#!/usr/bin/env python3
"""JAX interface for the *new* torch/specsimile-based fastGRAHSP emulator.

The emulator now consists of two specsimile ``LogDenseDecoder`` models:

* **GAL** – predicts log10(flux / 10^(log_stellar_mass - 10))
* **AGN** – predicts log10(flux / 10^(logLAGN - 44))

The combined (TOTAL) flux is:

    F_total[i] = F_GAL[i] + F_AGN[i]

where::

    F_GAL  = 10^(GAL_output)  * 10^(log_stellar_mass - 10)
    F_AGN  = 10^(AGN_output)  * 10^(logLAGN           - 44)

Only the filter columns that appear in *both* outputs are summed to TOTAL.

This module exposes two public functions that match the signatures expected
by ``difflike.py`` / ``megadifflike.py``:

* :func:`extract_fused_galagn_weights` – load the two .pt files into a flat
  parameter dict.  All required metadata (input parameter names, output
  column names) is read directly from the checkpoint files; no separate JSON
  sidecar is needed.
* :func:`jax_fastgrahsp_predict` – pure-JAX forward pass.
"""

from typing import Dict, List, Tuple

import jax.numpy as jnp
import numpy as np
import specsimile.jxrun as _specsimile
import torch

# ---------------------------------------------------------------------------
# Constants (hardcoded, matching buildemulator.py / prepare_data.py)
# ---------------------------------------------------------------------------
_GAL_NORM_OFFSET = 10.0   # log_stellar_mass reference
_AGN_NORM_OFFSET = 44.0   # logLAGN reference


# ---------------------------------------------------------------------------
# Weight extraction
# ---------------------------------------------------------------------------

def extract_fused_galagn_weights(
    gal_model_path: str,
    agn_model_path: str,
    verbose: bool = True,
) -> Tuple[Dict, Dict]:
    """
    Load the GAL and AGN specsimile emulators.

    All required metadata — input parameter names (``paramnames``) and output
    column names (``xnames``) — is read directly from the ``.pt`` checkpoint
    files.  No separate JSON sidecar is needed.

    Parameters
    ----------
    gal_model_path : str
        Path to the GAL specsimile ``.pt`` checkpoint.
    agn_model_path : str
        Path to the AGN specsimile ``.pt`` checkpoint.
    verbose : bool

    Returns
    -------
    params : dict
        Flat parameter bundle that :func:`jax_fastgrahsp_predict` consumes.
    info : dict
        Metadata dict with the same schema that the old JSON sidecar
        provided, reconstructed from checkpoint metadata::

            {
                "input": [...],          # full merged input column names
                "output": [
                    ["GAL",   [...]],
                    ["AGN",   [...]],
                    ["TOTAL", [...]]
                ]
            }

    Raises
    ------
    ValueError
        If either checkpoint is missing ``paramnames`` or ``xnames`` in its
        ``meta`` dict (i.e. it was saved before these fields were recorded).
    """
    # ------------------------------------------------------------------
    # Load raw checkpoints to extract metadata
    # ------------------------------------------------------------------
    if verbose:
        print("loading", gal_model_path)
    gal_ckpt = torch.load(gal_model_path, map_location='cpu', weights_only=False)
    if verbose:
        print("loading", agn_model_path)
    agn_ckpt = torch.load(agn_model_path, map_location='cpu', weights_only=False)

    gal_meta = gal_ckpt.get('meta', {})
    agn_meta = agn_ckpt.get('meta', {})

    # --- Input parameter names -------------------------------------------
    gal_paramnames: List[str] = gal_meta.get('paramnames', None)
    agn_paramnames: List[str] = agn_meta.get('paramnames', None)

    if gal_paramnames is None:
        raise ValueError(
            f"GAL checkpoint '{gal_model_path}' does not contain "
            "'paramnames' in its meta dict.  Re-train the emulator so that "
            "meta['paramnames'] is stored."
        )
    if agn_paramnames is None:
        raise ValueError(
            f"AGN checkpoint '{agn_model_path}' does not contain "
            "'paramnames' in its meta dict.  Re-train the emulator so that "
            "meta['paramnames'] is stored."
        )

    # --- Output column names ---------------------------------------------
    gal_colnames: List[str] = gal_meta.get('xnames', None)
    agn_colnames: List[str] = agn_meta.get('xnames', None)

    if gal_colnames is None:
        raise ValueError(
            f"GAL checkpoint '{gal_model_path}' does not contain "
            "'xnames' in its meta dict.  Re-train the emulator so that "
            "meta['xnames'] is stored (DatasetWriter must be called with "
            "xnames=GAL_output_colnames)."
        )
    if agn_colnames is None:
        raise ValueError(
            f"AGN checkpoint '{agn_model_path}' does not contain "
            "'xnames' in its meta dict.  Re-train the emulator so that "
            "meta['xnames'] is stored (DatasetWriter must be called with "
            "xnames=AGN_output_colnames)."
        )

    if verbose:
        print(f"  GAL paramnames ({len(gal_paramnames)}): {gal_paramnames}")
        print(f"  AGN paramnames ({len(agn_paramnames)}): {agn_paramnames}")
        print(f"  GAL outputs ({len(gal_colnames)}): {gal_colnames}")
        print(f"  AGN outputs ({len(agn_colnames)}): {agn_colnames}")

    # ------------------------------------------------------------------
    # Derive TOTAL columns as the intersection of GAL and AGN outputs,
    # preserving GAL order.  Strip 'GAL_' / 'AGN_' prefixes for matching.
    # ------------------------------------------------------------------
    gal_base = [c.replace('GAL_', '') for c in gal_colnames]
    agn_base = [c.replace('AGN_', '') for c in agn_colnames]
    agn_base_set = set(agn_base)

    total_colnames: List[str] = [
        base for base in gal_base if base in agn_base_set
    ]

    if verbose:
        print(f"  TOTAL outputs (intersection): {len(total_colnames)}")

    # Index arrays for selecting TOTAL columns out of GAL / AGN outputs
    gal_output_indices = np.array(
        [gal_base.index(c) for c in total_colnames], dtype=np.int32)
    agn_output_indices = np.array(
        [agn_base.index(c) for c in total_colnames], dtype=np.int32)

    # ------------------------------------------------------------------
    # Build the merged 'input_cols' list (union of both paramnames,
    # preserving a stable order: GAL first, then AGN-only extras).
    # This is used by downstream code that builds feature_dict keys.
    # ------------------------------------------------------------------
    seen: set = set()
    input_cols: List[str] = []
    for name in list(gal_paramnames) + list(agn_paramnames):
        if name not in seen:
            input_cols.append(name)
            seen.add(name)

    # ------------------------------------------------------------------
    # Load specsimile emulator weights into JAX-compatible bundles
    # ------------------------------------------------------------------
    if verbose:
        print(f"Loading GAL emulator from {gal_model_path}")
    gal_bundle = _specsimile.load_decoder_bundle(
        gal_model_path, prefix='gal', verbose=verbose)

    if verbose:
        print(f"Loading AGN emulator from {agn_model_path}")
    agn_bundle = _specsimile.load_decoder_bundle(
        agn_model_path, prefix='agn', verbose=verbose)

    # ------------------------------------------------------------------
    # Assemble the combined params dict
    # ------------------------------------------------------------------
    params: Dict = {}
    params.update(gal_bundle)
    params.update(agn_bundle)

    params['gal_output_indices'] = jnp.array(gal_output_indices)
    params['agn_output_indices'] = jnp.array(agn_output_indices)
    params['n_gal_outputs'] = len(gal_colnames)
    params['n_agn_outputs'] = len(agn_colnames)
    params['n_total_outputs'] = len(total_colnames)
    params['input_cols'] = input_cols
    params['gal_colnames'] = gal_colnames
    params['agn_colnames'] = agn_colnames
    params['total_colnames'] = total_colnames

    # ------------------------------------------------------------------
    # Build the info dict (replaces the old JSON sidecar)
    # ------------------------------------------------------------------
    info: Dict = {
        'input': input_cols,
        'output': [
            ['GAL', gal_colnames],
            ['AGN', agn_colnames],
            ['TOTAL', total_colnames],
        ],
    }

    if verbose:
        print(f"  GAL outputs: {len(gal_colnames)}")
        print(f"  AGN outputs: {len(agn_colnames)}")
        print(f"  TOTAL outputs (intersection): {len(total_colnames)}")

    return params, info


# ---------------------------------------------------------------------------
# JAX forward pass
# ---------------------------------------------------------------------------

def jax_fastgrahsp_predict(
    feature_dict: dict,
    params: Dict,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Pure-JAX forward pass for the specsimile-based fastGRAHSP emulator.

    Parameters
    ----------
    feature_dict : dict[str, jnp.ndarray]
        Mapping from feature name to ``(N,)`` JAX array.  Must contain at
        least all keys present in ``gal_paramnames`` and ``agn_paramnames``
        (as stored in the model checkpoints), plus ``'log_stellar_mass'``
        and ``'log_L_AGN'`` for the normalisation.
    params : dict
        Bundle returned by :func:`extract_fused_galagn_weights`.

    Returns
    -------
    total   : (N, n_total_outputs)  combined GAL + AGN flux  [mJy]
    gal_out : (N, n_gal_outputs)    GAL-only flux            [mJy]
    agn_out : (N, n_agn_outputs)    AGN-only flux            [mJy]
    """
    col_dict = feature_dict

    # Normalisation scalars
    log_stellar_mass = col_dict['log_stellar_mass']   # (N,)
    log_L_AGN = col_dict['log_L_AGN']          # (N,)

    # GAL forward pass — jax_encoder_forward_kw selects only the columns
    # that the GAL network was trained on (from gal_paramnames in the bundle)
    h_gal = _specsimile.jax_encoder_forward_kw(
        params, 'gal', col_dict,
    )   # (N, n_gal_outputs)  log10(flux / gal_norm)

    # AGN forward pass
    h_agn = _specsimile.jax_encoder_forward_kw(
        params, 'agn', col_dict,
    )   # (N, n_agn_outputs)  log10(flux / agn_norm)

    gal_norm = 10.0 ** (log_stellar_mass - _GAL_NORM_OFFSET)   # (N,)
    agn_norm = 10.0 ** (log_L_AGN - _AGN_NORM_OFFSET)   # (N,)

    # De-log and apply normalisation
    gal_out = (10.0 ** h_gal) * gal_norm[:, None]   # (N, n_gal_outputs)
    agn_out = (10.0 ** h_agn) * agn_norm[:, None]   # (N, n_agn_outputs)

    # Select shared (TOTAL) columns and sum
    gal_selected = gal_out[:, params['gal_output_indices']]   # (N, n_total)
    agn_selected = agn_out[:, params['agn_output_indices']]   # (N, n_total)
    total = gal_selected + agn_selected

    return total, gal_out, agn_out
