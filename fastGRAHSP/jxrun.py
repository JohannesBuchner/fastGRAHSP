#!/usr/bin/env python3
"""Functions for using the emulator from jax."""

import json
import os
from typing import Dict

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnx.numpy_helper as np_helper


def extract_onnx_weights(onnx_path: str) -> Dict[str, np.ndarray]:
    """Extract all initializer tensors from an ONNX model.

    Parameters
    ----------
    onnx_path: str
        path to model

    Returns
    -------
    weights: dict
        model data
    """
    model = onnx.load(onnx_path)
    weights = {}
    for init in model.graph.initializer:
        weights[init.name] = np_helper.to_array(init)
    return weights


def extract_fused_galagn_weights(onnx_path: str, json_path: str, verbose: bool = True):
    """Extract model weights.

    Parameters
    ----------
    onnx_path: str
        path to model data.
    json_path: str
        path to model metadata
    verbose: bool
        whether to print out debug information

    Returns
    -------
    params: dict
        model data
    info: dict
        model metadata
    """
    weights = extract_onnx_weights(onnx_path)
    info = json.load(open(json_path))

    params = {}
    params["gal_indices"] = weights["GAL_indices"].astype(np.int32)
    params["agn_indices"] = weights["AGN_indices"].astype(np.int32)
    params["gal_norm_index"] = weights["gal_norm_index"].astype(np.int32)
    params["agn_norm_index"] = weights["agn_norm_index"].astype(np.int32)
    params["gal_norm_offset"] = 10.0
    params["agn_norm_offset"] = 44.0
    params["gal_output_indices"] = weights["GAL_outputs_indices"].astype(np.int32)
    params["agn_output_indices"] = weights["AGN_outputs_indices"].astype(np.int32)

    params["gal_norm_mean"] = weights["sequential_1/normalization_1/Sub/y:0"].astype(
        np.float32
    )
    params["gal_norm_inv_std"] = weights[
        "ConstantFolding/sequential_1/normalization_1/truediv_recip:0"
    ].astype(np.float32)

    gal_layer_names = [
        (
            "sequential_1/dense_1/Cast/ReadVariableOp:0",
            "sequential_1/dense_1/BiasAdd/ReadVariableOp:0",
        ),
        (
            "sequential_1/dense_1_2/Cast/ReadVariableOp:0",
            "sequential_1/dense_1_2/BiasAdd/ReadVariableOp:0",
        ),
        (
            "sequential_1/dense_2_1/Cast/ReadVariableOp:0",
            "sequential_1/dense_2_1/BiasAdd/ReadVariableOp:0",
        ),
    ]
    for i, (w_name, b_name) in enumerate(gal_layer_names):
        params[f"gal_W{i}"] = weights[w_name].astype(np.float32)
        params[f"gal_b{i}"] = weights[b_name].astype(np.float32)
    params["gal_n_layers"] = len(gal_layer_names)

    params["agn_norm_mean"] = weights[
        "sequential_1_1/normalization_1_1/Sub/y:0"
    ].astype(np.float32)
    params["agn_norm_inv_std"] = weights[
        "ConstantFolding/sequential_1_1/normalization_1_1/truediv_recip:0"
    ].astype(np.float32)

    agn_layer_names = [
        (
            "sequential_1_1/dense_3_1/Cast/ReadVariableOp:0",
            "sequential_1_1/dense_3_1/BiasAdd/ReadVariableOp:0",
        ),
        (
            "sequential_1_1/dense_4_1/Cast/ReadVariableOp:0",
            "sequential_1_1/dense_4_1/BiasAdd/ReadVariableOp:0",
        ),
        (
            "sequential_1_1/dense_5_1/Cast/ReadVariableOp:0",
            "sequential_1_1/dense_5_1/BiasAdd/ReadVariableOp:0",
        ),
    ]
    for i, (w_name, b_name) in enumerate(agn_layer_names):
        params[f"agn_W{i}"] = weights[w_name].astype(np.float32)
        params[f"agn_b{i}"] = weights[b_name].astype(np.float32)
    params["agn_n_layers"] = len(agn_layer_names)

    n_gal_in = len(params["gal_indices"])
    n_agn_in = len(params["agn_indices"])
    if verbose:
        print(
            f"GAL network: input={n_gal_in} -> {params['gal_W0'].shape} -> {params['gal_W1'].shape} -> {params['gal_W2'].shape}"
        )
        print(
            f"AGN network: input={n_agn_in} -> {params['agn_W0'].shape} -> {params['agn_W1'].shape} -> {params['agn_W2'].shape}"
        )

    return params, info


def jax_normalize_grahsp(x, mean, inv_std):
    """Normalise x.

    Parameters
    ----------
    x: float
        value
    mean: float
        mean
    inv_std: float
        inverse standard deviation

    Returns
    -------
    float:
        mean-subtracted value, normalised by inv_std
    """
    return (x - mean) * inv_std


def _jax_grahsp_forward(h, params, n_layers):
    """Forward pass for GAL or AGN sub-network of fastGRAHSP.

    Parameters
    ----------
    h: float
        input layer values
    params: dict
        model data
    n_layers: int
        number of model layers

    Returns
    -------
    h: float
        output layer values
    """
    h = jax_normalize_grahsp(
        h, params["norm_mean"], params["norm_inv_std"]
    )
    for i in range(n_layers):
        h = h @ params[f"W{i}"] + params[f"b{i}"]
        if i < n_layers - 1:
            h = jax.nn.relu(h)
    return h


def jax_fastgrahsp_predict(X, params):
    """Predict total, GAL, AGN fluxes from fastGRAHSP inputs.

    X: (N, n_input_features)
    Returns: total (N, n_total), gal (N, n_gal_out), agn (N, n_agn_out)

    Parameters
    ----------
    X: float
        input data
    params: dict
        model data

    Returns
    -------
    flux: array
        flux in mJy.
    fluxGAL: array
        flux in mJy for galaxy alone
    fluxAGN: array
        flux in mJy for AGN alone
    """
    X = X.astype(jnp.float32)
    Xgal = X[:, params["gal_indices"]]
    Xagn = X[:, params["agn_indices"]]

    gal_norm_raw = X[:, params["gal_norm_index"]].reshape(-1)
    agn_norm_raw = X[:, params["agn_norm_index"]].reshape(-1)
    gal_norm = 10.0 ** (gal_norm_raw - params["gal_norm_offset"])
    agn_norm = 10.0 ** (agn_norm_raw - params["agn_norm_offset"])

    h_gal = _jax_grahsp_forward(Xgal, params, "gal", params["gal_n_layers"]).astype(
        jnp.float64
    )
    gal_out_linear = (10.0**h_gal) * gal_norm[:, None].astype(jnp.float64)

    h_agn = _jax_grahsp_forward(Xagn, params, "agn", params["agn_n_layers"]).astype(
        jnp.float64
    )
    agn_out_linear = (10.0**h_agn) * agn_norm[:, None].astype(jnp.float64)

    gal_selected = gal_out_linear[:, params["gal_output_indices"]]
    agn_selected = agn_out_linear[:, params["agn_output_indices"]]
    total = gal_selected + agn_selected

    return total, gal_out_linear, agn_out_linear


class JaxFastGRAHSP:
    """Reusable emulator class."""

    def __init__(self, width=128):
        """Initialise.

        Parameters
        ----------
        width: int
            Neural network model to use, must be one of 128, 256, 512.
            Larger size is slower but more accurate
        """
        key = f"GALAGN{width}"
        filename = os.path.join(os.path.dirname(__file__), key)
        self.params = extract_fused_galagn_weights(
            onnx_path=filename + ".onnx", json_path=filename + ".json"
        )

    def __call__(self, X):
        """Predict flux.

        Parameters
        ----------
        X: array
            Model parameter array (see example below):

                * log10 tau: time-scale of starformation in Myr
                * log10 age: age of oldest stars in Myr
                * log10 AFeII: amplitude of lines
                * log10 Alines: amplitude of lines
                * linewidth: width of broad emission lines
                * Si: strength of 12µm silicate feature (positive: emission, negative: absorption)
                * fcov: covering factor, based on the ratio of 12µm to 5100A luminosity ratio
                * COOLlam: central wavelength in µm of cold torus component
                * COOLwidth: width (in dex) of cold torus component
                * log10 HOTfcov: peak luminosity ratio of hot to cold torus component
                * HOTlam: central wavelength in µm of hot torus component
                * HOTwidth: width (in dex) of hot torus component
                * plslope: AGN power law slope
                * plbendloc: wavelength of UV bend.
                * log10 plbendwidth: width of UV bend (in dex).
                * log10 EBV: E(B-V) attenuation coefficient of entire system.
                * log10 EBV_AGN: E(B-V) attenuation coefficient of nucleus.
                * alpha: dust slope of Dale+ model
                * z: redshift
                * M: log10 of galaxy stellar mass in solar masses
                * L5100A: log10 of AGN luminosity at 5100A in erg/s.

        Returns
        -------
        flux: array
            flux in mJy.
        fluxGAL: array
            flux in mJy for galaxy alone
        fluxAGN: array
            flux in mJy for AGN alone
        """
        return jax_fastgrahsp_predict(X, self.params)
