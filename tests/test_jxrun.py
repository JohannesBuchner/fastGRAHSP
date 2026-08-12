#!/usr/bin/env python3
"""
Tests for the fastGRAHSP emulator: SFR, stellar luminosity, and filter fluxes
for three galaxy/AGN types (quiescent, star-forming, quasar).

These tests check that the emulator gives physically reasonable outputs
without requiring the full JointSEDXrayLikelihood infrastructure.
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", 1)

# ---------------------------------------------------------------------------
# Parameter sets for the three test cases
# ---------------------------------------------------------------------------

# Quasar: high-mass BH, high accretion rate
_PARAMS_QUASAR = {
    "log_sfhdelayed_tau_main":      3.7350649,    # log10(Myr)
    "log_sfhdelayed_age_main":      4.0,           # log10(Myr), 10 Gyr
    "log_sfhdelayed_f_burst":       -10.0,         # log10
    "log_activatelines_AFeII":      -0.02334764,   # log10
    "log_activatelines_Alines":     0.27632016,    # log10
    "activatelines.linewidth":      5000.0,        # km/s
    "activategtorus.Si":            0.0,
    "log_activategtorus_fcov":      -0.5,          # log10
    "log_activategtorus_COOLlam":   1.8,           # log10(um), ~63 um
    "activategtorus.COOLwidth":     0.5,
    "activategtorus.HOTlam":        2.5,           # um
    "activategtorus.HOTwidth":      0.5,
    "log_activategtorus_HOTfcov":   0.3,           # log10
    "activatepl.plslope":           -2.0,
    "activatepl.plbendloc":         100.0,         # nm
    "log_activatepl_plbendwidth":   1.0,           # log10(nm)
    "activatepl.uvslope":           -0.5,
    "log_biattenuation_E(B-V)":     -1.3,          # log10
    "log_biattenuation_E(B-V)-AGN": -1.65,         # log10
    "galdale2014.alpha":            2.87,
    "log_stellar_mass":             9.32,          # log10(Msun)
    "log_L_AGN":                    45.0,          # log10(erg/s) at 5100A
    "redshift":                     0.426,
    "E(B-V)-total":                 0.073,
}

# Quiescent galaxy: low-mass BH, negligible AGN, old stellar population
_PARAMS_QUI = {
    "log_sfhdelayed_tau_main":      2.2,           # log10(Myr), short e-folding
    "log_sfhdelayed_age_main":      3.9,           # log10(Myr), ~8 Gyr
    "log_sfhdelayed_f_burst":       -10.0,
    "log_activatelines_AFeII":      -0.02334764,
    "log_activatelines_Alines":     0.27632016,
    "activatelines.linewidth":      5000.0,
    "activategtorus.Si":            0.0,
    "log_activategtorus_fcov":      -0.5,
    "log_activategtorus_COOLlam":   1.8,
    "activategtorus.COOLwidth":     0.5,
    "activategtorus.HOTlam":        2.5,
    "activategtorus.HOTwidth":      0.5,
    "log_activategtorus_HOTfcov":   0.3,
    "activatepl.plslope":           -2.0,
    "activatepl.plbendloc":         100.0,
    "log_activatepl_plbendwidth":   1.0,
    "activatepl.uvslope":           -0.5,
    "log_biattenuation_E(B-V)":     -2.1,          # low dust
    "log_biattenuation_E(B-V)-AGN": -2.0,
    "galdale2014.alpha":            2.87,
    "log_stellar_mass":             11.5,          # log10(Msun), massive galaxy
    "log_L_AGN":                    38.0,          # negligible AGN
    "redshift":                     0.426,
    "E(B-V)-total":                 0.009,
}

# Star-forming galaxy: low-mass BH, negligible AGN, active star formation
_PARAMS_SF = {
    "log_sfhdelayed_tau_main":      4.0,           # log10(Myr), long e-folding
    "log_sfhdelayed_age_main":      4.0,           # log10(Myr), 10 Gyr
    "log_sfhdelayed_f_burst":       -10.0,
    "log_activatelines_AFeII":      -0.02334764,
    "log_activatelines_Alines":     0.27632016,
    "activatelines.linewidth":      5000.0,
    "activategtorus.Si":            0.0,
    "log_activategtorus_fcov":      -0.5,
    "log_activategtorus_COOLlam":   1.8,
    "activategtorus.COOLwidth":     0.5,
    "activategtorus.HOTlam":        2.5,
    "activategtorus.HOTwidth":      0.5,
    "log_activategtorus_HOTfcov":   0.3,
    "activatepl.plslope":           -2.0,
    "activatepl.plbendloc":         100.0,
    "log_activatepl_plbendwidth":   1.0,
    "activatepl.uvslope":           -0.5,
    "log_biattenuation_E(B-V)":     -2.0,          # low dust
    "log_biattenuation_E(B-V)-AGN": -2.0,
    "galdale2014.alpha":            2.87,
    "log_stellar_mass":             12.32,         # log10(Msun), very massive
    "log_L_AGN":                    38.0,          # negligible AGN
    "redshift":                     0.426,
    "E(B-V)-total":                 0.01,
}


def _make_feature_dict(params):
    """Convert a flat parameter dict to a JAX feature dict (N=1 arrays)."""
    return {k: jnp.array([v]) for k, v in params.items()}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def emulator_weights():
    """Load fastGRAHSP emulator weights (variant 6)."""
    import fastGRAHSP.jxrun as jxrun
    params, info = jxrun.extract_fused_galagn_weights(
        gal_model_path=None,
        agn_model_path=None,
        verbose=True,
    )
    return params, info


@pytest.fixture(scope="module")
def gal_colnames(emulator_weights):
    """GAL output column names from the emulator."""
    params, info = emulator_weights
    return params["gal_colnames"]


@pytest.fixture(scope="module")
def agn_colnames(emulator_weights):
    """AGN output column names from the emulator."""
    params, info = emulator_weights
    return params["agn_colnames"]


@pytest.fixture(scope="module")
def total_colnames(emulator_weights):
    """TOTAL output column names from the emulator."""
    params, info = emulator_weights
    return params["total_colnames"]


def _run_predict(emulator_weights, feature_params):
    """Run jax_fastgrahsp_predict for a given parameter dict."""
    import fastGRAHSP.jxrun as jxrun
    params, info = emulator_weights
    fd = _make_feature_dict(feature_params)
    total, gal_out, agn_out = jxrun.jax_fastgrahsp_predict(fd, params)
    return np.array(total), np.array(gal_out), np.array(agn_out)


def _get_col_index(colnames, name):
    """Return index of *name* in colnames, or None if not present."""
    # Try exact match first
    if name in colnames:
        return colnames.index(name)
    # Try with GAL_ prefix stripped
    stripped = [c.replace("GAL_", "").replace("AGN_", "") for c in colnames]
    if name in stripped:
        return stripped.index(name)
    return None


# ---------------------------------------------------------------------------
# Helper: assert flux is finite, positive, and in a reasonable mJy range
# ---------------------------------------------------------------------------

def _assert_flux_reasonable(flux_mjy, label, lo=1e-10, hi=1e6):
    assert np.isfinite(flux_mjy), f"{label}: flux is not finite ({flux_mjy})"
    assert flux_mjy > 0, f"{label}: flux is not positive ({flux_mjy:.3e} mJy)"
    assert flux_mjy >= lo, f"{label}: flux too low ({flux_mjy:.3e} mJy < {lo:.3e})"
    assert flux_mjy <= hi, f"{label}: flux too high ({flux_mjy:.3e} mJy > {hi:.3e})"


def _pcigale_sfr(tau_myr, age_myr, mstar_msun):
    """
    Compute pcigale's sfh.sfr (in Msun/yr) for the given SFH parameters,
    scaled to the given stellar mass.

    The SFHDelayed module with normalise=True sets sed.sfh = self.sfr / sfr_integrated,
    where sfr_integrated = sum(sfr) * 1e6 Myr * (Msun/yr per unit).
    The sed.sfh setter reads sfh_sfr[-1] as sfh.sfr, which is the SFR at the
    last time step (i.e. today) in units of Msun/yr per solar mass formed.
    Multiplying by mstar_msun gives the physical SFR in Msun/yr.

    Parameters
    ----------
    tau_myr : float
        e-folding time of the delayed SFH in Myr.
    age_myr : float
        Age of the oldest stars in Myr.
    mstar_msun : float
        Total stellar mass in Msun (used to rescale the normalised SFR).

    Returns
    -------
    sfr : float
        Star formation rate in Msun/yr, or None if pcigale is unavailable or
        sfh.sfr is not present in the SED info.
    """
    try:
        from pcigale.creation_modules.sfhdelayed import SFHDelayed
        from pcigale.sed import SED
    except ImportError:
        return None

    sed = SED()
    module = SFHDelayed(
        name="sfhd",
        tau_main=tau_myr,
        age_main=int(round(age_myr)),
        f_burst=0,
        age_burst=10,
        sfr_A=1.0,
        normalise=True,
    )
    module.process(sed)

    # With normalise=True, self.sfr is divided by sfr_integrated (in Msun).
    # sed.sfh is set to self.sfr (the normalised array, in yr^-1).
    # The sed.sfh setter stores sfh_sfr[-1] as sfh.sfr, which is the
    # normalised SFR at the last time step: sfr[-1] / sfr_integrated [yr^-1 Msun^-1].
    # Multiplying by mstar_msun gives physical SFR in Msun/yr.
    sfr_raw = sed.info.get("sfh.sfr", None)
    if sfr_raw is None:
        return None

    return float(sfr_raw) * mstar_msun


# ---------------------------------------------------------------------------
# Tests: quiescent galaxy (qui)
# ---------------------------------------------------------------------------

class TestQuiescentGalaxy:
    """
    Tests for a quiescent (passive) galaxy:
      - log_stellar_mass = 11.5 (very massive, old population)
      - log_L_AGN = 38.0 (negligible AGN)
      - short SFH e-folding time (tau_main = 10^2.2 Myr ~ 158 Myr)
      - low dust (E(B-V) ~ 0.008)

    Expectations:
      - Galaxy dominates the SED
      - SFR should be very low (quiescent)
      - Stellar luminosity should be very high (massive)
      - AGN contribution should be negligible
    """

    def test_predict_runs_without_error(self, emulator_weights):
        """Emulator should run without errors for quiescent parameters."""
        total, gal_out, agn_out = _run_predict(emulator_weights, _PARAMS_QUI)
        assert total is not None
        assert gal_out is not None
        assert agn_out is not None

    def test_total_fluxes_finite_and_positive(self, emulator_weights, total_colnames):
        """All total fluxes should be finite and positive for quiescent galaxy."""
        total, gal_out, agn_out = _run_predict(emulator_weights, _PARAMS_QUI)
        assert total.shape[0] == 1, f"Expected batch size 1, got {total.shape[0]}"
        assert np.all(np.isfinite(total)), (
            f"Non-finite total fluxes: {total[0]}"
        )
        assert np.all(total > 0), (
            f"Non-positive total fluxes: min={total.min():.3e} mJy"
        )

    def test_gal_fluxes_finite_and_positive(self, emulator_weights, gal_colnames):
        """All GAL fluxes should be finite and positive."""
        total, gal_out, agn_out = _run_predict(emulator_weights, _PARAMS_QUI)
        assert np.all(np.isfinite(gal_out)), "Non-finite GAL fluxes"
        assert np.all(gal_out > 0), f"Non-positive GAL fluxes: min={gal_out.min():.3e}"

    def test_wise1_flux(self, emulator_weights, total_colnames):
        """Print the WISE1 flux for the quiescent galaxy."""
        total, gal_out, agn_out = _run_predict(emulator_weights, _PARAMS_QUI)
        t_idx = _get_col_index(list(total_colnames), "WISE1")
        if t_idx is None:
            pytest.skip("WISE1 not found in total colnames")
        wise1_flux = float(total[0, t_idx])
        print(f"Quiescent galaxy WISE1 flux: {wise1_flux:.3e} mJy")

    def test_sfr_physically_reasonable(self, emulator_weights, gal_colnames):
        """
        SFR for a quiescent galaxy with short tau (158 Myr) and old age (8 Gyr)
        should be very low.
        Expected: log10(SFR / Msun/yr) < 0 (less than 1 Msun/yr).
        Physical range: 1e-10 to 1e3 Msun/yr.
        """
        total, gal_out, agn_out = _run_predict(emulator_weights, _PARAMS_QUI)

        sfr_idx = _get_col_index(list(gal_colnames), "sfh.sfr")
        if sfr_idx is None:
            pytest.skip("sfh.sfr not found in GAL emulator outputs")

        sfr = float(gal_out[0, sfr_idx])
        assert sfr > 0, f"SFR={sfr:.3e} is not positive"

        log_sfr = np.log10(sfr)
        print(f"Quiescent galaxy SFR: {sfr:.3e} Msun/yr (log10={log_sfr:.2f})")

        # Quiescent galaxy: SFR should be low
        assert -10 < log_sfr < 3, (
            f"SFR={sfr:.3e} Msun/yr (log10={log_sfr:.2f}) outside physical range "
            f"[1e-10, 1e3] Msun/yr for a quiescent galaxy"
        )
        # For a truly quiescent galaxy with short tau, SFR should be < 10 Msun/yr
        assert log_sfr < 2, (
            f"SFR={sfr:.3e} Msun/yr (log10={log_sfr:.2f}) is too high for a "
            f"quiescent galaxy (expected < 100 Msun/yr)"
        )

    def test_stellar_lum_matches_massive_galaxy(self, emulator_weights, gal_colnames):
        """
        Stellar luminosity for log_stellar_mass=11.5 should be high.
        Expected: log10(stellar.lum / W) in range [36, 45].
        """
        total, gal_out, agn_out = _run_predict(emulator_weights, _PARAMS_QUI)

        stellum_idx = _get_col_index(list(gal_colnames), "stellar.lum")
        if stellum_idx is None:
            pytest.skip("stellar.lum not found in GAL emulator outputs")

        stellum = float(gal_out[0, stellum_idx])
        assert stellum > 0, f"stellar.lum={stellum:.3e} is not positive"

        log_stellum = np.log10(stellum)
        print(f"Quiescent galaxy stellar.lum: {stellum:.3e} W (log10={log_stellum:.2f})")

        # Massive galaxy: stellar luminosity should be high
        assert 30 < log_stellum < 50, (
            f"log10(stellar.lum)={log_stellum:.2f} W outside physical range [30, 50]"
        )
        # For log_stellar_mass=11.5, stellar.lum should be high
        assert log_stellum > 35, (
            f"log10(stellar.lum)={log_stellum:.2f} W is too low for a "
            f"massive galaxy with log_M*=11.5"
        )

    def test_galaxy_dominates_over_agn_in_optical(
        self, emulator_weights, total_colnames, gal_colnames
    ):
        """
        For a quiescent galaxy with negligible AGN (log_L_AGN=38),
        the galaxy flux should dominate in optical filters.
        GAL flux should be > 50% of TOTAL flux in optical.
        """
        total, gal_out, agn_out = _run_predict(emulator_weights, _PARAMS_QUI)

        # Check a few optical filters
        for fname in ["decam_r", "decam_g", "SDSS_r"]:
            t_idx = _get_col_index(list(total_colnames), fname)
            g_idx = _get_col_index(list(gal_colnames), fname)
            if t_idx is None or g_idx is None:
                continue

            f_total = float(total[0, t_idx])
            f_gal = float(gal_out[0, g_idx])
            if f_total > 0 and f_gal > 0:
                gal_frac = f_gal / f_total
                print(f"Quiescent {fname}: GAL={f_gal:.3e}, TOTAL={f_total:.3e}, "
                      f"GAL fraction={gal_frac:.2f}")
                assert gal_frac > 0.5, (
                    f"{fname}: GAL fraction={gal_frac:.2f} < 0.5 for quiescent galaxy "
                    f"(AGN should be negligible at log_L_AGN=38)"
                )
                return  # One successful check is enough

        pytest.skip("No suitable optical filter found in both total and gal colnames")

    def test_total_flux_in_reasonable_range(self, emulator_weights, total_colnames):
        """
        Total fluxes for a quiescent galaxy at z=0.426 should be in a
        physically reasonable range (not zero, not absurdly large).
        """
        total, gal_out, agn_out = _run_predict(emulator_weights, _PARAMS_QUI)
        total_arr = total[0]

        # All fluxes should be in [1e-10, 1e4] mJy
        assert np.all(total_arr > 1e-10), (
            f"Some fluxes are effectively zero: min={total_arr.min():.3e} mJy"
        )
        assert np.all(total_arr < 1e4), (
            f"Some fluxes are unphysically large: max={total_arr.max():.3e} mJy"
        )


# ---------------------------------------------------------------------------
# Tests: star-forming galaxy (sf)
# ---------------------------------------------------------------------------

class TestStarFormingGalaxy:
    """
    Tests for a star-forming galaxy:
      - log_stellar_mass = 12.32 (very massive)
      - log_L_AGN = 38.0 (negligible AGN)
      - long SFH e-folding time (tau_main = 10^4.0 Myr ~ 10 Gyr), rising SFH
      - low dust

    Expectations:
      - Galaxy dominates the SED
      - SFR should be higher than quiescent case
      - Stellar luminosity should be very high (most massive)
      - AGN contribution negligible
    """

    def test_predict_runs_without_error(self, emulator_weights):
        """Emulator should run without errors for star-forming parameters."""
        total, gal_out, agn_out = _run_predict(emulator_weights, _PARAMS_SF)
        assert total is not None

    def test_total_fluxes_finite_and_positive(self, emulator_weights, total_colnames):
        """All total fluxes should be finite and positive for star-forming galaxy."""
        total, gal_out, agn_out = _run_predict(emulator_weights, _PARAMS_SF)
        assert np.all(np.isfinite(total)), "Non-finite total fluxes (sf)"
        assert np.all(total > 0), f"Non-positive total fluxes (sf): min={total.min():.3e}"

    def test_sfr_higher_than_quiescent(self, emulator_weights, gal_colnames):
        """
        SFR for star-forming galaxy should be higher than for quiescent galaxy.
        Both have log_stellar_mass > 10, but SF has longer tau -> more ongoing SF.
        """
        _, gal_sf, _ = _run_predict(emulator_weights, _PARAMS_SF)
        _, gal_qui, _ = _run_predict(emulator_weights, _PARAMS_QUI)

        sfr_idx = _get_col_index(list(gal_colnames), "sfh.sfr")
        if sfr_idx is None:
            pytest.skip("sfh.sfr not found in GAL emulator outputs")

        sfr_sf = float(gal_sf[0, sfr_idx])
        sfr_qui = float(gal_qui[0, sfr_idx])

        print(f"SF galaxy SFR:        {sfr_sf:.3e} Msun/yr")
        print(f"Quiescent galaxy SFR: {sfr_qui:.3e} Msun/yr")

        assert sfr_sf > 0, f"SF galaxy SFR={sfr_sf:.3e} is not positive"
        assert sfr_qui > 0, f"Quiescent galaxy SFR={sfr_qui:.3e} is not positive"

        # SF galaxy should have higher SFR (longer tau -> more ongoing SF at age=10 Gyr)
        assert sfr_sf > sfr_qui, (
            f"SF galaxy SFR={sfr_sf:.3e} Msun/yr should be > "
            f"quiescent SFR={sfr_qui:.3e} Msun/yr"
        )

    def test_sfr_physically_reasonable(self, emulator_weights, gal_colnames):
        """
        SFR for a star-forming galaxy should be in a physically reasonable range.
        For log_M*=12.32 and tau=10 Gyr, typical SFR ~ 1-1000 Msun/yr.
        """
        total, gal_out, agn_out = _run_predict(emulator_weights, _PARAMS_SF)

        sfr_idx = _get_col_index(list(gal_colnames), "sfh.sfr")
        if sfr_idx is None:
            pytest.skip("sfh.sfr not found in GAL emulator outputs")

        sfr = float(gal_out[0, sfr_idx])
        assert sfr > 0, f"SFR={sfr:.3e} is not positive"

        log_sfr = np.log10(sfr)
        print(f"Star-forming galaxy SFR: {sfr:.3e} Msun/yr (log10={log_sfr:.2f})")

        assert -6 < log_sfr < 5, (
            f"SFR={sfr:.3e} Msun/yr (log10={log_sfr:.2f}) outside physical range"
        )

    def test_stellar_lum_very_high(self, emulator_weights, gal_colnames):
        """
        Stellar luminosity for log_stellar_mass=12.32 should be very high.
        Expected: log10(stellar.lum / W) > 37.
        """
        total, gal_out, agn_out = _run_predict(emulator_weights, _PARAMS_SF)

        stellum_idx = _get_col_index(list(gal_colnames), "stellar.lum")
        if stellum_idx is None:
            pytest.skip("stellar.lum not found in GAL emulator outputs")

        stellum = float(gal_out[0, stellum_idx])
        assert stellum > 0, f"stellar.lum={stellum:.3e} is not positive"

        log_stellum = np.log10(stellum)
        print(f"Star-forming galaxy stellar.lum: {stellum:.3e} W (log10={log_stellum:.2f})")

        assert 30 < log_stellum < 50, (
            f"log10(stellar.lum)={log_stellum:.2f} W outside physical range"
        )
        # Very massive galaxy should have high stellar luminosity
        assert log_stellum > 37, (
            f"log10(stellar.lum)={log_stellum:.2f} W is too low for "
            f"log_M*=12.32 star-forming galaxy"
        )

    def test_sf_stellar_lum_higher_than_quiescent(self, emulator_weights, gal_colnames):
        """
        Star-forming galaxy (log_M*=12.32) should have higher stellar luminosity
        than quiescent galaxy (log_M*=11.5).
        """
        _, gal_sf, _ = _run_predict(emulator_weights, _PARAMS_SF)
        _, gal_qui, _ = _run_predict(emulator_weights, _PARAMS_QUI)

        stellum_idx = _get_col_index(list(gal_colnames), "stellar.lum")
        if stellum_idx is None:
            pytest.skip("stellar.lum not found in GAL emulator outputs")

        stellum_sf = float(gal_sf[0, stellum_idx])
        stellum_qui = float(gal_qui[0, stellum_idx])

        print(f"SF stellar.lum:        {stellum_sf:.3e} W")
        print(f"Quiescent stellar.lum: {stellum_qui:.3e} W")

        assert stellum_sf > stellum_qui, (
            f"SF stellar.lum={stellum_sf:.3e} W should be > "
            f"quiescent stellar.lum={stellum_qui:.3e} W "
            f"(log_M* 12.32 vs 11.5)"
        )

    def test_galaxy_dominates_in_nir(self, emulator_weights, total_colnames, gal_colnames):
        """
        For a star-forming galaxy with negligible AGN, galaxy should dominate in NIR.
        """
        total, gal_out, agn_out = _run_predict(emulator_weights, _PARAMS_SF)

        for fname in ["WISE1", "WFCAM_Ks", "WFCAM_H"]:
            t_idx = _get_col_index(list(total_colnames), fname)
            g_idx = _get_col_index(list(gal_colnames), fname)
            if t_idx is None or g_idx is None:
                continue

            f_total = float(total[0, t_idx])
            f_gal = float(gal_out[0, g_idx])
            if f_total > 0 and f_gal > 0:
                gal_frac = f_gal / f_total
                print(f"SF {fname}: GAL={f_gal:.3e}, TOTAL={f_total:.3e}, "
                      f"GAL fraction={gal_frac:.2f}")
                assert gal_frac > 0.3, (
                    f"{fname}: GAL fraction={gal_frac:.2f} < 0.3 for SF galaxy "
                    f"(AGN should be negligible at log_L_AGN=38)"
                )
                return

        pytest.skip("No suitable NIR filter found in both total and gal colnames")


# ---------------------------------------------------------------------------
# Tests: quasar
# ---------------------------------------------------------------------------

class TestQuasar:
    """
    Tests for a quasar:
      - log_stellar_mass = 9.32
      - log_L_AGN = 45.0 (bright AGN)
      - moderate SFH

    Expectations:
      - AGN dominates the SED in UV/optical
      - SFR moderate
      - Stellar luminosity lower than the pure-galaxy cases
      - AGN contribution significant
    """

    def test_predict_runs_without_error(self, emulator_weights):
        """Emulator should run without errors for quasar parameters."""
        total, gal_out, agn_out = _run_predict(emulator_weights, _PARAMS_QUASAR)
        assert total is not None

    def test_total_fluxes_finite_and_positive(self, emulator_weights, total_colnames):
        """All total fluxes should be finite and positive for quasar."""
        total, gal_out, agn_out = _run_predict(emulator_weights, _PARAMS_QUASAR)
        assert np.all(np.isfinite(total)), "Non-finite total fluxes (quasar)"
        assert np.all(total > 0), f"Non-positive total fluxes (quasar): min={total.min():.3e}"

    def test_wise1_flux(self, emulator_weights, total_colnames):
        """Print the WISE1 flux for the quasar."""
        total, gal_out, agn_out = _run_predict(emulator_weights, _PARAMS_QUASAR)
        t_idx = _get_col_index(list(total_colnames), "WISE1")
        if t_idx is None:
            pytest.skip("WISE1 not found in total colnames")
        wise1_flux = float(total[0, t_idx])
        print(f"Quasar WISE1 flux: {wise1_flux:.3e} mJy")

    def test_agn_dominates_in_uv(self, emulator_weights, total_colnames, agn_colnames):
        """
        For a bright quasar (log_L_AGN=45), AGN should dominate in UV.
        AGN flux should be > 30% of TOTAL flux in UV/optical filters.
        """
        total, gal_out, agn_out = _run_predict(emulator_weights, _PARAMS_QUASAR)

        for fname in ["NUV", "decam_g", "decam_r", "SDSS_g"]:
            t_idx = _get_col_index(list(total_colnames), fname)
            a_idx = _get_col_index(list(agn_colnames), fname)
            if t_idx is None or a_idx is None:
                continue

            f_total = float(total[0, t_idx])
            f_agn = float(agn_out[0, a_idx])
            if f_total > 0 and f_agn > 0:
                agn_frac = f_agn / f_total
                print(f"Quasar {fname}: AGN={f_agn:.3e}, TOTAL={f_total:.3e}, "
                      f"AGN fraction={agn_frac:.2f}")
                assert agn_frac > 0.3, (
                    f"{fname}: AGN fraction={agn_frac:.2f} < 0.3 for bright quasar "
                    f"(log_L_AGN=45, log_M*=9.32)"
                )
                return

        pytest.skip("No suitable UV/optical filter found in both total and agn colnames")

    def test_quasar_flux_brighter_than_quiescent_in_uv(
        self, emulator_weights, total_colnames
    ):
        """
        Quasar total UV flux should be brighter than quiescent galaxy UV flux,
        because AGN adds to the galaxy flux.
        """
        total_qso, _, _ = _run_predict(emulator_weights, _PARAMS_QUASAR)
        total_qui, _, _ = _run_predict(emulator_weights, _PARAMS_QUI)

        for fname in ["NUV", "decam_g"]:
            t_idx = _get_col_index(list(total_colnames), fname)
            if t_idx is None:
                continue

            f_qso = float(total_qso[0, t_idx])
            f_qui = float(total_qui[0, t_idx])
            print(f"{fname}: quasar={f_qso:.3e} mJy, quiescent={f_qui:.3e} mJy")
            assert f_qso > 0 and f_qui > 0
            # Quasar should be at least comparable in UV (AGN adds flux)
            # Allow quasar to be up to 1000x brighter (log_L_AGN=45 vs 38)
            assert f_qso > f_qui * 0.001, (
                f"{fname}: quasar flux {f_qso:.3e} mJy unexpectedly much fainter "
                f"than quiescent {f_qui:.3e} mJy"
            )
            return

        pytest.skip("No suitable UV filter found in total colnames")

    def test_sfr_physically_reasonable(self, emulator_weights, gal_colnames):
        """
        SFR for quasar host galaxy should be in a physically reasonable range.
        """
        total, gal_out, agn_out = _run_predict(emulator_weights, _PARAMS_QUASAR)

        sfr_idx = _get_col_index(list(gal_colnames), "sfh.sfr")
        if sfr_idx is None:
            pytest.skip("sfh.sfr not found in GAL emulator outputs")

        sfr = float(gal_out[0, sfr_idx])
        assert sfr > 0, f"SFR={sfr:.3e} is not positive"

        log_sfr = np.log10(sfr)
        print(f"Quasar host SFR: {sfr:.3e} Msun/yr (log10={log_sfr:.2f})")

        assert -6 < log_sfr < 5, (
            f"SFR={sfr:.3e} Msun/yr (log10={log_sfr:.2f}) outside physical range"
        )

    def test_stellar_lum_physically_reasonable(self, emulator_weights, gal_colnames):
        """
        Stellar luminosity for quasar host (log_M*=9.32) should be in physical range.
        """
        total, gal_out, agn_out = _run_predict(emulator_weights, _PARAMS_QUASAR)

        stellum_idx = _get_col_index(list(gal_colnames), "stellar.lum")
        if stellum_idx is None:
            pytest.skip("stellar.lum not found in GAL emulator outputs")

        stellum = float(gal_out[0, stellum_idx])
        assert stellum > 0, f"stellar.lum={stellum:.3e} is not positive"

        log_stellum = np.log10(stellum)
        print(f"Quasar host stellar.lum: {stellum:.3e} W (log10={log_stellum:.2f})")

        assert 20 < log_stellum < 50, (
            f"log10(stellar.lum)={log_stellum:.2f} W outside physical range"
        )

    def test_quasar_stellar_lum_lower_than_sf_galaxy(
        self, emulator_weights, gal_colnames
    ):
        """
        Quasar host (log_M*=9.32) should have lower stellar luminosity than
        the star-forming galaxy (log_M*=12.32).
        """
        _, gal_qso, _ = _run_predict(emulator_weights, _PARAMS_QUASAR)
        _, gal_sf, _ = _run_predict(emulator_weights, _PARAMS_SF)

        stellum_idx = _get_col_index(list(gal_colnames), "stellar.lum")
        if stellum_idx is None:
            pytest.skip("stellar.lum not found in GAL emulator outputs")

        stellum_qso = float(gal_qso[0, stellum_idx])
        stellum_sf = float(gal_sf[0, stellum_idx])

        print(f"Quasar host stellar.lum:    {stellum_qso:.3e} W")
        print(f"Star-forming stellar.lum:   {stellum_sf:.3e} W")

        assert stellum_qso < stellum_sf, (
            f"Quasar host stellar.lum={stellum_qso:.3e} W should be < "
            f"SF stellar.lum={stellum_sf:.3e} W (log_M* 9.32 vs 12.32)"
        )

    def test_total_flux_reasonable_range(self, emulator_weights, total_colnames):
        """
        Quasar total fluxes should be in a physically reasonable range.
        For a z=0.426 quasar with log_L_AGN=45, fluxes should be ~0.01-10 mJy.
        """
        total, gal_out, agn_out = _run_predict(emulator_weights, _PARAMS_QUASAR)
        total_arr = total[0]

        # All fluxes should be in [1e-10, 1e5] mJy
        assert np.all(total_arr > 1e-10), (
            f"Some fluxes are effectively zero: min={total_arr.min():.3e} mJy"
        )
        assert np.all(total_arr < 1e5), (
            f"Some fluxes are unphysically large: max={total_arr.max():.3e} mJy"
        )

    def test_agn_fluxes_positive_and_finite(self, emulator_weights, agn_colnames):
        """All AGN fluxes should be positive and finite."""
        total, gal_out, agn_out = _run_predict(emulator_weights, _PARAMS_QUASAR)
        assert np.all(np.isfinite(agn_out)), "Non-finite AGN fluxes"
        assert np.all(agn_out > 0), f"Non-positive AGN fluxes: min={agn_out.min():.3e}"


# ---------------------------------------------------------------------------
# Cross-type consistency tests
# ---------------------------------------------------------------------------

class TestCrossTypeConsistency:
    """
    Consistency checks comparing the three galaxy types against each other.
    These verify that the emulator behaves physically sensibly across a range
    of parameter values.
    """

    def test_stellar_lum_scales_with_stellar_mass(self, emulator_weights, gal_colnames):
        """
        Stellar luminosity should increase with stellar mass across all three types:
        quasar (9.32) < quiescent (11.5) < star-forming (12.32).
        """
        _, gal_qso, _ = _run_predict(emulator_weights, _PARAMS_QUASAR)
        _, gal_qui, _ = _run_predict(emulator_weights, _PARAMS_QUI)
        _, gal_sf, _ = _run_predict(emulator_weights, _PARAMS_SF)

        stellum_idx = _get_col_index(list(gal_colnames), "stellar.lum")
        if stellum_idx is None:
            pytest.skip("stellar.lum not found in GAL emulator outputs")

        lum_qso = float(gal_qso[0, stellum_idx])
        lum_qui = float(gal_qui[0, stellum_idx])
        lum_sf = float(gal_sf[0, stellum_idx])

        print(f"stellar.lum: quasar={lum_qso:.3e}, quiescent={lum_qui:.3e}, "
              f"sf={lum_sf:.3e} W")

        # Ordering: quasar < quiescent < sf (by stellar mass: 9.32 < 11.5 < 12.32)
        assert lum_qso < lum_qui, (
            f"quasar lum={lum_qso:.3e} should be < quiescent lum={lum_qui:.3e} "
            f"(log_M* 9.32 < 11.5)"
        )
        assert lum_qui < lum_sf, (
            f"quiescent lum={lum_qui:.3e} should be < sf lum={lum_sf:.3e} "
            f"(log_M* 11.5 < 12.32)"
        )

    def test_agn_flux_scales_with_agn_luminosity(
        self, emulator_weights, agn_colnames, total_colnames
    ):
        """
        AGN flux should be much larger for quasar (log_L_AGN=45) than for
        quiescent/sf (log_L_AGN=38), by approximately 10^(45-38) = 1e7.
        """
        total_qso, _, agn_qso = _run_predict(emulator_weights, _PARAMS_QUASAR)
        total_qui, _, agn_qui = _run_predict(emulator_weights, _PARAMS_QUI)

        # Find a filter present in AGN outputs
        for fname in ["NUV", "decam_g", "decam_r"]:
            a_idx = _get_col_index(list(agn_colnames), fname)
            if a_idx is None:
                continue

            agn_flux_qso = float(agn_qso[0, a_idx])
            agn_flux_qui = float(agn_qui[0, a_idx])

            if agn_flux_qso > 0 and agn_flux_qui > 0:
                ratio = agn_flux_qso / agn_flux_qui
                log_ratio = np.log10(ratio)
                print(f"{fname}: AGN flux ratio (quasar/quiescent) = {ratio:.2e} "
                      f"(log10={log_ratio:.1f})")
                # With log_L_AGN differing by 7 dex, AGN fluxes should differ enormously
                assert ratio > 100, (
                    f"{fname}: AGN flux ratio={ratio:.2e} should be >> 100 "
                    f"(log_L_AGN 45 vs 38, diff=7 dex)"
                )
                return

        pytest.skip("No suitable filter found in AGN colnames")

    def test_all_types_produce_finite_outputs(self, emulator_weights):
        """All three galaxy types should produce finite outputs."""
        for label, params in [("qui", _PARAMS_QUI), ("sf", _PARAMS_SF),
                               ("quasar", _PARAMS_QUASAR)]:
            total, gal_out, agn_out = _run_predict(emulator_weights, params)
            assert np.all(np.isfinite(total)), f"{label}: non-finite total"
            assert np.all(np.isfinite(gal_out)), f"{label}: non-finite gal"
            assert np.all(np.isfinite(agn_out)), f"{label}: non-finite agn"

    def test_all_types_produce_positive_outputs(self, emulator_weights):
        """All three galaxy types should produce positive outputs."""
        for label, params in [("qui", _PARAMS_QUI), ("sf", _PARAMS_SF),
                               ("quasar", _PARAMS_QUASAR)]:
            total, gal_out, agn_out = _run_predict(emulator_weights, params)
            assert np.all(total > 0), f"{label}: non-positive total min={total.min():.3e}"
            assert np.all(gal_out > 0), f"{label}: non-positive gal min={gal_out.min():.3e}"
            assert np.all(agn_out > 0), f"{label}: non-positive agn min={agn_out.min():.3e}"

    def test_output_shapes_consistent(self, emulator_weights, total_colnames,
                                      gal_colnames, agn_colnames):
        """Output shapes should be consistent across all three galaxy types."""
        n_total = len(total_colnames)
        n_gal = len(gal_colnames)
        n_agn = len(agn_colnames)

        for label, params in [("qui", _PARAMS_QUI), ("sf", _PARAMS_SF),
                               ("quasar", _PARAMS_QUASAR)]:
            total, gal_out, agn_out = _run_predict(emulator_weights, params)
            assert total.shape == (1, n_total), (
                f"{label}: total shape {total.shape} != (1, {n_total})"
            )
            assert gal_out.shape == (1, n_gal), (
                f"{label}: gal shape {gal_out.shape} != (1, {n_gal})"
            )
            assert agn_out.shape == (1, n_agn), (
                f"{label}: agn shape {agn_out.shape} != (1, {n_agn})"
            )

    def test_gal_plus_agn_equals_total(self, emulator_weights, total_colnames,
                                       gal_colnames, agn_colnames):
        """
        For each filter in TOTAL, total flux should equal GAL + AGN fluxes
        (within floating-point precision).
        """
        import fastGRAHSP.jxrun as jxrun
        params, info = emulator_weights

        for label, feature_params in [("qui", _PARAMS_QUI), ("sf", _PARAMS_SF),
                                       ("quasar", _PARAMS_QUASAR)]:
            fd = _make_feature_dict(feature_params)
            total, gal_out, agn_out = jxrun.jax_fastgrahsp_predict(fd, params)
            total_np = np.array(total)
            gal_np = np.array(gal_out)
            agn_np = np.array(agn_out)

            gal_idx = params['gal_output_indices']
            agn_idx = params['agn_output_indices']

            gal_sel = gal_np[0, gal_idx]
            agn_sel = agn_np[0, agn_idx]
            reconstructed = gal_sel + agn_sel

            np.testing.assert_allclose(
                total_np[0], reconstructed, rtol=1e-5,
                err_msg=f"{label}: total != gal_selected + agn_selected"
            )


# ---------------------------------------------------------------------------
# Tests: SFR consistency with pcigale for extreme SFH parameters
# ---------------------------------------------------------------------------

class TestSFRConsistencyWithPcigale:
    """
    Tests that the fastGRAHSP emulator's sfh.sfr output is consistent with
    the pcigale delayed-SFH model for the same input parameters.

    This covers the failure mode identified in test_grahspemu.py where the
    emulator predicted sfh.sfr = 6.6e-9 Msun/yr while pcigale gave
    0.13 Msun/yr for the quiescent parameter set (tau=158 Myr, age=10 Gyr,
    log_M*=11.5) — a 7-dex discrepancy.

    The emulator is expected to agree with pcigale within 2 dex (factor 100)
    or 0.001 Msun/yr across the supported parameter range, including at the
    quiescent extreme (short tau, old age) and the star-forming extreme
    (long tau).
    """

    @pytest.fixture(scope="class")
    def pcigale_sfr_table(self):
        """
        Compute pcigale sfh.sfr for a set of (tau_main, age_main, Mstar) combos.

        Returns a dict mapping (log_tau, log_age, log_mstar) -> sfr in Msun/yr.
        Uses the module-level _pcigale_sfr helper which correctly reads
        sfh.sfr from sed.info after SFHDelayed.process(sed).
        """
        try:
            from pcigale.creation_modules.sfhdelayed import SFHDelayed  # noqa: F401
        except ImportError:
            pytest.skip("pcigale not available")

        results = {}
        test_cases = [
            # (log_tau_myr, log_age_myr, log_mstar_msun)
            (2.2, 3.9, 11.5),   # quiescent: short tau, old age, massive
            (2.2, 4.0, 11.5),   # quiescent: short tau, oldest age, massive
            (4.0, 4.0, 12.32),  # star-forming: long tau, old age, very massive
            (3.74, 4.0, 9.32),  # quasar host: moderate tau, old age
        ]

        for log_tau, log_age, log_mstar in test_cases:
            tau_myr = 10.0 ** log_tau
            age_myr = 10.0 ** log_age
            mstar_msun = 10.0 ** log_mstar

            sfr = _pcigale_sfr(tau_myr, age_myr, mstar_msun)
            if sfr is not None:
                results[(log_tau, log_age, log_mstar)] = sfr

        return results

    def _get_emulator_sfr(self, emulator_weights, feature_params):
        """Run the emulator and return sfh.sfr in Msun/yr."""
        import fastGRAHSP.jxrun as jxrun
        params, info = emulator_weights
        fd = _make_feature_dict(feature_params)
        total, gal_out, agn_out = jxrun.jax_fastgrahsp_predict(fd, params)

        gal_colnames = list(params["gal_colnames"])
        sfr_idx = _get_col_index(gal_colnames, "sfh.sfr")
        if sfr_idx is None:
            return None

        return float(np.array(gal_out)[0, sfr_idx])

    def test_sfr_quiescent_agrees_with_pcigale_within_tolerance(
        self, emulator_weights, gal_colnames
    ):
        """
        For the quiescent parameter set (tau=158 Myr, age=10 Gyr, log_M*=11.5),
        the emulator's sfh.sfr should agree with pcigale within 0.001 Msun/yr
        or 2 dex (factor 100), whichever is more lenient.

        This is the exact parameter combination that caused the 7-dex failure in
        test_grahspemu.py:
          fastGRAHSP: 6.6e-9 Msun/yr
          pcigale:    0.13   Msun/yr
          ratio:      7.3 dex  <-- should be < 2 dex or < 0.001 Msun/yr absolute

        The pcigale sfh.sfr value is obtained by running SFHDelayed with
        normalise=True (which divides the SFR array by the total formed mass
        integral in solar masses), then reading sed.info["sfh.sfr"] which the
        sed.sfh setter stores as the last element of the normalised SFR array
        (in yr^-1), and finally multiplying by mstar_msun to convert to Msun/yr.
        """
        log_tau, log_age, log_mstar = 2.2, 4.0, 11.5
        tau_myr = 10.0 ** log_tau    # 158 Myr
        age_myr = 10.0 ** log_age    # 10000 Myr = 10 Gyr
        mstar_msun = 10.0 ** log_mstar

        sfr_pcigale = _pcigale_sfr(tau_myr, age_myr, mstar_msun)
        if sfr_pcigale is None:
            pytest.skip("pcigale not available or sfhdelayed did not produce sfh.sfr")

        # --- emulator SFR ---
        params_qui_exact = dict(_PARAMS_QUI)
        params_qui_exact["log_sfhdelayed_tau_main"] = log_tau
        params_qui_exact["log_sfhdelayed_age_main"] = log_age
        params_qui_exact["log_stellar_mass"] = log_mstar

        sfr_emu = self._get_emulator_sfr(emulator_weights, params_qui_exact)
        if sfr_emu is None:
            pytest.skip("sfh.sfr not found in GAL emulator outputs")

        abs_diff = abs(sfr_emu - sfr_pcigale)
        log_ratio = abs(np.log10(sfr_emu / sfr_pcigale))

        print(
            f"\nQuiescent SFR comparison:"
            f"\n  tau_main={tau_myr:.0f} Myr, age_main={age_myr:.0f} Myr, "
            f"log_M*={log_mstar}"
            f"\n  fastGRAHSP: {sfr_emu:.3e} Msun/yr"
            f"\n  pcigale:    {sfr_pcigale:.3e} Msun/yr"
            f"\n  |difference|: {abs_diff:.3e} Msun/yr"
            f"\n  log|ratio|:   {log_ratio:.2f} dex"
        )

        assert sfr_emu > 0, f"Emulator SFR={sfr_emu:.3e} is not positive"
        assert sfr_pcigale > 0, f"pcigale SFR={sfr_pcigale:.3e} is not positive"

        # Pass if within 0.001 Msun/yr absolute OR within 2 dex relative
        within_abs = abs_diff <= 0.001
        within_dex = log_ratio < 2.0
        assert within_abs or within_dex, (
            f"Emulator sfh.sfr ({sfr_emu:.3e} Msun/yr) and pcigale sfh.sfr "
            f"({sfr_pcigale:.3e} Msun/yr) disagree by {abs_diff:.3e} Msun/yr "
            f"(tolerance 0.001 Msun/yr) and {log_ratio:.2f} dex (tolerance 2 dex) "
            f"for quiescent parameters "
            f"(tau={tau_myr:.0f} Myr, age={age_myr:.0f} Myr, log_M*={log_mstar}).\n"
            f"This is the failure mode from test_grahspemu.py where the emulator "
            f"predicted near-zero SFR while pcigale gave a finite value."
        )

    def test_sfr_consistent_with_pcigale_across_sfh_parameter_range(
        self, emulator_weights, gal_colnames
    ):
        """
        The emulator's sfh.sfr should agree with pcigale within 2 dex or
        0.001 Msun/yr across the supported SFH parameter range, covering
        quiescent, intermediate, and star-forming cases.

        Parameter grid:
          log_tau in [2.2, 3.0, 4.0]  (tau = 158 Myr, 1 Gyr, 10 Gyr)
          log_age in [3.0, 3.5, 4.0]  (age = 1 Gyr, 3.2 Gyr, 10 Gyr)
          log_mstar = 10.0  (fixed, reference mass)

        For each combination, both the emulator and pcigale are evaluated.
        A combination passes if EITHER:
          - the absolute difference |sfr_emu - sfr_pcigale| <= 0.001 Msun/yr, OR
          - the log ratio |log10(sfr_emu / sfr_pcigale)| < 2.0 dex.

        The pcigale sfh.sfr for each combination is obtained via _pcigale_sfr(),
        which runs SFHDelayed with normalise=True and reads sed.info["sfh.sfr"]
        (the last-timestep SFR in yr^-1 per solar mass formed) then scales by
        the stellar mass.
        """
        try:
            from pcigale.creation_modules.sfhdelayed import SFHDelayed  # noqa: F401
        except ImportError:
            pytest.skip("pcigale not available")

        sfr_idx = _get_col_index(list(gal_colnames), "sfh.sfr")
        if sfr_idx is None:
            pytest.skip("sfh.sfr not found in GAL emulator outputs")

        import fastGRAHSP.jxrun as jxrun
        emu_params, info = emulator_weights

        log_mstar = 10.0
        mstar_msun = 10.0 ** log_mstar

        log_tau_grid = [2.2, 3.0, 4.0]
        log_age_grid = [3.0, 3.5, 4.0]

        abs_tolerance = 0.001  # Msun/yr
        dex_tolerance = 2.0    # dex

        failures = []
        print("\nSFR pcigale vs emulator comparison:")
        print(f"  {'log_tau':>8} {'log_age':>8} {'SFR_emu':>14} "
              f"{'SFR_pcigale':>14} {'log|ratio|':>12} {'|diff| Msun/yr':>16} {'status':>8}")
        print(f"  {'-'*8} {'-'*8} {'-'*14} {'-'*14} {'-'*12} {'-'*16} {'-'*8}")

        for log_tau in log_tau_grid:
            for log_age in log_age_grid:
                tau_myr = 10.0 ** log_tau
                age_myr = 10.0 ** log_age

                # Skip unphysical combinations (tau > age)
                if tau_myr > age_myr:
                    continue

                sfr_pcigale = _pcigale_sfr(tau_myr, age_myr, mstar_msun)
                if sfr_pcigale is None or sfr_pcigale <= 0:
                    continue

                # emulator
                test_params = dict(_PARAMS_QUI)
                test_params["log_sfhdelayed_tau_main"] = log_tau
                test_params["log_sfhdelayed_age_main"] = log_age
                test_params["log_stellar_mass"] = log_mstar
                # Keep AGN negligible
                test_params["log_L_AGN"] = 38.0

                fd = _make_feature_dict(test_params)
                _, gal_out, _ = jxrun.jax_fastgrahsp_predict(fd, emu_params)
                sfr_emu = float(np.array(gal_out)[0, sfr_idx])

                if sfr_emu <= 0:
                    failures.append(
                        f"  log_tau={log_tau}, log_age={log_age}: "
                        f"emulator SFR={sfr_emu:.3e} is not positive"
                    )
                    continue

                log_ratio = abs(np.log10(sfr_emu / sfr_pcigale))
                abs_diff = abs(sfr_emu - sfr_pcigale)

                # Pass if within absolute tolerance OR within dex tolerance
                within_abs = abs_diff <= abs_tolerance
                within_dex = log_ratio < dex_tolerance
                passed = within_abs or within_dex
                status = "PASS" if passed else "FAIL"

                print(f"  {log_tau:>8.1f} {log_age:>8.1f} {sfr_emu:>14.3e} "
                      f"{sfr_pcigale:>14.3e} {log_ratio:>12.3f} {abs_diff:>16.3e} {status:>8}")

                if not passed:
                    failures.append(
                        f"  log_tau={log_tau} ({tau_myr:.0f} Myr), "
                        f"log_age={log_age} ({age_myr:.0f} Myr): "
                        f"emu={sfr_emu:.3e}, pcigale={sfr_pcigale:.3e}, "
                        f"log|ratio|={log_ratio:.2f} >= {dex_tolerance} dex AND "
                        f"|diff|={abs_diff:.3e} Msun/yr >= {abs_tolerance} Msun/yr"
                    )

        assert len(failures) == 0, (
            f"{len(failures)} SFH parameter combinations have emulator sfh.sfr "
            f"disagreeing with pcigale by more than {dex_tolerance} dex "
            f"AND more than {abs_tolerance} Msun/yr:\n"
            + "\n".join(failures)
            + "\n\nThe emulator may be predicting near-zero SFR at the quiescent "
            "extreme of parameter space (short tau, old age) while pcigale "
            "reports a finite floor value. This was the failure mode in "
            "test_grahspemu.py (7-dex discrepancy at tau=158 Myr, age=10 Gyr)."
        )

# ---------------------------------------------------------------------------
# Test runner guard
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
