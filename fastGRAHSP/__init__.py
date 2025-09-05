#!/usr/bin/env python
# coding: utf-8
"""fastGRAHSP: fast neural network simulator for fluxes of galaxies and quasars."""

__version__ = "1.0.3"
__author__ = "Johannes Buchner"

import json
import os

import numpy as np
import onnxruntime as rt


class Emulator:
    """Holder for a ONNX computation model."""

    def __init__(self, filename, **kwargs):
        """Load model.

        Parameters
        -----------
        filename: str
            ONNX file name
        **kwargs: dict
            additional arguments passed to `onnxruntime.InferenceSession`.
        """
        session = sess = rt.InferenceSession(filename, **kwargs)
        self.session = session
        self.input_name = sess.get_inputs()[0].name
        self.output_names = [output.name for output in sess.get_outputs()]

    def predict(self, X):
        """Predict.

        Parameters
        -----------
        X: array
            input features

        Returns
        ----------
        y: array
            prediction of the model.
        """
        output = self.session.run(
            self.output_names,
            {self.input_name: X.astype(np.float32)}
        )
        return {outputname: output[i] for i, outputname in enumerate(self.output_names)}


emulators = {}


def predict_fluxes(Xextended, width=128, providers=["CPUExecutionProvider"], **kwargs):
    """Predict flux.

    Parameters
    ----------
    Xextended: array
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

    width: int
        Neural network model to use, must be one of 128, 256, 512.
        Larger size is slower but more accurate
    providers: list
        passed to `onnxruntime.InferenceSession`
    **kwargs: dict
        additional arguments passed to `onnxruntime.InferenceSession`

    Returns
    -------
    flux: array
        flux in mJy.
    fluxGAL: array
        flux in mJy for galaxy alone
    fluxAGN: array
        flux in mJy for AGN alone
    colnames: list
        name of columns of *flux*.
    colnamesGAL: float
        name of columns of *fluxGAL*.
    colnamesAGN: float
        name of columns of *fluxAGN*.


    Examples
    --------
    ::
        N = 1000
        df = dict(
            tau=np.random.uniform(1, 10000, size=N),
            age=np.random.uniform(1, 10000, size=N),
            AFeII=10**np.random.uniform(-1, 1, size=N),
            Alines=10**np.random.uniform(-1, 1, size=N),
            linewidth=np.random.uniform(1000, 30000, size=N),
            Si=np.random.uniform(-5, 5, size=N),
            fcov=np.random.uniform(0, 1, size=N),
            COOLlam=np.random.uniform(12, 28, size=N),
            COOLwidth=np.random.uniform(0.3, 0.9, size=N),
            HOTfcov=np.random.uniform(0, 10, size=N),
            HOTlam=np.random.uniform(1.1, 4.3, size=N),
            HOTwidth=np.random.uniform(0.3, 0.9, size=N),
            plslope=np.random.uniform(-2.6, -1.3, size=N),
            plbendloc=10**np.random.uniform(1.7, 2.3, size=N),
            plbendwidth=10**np.random.uniform(-2, 0, size=N),
            uvslope = np.zeros(N),
            EBV=10**np.random.uniform(-2, -1, size=N),
            EBV_AGN=10**np.random.uniform(-2, 1, size=N),
            alpha=np.random.uniform(1.2, 2.7, size=N),
            z=np.clip(np.random.normal(size=N)**2, 0, 6),
            M=10**np.random.uniform(7, 12, size=N),
            L5100A=10**np.random.uniform(38, 46, size=N),
        )

        emulator_args = np.transpose([
            np.log10(df['tau']),
            np.log10(df['age']),
            np.log10(df['AFeII']),
            np.log10(df['Alines']),
            df['linewidth'],
            df['Si'],
            df['fcov'],
            df['COOLlam'],
            df['COOLwidth'],
            df['HOTlam'],
            df['HOTwidth'],
            np.log10(df['HOTfcov']),
            df['plslope'],
            df['plbendloc'],
            np.log10(df['plbendwidth']),
            df['uvslope'],
            np.log10(df['EBV']),
            np.log10(df['EBV_AGN']),
            df['alpha'],
            df['M'],
            df['L5100A'],
            df['z'],
            df['EBV'] + df['EBV_AGN'],
        ])
        results = predict_fluxes(emulator_args)
        total_fluxes, total_columns, GAL_fluxes, GAL_columns, AGN_fluxes, AGN_columns = results
        i = total_columns.index('WISE1')
        print(total_fluxes[:, i])  # WISE1 flux in mJy

    """
    key = f"GALAGN{width}"
    if key not in emulators:
        print("loading emulators...")
        filename = os.path.join(os.path.dirname(__file__), key)

        info = json.load(open(filename + ".json"))
        print("emulator expects as input:", info["input"])
        assert Xextended.shape[1] == len(info["input"]), (
            Xextended.shape,
            len(info["input"]),
        )
        for subkey, colnames in info["output"]:
            emulators[key + subkey + "colnames"] = colnames
        emulators[key] = Emulator(
            filename + ".onnx", providers=providers, **kwargs
        )

    outGALAGN = emulators[key].predict(Xextended)
    outGAL = outGALAGN["GAL_outputs_linear_scaled"]
    outAGN = outGALAGN["AGN_outputs_linear_scaled"]
    out = outGALAGN["both_val"]
    assert np.all(outGAL >= 0)
    assert np.all(outAGN >= 0)
    assert np.all(out >= 0)
    return (
        out,
        emulators[key + "TOTALcolnames"],
        outGAL,
        emulators[key + "GALcolnames"],
        outAGN,
        emulators[key + "AGNcolnames"],
    )
