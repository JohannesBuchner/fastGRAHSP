import numpy as np
from fastGRAHSP import predict_fluxes


def test_mock():
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
    assert total_fluxes.shape == (N, len(total_columns))
    assert GAL_fluxes.shape == (N, len(GAL_columns))
    assert AGN_fluxes.shape == (N, len(AGN_columns))
