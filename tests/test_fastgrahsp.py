import matplotlib.pyplot as plt
import numpy as np
from fastGRAHSP import predict_fluxes
from numpy.testing import assert_allclose


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
        M=np.random.uniform(7, 12, size=N),
        L5100A=np.random.uniform(38, 46, size=N),
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
    for width in 128, 256, 1024:
        print("NN width:", width)
        results = predict_fluxes(emulator_args, width=width)
        total_fluxes, total_columns, GAL_fluxes, GAL_columns, AGN_fluxes, AGN_columns = results
        i = total_columns.index('WISE1')
        #print(total_fluxes[:, i])  # WISE1 flux in mJy
        assert total_fluxes.shape == (N, len(total_columns))
        assert GAL_fluxes.shape == (N, len(GAL_columns))
        assert AGN_fluxes.shape == (N, len(AGN_columns))
        for i, band in enumerate(total_columns):
            if not ('GAL_' + band in GAL_columns):
                continue
            if not ('AGN_' + band in AGN_columns):
                continue
            j = GAL_columns.index('GAL_' + band)
            k = AGN_columns.index('AGN_' + band)
            #print(i, j, k, band)
            assert np.isfinite(total_fluxes[:,i]).all()
            assert np.isfinite(AGN_fluxes[:,k]).all()
            assert np.isfinite(GAL_fluxes[:,j]).all()
            assert_allclose(GAL_fluxes[:,j] + AGN_fluxes[:,k], total_fluxes[:,i])


# Helper: construct emulator_args from a parameter dictionary (vectorized over arrays)
def make_emulator_args(params, defaults):
    # params: dict of arrays (same length) or scalars; returns (n, 23) array
    # Make sure arrays
    def arr(x, n):
        if np.ndim(x) == 0:
            return np.full(n, x, dtype=float)
        x = np.asarray(x, dtype=float)
        return x

    # Determine n
    n = None
    for v in params.values():
        if np.ndim(v) > 0:
            n = len(v)
            break
    if n is None:
        n = 1

    p = {k: arr(params.get(k, defaults[k]), n) for k in defaults.keys()}

    # Safety for logs: replace non-positive with tiny positive to avoid -inf
    tiny = 1e-30
    for key in ['tau','age','AFeII','Alines','HOTfcov','plbendwidth','EBV','EBV_AGN']:
        if np.any(p[key] <= 0):
            p[key] = np.maximum(p[key], tiny)

    args_list = [
        np.log10(p['tau']),
        np.log10(p['age']),
        np.log10(p['AFeII']),
        np.log10(p['Alines']),
        p['linewidth'],
        p['Si'],
        p['fcov'],
        p['COOLlam'],
        p['COOLwidth'],
        p['HOTlam'],
        p['HOTwidth'],
        np.log10(p['HOTfcov']),
        p['plslope'],
        p['plbendloc'],
        np.log10(p['plbendwidth']),
        p['uvslope'],
        np.log10(p['EBV']),
        np.log10(p['EBV_AGN']),
        p['alpha'],
        p['M'],
        p['L5100A'],
        p['z'],
        p['EBV'] + p['EBV_AGN'],
    ]
    return np.vstack(args_list).T  # shape (n, 23)


# Plotting loop
def scan_and_plot(predict_fluxes,
                  param_specs,
                  defaults,
                  n_points=100,
                  widths=(256,),   # set to (128,256,512) if you want all
                  y_clip_min=1e-30, plot=True):

    for name, scale, vmin, vmax, default in param_specs:
        # Skip parameters without a range
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            continue

        # Create scan values
        if scale == 'log':
            # Ensure strictly positive
            vmin_pos = vmin if vmin > 0 else 1e-30
            vmax_pos = vmax if vmax > 0 else 1e-30
            xs = np.logspace(np.log10(vmin_pos), np.log10(vmax_pos), n_points)
        else:
            xs = np.linspace(vmin, vmax, n_points)

        print(f"Parameter grid for {name}: {xs}")
        # Build parameter dict for this scan (others fixed at defaults)
        params = {k: defaults[k] for k in defaults.keys()}
        params[name] = xs

        # Build emulator args and predict
        emulator_args = make_emulator_args(params, defaults=defaults)

        # For each band, make one plot with GAL, AGN, TOTAL vs parameter
        # Assume columns align across total/GAL/AGN
        figures = {}
        for width in widths:
            total_fluxes, total_columns, GAL_fluxes, GAL_columns, AGN_fluxes, AGN_columns = predict_fluxes(
                emulator_args, width=width
            )
            for bi, band in enumerate(total_columns):
                if not ('GAL_' + band in GAL_columns):
                    continue
                if not ('AGN_' + band in AGN_columns):
                    continue
                j = GAL_columns.index('GAL_' + band)
                k = AGN_columns.index('AGN_' + band)
                y_total = np.clip(total_fluxes[:, bi], y_clip_min, None)
                y_gal   = np.clip(GAL_fluxes[:, j],   y_clip_min, None)
                y_agn   = np.clip(AGN_fluxes[:, k],   y_clip_min, None)

                assert np.isfinite(y_total).all()
                assert np.isfinite(y_gal).all()
                assert np.isfinite(y_agn).all()
                logstd_agn = np.std(np.log10(y_agn))
                logstd_gal = np.std(np.log10(y_gal))
                eps = 0.04
                if logstd_agn > eps and logstd_gal > eps:
                    print(f"    {name} influences {band}, GAL & AGN")
                elif logstd_agn > eps:
                    print(f"    {name} influences {band}, AGN")
                elif logstd_gal > eps:
                    print(f"    {name} influences {band}, GAL")
                else:
                    print(f"    {name} DOES NOT influence {band}")
                if not plot:
                    continue

                xtitle = f"{name} ({scale})"
                print(f"plotting {name} {width} {band} ...")
                if band not in figures:
                    figures[band] = plt.subplots(figsize=(6, 4.5))
                fig, ax = figures[band]

                ax.plot(xs, y_total, label='TOTAL', color='k', lw=2)
                ax.plot(xs, y_gal,   label='GAL',   color='C0', lw=1.8)
                ax.plot(xs, y_agn,   label='AGN',   color='C3', lw=1.8)

                ax.set_xscale('log' if scale == 'log' else 'linear')
                ax.set_yscale('log')
                ax.set_xlabel(xtitle)
                ax.set_ylabel(f'Flux in {band} [mJy]')
                ax.set_title(f'{band} vs {name} (width={width})')
                ax.grid(True, which='both', ls=':', alpha=0.5)
                ax.legend()

                # Save under parameter name (include band and width to avoid overwrite)
        for band, (fig, ax) in figures.items():
            if not plot:
                continue
            fig.savefig(f"test_sens_{name}_{band}.pdf")
            plt.close(fig)
            print(f"  plotted to test_sens_{name}_{band}.pdf")

def test_sensitivity(plot=False):
    defaults = dict(
        tau=10000,
        age=10000,
        AFeII=1,
        Alines=1,
        linewidth=2000,
        Si=0,
        fcov=0.5,
        COOLlam=15,
        COOLwidth=0.4,
        HOTfcov=1,
        HOTlam=2.0,
        HOTwidth=0.4,
        plslope=-2,
        plbendloc=2.0,
        plbendwidth=0.1,
        uvslope=0,
        EBV=0.01,
        EBV_AGN=0.01,
        alpha=2.0,
        z=0.5,
        M=9,
        L5100A=42,
    )

    # Condensed parameter specifications:
    # (name, 'log' or 'linear', min, max, default)
    # The ranges match the first dictionary you provided, converting 10**uniform to explicit numeric ranges.
    param_specs = [
        ('tau',          'log',     1.0,              1.0e4,           defaults['tau']),
        ('age',          'log',     1.0,              1.0e4,           defaults['age']),
        ('AFeII',        'log',     1.0e-1,           1.0e1,           defaults['AFeII']),
        ('Alines',       'log',     1.0e-1,           1.0e1,           defaults['Alines']),
        ('linewidth',    'linear',  1.0e3,            3.0e4,           defaults['linewidth']),
        ('Si',           'linear', -5.0,              5.0,             defaults['Si']),
        ('fcov',         'linear',  0.0,              1.0,             defaults['fcov']),
        ('COOLlam',      'linear', 12.0,              28.0,            defaults['COOLlam']),
        ('COOLwidth',    'linear',  0.3,              0.9,             defaults['COOLwidth']),
        # HOTfcov was sampled 0..10 but emulator uses log10(HOTfcov); avoid 0 in log-axis by using a tiny positive floor
        ('HOTfcov',      'log',     1.0e-6,           1.0e1,           defaults['HOTfcov']),
        ('HOTlam',       'linear',  1.1,              4.3,             defaults['HOTlam']),
        ('HOTwidth',     'linear',  0.3,              0.9,             defaults['HOTwidth']),
        ('plslope',      'linear', -2.6,             -1.3,             defaults['plslope']),
        # plbendloc sampling was 10**uniform(1.7, 2.3) ≈ 50.1..199.5
        ('plbendloc',    'log',     10**1.7,          10**2.3,         defaults['plbendloc']),
        ('plbendwidth',  'log',     10**(-2.0),       10**0.0,         defaults['plbendwidth']),
        # uvslope had no range in your generator (always 0). Keep for completeness; loop will skip since min==max.
        ('uvslope',      'linear',  0.0,              0.0,             defaults['uvslope']),
        ('EBV',          'log',     10**(-2.0),       10**(-1.0),      defaults['EBV']),
        ('EBV_AGN',      'log',     10**(-2.0),       10**(1.0),       defaults['EBV_AGN']),
        ('alpha',        'linear',  1.2,              2.7,             defaults['alpha']),
        # z was clipped to [0, 6] in your generator
        ('z',            'linear',  0.0,              6.0,             defaults['z']),
        ('M',            'linear',  7,                12,              defaults['M']),
        ('L5100A',       'linear',  38,               46,              defaults['L5100A']),
    ]

    scan_and_plot(
        predict_fluxes,
        param_specs,
        defaults,
        n_points=100,
        widths=(128,),   # set to (128,256,512) if you want all
        y_clip_min=1e-30,
        plot=plot)

if __name__ == '__main__':
    test_sensitivity(plot=True)
