import os
import numpy as np
from scipy.interpolate import RegularGridInterpolator


class XFluxModel:
    """Conversion model to compute observed fluxes for unobscured and obscured Active Galactic Nuclei."""

    def __init__(self, colname, lumdistfunc, tablename='28-0.4-avg-400-0'):
        """Initialise.

        Parameters
        -----------
        colname: str
            Column name to use from table, must be one of:
            
                * 'soft': Chandra 0.5-2keV
                * 'full': Chandra 0.5-7keV -- actually 
                * 'hard': Chandra 2-7keV -- can be compared to results from a ECF of 2.625E-11 corresponding to a PhoIndex=1.4
                * 'ehard': eROSITA 2.3-5keV -- can be compared to results from a ECF of (1 / 1.102e+11 * 1.56 * 1.8) corresponding to a PhoIndex=2
                * 'emain': eROSITA 0.6-2.3keV  -- can be compared to results from a ECF of (1 / 1.054e+12 * 0.29 * 1.8) corresponding to a PhoIndex=2
                * 'uhrd': NuSTAR 3-24keV
                * 'bat': BAT 14-195keV
            
            Where indicated above, the fluxes are actually converted to net source counts,
            and then converted back to fluxes with a ECF. This gives a more realistic comparison 
            to real data.
            Responses used are from COSMOS (NuSTAR), 
            CDF-S 4Ms (Chandra, see BXA example source 179) and eFEDS (eROSITA).

        lumdistfunc: func
            
            Function to compute the distance in cm^2.
            
            if using cosmolopy:

                cosmo = {'omega_M_0' : 0.3, 'omega_lambda_0' : 0.7, 'h' : 0.7}
                cosmo = cosmolopy.distance.set_omega_k_0(cosmo)
                dist = cosmolopy.distance.quick_distance_function(cosmolopy.distance.luminosity_distance, zmax=8, **cosmo)
                def lumdistfunc(z):
                    return dist(z) * cosmolopy.constants.Mpc_cm
            
            if using astropy:
            
                from astropy.cosmology import FlatLambdaCDM
                import astropy.units as u
                cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
                def lumdistfunc(z):
                    return cosmo.luminosity_distance(z).to(u.cm)

        tablename: array

            Obscurer geometry to choose (differences only matter for Compton-thick AGN).
            These give the X-ray flux conversion tables from UXCLUMPY. 

            Available geometries are named: 
            `{TORSigma}-{CTKcover}-avg-{Ecut}-{fracscat}`,
            where the allowed values are:

                * TORSigma (in degrees): 8, 28, 60 
                * CTKcover: 0, 0.1, 0.2, 0.4
                * Ecut: 400, 300
                * fracscat: 0, 0.1, 0.01 (relative normalisation of second, unobscured powerlaw)

            The default and recommended geometry is '28-0.4-avg-400-0'.

        Returns
        ----------
        functions: dict
            dictionary of functions, namely:
            
            * photon_log10flux_for
            * areafunc
            * invareafunc
            * get_grid
            * get_grid_averaged_flux
        """

        table = np.loadtxt(os.path.join(os.path.dirname(__file__), f'toruscruxux-{tablename}.txt.gz'))
        ncolumns = table.shape[1]
        columns = ['Gamma', 'nH', 'z',
                'softref', 'fullref', 'hardref', 'ehardref', 'emainref', 'uhrdref', 'batref',
                'soft', 'full', 'hard', 'ehard', 'emain', 'uhrd', 'bat'][:ncolumns]
        z = table[:,2]
        iref = columns.index(colname + 'ref')
        iobs = columns.index(colname)

        F = table[:,iref]
        L = F * (lumdistfunc(z)**2 * 4 * np.pi)
        FperL = table[:,iobs] / L


        zbins = np.concatenate((np.logspace(-3, -1, 15)[:-1], np.linspace(0.1, 7, 81)))
        table_grids = [np.linspace(1., 3., 11), np.linspace(20, 26, 43), zbins]

        # now we have luminosity columns that convert to fluxes
        # namely f/L conversion factors
        def get_grid(Gamma, log10L):
            """Get integration grid.
        
        
            Parameters
            -----------
            Gamma: array
                value of gamma to look up
            log10L: array
                log-luminosities to apply

            Returns
            ----------
            log10L: array
                log-luminosities
            z: array
                redshift
            nH: array
                log-column densities
            flux: array
                log-fluxes
            """
            mask = table[:,0] == Gamma
            maskedTable = table[mask]
            nH = maskedTable[:,1]
            z = maskedTable[:,2]
            flux = log10L + np.log10(FperL[mask]).reshape((-1,1))
            return log10L + 0*flux, z, nH, flux

        # restore to hyper-grid
        l = 0
        grid = np.nan * np.zeros((len(table_grids[0]), len(table_grids[1]), len(table_grids[2])))
        for i, Gamma in enumerate(table_grids[0]):
            for j, nHi in enumerate(table_grids[1]):
                lslice = slice(l, l + len(table_grids[2]))
                np.testing.assert_allclose(table[lslice, 0], Gamma)
                np.testing.assert_allclose(table[lslice, 1], nHi)
                np.testing.assert_allclose(table[lslice, 2], table_grids[2])
                grid[i,j] = FperL[lslice]
                l += len(table_grids[2])
        assert np.isfinite(grid).all()

        # nice that github user "JohannesBuchner" put this class into scipy
        self.interpolator = RegularGridInterpolator(table_grids, grid)

    def obsflux_per_restL(self, Gamma, nH, z):
        """compute the flux-to-luminosity ratio.
        
        Specifically, this gives the difference of the
        absorbed, observed-frame flux in the band (colname) in erg/s/cm^2
        to the intrinsic 2-10keV rest-frame luminosity in erg/s.

        Parameters
        -----------
        Gamma: array
            value of the photon index to look up
        nH: array
            log-column densities
        z: array
            redshift

        Returns
        -------
        array
            grid values at given location
        """
        #print 'interpolating for', Gamma, nH, z
        inputshape = np.shape(nH)
        assert np.shape(Gamma) == inputshape
        assert np.shape(z) == inputshape
        coords = [np.asarray(v).reshape(-1,1) for v in [Gamma, nH, z]]
        coordsT = np.transpose(coords)
        coordsT2 = np.column_stack([Gamma, nH, z])
        assert np.allclose(coordsT, coordsT2), (coordsT.shape, coordsT2.shape)
        return self.interpolator(coordsT2).reshape(inputshape)

    def photon_log10flux_for(self, log10L, Gamma, nH, z):
        """Compute observed-frame flux.
    
        Parameters
        -----------
        log10L: array
            log-luminosities, in erg/s, log10
        Gamma: array
            Photon index
        z: array
            redshift
        nH: array
            column densities in 1/cm^2, log10

        Returns
        ----------
        flux: array
            observed-frame flux in erg/s/cm^2, log10.
        """
        return np.log10(self.obsflux_per_restL(Gamma, nH, z)) + log10L

def main():
    """example run."""
    import cosmolopy
    cosmo = {'omega_M_0' : 0.3, 'omega_lambda_0' : 0.7, 'h' : 0.7}
    cosmo = cosmolopy.distance.set_omega_k_0(cosmo)
    dist = cosmolopy.distance.quick_distance_function(cosmolopy.distance.luminosity_distance, zmax=8, **cosmo)
    def lumdistfunc1(z):
        return dist(z) * cosmolopy.constants.Mpc_cm

    from astropy.cosmology import FlatLambdaCDM
    import astropy.units as u
    cosmo2 = FlatLambdaCDM(H0=70, Om0=0.3)
    def lumdistfunc2(z):
        return cosmo2.luminosity_distance(z).to(u.cm)
    
    for band in 'hard', 'soft':
        for lumdistfunc in lumdistfunc1, lumdistfunc2:
            model = XFluxModel(band, lumdistfunc)
            import matplotlib.pyplot as plt
            N = 100
            for zi in 0.1, 1, 2, 3:
                Gamma = 2 + np.zeros(N)
                nH = np.linspace(20.01, 25.9, N)
                z = zi + np.zeros(N)
                L = 45 + np.zeros(N)
                plt.plot(nH, model.photon_log10flux_for(L, Gamma, nH, z), label=zi)
            plt.legend(title='Redshift')
            plt.xlabel(r'Hydrogen-equivalent Column density $N_\mathrm{H}$ [cm$^{-2}$], log')
            plt.ylabel('Observed X-ray Flux (Chandra 2-7keV band)')
            plt.savefig(f'Xflux_nH_{band}.pdf')
            plt.close()
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw=dict(hspace=0.04))
            Gamma = 2 + np.zeros(N)
            z = np.linspace(0.1, 6, N)
            nH = 21 + np.log10((1 + z)**3.3)  # Gilli+22 suggestion
            L = 45 + np.zeros(N)
            ax1.plot(z, nH, label='Gilli+22')
            ax1.set_ylabel(r'$\log N_\mathrm{H}$ [cm$^{-2}$]')
            ax1.legend()
            ax2.plot(z, model.photon_log10flux_for(L, Gamma, nH, z), label=r'$N_\mathrm{H}$ increases with redshift')
            nH = 20 + 0 * z
            ax2.plot(z, model.photon_log10flux_for(L, Gamma, nH, z), color='lightgray', ls='--', label='unobscured for comparison')
            ax2.set_ylabel('Observed X-ray Flux (2-7keV)')
            ax2.set_xlabel('Redshift')
            ax2.legend()
            plt.savefig(f'Xflux_z_{band}.pdf')
            plt.close()

        
if __name__ == '__main__':
    main()
