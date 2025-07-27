#!/usr/bin/env python3

import os

core_num = "1"
os.environ["OMP_NUM_THREADS"] = core_num  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = core_num  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = core_num  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = core_num  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = core_num  # export NUMEXPR_NUM_THREADS=6
import numpy as np
import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# from template_bank.template_bank_generation_parameters import *
from template_bank_generation_parameters import *
from astropy.io import fits
import pipeline_parameters as pp


def get_masses_radii_densities_selection():
    # Looking at the Gaia info of the Kepler stars to see which limits we have and what is the distribution
    gaia_file = fits.open(pp.Gaia_kepler_file)

    all_b_r = []
    all_abs_gmag = []
    all_teff = []
    all_masses = []
    all_radii = []

    for index in range(len(gaia_file[1].data['kepid'])):
        b_r = (gaia_file[1].data['bp_rp'])[index]  # color
        parallax = (gaia_file[1].data['parallax'])[index]  # parallaxes (needed for absolute magnitude)
        phot_g_mag = (gaia_file[1].data['phot_g_mean_mag'])[index]  # photometric g magnitudes
        teff = (gaia_file[1].data['teff'])[index]
        if parallax > 0:
            abs_gmag = phot_g_mag + 5. * (np.log10(parallax / 1.e3) + 1.)  # calculating absolute magnitude from Bailer-Jones distances
            all_b_r.append(b_r)
            all_abs_gmag.append(abs_gmag)
            all_teff.append(teff)
            all_masses.append((gaia_file[1].data['mass'])[index])  # in sol masses
            all_radii.append((gaia_file[1].data['radius'])[index])  # in sol radii

    all_b_r = np.array(all_b_r)
    all_abs_gmag = np.array(all_abs_gmag)
    all_teff = np.array(all_teff)
    all_masses = np.array(all_masses)
    all_radii = np.array(all_radii)

    good_inds = ((all_radii <= Max_stellar_radius_in_solar_radii)
                 & (all_radii >= Min_stellar_radius_in_solar_radii)
                 & (all_masses <= Max_stellar_mass_in_solar_masses)
                 & (all_masses >= Min_stellar_mass_in_solar_masses)
                 & (~np.isnan(all_masses)) & (~np.isnan(all_radii)))
    # -> radii 10 are for Jupiter-likes, masses are because of the stellar type

    len(all_abs_gmag[good_inds]) / len(all_abs_gmag)

    good_masses = all_masses[good_inds]
    good_radii = all_radii[good_inds]
    m_1_3_r_all = all_masses ** (1 / 3) / all_radii
    m_1_3_r_good = good_masses ** (1 / 3) / good_radii
    return all_masses, all_radii, m_1_3_r_all, good_masses, good_radii, m_1_3_r_good, all_b_r, all_abs_gmag, all_teff


if __name__ == "__main__":
    # import matplotlib
    import matplotlib.pyplot as plt

    plt.rcParams.update({'figure.figsize': (8, 5), 'font.size': 13, 'font.serif': 'Times New Roman',
                         'lines.linewidth': 2, 'legend.fontsize': 12})
    def_cols = plt.rcParams['axes.prop_cycle'].by_key()['color']

    All_masses, All_radii, M_1_3_r_all, Good_masses, Good_radii, M_1_3_r_good, _, _, _ = get_masses_radii_densities_selection()
    np.save(M_1_3_r_selected_list_file, M_1_3_r_good, allow_pickle=True)

    fig = plt.figure()
    bins = np.linspace(0, 2.5, 100)
    plt.hist(M_1_3_r_all, bins=bins, color=(0.8, 0.8, 0.8), label='All stars')

    plt.hist(M_1_3_r_all[All_radii < Min_stellar_radius_in_solar_radii],
             bins=bins, linewidth=2,
             label='R<{:.1f}'.format(Min_stellar_radius_in_solar_radii) + r' $R_{\odot}$')
    plt.hist(M_1_3_r_all[All_radii > Max_stellar_radius_in_solar_radii],
             bins=bins, linewidth=2,
             label='R>{:.1f}'.format(Max_stellar_radius_in_solar_radii) + r' $R_{\odot}$')

    plt.hist(M_1_3_r_all[All_masses > Max_stellar_mass_in_solar_masses],
             bins=bins, linewidth=2, histtype='step',
             label='M>{:.1f}'.format(Max_stellar_mass_in_solar_masses) + r' $M_{\odot}$')
    plt.hist(M_1_3_r_all[All_masses < Min_stellar_mass_in_solar_masses],
             bins=bins, linewidth=2, histtype='step',
             label='M<{:.1f}'.format(Min_stellar_mass_in_solar_masses) + r' $M_{\odot}$')

    plt.hist(M_1_3_r_good, bins=bins, histtype='step', linewidth=2, label='selected stars')

    for i_lim, lim in enumerate(M13r_bin_edges_template_prior):
        plt.axvline(lim, color='k', linestyle='--', linewidth=2, label='Prior bin edges' if i_lim == 0 else None)

    plt.xlabel(r'$(M/M_{\odot})^{1/3} / (R/R_{\odot})$')
    plt.ylabel('Number of stars in Gaia-Kepler catalog')
    plt.yscale('log')
    plt.legend()
    plt.grid(zorder=0, color=(0.5, 0.5, 0.5), linewidth=0.5)

    fig.savefig(os.path.join(Templates_preparation_folder, 'Stars_densities_distribution.png'), dpi=320, bbox_inches='tight')
    plt.close(fig)
    plt.show()

    # print((1/109.076)**2)  # earth-sun depth
    # print((1/(109.076 * 0.5))**2)  # earth-half-sun depth
    # print(2/(2*np.pi*215) * 365 * 24)  # duration earth-sun
    # print(0.2/(2*np.pi*215) * 50 * 24 / 0.5)  # duration smaller star short period
