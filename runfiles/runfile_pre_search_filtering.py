#!/usr/bin/env python3

import os
os.environ["OMP_NUM_THREADS"] = "1"  #
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"  #
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import numpy as np
from astropy.io import fits
from astropy.table import Table
import pickle
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from template_bank.template_bank_generation_parameters import (Max_stellar_radius_in_solar_radii,
                                                               Min_stellar_radius_in_solar_radii, Max_stellar_mass_in_solar_masses,
                                                               Min_stellar_mass_in_solar_masses,
                                                               Min_logg, Max_logg, Min_teff, Max_teff)
import search_pipeline as spip
import pipeline_parameters as pp
import auxiliary_functions as af
from locations import (Gaia_kepler_file, Villanova_dict_file, Koi_dict_file,
                       Good_tids_search_postfix, Good_tids_candidates_postfix,
                       Pre_search_filtering_folder)

import warnings
# from astropy.utils.exceptions import AstropyWarning, UnitsWarning
from astropy.units import UnitsWarning
# warnings.simplefilter('ignore', category=UnitsWarning)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--prefix", type=str, nargs=1, help="KIC number prefix", default="0030")
parser.add_argument("--last_digit", type=str, nargs=1, help="last digit of KIC number", default="0")
args = parser.parse_args()
prefix = args.prefix[0]
last_digit = args.last_digit[0]


# Functions
#%%
def bad_fft_check_local(orig_fluxes, amp_max_bad_fft=pp.Amp_max_bad_fft, dif_max_bad_fft=pp.Dif_max_bad_fft):
    """Same as bad_fft_check of target_class.KeplerTarget, but for local use."""
    is_good = True
    for q in range(len(orig_fluxes)):
        lc = np.array(orig_fluxes[q] - np.nanmean(orig_fluxes[q]))
        if len(lc) > 2000:
            lc[np.where(np.isnan(lc))] = np.nanmean(lc)
            is_good_1 = not af.is_bad_fft(np.abs(np.fft.rfft(lc)), amp_max=amp_max_bad_fft, dif_max=dif_max_bad_fft)
            is_good &= is_good_1
    return is_good


def check_detectability_local(target_id, data_location, output_location, stellar_radius,
                              rp_re_max=pp.Rp_re_max, rp_rs_max=pp.Rp_rs_max, per_min_candidate=pp.Per_min_candidate):
    """Almost same as check_detectable_radius of target_class.KeplerTarget, but for local use."""
    # --------------------------------------------------------------------------------------------
    # GETTING THE PSD

    klc = spip.KeplerLightCurve(target_id=target_id,
                                data_location=data_location,
                                output_location=output_location,
                                allow_data_folder_creation=False, allow_output_folder_creation=False)

    is_good_fft = bad_fft_check_local(klc.orig_fluxes)
    n_quarters = len(klc.quarters)
    mean_flux = klc.mean_total

    nsig_outliers_0 = spip.Nsig_outliers_0
    nsig_outliers_0_pos = spip.Nsig_outliers_0_pos
    nsig_outliers_0_neg = spip.Nsig_outliers_0_neg
    nsig_outliers_psd = spip.Nsig_outliers_psd
    nperseg_psd = spip.Nperseg_psd
    npsdsmin = spip.Npsdsmin
    max_small_hole = spip.Max_small_hole
    detr_length_outliers = spip.Detr_length_outliers
    tukey_steps_len = spip.Tukey_steps_len_psd
    save_low_trends = False
    klc.nperseg_psd = nperseg_psd
    try:
        klc.subtract_means(save_means=True, masks=None)
        klc.mask_flux_outliers(nsig=nsig_outliers_0, nsig_positive=nsig_outliers_0_pos,
                               nsig_negative=nsig_outliers_0_neg,
                               nsig_clean=nsig_outliers_psd, detr_length_outliers=detr_length_outliers,
                               in_whitened_fluxes=False, modify_clean_masks=True)
        klc.subtract_means(save_means=True, masks=klc.masks)
        klc.normalize_by_means()
        klc.make_big_hole_masks(max_small_hole=max_small_hole, exclude_individual_points=True)
        klc.interpolate_fluxes(interpolate_clean_fluxes=True)
        klc.subtract_low_trends(nperseg=klc.nperseg_psd, save_low_trends=save_low_trends)
        klc.fill_additional_outliers(nsig=nsig_outliers_0, nsig_positive=nsig_outliers_0_pos,
                                     nsig_negative=nsig_outliers_0_neg,
                                     nsig_clean=nsig_outliers_psd, detr_length_outliers=detr_length_outliers,
                                     in_whitened_fluxes=False, modify_clean_masks=True,
                                     interpolate_clean_fluxes=True)  # again after piecewise detrending
        klc.prepare_for_psd(nperseg=klc.nperseg_psd, npsdsmin=npsdsmin, max_small_hole=max_small_hole)
        klc.pad_by_zeros()  # Needed to avoid crosstalk in whitening filter. And we want the end zeros to be actual zeros
        klc.measure_psds(nperseg=klc.nperseg_psd, tukey_steps_len=tukey_steps_len)
        klc.whiten_fluxes(whiten_clean_fluxes=True)
        klc.condition_templates()
    except ValueError as e:
        with open('/home/projects/barakz/orynai/run_13/hpc/tid_errors.txt', 'a') as f:
            print(target_id, f"Error in target {target_id}: {e}", file=f)
            print(file=f)
            del klc
            raise ValueError(f"Error in target {target_id}: {e}")

    # From `target_class.KeplerTarget.check_detectable_radius`
    units_of_re_to_units_of_r_star_factor = 1 / (stellar_radius * pp.R_sun_R_e)
    rp_rs = units_of_re_to_units_of_r_star_factor * rp_re_max
    rp_rs = min(rp_rs, rp_rs_max)

    n_transits_max = int(np.ceil(len(klc.quarters) * 90 / per_min_candidate))

    norms = np.array([np.nanmax(klc.tb.templates_shifts[kk]) for kk in range(len(klc.whitened_templates_td))])
    prods = np.array([np.mean([np.linalg.norm(klc.whitened_templates_td[kk][q])
                                   for q in klc.quarters], axis=0) / np.nanmax(klc.tb.templates_shifts[kk])
                          for kk in range(len(klc.whitened_templates_td))])
    if np.isnan(prods).any():
        with open('/home/projects/barakz/orynai/run_13/hpc/tid_errors.txt', 'a') as f:
            print(target_id, n_transits_max, file=f)
            print(norms, file=f)
            print(prods, file=f)
            print(file=f)
        raise ValueError(f"ALL-NaN found in prods for target {target_id}")


    snr_max = (np.nanmax([np.mean([np.linalg.norm(klc.whitened_templates_td[kk][q])
                                   for q in klc.quarters], axis=0) / np.nanmax(klc.tb.templates_shifts[kk])
                          for kk in range(len(klc.whitened_templates_td))])
               * klc.mean_total * n_transits_max ** 0.5 * (rp_rs ** 2))
    # -> for an exact conversion between the snr and the radius, look dedicated functions

    del klc
    return snr_max, is_good_fft, n_quarters, mean_flux

#%%
# --------------------------------------------------------------------------------------------


Baseline_dir = '/home/projects/barakz'
Working_dir = os.path.join(Baseline_dir, 'Collaboration-Kepler')
Rotation_periods_file = os.path.join(Working_dir, 'KeplerFiles', 'rotation_periods_tsevi', 'J_ApJS_211_24_table1.dat.gz.fits')
Data_folder = os.path.join(Working_dir, 'KeplerData')
target_ids = [tid for tid in os.listdir(os.path.join(Data_folder, prefix))
              if (os.path.isdir(os.path.join(Data_folder, prefix, tid))
              and (len(tid) == 9)
              and tid.isnumeric()
              and tid[-1] == last_digit)]

dict_results = {}
dict_results['target_id'] = np.array(target_ids, dtype=str)

# ----------------------------------------------------------------------
# Gaia
gaia_file = fits.open(Gaia_kepler_file)
kepid_array = gaia_file[1].data['kepid']
id_to_index = {k: i for i, k in enumerate(kepid_array)}
gaia_indices = np.array([id_to_index.get(int(tid), -1) for tid in target_ids], dtype=int)
mask_in_gaia = (gaia_indices >= 0)
gaia_indices_in_gaia = gaia_indices[mask_in_gaia]

names_gaia_dict = [
    ['radius', pp.Radius_field_name_gaia],
    ['mass', pp.Mass_field_name_gaia],
    ['teff', pp.Teff_field_name_gaia],
    ['metallicity', pp.Metallicity_field_name_gaia],
    ['logg', pp.Logg_field_name_gaia],
    ['gaia_phot_g_mean_mag', pp.Phot_g_mean_mag_field_name_gaia],
    ['kepmag', pp.Kepmag_field_name_gaia],
    ['gaia_parallax', pp.Parallax_field_name_gaia],
    ['gaia_bp_rp', pp.Bp_rp_field_name_gaia],
    ['gaia_ra', pp.Ra_field_name_gaia],
    ['gaia_dec', pp.Dec_field_name_gaia],
    ['gaia_ruwe', pp.Ruwe_field_name_gaia],
    ['nconfp', pp.Nconfp_field_name_gaia],
    ['gaia_non_single_star', pp.Non_single_star_field_name_gaia],
    ['gaia_phot_variable_flag', pp.Phot_variable_flag_field_name_gaia]]

for name_dict, name_table in names_gaia_dict:
    dtype = float
    if (name_table == pp.Nconfp_field_name_gaia) or (name_table == pp.Non_single_star_field_name_gaia):
        dtype = int
    elif name_table == pp.Phot_variable_flag_field_name_gaia:
        dtype = gaia_file[1].data[name_table].dtype
    if (dtype == float) or (dtype == np.float64) or (dtype == np.float32):
        vals = np.ones(len(target_ids), dtype=dtype) * np.nan
    else:
        vals = np.zeros(len(target_ids), dtype=dtype)
    vals[mask_in_gaia] = (gaia_file[1].data[name_table][gaia_indices_in_gaia].astype(dtype))
    dict_results[name_dict] = vals
gaia_file.close()

dict_results['provenance'] = np.array([''] * len(target_ids), dtype="<U10")
dict_results['provenance'][mask_in_gaia] = 'gaia'

# ----------------------------------------------------------------------
# Berger 2020 stellar properties. Using if there are no gaia values
from catalog_functions import read_berger_table, Berger_field_names_dict
from locations import Berger_table_file
names_berger_dict = [
    ['radius', Berger_field_names_dict['radius']],
    ['mass', Berger_field_names_dict['mass']],
    ['teff', Berger_field_names_dict['teff']],
    ['metallicity', Berger_field_names_dict['metallicity']],
    ['logg', Berger_field_names_dict['logg']]]

berger_table = read_berger_table(berger_table_file=Berger_table_file)
kepid_array = berger_table['target_id']
id_to_index = {k: i for i, k in enumerate(kepid_array)}
berger_indices = np.array([id_to_index.get(tid, -1) for tid in target_ids], dtype=int)
mask_in_berger = (berger_indices >= 0)
mask_no_properties_and_in_berger = ((np.isnan(dict_results['radius'])
                                    | np.isnan(dict_results['mass'])
                                    | np.isnan(dict_results['logg'])
                                    | np.isnan(dict_results['teff'])
                                    | np.isnan(dict_results['metallicity']))
                                    & mask_in_berger)
indices_for_tids_replace_by_berger = berger_indices[mask_no_properties_and_in_berger]

for name_dict, name_table in names_berger_dict:
    dict_results[name_dict][mask_no_properties_and_in_berger] = ((berger_table[name_table][
        indices_for_tids_replace_by_berger]).data).astype(float)

dict_results['provenance'][mask_no_properties_and_in_berger] = 'berger'

# ----------------------------------------------------------------------
dict_results['is_in_catalog'] = np.zeros(len(target_ids), dtype=bool)
dict_results['is_in_gaia'] = np.zeros(len(target_ids), dtype=bool)
dict_results['is_in_gaia'][mask_in_gaia] = True
dict_results['is_in_catalog'][mask_in_gaia] = True
dict_results['is_in_catalog'][mask_in_berger] = True
dict_results['is_bad_properties'] = np.zeros(len(target_ids), dtype=bool)
dict_results['is_bad_properties'][np.isnan(dict_results['mass'])] = True
dict_results['is_bad_properties'][dict_results['gaia_parallax'] < 0] = True
dict_results['has_gaia_properties'] = np.ones(len(target_ids), dtype=bool)
dict_results['has_gaia_properties'][~mask_in_gaia] = False
dict_results['has_gaia_properties'][mask_no_properties_and_in_berger] = False
dict_results['is_single_gaia'] = np.ones(len(target_ids), dtype=bool)
dict_results['is_single_gaia'][dict_results['gaia_ruwe'] > 1.4] = False  # ATTENTION: NEW in Run_13!
dict_results['is_single_gaia'][dict_results['gaia_non_single_star'] > 0] = False
dict_results['is_single_gaia'][dict_results['gaia_phot_variable_flag'] == 'VARIABLE'] = False
dict_results['is_good_selection'] = np.ones(len(target_ids), dtype=bool)
dict_results['is_good_selection'] &= ((dict_results['radius'] <= Max_stellar_radius_in_solar_radii)
                                      & (dict_results['radius'] >= Min_stellar_radius_in_solar_radii)
                                      & (dict_results['mass'] <= Max_stellar_mass_in_solar_masses)
                                      & (dict_results['mass'] >= Min_stellar_mass_in_solar_masses)
                                      & (dict_results['logg'] <= Max_logg)
                                      & (dict_results['logg'] >= Min_logg)
                                      & (dict_results['teff'] <= Max_teff)
                                      & (dict_results['teff'] >= Min_teff))

msk = dict_results['gaia_parallax'] >= 0
dict_results['gaia_abs_gmag'] = np.ones(len(target_ids), dtype=float) * np.nan
dict_results['gaia_abs_gmag'][msk] = (dict_results['gaia_phot_g_mean_mag'][msk] + 5. * (
        np.log10(dict_results['gaia_parallax'][msk] / 1.e3) + 1.))

dict_results['m_1_3_r_s'] = dict_results['mass'] ** (1 / 3) / dict_results['radius']

# ----------------------------------------------------------------------
# Rotators
# ----------------------------------------------------------------------
with warnings.catch_warnings():
    warnings.simplefilter('ignore', category=UnitsWarning)

    table_rotation_tsevi = Table.read(Rotation_periods_file, format='fits')
    rotating_tids = table_rotation_tsevi['KIC'].data
    rotating_tids = np.array([str(tid).rjust(9, '0') for tid in rotating_tids])
    id_to_index = {k: i for i, k in enumerate(rotating_tids)}
    rotator_indices = np.array([id_to_index.get(tid, -1) for tid in target_ids], dtype=int)
    mask_in_rotators = (rotator_indices >= 0)
    rotator_indices_in_rotators = rotator_indices[mask_in_rotators]

dict_results['is_rotator'] = np.zeros(len(target_ids), dtype=bool)
dict_results['is_rotator'][mask_in_rotators] = True

dict_results['rotator_prot'] = np.ones(len(target_ids), dtype=float) * np.nan
dict_results['rotator_prot'][mask_in_rotators] = table_rotation_tsevi['Prot'][rotator_indices_in_rotators]
dict_results['rotator_rper'] = np.ones(len(target_ids), dtype=float) * np.nan
dict_results['rotator_rper'][mask_in_rotators] = table_rotation_tsevi['Rper'][rotator_indices_in_rotators]

# ----------------------------------------------------------------------
# Check villanova
# ----------------------------------------------------------------------

with open(Villanova_dict_file, 'rb') as villanova_catalog_dict_file:
    villanova_dict = pickle.load(villanova_catalog_dict_file)

dict_results['is_villanova_binary'] = np.zeros(len(target_ids), dtype=bool)
for i_target_id, target_id in enumerate(target_ids):
    if target_id in villanova_dict['KIC']:
        ind = villanova_dict['KIC'].index(target_id)
        dict_results['is_villanova_binary'][i_target_id] = True

# ----------------------------------------------------------------------
# Getting koi info
# ----------------------------------------------------------------------
with open(Koi_dict_file, 'rb') as koi_catalog_dict_file:
    koi_info_dict = pickle.load(koi_catalog_dict_file)

dict_results['is_koi'] = np.zeros(len(target_ids), dtype=bool)
dict_results['n_kois'] = np.zeros(len(target_ids), dtype=int)
dict_results['max_ses_koi'] = np.ones(len(target_ids), dtype=float) * np.nan
dict_results['max_mes_koi'] = np.ones(len(target_ids), dtype=float) * np.nan

kepids = np.array(koi_info_dict['kepid'])  # important to have an array!
koi_max_sngle_evs = np.array(koi_info_dict['koi_max_sngle_ev'])
koi_max_mult_evs = np.array(koi_info_dict['koi_max_mult_ev'])

for i_target_id, target_id in enumerate(target_ids):
    koi_dict = {}
    if target_id in kepids:
        dict_results['is_koi'][i_target_id] = True
        inds = np.where(kepids == target_id)[0]
        dict_results['n_kois'][i_target_id] = len(inds)
        sess = koi_max_sngle_evs[inds]
        sess = np.array([s if s != '' else np.nan for s in sess], dtype=float)
        mess = koi_max_mult_evs[inds]
        mess = np.array([s if s != '' else np.nan for s in mess], dtype=float)
        if np.isnan(sess).all() or len(sess) == 0:
            dict_results['max_ses_koi'][i_target_id] = np.nan
        else:
            dict_results['max_ses_koi'][i_target_id] = np.nanmax(sess)
        if np.isnan(mess).all() or len(mess) == 0:
            dict_results['max_mes_koi'][i_target_id] = np.nan
        else:
            dict_results['max_mes_koi'][i_target_id] = np.nanmax(mess)

# ----------------------------------------------------------------------
# Going target by target and checking the detectable radius

snr_thresh = pp.Score_candidate_search ** 0.5 - 1

dict_results['snr_max_detectable_radius'] = np.ones(len(target_ids), dtype=float) * np.nan
dict_results['is_bad_fft'] = np.zeros(len(target_ids), dtype=bool)
dict_results['n_quarters'] = np.zeros(len(target_ids), dtype=int)
dict_results['mean_flux'] = np.ones(len(target_ids), dtype=float) * np.nan
dict_results['is_too_short'] = np.zeros(len(target_ids), dtype=bool)

for i_target_id, target_id in enumerate(target_ids):
    data_location = os.path.join(Data_folder, target_id[:4], target_id)
    n_data_files = len([f for f in os.listdir(data_location) if 'fits' in f])
    if n_data_files >= 2:
        snr_max, is_good_fft, n_quarters, mean_flux = check_detectability_local(target_id,
                                                                                stellar_radius=dict_results['radius'][i_target_id],
                                                                                data_location=data_location,
                                                                                output_location=None, rp_re_max=pp.Rp_re_max,
                                                                                rp_rs_max=pp.Rp_rs_max,
                                                                                per_min_candidate=pp.Per_min_candidate)
    else:
        snr_max, is_good_fft, n_quarters, mean_flux = (np.nan, False, n_data_files, np.nan)
    dict_results['snr_max_detectable_radius'][i_target_id] = snr_max
    dict_results['is_bad_fft'][i_target_id] = (not is_good_fft)
    dict_results['n_quarters'][i_target_id] = n_quarters
    dict_results['mean_flux'][i_target_id] = mean_flux
    dict_results['is_too_short'][i_target_id] = (n_quarters < pp.Min_quarters_to_process)

# ----------------------------------------------------------------------

dict_results['is_good_for_search'] = ((np.abs(dict_results['snr_max_detectable_radius']) >= snr_thresh)
                                      & (~dict_results['is_villanova_binary'])
                                      & (~dict_results['is_bad_fft'])
                                      & (~dict_results['is_too_short'])
                                      & (dict_results['is_in_gaia'])  # we want the target to be in gaia, although
                                      # not necessarily have all the properties inferred: those we can supplement from berger
                                      & (~dict_results['is_bad_properties'])
                                      & (dict_results['is_good_selection']))

dict_results['is_good_for_candidates'] = ((np.abs(dict_results['snr_max_detectable_radius']) >= snr_thresh)
                                          & (~dict_results['is_villanova_binary'])
                                          & (~dict_results['is_bad_fft'])
                                          & (~dict_results['is_too_short'])
                                          & (dict_results['is_in_gaia'])  # we want the target to be in gaia, although
                                          # not necessarily have all the properties inferred: those we can supplement from berger
                                          & (dict_results['is_single_gaia'])
                                          & (~dict_results['is_bad_properties'])
                                          & (dict_results['is_good_selection']))

# --------------------------------------------------------------------------
# Saving
Results_folder = Pre_search_filtering_folder
prefix_results_folder = os.path.join(Results_folder, prefix)
if not os.path.isdir(prefix_results_folder):
    os.mkdir(prefix_results_folder, mode=0o777)
file_path = os.path.join(prefix_results_folder, prefix + '_' + last_digit + '_stars_table.ecsv')

table_results = Table(dict_results)
table_results.write(file_path, format='ascii.ecsv', overwrite=True)

# Good for search:
good_tids_prefix = dict_results['target_id'][dict_results['is_good_for_search']]
good_tids_prefix_file = os.path.join(Results_folder, prefix + '_' + Good_tids_search_postfix)
with open(good_tids_prefix_file, 'a') as f:
    for tid in good_tids_prefix:
        print(tid, file=f)

# Good for candidates:
good_tids_prefix = dict_results['target_id'][dict_results['is_good_for_candidates']]
good_tids_prefix_file = os.path.join(Results_folder, prefix + '_' + Good_tids_candidates_postfix)
with open(good_tids_prefix_file, 'a') as f:
    for tid in good_tids_prefix:
        print(tid, file=f)
