#!/usr/bin/env python3

import os

os.environ["OMP_NUM_THREADS"] = "1"  #
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"  #
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pipeline_parameters as pp


# ################################################################################################
# FILES
# ################################################################################################
Gaia_kepler_file = pp.Gaia_kepler_file
Gaia_fields = [
    'SOURCE_ID',
    'ra',  # Barycentric right ascension of the source in ICRS at the reference epoch ref_epoch
    'ra_error',
    'dec',  # Barycentric declination of the source in ICRS at the reference epoch ref_epoch
    'dec_error',
    'parallax',
    'parallax_error',
    'pm',
    'pmra',
    'pmra_error',
    'pmdec',
    'pmdec_error',
    'astrometric_excess_noise',
    'astrometric_excess_noise_sig',
    'pseudocolour',
    'pseudocolour_error',
    'astrometric_sigma5d_max',
    'ruwe',
    'phot_g_mean_flux',
    'phot_g_mean_flux_error',
    'phot_g_mean_mag',
    'phot_bp_rp_excess_factor',
    'bp_rp',
    'bp_g',
    'g_rp',
    'radial_velocity',
    'radial_velocity_error',
    'rv_expected_sig_to_noise',
    'rv_renormalised_gof',
    'rv_chisq_pvalue',
    'has_rvs',
    'has_epoch_rv',
    'vbroad',
    'vbroad_error',
    'non_single_star',
    'phot_variable_flag',
    'dist',
    'rv_template_fe_h',
    'rv_template_logg',
    'rv_template_teff',
    'teff_gspphot',
    'logg_gspphot',
    'rv_nb_transits'
]
Koi_dict_file = pp.Koi_dict_file
Ttv_dict_file = pp.Ttv_dict_file
Villanova_dict_file = pp.Villanova_dict_file
Target_pixels_location = pp.Target_pixels_location

Kepler_files_folder = pp.Kepler_files_folder

Earth_pointing_times_file = os.path.join(Kepler_files_folder, 'earth_pointing_times.txt')
Quarter_start_times_file = os.path.join(Kepler_files_folder, 'quarter_start_times.txt')
Quarter_end_times_file = os.path.join(Kepler_files_folder, 'quarter_end_times.txt')
Villanova_catalog_file = os.path.join(Kepler_files_folder, 'villanova_catalog')
Villanova_periods_phases_kmags_relevant_filename = 'villanova_periods_phases_kmags_relevant.npy'
Villanova_periods_phases_kmags_relevant_file = os.path.join(Kepler_files_folder, Villanova_periods_phases_kmags_relevant_filename)
Table_koi_file = os.path.join(Kepler_files_folder, 'KOI_table_2024.csv')

Initial_trigger_list_filename = pp.Threshold_crossing_list_filename
Suspicious_frequency_reasons_triggers_filename = 'suspicious_frequency_reasons_triggers.npy'
Suspicious_frequency_filtered_triggers_filename = 'suspicious_frequency_filtered_triggers.npy'
Gaia_trigger_table_filename = 'gaia_trigger_table.ecsv'

Initial_trigger_list_file = os.path.join(pp.Threshold_crossing_lists_folder, Initial_trigger_list_filename)
Suspicious_frequency_reasons_triggers_file = os.path.join(pp.Threshold_crossing_lists_folder,
                                                          Suspicious_frequency_reasons_triggers_filename)
Suspicious_frequency_filtered_triggers_file = os.path.join(pp.Threshold_crossing_lists_folder,
                                                           Suspicious_frequency_filtered_triggers_filename)
Gaia_trigger_file = os.path.join(pp.Threshold_crossing_lists_folder, Gaia_trigger_table_filename)

Neighbors_folder = os.path.join(pp.Triggers_output_location, 'Neighbors_run_' + pp.Run_name)
Summary_tce_txt_filename_postfix = 'tce_summary.txt'
Dict_tce_txt_filename_postfix = 'tce_class_dict.txt'
Pickle_tce_small_attributes_filename_postfix = 'tce_class_small_attributes.pickle'
Pickle_tce_all_attributes_filename_postfix = 'tce_class_all_attributes.pickle'
Problems_tce_txt_filename_postfix = 'tce_problems.txt'

# ################################################################################################
# TECHNICAL PARAMS
# ################################################################################################
Pixel_scale = 3.98  # arcsec
Neighbors_search_radius_pix = 5  # how many pixels around to search for neighbors
# ----------------------------------------------------------------------
# PSF:
# The resulting 95% encircled energy diameters range from 3.2 to 7.5 pixels
# https://ntrs.nasa.gov/api/citations/20190028937/downloads/20190028937.pdf
#
# The Kepler field of view spans 115.6 square degrees over 95 million detector pixels, with a 3.98 x 3.98 arcsec
# pixels and 3.1-7.5 arcsec FWHM imaging point spread function.
# https://nexsci.caltech.edu/workshop/2012/keplergo/DataAnalysisProducts.shtml#:~:text=The%20Kepler
# %20field%20of%20view,FWHM%20imaging%20point%20spread%20function.
# ----------------------------------------------------------------------

# CONSTANTS AND ARGUMENTS
Quarter_16_end = 1557.9589375032883
# Kepler time is measured in BJD - 2454833
# Villanova phases are in BJD - 2400000
# And we corrected for this


# ################################################################################################
# TESTS PARAMETERS
# ################################################################################################
# Veto target pixels parameters:
Part_of_cumsum_2_targ_pix = 0.85
Nsig_outliers_0_pos_tests = 5.
Nsig_outliers_0_neg_tests = 15.
Scaling_factor_centroid_flux = 10000.  # Multiplying centroid series by this to avoid numerical precision issues
Threshold_centroid_snr_2 = 45.
Threshold_false_negative_bg_target_pixels_mean = 0.01
Threshold_false_negative_bg_target_pixels_max = 0.01
Fn_fold_veto = 0.01
Threshold_fraction_same_season = 0.76  # if 3 out of 4 transits are same season, it is yet ok

# ################################################################################################
# DEFINING HOW CLOSE TO KNOWN PERIODICITIES THE SIGNALS ARE ALLOWED TO BE
# there are some harmonics which are fast:
# exposure time 29.424 min
# reaction wheels 3.03 days, 0.133 days
Period_year = 370.  # 365.25 #year
Period_quarter = 93.  # quarter
# 30., #monthly downlink is very irregular and will be considered separately
Period_error_year = 10.  # duty cycles as estimation for year period error
Period_error_year_harm = 2.
Period_error_quarter = 2.  # let it be for quarter
Period_error_quarter_harm = 1.
Period_error = 0.03  # days
Phase_error = 0.07  # around 3 lc

# Kepler time is measured in BJD - 2454833
# Villanova phases are in BJD - 2400000
# And we corrected for this

Dict_bad_periodicity_reason_names = {
    'y': 'year',
    'q': 'quarter',
    'v': 'villanova',
    'd': 'downlink',
    'k': 'koi',
    'yh': 'year_harmonics',
    'qh': 'quarter_harmonics',
    'vh': 'villanova_harmonics',
    '': '',
}

# ################################################################################################
# ERRORS AROUND SPECIFIC TIMES
Timing_error_quarter_edge = 2.  # days. contamination from quarter edge can last. just a random estimate
Timing_error_earth_downlink = 1.  # days. contamination from earth pointing and downlink can last. just a random estimate
Min_fraction_valid_transits = 0.3  # this part of transits maximally can be at the quarter edges


# ################################################################################################
# LOADING SUSPICIOUS TIMES

def get_earth_pointing_times(earth_pointing_times_file=Earth_pointing_times_file):
    with open(earth_pointing_times_file, 'r') as fil1:
        earth_pointing_times = np.array(fil1.read().split()).astype(float)
    return earth_pointing_times


def get_quarter_start_end_times(quarter_start_times_file=Quarter_start_times_file):
    with open(quarter_start_times_file, 'r') as fil1:
        quarter_start_times = np.array(fil1.read().split()).astype(float)

    with open(Quarter_end_times_file, 'r') as fil1:
        quarter_end_times = np.array(fil1.read().split()).astype(float)
    return quarter_start_times, quarter_end_times


def get_villanova_relevant_periods_phases_kmags(villanova_periods_phases_kmags_relevant_file=Villanova_periods_phases_kmags_relevant_file):
    """

    :param villanova_periods_phases_kmags_relevant_file:
    :type villanova_periods_phases_kmags_relevant_file:
    :return: vil_periods, vil_phases, vil_kmags
    :rtype: numpy.ndarray of float, numpy.ndarray of float, numpy.ndarray of float
    """
    vil_periods, vil_phases, vil_kmags = np.load(villanova_periods_phases_kmags_relevant_file, allow_pickle=True)
    return vil_periods, vil_phases, vil_kmags


def get_strong_koi_periodicities(koi_table_file=Table_koi_file):
    # KNOWN PLANETS
    with open(koi_table_file, 'r') as table_koi_file:
        lines = table_koi_file.readlines()
    lines = [line for line in lines if line[0] != '#']
    line_descriptions = (lines[0].strip()).split(',')
    koi_model_snr_index = line_descriptions.index('koi_model_snr')
    koi_period_index = line_descriptions.index('koi_period')
    koi_time0bk_index = line_descriptions.index('koi_time0bk')
    koi_kepid_index = line_descriptions.index('kepid')

    lines = lines[1:]  # remove header
    lines = [(line.strip()).split(',') for line in lines]
    lines = [line for line in lines if (len(line[koi_model_snr_index]) > 0)]  # select known snr
    lines = [line for line in lines if (float(line[koi_model_snr_index]) > 40.)]  # select big snr

    periods_koi = np.array([float(line[koi_period_index]) for line in lines])
    ftts_koi = np.array([float(line[koi_time0bk_index]) for line in lines])
    kid_kois = np.array([line[koi_kepid_index].rjust(9, '0') for line in lines])
    return periods_koi, ftts_koi, kid_kois
