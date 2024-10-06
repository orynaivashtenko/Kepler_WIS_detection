#!/usr/bin/env python3

import os

os.environ["OMP_NUM_THREADS"] = "1"  #
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"  #
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
from locations import *
from catalog_functions import read_table, get_koi_ttv_dicts
from search_pipeline_parameters import Nperseg_psd


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


# ################################################################################################
# TECHNICAL PARAMS
# ################################################################################################
Pixel_scale = 3.98  # arcsec
Neighbors_search_radius_pix = 5  # how many pixels around to search for neighbors
Overlapping_transits_check_radius_pixels = 100
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
Part_of_flux_peak_threshold = 0.3  # One-sigma of a Gaussian 2-d PSF is 0.3 of the peak
Part_of_cumsum_2_targ_pix = None  # 0.85  # Not used, overriden by Part_of_flux_peak_threshold
Number_of_pixels_cut = None
Part_of_light = None
Nsig_outliers_0_pos_tests = 5.
Nsig_outliers_0_neg_tests = 15.
Scaling_factor_centroid_flux = 10000.  # Multiplying centroid series by this to avoid numerical precision issues
Threshold_centroid_snr_2 = 45.
Threshold_false_negative_bg_target_pixels_mean = 0.01
Threshold_false_negative_bg_target_pixels_max = 0.01
# Veto folds parameters:
Fn_fold_veto = 0.01
Fn_parts_veto = 0.005
Threshold_fraction_same_season = 0.76  # if 3 out of 4 transits are same season, it is yet ok
Min_len_part_bins_veto_parts_fold = 4

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
Overlap_timing_error = 0.08  # days  # Error fof defining when our transit overlaps with a known thing

# ################################################################################################
# CATALOG PARAMETERS
Catalog_names_properties_dict = dict([  # name,
                                        # type, dtype, unit, fill_value, description
    ['target_id',
     [str,  '<U10', None, 'None', '',
      'Kepler Input Catalog ID of the target star']],
    ['prob_density_planet_not_noise',
     [float, float, 'prob', np.nan,
      'Probability per period range per trigger that the signal is a planet and not the background noise']],
    ['false_alarm_prob_trigger',
     [float, float, 'prob', np.nan,
      'False alarm probability for best_stat obtained by integrating the extrapolated timeslides histogram']],
    ['best_int_score',
     [float, float, 'SNR^2', np.nan,
      'Best statistical score obtained by the pipeline']],
    ['best_per',
     [float, float, 'days', np.nan,
      'Best period obtained by the pipeline']],
    ['best_stat',
     [float, float, 'SNR^2', np.nan,
      'Best bare statistic obtained by the pipeline']],
    ['best_ftt',
     [float, float, 'days after Kepler reference time', np.nan,
      'Best first transit time obtained by the pipeline']],
    ['transit_duration_hours',
     [float, float, 'hours', np.nan,
      'Mean transit duration corresponding to the best folded template']],
    ['rp_to_r_earth',
     [float, float, 'R_earth', np.nan,
      'Radius of the planet in units of Earth radii inferred by the pipeline and stellar radius by Gaia']],
    ['rp_to_r_earth_std',
     [float, float, 'R_earth', np.nan,
      'Error of the rp_to_r_earth']],
    ['rp_to_rs',
     [float, float, 'stellar_radius', np.nan,
      'Ratio between the planetary radius and the stellar radius inferred by the pipeline']],
    ['rp_to_rs_std',
     [float, float, 'stellar_radius', np.nan,
      'Error of the rp_to_rs']],
    ['koi_status',
     [str, '<U200', '', '',
      'confirmed, false_positive, candidate if is Kepler object of interest']],
    ['kepler_tces',
     [str, '<U200', '', '',
      'periods of tce if is Kepler TCE']],
    ['n_transits',
     [int, int, '', 'None',
      'Number of transits inside the processed data']],
    ['n_valid_transits',
     [int, int, '', 'None',
      'Number of transits inside the processed data that passed the quality tests']],
    ['n_overlapping_koi_transits',
     [int, int, '', 'None',
      'Number of transits overlapping with the known Kepler objects of interest of this target']],
    ['max_season_count',
     [float, float, '', np.nan,
      'Maximal number of transits of the same season']],
    ['centroid_snr_2',
     [float, float, 'SNR^2', np.nan,
      'Statistical score of the centroid deviation during the transits']],
    ['p_value_mean_bg_pixels',
     [float, float, '', np.nan,
      'P-value that the background pixel signal is stronger than the target pixel signal']],
    ['p_value_worst_bg_pixels',
     [float, float, '', np.nan,
      'P-value that the background pixel signal is stronger than the target pixel signal for the worst transit']],
    ['p_value_fold_residuals_time',
     [float, float, '', np.nan,
      'P-value that the residuals of the fold after subtracting the best-fit transit model are deviating']],
    ['duration_folded_hours',
     [float, float, '', np.nan,
      'Duration of transit in hours from the fit of templates to the folded flux']],
    ['p_value_mf_fold_residuals',
     [float, float, '', np.nan,
      """P-value that the matched filtering of the templates with the residuals of the fold after 
      subtracting the best-fit transit model is deviating"""]],
    ['p_value_veto_fold_parts',
     [float, float, '', np.nan,
      'P-value that amplitudes of pieces of folded template are inconsistent with each other']],
    ['p_value_veto_fold_parts_fourier',
     [float, float, '', np.nan,
      'P-value that amplitudes of pieces in Fourier domain of folded template are inconsistent with each other']],
    ['p_value_even_odd_ampl',
     [float, float, '', np.nan,
      'P-value that amplitudes of even and odd transits are inconsistent with each other']],
    ['mean_transit_ppm',
     [float, float, '', np.nan,
      'Mean transit depth in parts per million']],
    ['mean_transit_ppm_std',
     [float, float, '', np.nan,
      'Standard deviation of the mean transit depth in parts per million']],
    ['mean_ppm_variability',
     [float, float, '', np.nan,
      'Additional spread in the ppm of the transits which cannot be explained by the error of the individual transits ppm']],
    ['rp_to_r_earth_std_conservative',
     [float, float, 'Stellar radii', np.nan,
      'Radius error obtained from separate transits']],
    ['best_temp',
     [int, int, 'Index', '',
      'Best template obtained by the pipeline']],
    ['best_temp_folded',
     [int, int, 'Index', '',
      'Best template obtained after the folding and binning']],
    ['best_gauss_stat',
     [float, float, 'SNR^2', np.nan,
      'Best bare statistic after the gaussianization']],
    ['n_quarters',
     [int, int, '', '',
      'Number of Kepler quarters in the data that was processed']],
    ['mean_flux',
     [float, float, 'Kepler Flux Units', np.nan,
      'Mean Kepler flux of the star']],
    ['stellar_radius',
     [float, float, 'R_sun', np.nan,
      'Radius of the target star by Gaia']],
    ['m_1_3_r_s',
     [float, float, 'M_sun^{1/3}/R_sun', np.nan,
      'Function of mass and radius needed for injections']],
    ['stellar_mass',
     [float, float, 'M_sun', np.nan,
      'Mass of the target star by Gaia']],
    ['kepmag',
     [float, float, 'mag', np.nan,
      'Kepler magnitude']],
    ['phot_g_mean_mag',
     [float, float, '', np.nan,
      'Mean g-magnitude of the target by Gaia']],
    ['parallax',
     [float, float, 'mas', np.nan,
      'Parallax of the target by Gaia']],
    ['gaia_abs_gmag',
     [float, float, '', np.nan,
      'Absolute magnitude obtained from parallax']],
    ['gaia_b_r',
     [float, float, '', np.nan,
      'Color by Gaia']],
    ['teff',
     [float, float, 'K', np.nan,
      'Effective surface temperature of the target by Gaia']],
    ['metallicity',
     [float, float, '', np.nan,
      'Metallicity of the target by Gaia']],
    ['logg',
     [float, float, '', np.nan,
      'Log gravity of the target by Gaia']],
    ['ra',
     [float, float, 'deg', np.nan,
      'Right ascension of the target']],
    ['dec',
     [float, float, 'deg', np.nan,
      'Declination of the target']],
    ['ruwe',
     [float, float, 'deg', np.nan,
      'Ruwe of the target by Gaia']],
    ['gaia_nconfp',
     [int, int, '', '',
      'Number of confirmed planets of this target by Gaia']],
    ['gaia_non_single_star_flag',
     [str, '<U36', '', '',
      'Whether is a non-single star by Gaia']],
    ['gaia_variable_star_flag',
     [str, '<U36', '', '',
      'Whether is a variable star by Gaia']],
    ['is_villanova_binary',
     [bool, bool, '', '',
      'Whether is in the Villanova eclipsing binary catalog']],
    ['is_koi_planet_masked',
     [bool, bool, '', '',
      'Whether is Kepler object of interest was masked']],
    ['was_notched',
     [bool, bool, '', '',
      'Whether the pipeline detected and masked harmonic peaks in the spectrum']],
    # ['failures',
    #  [str, '<U200', '', '',
    #   'Some failures that happened in the pipeline run']],
    ['best_g_stat',
     [float, float, 'SNR^2',  np.nan,
      'Like best_stat but from the gaussianization']],
    ['best_g_per',
     [float, float, 'days', np.nan,
      'Like best_per but from the gaussianization']],
    ['best_g_temp',
     [int, int, '', '',
      'Like best_temp but from the gaussianization']],
    ['best_g_ftt',
     [float, float, 'days after Kepler reference time', np.nan,
      'Like best_ftt but from the gaussianization']],
    ['best_g_gauss_stat',
     [float, float, '', np.nan,
      'Like best_gauss_stat but from the gaussianization']],
    ['best_g_int_score',
     [float, float, '', np.nan,
      'Like best_int_score but from the gaussianization']],
    ['prob_density_h0_trigger',
     [float, float, 'prob', np.nan,
      'Probability density for the best_stat value in the background distribution']],
    ['prob_density_h1_trigger',
     [float, float, 'prob', np.nan,
      'Probability density for the best_stat value in the expected true planet distribution']],
    ['is_good_pre_filtering', [bool, bool, '', 'All the tests before the statistical significance estimation were passed']],
    ['too_few_valid_transits', [bool, bool, '', 'Too few transits of good quality and not nan']],
    ['is_bad_seasons', [bool, bool, '', 'Too many transits fall in the same season']],
    ['is_bad_fold_residuals', [bool, bool, '', 'A contaminant was detected in the residuals after folding']],
    ['is_bad_fold_residuals_fft', [bool, bool, '', 'A contaminant was detected in the fft of residuals after folding']],
    ['is_bad_fold_residuals_parts', [bool, bool, '', 'Amplitudes of pieces of template after folding are inconsistent']],
    ['is_bad_fold_parts_time', [bool, bool, '', 'Amplitudes of pieces of template after folding are inconsistent in time domain']],
    ['is_bad_fold_parts_fourier', [bool, bool, '', 'Amplitudes of pieces of template in Fourier after folding are inconsistent']],
    ['is_centroid_bad', [bool, bool, '', 'Centroid deviation has a too high SNR']],
    ['is_bad_bg_pixels', [bool, bool, '',
                          'Depth of the transit of light curve made from background pixels is too high compared to the star pixels']],
    ['is_period_short', [bool, bool, '', 'The best period is shorter than the considered range']],
    ['is_bad_periodicity', [bool, bool, '', 'The periodicity was found to coincide with a known contaminant']],
    ['bad_periodicity_reasons', [str, '', '', 'The contaminants that the periodicity was found to coincide with']],
    ['is_bad_even_odd_ampl', [bool, bool, '', 'The amplitudes of the even and odd transits are inconsistent']],
    ['is_too_short', [bool, bool, '', 'Was marked as too short data for the search']],
    ['is_bad_fft', [bool, bool, '', 'Was marked by the pipeline as not quiet enough spectrum']],
    ['is_koi', [bool, bool, '', 'The star is Kepler object of interest']]
    ])

Flags = dict([  # name, type, value
    ['too_few_valid_transits', [bool, False]],
    ['is_bad_seasons', [bool, False]],
    ['is_bad_fold_residuals', [bool, False]],
    ['is_bad_fold_residuals_fft', [bool, False]],
    ['is_bad_fold_residuals_parts', [bool, False]],
    ['is_bad_fold_parts_time', [bool, False]],
    ['is_bad_fold_parts_fourier', [bool, False]],
    ['is_centroid_bad', [bool, False]],
    ['is_bad_bg_pixels', [bool, False]],
    ['is_period_short', [bool, False]],
    ['is_bad_periodicity', [bool, False]],
    ['is_bad_even_odd_ampl', [bool, False]],
    ['is_good_pre_filtering', [bool, True]],
    ['is_too_short', [bool, False]],
    ['is_bad_fft', [bool, False]],
    ])


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
    vil_kids, vil_periods, vil_phases, vil_kmags = np.load(villanova_periods_phases_kmags_relevant_file, allow_pickle=True)
    return vil_periods, vil_phases, vil_kmags


def get_villanova_table(villanova_kic_relevant_file=Villanova_kic_relevant_file):
    return read_table(file_path=villanova_kic_relevant_file, fill_values=None, table_format=None)


def get_strong_koi_periodicities(koi_table_file=Table_koi_file, return_coords=False, return_magnitudes=False,
                                 snr_model_trheshold=40., koi_max_sngle_ev=None, koi_kepmag=None):
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
    lines = [line for line in lines if line[koi_model_snr_index] != '' and (
            float(line[koi_model_snr_index]) > snr_model_trheshold)]  # select big snr
    if koi_max_sngle_ev is not None:
        koi_max_sngle_ev_index = line_descriptions.index('koi_max_sngle_ev')
        lines = [line for line in lines if line[koi_max_sngle_ev_index] != '' and (float(line[koi_max_sngle_ev_index]) > koi_max_sngle_ev)]
    if koi_kepmag is not None:
        koi_kepmag_index = line_descriptions.index('koi_kepmag')
        lines = [line for line in lines if line[koi_kepmag_index] == '' or (float(line[koi_kepmag_index]) < koi_kepmag)]

    periods_koi = np.array([float(line[koi_period_index]) for line in lines])
    ftts_koi = np.array([float(line[koi_time0bk_index]) for line in lines])
    kid_kois = np.array([line[koi_kepid_index].rjust(9, '0') for line in lines])

    res_list = [periods_koi, ftts_koi, kid_kois]
    if return_coords:
        ra_index = line_descriptions.index('ra')  # in decimal degrees
        dec_index = line_descriptions.index('dec')  # in decimal degrees
        ras = np.array([float(line[ra_index]) for line in lines])
        decs = np.array([float(line[dec_index]) for line in lines])
        res_list = res_list + [ras, decs]
    if return_magnitudes:
        koi_kepmag_index = line_descriptions.index('koi_kepmag')
        kepmags = np.array([line[koi_kepmag_index] for line in lines])
        kepmags = np.array([float(km) if km != '' else np.nan for km in kepmags])
        res_list = res_list + [kepmags]
    return res_list

