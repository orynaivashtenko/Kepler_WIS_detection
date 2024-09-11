#!/usr/bin/env python3

import os

os.environ["OMP_NUM_THREADS"] = "1"  #
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"  #
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
from locations import *


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
      'Number of transits overlapping with the known Kepler objects of interest']],
    ['centroid_snr_2',
     [float, float, 'SNR^2', np.nan,
      'Statistical score of the centroid deviation during the transits']],
    ['p_value_mean_bg_pixels',
     [float, float, '', np.nan,
      'P-value that the background pixel signal is stronger than the target pixel signal']],
    ['p_value_worst_bg_pixels',
     [float, float, '', np.nan,
      'P-value that the background pixel signal is stronger than the target pixel signal for the worst transit']],
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
    ['failures',
     [str, '<U200', '', '',
      'Some failures that happened in the pipeline run']],
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
    ])

Pre_filtered_properties = dict([  # name, type, value
    ['too_few_valid_transits', [bool, False]],
    ['is_bad_seasons', [bool, False]],
    ['is_bad_fold_residuals', [bool, False]],
    ['is_centroid_bad', [bool, False]],
    ['is_bad_bg_pixels', [bool, False]],
    ['is_bad_periodicity', [bool, False]],
    ['bad_periodicity_reasons', [str, '']],
    ['is_good_pre_filtering', [bool, True]],
    ['is_too_short', [bool, False, '', 'Whether was marked as too short for the search']],
    ['is_bad_fft', [bool, False, '', 'Whether was marked by the pipeline as not quiet enough spectrum']],
    ])

# Others:
# ['is_koi', [bool, bool, '', 'Whether is Kepler object of interest']],
# ['n_slides', [int, int, '', 'Number of timeslides performed on the target']],
# ['mean_transit_amplitude',
#      [float, float, '', np.nan,
#       'Mean transit amplitude corresponding to the normalized template']],

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
