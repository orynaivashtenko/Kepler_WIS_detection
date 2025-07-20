#!/usr/bin/env python3
import os
core_num = "1"
os.environ["OMP_NUM_THREADS"] = core_num  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = core_num  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = core_num  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = core_num  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = core_num  # export NUMEXPR_NUM_THREADS=6

import copy
import tce_class
from target_class import get_target_for_tce
from tce_parameters import *
Durations = tce_class.Durations
from locations import Data_location, Output_location, Target_pixels_location, Triggers_plots_location, Neighbors_folder
from auxiliary_functions import get_saving_file_path

from reportlab.lib.pagesizes import letter, landscape, portrait
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import cm
from datetime import datetime
from reportlab.platypus import Frame, BaseDocTemplate, Paragraph, Table, TableStyle, Spacer, Image, NextPageTemplate, PageBreak
from PIL import Image as PILImage

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--target_id", type=str, nargs=1, help="KIC number", default="003097222")
args = parser.parse_args()
tid = args.target_id
if isinstance(tid, list):
    tid = tid[0]


# %%

tce = tce_class.WisTCE(target_id=tid, output_location=Output_location, data_location=Data_location,
                       target_pixels_location=Target_pixels_location,
                       load_pickled_attributes=False, load_target_attributes=False, problems_folder=None)
tce.unpickle_small_attributes(load_target_attributes=True, overwrite_existing_by_target=True,
                              overwrite_location_parameters=False)

targ = get_target_for_tce(tce)
targ.unpickle_small_attributes()
tce.get_neighbors_info(neighbors_folder=Neighbors_folder)
# tce.get_target_gaia_table(file_path='home/projects/barakz/Collaboration-Kepler/orynai/run_11_results/tids_table_gaia.ecsv')

pdf_location = os.path.join(Triggers_plots_location, tce.target_id)
pdf_path = get_saving_file_path(output_location=str(pdf_location),
                                target_id=tce.target_id,
                                filename_postfix=Trigger_pdf_report_filename_postfix,
                                file_format='png')

# %% # BUILDING THE DATA TABLES

# Data for the tables

# Summary table
summary_data = [
    ['Trigger summary', '', '',],
    ['Parameter', 'Value', 'Description'],
    ['p_planet',
     '{:.3e}'.format(tce.prob_density_planet_not_noise) if tce.prob_density_planet_not_noise is not None else '--',
     'Probability to be a\nplanet and not noise'],
    ['FAR',
     '{:.3e}'.format(tce.false_alarm_prob_trigger) if tce.false_alarm_prob_trigger is not None else '--',
     'False alarm probability\non the noise background'],
    ['score',
     '{:.2f}'.format(tce.best_int_score),
     '[SNR^2], Statistical prior-informed\nscore of the pipeline'],
    ['period',
     '{:.2f}'.format(tce.best_per),
     '[days] Best period found\nby the pipeline'],
    ['duration',
     '{:.2f}'.format(tce.transit_duration_hours) if tce.transit_duration_hours is not None else '--',
     '[hours] Duration corresponding\nto the best template'],
    ['rp/re',
     '{:.2f}'.format(tce.rp_to_r_earth) + u'\u00B1' + '{:.2f}'.format(tce.rp_to_r_earth_std) if tce.rp_to_r_earth is not None else '--',
     '[earth radii] Estimated planet\nradius in units of Earth radii'],
    ['rp/rs',
     '{:.3f}'.format(tce.rp_to_rs) + u'\u00B1' + '{:.3f}'.format(tce.rp_to_rs_std) if tce.rp_to_rs is not None else '--',
     '[earth radii] Estimated planet\nradius in units of star radii'],
    ['depth',
     '{:.1f}'.format(tce.mean_transit_ppm) + u'\u00B1' + '{:.1f}'.format(
         tce.mean_transit_ppm_std) + '(' + u'\u00B1' + '{:.1f}'.format(
         tce.mean_ppm_variability) + ')' if tce.rp_to_rs is not None else '--',
     '[part per million] Estimated\ntransit depth'],
    ['t0',
     '{:.2f}'.format(tce.best_ftt),
     '[KBJD] First detected\ntransit time']]


# Star properties table
stellar_data = [
    ['Star properties', '', '',],
    ['Parameter', 'Value', 'Description'],
    ['r_star',
     '{:.2f}'.format(tce.stellar_radius) if tce.stellar_radius is not None else '--',
     '[r_sun] Stellar radius'],
    ['m_star',
     '{:.2f}'.format(tce.stellar_mass) if tce.stellar_mass is not None else '--',
     '[m_sun] Stellar mass'],
    ['mean_flux',
     '{:.2f}'.format(tce.mean_flux) if tce.mean_flux is not None else '--',
     '[e/s] Mean Kepler flux'],
    ['kepmag',
     '{:.2f}'.format(tce.kepmag) if tce.kepmag is not None else '--',
     'Kepler magnitude'],
    ['phot_g_mean_mag \ngaia_abs_gmag',
     ('{:.2f}'.format(tce.phot_g_mean_mag) if tce.phot_g_mean_mag is not None else '--')
     + ('\n' + '{:.2f}'.format(tce.gaia_abs_gmag) if tce.gaia_abs_gmag is not None else '--'),
     'g-magnitude'],
    ['gaia_b_r',
     '{:.2f}'.format(tce.gaia_b_r) if tce.gaia_b_r is not None else '--',
     'Color by Gaia'],
    ['parallax',
     '{:.2f}'.format(tce.parallax) if tce.parallax is not None else '--',
     '[mas] Parallax'],
    ['teff',
     '{:.1f}'.format(tce.teff) if tce.teff is not None else '--',
     '[K] Stellar temperature'],
    ['metallicity',
     '{:.2f}'.format(tce.metallicity) if tce.metallicity is not None else '--',
     '[Fe/H] Stellar metallicity'],
    ['logg',
     '{:.2f}'.format(tce.logg) if tce.logg is not None else '--',
     'Log surface gravity'],
    ['ra',
     '{:.2f}'.format(tce.ra) if tce.ra is not None else '--',
     '[deg] Right ascention'],
    ['dec',
     '{:.2f}'.format(tce.dec) if tce.dec is not None else '--',
     '[deg] Declination'],
    ['ruwe',
     '{:.3f}'.format(tce.ruwe) if tce.ruwe is not None else '--',
     'Gaia RUWE'],
    ['n_planets_prior',
     '{:.2e}'.format(tce.prior_planets_in_period_range_one_star) if tce.prior_planets_in_period_range_one_star is not None else '--',
     'Prior planets rate\nin period range '
     + ('{:.1e}'.format(tce.injections_period_range[0]) + '-' + '{:.1e}'.format(tce.injections_period_range[-1]) + 'days'
        if tce.injections_period_range is not None else '')],
    ['transit_prob',
     '{:.2e}'.format(tce.transit_probability) if tce.transit_probability is not None else '--',
     'Geometric transit probability'],
    ['n_planets_relevant',
     '{:.2e}'.format(tce.transiting_planets_in_relevant_range) if tce.transiting_planets_in_relevant_range is not None else '--',
     'Rate of planets in period\nrange and for relevant radii'],
    ['is_koi',
     str(tce.is_koi),
     'Whether is a known KOI'],
]


veto_flags_data = [
    ['Veto summary', '',],
    ['Test', 'Value', ],
    ['failed vetos', str(not tce.is_good_pre_filtering)],  # 'Passed all the tests and got to scrambling and injection stage'
    ['too few valid transits', str(tce.too_few_valid_transits)],
    ['bad seasons', str(tce.is_bad_seasons)],
    ['bad fold residuals', str(tce.is_bad_fold_residuals)],
    ['bad fold residuals fft', str(tce.is_bad_fold_residuals_fft)],
    ['bad fold parts', str(tce.is_bad_fold_residuals_parts)],
    ['bad fold parts time', str(tce.is_bad_fold_parts_time)],
    ['bad fold parts fft', str(tce.is_bad_fold_parts_fourier)],
    ['bad centroid', str(tce.is_centroid_bad)],
    ['bad bg pixels', str(tce.is_bad_bg_pixels)],
    ['period short', str(tce.is_period_short)],
    ['bad periodicity', str(tce.is_bad_periodicity)],
    ['bad even-odd', str(tce.is_bad_even_odd_ampl)],
    ['too short data', str(tce.is_too_short)],
    ['bad spectrum', str(tce.is_bad_fft)],
]

additional_stat_data = [
    ['Additional statistics data', '', '',],
    ['Parameter', 'Value', 'Description'],
    ['best statistic',
     '{:.2f}'.format(tce.best_stat),
     'Statistic, maximal SNR squared without phase space integration'],
    ['best gaussianized statistic',
     '{:.2f}'.format(tce.best_gauss_stat),
     'Statistic after the gaussianization at the trigger place'],
    ['best (g) gaussianized statistic',
     '{:.2f}'.format(tce.best_g_gauss_stat) if tce.best_g_gauss_stat is not None else '--',
     'Best gaussianized statistic'],
    ['best (g) score',
     '{:.2f}'.format(tce.best_g_int_score) if tce.best_g_int_score is not None else '--',
     'Best gaussianized detection score'],
    ['best (g) period',
     '{:.2f}'.format(tce.best_g_per) if tce.best_g_per is not None else '--',
     'Period for best gaussianized statistic'],
    ['best (g) first transit time',
     '{:.2f}'.format(24 * Durations[int(tce.best_g_temp)]) if (tce.best_g_temp is not None) and ~np.isnan(tce.best_g_temp) else '--',
     'Duration in hours for best gaussianized statistic'],
    ['best (g) t0',
     '{:.2f}'.format(tce.best_g_ftt) if tce.best_g_ftt is not None else '--',
     'First transit time for best gaussianized statistic'],
    ['best (g) distribution-corrected sttistic',
     '{:.2f}'.format(tce.best_g_stat) if tce.best_g_stat is not None else '--',
     'Regular (distribution-corrected) statistic for best gaussian score'],
    ['Lightcurve had frequency peaks',
     str(tce.was_notched) if tce.was_notched is not None else '--',
     'The power spectrum had significant narrow peaks, \nand notch filter was applied in the processing'],
    ['KOI masked',
     str(tce.is_koi_planet_masked) if tce.is_koi_planet_masked is not None else '--',
     'There were a significant KOI which was masked'],
    ['gaia variable star flag', str(tce.gaia_variable_star_flag) if tce.gaia_variable_star_flag is not None else '--',
     'Is marked by Gaia as variable star'],
    ['gaia non-single star flag', str(tce.gaia_non_single_star_flag) if tce.gaia_non_single_star_flag is not None else '--',
     'Is marked by Gaia as non-single star'],
    ['gaia nconfp', str(tce.gaia_nconfp) if tce.gaia_nconfp is not None else '--',
     'Number of confirmed planets from Gaia'],
    ['Is Villanova binary', str(tce.is_villanova_binary) if tce.is_villanova_binary is not None else '--',
     'Is present in Kepler Villanova eclipsing binaries catalog']
]

injections_etc_data = [
    ['Injection-recovery and scrambling parameters', ''],
    ['# injections',
     str(targ.number_of_injections)
     if targ.number_of_injections is not None and ~np.isnan(targ.number_of_injections) else '--'],
    ['# of success injections', str(targ.number_of_success_injections)
     if targ.number_of_success_injections is not None and ~np.isnan(targ.number_of_success_injections) else '--'],
    ['periods of injections',
     str(targ.injections_period_range[0]) + '-' + str(targ.injections_period_range[1])
     if targ.injections_period_range is not None and ~np.isnan(targ.injections_period_range) else '--'],
    ['# scramblings',
     str(targ.n_slides)
     if targ.n_slides is not None and ~np.isnan(targ.n_slides) else '--'],
    ['periods of scrambled search',
     str(targ.slides_period_range[0]) + '-' + str(targ.slides_period_range[1])
     if targ.slides_period_range is not None and ~np.isnan(targ.slides_period_range) else '--'],
    ['prior # planets per star in period range',
     '{:.2f}'.format(targ.prior_planets_in_period_range_one_star)
     if targ.prior_planets_in_period_range_one_star is not None and ~np.isnan(targ.prior_planets_in_period_range_one_star) else '--'],
    # ['', '{:.2f}'.format(targ.prior_radii_distribution)
    #  if targ.prior_radii_distribution is not None and ~np.isnan(targ.prior_radii_distribution) else '--'],
    ['transit probability', '{:.2e}'.format(targ.transit_probability)
     if targ.transit_probability is not None and ~np.isnan(targ.transit_probability) else '--'],
    ['# transiting planets in period range', '{:.2e}'.format(targ.transiting_planets_of_radii_in_per_range_one_star)
     if targ.transiting_planets_of_radii_in_per_range_one_star is not None and ~np.isnan(
        targ.transiting_planets_of_radii_in_per_range_one_star) else '--'],
    ['probability to transit and be in period range and relevant radii', '{:.2e}'.format(targ.transit_and_range_probability)
     if targ.transit_and_range_probability is not None and ~np.isnan(targ.transit_and_range_probability) else '--'],
    ['# transiting planets in period range and relevant radii', '{:.2e}'.format(targ.transiting_planets_in_relevant_range)
     if targ.transiting_planets_in_relevant_range is not None and ~np.isnan(targ.transiting_planets_in_relevant_range) else '--'],
    ['probability density of trigger from injection-recovery', '{:.2e}'.format(targ.prob_density_h1_trigger)
     if targ.prob_density_h1_trigger is not None and ~np.isnan(targ.prob_density_h1_trigger) else '--'],
    ['probability density of trigger from the scrambling', '{:.2e}'.format(targ.prob_density_h0_trigger)
     if targ.prob_density_h0_trigger is not None and ~np.isnan(targ.prob_density_h0_trigger) else '--'],
    ['probability density of trigger from the scrambling after trigger removal', '{:.2e}'.format(targ.prob_density_h0_clean_trigger)
     if targ.prob_density_h0_clean_trigger is not None and ~np.isnan(targ.prob_density_h0_clean_trigger) else '--'],
    ['probability density (in this period range) to be planet and not noise', '{:.2e}'.format(targ.prob_density_planet_not_noise)
     if targ.prob_density_planet_not_noise is not None and ~np.isnan(targ.prob_density_planet_not_noise) else '--'],
    ['false alarm probability', '{:.2e}'.format(targ.false_alarm_prob_trigger)
     if targ.false_alarm_prob_trigger is not None and ~np.isnan(targ.false_alarm_prob_trigger) else '--'],
    ['(M/M_sun)^{1/3}/(R/R_sun)', '{:.2f}'.format(targ.m_1_3_r_s)
     if targ.m_1_3_r_s is not None and ~np.isnan(targ.m_1_3_r_s) else '--'],
    # ['', '{:.2f}'.format() if  is not None and ~np.isnan() else '--'],
]

# Transit Info
transit_info_data = [
    ['Transit info', ''],
    ['Parameter', 'Value'],
    ['too few valid transits', str(tce.too_few_valid_transits)],
    ['# transits', str(tce.n_transits)],
    ['# valid transits', str(tce.n_valid_transits)],
    ['# same-season transits', str(tce.max_season_count)],
    ['fraction same-season transits', str(np.round(tce.fraction_same_season, 2)) + '(/' + str(Threshold_fraction_same_season) + ')'],
    ['n overlapping koi transits', str(tce.n_overlapping_koi_transits)]
]

# Transit overlaps with other KOIs
transit_overlap_data = [
    ['Transit overlaps with known transits', ''],
    ['target koi overlap mask', str(tce.koi_overlap_mask.astype(int))],
    ['target kepler tce overlap mask', str(tce.kepler_tce_overlap_mask.astype(int))],
    ['other overlapping kois', str(tce.overlapping_kois)],
    ['other overlapping villanovas', str(tce.overlapping_villanovas)],
]

# Info of this target's KOIs
if len(tce.koi_dict) > 0:
    kepler_koi_data = [
        ['kepoi_name'] + [n for n in tce.koi_dict['kepoi_name']],
        ['koi_disposition'] + [n for n in tce.koi_dict['koi_disposition']],
        ['koi_period'] + [np.round(float(n), 3) if n != '' else '--' for n in tce.koi_dict['koi_period']],
        ['koi_time0bk'] + [np.round(float(n), 3) if n != '' else '--' for n in tce.koi_dict['koi_time0bk']],
        ['koi_duration'] + [np.round(float(n), 3) if n != '' else '--' for n in tce.koi_dict['koi_duration']],
        ['koi_model_snr'] + [np.round(float(n), 3) if n != '' else '--' for n in tce.koi_dict['koi_model_snr']],
        ['koi_model_snr'] + [np.round(float(n), 3) if n != '' else '--' for n in tce.koi_dict['koi_model_snr']],
        ['koi_max_mult_ev'] + [np.round(float(n), 3) if n != '' else '--' for n in tce.koi_dict['koi_max_mult_ev']],
        ['koi_max_sngle_ev'] + [np.round(float(n), 3) if n != '' else '--' for n in tce.koi_dict['koi_max_mult_ev']]
    ]
    kepler_koi_data = [['KOI info'] + ['' for _ in range(len(kepler_koi_data[0]) - 1)]] + kepler_koi_data
else:
    kepler_koi_data = [['KOI info', ''], ['Target is not KOI', '']]

# Info of this target's Kepler TCEs
if len(tce.tce_dict) > 0:
    kepler_tce_data = [
        ['tce_period'] + [str(np.round(float(n), 3)) if n != '' else '--' for n in tce.tce_dict['tce_period']],
        ['tce_time0bk'] + [str(np.round(float(n), 2)) if n != '' else '--' for n in tce.tce_dict['tce_time0bk']],
        ['tce_duration'] + [str(np.round(float(n), 2)) if n != '' else '--' for n in tce.tce_dict['tce_duration']],
        ['tce_model_snr'] + [str(np.round(float(n), 3)) if n != '' else '--' for n in tce.tce_dict['tce_model_snr']],
        ['tce_max_mult_ev'] + [str(np.round(float(n), 3)) if n != '' else '--' for n in tce.tce_dict['tce_max_mult_ev']],
    ]
    kepler_tce_data = [['Kepler TCE info'] + ['' for _ in range(len(kepler_tce_data[0]) - 1)]] + kepler_tce_data
else:
    kepler_tce_data = [['Kepler TCE info', ''], ['Target is not Kepler TCE', '']]

# Veto fold info
resid_veto_data = [
    ['Residuals of the fold veto results', ''],
    ['bad results', str(tce.is_bad_fold_residuals)],
    ['p-value MF fold residuals', ('{:.3f}'.format(tce.p_value_mf_fold_residuals) if tce.p_value_mf_fold_residuals > 1e-3
                                   else '{:.2e}'.format(
        tce.p_value_mf_fold_residuals)) if tce.p_value_mf_fold_residuals is not None else '--'],
    ['p-value fold residuals fft', ('{:.3f}'.format(tce.p_value_fold_residuals_fft) if tce.p_value_fold_residuals_fft > 1e-3
                                    else '{:.2e}'.format(
        tce.p_value_fold_residuals_fft)) if tce.p_value_fold_residuals_fft is not None else '--'],
    ['p-value fold residuals', ('{:.3f}'.format(tce.p_value_fold_residuals_time) if tce.p_value_fold_residuals_time > 1e-3
                                else '{:.2e}'.format(
        tce.p_value_fold_residuals_time)) if tce.p_value_fold_residuals_time is not None else '--'],
    ['Duration of template (hours) and score of fold contaminant',
     str(np.round(Durations[int(tce.template_and_stat_of_fold_contaminant[0])], 2)) + ', '
     + str(np.round(tce.template_and_stat_of_fold_contaminant[1], 2)) if tce.template_and_stat_of_fold_contaminant is not None else '--'],
    ['Duration (hours) of best folded template', str(np.round(tce.duration_folded_hours, 2)
                                                     ) + ' h' + ' (search ' + '{:.2f}'.format(
        tce.transit_duration_hours) if tce.transit_duration_hours is not None else '--' + ')']
]  # self.fn_fold_veto = fn_fold_veto

# Veto parts in Fourier domain info
veto_parts_data_fourier_domain = [
    ['Partial amplitudes veto of the fold in Fourier domain', ''],
    ['bad results in Fourier domain', str(tce.is_bad_fold_parts_fourier)],
    ['# parts', str(int(tce.n_parts_veto_fold_parts_fourier))],
    ['p-value parts Fourier domain',
     ((('{:.3f}'.format(tce.p_values_veto_fold_parts_fourier[1]) if tce.p_values_veto_fold_parts_fourier[1] > 1e-3
        else '{:.2e}'.format(tce.p_values_veto_fold_parts_fourier[1])) if not np.isnan(tce.p_values_veto_fold_parts_fourier[1]) else 'nan')
      if tce.p_values_veto_fold_parts_fourier is not None else '--')],
    ['p-value parts Fourier domain no renorm',
     ((('{:.3f}'.format(tce.p_values_veto_fold_parts_fourier[0]) if tce.p_values_veto_fold_parts_fourier[0] > 1e-3
        else '{:.2e}'.format(tce.p_values_veto_fold_parts_fourier[0])) if not np.isnan(tce.p_values_veto_fold_parts_fourier[0]) else 'nan')
      if tce.p_values_veto_fold_parts_fourier is not None else '--')],
]

# Veto parts in time domain info
veto_parts_data_real_domain = [
    ['Partial amplitudes veto of the fold in time domain', ''],
    ['bad results in time domain', str(tce.is_bad_fold_residuals_parts)],
    ['p-value parts time domain', '{:.3f}'.format(tce.p_value_veto_fold_parts) if tce.p_value_veto_fold_parts > 1e-3
     else '{:.2e}'.format(tce.p_value_veto_fold_parts)],
]

# Veto parts in time domain, details
veto_parts_details_real_domain = [
    ['Partial amplitudes of the fold in time domain', ''],
    ['# parts', 'p-value total', 'p-value worst'],
]

for i in range(len(tce.veto_fold_parts_results)):
    veto_parts_details_real_domain.append(
        [str(int(tce.veto_fold_parts_results[i][0])),
         '{:.3f}'.format(tce.veto_fold_parts_results[i][1]) if tce.veto_fold_parts_results[i][1] > 1e-3
         else '{:.2e}'.format(tce.veto_fold_parts_results[i][1]),
         '{:.3f}'.format(tce.veto_fold_parts_results[i][2]) if tce.veto_fold_parts_results[i][2] > 1e-3
         else '{:.2e}'.format(tce.veto_fold_parts_results[i][2])]
    )

# Veto centroid info
veto_centroid_data = [
    ['Centroid veto', ''],
    ['SNR squared of centroid deviation', '{:.2f}'.format(tce.centroid_snr_2) if tce.centroid_snr_2 is not None else '--'],
    ['SNR of ra, dec deviations', (('{:.2f}'.format(tce.centroid_snrs_ra_dec[0]) if tce.centroid_snrs_ra_dec[0] is not None else '--')
                                   + ', ' + (
                                       '{:.2f}'.format(tce.centroid_snrs_ra_dec[1]) if tce.centroid_snrs_ra_dec[1] is not None else '--'))],
    ['Durations of ra, dec deviations (hours)', str(np.round(24 * Durations[int(tce.centroid_templates_ra_dec[0])], 2)
                                                    ) + ', ' + str(np.round(24 * Durations[int(tce.centroid_templates_ra_dec[1])], 2)
                                                                   ) if tce.centroid_templates_ra_dec is not None else '--, --']]

# Veto star-bg pixels info
veto_star_bg_pixels_data = [
    ['Star vs background transit depth veto', ''],
    ['bad result', str(tce.is_bad_bg_pixels)],
    ['p-value that the depth from the background pixels is too large',
     (('{:.3f}'.format(tce.p_value_mean_bg_pixels) if tce.p_value_mean_bg_pixels > 1e-3
       else '{:.2e}'.format(tce.p_value_mean_bg_pixels)) if not np.isnan(
         tce.p_value_mean_bg_pixels) else 'nan') if tce.p_value_mean_bg_pixels is not None else '--'],
    ['p-value that the deviation is too large for the worst transit',
     (('{:.3f}'.format(tce.p_value_worst_bg_pixels) if tce.p_value_worst_bg_pixels > 1e-3
       else '{:.2e}'.format(tce.p_value_worst_bg_pixels)) if not np.isnan(
         tce.p_value_worst_bg_pixels) else 'nan') if tce.p_value_worst_bg_pixels is not None else '--'],
    ['valid transit mask for star', str(list(tce.star_valid_transit_mask.astype(int)))],
    ['valid transit mask for background', str(list(tce.bg_valid_transit_mask.astype(int)))],
]

# Even-odd veto info
veto_even_odd_data = [
    ['Even-odd veto', ''],
    ['bad result', str(tce.is_bad_even_odd_ampl)],
    ['p-value that even and odd are different', ('{:.3f}'.format(tce.p_value_even_odd_ampl) if tce.p_value_even_odd_ampl > 1e-3
                                                 else '{:.2e}'.format(
        tce.p_value_even_odd_ampl)) if tce.p_value_even_odd_ampl is not None else '--'],
    ['Peak scores even/odd', (('{:.2f}'.format(tce.scores_ampls_even_odd[0][0]) if tce.scores_ampls_even_odd[0][0] is not None else '--')
                              + ', ' + ('{:.2f}'.format(tce.scores_ampls_even_odd[1][0]) if tce.scores_ampls_even_odd[1][
                                                                                                0] is not None else '--'))],
    ['Difference in positions, bins', str(np.round(tce.scores_ampls_even_odd[0][-1] - tce.scores_ampls_even_odd[1][-1],
                                                   1)) if tce.scores_ampls_even_odd is not None else '--']]

# %% # DOCUMENT DEFINITIONS, FUNCTIONS, STYLES
tid_plots_location = os.path.join(Triggers_plots_location, tce.target_id)
# ------------------------------------------------------------------------------------------------
# Defining Pages layouts
pagesize = letter
page_width_portrait, page_height_portrait = portrait(pagesize)
page_width_landscape, page_height_landscape = landscape(pagesize)
margin_cm = 0.3

padding = dict(
  leftPadding=cm * margin_cm,
  rightPadding=cm * margin_cm,
  topPadding=cm * margin_cm,
  bottomPadding=cm * margin_cm)

portrait_frame = Frame(0, 0, *pagesize, **padding)
landscape_frame = Frame(0, 0, *landscape(pagesize), **padding)


# noinspection PyUnusedLocal, PyShadowingNames
def on_page(canvas, doc, pagesize=pagesize):
    page_num = canvas.getPageNumber()
    pass


# noinspection PyUnusedLocal, PyShadowingNames
def on_page_landscape(canvas, doc):
    return on_page(canvas, doc, pagesize=landscape(pagesize))


from reportlab.platypus import PageTemplate

portrait_template = PageTemplate(
  id='portrait',
  frames=portrait_frame,
  onPage=on_page,
  pagesize=pagesize)

landscape_template = PageTemplate(
  id='landscape',
  frames=landscape_frame,
  onPage=on_page_landscape,
  pagesize=landscape(pagesize))

# ------------------------------------------------------------------------------------------------
# Defining Styles

styles = getSampleStyleSheet()
fontsize_text = 8
font_family_text = 'Helvetica'
font_family_bold = 'Helvetica-Bold'

styles['Normal'].fontSize = fontsize_text
styles['Normal'].fontName = font_family_text
title_style = ParagraphStyle(name='TitleStyle', fontName=font_family_text, fontSize=15, alignment=TA_CENTER, leading=24)
subheader_style = ParagraphStyle(name='SubheaderStyle', fontName=font_family_text, fontSize=11, alignment=TA_CENTER, leading=20)
table_header_color = colors.HexColor('#eeeeee')
table_pad = 0.5

centered_paragraph = ParagraphStyle(name='Centered', alignment=TA_CENTER, fontSize=fontsize_text, fontName=font_family_text)

table_style_one_header = TableStyle([
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
    ('BACKGROUND', (0, 0), (-1, 0), table_header_color),  # Title background
    ('BACKGROUND', (0, 1), (0, -1), table_header_color),  # Darker for leftmost column
    ('GRID', (0, 0), (-1, -1), 0.1, colors.black),  # Apply grid lines
    ('SPAN', (0, 0), (-1, 0)),  # Span the title across all columns
    ('FONTNAME', (0, 0), (-1, -1), font_family_text),
    ('FONTNAME', (0, 0), (-1, 0), font_family_bold),  # Make title bold
    ('FONTSIZE', (0, 0), (-1, -1), fontsize_text),
    ('TOPPADDING', (0, 0), (-1, -1), table_pad),  # Set top padding
    ('BOTTOMPADDING', (0, 0), (-1, -1), table_pad),  # Set bottom padding
])
table_style = copy.deepcopy(table_style_one_header)
table_style.add('BACKGROUND', (0, 1), (-1, 1), table_header_color)  # Header background


# ------IMAGES-----------------------------------------------------------------------------------
# Define target widths for images in centimeters
image_width_cm_sq = 8.5
image_width_full_landscape_cm = (page_width_landscape / cm - 2 * margin_cm) * 0.99
image_width_full_portrait_cm = (page_width_portrait / cm - 2 * margin_cm) * 0.99


def calculate_aspect_ratio(img_path, image_width_cm):
    # Preserved aspect ratio
    target_width = image_width_cm * 28.35  # Convert cm to points (1 cm ≈ 28.35 points)
    # points_to_cm = 0.0352778
    with PILImage.open(img_path) as img:
        original_width, original_height = img.size
        aspect_ratio = original_height / original_width
        target_height = target_width * aspect_ratio
    return target_width, target_height


def get_img(fig_filename_postfix, image_width_cm=image_width_full_landscape_cm, plots_location=tid_plots_location):
    if not (isinstance(fig_filename_postfix, list) or isinstance(fig_filename_postfix, tuple
                                                                 ) or isinstance(fig_filename_postfix, np.ndarray)):
        was_list = False
        fig_filename_postfix = [fig_filename_postfix]
    else:
        was_list = True
    if not hasattr(image_width_cm, '__len__'):
        image_width_cm = [image_width_cm for _ in range(len(fig_filename_postfix))]
    img_list = []
    for ii in range(len(fig_filename_postfix)):
        fig1_file = os.path.join(str(plots_location), str(tce.target_id + '_' + fig_filename_postfix[ii]))
        if os.path.exists(fig1_file):
            # Calculate dimensions for images
            img1_width, img1_height = calculate_aspect_ratio(fig1_file, image_width_cm[ii])
            img1 = Image(fig1_file, width=img1_width, height=img1_height)
            img1.allowDuplicate = True
            img_list.append(img1)
        else:
            img_list.append(Paragraph('This figure was not produced', centered_paragraph))
    if was_list:
        img_table = Table([img_list])
        img_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
        ]))
        element = img_table
    else:
        element = img_list[0]
    return element


# %% # MAKING THE DOCUMENT
elements = []
page_templates = []
# ------------------------------------------------------------------------------------------------
# Title
# Header
pre_header = Paragraph('WIS-Kepler Pipeline | ' +
                       'Document written on ' + datetime.today().strftime('%d %b %Y') + ' | ' +
                       "Author Oryna Ivashtenko",
                       centered_paragraph)
header = Paragraph("Trigger Report for KIC" + str(tce.target_id), title_style)
subheader = Paragraph("Best trigger results of search, vetting, and significance estimation", subheader_style)

elements.append(pre_header)
elements.append(header)
elements.append(subheader)
elements.append(Spacer(1, cm * 0.2))

# ------------------------------------------------------------------------------------------------
# Page 1. Summary

elements.append(get_img(Background_injection_histograms_plot_filename_postfix, image_width_cm=image_width_full_landscape_cm))
elements.append(Spacer(1, cm * 0.2))  # Spacer of 0.5 cm

summary_table = Table(summary_data)
summary_table.setStyle(table_style)
# ------------
stellar_table = Table(stellar_data)
stellar_table.setStyle(table_style)
# ------------
veto_flags_table = Table(veto_flags_data)
veto_table_style = copy.deepcopy(table_style)

for i in range(1, len(veto_flags_data)):  # Start from 1 to skip header
    if veto_flags_data[i][1] == 'True':  # Check the status
        veto_table_style.add('BACKGROUND', (1, i), (1, i), colors.HexColor('#ffcccc'))  # Light red background
    elif veto_flags_data[i][1] == 'None':
        veto_table_style.add('BACKGROUND', (1, i), (1, i), colors.HexColor('#d3d3d3'))  # Light gray background

veto_flags_table.setStyle(veto_table_style)
# ------------
spacer_size_cm = 0.2
combined_data = [[summary_table, Spacer(spacer_size_cm * cm, 0), stellar_table, Spacer(spacer_size_cm * cm, 0
                                                                                       ), veto_flags_table]]  # Spacer to create a gap
combined_table = Table(combined_data)
combined_table_style = TableStyle([('VALIGN', (0, 0), (-1, -1), 'TOP')])  # , ('ALIGN', (0, 0), (-1, -1), 'CENTER')])
combined_table.setStyle(combined_table_style)
# Adjust the width of the combined table to fit content and allow space on the right
combined_table_widths = [summary_table.wrap(0, 0)[0], spacer_size_cm*cm, stellar_table.wrap(
    0, 0)[0], spacer_size_cm*cm, veto_flags_table.wrap(0, 0)[0]]
combined_table._argW = list(np.array(combined_table_widths) / np.sum(combined_table_widths) * (page_width_landscape - 2 * margin_cm * cm))

elements.append(combined_table)

elements.append(NextPageTemplate('landscape'))
elements.append(PageBreak())
page_templates.append(landscape_template)

# ------------------------------------------------------------------------------------------------
# Page 2. Fluxes

elements.append(get_img(fig_filename_postfix=All_series_plot_filename_postfix, image_width_cm=image_width_full_landscape_cm))

elements.append(NextPageTemplate('landscape'))
elements.append(PageBreak())
page_templates.append(landscape_template)

# ------------------------------------------------------------------------------------------------
# Page 3. Folds

for filename_postfix_plot in [All_folds_plot_filename_postfix, All_folds_zoomed_filename_postfix]:
    elements.append(get_img(filename_postfix_plot, image_width_cm=image_width_full_landscape_cm))

elements.append(NextPageTemplate('landscape'))
elements.append(PageBreak())
page_templates.append(landscape_template)

# ------------------------------------------------------------------------------------------------
# Page 4. Transit info
table_style_transit_info = copy.deepcopy(table_style_one_header)
table_style_transit_info.add('LEFTPADDING', (0, 0), (-1, -1), 2)  # Reduce left padding
table_style_transit_info.add('RIGHTPADDING', (0, 0), (-1, -1), 2)  # Reduce right padding

elements.append(Paragraph("Transit info", subheader_style))
elements.append(Spacer(0, cm * 0.5))

transit_info_table = Table(transit_info_data)  # , colWidths=left_table_widths)
transit_info_table.setStyle(table_style_transit_info)
kepler_koi_table = Table(kepler_koi_data)
kepler_koi_table.setStyle(table_style_transit_info)
kepler_koi_table.hAlign = 'RIGHT'
kepler_tce_table = Table(kepler_tce_data)
kepler_tce_table.setStyle(table_style_transit_info)
kepler_tce_table.hAlign = 'RIGHT'

combined_data = [[transit_info_table, Spacer(0.1 * cm, 0), kepler_koi_table, Spacer(0.1 * cm, 0), kepler_tce_table]]
# -> Spacer to create a gap
combined_table = Table(combined_data)
combined_table_style = TableStyle([('VALIGN', (0, 0), (-1, -1), 'TOP')])  # , ('ALIGN', (0, 0), (-1, -1), 'CENTER')])
combined_table.setStyle(combined_table_style)
elements.append(combined_table)

transit_overlap_table = Table(transit_overlap_data)
transit_overlap_table.setStyle(table_style_one_header)
transit_overlap_table.hAlign = 'LEFT'
elements.append(transit_overlap_table)
elements.append(Spacer(0, cm * 0.5))

elements.append(get_img(Transit_info_plot_filename_postfix, image_width_cm=image_width_full_landscape_cm))

elements.append(NextPageTemplate('landscape'))
elements.append(PageBreak())
page_templates.append(landscape_template)

# ------------------------------------------------------------------------------------------------
# Page 5. Chunk periodogram

elements.append(Paragraph("Maximized periodogram", subheader_style))
elements.append(Spacer(0, cm * 0.5))
if tce.is_bad_periodicity:
    elements.append(Paragraph('Bad periodicity:' + tce.bad_periodicity_reasons, styles['Normal']))
elements.append(Spacer(0, cm * 0.5))
elements.append(get_img(Periodogram_chunks_plot_filename_postfix, image_width_cm=image_width_full_landscape_cm))

elements.append(NextPageTemplate('landscape'))
elements.append(PageBreak())
page_templates.append(landscape_template)

# ------------------------------------------------------------------------------------------------
# Page 6. Residuals fold

elements.append(Paragraph("Folded lightcurve residuals veto", subheader_style))
elements.append(Spacer(0, cm * 0.5))

resid_veto_table = Table(resid_veto_data)
resid_veto_table.setStyle(table_style_one_header)
elements.append(resid_veto_table)
elements.append(Spacer(0, cm * 0.5))
elements.append(get_img(Residuals_fold_veto_histogram_plot_filename_postfix, image_width_cm=image_width_full_landscape_cm/18*13))
elements.append(Spacer(0, cm * 2))

# -------------------------
# Veto parts fold (same page)

elements.append(Paragraph("Veto partial amplitudes of fold", subheader_style))

veto_parts_table_fourier_domain = Table(veto_parts_data_fourier_domain)
veto_parts_table_fourier_domain.setStyle(table_style_one_header)

veto_parts_table_real_domain = Table(veto_parts_data_real_domain)
veto_parts_table_real_domain.setStyle(table_style_one_header)

veto_parts_details_table_real_domain = Table(veto_parts_details_real_domain)
veto_parts_details_table_real_domain.setStyle(table_style)

combined_data = [[veto_parts_table_fourier_domain, Spacer(spacer_size_cm * cm, 0),
                  veto_parts_table_real_domain, Spacer(spacer_size_cm * cm, 0),
                  veto_parts_details_table_real_domain]]  # Spacer to create a gap
combined_table = Table(combined_data)
combined_table_style = TableStyle([('VALIGN', (0, 0), (-1, -1), 'TOP')])  # , ('ALIGN', (0, 0), (-1, -1), 'CENTER')])
combined_table.setStyle(combined_table_style)
elements.append(combined_table)

elements.append(NextPageTemplate('landscape'))
elements.append(PageBreak())
page_templates.append(landscape_template)

# ------------------------------------------------------------------------------------------------
# Page 7. Veto parts fold (continuation)

elements.append(get_img(Veto_parts_fold_plot_filename_postfix, image_width_cm=image_width_full_landscape_cm))

elements.append(NextPageTemplate('landscape'))
elements.append(PageBreak())
page_templates.append(landscape_template)

# ------------------------------------------------------------------------------------------------
# Page 8. Veto star-bg pixels

elements.append(Paragraph("Veto depth of star and background pixels", subheader_style))
elements.append(Spacer(0, cm * 0.3))

veto_star_bg_pixels_table = Table(veto_star_bg_pixels_data)
veto_star_bg_pixels_table.setStyle(table_style_one_header)
elements.append(veto_star_bg_pixels_table)
elements.append(Spacer(0, cm * 0.5))

elements.append(get_img(fig_filename_postfix=Star_bg_pixels_ppm_plot_filename_postfix, image_width_cm=image_width_full_landscape_cm))
elements.append(Spacer(0, cm * 0.3))
elements.append(get_img(fig_filename_postfix=Star_bg_full_stats_plot_filename_postfix, image_width_cm=image_width_full_landscape_cm))

elements.append(NextPageTemplate('landscape'))
elements.append(PageBreak())
page_templates.append(landscape_template)

# ------------------------------------------------------------------------------------------------
# Page 9. Veto centroid

elements.append(Paragraph("Veto centroid", subheader_style))
elements.append(Spacer(0, cm * 0.1))

veto_centroid_table = Table(veto_centroid_data)
veto_centroid_table.setStyle(table_style_one_header)
elements.append(veto_centroid_table)

elements.append(Spacer(0, cm * 0.5))

elements.append(get_img(fig_filename_postfix=[Centroid_cloud_plot_filename_postfix, Centroid_snr2_seasons_plot_filename_postfix],
                        image_width_cm=[image_width_cm_sq, image_width_full_landscape_cm * 0.9 - image_width_cm_sq]))

elements.append(get_img(fig_filename_postfix=Centroid_periodic_plot_filename_postfix, image_width_cm=image_width_full_landscape_cm*0.7))

elements.append(NextPageTemplate('landscape'))
elements.append(PageBreak())
page_templates.append(landscape_template)

# ------------------------------------------------------------------------------------------------
# Page 10. Veto centroid 2

elements.append(Paragraph("Veto centroid", subheader_style))
elements.append(Spacer(0, cm * 0.5))
elements.append(get_img(fig_filename_postfix=Centroid_coordinate_pieces_plot_filename_postfix,
                        image_width_cm=image_width_full_landscape_cm))
elements.append(Spacer(0, cm * 0.3))
elements.append(get_img(fig_filename_postfix=Centroid_full_stats_plot_filename_postfix, image_width_cm=image_width_full_landscape_cm))

elements.append(NextPageTemplate('landscape'))
elements.append(PageBreak())
page_templates.append(landscape_template)

# ------------------------------------------------------------------------------------------------------------------------
# Page 11. Even-odd
elements.append(Paragraph("Veto even-odd", subheader_style))
elements.append(Spacer(0, cm * 0.5))
elements.append(get_img(fig_filename_postfix=Even_odd_wh_flux_and_snr_plot_filename_postfix,
                        image_width_cm=image_width_full_landscape_cm))
elements.append(Spacer(0, cm * 1.))
elements.append(get_img(fig_filename_postfix=Even_odd_wh_flux_from_veto_plot_filename_postfix,
                        image_width_cm=image_width_full_landscape_cm))

elements.append(NextPageTemplate('landscape'))
elements.append(PageBreak())
page_templates.append(landscape_template)

# ------------------------------------------------------------------------------------------------------------------------
# Page 12. Seasonal

elements.append(Paragraph("Seasonal folds", subheader_style))
elements.append(Spacer(0, cm * 0.5))
elements.append(get_img(fig_filename_postfix=Season_pieces_wh_flux_plot_filename_postfix,
                        image_width_cm=image_width_full_landscape_cm))
elements.append(Spacer(0, cm * 0.5))
elements.append(get_img(fig_filename_postfix=Season_pieces_snr_plot_filename_postfix,
                        image_width_cm=image_width_full_landscape_cm))

elements.append(NextPageTemplate('landscape'))
elements.append(PageBreak())
page_templates.append(landscape_template)


# ------------------------------------------------------------------------------------------------------------------------
# Page 13. Injection-recovery
elements.append(Paragraph("Injection-recovery parameters and efficiency", subheader_style))
injections_etc_table = Table(injections_etc_data)
injections_etc_table.setStyle(table_style_one_header)
elements.append(injections_etc_table)
elements.append(Spacer(0, cm * 0.5))
elements.append(get_img(fig_filename_postfix=Injections_radii_durations_score_plot_filename_postfix,
                        image_width_cm=image_width_full_landscape_cm))

elements.append(NextPageTemplate('portrait'))
elements.append(PageBreak())
page_templates.append(portrait_template)

# ------------------------------------------------------------------------------------------------------------------------
# Page 14. Injection-recovery

elements.append(get_img(fig_filename_postfix=Corner_injections_orbital_plot_filename_postfix,
                        image_width_cm=image_width_full_portrait_cm))

elements.append(NextPageTemplate('portrait'))
elements.append(PageBreak())
page_templates.append(portrait_template)

# ------------------------------------------------------------------------------------------------------------------------
# Page 15. Injection-recovery
elements.append(get_img(fig_filename_postfix=Corner_injections_times_plot_filename_postfix,
                        image_width_cm=image_width_full_portrait_cm))

elements.append(NextPageTemplate('portrait'))
elements.append(PageBreak())
page_templates.append(portrait_template)

# ------------------------------------------------------------------------------------------------------------------------
# Page 16. Injection-recovery
elements.append(get_img(fig_filename_postfix=Corner_injections_scores_plot_filename_postfix, image_width_cm=image_width_full_portrait_cm))

elements.append(NextPageTemplate('portrait'))
elements.append(PageBreak())
page_templates.append(portrait_template)

# ------------------------------------------------------------------------------------------------------------------------
# Page 17. Neighbors
# elements.append(Paragraph("Pixel map and neighboring stars", subheader_style))
elements.append(get_img(fig_filename_postfix=Neighbors_maps_plot_filename_postfix, image_width_cm=image_width_full_portrait_cm))

elements.append(NextPageTemplate('portrait'))
elements.append(PageBreak())
page_templates.append(portrait_template)

# ------------------------------------------------------------------------------------------------------------------------
# Page 18. Additional statistical info

elements.append(Paragraph("Additional parameters of data processing", subheader_style))
additional_stat_table = Table(additional_stat_data)
additional_stat_table.setStyle(table_style)
elements.append(additional_stat_table)
elements.append(Spacer(0, cm * 0.5))

elements.append(PageBreak())
page_templates.append(portrait_template)


# %%

# ------------------------------------------------------------------------------------------------

doc = BaseDocTemplate(
  pdf_path,
  pageTemplates=page_templates)

doc.build(elements)
