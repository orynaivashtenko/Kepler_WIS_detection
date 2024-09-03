import numpy as np
import os
# from scipy.interpolate import RegularGridInterpolator

######################################################################################################################################
# PHYSICAL CONSTANTS AND RATES
# ------------------------------------------------------------------------------------------------------------------------------------

# Physical constants is CGS
R_sun = 6.9634e10  # cm #solar radius
au = 1.495978707e13  # cm #astronomical unit
M_sun = 1.989e33  # g #solar mass
G = 6.67430e-8  # dyn⋅cm2/g2 #gravitational constant
R_sun_R_e = 109.076  # sun radii to Earth radii
Default_limb_darkening_coefs = (0.47, 0.23)

Pipeline_folder = os.path.dirname(__file__)
force_data_output_locations_creation = False
Working_place = None  # On which machine it is running: my office Mac ('Home'), Marvin or WEXAC
# Working_place = 'Home' # 'Marvin'  # 'WEXAC'
if Working_place is None:
    if '/Users/orynai/Work/Kepler_Project' in Pipeline_folder:
        Working_place = 'Home'
    elif '/SMALLDATA' in Pipeline_folder or '/home/orynai/' in Pipeline_folder:
        Working_place = 'Marvin'
    elif '/home/labs/barakz' in Pipeline_folder:
        Working_place = 'WEXAC'


# Way to use the occurrence table:
def get_occurrence_table_old(stellar_radius):
    """
    Per one star
    :param stellar_radius:
    :type stellar_radius:
    :return:
    :rtype:
    """
    # We are not putting the following occurrence rate info to a separate file because it is easier like this.
    # Generating rp according to the prior
    log_rp_edges = np.log(np.array([0.5, 1, 2, 4, 8, 20]) / stellar_radius / R_sun_R_e)
    # -> This was measured in Earth radii, we convert to stellar radii
    period_edges = 10 ** (np.linspace(1, 3, 11))[2:9]  # An exact empirical formula for the values in the table.
    occurrences_table = np.array([
        # zhu and dong, planets per 100 stars (WARNING: WE ONLY DETECT ONE PER STAR, DISTRIBUTION CAN BE DIFFERENT!)
        [4.7, 4.1, 7.2, 0.66, 0.23],  # P = 25.1  - 39.8  days
        [4.2, 4.2, 8.1, 0.58, 0.52],  # P = 39.8  - 63.1  days
        [3.1, 3.1, 6.9, 1.50, 0.44],  # P = 63.0  - 100.0 days
        [4.2, 4.2, 6.2, 1.30, 1.30],  # P = 100.0 - 158.5 days
        [4.9, 4.9, 7.5, 2.40, 1.20],  # P = 158.5 - 251.2 days
        [10.0, 10.0, 6.9, 1.70, 1.70]]  # P = 251.2 - 398.1 days
    ) / 100
    return log_rp_edges, period_edges, occurrences_table


# Way to use the occurrence table:
def get_occurrence_table():
    """
    Per one star
    :return:
    :rtype:
    """
    # We are not putting the following occurrence rate info to a separate file because it is easier like this.
    # Generating rp according to the prior
    log2_rp_re_edges = np.array([-1, 0, 1, 2, 3, np.log2(20)])  # planetary radius in Earth radii
    log10_per_edges = np.linspace(1, 3, 11)[2:9]  # days # An exact empirical formula for the values in the table.
    occurrences_table = np.array([
        # zhu and dong, planets per 100 stars (WARNING: WE ONLY DETECT ONE PER STAR, DISTRIBUTION CAN BE DIFFERENT!)
        [4.7, 4.1, 7.2, 0.66, 0.23],  # P = 25.1  - 39.8  days
        [4.2, 4.2, 8.1, 0.58, 0.52],  # P = 39.8  - 63.1  days
        [3.1, 3.1, 6.9, 1.50, 0.44],  # P = 63.0  - 100.0 days
        [4.2, 4.2, 6.2, 1.30, 1.30],  # P = 100.0 - 158.5 days
        [4.9, 4.9, 7.5, 2.40, 1.20],  # P = 158.5 - 251.2 days
        [10.0, 10.0, 6.9, 1.70, 1.70]]  # P = 251.2 - 398.1 days
    ) / 100  # -> we report per one star
    return log2_rp_re_edges, log10_per_edges, occurrences_table


def get_occurrence_table_enlarged():
    """
    """
    # We are not putting the following occurrence rate info to a separate file because it is easier like this.
    # zhu and dong, planets per 100 stars (WARNING: WE ONLY DETECT ONE PER STAR, DISTRIBUTION CAN BE DIFFERENT!)
    # I repeated some rows to extrapolate in order to interpolate
    # log2_rp_re_edges = np.array([-1, 0, 1, 2, 3, np.log2(20)])  # planetary radius in Earth radii
    log2_rp_re_edges_enlarged = np.array([-1, 0., 1, 2, 3, np.log2(20)])
    log10_per_edges_enlarged = np.array([1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6, 2.8])
    occurrences_table_enlarged = np.array([
        [0.052, 0.046, 0.064, 0.0045, 0.0017],
        [0.047, 0.041, 0.072, 0.0066, 0.0023],
        [0.042, 0.042, 0.081, 0.0058, 0.0052],
        [0.031, 0.031, 0.069, 0.015,  0.0044],
        [0.042, 0.042, 0.062, 0.013,  0.013],
        [0.049, 0.049, 0.075, 0.024,  0.012],
        [0.1,   0.1,   0.069, 0.017,  0.017],
        [0.1,   0.1,   0.069, 0.017,  0.017]])
    return log2_rp_re_edges_enlarged, log10_per_edges_enlarged, occurrences_table_enlarged

# def get_occurrence_rate(stellar_radius, per_days, rp_rearth, rp_rstar=None):
#     """
#     Per one star
#     :param stellar_radius:
#     :type stellar_radius: float, in solar radii
#     :param per_days: desired period or periods in days
#     :type per_days: ndarray(float) or float
#     :param rp_rearth: desired planetary radius in Earth radii. if None, `rp_rstar` will be used
#     :type rp_rearth: ndarray(float) or float or None
#     :param rp_rstar: desired planetary radius in stellar radii. Will be used if `rp_rearth` is None
#     :type rp_rstar: ndarray(float) or float or None
#     :return:
#     :rtype:
#     """
#     # We are not putting the following occurrence rate info to a separate file because it is easier like this.
#     # zhu and dong, planets per 100 stars (WARNING: WE ONLY DETECT ONE PER STAR, DISTRIBUTION CAN BE DIFFERENT!)
#     # I repeated some rows to extrapolate in order to interpolate
#     occurrences_table_enlarged = np.array([
#         [0.052, 0.052, 0.046, 0.064, 0.0045, 0.0017, 0.0017],
#         [0.047, 0.047, 0.041, 0.072, 0.0066, 0.0023, 0.0023],
#         [0.042, 0.042, 0.042, 0.081, 0.0058, 0.0052, 0.0052],
#         [0.031, 0.031, 0.031, 0.069, 0.015, 0.0044, 0.0044],
#         [0.042, 0.042, 0.042, 0.062, 0.013, 0.013, 0.013],
#         [0.049, 0.049, 0.049, 0.075, 0.024, 0.012, 0.012],
#         [0.1, 0.1, 0.1, 0.069, 0.017, 0.017, 0.017],
#         [0.1, 0.1, 0.1, 0.069, 0.017, 0.017, 0.017]])
#     # the following comes from the zhu and dong table labels, periods are in days
#     period_bin_centers_enlarged = 10 ** (np.array([1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6, 2.8]) + 0.1)[:-1]
#     # -> An exact empirical formula for the values in the table.
#     log10_period_bin_centers_enlarged = np.log10(period_bin_centers_enlarged)
#     # the planetary radius from the table was given in the earth radii
#     log2_rp_re_edges_enlarged = [-2, -1, 0., 1, 2, 3, 4, 5]
#     log2_rp_re_bin_centers_enlarged = (log2_rp_re_edges_enlarged + np.roll(log2_rp_re_edges_enlarged, -1))[:-1] / 2
#     SHOULD BE DIVIDED BY THE PERIOD STEP TO GET THE TRUE RATE, OTHERWISE IT IS NOT RATE, IT IS
#     JUST THE NUMBER OF EVENTS PER BIN!!!!!
#     oc_tab_enl_interp = RegularGridInterpolator(points=(log10_period_bin_centers_enlarged, log2_rp_re_bin_centers_enlarged),
#                                                 values=occurrences_table_enlarged,
#                                                 method='linear')  # occurrence table enlarged interpolated
#     if rp_rearth is None:
#         rp_rearth = rp_rstar * stellar_radius * R_sun_R_e
#     if not hasattr(rp_rearth, '__len__'):
#         rp_rearth = [rp_rearth]
#     if not hasattr(per_days, '__len__'):
#         per_days = [per_days]
#     points = [[np.log10(per_days[i]), np.log2(rp_rearth[i])] for i in range(len(per_days))]
#     result = oc_tab_enl_interp(points)
#     if len(result) == 1:
#         result = result[0]
#     return result


# MASKING KOIs
Mask_kepler_candidates = False
Mask_kepler_bright_candidates = True
Kepler_bright_candidate_mes = 20.
Kepler_bright_candidate_ses = 8.
Mask_kepler_false_positives = False

# TARGET CLASS PARAMETERS

Min_quarters_to_process = 4  # If there are fewer quarters in the data, the search will not run
# Bad fft check:
Amp_max_bad_fft = 300.  # Limit amplitude of the fft maximum (above it the maximum is bad)
Dif_max_bad_fft = 11.  # Maximal difference between the value at point and the surrounding. Bigger is bad.

# Search
P_min_search = 30  # days
P_max_search = 500  # days
P_short_min = 10  # days - short periodicity search for verification
P_short_max = 30  # days - short periodicity search for verification
Per_min_candidate = 50.  # days  # Only consider such triggers for follow-up after the search
Baseline_rate_period = P_min_search
Rp_re_max = 11  # this number of earth radii we search for now. 11 is Jupiter
Rp_rs_max = 0.05  # this relative radius we search for now. 0.04 is like Neptune to Sun
# this one is used for the main search
# P_chunks_search = 360  # will give approx 2GB per core, I increased it when changed duty cycle. # is not used for now
# -> this one used to be when I had 25 templates. Now I have more, so we must increase
# P_chunks_search = 1000  # 1440
Per_phases_per_chunk = 1200000  # It was 1300000 in run 12, but injection search took too much memory # With 1500000 some jobs got killed
# -> instead of specifying number of chunks, it is easier to specify the time-size of the array going to the memory
# this one is used for the timeslides:
# N_periods_per_chunk = 2000  # will give 1.2GB per core, but it was when I only ran for 1 template
# N_periods_per_chunk = 80  # Now I run for the full bank
# I don't know why the previous line (with P_chunks_search = 360) is so (it would give 222 periods per chunk)
Score_candidate_search = 40.  # mes in units of SNR SQUARED TEMPLATE-MARGINALIZED
Score_candidate_postprocess = 49.
# The following `close` parameters are for the maximal score selection. Sometimes the maximal integral corresponds
# to the chunk that does not capture the true peak. But the integral does capture the true peak because of the
# period overlapping. We check whether there is a score which is no less
# than the maximum minus `Delta_integral_close` and gives the gain in statistic more than `Delta_stat_close`. If
# there is such a score, it is selected as a maximum.
Delta_integral_close = 0.2
Delta_stat_close = 1.

N_slides_total = 50000  # for one template and narrow periods for all the runs
# I thought to make chunk 50000 for one template and narrow periods per one run (about 14 hours) (rough 1 sec per one periodicity search)
# But there was a timeout, so
N_slides_chunk = 200
N_injections_total = 10000  # total number of all injections (all radii and periods etc.)
N_injections_chunk = 40  # the previous one was perhaps run_8, but now we have full bank
# N_injections_chunk = 20  # number of injections per job so that it will take 10/3 hours (120/3 sec per one inj detection)
# I decreased it from 50 to 20 in run_7 because now we need more time for veto
Per_wing_injection_and_slides_relative = 2 / 50  # Range to the left and to the right of the best period in units of period. Will be used
# to generate injections and time slides with variable periods around the best one
Per_wing_injection_and_slides_absolute = 2 / 50  # Range to the left and to the right of the best period in days. Will be used
Per_wing_injection_detection = 0.5  # days. Range to the left and to the right of the injected period used in the injection recovery
# Max_phase_mismatch_rel = 0.001  # Maximal relative phase difference for defining successful injections
Max_phase_mismatch_to_dur = 1.5  # Maximal phase difference in units of template duration for defining successful injections
Default_min_plot_stat = 10
Default_max_plot_stat = 140

Data_location = os.path.join(os.getcwd(), 'light_curves')  # in this directory, a file named after target_id will be created
Output_location = os.path.join(os.getcwd(), 'outputs')  # in this directory, a file named after target_id will be created
Triggers_output_location = os.path.join(os.getcwd(), 'triggers_outputs')
Target_pixels_location = os.path.join(os.getcwd(), 'target_pixels')
Run_root_folder = os.path.join(os.getcwd(), 'outputs')
Problems_folder = None
Run_name = 'run_11'

# Kepler_files_folder = '/Users/orynai/Work/Kepler_Project/Kepler_Earths/KeplerFiles'
Kepler_files_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'KeplerFiles')

if Working_place == 'Home':
    pass
    # Kepler_files_folder = '/Users/orynai/Work/Kepler_Project/Kepler_Earths/KeplerFiles'

if Working_place == 'Marvin':
    Data_location = '/Collaboration-Kepler/KeplerData'
    Output_location = '/Collaboration-Kepler/orynai/output_' + str(Run_name)
    Triggers_output_location = '/Collaboration-Kepler/orynai/triggers_output_' + str(Run_name)
    Run_root_folder = '/SMALLDATA/orynai/' + str(Run_name)
    Target_pixels_location = '/Collaboration-Kepler/KeplerTargetPixels'

if Working_place == 'WEXAC':
    Data_location = '/home/labs/barakz/Collaboration-Kepler/KeplerData'
    Output_location = '/home/labs/barakz/Collaboration-Kepler/orynai/output_' + str(Run_name)
    Triggers_output_location = '/home/labs/barakz/Collaboration-Kepler/orynai/triggers_output_' + str(Run_name)
    # Kepler_files_folder = '/home/labs/barakz/Collaboration-Kepler/KeplerFiles'
    Run_root_folder = '/home/labs/barakz/orynai/' + str(Run_name)
    Target_pixels_location = '/home/labs/barakz/Collaboration-Kepler/KeplerTargetPixels'
    Problems_folder = '/home/labs/barakz/orynai/' + str(Run_name) + '/problems_stars'

Koi_dict_filename = 'KOI_catalog_dict.pickle'
Ttv_dict_filename = 'TTV_catalog_dict.pickle'
Gaia_kepler_filename = 'kepler_dr3_good.fits'
Villanova_dict_filename = 'villanova_catalog_dict.pickle'
Limb_darkening_array_filename = 'limb_darkening_kepler_array.npy'
Flares_filename = 'Flares_Kepler.txt'

Koi_dict_file = os.path.join(Kepler_files_folder, Koi_dict_filename)
Ttv_dict_file = os.path.join(Kepler_files_folder, Ttv_dict_filename)
Gaia_kepler_file = os.path.join(Kepler_files_folder, Gaia_kepler_filename)
Villanova_dict_file = os.path.join(Kepler_files_folder, Villanova_dict_filename)
Limb_darkening_array_file = os.path.join(Kepler_files_folder, Limb_darkening_array_filename)
Flares_file = os.path.join(Kepler_files_folder, Flares_filename)

# Gaia_kepler_file = '/home/orynai/Kepler_Notebooks/kepler_dr3_good.fits'
# Default_koi_file = '/home/orynai/Kepler_Notebooks/cumulative_2022.05.30_11.44.28.csv'
# Default_targets_info_file = '/Users/orynaivashtenko/Work/Kepler_Project/input_stars_R2_and_R3_mass.txt'
# Default_koi_file = '/Users/orynaivashtenko/Work/Kepler_Project/cumulative_2022.05.30_11.44.28.csv'  # used in run_5 for validation
# # # Default_koi_catalog_file = '/Users/orynaivashtenko/Work/Kepler_Project/cumulative_2022.10.18_03.41.17.csv'
# containing koi names and many fields
# Default_ttv_catalog_file = '/Users/orynaivashtenko/Work/Kepler_Project/TTV_Catalog_Table3_O-C.txt'

if force_data_output_locations_creation:
    if not os.path.isdir(Data_location):
        os.mkdir(Data_location)
    if not os.path.isdir(Output_location):
        os.mkdir(Output_location)

Search_history_targets_filename = 'search_history_targets.txt'
Pre_filtering_history_targets_filename = 'pre_filtering_history_targets.txt'

Threshold_crossing_list_filename = 'threshold_crossing_kid_info.txt'
Threshold_g_crossing_list_filename = 'threshold_g_crossing_kid_info.txt'
Threshold_crossing_list_multiple_triggers_filename = 'threshold_crossing_list_multiple.txt'
Threshold_g_crossing_list_multiple_triggers_filename = 'threshold_g_crossing_list_multiple.txt'
Target_ids_prefiltered_filename = 'triggers_prefiltered.txt'
Target_ids_g_prefiltered_filename = 'triggers_g_prefiltered.txt'
Maximal_short_periodicity_triggers_filename = 'short_periods_veto_maximal_triggers.txt'
Maximal_short_periodicity_g_triggers_filename = 'short_periods_veto_maximal_g_triggers.txt'
Triggers_short_periodicity_filtered_filename = 'triggers_short_periodicity_filtered.txt'
Triggers_g_short_periodicity_filtered_filename = 'triggers_g_short_periodicity_filtered.txt'
Triggers_for_slides_injections_filename = 'triggers_for_slides_injections.txt'
Triggers_g_for_slides_injections_filename = 'triggers_g_for_slides_injections.txt'
Tids_that_did_not_run_filename = 'Tids_that_did_not_run.txt'
Non_slidable_tids_short_period_filename = 'non_slidable_tids__having_10_30_trigger.txt'
Seeds_scrambling_run_filename = 'seeds_scrambling_run.txt'
Threshold_crossing_scrambled_list_filename = 'threshold_crossing_scrambled_kid_info.txt'
Threshold_g_crossing_scrambled_list_filename = 'threshold_g_crossing_scrambled_kid_info.txt'
Threshold_crossing_scrambled_list_multiple_triggers_filename = 'threshold_crossing_scrambled_list_multiple.txt'
Threshold_g_crossing_scrambled_list_multiple_triggers_filename = 'threshold_g_crossing_scrambled_list_multiple.txt'

Search_history_targets_file = os.path.join(Run_root_folder, Search_history_targets_filename)
Pre_filtering_history_targets_file = os.path.join(Run_root_folder, Pre_filtering_history_targets_filename)
# 1) Pipeline selects the threshold-crossing events:
Threshold_crossing_list_file = os.path.join(Run_root_folder, Threshold_crossing_list_filename)
Threshold_g_crossing_list_file = os.path.join(Run_root_folder, Threshold_g_crossing_list_filename)
# 2) Pre-filtering vetos check those events:
Triggers_prefiltered_file = os.path.join(Run_root_folder, Target_ids_prefiltered_filename)
Triggers_g_prefiltered_file = os.path.join(Run_root_folder, Target_ids_g_prefiltered_filename)
# 3) There is a short periodicity search for the pre-filtered events:
Maximal_short_periodicity_triggers_file = os.path.join(Run_root_folder, Maximal_short_periodicity_triggers_filename)
Maximal_short_periodicity_g_triggers_file = os.path.join(Run_root_folder, Maximal_short_periodicity_g_triggers_filename)
Triggers_short_periodicity_filtered_file = os.path.join(Run_root_folder, Triggers_short_periodicity_filtered_filename)
Triggers_g_short_periodicity_filtered_file = os.path.join(Run_root_folder, Triggers_g_short_periodicity_filtered_filename)
# 4) After the short periodicity search, triggers are selected for the slides and injections:
Triggers_for_slides_injections_file = os.path.join(Run_root_folder, Triggers_for_slides_injections_filename)
Triggers_g_for_slides_injections_file = os.path.join(Run_root_folder, Triggers_g_for_slides_injections_filename)
Non_slidable_tids_short_period_file = os.path.join(Run_root_folder, Non_slidable_tids_short_period_filename)
# x) If there is a scrambled run:
Seeds_scrambling_run_file = os.path.join(Run_root_folder, Seeds_scrambling_run_filename)
Threshold_crossing_scrambled_list_file = os.path.join(Run_root_folder, Threshold_crossing_scrambled_list_filename)
Threshold_g_crossing_scrambled_list_file = os.path.join(Run_root_folder, Threshold_g_crossing_scrambled_list_filename)


Summary_target_txt_filename_postfix = 'target_summary.txt'
Problems_target_txt_filename_postfix = 'problems.txt'
Dict_target_txt_filename_postfix = 'target_class_dict.txt'
Pickle_target_small_attributes_filename_postfix = 'target_class_small_attributes.pickle'
Conditioned_data_filename_postfix = 'conditioned_data'
No_veto_no_score_cor_stats_filename_postfix = 'no_veto_no_cor_stats'
Connected_statistic_data_filename_postfix = 'connected_stats'
Best_periodic_scores_filename_postfix = 'best_periodic_scores_all_p'
Best_scrambled_periodic_scores_filename_postfix = 'best_scrambled_periodic_scores_all_p'
Best_short_periodic_scores_filename_postfix = '_best_periodic_scores_'+str(P_short_min)+'_'+str(P_short_max)+'.npy'
Injection_parameters_filename_postfix = 'injection_parameter_sets.txt'
All_injection_parameters_filename_postfix = 'all_injection_parameter_sets'
Injections_filename_postfix = 'injections.txt'
# these stupid names are because of the search which checks `in`, and I need one file not to be in another one
Slides_res_filename_postfix = 'scramble_res_scramble'
Slides_clean_res_filename_postfix = 'scramble_res_clean'

#########################################################################################################################
# Template Bank
#########################################################################################################################
Template_bank_folder_name = 'template_bank'
Default_template_bank_folder = os.path.join(os.path.dirname(__file__), Template_bank_folder_name)
Default_templates_filename = 'templates.npy'
Default_templates_shifts_filename = 'templates_shifts.npy'
Default_shifts_filename = 'shifts.npy'
Default_durations_filename = 'durations.npy'  # days
Default_indices_of_templates_filename = 'indices_of_templates.npy'
Default_templates_of_indices_filename = 'templates_of_indices.npy'
Default_templates_fd_for_injections_filename = 'templates_whitened_normalized_fd_for_injections.npy'
Default_template_params_filename = 'params_templates.npy'
Precomputed_tables_location = Default_template_bank_folder
Log_occurrences_templates_periods_m13r_per_edges_filename = 'log_occurrences_templates_periods_m13r_per_edges.npy'
Log_second_derivs_cos2_shifts_templates_filename = 'log_second_derivs_cos2_shifts_templates.npy'
Log_cos2_derivs_templates_filename = 'log_cos2_derivs_templates.npy'
Log_template_prior_m13r_per_edges_filename = 'log_template_prior_m13r_per_edges.npy'
Transit_probabilities_m13r_periods_filename = 'm13r_period_transit_probabilities.npy'

# -> effective_filter_whitened normalized to have unit time-domain norm, shifter all to have peaks to the left
# (even the subbin shifts are forcefully shifted, as will the models be)

#########################################################################################################################
# Class parameters
#########################################################################################################################
Histogram_mes_binwid = 1.  # Width of the bin in the MES[SNR^2] histogram of background and injections

Location_attributes = ['output_location', 'data_location', 'gaia_kepler_file', 'koi_dict_file',
                       'ttv_dict_file', 'villanova_dict_file', 'limb_darkening_array_file']

Star_attributes = ['stellar_radius', 'stellar_mass', 'kepmag', 'phot_g_mean_mag', 'parallax',
                   'gaia_abs_gmag', 'gaia_b_r', 'teff', 'ra', 'dec', 'm_1_3_r_s']


#########################################################################################################################
# Simulations
#########################################################################################################################
Simulations_folder_name = 'simulations'
Default_stored_asd_file_filename = 'stored_smoothed_asd_from_fft.txt'  # Median of many measurements
Default_effective_wf_fd_filename = 'effective_whitening_filter_fourier_domain.npy'  # it is better to use this, it is with Tuckey
