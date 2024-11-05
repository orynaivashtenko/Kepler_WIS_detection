#!/usr/bin/env python3

import os

Pipeline_folder = os.path.dirname(__file__)
force_data_output_locations_creation = False
# --------------------------------------------------------------------------------------------
Run_name = 'run_12'
# --------------------------------------------------------------------------------------------
Working_place = None  # On which machine it is running: my office Mac ('Home'), Marvin or WEXAC
# Working_place = 'Home' # 'Marvin'  # 'WEXAC'
if Working_place is None:
    if '/Users/orynai/Work/Kepler_Project' in Pipeline_folder:
        Working_place = 'Home'
    elif '/SMALLDATA' in Pipeline_folder or '/home/orynai/' in Pipeline_folder:
        Working_place = 'Marvin'
    elif '/home/labs/barakz' in Pipeline_folder:
        Working_place = 'WEXAC'

# Definitions for Home:
Data_location = '/Users/orynai/Work/Kepler_Project/KeplerDataLocal'  # in this directory, a file named after target_id will be created
Target_pixels_location = '/Users/orynai/Work/Kepler_Project/KeplerTargetPixelsLocal'
Run_outputs_folder = '/Users/orynai/Work/Kepler_Project/run_12_local'
Output_location = os.path.join(Run_outputs_folder, 'output_' + str(Run_name))
Triggers_plots_location = os.path.join(os.getcwd(), 'triggers_outputs')
Run_technical_folder = Run_outputs_folder

if Working_place == 'Home':
    pass

if Working_place == 'Marvin':
    Data_location = '/Collaboration-Kepler/KeplerData'
    Target_pixels_location = '/Collaboration-Kepler/KeplerTargetPixels'
    Run_outputs_folder = '/Collaboration-Kepler/orynai/' + str(Run_name)
    Run_technical_folder = '/SMALLDATA/orynai/' + str(Run_name)

if Working_place == 'WEXAC':
    Data_location = '/home/labs/barakz/Collaboration-Kepler/KeplerData'
    Target_pixels_location = '/home/labs/barakz/Collaboration-Kepler/KeplerTargetPixels'
    Run_outputs_folder = '/home/labs/barakz/Collaboration-Kepler/orynai/' + str(Run_name)
    Run_technical_folder = '/home/labs/barakz/orynai/' + str(Run_name)

Output_location = os.path.join(Run_outputs_folder, 'output_' + str(Run_name))
Triggers_plots_location = os.path.join(Run_outputs_folder, 'triggers_plots_' + str(Run_name))
Trigger_lists_folder = os.path.join(Run_outputs_folder, 'triggers_' + str(Run_name))
Problems_folder = os.path.join(Run_outputs_folder, 'problems_stars')
Problems_folder_scrambled = os.path.join(Run_outputs_folder, 'problems_stars_scrambled')
Kepler_files_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'KeplerFiles')

if force_data_output_locations_creation:
    if not os.path.isdir(Data_location):
        os.mkdir(Data_location)
    if not os.path.isdir(Target_pixels_location):
        os.mkdir(Target_pixels_location)
    if not os.path.isdir(Run_outputs_folder):
        os.mkdir(Run_outputs_folder)
    if not os.path.isdir(Output_location):
        os.mkdir(Output_location)

# ############################################################################################################
# Kepler, Gaia and other files
Koi_dict_filename = 'KOI_catalog_dict.pickle'
Ttv_dict_filename = 'TTV_catalog_dict.pickle'
Gaia_kepler_filename = 'kepler_dr3_good.fits'
Villanova_dict_filename = 'villanova_catalog_dict.pickle'
Limb_darkening_array_filename = 'limb_darkening_kepler_array.npy'
Flares_filename = 'Flares_Kepler.txt'
Kepler_tce_table_filename = 'q1_q17_dr25_tce_2024.csv'

Koi_dict_file = os.path.join(Kepler_files_folder, Koi_dict_filename)
Ttv_dict_file = os.path.join(Kepler_files_folder, Ttv_dict_filename)
Gaia_kepler_file = os.path.join(Kepler_files_folder, Gaia_kepler_filename)
Villanova_dict_file = os.path.join(Kepler_files_folder, Villanova_dict_filename)
Limb_darkening_array_file = os.path.join(Kepler_files_folder, Limb_darkening_array_filename)
Flares_file = os.path.join(Kepler_files_folder, Flares_filename)
Kepler_tce_table_file = os.path.join(Kepler_files_folder, Kepler_tce_table_filename)
Earth_pointing_times_file = os.path.join(Kepler_files_folder, 'earth_pointing_times.txt')
Quarter_start_times_file = os.path.join(Kepler_files_folder, 'quarter_start_times.txt')
Quarter_end_times_file = os.path.join(Kepler_files_folder, 'quarter_end_times.txt')
Villanova_catalog_file = os.path.join(Kepler_files_folder, 'villanova_catalog')
Villanova_periods_phases_kmags_relevant_filename = 'villanova_periods_phases_kmags_relevant.npy'
Villanova_periods_phases_kmags_relevant_file = os.path.join(Kepler_files_folder, Villanova_periods_phases_kmags_relevant_filename)
Villanova_kic_relevant_filename = 'villanova_from_kic_relevant.ecsv'
Villanova_kic_relevant_file = os.path.join(Kepler_files_folder, Villanova_kic_relevant_filename)
Villanova_kic_query_relevant_filename = 'villanova_kic_query.ecsv'
Villanova_kic_query_relevant_file = os.path.join(Kepler_files_folder, Villanova_kic_query_relevant_filename)
Table_koi_file = os.path.join(Kepler_files_folder, 'KOI_table_2024.csv')

# ############################################################################################################
# Run results, triggers
# ############################################################################################################
Search_history_targets_filename = 'search_history_targets.txt'
Pre_filtering_history_targets_filename = 'pre_filtering_history_targets.txt'

Threshold_crossing_list_filename = 'threshold_crossing_kid_info.txt'
Threshold_g_crossing_list_filename = 'threshold_g_crossing_kid_info.txt'
Threshold_crossing_list_multiple_triggers_filename = 'threshold_crossing_list_multiple.txt'
Threshold_g_crossing_list_multiple_triggers_filename = 'threshold_g_crossing_list_multiple.txt'
Triggers_prefiltered_filename = 'triggers_prefiltered.txt'
Triggers_g_prefiltered_filename = 'triggers_g_prefiltered.txt'
Maximal_short_periodicity_triggers_filename = 'short_periods_veto_maximal_triggers.txt'
Maximal_short_periodicity_g_triggers_filename = 'short_periods_veto_maximal_g_triggers.txt'
Triggers_short_periodicity_filtered_filename = 'triggers_short_periodicity_filtered.txt'
Triggers_g_short_periodicity_filtered_filename = 'triggers_g_short_periodicity_filtered.txt'
Triggers_for_slides_injections_filename = 'triggers_for_slides_injections.txt'
Triggers_g_for_slides_injections_filename = 'triggers_g_for_slides_injections.txt'
Tids_that_did_not_run_filename = 'tids_that_did_not_run.txt'
Non_slidable_tids_short_period_filename = 'non_slidable_tids__having_10_30_trigger.txt'
Seeds_scrambling_run_filename = 'seeds_scrambling_run.txt'
Threshold_crossing_scrambled_list_filename = 'threshold_crossing_scrambled_kid_info.txt'
Threshold_g_crossing_scrambled_list_filename = 'threshold_g_crossing_scrambled_kid_info.txt'
Threshold_crossing_scrambled_list_multiple_triggers_filename = 'threshold_crossing_scrambled_list_multiple.txt'
Threshold_g_crossing_scrambled_list_multiple_triggers_filename = 'threshold_g_crossing_scrambled_list_multiple.txt'

Search_history_targets_file = os.path.join(Run_outputs_folder, Search_history_targets_filename)
Pre_filtering_history_targets_file = os.path.join(Run_outputs_folder, Pre_filtering_history_targets_filename)
# 1) Pipeline selects the threshold-crossing events:
Threshold_crossing_list_file = os.path.join(Trigger_lists_folder, Threshold_crossing_list_filename)
Threshold_g_crossing_list_file = os.path.join(Trigger_lists_folder, Threshold_g_crossing_list_filename)
# 2) Pre-filtering vetos check those events:
Triggers_prefiltered_file = os.path.join(Trigger_lists_folder, Triggers_prefiltered_filename)
Triggers_g_prefiltered_file = os.path.join(Trigger_lists_folder, Triggers_g_prefiltered_filename)
# 3) There is a short periodicity search for the pre-filtered events:
Maximal_short_periodicity_triggers_file = os.path.join(Trigger_lists_folder, Maximal_short_periodicity_triggers_filename)
Maximal_short_periodicity_g_triggers_file = os.path.join(Trigger_lists_folder, Maximal_short_periodicity_g_triggers_filename)
Triggers_short_periodicity_filtered_file = os.path.join(Trigger_lists_folder, Triggers_short_periodicity_filtered_filename)
Triggers_g_short_periodicity_filtered_file = os.path.join(Trigger_lists_folder, Triggers_g_short_periodicity_filtered_filename)
# 4) After the short periodicity search, triggers are selected for the slides and injections:
Triggers_for_slides_injections_file = os.path.join(Trigger_lists_folder, Triggers_for_slides_injections_filename)
Triggers_g_for_slides_injections_file = os.path.join(Trigger_lists_folder, Triggers_g_for_slides_injections_filename)
Non_slidable_tids_short_period_file = os.path.join(Trigger_lists_folder, Non_slidable_tids_short_period_filename)
# x) If there is a scrambled run:
Seeds_scrambling_run_file = os.path.join(Run_outputs_folder, Seeds_scrambling_run_filename)
Threshold_crossing_scrambled_list_file = os.path.join(Run_outputs_folder, Threshold_crossing_scrambled_list_filename)
Threshold_g_crossing_scrambled_list_file = os.path.join(Run_outputs_folder, Threshold_g_crossing_scrambled_list_filename)

# ################################################################################################
# WisTCE, pre-filtering, and Catalog
# ################################################################################################

# Triggers and results files
Summary_tce_txt_filename_postfix = 'tce_summary.txt'
Dict_tce_txt_filename_postfix = 'tce_class_dict.txt'
Pickle_tce_small_attributes_filename_postfix = 'tce_class_small_attributes.pickle'
Pickle_tce_all_attributes_filename_postfix = 'tce_class_all_attributes.pickle'
Gaia_trigger_table_filename = 'gaia_trigger_table.ecsv'
Gaia_trigger_table_file = os.path.join(Trigger_lists_folder, Gaia_trigger_table_filename)
"""The Gaia table corresponding to the individual Kepler target"""
Neighbors_folder = os.path.join(Run_outputs_folder, 'neighbors_' + Run_name)
Problems_tce_txt_filename_postfix = 'tce_problems.txt'

# ----------------------------------------------------------------------
# Pre-filtering byproducts
Suspicious_frequency_reasons_triggers_filename = 'suspicious_frequency_reasons_triggers.npy'
Suspicious_frequency_filtered_triggers_filename = 'suspicious_frequency_filtered_triggers.npy'
Suspicious_frequency_reasons_triggers_file = os.path.join(Trigger_lists_folder, Suspicious_frequency_reasons_triggers_filename)
Suspicious_frequency_filtered_triggers_file = os.path.join(Trigger_lists_folder, Suspicious_frequency_filtered_triggers_filename)

# ----------------------------------------------------------------------
# Catalog files
Trigger_table_filename = 'trigger_table.ecsv'
Trigger_table_file = os.path.join(Trigger_lists_folder, Trigger_table_filename)
Trigger_table_format = 'ascii.ecsv'

# ############################################################################################################
# TCE data saving postfixes
# ############################################################################################################
Connected_stats_centroid_x_filename_postfix = 'connected_stats_centroid_x'
Periodic_stats_centroid_x_filename_postfix = 'periodic_stats_centroid_x'
Connected_stats_centroid_y_filename_postfix = 'connected_stats_centroid_y'
Periodic_stats_centroid_y_filename_postfix = 'periodic_stats_centroid_y'

Connected_stats_star_pixels_filename_postfix = 'connected_stats_star_pixels'
Periodic_stats_star_pixels_filename_postfix = 'periodic_stats_star_pixels'
Connected_stats_bg_pixels_filename_postfix = 'connected_stats_bg_pixels'
Periodic_stats_bg_pixels_filename_postfix = 'periodic_stats_bg_pixels'

Fold_residuals_flux_fd_chi2_filename_postfix = 'fold_residuals_flux_fd_chi2'
Mf_scores_residuals_filename_postfix = 'mf_scores_residuals'
Folded_binned_flux_template_filename_postfix = 'folded_binned_flux_template'
Fold_residuals_flux_td_chi_filename_postfix = 'fold_residuals_flux_td_chi'
Folded_binned_wh_flux_model_even_odd_filename_postfix = 'folded_binned_wh_flux_model_even_odd'
# ############################################################################################################
# Search pipeline data saving postfixes
# ############################################################################################################
Summary_target_txt_filename_postfix = 'target_summary.txt'
Problems_target_txt_filename_postfix = 'problems.txt'
Dict_target_txt_filename_postfix = 'target_class_dict.txt'
Pickle_target_small_attributes_filename_postfix = 'target_class_small_attributes.pickle'
Conditioned_data_filename_postfix = 'conditioned_data'
No_veto_no_score_cor_stats_filename_postfix = 'no_veto_no_cor_stats'
Connected_statistic_data_filename_postfix = 'connected_stats'
Best_periodic_scores_filename_postfix = 'best_periodic_scores_all_p'
Best_scrambled_periodic_scores_filename_postfix = 'best_scrambled_periodic_scores_all_p'
Best_short_periodic_scores_filename_postfix = 'best_periodic_scores_short'
Injection_parameters_filename_postfix = 'injection_parameter_sets.txt'
All_injection_parameters_filename_postfix = 'all_injection_parameter_sets'
Injections_filename_postfix = 'injections.txt'
# these stupid names are because of the search which checks `in`, and I need one file not to be in another one
Slides_res_filename_postfix = 'scramble_res_scramble'
Slides_clean_res_filename_postfix = 'scramble_res_clean'
# Other:
Psd_for_injections_filename_postfix = 'psd_for_injections.npy'  # was not used eventually
Periodic_stats_filename_postfix = 'stats_periodic'

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
# Simulations
#########################################################################################################################
Simulations_folder_name = 'simulations'
Default_stored_asd_file_filename = 'stored_smoothed_asd_from_fft.txt'  # Median of many measurements
Default_effective_wf_fd_filename = 'effective_whitening_filter_fourier_domain.npy'  # it is better to use this, it is with Tuckey

#########################################################################################################################
# Plotting
#########################################################################################################################
Trigger_pdf_report_filename_postfix = 'trigger_report.pdf'

Background_injection_histograms_plot_filename_postfix = 'background_injection_histograms.png'
Orig_fluxes_plot_filename_postfix = 'orig_fluxes.png'
Red_fluxes_plot_filename_postfix = 'red_fluxes.png'
Whitened_fluxes_plot_filename_postfix = 'whitened_fluxes.png'
Single_event_statistics_plot_filename_postfix = 'single_event_statistics.png'
All_series_plot_filename_postfix = 'all_series.png'
Folded_whitened_fluxes_plot_filename_postfix = 'folded_whitened_fluxes.png'
Folded_whitened_fluxes_zoomed_plot_filename_postfix = 'folded_whitened_fluxes_zoomed.png'
Folded_statistics_plot_filename_postfix = 'folded_statistics.png'
Folded_statistics_zoomed_plot_filename_postfix = 'folded_statistics_zoomed.png'
All_folds_plot_filename_postfix = 'all_folds.png'
All_folds_zoomed_filename_postfix = 'all_folds_zoomed.png'
Periodogram_chunks_plot_filename_postfix = 'periodogram_chunks.png'
Residuals_fold_veto_histogram_plot_filename_postfix = 'residuals_fold_veto_histogram.png'
Veto_parts_fold_plot_filename_postfix = 'veto_parts_fold.png'
Neighbor_maps_plot_filename_postfix = 'neighbor_maps.png'
Injections_radii_durations_score_plot_filename_postfix = 'inj_radii_dur_score.png'
Corner_injections_orbital_plot_filename_postfix = 'corner_inj_orbital.png'
Corner_injections_times_plot_filename_postfix = 'corner_inj_times.png'
Corner_injections_scores_plot_filename_postfix = 'corner_inj_scores.png'
Centroid_cloud_plot_filename_postfix = 'centroid_cloud.png'
Centroid_periodic_plot_filename_postfix = 'centroid_periodic.png'
Centroid_coordinate_pieces_plot_filename_postfix = 'centroid_coordinate_pieces.png'
Centroid_snr2_seasons_plot_filename_postfix = 'centroid_snr2_seasons.png'
Centroid_full_stats_plot_filename_postfix = 'centroid_full_stats.png'
Star_bg_pixels_ppm_plot_filename_postfix = 'star_bg_pixels_ppm.png'
Season_pieces_snr_plot_filename_postfix = 'season_pieces_snr.png'
Season_pieces_wh_flux_plot_filename_postfix = 'season_pieces_wh_flux.png'
Even_odd_wh_flux_plot_filename_postfix = 'even_odd_wh_flux.png'
Even_odd_snr_plot_filename_postfix = 'even_odd_snr.png'
Even_odd_wh_flux_and_snr_plot_filename_postfix = 'even_odd_wh_flux_and_snr.png'
Even_odd_wh_flux_from_veto_plot_filename_postfix = 'even_odd_wh_flux_from_veto.png'
Neighbors_maps_plot_filename_postfix = 'neighbors_maps.png'
Star_bg_full_stats_plot_filename_postfix = 'star_bg_full_stats.png'
Transit_info_plot_filename_postfix = 'transit_info.png'

# locs.Folded_red_fluxes_plot_filename_postfix
# locs.Neighbor_maps_plot_filename_postfix