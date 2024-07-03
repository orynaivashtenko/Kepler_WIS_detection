import os

os.environ["OMP_NUM_THREADS"] = "1"  #
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"  #
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import pipeline_parameters as pp

Gaia_kepler_file = pp.Gaia_kepler_file
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

Initial_trigger_list_file = os.path.join(pp.Threshold_crossing_lists_folder, Initial_trigger_list_filename)
Suspicious_frequency_reasons_triggers_file = os.path.join(pp.Threshold_crossing_lists_folder,
                                                          Suspicious_frequency_reasons_triggers_filename)
Suspicious_frequency_filtered_triggers_file = os.path.join(pp.Threshold_crossing_lists_folder,
                                                           Suspicious_frequency_filtered_triggers_filename)


Summary_tce_txt_filename_postfix = 'tce_summary.txt'
Dict_tce_txt_filename_postfix = 'tce_class_dict.txt'
Pickle_tce_small_attributes_filename_postfix = 'tce_class_small_attributes.pickle'
Pickle_tce_all_attributes_filename_postfix = 'tce_class_all_attributes.pickle'

# Veto target pixels parameters:
Part_of_cumsum_2_targ_pix = 0.85
Nsig_outliers_0_pos_tests = 5.
Nsig_outliers_0_neg_tests = 15.
Threshold_centroid_snr_2 = 45.
Threshold_false_negative_bg_target_pixels_mean = 0.01
Threshold_false_negative_bg_target_pixels_max = 0.01
