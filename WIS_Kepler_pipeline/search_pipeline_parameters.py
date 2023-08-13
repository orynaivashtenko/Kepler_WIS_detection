import os

# PARAMETERS INITIALIZATION

# Masking:
Nsig_outliers_0 = 6.
Nsig_outliers_0_pos = 6.  # 5. #it is possible to mask positive and negative outliers differently
# we want to make inversion runs, therefore we need this masking to be equivalent in positive and negative directions
Nsig_outliers_0_neg = 6.
Nsig_outliers_psd = 4
Max_small_hole = 3
Nsig_outliers_w = 6
Nsig_outliers_w_pos = 6.
Nsig_outliers_w_neg = 6.
Nsig_outliers_stat = 3.
# Masking. Detrending in outliers detection:
Detr_length_outliers = 128
Detr_method_outliers = 'median'
Detr_endmethod_outliers = 'mean'
Detr_ends_outliers = 'symmetric'
Detr_endlen_outliers = None

# Measuring PSD:
Nperseg_psd = 128
Noverlap_psd = 0.5
Window_psd = 'hann'
Npsdsmin = 40
Use_rfft = True
Subtract_mean_in_psd = True
Smoothen_psds = True
Tukey_steps_len_psd = 16
Conditional_zero_zeroth_freq_wf = True
Always_zero_zeroth_freq_wf = False

# Notch:
Nsig_notch = 5.
Irl_notch = 3000.
Htol_notch = 1e-5
Minval_notch = 0

# Detrending original fluxes to find the trends:
Detr_length = 800  # in order not to spoil even the largest transit
Detr_method = 'median'
Detr_endmethod = 'mean'
Detr_ends = 'symmetric'
Detr_endlen = Detr_length // 4
Detr_return_trend_ = False
Detr_keep_overall_level = False

# Templates and detection
Template_bank_from_file = True
Templates_subbin_shift = True
Template_ind_detection = 'all'
Margin_snr_veto = 2.
Fn_fit_veto = 0.01
Maxdev_fit_veto = 4.0
Zero_ends_len_abs_fit_veto = None  # this means that zero_ends_len_rel_fit_veto will be used
Zero_ends_len_rel_fit_veto = 1  # this means that it will be equal to the 1 transit (template) duration
Threshold_mes_veto = 5.
Veto_threshold_for_mes = 2.05  # false alarm 5%
N_parts_veto_snr = 4
N_parts_veto_len = 6
Inn_prod_min_veto = 0.2
Delta_snr2_wide_templ_veto = 4.  # should correspond to 2 sigma
Threshold_to_std_veto = 35 ** 0.5

# Processing the statistic
Nperseg_moving_var = 512
# ->make sure it is smaller than period
Ngap_moving_var = 192
Nsamples_min_moving_var = 128
Apply_score_correction = True

Pmin = 30  # days
Pmax = 500
Duty_cycle = 0.05  # this was 0.1, but I changed it because sometimes it is not enough
Std_time_scrambling = 10.  # days # for time slides
Max_hole_pieces_scrampling = 30  # <1 day # Hole which is considered a reason to split the statistic to two pieces which can be shuffled
Max_piece_scrambling = int(50 * 24 * 2)  # 50 days # Maximal possible length of a piece which will be shuffled
Min_piece_scrambling = int(5 * 24 * 2)  # 5 days # Minimal possible length of a piece which will be shuffled
N_per_temp_saving = 10  # how many best candidates per template to save
# track_process = False

Search_results_index_per = 0
Search_results_index_stat = 1
Search_results_index_sum_tr_times = 2
Search_results_index_sum_stats = 3
Search_results_index_sum_trends = 4
Search_results_index_sum_vars = 5
Search_results_index_shifts = 6
Search_results_index_per_list = 7
Search_results_length = 8

Default_output_location = os.getcwd()  # os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'Outputs_of_pipeline')
Default_data_location = os.getcwd()

# parameters_file = os.path.join(os.path.dirname(__file__), 'pipeline_parameters.txt')
# PARAMETERS CAN BE SPECIFIED IN THE FILE, AND THEY WILL BE READ FROM IT
# variables_from_file(parameters_file)
Template_bank_file = os.path.join(os.path.dirname(__file__), 'templates', 'template_bank.txt')
Shifts_template_bank_file = os.path.join(os.path.dirname(__file__), 'templates', 'shifts_template_bank.txt')
