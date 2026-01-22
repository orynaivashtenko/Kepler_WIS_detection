import os
os.environ["OMP_NUM_THREADS"] = "1"  #
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"  #
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import numpy as np

# Those are needed for the pipeline, they will be imported there
from pipeline_parameters import Conditioned_data_filename_postfix, No_veto_no_score_cor_stats_filename_postfix, \
    Connected_statistic_data_filename_postfix, Best_periodic_scores_filename_postfix, \
    Psd_for_injections_filename_postfix, Periodic_stats_filename_postfix

# Locations:
Default_output_location = os.getcwd()  # os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'Outputs_of_pipeline')
Default_data_location = os.getcwd()

# PARAMETERS INITIALIZATION

Default_end_time = 1560.  # If the end of the mission is needed for some estimations
Default_m_1_3_r_s = 1.  # Mass of the star in units of solar mass divided by the radius of the star to the power 1/3
# Normalization
Average_flux_method = 'mean'  # can be 'mean' or 'median

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
Nsig_outliers_stat = 4.  # is supposed to happen less than once per quarter

# Masking. Detrending in outliers detection:
Detr_length_outliers = 256
Detr_method_outliers = 'median'
Detr_endmethod_outliers = 'mean'

# Measuring PSD:
Nperseg_psd = 128
Noverlap_psd = 0.5
Window_psd = 'hann'
Npsdsmin = 40
Use_rfft = True
Subtract_mean_in_psd = True
Multi_resolution_psds = True
Always_zero_zeroth_freq_psd = False
Tukey_steps_len_psd = 16
Wf_td_window_extension_coef = 2.5
Reject_outliers_psd = True
Psd_outlier_sf = 2e-2  # Reject approximately one slice
N_repetitions_outliers_psd = 4

# Ends treatment:
Alpha_windowing_fluxes = 0.1  # tukey window smoothness parameter, roughly part of the data which is affected by the window

# Notch:
Nsig_notch = 5.  # this orig_means that such a deviation will happen more rarely than once per 10 stars
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

# Templates and detection:
Template_bank_from_file = True
Templates_subbin_shift = True
Template_ind_detection = 'all'

# Veto single-event:
Margin_snr_veto = 2.
# Max_abs_stat_veto = 6.5  # something more than this will not be considered # RUN_13 VALUE IS 6.5!
Max_abs_stat_veto = 7.5  # something more than this will not be considered  # 8 is better in not killing injections....
Fn_fit_veto = 0.01
N_parts_veto_snr = 4
Min_bins_veto_part_time = 4  # Minimal size of one part in veto_parts_time
Lobe_hlen_veto_part_time = 4  # Half-size of lobe if veto_parts_time has only one part, the main lobe
N_parts_veto_fourier = 3
Wind_len_overlap_fourier = 64  # Window length of the overlapping parts into which the Fourier template will be split.
Threshold_peak_sigma_veto = 3.  # there will also be look-elsewhere

# Processing the statistic:
Nwind_moving_var_min = 1024  # trying to make the variance error few %
# ->make sure max is smaller than period
Ngap_moving_var_min = 8  # the actual gap will be according to the template width, but this is the minimal one
Nsamples_min_moving_var_coef = 0.8
Get_gaussianized_ses = True
# Measuring single-event statistics distributions:
N_bins_hist_ses = 50
Snr_0_fit_hist_ses = 2.5
Beg_min_hist_ses = 3

# Periodicity search:
P_bins_overlap = 5  # add this to the periods array for the best trigger not to be at the boundary (integral score will not be full)
Duty_cycle = 0.02  # this was 0.1, but I changed it because sometimes it is not enough
Fn_threshold_veto_mes = 0.02  # False negative for veto MES
Min_required_nonzero_transits_number = 3  # Minimal number of valid transits that is required for a trigger
Min_required_nonzero_transits_fraction = 0.3  # Fraction of transits that are required to be non-zero to pass veto_mes
Max_mes_veto_trials = None  # If None, all the possibilities will be tried
# -> Try these many triggers for the best, and if they are all bad, then stop
Std_time_scrambling = 10.  # days # for time slides
Max_hole_pieces_scrampling = 30  # <1 day # Hole which is considered a reason to split the statistic to two pieces which can be shuffled
Max_piece_scrambling = int(50 * 24 * 2)  # 50 days # Maximal possible length of a piece which will be shuffled
Min_piece_scrambling = int(5 * 24 * 2)  # 5 days # Minimal possible length of a piece which will be shuffled

# After periodicity search:
D_phase_integral_score = 50  # Bins. The size of the region taken for selecting the phase indices participating in the integral score
D_period_integral_score = 50  # Bins. The size of the region taken for selecting the period indices participating in the integral score
Minimal_trackable_score = 20  # The score which is considered interesting. If all the scores are lower, the search is stopped

# Best stat formatting:
Best_per_results_index_per = 0
Best_per_results_index_ftt = 1  # first transit time
Best_per_results_index_stats = 2
Best_per_results_index_stats_gauss = 3
Best_per_results_index_best_integral = 4  # Integral score
Best_per_results_index_template = 5
Best_per_results_index_per_list = 6
Best_per_results_length = 7

# Data types used in the search (memory concerns):
Dtype_r = np.float32
Dtype_c = np.complex64
Dtype_timings = np.float64  # Times in the periodicity summation should have a good resolution, otherwise
# the resolution of the folded periodic timings will suffer
Use_pyfftw = False  # Use pyfftw for the FFTs, otherwise use numpy
# It creates some numerical difference in SNR, which sometimes can reach 1e-3, so I preferred not to.


