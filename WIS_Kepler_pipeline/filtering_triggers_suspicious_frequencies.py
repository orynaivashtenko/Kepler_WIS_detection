# #!/usr/bin/env python3
#
# # ########################################################################################
# # This file filters the triggers which coincide with villanova binaries
# # Or frequencies of quarters and their harmonics or Earth down-links
# # ########################################################################################
# import os
#
# import trigger_class
#
# os.environ["OMP_NUM_THREADS"] = "1"  #
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"  #
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"
#
# import numpy as np
# from trigger_class import load_triggers
# from tce_parameters import (Earth_pointing_times_file, Quarter_start_times_file, Quarter_end_times_file,
#                             Villanova_catalog_file, Villanova_periods_phases_kmags_relevant_file,
#                             Table_koi_file, Suspicious_frequency_reasons_triggers_file,
#                             Initial_trigger_list_file, Suspicious_frequency_filtered_triggers_file)
#
# Trigger_list_file = Initial_trigger_list_file
#
#
#
# # ################################################################################################
# # LOADING VILLANOVA
# vil_periods, vil_phases, vil_kmags = np.load(Villanova_periods_phases_kmags_relevant_file, allow_pickle=True)
#
# # ################################################################################################
# # LOADING KNOWN PLANETS
# with open(Table_koi_file, 'r') as table_koi_file:
#     lines = table_koi_file.readlines()
# lines = [line for line in lines if line[0] != '#']
# header_list = (lines[0].strip()).split(',')
# header_list = [e.strip() for e in header_list]
# lines = lines[1:]  # remove header
# lines = [(line.strip()).split(',') for line in lines]
# lines = [line for line in lines if (len(line[18]) > 0)]  # select known snr
# lines = [line for line in lines if (float(line[18]) > 40.)]  # select big snr
#
# index_kepid = header_list.index('kepid')
# index_period = header_list.index('koi_period')
# index_ftt = header_list.index('koi_time0bk')
# periods_koi = np.array([float(line[index_period]) for line in lines])
# phases_koi = np.array([float(line[index_ftt]) for line in lines])
# kids_koi = np.array([line[index_kepid].rjust(9, '0') for line in lines])
#
# # ################################################################################################
# # LOADING THE TRIGGERS
#
# target_ids, integral_scores, stats, pers, ftts, gauss_stats, templates = load_triggers(Trigger_list_file, sep=' ')
# best_periods = np.array(pers)
# best_ftts = np.array(ftts)
#
# dict_timing_reason_names = {
#     'y': 'year',
#     'q': 'quarter',
#     'v': 'villanova',
#     'd': 'downlink',
#     'k': 'koi',
#     'yh': 'year_harmonics',
#     'qh': 'quarter_harmonics',
#     'vh': 'villanova_harmonics',
#     'dh': 'downlink_harmonics',
#     'kh': 'koi_harmonics',
#     'qn': 'inter_quarter_nans',
#     'dn': 'inter_downlink_nans',
#     '': '',
# }
#
# # ################################################################################################
# # DEFINING HOW CLOSE TO KNOWN PERIODICITIES THE SIGNALS ARE ALLOWED TO BE
# # there are some harmonics which are fast:
# # exposure time 29.424 min
# # reaction wheels 3.03 days, 0.133 days
# year_days = 370.  # 365.25 #year
# quarter_days = 93.  # quarter
# # 30., #monthly downlink is very irregular and will be considered separately
#
# error_year_days = 10.  # duty cycles as estimation for year period error
# error_year_harm_days = 2.
# error_quarter_days = 2.  # let it be for quarter
#
# per_err = 0.03  # days
# phase_err = 0.07  # around 3 lc
# hole_err = 2.  # days. contamination from hole can last. just a random estimate
# # Kepler time is measured in BJD - 2454833
# # Villanova phases are in BJD - 2400000
# # And we corrected for this
#
# quarter_coef_threshold = 0.3  # this part of transits maximally can be at the quarter edges
# downlink_coef_threshold = 0.3
#
# # ################################################################################################
# # FILTERING ALL THE BAD TIMES THAT WERE LOADED
#
# is_bad_flag = np.empty(len(target_ids), dtype=bool)
# bad_times_reason = np.empty(len(target_ids), dtype='U128')
# for i in range(len(target_ids)):
#     reason = ''
#     bad_vil = ((vil_periods > best_periods[i] - per_err) &
#                (vil_periods < best_periods[i] + per_err) &
#                (((vil_phases > best_ftts[i] - phase_err) &
#                  (vil_phases < best_ftts[i] + phase_err)) |
#                 ((vil_phases + vil_periods > best_ftts[
#                     i] - phase_err) &  # maybe one transit to the left or to the right was missing
#                  (vil_phases + vil_periods < best_ftts[i] + phase_err)) |
#                 ((vil_phases - vil_periods > best_ftts[i] - phase_err) &
#                  (vil_phases - vil_periods < best_ftts[i] + phase_err))
#                 )).any()
#
#     bad_kois = ((periods_koi > best_periods[i] - per_err) &
#                 (periods_koi < best_periods[i] + per_err) &
#                 (((phases_koi > best_ftts[i] - phase_err) &
#                   (phases_koi < best_ftts[i] + phase_err)) |
#                  ((phases_koi + periods_koi > best_ftts[
#                      i] - phase_err) &  # maybe one transit to the left or to the right was missing
#                   (phases_koi + periods_koi < best_ftts[i] + phase_err)) |
#                  ((phases_koi - periods_koi > best_ftts[i] - phase_err) &
#                   (phases_koi - periods_koi < best_ftts[i] + phase_err))
#                  ))
#     bad_koi = bad_kois.any()
#     bad_kids = ''
#     if bad_koi:
#         bad_kids = kids_koi[bad_kois]  # which exactly KOI leaked
#
#     bad_year = (np.abs(year_days - best_periods[i]) < error_year_days)  # check that period is close to year
#
#     bad_quarter = ((quarter_days > best_periods[i] - error_quarter_days) &
#                    (quarter_days < best_periods[i] + error_quarter_days))  # check that period is close to quarter
#     quarter_coef = np.array(
#         [np.array([((best_ftts[i] + ii * best_periods[i] > quarter_end_times[j] - hole_err)
#                     & (best_ftts[i] + ii * best_periods[i] < quarter_start_times[j + 1] + hole_err))
#                    for j in range(len(quarter_start_times) - 1)]).any()  # is inside at least one quarter hole
#          for ii in range(int(np.ceil(1300 / best_periods[i])))])  # for all the transits
#     quarter_coef = np.mean(quarter_coef.astype(float))
#
#     if quarter_coef > quarter_coef_threshold:
#         bad_quarter = True
#
#     bad_downlinks = np.array([float(np.nanmin(np.abs(best_ftts[i] + ii * best_periods[i] - earth_pointing_times)) < hole_err)
#                               for ii in range(int(np.ceil(1300 / best_periods[i])))])  # in mission time
#     downlink_coef = np.mean(bad_downlinks.astype(float))
#     if downlink_coef >= downlink_coef_threshold:
#         bad_downlinks = True
#     else:
#         bad_downlinks = False
#
#     if bad_vil:
#         reason = reason + 'v '
#     if bad_year:
#         reason = reason + 'y '
#     if bad_quarter:
#         reason = reason + 'qn' + str(np.round(quarter_coef, 1)) + ' '
#     if bad_downlinks:
#         reason = reason + 'dn' + str(np.round(downlink_coef, 1)) + ' '
#     if bad_koi and bad_kids != '':
#         reason = reason + 'k' + '_'.join(bad_kids) + ' '
#
#     # ################################################################################################
#     # HARMONICS
#
#     else:
#         # harmonics:
#         bad_vil = False
#         bad_year = False
#         bad_quarter = False
#         bad_harms_vil = ''
#         bad_harms_year = ''
#         bad_harms_quarter = ''
#         for harm in [1 / 4, 1 / 3, 1 / 2, 2., 3., 4., 5., 6., 7., 8.]:
#             bad_vil_1 = ((vil_periods / harm > best_periods[i] - per_err) &
#                          (vil_periods / harm < best_periods[i] + per_err) &
#                          (((vil_phases > best_ftts[i] - phase_err) &
#                            (vil_phases < best_ftts[i] + phase_err)) |
#                           ((vil_phases + vil_periods > best_ftts[
#                               i] - phase_err) &  # maybe one transit to the left or to the right was missing
#                            (vil_phases + vil_periods < best_ftts[i] + phase_err)) |
#                           ((vil_phases - vil_periods > best_ftts[i] - phase_err) &
#                            (vil_phases - vil_periods < best_ftts[i] + phase_err))
#                           )).any()
#
#             bad_vil = bad_vil | bad_vil_1
#             if bad_vil_1:
#                 bad_harms_vil = bad_harms_vil + '' + str(np.round(harm, 2))
#
#             bad_year_1 = ((year_days / harm > best_periods[
#                 i] - error_year_harm_days) &  # check that period is close to harmonics of year
#                           (year_days / harm < best_periods[i] + error_year_harm_days))
#             bad_year = bad_year | bad_year_1
#             if bad_year_1:
#                 bad_harms_year = bad_harms_year + '' + str(np.round(harm, 2))
#
#             bad_quarter_1 = ((quarter_days / harm > best_periods[
#                 i] - error_quarter_days) &  # check that period is close to harmonics of quarter
#                              (quarter_days / harm < best_periods[i] + error_quarter_days))
#
#             bad_quarter = bad_quarter | bad_quarter_1
#             if bad_quarter_1:
#                 bad_harms_quarter = bad_harms_quarter + '' + str(np.round(harm, 2))
#
#         if bad_vil:
#             reason = reason + ' vh' + str(bad_harms_vil) + ' '
#         if bad_year:
#             reason = reason + ' yh' + bad_harms_year + ' '
#         if bad_quarter:
#             # if quarter_coef >= 0.1:
#             reason = reason + ' qh' + str(bad_harms_quarter) + ' qn' + str(np.round(quarter_coef, 1)) + ' '
#
#     bad_times_reason[i] = reason
#     if reason != '':
#         is_bad_flag[i] = True
#
# triggers_with_filtering_reasons = np.vstack((
#         np.array(target_ids),
#         np.array(integral_scores),
#         np.array(stats),
#         np.array(pers),
#         np.array(ftts),
#         np.array(gauss_stats),
#         np.array(templates),
#         np.array(is_bad_flag),
#         np.array(bad_times_reason)), dtype=object)
#
# np.save(Suspicious_frequency_reasons_triggers_file, triggers_with_filtering_reasons, allow_pickle=True)
#
# is_good_indices = ~is_bad_flag
# trigger_class.print_triggers(target_ids=np.array(target_ids[is_good_indices]),
#                              integral_scores=np.array(integral_scores[is_good_indices]),
#                              stats=np.array(stats[is_good_indices]),
#                              pers=np.array(pers[is_good_indices]),
#                              ftts=np.array(ftts[is_good_indices]),
#                              gauss_stats=np.array(gauss_stats[is_good_indices]),
#                              templates=np.array(templates[is_good_indices]),
#                              file=Suspicious_frequency_filtered_triggers_file,
#                              sep=' ', end='\n')