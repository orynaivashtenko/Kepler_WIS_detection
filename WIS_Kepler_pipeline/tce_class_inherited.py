# #!/usr/bin/env python3
#
# import copy
# import os
# core_num = "1"
# os.environ["OMP_NUM_THREADS"] = core_num
# os.environ["OPENBLAS_NUM_THREADS"] = core_num
# os.environ["MKL_NUM_THREADS"] = core_num
# os.environ["VECLIB_MAXIMUM_THREADS"] = core_num
# os.environ["NUMEXPR_NUM_THREADS"] = core_num
# # import numpy as np
# import pickle
# from sys import getsizeof
#
# import target_class
# from pipeline_parameters import Limb_darkening_array_file, Problems_folder, R_sun_R_e
# from tce_parameters import *
# import auxiliary_functions as af
# import search_pipeline as spip
# import utils as ut
# import template_bank.template_module as tm
# from scipy.stats import chi2
# # from astropy.table import Table
#
#
# class WisTCE(target_class.KeplerTarget):
#     def __init__(self, target_id='003097222',
#                  data_location=pp.Data_location, output_location=pp.Output_location,
#                  target_pixels_location=Target_pixels_location,
#                  load_pickled_attributes=False,
#                  unpickle_target_attributes=True,
#                  get_transit_and_binary_info=True,
#                  gaia_kepler_file_path=Gaia_kepler_file,
#                  koi_dict_file=Koi_dict_file,
#                  ttv_dict_file=Ttv_dict_file,
#                  villanova_dict_file=Villanova_dict_file,
#                  limb_darkening_array_file=Limb_darkening_array_file,
#                  overwrite_location_parameters_by_unpickle=False,
#                  allow_output_folder_creation=True,
#                  allow_data_folder_creation=True,
#                  problems_folder=Problems_folder,
#                  add_prefix_to_problems_folder=True,
#                  clean_problems_file=False,
#                  overwrite_parameters_by_unpickle=False):
#
#         super().__init__(target_id=target_id, data_location=data_location, output_location=output_location,
#                          load_pickled_attributes=False,
#                          get_transit_and_binary_info=get_transit_and_binary_info,
#                          gaia_kepler_file_path=gaia_kepler_file_path,
#                          koi_dict_file=koi_dict_file,
#                          ttv_dict_file=ttv_dict_file,
#                          villanova_dict_file=villanova_dict_file,
#                          limb_darkening_array_file=limb_darkening_array_file,
#                          overwrite_location_parameters_by_unpickle=overwrite_location_parameters_by_unpickle,
#                          allow_output_folder_creation=allow_output_folder_creation,
#                          allow_data_folder_creation=allow_data_folder_creation,
#                          problems_folder=problems_folder,
#                          add_prefix_to_problems_folder=add_prefix_to_problems_folder,
#                          clean_problems_file=clean_problems_file)
#
#         data_location = af.get_correct_tid_folder(target_id, data_location, allow_creation=allow_data_folder_creation)
#         output_location = af.get_correct_tid_folder(target_id, output_location, allow_creation=allow_output_folder_creation)
#         target_pixels_location = af.get_correct_tid_folder(target_id, target_pixels_location, allow_creation=allow_data_folder_creation)
#
#         self.target_pixels_location = target_pixels_location
#         self.star_pixels_ppms = None
#         self.star_pixels_stds_of_ppms = None
#         self.bg_pixels_ppms = None
#         self.bg_pixels_stds_of_ppms = None
#         self.p_value_mean_bg_pixels = None
#         self.p_value_worst_bg_pixels = None
#         self.is_bad_bg_pixels = None
#         self.klc_star_pixels = None
#         self.klc_bg_pixels = None
#         self.star_valid_transit_mask = None
#         self.bg_valid_transit_mask = None
#         self.centroid_snrs_ra_dec = None
#         self.centroid_snr_2 = None
#         self.is_centroid_bad = None
#         self.klc_centroid_x = None
#         self.klc_centroid_y = None
#
#         self.badness_of_transits = None
#         self.valid_transit_mask = None
#         self.n_valid_transits = None
#         self.n_transits = None
#         self.too_few_valid_transits = None
#         self.is_bad_periodicity = None
#         self.bad_periodicity_reasons = None
#         self.p_value_fold_residuals = None
#         self.p_value_mf_fold_residuals = None
#         self.is_bad_fold_residuals = None
#         self.best_temp_folded = None
#         self.template_and_stat_of_fold_contaminant = None
#         self.is_good_pre_filtering = None
#
#         self.mean_transit_amplitude = None
#         self.mean_transit_ppm = None
#         self.mean_transit_ppm_std = None
#         self.transit_amplitudes = None
#         self.transit_amplitude_stds = None
#         self.transit_normalized_amplitudes = None
#         self.transit_normalized_amplitude_stds = None
#         self.transit_ppms = None
#         self.transit_ppm_stds = None
#         self.mean_two_noises_model_ppm_std = None
#         self.rp_to_rs_std_from_mean_ppm = None
#         self.rp_to_rs_std_conservative = None
#
#         if load_pickled_attributes:
#             self.unpickle_small_attributes(unpickle_target_attributes=unpickle_target_attributes)
#             if not overwrite_parameters_by_unpickle:
#                 self.data_location = data_location
#                 self.output_location = output_location
#                 self.gaia_kepler_file = gaia_kepler_file_path
#                 self.koi_dict_file = koi_dict_file
#                 self.ttv_dict_file = ttv_dict_file
#                 self.villanova_dict_file = villanova_dict_file
#                 self.target_pixels_location = target_pixels_location
#
#     def get_transit_times_info(self, period=None, first_transit_time=None, template=None, from_real_klc=True, max_time=Quarter_16_end,
#                                data_location=None, output_location=None):
#         if period is None:
#             period = self.best_per
#         if first_transit_time is None:
#             first_transit_time = self.best_ftt
#         if template is None:
#             template = self.best_temp
#         if data_location is None:
#             data_location = self.data_location
#         if output_location is None:
#             output_location = self.output_location
#         if from_real_klc:
#             klc = spip.get_fold_quickly(target_id=self.target_id, data_location=data_location,
#                                         output_location=output_location, period=period,
#                                         overlap_period_bins=0)
#             klc.get_summed_values(first_transit_time=first_transit_time, period=period, template=template)
#             self.summed_transit_times = klc.summed_times
#             self.summed_raw_stats = klc.summed_stats_raw
#             self.summed_vars = klc.summed_vars
#             self.summed_corrections = klc.summed_corrections
#             self.summed_gauss_stats = klc.summed_stats_raw_gauss
#             self.mean_flux = klc.mean_total
#             self.valid_transit_mask = (~np.isnan(self.summed_raw_stats)) & (~np.isnan(self.summed_vars)) & (self.summed_vars > 0)
#         else:
#             self.summed_transit_times = np.array([first_transit_time + i * period
#                                                   for i in range(int(np.ceil((max_time - first_transit_time) / period)))])
#
#     def pickle_small_attributes(self, output_location=None,
#                                 pickle_small_attributes_filename_postfix=Pickle_tce_small_attributes_filename_postfix,
#                                 pickle_target_attributes=False,
#                                 pickle_target_small_attributes_filename_postfix=pp.Pickle_target_small_attributes_filename_postfix):
#         """
#         Pickles all attributes which are of type str, float, int, np.float, np.integer, np.ndarray, list, dict
#         TODO: we should avoid pickling too long arrays
#         :param pickle_target_attributes: Whether to perform a separate pickling for the target_class object from which this is inherited.
#                                          Not recommended because the target properties are not supposed to be modified.
#         :type pickle_target_attributes:
#         :param pickle_target_small_attributes_filename_postfix: name of the file for the target_class pickle
#         :type pickle_target_small_attributes_filename_postfix:
#         :param pickle_small_attributes_filename_postfix:
#         :type pickle_small_attributes_filename_postfix:
#         :param output_location:
#         :type output_location
#         :return:
#         """
#         if pickle_target_attributes:
#             super().pickle_small_attributes(output_location=output_location,
#                                             pickle_small_attributes_filename_postfix=pickle_target_small_attributes_filename_postfix)
#         if output_location is None:
#             output_location = self.output_location
#         dict_to_pickle = {}
#         for key in self.__dict__:
#             value = self.__dict__[key]
#             if isinstance(value, (str, float, int, np.float32, np.float64, np.complex64, np.complex128,
#                                   np.integer, np.ndarray, list, dict)):
#                 if (((not hasattr(value, '__len__')) or len(value) < 1000) and getsizeof(value) < 80128) or isinstance(value, str):
#                     # size in bytes of array 100*100
#                     dict_to_pickle[key] = value
#
#         with open(os.path.join(output_location, self.target_id + '_' + pickle_small_attributes_filename_postfix), 'wb') as pickle_file:
#             pickle.dump(dict_to_pickle, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
#
#     def unpickle_small_attributes(self, output_location=None,
#                                   pickle_tce_small_attributes_filename_postfix=Pickle_tce_small_attributes_filename_postfix,
#                                   unpickle_target_attributes=False,
#                                   pickle_target_small_attributes_filename_postfix=pp.Pickle_target_small_attributes_filename_postfix,
#                                   overwrite_location_parameters=False):
#         if output_location is None:
#             output_location = self.output_location
#
#         old_output_location = self.output_location
#         old_data_location = self.data_location
#         old_target_pixels_location = self.target_pixels_location
#         old_gaia_kepler_file_path = self.gaia_kepler_file
#         old_koi_dict_file = self.koi_dict_file
#         old_ttv_dict_file = self.ttv_dict_file
#         old_villanova_dict_file = self.villanova_dict_file
#         old_limb_darkening_array_file = self.limb_darkening_array_file
#
#         if unpickle_target_attributes:
#             super().unpickle_small_attributes(output_location=output_location,
#                                               pickle_small_attributes_filename_postfix=pickle_target_small_attributes_filename_postfix,
#                                               overwrite_location_parameters=overwrite_location_parameters)
#
#         file_path = af.get_saving_file_path(output_location=output_location, target_id=self.target_id,
#                                             filename_postfix=pickle_tce_small_attributes_filename_postfix, file_format='pickle')
#         if os.path.exists(file_path):
#             with open(file_path, 'rb') as pickle_file:
#                 dict1 = pickle.load(pickle_file)
#                 self.__dict__.update(dict1.items())
#
#         if not overwrite_location_parameters:
#             self.data_location = old_data_location
#             self.output_location = old_output_location
#             self.target_pixels_location = old_target_pixels_location
#             self.gaia_kepler_file = old_gaia_kepler_file_path
#             self.koi_dict_file = old_koi_dict_file
#             self.ttv_dict_file = old_ttv_dict_file
#             self.villanova_dict_file = old_villanova_dict_file
#             self.limb_darkening_array_file = old_limb_darkening_array_file
#
#     def pickle_all_attributes(self, output_location=None,
#                               pickle_all_attributes_filename_postfix=Pickle_tce_all_attributes_filename_postfix):
#         """
#         Pickles all attributes which are of type str, float, int, np.float, np.integer, np.ndarray, list, dict
#         TODO: we should avoid pickling too long arrays
#
#         :param pickle_all_attributes_filename_postfix:
#         :type pickle_all_attributes_filename_postfix:
#         :param output_location:
#         :type output_location
#         :return:
#         """
#         if output_location is None:
#             output_location = self.output_location
#         dict_to_pickle = self.__dict__
#
#         with open(os.path.join(output_location, self.target_id + '_' + pickle_all_attributes_filename_postfix), 'wb') as pickle_file:
#             pickle.dump(dict_to_pickle, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
#
#     def unpickle_all_attributes(self, output_location=None,
#                                 pickle_all_attributes_filename_postfix=Pickle_tce_all_attributes_filename_postfix):
#         if output_location is None:
#             output_location = self.output_location
#         with open(os.path.join(output_location, self.target_id + '_' + pickle_all_attributes_filename_postfix), 'rb') as pickle_file:
#             dict1 = pickle.load(pickle_file)
#             self.__dict__.update(dict1.items())
#
#     def print_dict_file(self, output_location=None, dict_txt_filename_postfix=Dict_tce_txt_filename_postfix):
#         """
#
#         :param output_location:
#         :type output_location:
#         :param dict_txt_filename_postfix:
#         :type dict_txt_filename_postfix:
#         :return:
#         :rtype:
#         """
#         fields_not_to_print = ['gaia_kepler_file',
#                                'koi_dict_file',
#                                'ttv_dict_file',
#                                'villanova_dict_file',
#                                'limb_darkening_array_file']
#
#         # numpy.savetxt()
#         if output_location is None:
#             output_location = self.output_location
#         dict_to_print = {}
#         for key in self.__dict__:
#             if key not in fields_not_to_print:
#                 value = self.__dict__[key]
#                 if isinstance(value, (str, float, int, np.float32, np.float64, np.complex64, np.complex128,
#                                       np.integer, np.ndarray, list, dict)) or value is None:
#                     ok_to_print = True
#                     if isinstance(value, (str, float, int, np.float32, np.float64, np.complex64, np.complex128,
#                                           np.integer, np.ndarray, list, dict)):
#                         if hasattr(value, '__len__'):
#                             if isinstance(value, dict) and len(list(value.keys())) > 0 and hasattr(value[list(value.keys())[0]], '__len__'):
#                                 ok_to_print = False  # TODO: maybe make the selection process look at all the dict keys
#                             elif ((not isinstance(value, dict) and not (value is None))
#                                   and (len(value) > 0 and (not isinstance(value[0], str)) and hasattr(value[0], '__len__'))):
#                                 ok_to_print = False  # not printing nested lists
#                             elif (not isinstance(value, str)) and len(value) > 50:
#                                 # roughly to be able to set summed values for planets with period 30
#                                 ok_to_print = False
#                     if ok_to_print:
#                         dict_to_print[key] = value
#         with open(os.path.join(output_location, self.target_id + '_' + dict_txt_filename_postfix), 'w') as fil:
#             for key in dict_to_print:
#                 print(key, ":", dict_to_print[key], file=fil)
#
#     def get_trigger_star_bg_ppms(self, target_pixels_folder=None,
#                                  number_of_pixels_cut=None, part_of_cumsum_2=Part_of_cumsum_2_targ_pix,
#                                  part_of_light=None,
#                                  period=None, template_index=None, first_transit_time=None,
#                                  nsig_outliers_0_neg=Nsig_outliers_0_neg_tests, nsig_outliers_0_pos=Nsig_outliers_0_pos_tests,
#                                  save_klcs=False):
#         if target_pixels_folder is None:
#             target_pixels_folder = self.target_pixels_location
#         if period is None:
#             period = self.best_per
#         if template_index is None:
#             template_index = self.best_temp
#         if first_transit_time is None:
#             first_transit_time = self.best_ftt
#
#         target_pixels_folder = af.get_correct_tid_folder(target_id=self.target_id, location=target_pixels_folder)
#         pixel_times, target_pixel_fluxes = af.get_target_pixel_data_fun(self.target_id, data_folder=target_pixels_folder,
#                                                                         remove_zeroth_quarter=True,
#                                                                         remove_quarter_17=True)
#         star_part_sap_fluxes, bg_part_sap_fluxes = af.get_splitted_saps_from_pixels(target_pixel_fluxes,
#                                                                                     number_of_pixels_cut=number_of_pixels_cut,
#                                                                                     part_of_cumsum_2=part_of_cumsum_2,
#                                                                                     part_of_light=part_of_light)
#         ppms_and_stds_star_and_bg = []
#         klcs = []
#         for fluxes in [star_part_sap_fluxes, bg_part_sap_fluxes]:
#             # making the lightcurve out of this whatever data and doing search:
#             klc = spip.get_minimal_processed_folded_klc(target_id=None, get_data=False, times=pixel_times, fluxes=copy.deepcopy(fluxes),
#                                                         period=period, template_index=template_index, cut_edge_nans=False,
#                                                         nsig_outliers_0_neg=nsig_outliers_0_neg,
#                                                         nsig_outliers_0_pos=nsig_outliers_0_pos,
#                                                         data_location=self.data_location, output_location=self.output_location)
#             # -> we do not marginalize, so statistics are still not in units of snr squared
#
#             # determine the depth in the needed position:
#             ti = 0  # we have only one template initialized, and we
#             ti_sh = klc.indices_of_templates[ti][0]
#             if klc.indices_of_templates[ti][-1] - klc.indices_of_templates[ti][0] > 1:
#                 is_interleaved = True
#                 phases = klc._phases_interleaved
#                 time = klc.time_interleaved
#                 statistics_periodic = klc.statistics_periodic_interleaved
#                 statistics = klc.statistics_interleaved
#                 stat_vars = klc.stat_vars_interleaved
#             else:
#                 is_interleaved = False
#                 phases = klc._phases_not_interleaved
#                 time = klc.time_not_interleaved
#                 statistics_periodic = klc.statistics_periodic_not_interleaved
#                 statistics = klc.statistics_not_interleaved
#                 stat_vars = klc.stat_vars_not_interleaved
#
#             # Finding  correlation size:
#             period_index_left, period_index_right = klc.get_influence_period_indices(period=period, part_snr_2=0.5)
#             phase_index_left, phase_index_right = klc.get_influence_phase_indices(first_transit_time=first_transit_time,
#                                                                                   in_interleaved=is_interleaved, part_snr_2=0.5)
#
#             # Allow the actual minimum to be a little bit shifted: ## TODO is it okay that it is independent in star and bg?
#             # redefining the best period index with a small freedom:
#             period_index = period_index_left + np.argmin(np.nanmin(statistics_periodic[period_index_left:period_index_right, 0,
#                                                                    phase_index_left:phase_index_right], axis=-1))
#             # redefining the best phase index with a small freedom:
#             phase_index = phase_index_left + np.argmin(statistics_periodic[period_index][0][
#                                                        phase_index_left:phase_index_right])
#             # -> allow the actual minimum to be a little bit shifted
#             period_new = klc.periods[period_index]
#             first_transit_time_new = phases[phase_index] + klc.tmin
#             expected_transit_times = [first_transit_time_new + i * period_new  # I am not sure tmin does not miss some nan in the beginning
#                                       for i in range(int(np.ceil((klc.tmax - first_transit_time_new) / period_new)))]
#             summed_transit_points = [np.nanargmin(np.abs(time - expected_transit_times[i])) for i in range(len(expected_transit_times))]
#             # summed_transit_times = np.array(time[summed_transit_points])
#             summed_raw_stats = statistics[ti][summed_transit_points]
#             summed_vars = stat_vars[ti][summed_transit_points]
#
#             (ampl, ppm, ampls, ampls_stds, normalized_ampls, normalized_ampls_stds,
#              ppms, ppms_stds) = af.estimate_ampls_ppms_from_stats(summed_raw_stats=summed_raw_stats, summed_stat_vars=summed_vars,
#                                                                   templ_max=np.max(np.abs(klc.tb.templates_shifts[ti_sh])),
#                                                                   flux_trend=klc.mean_total, trends_std=None)
#             ppms_and_stds_star_and_bg.append([ppms, ppms_stds])
#             if save_klcs:
#                 klcs.append(copy.deepcopy(klc))
#
#         self.star_pixels_ppms = ppms_and_stds_star_and_bg[0][0]
#         self.star_pixels_stds_of_ppms = ppms_and_stds_star_and_bg[0][1]
#         self.bg_pixels_ppms = ppms_and_stds_star_and_bg[1][0]
#         self.bg_pixels_stds_of_ppms = ppms_and_stds_star_and_bg[1][1]
#         if save_klcs:
#             self.klc_star_pixels = klcs[0]
#             self.klc_bg_pixels = klcs[1]
#
#     def test_target_pixels_amplitudes(self,
#                                       threshold_false_negative_bg_target_pixels_mean=Threshold_false_negative_bg_target_pixels_mean,
#                                       threshold_false_negative_bg_target_pixels_max=Threshold_false_negative_bg_target_pixels_max,
#                                       part_of_cumsum_2=Part_of_cumsum_2_targ_pix):
#         if (self.star_pixels_ppms is None or self.star_pixels_stds_of_ppms is None
#                 or self.bg_pixels_ppms is None or self.bg_pixels_stds_of_ppms is None):
#             self.get_trigger_star_bg_ppms(
#                 target_pixels_folder=self.target_pixels_location,
#                 number_of_pixels_cut=None, part_of_cumsum_2=part_of_cumsum_2, part_of_light=None)
#
#         # First, see what happened in the transits because they can happen to be nans if the transit in one of them is too deep:
#         self.star_valid_transit_mask = (~np.isnan(self.star_pixels_ppms)) & (~np.isnan(self.star_pixels_stds_of_ppms)
#                                                                              & (self.star_pixels_stds_of_ppms > 0))
#         self.bg_valid_transit_mask = (~np.isnan(self.bg_pixels_ppms)) & (~np.isnan(self.bg_pixels_stds_of_ppms)
#                                                                          & (self.bg_pixels_stds_of_ppms > 0))
#         valid_star_bg_transit_mask = self.star_valid_transit_mask & self.bg_valid_transit_mask
#         if len(np.where(valid_star_bg_transit_mask)[0]) < 3:
#             self.is_bad_bg_pixels = True
#         else:
#             # If everything is fine, then look at the ppm difference of the star and the background pixels:
#             diff_ppms = self.star_pixels_ppms - self.bg_pixels_ppms
#             var_diff_ppms = self.star_pixels_stds_of_ppms ** 2 + self.bg_pixels_stds_of_ppms ** 2
#             valid_mask = (~np.isnan(diff_ppms)) & (~np.isnan(var_diff_ppms) & ~np.isclose(var_diff_ppms, 0))
#             test_stat = np.nansum(diff_ppms[valid_mask] / var_diff_ppms[valid_mask]) / np.nansum(1 / var_diff_ppms[valid_mask])**0.5
#
#             false_neg = ut.get_p_value_gaussian(test_stat, 0, 1)
#             self.p_value_mean_bg_pixels = false_neg
#             if test_stat > 0:
#                 if false_neg < threshold_false_negative_bg_target_pixels_mean:
#                     self.is_bad_bg_pixels = True
#                 else:
#                     self.is_bad_bg_pixels = False
#             else:
#                 self.is_bad_bg_pixels = False
#
#             # Checking the worst deviation:
#             if (diff_ppms[valid_mask] > 0).any():
#                 inds = (diff_ppms > 0) & valid_mask
#                 diffs_single_ppm = diff_ppms[inds]
#                 stds_single_ppm = var_diff_ppms[inds]**0.5
#                 false_neg_singles = [ut.get_p_value_gaussian(diffs_single_ppm[i] / stds_single_ppm[i], 0, 1.)
#                                      for i in range(len(diffs_single_ppm))]
#                 worst_false_neg_single = np.nanmin(false_neg_singles)
#                 self.p_value_worst_bg_pixels = worst_false_neg_single
#
#                 # look-elsewhere:
#                 # 1 - (1 - p1)**n = p:
#                 # 1 - p1 = (1 - p)**(1/n)
#                 false_neg_1 = 1 - (1 - threshold_false_negative_bg_target_pixels_max)**(1 / len(np.where(valid_mask)[0]))
#                 if worst_false_neg_single < false_neg_1:
#                     self.is_bad_bg_pixels = True
#
#     def test_centroid_transits_snr(self, threshold_snr_2=Threshold_centroid_snr_2):
#         if not hasattr(self, 'self.centroid_snrs_ra_dec') or self.centroid_snrs_ra_dec is None:
#             self.find_centroid_transits_snr(period=self.best_per, ftt=self.best_ftt,
#                                             nsig_outliers_0_pos=Nsig_outliers_0_pos_tests,
#                                             nsig_outliers_0_neg=Nsig_outliers_0_neg_tests)
#         self.centroid_snr_2 = self.centroid_snrs_ra_dec[0]**2 + self.centroid_snrs_ra_dec[1]**2
#         # self.centroid_snr_2 = self.centroid_snrs_ra_dec[0]**2 * np.cos(self.dec)**2 + self.centroid_snrs_ra_dec[1]**2
#         # -> we did not use the spherical basis geometry to make the length element. The reason is that we sum the
#         # snr, which will already include its impact. Say we are on the equator, then the snr is the same for both
#         # directions. If we are on the pole, then the snr is less for ra because the noise is larger. So the effect
#         # of that cosine is taken into account by the snr of ra.
#         # this is statistically not really correct, but we at least know what are the snrs we are talking about.
#         # we expect the centroid to deviate even if the planet is real. if it deviates in the incorrect direction wrt
#         # the neighboring stars, then it is bad.
#         if self.centroid_snr_2 > threshold_snr_2:
#             self.is_centroid_bad = True
#         else:
#             self.is_centroid_bad = False
#
#     # noinspection PyProtectedMember
#     def find_centroid_transits_snr(self, period=None, ftt=None, nsig_outliers_0_pos=Nsig_outliers_0_pos_tests,
#                                    nsig_outliers_0_neg=Nsig_outliers_0_neg_tests,
#                                    save_klcs=False, scaling_factor=Scaling_factor_centroid_flux):
#         if period is None:
#             period = self.best_per
#         if ftt is None:
#             ftt = self.best_ftt
#         # times, target_pixel_fluxes = af.get_target_pixel_data_fun(self.target_id, data_folder=self.target_pixels_location)
#         times, target_pixel_fluxes, sky_coords, time_masks, aperture_masks = af.get_target_pixel_sky_coords_fun(
#             target_id=self.target_id, data_location=self.target_pixels_location,
#             remove_zeroth_quarter=True, remove_quarter_17=True, cut_edge_nans=True)
#         centroid_x_quarters, centroid_y_quarters = af.get_centroid_series(target_pixel_fluxes=target_pixel_fluxes,
#                                                                           sky_coords=sky_coords,
#                                                                           time_masks=time_masks, aperture_masks=aperture_masks)
#         # Adding the scaling factor to avoid the numerical precision issues:
#         centroid_x_quarters = [centroid_x_quarters[q] * scaling_factor for q in range(len(centroid_x_quarters))]
#         centroid_y_quarters = [centroid_y_quarters[q] * scaling_factor for q in range(len(centroid_y_quarters))]
#         max_coords_deviation_periodic = []
#         klcs = []
#         for fluxes in [centroid_x_quarters, centroid_y_quarters]:
#             # making the lightcurve out of this whatever data and doing search:
#             klc = spip.get_minimal_processed_folded_klc(target_id=None, get_data=False, times=times, fluxes=fluxes,
#                                                         period=period, template_index='all',
#                                                         nsig_outliers_0_neg=nsig_outliers_0_neg,
#                                                         nsig_outliers_0_pos=nsig_outliers_0_pos,
#                                                         data_location=self.data_location, output_location=self.output_location)
#
#             # Finding  correlation size:
#             per_ind_left, per_ind_right = klc.get_influence_period_indices(period=period, part_snr_2=0.5)
#
#             max_deviation = []
#             for in_interl, phases, stats_periodic in [[True, klc._phases_interleaved, klc.statistics_periodic_interleaved],
#                                                       [False, klc._phases_not_interleaved, klc.statistics_periodic_not_interleaved]]:
#                 phase_ind_left, phase_ind_right = klc.get_influence_phase_indices(first_transit_time=ftt, in_interleaved=in_interl,
#                                                                                   part_snr_2=0.5)
#                 max_pos_dev = np.nanmax(stats_periodic[per_ind_left:per_ind_right,
#                                         :, phase_ind_left:phase_ind_right])
#                 max_neg_dev = np.nanmin(stats_periodic[per_ind_left:per_ind_right,
#                                         :, phase_ind_left:phase_ind_right])
#                 if np.abs(max_pos_dev) > np.abs(max_neg_dev):
#                     max_deviation.append(max_pos_dev)
#                 else:
#                     max_deviation.append(max_neg_dev)
#
#             if np.abs(max_deviation[0]) > np.abs(max_deviation[1]):
#                 max_coords_deviation_periodic.append(max_deviation[0])
#             else:
#                 max_coords_deviation_periodic.append(max_deviation[1])
#             if save_klcs:
#                 klcs.append(copy.deepcopy(klc))
#         if save_klcs:
#             self.klc_centroid_x = klcs[0]
#             self.klc_centroid_y = klcs[1]
#         self.centroid_snrs_ra_dec = max_coords_deviation_periodic
#
#     def check_transit_times_quality_flags(self):
#         """
#         Opens the fits data and accesses the quality flag series. looks in the vicinity of every summed transit point
#         and records all that was there.
#         :return:
#         :rtype:
#         """
#         self.get_transit_times_info(from_real_klc=True, data_location=self.data_location, output_location=self.output_location)
#         # times, _, _, kepler_quality_masks, _ = af.get_target_pixel_sky_coords_fun(
#         #     target_id=self.target_id, data_location=self.target_pixels_location,
#         #     remove_zeroth_quarter=True, remove_quarter_17=True, cut_edge_nans=True)
#         times, kepler_quality_masks = af.get_kepler_quality_masks(
#             target_id=self.target_id, data_location=self.target_pixels_location,
#             remove_zeroth_quarter=True, remove_quarter_17=True, cut_edge_nans=True)
#         # times = res[0]
#         # kepler_quality_masks = res[3]
#
#         concatenated_time = np.concatenate(times)
#         concatenated_kepler_quality_mask = np.concatenate(kepler_quality_masks).astype(bool)
#
#         earth_pointing_times = get_earth_pointing_times()
#         quarter_start_times, quarter_end_times = get_quarter_start_end_times()
#
#         badness_of_transits = ['' for _ in range(len(self.summed_transit_times))]
#         valid_transit_mask = np.ones(len(self.summed_transit_times), dtype=bool)
#         valid_transit_mask[np.isnan(self.summed_raw_stats) | np.isnan(self.summed_vars) | np.isclose(self.summed_vars, 0)] = False
#         for i in range(len(valid_transit_mask)):
#             if not valid_transit_mask[i]:
#                 badness_of_transits[i] += 'nan '
#
#         for i, transit_time in enumerate(self.summed_transit_times):
#             between_quarters = np.array([((transit_time > quarter_end_times[j] - Timing_error_quarter_edge)
#                                           & (transit_time < quarter_start_times[j + 1] + Timing_error_quarter_edge))
#                                          for j in range(len(quarter_start_times) - 1)]).any()
#             if between_quarters:
#                 badness_of_transits[i] += 'between_quarters '
#                 valid_transit_mask[i] = False
#
#             index_in_times = np.searchsorted(concatenated_time, transit_time)
#             if ((not concatenated_kepler_quality_mask[index_in_times])
#                     or (not concatenated_kepler_quality_mask[max(0, index_in_times - 1)])):
#                 badness_of_transits[i] += 'kepler_quality '
#                 valid_transit_mask[i] = False
#
#             if np.min(np.abs(earth_pointing_times - transit_time)) < Timing_error_earth_downlink:
#                 badness_of_transits[i] += 'downlink '
#                 valid_transit_mask[i] = False
#
#         self.badness_of_transits = badness_of_transits
#         self.valid_transit_mask = valid_transit_mask
#         self.n_valid_transits = len(np.where(valid_transit_mask)[0])
#         self.n_transits = len(self.summed_raw_stats)
#         if self.n_valid_transits < 3 or self.n_valid_transits / self.n_transits < Min_fraction_valid_transits:
#             self.too_few_valid_transits = True
#         else:
#             self.too_few_valid_transits = False
#
#     # noinspection DuplicatedCode
#     def check_periodicity_contaminants(self, period_error=Period_error, phase_error=Phase_error,
#                                        period_year=Period_year, period_quarter=Period_quarter,
#                                        period_error_year=Period_error_year,
#                                        period_error_quarter=Period_error_quarter,
#                                        period_error_year_harm=Period_error_year_harm,
#                                        period_error_quarter_harm=Period_error_quarter_harm):
#         # Loading known Kepler periodicities of potentially bright contaminants:
#         vil_periods, vil_phases, vil_kmags = get_villanova_relevant_periods_phases_kmags()
#         periods_koi, ftts_koi, kid_kois = get_strong_koi_periodicities()
#         vil_periods = np.array(vil_periods, dtype=float)
#         vil_phases = np.array(vil_phases, dtype=float)
#         periods_koi = np.array(periods_koi, dtype=float)
#         ftts_koi = np.array(ftts_koi, dtype=float)
#
#         # ################################################################################################
#         # FILTERING ALL THE BAD PERIODICITIES
#         self.bad_periodicity_reasons = ''
#
#         bad_vil = ((vil_periods > self.best_per - period_error) &
#                    (vil_periods < self.best_per + period_error) &
#                    (((vil_phases > self.best_ftt - phase_error) &
#                      (vil_phases < self.best_ftt + phase_error)) |
#                     ((vil_phases + vil_periods > self.best_ftt - phase_error) &  # maybe one transit to the left or to the right was missing
#                      (vil_phases + vil_periods < self.best_ftt + phase_error)) |
#                     ((vil_phases - vil_periods > self.best_ftt - phase_error) &
#                      (vil_phases - vil_periods < self.best_ftt + phase_error))
#                     ))
#         if bad_vil.any():
#             self.bad_periodicity_reasons += 'v '
#             self.is_bad_periodicity = True
#
#         bad_kois = ((periods_koi > self.best_per - period_error) &
#                     (periods_koi < self.best_per + period_error) &
#                     (((ftts_koi > self.best_ftt - phase_error) &
#                       (ftts_koi < self.best_ftt + phase_error)) |
#                      ((ftts_koi + periods_koi > self.best_ftt - phase_error) &  # maybe one transit to the left or to the right was missing
#                       (ftts_koi + periods_koi < self.best_ftt + phase_error)) |
#                      ((ftts_koi - periods_koi > self.best_ftt - phase_error) &
#                       (ftts_koi - periods_koi < self.best_ftt + phase_error))
#                      ))
#         if bad_kois.any():
#             self.bad_periodicity_reasons += 'k '
#             self.is_bad_periodicity = True
#
#         bad_year = (np.abs(period_year - self.best_per) < period_error_year)  # check that period is close to year
#         if bad_year:
#             self.bad_periodicity_reasons += 'y '
#             self.is_bad_periodicity = True
#
#         bad_quarter = ((period_quarter > self.best_per - period_error_quarter) &
#                        (period_quarter < self.best_per + period_error_quarter))  # check that period is close to quarter
#         if bad_quarter:
#             self.bad_periodicity_reasons += 'q '
#             self.is_bad_periodicity = True
#
#         # ################################################################################################
#         # HARMONICS
#         if self.is_bad_periodicity is None:
#             bad_vil = False
#             bad_year = False
#             bad_quarter = False
#             bad_harms_vil = ''
#             bad_harms_year = ''
#             bad_harms_quarter = ''
#             for harm in [1 / 4, 1 / 3, 1 / 2, 4/3, 2., 3., 4., 5., 6., 7., 8.]:  # warning, 4/3 was not there before
#                 bad_vil_1 = ((vil_periods / harm > self.best_per - period_error) &
#                              (vil_periods / harm < self.best_per + period_error) &
#                              (((vil_phases > self.best_ftt - phase_error) &
#                                (vil_phases < self.best_ftt + phase_error)) |
#                               ((vil_phases + vil_periods > self.best_ftt - phase_error) &  # maybe one transit was missing
#                                (vil_phases + vil_periods < self.best_ftt + phase_error)) |
#                               ((vil_phases - vil_periods > self.best_ftt - phase_error) &  # maybe one transit was missing
#                                (vil_phases - vil_periods < self.best_ftt + phase_error))
#                               )).any()
#
#                 bad_vil = bad_vil | bad_vil_1
#                 if bad_vil_1:
#                     bad_harms_vil = bad_harms_vil + '' + str(np.round(harm, 2))
#                 # Check that period is close to harmonics of year
#                 bad_year_1 = ((period_year / harm > self.best_per - period_error_year_harm) &
#                               (period_year / harm < self.best_per + period_error_year_harm))
#                 bad_year = bad_year | bad_year_1
#                 if bad_year_1:
#                     bad_harms_year = bad_harms_year + '' + str(np.round(harm, 2))
#                 # Check that period is close to harmonics of quarter
#                 bad_quarter_1 = ((period_quarter / harm > self.best_per - period_error_quarter_harm) &
#                                  (period_quarter / harm < self.best_per + period_error_quarter_harm))
#
#                 bad_quarter = bad_quarter | bad_quarter_1
#                 if bad_quarter_1:
#                     bad_harms_quarter = bad_harms_quarter + '' + str(np.round(harm, 2))
#
#             if bad_vil:
#                 self.bad_periodicity_reasons += ' vh' + str(bad_harms_vil) + ' '
#                 self.is_bad_periodicity = True
#             if bad_year:
#                 self.bad_periodicity_reasons += ' yh' + str(bad_harms_year) + ' '
#                 self.is_bad_periodicity = True
#             if bad_quarter:
#                 self.bad_periodicity_reasons += ' qh' + str(bad_harms_quarter)
#                 # This one we do not want to make the periodicity bad, but we want to still know if it is there
#
#         if self.is_bad_periodicity is None:
#             self.is_bad_periodicity = False
#
#     def check_harmonic_coincidence(self, period_check, ftt_check, periods_reference, phases_reference,
#                                    period_error=Period_error, phase_error=Phase_error,
#                                    harmonics_list=(1., 1./4, 1./3, 1./2, 4./3, 2., 3., 4., 5., 6., 7., 8.),
#                                    harmonics_nums=
#                                    (1, 1, 1, 1, 4, 2, 3, 4, 5, 6, 7, 8),
#                                    harmonics_denoms=
#                                    (1, 4, 3, 2, 3, 1, 1, 1, 1, 1, 1, 1)):
#
#         # min_per =
#         #
#         #
#         # periods_reference = np.array(periods_reference)
#         # phases_reference = np.array(phases_reference)
#         # ###
#         # bad_vil = ((vil_periods > self.best_per - period_error) &
#         #            (vil_periods < self.best_per + period_error) &
#         #            (((vil_phases > self.best_ftt - phase_error) &
#         #              (vil_phases < self.best_ftt + phase_error)) |
#         #             ((vil_phases + vil_periods > self.best_ftt - phase_error
#         #             ) &  # maybe one transit to the left or to the right was missing
#         #              (vil_phases + vil_periods < self.best_ftt + phase_error)) |
#         #             ((vil_phases - vil_periods > self.best_ftt - phase_error) &
#         #              (vil_phases - vil_periods < self.best_ftt + phase_error))
#         #             ))
#         # ###
#
#         bad = False
#         bad_harms = ''
#         for harm in harmonics_list:  # warning, 4/3 was not there before
#
#             bad_1 = ((periods_reference / harm > period_check - period_error) &
#                      (periods_reference / harm < period_check + period_error) &
#                      (((phases_reference > ftt_check - phase_error) &
#                        (phases_reference < ftt_check + phase_error)) |
#                       ((phases_reference + periods_reference > ftt_check - phase_error) &  # maybe one transit was missing
#                        (phases_reference + periods_reference < ftt_check + phase_error)) |
#                       ((phases_reference - periods_reference > ftt_check - phase_error) &  # maybe one transit was missing
#                        (phases_reference - periods_reference < ftt_check + phase_error))
#                       )).any()
#
#             bad = bad | bad_1
#             if bad_1:
#                 bad_harms = bad_harms + ' ' + str(np.round(harm, 2))
#
#     def estimate_biased_rp(self):
#         if self.summed_raw_stats is None or self.summed_vars is None or self.mean_flux is None:
#             self.get_transit_times_info()
#         # ATTENTION! It uses the raw statistics after the SES veto, so some transits may be missed, and the estimate may be biased.
#         # if self.best_ppms_no_filt is None or self.best_ppm_no_filt is None or self.best_ppms_stds_no_filt is None:
#         #     self.get_summed_values_no_filters(nsig_outliers_0_pos=nsig_outliers_0_pos,
#         #                                       nsig_outliers_0_neg=nsig_outliers_0_neg)
#         if len(np.where(self.valid_transit_mask)[0]) == 0:
#             self.mean_transit_amplitude = np.nan
#             self.mean_transit_ppm = np.nan
#             self.mean_transit_ppm_std = np.nan
#             self.transit_amplitudes = np.nan
#             self.transit_amplitude_stds = np.nan
#             self.transit_normalized_amplitudes = np.nan
#             self.transit_normalized_amplitude_stds = np.nan
#             self.transit_ppms = np.nan
#             self.transit_ppm_stds = np.nan
#
#             self.rp_to_rs = np.nan
#             self.rp_to_rs_std = np.nan
#             self.rp_to_r_earth = np.nan
#             self.rp_to_r_earth_std = np.nan
#         else:
#             tb = tm.TemplateBank(templates=None, shifts=None, template_index='all', load_essentials=True)
#             ti_sh = tb.indices_of_templates[self.best_temp][0]
#             templ_max = np.max(np.abs(tb.templates_shifts[ti_sh]))
#             (self.mean_transit_amplitude, self.mean_transit_ppm, self.transit_amplitudes, self.transit_amplitude_stds,
#              self.transit_normalized_amplitudes, self.transit_normalized_amplitude_stds, self.transit_ppms,
#              self.transit_ppm_stds) = af.estimate_ampls_ppms_from_stats(summed_raw_stats=self.summed_raw_stats,
#                                                                         summed_stat_vars=self.summed_vars, templ_max=templ_max,
#                                                                         flux_trend=self.mean_flux, trends_std=None,
#                                                                         valid_transit_mask=self.valid_transit_mask)
#             self.mean_transit_ppm_std = (np.nansum(self.transit_ppm_stds[self.valid_transit_mask] ** 2
#                                                    ) / len(self.transit_ppm_stds[self.valid_transit_mask]) ** 2) ** 0.5
#             self.mean_two_noises_model_ppm_std = ut.get_std_of_mean_two_sigmas_model(self.transit_ppm_stds[self.valid_transit_mask])
#
#         ppm_factor = 1e3
#         self.rp_to_rs = - np.sign(self.mean_transit_ppm) * (np.abs(self.mean_transit_ppm)) ** 0.5 / ppm_factor
#         # -> Radius estimation is subjected to big error and bias! The template contains depth factor at least,
#         # and the depth factor can be different even if the same template triggered. And of course, all the
#         # templates have different depth factors for their parameters.
#         n_samp = len(self.transit_ppm_stds[self.valid_transit_mask])
#         sum_ppm = np.sum(self.transit_ppm_stds[self.valid_transit_mask])
#         sum_var_ppm = np.sum(self.transit_ppm_stds[self.valid_transit_mask] ** 2)
#         self.rp_to_rs_std = (1 / (4 * n_samp) * 1 / sum_ppm * sum_var_ppm) ** 0.5 / ppm_factor
#         self.rp_to_rs_std_from_mean_ppm = self.mean_transit_ppm_std / (2 * np.abs(self.mean_transit_ppm)**0.5) / ppm_factor
#         self.rp_to_rs_std_conservative = self.mean_two_noises_model_ppm_std / (2 * np.abs(self.mean_transit_ppm) ** 0.5) / ppm_factor
#         # self.rp_to_rs_std = 1 / (2 * self.rp_to_rs) * (np.sum(self.transit_ppm_stds[self.valid_transit_mask] ** 2)
#         #                                                / len(self.transit_ppm_stds[self.valid_transit_mask])) ** 0.5 / 1e3 ???
#         self.rp_to_r_earth = self.rp_to_rs * self.stellar_radius * R_sun_R_e
#         self.rp_to_r_earth_std = self.rp_to_rs_std * self.stellar_radius * R_sun_R_e
#
#     def veto_fold_residuals_quick(self, fn_fold_veto=Fn_fold_veto):
#         self.is_bad_fold_residuals = False
#         self.initialize_kepler_light_curve()
#         self.klc.load_conditioned_data()
#         indices_of_templates = tm.Indices_of_templates
#         whitened_templates_td = copy.deepcopy(self.klc.whitened_templates_td)
#         dt_days = self.klc.dt_days
#         period_fold = self.best_per
#         dt_steps_high_pass = self.klc.nperseg_psd
#
#         # ------------------------------------------------------------------------------------------------------------
#         # FOLDING WHITENED FLUX AND BINNING
#         binwid = self.klc.dt_days  # min(self.templates_durations_hours[temp_number] / 24 / 40, 0.1)
#         bins_time, bins_flux, bins_counts, folds_times, folds_fluxes = af.fold_fluxes(
#             times=self.klc.times, fluxes=self.klc.whitened_fluxes_td, period=period_fold, first_time=self.best_ftt,
#             masks=self.klc.big_hole_masks, binwid=binwid, time_fold=period_fold,
#             dt=self.klc.dt_days, means=None, template=None, return_std=False,
#             normalize_by_counts=False)  # np.array(self.klc.templates[self.klc.indices_of_templates[temp_number][0]]))
#         flux_normalized_vars = np.zeros(len(bins_flux))
#         flux_normalized_vars[bins_counts > 0] = bins_flux[bins_counts > 0] / bins_counts[bins_counts > 0] ** 0.5
#         # -> this will have the same variance for all the bins
#
#         # ------------------------------------------------------------------------------------------------------------
#         # GETTING THE BEST TEMPLATE INDEX
#         # The best template index in the fold may not be the best template index from the search due to different binning,
#         # corrections, veto etc. So we have to find the best template index in the folded data.
#         n_time_templ = int(np.round(period_fold / dt_days))
#
#         averaged_templates, time_templates = np.array([af.get_average_whitened_template(
#             whitened_templates_td=whitened_templates_td, temp_number=tn, n_time_templ=n_time_templ,
#             dt_days=dt_days, indices_of_templates=indices_of_templates)
#             for tn in range(len(indices_of_templates))]).transpose(1, 0, 2)
#         # Interpolating the best averaged template
#         averaged_templates = np.array([np.interp(bins_time, time_templates[i], averaged_templates[i])
#                                  for i in range(len(averaged_templates))])
#         averaged_templates = np.array([- temp_interp / np.linalg.norm(temp_interp) for temp_interp in averaged_templates])
#         del time_templates
#
#         mf_scores, ampl_scores = np.array([af.mf_high_pass(data=flux_normalized_vars, template=temp_interp, template_rolled=None,
#                                                            dt=binwid, dt_pass=dt_steps_high_pass * dt_days,
#                                                            renorm_psd_drift=True)
#                                            for temp_interp in averaged_templates])
#
#         rolling_args = np.argmax(mf_scores, axis=-1)
#         rolling_vals_peaks = np.array([rolling_arg if rolling_arg <= len(mf_scores[i]) / 2 else len(mf_scores[i]) - rolling_arg
#                                        for i, rolling_arg in enumerate(rolling_args)])
#         mf_scores_peaks = np.max(mf_scores, axis=-1)
#
#         max_dist_from_detected_peak_bins = 5  # how far away the pipeline peak can be from the folding peak for the best template
#         if rolling_vals_peaks[np.argmax(mf_scores_peaks)] > max_dist_from_detected_peak_bins:
#             self.is_bad_fold_residuals = True  # There is some largest peak in the wrong place
#             self.template_and_stat_of_fold_contaminant = (np.argmax(mf_scores_peaks), np.max(mf_scores_peaks))
#         self.best_temp_folded = np.argmax(mf_scores_peaks * (rolling_vals_peaks <= max_dist_from_detected_peak_bins).astype(float))
#         temp_number = self.best_temp_folded
#         rolling_arg = rolling_args[temp_number]
#         amp_best_fit = ampl_scores[temp_number][rolling_arg]
#         # ------------------------------------------------------------------------------------------------------------
#         # CALCULATING THE MATCHED FILTER SCORE ON THE BINNED WHITENED FLUX
#         # Matched-filtering of the folded binned flux with the templates corresponding to their expected values.
#         # Takes the original `whitened_templates_td` that were used fot the search, in their original time resolution,
#         # averages them over sub-bin shifts, and interpolates them (not integrates, which is in general wrong!)
#         # to the binned time `bins_time`. The binned template is normalized by the `bins_counts`, as well as the
#         # binned flux `bins_flux`, so that the expectations are consistent, and the variances are uniform.
#         # Then, the binned flux `bins_flux` is matched-filtered with the binned and the normalized templates
#         # The folded template is proportional to the counts, but the matched filtering should be done
#         # on the whitened template, and the standard deviation is proportional to the square root of counts.
#         # Important: template multiplied by counts is not shift-invariant because counts are different
#         # in every bin. Therefore, convolution cannot be used.
#
#         # GETTING THE RESIDUALS
#         residuals_flux_binned = flux_normalized_vars - amp_best_fit * np.roll(averaged_templates[temp_number], rolling_arg)
#         # Renormalizing to account for blueness:
#         residuals_flux_binned = af.renorm_noise_high_pass(data=residuals_flux_binned, dt=binwid, dt_pass=dt_steps_high_pass * dt_days)
#
#         # ------------------------------------------------------------------------------------------------------------
#         # MATCHED_FILTERING OF THE RESIDUALS TO FIND SUSPICIOUS DEVIATIONS
#         # THE TEMPLATES DO NOT HAVE TO BE OUR TEMPLATE BANK, THEY CAN BE WHATEVER
#
#         mf_scores_resid = np.array([af.mf_high_pass(data=residuals_flux_binned, template=temp_interp, template_rolled=None,
#                                                     dt=binwid, dt_pass=dt_steps_high_pass * dt_days,
#                                                     renorm_psd_drift=True)[0]
#                                     for temp_interp in averaged_templates]).transpose(1, 0, 2)
#         # ------------------------------------------------------------------------------------------------------------
#         # Calculating p-values:
#         p_val_resid = 1 - (1 - chi2(df=1, scale=1).sf(np.max((residuals_flux_binned / np.std(residuals_flux_binned)) ** 2))
#                            ) ** len(residuals_flux_binned)
#         p_val_mf = 1. - (1 - chi2(df=1, scale=1).sf(np.max(mf_scores_resid ** 2))) ** len(mf_scores_resid.flatten())
#         self.p_value_fold_residuals = p_val_resid
#         self.p_value_mf_fold_residuals = p_val_mf
#         if self.p_value_mf_fold_residuals < fn_fold_veto:  # or self.p_value_fold_residuals < fn_fold_veto:
#             self.is_bad_fold_residuals = True
#
#     def veto_fold_residuals(self, fn_fold_veto=Fn_fold_veto):
#         self.is_bad_fold_residuals = False
#         self.initialize_kepler_light_curve()
#         self.klc.load_conditioned_data()
#         indices_of_templates = tm.Indices_of_templates
#         whitened_templates_td = copy.deepcopy(self.klc.whitened_templates_td)
#         dt_days = self.klc.dt_days
#         period_fold = self.best_per
#         dt_steps_high_pass = self.klc.nperseg_psd
#
#         # ------------------------------------------------------------------------------------------------------------
#         # FOLDING WHITENED FLUX AND BINNING
#         binwid = self.klc.dt_days  # min(self.templates_durations_hours[temp_number] / 24 / 40, 0.1)
#         bins_time, bins_flux, bins_counts, folds_times, folds_fluxes = af.fold_fluxes(
#             times=self.klc.times, fluxes=self.klc.whitened_fluxes_td, period=period_fold, first_time=self.best_ftt,
#             masks=self.klc.big_hole_masks, binwid=binwid, time_fold=period_fold,
#             dt=self.klc.dt_days, means=None, template=None, return_std=False,
#             normalize_by_counts=False)  # np.array(self.klc.templates[self.klc.indices_of_templates[temp_number][0]]))
#         flux_normalized_vars = np.zeros(len(bins_flux))
#         flux_normalized_vars[bins_counts > 0] = bins_flux[bins_counts > 0] / bins_counts[bins_counts > 0] ** 0.5
#         # -> this will have the same variance for all the bins
#
#         # ------------------------------------------------------------------------------------------------------------
#         # GETTING THE BEST TEMPLATE INDEX
#         # The best template index in the fold may not be the best template index from the search due to different binning,
#         # corrections, veto etc. So we have to find the best template index in the folded data.
#         n_time_templ = int(np.round(period_fold / dt_days))
#         res = [af.matched_filter_folded_fluxes(
#             template_number=template_number, whitened_templates_td=whitened_templates_td, n_time_templ=n_time_templ, binwid=binwid,
#             indices_of_templates=indices_of_templates, bins_time=bins_time, bins_counts=bins_counts,
#             flux_normalized_vars=flux_normalized_vars, dt_steps_high_pass=dt_steps_high_pass, dt_days=dt_days, renorm_psd_drift=True)
#             for template_number in range(len(indices_of_templates))]  # ).transpose(1, 0, 2)
#
#         mf_scores_peaks = np.array([r[0] for r in res])
#         ampl_scores_peaks = np.array([r[1] for r in res])
#         rolling_args = np.array([r[2] for r in res])
#         folded_templates_rolled = np.array([r[3] for r in res])
#
#         rolling_vals_peaks = np.array([rolling_arg if rolling_arg <= len(flux_normalized_vars) / 2
#                                        else len(flux_normalized_vars) - rolling_arg for rolling_arg in rolling_args])
#
#         max_dist_from_detected_peak_bins = 5  # how far away the pipeline peak can be from the folding peak for the best template
#         if rolling_vals_peaks[np.argmax(mf_scores_peaks)] > max_dist_from_detected_peak_bins:
#             self.is_bad_fold_residuals = True  # There is some largest peak in the wrong place
#             self.template_and_stat_of_fold_contaminant = (np.argmax(mf_scores_peaks), np.max(mf_scores_peaks))
#         self.best_temp_folded = np.argmax(mf_scores_peaks * (rolling_vals_peaks <= max_dist_from_detected_peak_bins).astype(float))
#         temp_number = self.best_temp_folded
#         amp_best_fit = ampl_scores_peaks[temp_number]
#         # ------------------------------------------------------------------------------------------------------------
#         # CALCULATING THE MATCHED FILTER SCORE ON THE BINNED WHITENED FLUX
#         # Matched-filtering of the folded binned flux with the templates corresponding to their expected values.
#         # Takes the original `whitened_templates_td` that were used fot the search, in their original time resolution,
#         # averages them over sub-bin shifts, and interpolates them (not integrates, which is in general wrong!)
#         # to the binned time `bins_time`. The binned template is normalized by the `bins_counts`, as well as the
#         # binned flux `bins_flux`, so that the expectations are consistent, and the variances are uniform.
#         # Then, the binned flux `bins_flux` is matched-filtered with the binned and the normalized templates
#         # The folded template is proportional to the counts, but the matched filtering should be done
#         # on the whitened template, and the standard deviation is proportional to the square root of counts.
#         # Important: template multiplied by counts is not shift-invariant because counts are different
#         # in every bin. Therefore, convolution cannot be used.
#
#         # GETTING THE RESIDUALS
#         residuals_flux_binned = flux_normalized_vars - amp_best_fit * folded_templates_rolled[temp_number]
#         # Renormalizing to account for blueness:
#         residuals_flux_binned = af.renorm_noise_high_pass(data=residuals_flux_binned, dt=binwid, dt_pass=dt_steps_high_pass * dt_days)
#
#         # ------------------------------------------------------------------------------------------------------------
#         # MATCHED_FILTERING OF THE RESIDUALS TO FIND SUSPICIOUS DEVIATIONS
#         # THE TEMPLATES DO NOT HAVE TO BE OUR TEMPLATE BANK, THEY CAN BE WHATEVER
#
#         mf_scores_resid = np.array([af.matched_filter_folded_fluxes(
#             template_number=template_number, whitened_templates_td=whitened_templates_td, n_time_templ=n_time_templ, binwid=binwid,
#             indices_of_templates=indices_of_templates, bins_time=bins_time, bins_counts=bins_counts,
#             flux_normalized_vars=residuals_flux_binned, dt_steps_high_pass=dt_steps_high_pass,
#             dt_days=dt_days, renorm_psd_drift=True)[0]
#             for template_number in range(len(indices_of_templates))])
#
#         # ------------------------------------------------------------------------------------------------------------
#         # Calculating p-values:
#         p_val_resid = 1 - (1 - chi2(df=1, scale=1).sf(np.max((residuals_flux_binned / np.std(residuals_flux_binned)) ** 2))
#                            ) ** len(residuals_flux_binned)
#         p_val_mf = 1. - (1 - chi2(df=1, scale=1).sf(np.max(mf_scores_resid ** 2))) ** len(mf_scores_resid.flatten())
#         self.p_value_fold_residuals = p_val_resid
#         self.p_value_mf_fold_residuals = p_val_mf
#         if self.p_value_mf_fold_residuals < fn_fold_veto:  # or self.p_value_fold_residuals < fn_fold_veto:
#             self.is_bad_fold_residuals = True
#
#     def run_all_checks_exclusive(self):
#         self.get_transit_times_info()
#         self.check_transit_times_quality_flags()
#         self.check_periodicity_contaminants()
#         if (not self.too_few_valid_transits) and (not self.is_bad_periodicity):
#             self.test_target_pixels_amplitudes()
#             if not self.is_bad_bg_pixels:
#                 self.test_centroid_transits_snr()
#                 if not self.is_centroid_bad:
#                     self.veto_fold_residuals()
#                     if not self.is_bad_fold_residuals:
#                         self.estimate_biased_rp()
#                         self.is_good_pre_filtering = True
#         if self.is_good_pre_filtering is not True:
#             self.is_good_pre_filtering = False
#
