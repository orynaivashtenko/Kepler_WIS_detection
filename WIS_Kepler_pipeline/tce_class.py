import copy
import os
core_num = "1"
os.environ["OMP_NUM_THREADS"] = core_num
os.environ["OPENBLAS_NUM_THREADS"] = core_num
os.environ["MKL_NUM_THREADS"] = core_num
os.environ["VECLIB_MAXIMUM_THREADS"] = core_num
os.environ["NUMEXPR_NUM_THREADS"] = core_num
import numpy as np
import pickle
from sys import getsizeof

import target_class
from pipeline_parameters import Limb_darkening_array_file, Problems_folder
from tce_parameters import *
import auxiliary_functions as af
import search_pipeline as spip
import utils as ut


class KeplerTCE(target_class.KeplerTarget):
    def __init__(self, target_id='003097222',
                 data_location=pp.Data_location, output_location=pp.Output_location,
                 target_pixels_location=Target_pixels_location,
                 load_pickled_attributes=False,
                 get_transit_and_binary_info=True,
                 gaia_kepler_file_path=Gaia_kepler_file,
                 koi_dict_file=Koi_dict_file,
                 ttv_dict_file=Ttv_dict_file,
                 villanova_dict_file=Villanova_dict_file,
                 limb_darkening_array_file=Limb_darkening_array_file,
                 overwrite_location_parameters_by_unpickle=False,
                 allow_output_folder_creation=True,
                 allow_data_folder_creation=True,
                 problems_folder=Problems_folder,
                 add_prefix_to_problems_folder=True,
                 clean_problems_file=False,
                 overwrite_parameters_by_unpickle=False):

        super().__init__(target_id=target_id, data_location=data_location, output_location=output_location,
                         load_pickled_attributes=load_pickled_attributes,
                         get_transit_and_binary_info=get_transit_and_binary_info,
                         gaia_kepler_file_path=gaia_kepler_file_path,
                         koi_dict_file=koi_dict_file,
                         ttv_dict_file=ttv_dict_file,
                         villanova_dict_file=villanova_dict_file,
                         limb_darkening_array_file=limb_darkening_array_file,
                         overwrite_location_parameters_by_unpickle=overwrite_location_parameters_by_unpickle,
                         allow_output_folder_creation=allow_output_folder_creation,
                         allow_data_folder_creation=allow_data_folder_creation,
                         problems_folder=problems_folder,
                         add_prefix_to_problems_folder=add_prefix_to_problems_folder,
                         clean_problems_file=clean_problems_file)

        data_location = af.get_correct_tid_folder(target_id, data_location, allow_creation=allow_data_folder_creation)
        output_location = af.get_correct_tid_folder(target_id, output_location, allow_creation=allow_output_folder_creation)
        target_pixels_location = af.get_correct_tid_folder(target_id, target_pixels_location, allow_creation=allow_data_folder_creation)

        self.target_pixels_location = target_pixels_location
        self.star_pixels_ppms = None
        self.star_pixels_stds_of_ppms = None
        self.bg_pixels_ppms = None
        self.bg_pixels_stds_of_ppms = None
        self.p_value_mean_bg_pixels = None
        self.p_value_worst_bg_pixels = None
        self.is_bad_bg_pixels = None
        self.klc_star_pixels = None
        self.klc_bg_pixels = None
        self.centroid_snrs_ra_dec = None
        self.target_pixels_stats = None
        self.target_pixels_means = None
        self.target_pixels_ppms = None
        self.target_pixels_stds_of_ppms = None
        self.centroid_snr_2 = None
        self.is_centroid_bad = None

        if load_pickled_attributes:
            self.unpickle_small_attributes()
            if not overwrite_parameters_by_unpickle:
                self.data_location = data_location
                self.output_location = output_location
                self.gaia_kepler_file = gaia_kepler_file_path
                self.koi_dict_file = koi_dict_file
                self.ttv_dict_file = ttv_dict_file
                self.villanova_dict_file = villanova_dict_file
                self.target_pixels_location = target_pixels_location

    def pickle_small_attributes(self, output_location=None,
                                pickle_small_attributes_filename_postfix=Pickle_tce_small_attributes_filename_postfix,
                                pickle_target_attributes=False,
                                pickle_target_small_attributes_filename_postfix=pp.Pickle_target_small_attributes_filename_postfix):
        """
        Pickles all attributes which are of type str, float, int, np.float, np.integer, np.ndarray, list, dict
        TODO: we should avoid pickling too long arrays
        :param pickle_target_attributes: Whether to perform a separate pickling for the target_class object from which this is inherited.
                                         Not recommended because the target properties are not supposed to be modified.
        :type pickle_target_attributes:
        :param pickle_target_small_attributes_filename_postfix: name of the file for the target_class pickle
        :type pickle_target_small_attributes_filename_postfix:
        :param pickle_small_attributes_filename_postfix:
        :type pickle_small_attributes_filename_postfix:
        :param output_location:
        :type output_location
        :return:
        """
        if pickle_target_attributes:
            super().pickle_small_attributes(output_location=output_location,
                                            pickle_small_attributes_filename_postfix=pickle_target_small_attributes_filename_postfix)
        if output_location is None:
            output_location = self.output_location
        dict_to_pickle = {}
        for key in self.__dict__:
            value = self.__dict__[key]
            if isinstance(value, (str, float, int, np.float32, np.float64, np.complex64, np.complex128,
                                  np.integer, np.ndarray, list, dict)):
                if (((not hasattr(value, '__len__')) or len(value) < 1000) and getsizeof(value) < 80128) or isinstance(value, str):
                    # size in bytes of array 100*100
                    dict_to_pickle[key] = value

        with open(os.path.join(output_location, self.target_id + '_' + pickle_small_attributes_filename_postfix), 'wb') as pickle_file:
            pickle.dump(dict_to_pickle, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

    def unpickle_small_attributes(self, output_location=None,
                                  pickle_tce_small_attributes_filename_postfix=Pickle_tce_small_attributes_filename_postfix,
                                  unpickle_target_attributes=False,
                                  pickle_target_small_attributes_filename_postfix=pp.Pickle_target_small_attributes_filename_postfix,
                                  overwrite_location_parameters=False):
        if output_location is None:
            output_location = self.output_location

        old_output_location = self.output_location
        old_data_location = self.data_location
        old_target_pixels_location = self.target_pixels_location
        old_gaia_kepler_file_path = self.gaia_kepler_file
        old_koi_dict_file = self.koi_dict_file
        old_ttv_dict_file = self.ttv_dict_file
        old_villanova_dict_file = self.villanova_dict_file
        old_limb_darkening_array_file = self.limb_darkening_array_file

        if unpickle_target_attributes:
            super().unpickle_small_attributes(output_location=output_location,
                                              pickle_small_attributes_filename_postfix=pickle_target_small_attributes_filename_postfix,
                                              overwrite_location_parameters=overwrite_location_parameters)

        file_path = af.get_saving_file_path(output_location=output_location, target_id=self.target_id,
                                            filename_postfix=pickle_tce_small_attributes_filename_postfix, file_format='pickle')
        if os.path.exists(file_path):
            with open(file_path, 'rb') as pickle_file:
                dict1 = pickle.load(pickle_file)
                self.__dict__.update(dict1.items())

        if not overwrite_location_parameters:
            self.data_location = old_data_location
            self.output_location = old_output_location
            self.target_pixels_location = old_target_pixels_location
            self.gaia_kepler_file = old_gaia_kepler_file_path
            self.koi_dict_file = old_koi_dict_file
            self.ttv_dict_file = old_ttv_dict_file
            self.villanova_dict_file = old_villanova_dict_file
            self.limb_darkening_array_file = old_limb_darkening_array_file

    def pickle_all_attributes(self, output_location=None,
                              pickle_all_attributes_filename_postfix=Pickle_tce_all_attributes_filename_postfix):
        """
        Pickles all attributes which are of type str, float, int, np.float, np.integer, np.ndarray, list, dict
        TODO: we should avoid pickling too long arrays

        :param pickle_all_attributes_filename_postfix:
        :type pickle_all_attributes_filename_postfix:
        :param output_location:
        :type output_location
        :return:
        """
        if output_location is None:
            output_location = self.output_location
        dict_to_pickle = self.__dict__

        with open(os.path.join(output_location, self.target_id + '_' + pickle_all_attributes_filename_postfix), 'wb') as pickle_file:
            pickle.dump(dict_to_pickle, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

    def unpickle_all_attributes(self, output_location=None,
                                pickle_all_attributes_filename_postfix=Pickle_tce_all_attributes_filename_postfix):
        if output_location is None:
            output_location = self.output_location
        with open(os.path.join(output_location, self.target_id + '_' + pickle_all_attributes_filename_postfix), 'rb') as pickle_file:
            dict1 = pickle.load(pickle_file)
            self.__dict__.update(dict1.items())

    def print_dict_file(self, output_location=None, dict_txt_filename_postfix=Dict_tce_txt_filename_postfix):
        """

        :param output_location:
        :type output_location:
        :param dict_txt_filename_postfix:
        :type dict_txt_filename_postfix:
        :return:
        :rtype:
        """
        # numpy.savetxt()
        if output_location is None:
            output_location = self.output_location
        dict_to_print = {}
        for key in self.__dict__:
            value = self.__dict__[key]
            if isinstance(value, (str, float, int, np.float32, np.float64, np.complex64, np.complex128,
                                  np.integer, np.ndarray, list, dict)):
                ok_to_print = True
                if isinstance(value, (str, float, int, np.float32, np.float64, np.complex64, np.complex128,
                                      np.integer, np.ndarray, list, dict)):
                    if hasattr(value, '__len__'):
                        if isinstance(value, dict) and hasattr(value[list(value.keys())[0]], '__len__'):
                            ok_to_print = False  # TODO: maybe make the selection process look at all the dict keys
                        elif (not isinstance(value, dict)) and (len(value) > 0 and hasattr(value[0], '__len__')):
                            ok_to_print = False  # not printing nested lists
                        elif (not isinstance(value, str)) and len(value) > 50:
                            # roughly to be able to set summed values for planets with period 30
                            ok_to_print = False
                if ok_to_print:
                    dict_to_print[key] = value
        with open(os.path.join(output_location, self.target_id + '_' + dict_txt_filename_postfix), 'w') as fil:
            for key in dict_to_print:
                print(key, ":", dict_to_print[key], file=fil)

    # def get_trigger_star_bg_ppms(self, target_pixels_folder=None,
    #                              number_of_pixels_cut=None, part_of_cumsum_2=Part_of_cumsum_2_targ_pix,
    #                              part_of_light=None,
    #                              period=None, template_index=None, first_transit_time=None,
    #                              nsig_outliers_0_neg=Nsig_outliers_0_neg_tests, nsig_outliers_0_pos=Nsig_outliers_0_pos_tests):
    #     if target_pixels_folder is None:
    #         target_pixels_folder = self.target_pixels_location
    #     if period is None:
    #         period = self.best_per
    #     if template_index is None:
    #         template_index = self.best_temp
    #     if first_transit_time is None:
    #         first_transit_time = self.best_ftt
    #     star_part_ppms, star_part_stds_of_ppms, bg_part_ppms, bg_part_stds_of_ppms = af.get_trigger_star_bg_ppms(
    #                                        target_id=self.target_id, first_transit_time=first_transit_time, period=period,
    #                                        template_index=template_index, target_pixels_folder=target_pixels_folder,
    #                                        number_of_pixels_cut=number_of_pixels_cut, part_of_cumsum_2=part_of_cumsum_2,
    #                                        part_of_light=part_of_light,
    #                                        nsig_outliers_0_neg=nsig_outliers_0_neg, nsig_outliers_0_pos=nsig_outliers_0_pos)
    #     self.star_pixels_ppms = star_part_ppms
    #     self.star_pixels_stds_of_ppms = star_part_stds_of_ppms
    #     self.bg_pixels_ppms = bg_part_ppms
    #     self.bg_pixels_stds_of_ppms = bg_part_stds_of_ppms

    def get_trigger_star_bg_ppms(self, target_pixels_folder=None,
                                 number_of_pixels_cut=None, part_of_cumsum_2=Part_of_cumsum_2_targ_pix,
                                 part_of_light=None,
                                 period=None, template_index=None, first_transit_time=None,
                                 nsig_outliers_0_neg=Nsig_outliers_0_neg_tests, nsig_outliers_0_pos=Nsig_outliers_0_pos_tests,
                                 save_klcs=False):
        if target_pixels_folder is None:
            target_pixels_folder = self.target_pixels_location
        if period is None:
            period = self.best_per
        if template_index is None:
            template_index = self.best_temp
        if first_transit_time is None:
            first_transit_time = self.best_ftt

        target_pixels_folder = af.get_correct_tid_folder(target_id=self.target_id, location=target_pixels_folder)
        pixel_times, target_pixel_fluxes = af.get_target_pixel_data_fun(self.target_id, data_folder=target_pixels_folder,
                                                                        remove_zeroth_quarter=True,
                                                                        remove_quarter_17=True)
        star_part_sap_fluxes, bg_part_sap_fluxes = af.get_splitted_saps_from_pixels(target_pixel_fluxes,
                                                                                    number_of_pixels_cut=number_of_pixels_cut,
                                                                                    part_of_cumsum_2=part_of_cumsum_2,
                                                                                    part_of_light=part_of_light)
        ppms_and_stds_star_and_bg = []
        if save_klcs:
            klcs = []
        for fluxes in [star_part_sap_fluxes, bg_part_sap_fluxes]:
            # making the lightcurve out of this whatever data and doing search:
            klc = spip.get_minimal_processed_folded_klc(target_id=None, get_data=False, times=pixel_times, fluxes=copy.deepcopy(fluxes),
                                                        period=period, template_index=template_index,
                                                        nsig_outliers_0_neg=nsig_outliers_0_neg,
                                                        nsig_outliers_0_pos=nsig_outliers_0_pos)
            # -> we do not marginalize, so statistics are still not in units of snr squared

            # determine the depth in the needed position:
            ti = 0  # we have only one template initialized, and we
            ti_sh = klc.indices_of_templates[ti][0]
            if klc.indices_of_templates[ti][-1] - klc.indices_of_templates[ti][0] > 1:
                is_interleaved = True
                phases = klc._phases_interleaved
                time = klc.time_interleaved
                statistics_periodic = klc.statistics_periodic_interleaved
                statistics = klc.statistics_interleaved
                stat_vars = klc.stat_vars_interleaved
            else:
                is_interleaved = False
                phases = klc._phases_not_interleaved
                time = klc.time_not_interleaved
                statistics_periodic = klc.statistics_not_interleaved
                statistics = klc.statistics_not_interleaved
                stat_vars = klc.stat_vars_not_interleaved

            # Finding  correlation size:
            period_index_left, period_index_right = klc.get_influence_period_indices(period=period, part_snr_2=0.5)
            phase_index_left, phase_index_right = klc.get_influence_phase_indices(first_transit_time=first_transit_time,
                                                                                  in_interleaved=is_interleaved, part_snr_2=0.5)

            # Allow the actual minimum to be a little bit shifted:
            # redefining the best period index with a small freedom:
            period_index = period_index_left + np.argmin(np.nanmin(statistics_periodic[period_index_left:period_index_right, 0,
                                                                   phase_index_left:phase_index_right], axis=-1))
            # redefining the best phase index with a small freedom:
            phase_index = phase_index_left + np.argmin(statistics_periodic[period_index][0][
                                                       phase_index_left:phase_index_right])
            # -> allow the actual minimum to be a little bit shifted
            period_new = klc.periods[period_index]
            first_transit_time_new = phases[phase_index] + klc.tmin
            expected_transit_times = [first_transit_time_new + i * period_new  # I am not sure tmin does not miss some nan in the beginning
                                      for i in range(int(np.ceil((klc.tmax - first_transit_time_new) / period_new)))]
            summed_transit_points = [np.nanargmin(np.abs(time - expected_transit_times[i])) for i in range(len(expected_transit_times))]
            # summed_transit_times = np.array(time[summed_transit_points])
            summed_raw_stats = statistics[ti][summed_transit_points]
            summed_vars = stat_vars[ti][summed_transit_points]

            (ampl, ppm, ampls, ampls_stds, normalized_ampls, normalized_ampls_stds,
             ppms, ppms_stds) = af.estimate_ampls_ppms_from_stats(summed_raw_stats=summed_raw_stats, summed_stat_vars=summed_vars,
                                                                  templ_max=np.max(np.abs(klc.tb.templates_shifts[ti_sh])),
                                                                  flux_trend=klc.mean_total, trends_std=None)
            ppms_and_stds_star_and_bg.append([ppms, ppms_stds])
            if save_klcs:
                klcs.append(copy.deepcopy(klc))

        #
        # ############################################################
        #
        # star_part_ppms, star_part_stds_of_ppms, bg_part_ppms, bg_part_stds_of_ppms = af.get_trigger_star_bg_ppms(
        #                                    target_id=self.target_id, first_transit_time=first_transit_time, period=period,
        #                                    template_index=template_index, target_pixels_folder=target_pixels_folder,
        #                                    number_of_pixels_cut=number_of_pixels_cut, part_of_cumsum_2=part_of_cumsum_2,
        #                                    part_of_light=part_of_light,
        #                                    nsig_outliers_0_neg=nsig_outliers_0_neg, nsig_outliers_0_pos=nsig_outliers_0_pos)
        self.star_pixels_ppms = ppms_and_stds_star_and_bg[0][0]
        self.star_pixels_stds_of_ppms = ppms_and_stds_star_and_bg[0][1]
        self.bg_pixels_ppms = ppms_and_stds_star_and_bg[1][0]
        self.bg_pixels_stds_of_ppms = ppms_and_stds_star_and_bg[1][1]
        if save_klcs:
            self.klc_star_pixels = klcs[0]
            self.klc_bg_pixels = klcs[1]

    def test_target_pixels_amplitudes(self,
                                      threshold_false_negative_bg_target_pixels_mean=Threshold_false_negative_bg_target_pixels_mean,
                                      threshold_false_negative_bg_target_pixels_max=Threshold_false_negative_bg_target_pixels_max,
                                      part_of_cumsum_2=Part_of_cumsum_2_targ_pix):
        if (self.star_pixels_ppms is None or self.star_pixels_stds_of_ppms is None
                or self.bg_pixels_ppms is None or self.bg_pixels_stds_of_ppms is None):
            self.get_trigger_star_bg_ppms(
                target_pixels_folder=self.target_pixels_location,
                number_of_pixels_cut=None, part_of_cumsum_2=part_of_cumsum_2, part_of_light=None)

        # diff_mean_ppm = np.nanmean(self.star_pixels_ppms) - np.nanmean(self.bg_pixels_ppms)
        # var_mean_ppm = (np.nansum(self.star_pixels_stds_of_ppms**2) / len(self.star_pixels_stds_of_ppms)**2
        #                 + np.nansum(self.bg_pixels_stds_of_ppms**2) / len(self.bg_pixels_stds_of_ppms)**2)
        diff_ppms = self.star_pixels_ppms - self.bg_pixels_ppms
        var_diff_ppms = self.star_pixels_stds_of_ppms ** 2 + self.bg_pixels_stds_of_ppms ** 2
        valid_mask = (~np.isnan(diff_ppms)) & (~np.isnan(var_diff_ppms) & ~np.isclose(var_diff_ppms, 0))
        test_stat = np.nansum(diff_ppms[valid_mask] / var_diff_ppms[valid_mask]) / np.nansum(1 / var_diff_ppms[valid_mask])**0.5

        # sig_big = (np.nanmean(star_part_stds_of_ppms ** 2) + np.nanmean(bg_part_stds_of_ppms ** 2)
        # + np.nanvar(star_part_ppms) + np.nanvar(bg_part_ppms)) ** 0.5
        if test_stat > 0:
            false_neg = ut.get_p_value_gaussian(test_stat, 0, 1)
            self.p_value_mean_bg_pixels = false_neg
            if false_neg < threshold_false_negative_bg_target_pixels_mean:
                self.is_bad_bg_pixels = True
            else:
                self.is_bad_bg_pixels = False
        else:
            self.is_bad_bg_pixels = False

        # Checking the worst deviation:
        if (diff_ppms[valid_mask] > 0).any():
            inds = (diff_ppms > 0) & valid_mask
            diffs_single_ppm = diff_ppms[inds]
            stds_single_ppm = var_diff_ppms[inds]**0.5
            false_neg_singles = [ut.get_p_value_gaussian(diffs_single_ppm[i] / stds_single_ppm[i], 0, 1.)
                                 for i in range(len(diffs_single_ppm))]
            worst_false_neg_single = np.nanmin(false_neg_singles)
            self.p_value_worst_bg_pixels = worst_false_neg_single

            # look-elsewhere:
            # 1 - (1 - p1)**n = p:
            # 1 - p1 = (1 - p)**(1/n)
            false_neg_1 = 1 - (1 - threshold_false_negative_bg_target_pixels_max)**(1 / len(np.where(valid_mask)[0]))
            if worst_false_neg_single < false_neg_1:
                self.is_bad_bg_pixels = True

    def test_centroid_transits_snr(self, threshold_snr_2=Threshold_centroid_snr_2):
        max_coords_deviation_periodic = self.find_centroid_transits_snr(period=self.best_per, ftt=self.best_ftt,
                                                                        nsig_outliers_0_pos=Nsig_outliers_0_pos_tests,
                                                                        nsig_outliers_0_neg=Nsig_outliers_0_neg_tests)
        self.centroid_snrs_ra_dec = max_coords_deviation_periodic
        self.centroid_snr_2 = max_coords_deviation_periodic[0]**2 + max_coords_deviation_periodic[1]**2
        if self.centroid_snr_2 > threshold_snr_2:
            self.is_centroid_bad = True
        else:
            self.is_centroid_bad = False

    # noinspection PyProtectedMember
    def find_centroid_transits_snr(self, period=None, ftt=None, nsig_outliers_0_pos=Nsig_outliers_0_pos_tests,
                                   nsig_outliers_0_neg=Nsig_outliers_0_neg_tests):
        if period is None:
            period = self.best_per
        if ftt is None:
            ftt = self.best_ftt
        # times, target_pixel_fluxes = af.get_target_pixel_data_fun(self.target_id, data_folder=self.target_pixels_location)
        times, target_pixel_fluxes, sky_coords, time_masks, aperture_masks = af.get_target_pixel_sky_coords_fun(
            target_id=self.target_id, data_location=self.target_pixels_location,
            remove_zeroth_quarter=True, remove_quarter_17=True, cut_edge_nans=True)
        centroid_x_quarters, centroid_y_quarters = af.get_centroid_series(target_pixel_fluxes=target_pixel_fluxes,
                                                                          sky_coords=sky_coords,
                                                                          time_masks=time_masks, aperture_masks=aperture_masks)

        max_coords_deviation_periodic = []
        for fluxes in [centroid_x_quarters, centroid_y_quarters]:
            # making the lightcurve out of this whatever data and doing search:
            klc = spip.get_minimal_processed_folded_klc(target_id=None, get_data=False, times=times, fluxes=fluxes,
                                                        period=period, template_index='all',
                                                        nsig_outliers_0_neg=nsig_outliers_0_neg,
                                                        nsig_outliers_0_pos=nsig_outliers_0_pos)

            # Finding  correlation size:
            per_ind_left, per_ind_right = klc.get_influence_period_indices(period=period, part_snr_2=0.5)
            # part_snr_2 = 0.5  # define this as the end of correlation
            # d_per = klc.periods[1] - klc.periods[0]
            # dur_max_templ = klc.tb.durations[-1]
            # # n_transits = int(np.round(klc.t_mission / period))
            # delta_per_bins = int(np.round((period * 2 * dur_max_templ / klc.t_mission * (1 - part_snr_2 ** 0.5)) / d_per))
            # close_period_ind = np.nanargmin(np.abs(klc.periods - period))
            # per_ind_left = max(0, close_period_ind - delta_per_bins)  # we do not divide delta_per_bins by 2 on purpose
            # per_ind_right = min(len(klc.periods), close_period_ind + delta_per_bins)

            max_deviation = []
            for in_interl, phases, stats_periodic in [[True, klc._phases_interleaved, klc.statistics_periodic_interleaved],
                                                      [False, klc._phases_not_interleaved, klc.statistics_periodic_not_interleaved]]:
                phase_ind_left, phase_ind_right = klc.get_influence_phase_indices(first_transit_time=ftt, in_interleaved=in_interl,
                                                                                  part_snr_2=0.5)
                # d_ftt = phases[1] - phases[0]
                # delta_ftt_bins = int(np.round((dur_max_templ * (1 - part_snr_2 ** 0.5)) / d_ftt))
                # close_phase_ind = np.nanargmin(np.abs(phases + klc.tmin - ftt))
                # phase_ind_left = max(0, close_phase_ind - delta_ftt_bins)  # we do not divide delta_ftt_bins by 2 on purpose
                # phase_ind_right = min(len(phases), close_phase_ind + delta_ftt_bins)
                max_pos_dev = np.nanmax(stats_periodic[per_ind_left:per_ind_right,
                                        :, phase_ind_left:phase_ind_right])
                max_neg_dev = np.nanmin(stats_periodic[per_ind_left:per_ind_right,
                                        :, phase_ind_left:phase_ind_right])
                if np.abs(max_pos_dev) > np.abs(max_neg_dev):
                    max_deviation.append(max_pos_dev)
                else:
                    max_deviation.append(max_neg_dev)

            if np.abs(max_deviation[0]) > np.abs(max_deviation[1]):
                max_coords_deviation_periodic.append(max_deviation[0])
            else:
                max_coords_deviation_periodic.append(max_deviation[1])
        return max_coords_deviation_periodic

    # def test_single_pixels(self):
    #     """
    #     Takes every pixel in the optimal aperture and checks the statistics at the transit times
    #     :return:
    #     :rtype:
    #     """
    #     pixel_times, target_pixel_fluxes, bit_aperture_masks = af.get_target_pixel_data_fun(target_id=self.target_id,
    #                                                                                         data_folder=self.target_pixels_location,
    #                                                                                         remove_zeroth_quarter=True,
    #                                                                                         remove_quarter_17=True,
    #                                                                                         return_aperture_masks=True)
    #
    #     target_pixels_stats, target_pixels_means, \
    #         target_pixels_ppms, target_pixels_stds_of_ppms = af.get_single_pixels_stats(times=pixel_times,
    #                                                                                     target_pixel_fluxes=target_pixel_fluxes,
    #                                                                                     bit_aperture_masks=bit_aperture_masks,
    #                                                                                     summed_transit_times=self.summed_transit_times,
    #                                                                                     best_temp=self.best_temp)
    #     self.target_pixels_stats = target_pixels_stats
    #     self.target_pixels_means = target_pixels_means
    #     self.target_pixels_ppms = target_pixels_ppms
    #     self.target_pixels_stds_of_ppms = target_pixels_stds_of_ppms

    def check_quality_flags(self):
        """
        Opens the fits data and accesses the quality flag series. looks in the vicinity of every summed transit point
        and records all that was there.
        :return:
        :rtype:
        """
        pass
