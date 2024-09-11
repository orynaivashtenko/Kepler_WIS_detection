#!/usr/bin/env python3

import copy
import os
core_num = "1"
os.environ["OMP_NUM_THREADS"] = core_num
os.environ["OPENBLAS_NUM_THREADS"] = core_num
os.environ["MKL_NUM_THREADS"] = core_num
os.environ["VECLIB_MAXIMUM_THREADS"] = core_num
os.environ["NUMEXPR_NUM_THREADS"] = core_num
# import numpy as np
import pickle
from sys import getsizeof
# import target_class
from pipeline_parameters import R_sun_R_e, Location_attributes, Per_min_candidate, Delta_stat_close, Delta_integral_close
import locations as locs
from tce_parameters import *
import auxiliary_functions as af
import search_pipeline as spip
import utils as ut
import template_bank.template_module as tm
from catalog_functions import get_koi_ttv_dicts, get_kepler_tce_dict, read_neighbor_info, read_gaia_table_for_target_id
from scipy.stats import chi2
# from astropy.table import Table


class WisTCE(object):
    def __init__(self, target_id='003097222',
                 data_location=locs.Data_location, output_location=locs.Output_location,
                 target_pixels_location=Target_pixels_location,
                 load_pickled_attributes=False,
                 load_target_attributes=True,
                 gaia_kepler_file_path=Gaia_kepler_file,
                 koi_dict_file=Koi_dict_file,
                 ttv_dict_file=Ttv_dict_file,
                 villanova_dict_file=Villanova_dict_file,
                 limb_darkening_array_file=Limb_darkening_array_file,
                 allow_output_folder_creation=True,
                 allow_data_folder_creation=True,
                 problems_folder=Problems_folder,
                 add_prefix_to_problems_folder=True,
                 clean_problems_file=False,
                 overwrite_location_parameters_by_unpickle=False,
                 plots_location=None,
                 allow_plots_folder_creation=True):

        data_location = af.get_correct_tid_folder(target_id, data_location, allow_creation=allow_data_folder_creation)
        output_location = af.get_correct_tid_folder(target_id, output_location, allow_creation=allow_output_folder_creation)
        target_pixels_location = af.get_correct_tid_folder(target_id, target_pixels_location, allow_creation=allow_data_folder_creation)

        # Basic attributes:
        self.target_id = str(target_id).rjust(9, '0')
        self.mean_flux = None
        self.n_quarters = None

        # Location attributes:
        self.data_location = data_location
        self.output_location = output_location
        self.gaia_kepler_file = gaia_kepler_file_path
        self.koi_dict_file = koi_dict_file
        self.ttv_dict_file = ttv_dict_file
        self.villanova_dict_file = villanova_dict_file
        self.target_pixels_location = target_pixels_location
        self.limb_darkening_array_file = limb_darkening_array_file
        if plots_location is not None:
            plots_location = af.get_correct_tid_folder(target_id, plots_location,
                                                       allow_creation=allow_plots_folder_creation,
                                                       with_prefix_folder=False)
            self.plots_location = plots_location
        else:
            self.plots_location = None

        if problems_folder is None:
            problems_folder = self.output_location
        elif add_prefix_to_problems_folder:
            if (os.path.basename(problems_folder) != self.target_id[:4]
                    and os.path.basename(problems_folder) != self.target_id):
                if self.target_id[:4] in os.listdir(problems_folder):
                    problems_folder = os.path.join(problems_folder, target_id[:4])
        self.problems_file_location = os.path.join(problems_folder, target_id + '_' + Problems_tce_txt_filename_postfix)
        if clean_problems_file:
            if os.path.exists(self.problems_file_location):
                os.remove(self.problems_file_location)

        # Trigger attributes
        self.best_int_score = None
        self.best_stat = None
        self.best_per = None
        self.best_ftt = None
        self.best_temp = None
        self.best_gauss_stat = None
        self.best_g_stat = None
        self.best_g_per = None
        self.best_g_temp = None
        self.best_g_ftt = None
        self.best_g_gauss_stat = None
        self.best_g_int_score = None

        # Significance attributes:
        self.prob_density_planet_not_noise = None
        self.false_alarm_prob_trigger = None
        self.prob_density_h0_trigger = None
        self.prob_density_h1_trigger = None

        # Stellar and Kepler attributes:
        self.stellar_radius = None
        self.m_1_3_r_s = None
        self.stellar_mass = None
        self.kepmag = None
        self.phot_g_mean_mag = None
        self.parallax = None
        self.gaia_abs_gmag = None
        self.gaia_b_r = None
        self.teff = None
        self.metallicity = None
        self.logg = None
        self.ra = None
        self.dec = None
        self.ruwe = None
        self.gaia_nconfp = None
        self.gaia_non_single_star_flag = None
        self.gaia_variable_star_flag = None
        self.is_villanova_binary = None

        # Koi and Kepler TCE attributes:
        self.is_koi = None
        self.is_tce = None
        self.koi_status = None
        self.kepler_tces = None
        self.koi_dict = None
        self.ttv_dict = None
        self.tce_dict = None
        self.n_overlapping_koi_transits = None
        self.koi_overlap_mask = None
        self.kepler_tce_overlap_mask = None

        # Transits attributes:
        self.n_transits = None
        self.n_valid_transits = None
        self.badness_of_transits = None
        self.valid_transit_mask = None

        self.summed_transit_times = None
        self.summed_raw_stats = None
        self.summed_vars = None
        self.summed_corrections = None
        self.summed_gauss_stats = None

        self.transit_orig_quarters = None
        self.transit_seasons = None
        self.transit_amplitudes = None
        self.mean_transit_amplitude = None
        self.transit_amplitude_stds = None
        self.transit_normalized_amplitudes = None
        self.transit_normalized_amplitude_stds = None
        self.transit_ppms = None
        self.transit_ppm_stds = None
        self.mean_transit_ppm = None
        self.mean_transit_ppm_std = None
        self.mean_ppm_variability = None
        self.rp_to_r_earth = None
        self.rp_to_r_earth_std = None
        self.rp_to_r_earth_std_conservative = None
        self.rp_to_rs = None
        self.rp_to_rs_std = None
        self.mean_transit_amplitude = None
        self.mean_transit_ppm = None
        self.mean_transit_ppm_std = None
        self.transit_duration_hours = None

        # Vetting attributes:
        self.too_few_valid_transits = None
        self.star_pixels_ppms = None
        self.star_pixels_stds_of_ppms = None
        self.bg_pixels_ppms = None
        self.bg_pixels_stds_of_ppms = None
        self.p_value_mean_bg_pixels = None
        self.p_value_worst_bg_pixels = None
        self.is_bad_bg_pixels = None
        self.klc_star_pixels = None
        self.klc_bg_pixels = None
        self.star_valid_transit_mask = None
        self.bg_valid_transit_mask = None
        self.centroid_snrs_ra_dec = None
        self.centroid_templates_ra_dec = None
        self.centroid_snr_2 = None
        self.is_centroid_bad = None
        self.klc_centroid_x = None
        self.klc_centroid_y = None
        self.is_bad_periodicity = None
        self.bad_periodicity_reasons = None
        self.p_value_fold_residuals = None
        self.p_value_mf_fold_residuals = None
        self.fn_fold_veto = None  # the parameter used in the veto, not the result of the veto!
        self.is_bad_fold_residuals = None
        self.best_temp_folded = None
        self.template_and_stat_of_fold_contaminant = None
        self.is_bad_fold_residuals_fft = None
        self.max_season_count = None
        self.is_bad_seasons = None
        self.is_good_pre_filtering = None
        # Short periodicity search results attributes
        self.best_short_int_score = None
        self.best_short_stat = None
        self.best_short_per = None
        self.best_short_ftt = None
        self.best_short_temp = None
        self.best_short_gauss_stat = None
        self.is_short_periodicity_trigger = None
        self.is_period_short = None

        # Tables:
        self.neighbors_table = None
        self.gaia_target_table = None

        # Search attributes:
        self.is_koi_planet_masked = None
        self.was_notched = None
        self.failures = None
        self.n_slides = None

        if load_target_attributes:
            self.get_attributes_from_target(overwrite_location_parameters=False, overwrite_existing=True)

        if load_pickled_attributes:
            self.unpickle_small_attributes(overwrite_location_parameters=overwrite_location_parameters_by_unpickle)

    def get_transit_times_info(self, period=None, first_transit_time=None, template=None, from_real_klc=True, max_time=Quarter_16_end,
                               data_location=None, output_location=None):
        if period is None:
            period = self.best_per
        if first_transit_time is None:
            first_transit_time = self.best_ftt
        if template is None:
            template = self.best_temp
        if data_location is None:
            data_location = self.data_location
        if output_location is None:
            output_location = self.output_location
        if from_real_klc:
            klc = spip.get_fold_quickly(target_id=self.target_id, data_location=data_location,
                                        output_location=output_location, period=period,
                                        overlap_period_bins=0)
            klc.get_summed_values(first_transit_time=first_transit_time, period=period, template=template)
            self.summed_transit_times = klc.summed_times
            self.summed_raw_stats = klc.summed_stats_raw
            self.summed_vars = klc.summed_vars
            self.summed_corrections = klc.summed_corrections
            self.summed_gauss_stats = klc.summed_stats_raw_gauss
            self.mean_flux = klc.mean_total
            self.valid_transit_mask = (~np.isnan(self.summed_raw_stats)) & (~np.isnan(self.summed_vars)) & (self.summed_vars > 0)
        else:
            self.summed_transit_times = ut.get_transit_times(first_transit_time=first_transit_time,
                                                             period=period, max_time=max_time)
        self.transit_orig_quarters = af.get_orig_quarters(self.summed_transit_times)
        self.transit_seasons = self.transit_orig_quarters % 4

    def seasonal_test(self, threshold_fraction_same_season=Threshold_fraction_same_season):
        season_options = np.arange(4)
        transit_seasons_list = list(self.transit_seasons)
        season_counts = [transit_seasons_list.count(season) for season in season_options]
        self.max_season_count = np.max(season_counts)
        fraction_same_season = self.max_season_count / len(transit_seasons_list)
        if fraction_same_season > threshold_fraction_same_season:
            self.is_bad_seasons = True
        else:
            self.is_bad_seasons = False

    def get_attributes_from_target(self, overwrite_location_parameters=False,
                                   output_location=None,
                                   overwrite_existing=True,
                                   pickle_target_small_attributes_filename_postfix=locs.Pickle_target_small_attributes_filename_postfix):
        if output_location is None:
            output_location = self.output_location

        file_path = af.get_saving_file_path(output_location=output_location, target_id=self.target_id,
                                            filename_postfix=pickle_target_small_attributes_filename_postfix, file_format='pickle')

        dict_of_old_locations = {}
        for key in Location_attributes:
            if key in self.__dict__.keys():
                dict_of_old_locations[key] = self.__dict__[key]

        if overwrite_existing:
            dict_old = None
        else:
            dict_old = copy.deepcopy(self.__dict__)

        with open(file_path, 'rb') as pickle_file:
            dict1 = pickle.load(pickle_file)
            self.__dict__.update(dict1.items())

        if not overwrite_existing:
            for key in dict_old:
                if dict_old[key] is not None:
                    self.__dict__[key] = dict_old[key]

        if not overwrite_location_parameters:
            for key in Location_attributes:
                if key in self.__dict__.keys():
                    self.__dict__[key] = dict_of_old_locations[key]

    def pickle_small_attributes(self, output_location=None,
                                pickle_small_attributes_filename_postfix=Pickle_tce_small_attributes_filename_postfix):
        """
        Pickles all attributes which are of type str, float, int, np.float, np.integer, np.ndarray, list, dict
        :param pickle_small_attributes_filename_postfix:
        :type pickle_small_attributes_filename_postfix:
        :param output_location:
        :type output_location
        :return:
        """
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

        file_path = af.get_saving_file_path(output_location=output_location, target_id=self.target_id,
                                            filename_postfix=pickle_small_attributes_filename_postfix, file_format='pickle')
        with open(file_path, 'wb') as pickle_file:
            pickle.dump(dict_to_pickle, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

    def unpickle_small_attributes(self, output_location=None,
                                  pickle_tce_small_attributes_filename_postfix=Pickle_tce_small_attributes_filename_postfix,
                                  overwrite_location_parameters=False,
                                  load_target_attributes=True, overwrite_existing_by_target=True,
                                  pickle_target_small_attributes_filename_postfix=locs.Pickle_target_small_attributes_filename_postfix):
        if output_location is None:
            output_location = self.output_location

        dict_of_old_locations = {}
        for key in Location_attributes + ['target_pixels_location']:
            dict_of_old_locations[key] = self.__dict__[key]

        if load_target_attributes:
            self.get_attributes_from_target(
                pickle_target_small_attributes_filename_postfix=pickle_target_small_attributes_filename_postfix,
                overwrite_location_parameters=overwrite_location_parameters,
                overwrite_existing=overwrite_existing_by_target, output_location=output_location)

        file_path = af.get_saving_file_path(output_location=output_location, target_id=self.target_id,
                                            filename_postfix=pickle_tce_small_attributes_filename_postfix, file_format='pickle')
        if os.path.exists(file_path):
            with open(file_path, 'rb') as pickle_file:
                dict1 = pickle.load(pickle_file)
                self.__dict__.update(dict1.items())

        if not overwrite_location_parameters:
            for key in Location_attributes + ['target_pixels_location']:
                if key in self.__dict__.keys():
                    self.__dict__[key] = dict_of_old_locations[key]

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

        file_path = af.get_saving_file_path(output_location=output_location, target_id=self.target_id,
                                            filename_postfix=pickle_all_attributes_filename_postfix, file_format='pickle')
        with open(file_path, 'wb') as pickle_file:
            pickle.dump(dict_to_pickle, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

    def unpickle_all_attributes(self, output_location=None,
                                pickle_all_attributes_filename_postfix=Pickle_tce_all_attributes_filename_postfix):
        if output_location is None:
            output_location = self.output_location
        file_path = af.get_saving_file_path(output_location=output_location, target_id=self.target_id,
                                            filename_postfix=pickle_all_attributes_filename_postfix, file_format='pickle')
        with open(file_path, 'rb') as pickle_file:
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
        fields_not_to_print = ['gaia_kepler_file',
                               'koi_dict_file',
                               'ttv_dict_file',
                               'villanova_dict_file',
                               'limb_darkening_array_file']

        # numpy.savetxt()
        if output_location is None:
            output_location = self.output_location
        dict_to_print = {}
        for key in self.__dict__:
            if key not in fields_not_to_print:
                value = self.__dict__[key]
                if isinstance(value, (str, float, int, np.float32, np.float64, np.complex64, np.complex128,
                                      np.integer, np.ndarray, list, dict)) or value is None:
                    ok_to_print = True
                    if isinstance(value, (str, float, int, np.float32, np.float64, np.complex64, np.complex128,
                                          np.integer, np.ndarray, list, dict)):
                        if hasattr(value, '__len__'):
                            if isinstance(value, dict) and len(list(value.keys())) > 0 and hasattr(value[list(value.keys())[0]], '__len__'):
                                ok_to_print = False
                            elif ((not isinstance(value, dict) and not (value is None))
                                  and (len(value) > 0 and (not isinstance(value[0], str)) and hasattr(value[0], '__len__'))):
                                ok_to_print = False  # not printing nested lists
                            elif (not isinstance(value, str)) and len(value) > 50:
                                # roughly to be able to set summed values for planets with period 30
                                ok_to_print = False
                    if ok_to_print:
                        dict_to_print[key] = value
        file_path = af.get_saving_file_path(output_location=output_location, target_id=self.target_id,
                                            filename_postfix=dict_txt_filename_postfix, file_format='txt')
        with open(file_path, 'w') as fil:
            for key in dict_to_print:
                print(key, ":", dict_to_print[key], file=fil)

    def get_periodicity_results(self, short_periodicity=True,
                                best_periodic_scores_filename_postfix=locs.Best_periodic_scores_filename_postfix,
                                best_short_periodic_scores_filename_postfix=locs.Best_short_periodic_scores_filename_postfix,
                                return_lists=False, set_attributes=True,
                                delta_stat_close=Delta_stat_close,
                                delta_integral_close=Delta_integral_close):
        return_array = af.get_periodicity_results(target_id=self.target_id, output_location=self.output_location,
                                                  short_periodicity=short_periodicity,
                                                  best_periodic_scores_filename_postfix=best_periodic_scores_filename_postfix,
                                                  best_short_periodic_scores_filename_postfix=best_short_periodic_scores_filename_postfix)
        if set_attributes:
            ind_max, _, __ = af.get_best_int_score_and_stat_index(int_scores=return_array[spip.Best_per_results_index_best_integral],
                                                                  stats=return_array[spip.Best_per_results_index_stats],
                                                                  delta_stat_close=delta_stat_close,
                                                                  delta_integral_close=delta_integral_close)
            # ind_max = np.argmax(return_array[spip.Best_per_results_index_best_integral])
            if short_periodicity:
                self.best_short_int_score = return_array[spip.Best_per_results_index_best_integral][ind_max]
                self.best_short_stat = return_array[spip.Best_per_results_index_stats][ind_max]
                self.best_short_per = return_array[spip.Best_per_results_index_per][ind_max]
                self.best_short_ftt = return_array[spip.Best_per_results_index_ftt][ind_max]
                self.best_short_temp = return_array[spip.Best_per_results_index_template][ind_max]
                self.best_short_gauss_stat = return_array[spip.Best_per_results_index_stats_gauss][ind_max]
            else:
                self.best_int_score = return_array[spip.Best_per_results_index_best_integral][ind_max]
                self.best_stat = return_array[spip.Best_per_results_index_stats][ind_max]
                self.best_per = return_array[spip.Best_per_results_index_per][ind_max]
                self.best_ftt = return_array[spip.Best_per_results_index_ftt][ind_max]
                self.best_temp = return_array[spip.Best_per_results_index_template][ind_max]
                self.best_gauss_stat = return_array[spip.Best_per_results_index_stats_gauss][ind_max]
        if return_lists:
            return return_array

    def veto_short_periodicity(self, best_short_periodic_scores_filename_postfix=locs.Best_short_periodic_scores_filename_postfix,
                               best_periodic_scores_filename_postfix=locs.Best_periodic_scores_filename_postfix,
                               minimal_period=Per_min_candidate):
        self.get_periodicity_results(short_periodicity=True,
                                     best_short_periodic_scores_filename_postfix=best_short_periodic_scores_filename_postfix,
                                     return_lists=False, set_attributes=True)
        if self.best_int_score is None:
            self.get_periodicity_results(short_periodicity=False,
                                         best_periodic_scores_filename_postfix=best_periodic_scores_filename_postfix,
                                         return_lists=False, set_attributes=True)
        if self.best_short_int_score > self.best_int_score:
            self.is_short_periodicity_trigger = True
        else:
            self.is_short_periodicity_trigger = False
        if self.best_per < minimal_period:
            self.is_period_short = True
        else:
            self.is_period_short = False

    def get_trigger_star_bg_ppms(self, target_pixels_folder=None,
                                 part_of_flux_peak_threshold=Part_of_flux_peak_threshold,
                                 number_of_pixels_cut=None, part_of_cumsum_2=Part_of_cumsum_2_targ_pix,
                                 part_of_light=None, period=None, template_index=None, first_transit_time=None,
                                 nsig_outliers_0_neg=Nsig_outliers_0_neg_tests,
                                 nsig_outliers_0_pos=Nsig_outliers_0_pos_tests,
                                 save_klcs=False, save_data_to_file=False):
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
                                                                                    part_of_peak=part_of_flux_peak_threshold,
                                                                                    number_of_pixels_cut=number_of_pixels_cut,
                                                                                    part_of_cumsum_2=part_of_cumsum_2,
                                                                                    part_of_light=part_of_light)
        filename_postfixes = [[Connected_stats_star_pixels_filename_postfix, Periodic_stats_star_pixels_filename_postfix],
                              [Connected_stats_bg_pixels_filename_postfix, Periodic_stats_bg_pixels_filename_postfix]]

        ppms_and_stds_star_and_bg = []
        klcs = []
        for i_flux, fluxes in enumerate([star_part_sap_fluxes, bg_part_sap_fluxes]):
            # making the lightcurve out of this whatever data and doing search:
            klc = spip.get_minimal_processed_folded_klc(target_id=self.target_id, get_data=False,  # target_id is needed for saving name
                                                        times=pixel_times, fluxes=copy.deepcopy(fluxes),
                                                        period=period, template_index=template_index,
                                                        cut_edge_nans=False, delete_fulls=False,
                                                        nsig_outliers_0_neg=nsig_outliers_0_neg,
                                                        nsig_outliers_0_pos=nsig_outliers_0_pos,
                                                        data_location=None, output_location=None)
            # -> we do not marginalize, so statistics are still not in units of snr squared
            if save_data_to_file:
                klc.save_connected_stats(filename_postfix=filename_postfixes[i_flux][0],
                                         output_location=self.output_location)
                klc.save_statistics_periodic(output_location=self.output_location,
                                             filename_postfix=filename_postfixes[i_flux][1], specify_periods=True)

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
                statistics_periodic = klc.statistics_periodic_not_interleaved
                statistics = klc.statistics_not_interleaved
                stat_vars = klc.stat_vars_not_interleaved

            # Finding  correlation size:
            period_index_left, period_index_right = klc.get_influence_period_indices(period=period, part_snr_2=0.5)
            phase_index_left, phase_index_right = klc.get_influence_phase_indices(first_transit_time=first_transit_time,
                                                                                  in_interleaved=is_interleaved, part_snr_2=0.3)

            # Allow the actual minimum to be a little bit shifted:
            # Is it okay that it is independent in star and bg? I think so because maybe the actual minimum in the BG was a bit off,
            # and the star caught it in a different time due to its processed. We are interested in what happened in
            # the background in the worst case.
            # Redefining the best period index with a small freedom:
            period_index = period_index_left + np.argmin(np.nanmin(statistics_periodic[period_index_left:period_index_right, 0,
                                                                   phase_index_left:phase_index_right], axis=-1))
            # Warning: we are taking minima here assuming that the statistics are not squared, and that the deviation is actually negative.
            # redefining the best phase index with a small freedom:
            phase_index = phase_index_left + np.argmin(statistics_periodic[period_index][0][
                                                       phase_index_left:phase_index_right])
            # -> allow the actual minimum to be a little bit shifted
            period_new = klc.periods[period_index]
            first_transit_time_new = phases[phase_index] + klc.tmin
            expected_transit_times = ut.get_transit_times(first_transit_time=first_transit_time_new, period=period_new, max_time=klc.tmax)
            # -> I am not sure tmin does not miss some nan in the beginning
            summed_transit_points = [np.nanargmin(np.abs(time - expected_transit_times[i])) for i in range(len(expected_transit_times))]
            # summed_transit_times = np.array(time[summed_transit_points])
            summed_raw_stats = statistics[ti][summed_transit_points]
            summed_vars = stat_vars[ti][summed_transit_points]

            (ppm, ppm_std, ppm_variability, ppms, ppms_stds,
             ampl, ampl_std, ampl_variability, ampls, ampls_stds,
             normalized_ampls, normalized_ampls_stds) = af.estimate_ampls_ppms_from_stats(
                summed_raw_stats=summed_raw_stats, summed_stat_vars=summed_vars,
                templ_max=np.max(np.abs(klc.tb.templates_shifts[ti_sh])), flux_trend=klc.mean_total, trends_std=None)
            ppms_and_stds_star_and_bg.append([ppms, ppms_stds])
            if save_klcs:
                klcs.append(copy.deepcopy(klc))

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
                                      part_of_cumsum_2=Part_of_cumsum_2_targ_pix,
                                      save_byproducts_to_file=False):
        if (self.star_pixels_ppms is None or self.star_pixels_stds_of_ppms is None
                or self.bg_pixels_ppms is None or self.bg_pixels_stds_of_ppms is None):
            self.get_trigger_star_bg_ppms(
                target_pixels_folder=self.target_pixels_location,
                number_of_pixels_cut=None, part_of_cumsum_2=part_of_cumsum_2, part_of_light=None,
                save_data_to_file=save_byproducts_to_file)

        # First, see what happened in the transits because they can happen to be nans if the transit in one of them is too deep:
        self.star_valid_transit_mask = (~np.isnan(self.star_pixels_ppms)) & (~np.isnan(self.star_pixels_stds_of_ppms)
                                                                             & (self.star_pixels_stds_of_ppms > 0))
        self.bg_valid_transit_mask = (~np.isnan(self.bg_pixels_ppms)) & (~np.isnan(self.bg_pixels_stds_of_ppms)
                                                                         & (self.bg_pixels_stds_of_ppms > 0))
        valid_star_bg_transit_mask = self.star_valid_transit_mask & self.bg_valid_transit_mask
        if len(np.where(valid_star_bg_transit_mask)[0]) < 3:
            self.is_bad_bg_pixels = True
        else:
            # If everything is fine, then look at the ppm difference of the star and the background pixels:
            diff_ppms = self.star_pixels_ppms - self.bg_pixels_ppms
            var_diff_ppms = self.star_pixels_stds_of_ppms ** 2 + self.bg_pixels_stds_of_ppms ** 2
            valid_mask = (~np.isnan(diff_ppms)) & (~np.isnan(var_diff_ppms) & ~np.isclose(var_diff_ppms, 0))
            test_stat = np.nansum(diff_ppms[valid_mask] / var_diff_ppms[valid_mask]) / np.nansum(1 / var_diff_ppms[valid_mask])**0.5

            false_neg = ut.get_p_value_gaussian(test_stat, 0, 1)
            self.p_value_mean_bg_pixels = false_neg
            if test_stat > 0:
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

    def test_centroid_transits_snr(self, threshold_snr_2=Threshold_centroid_snr_2,
                                   save_byproducts_to_file=False):
        if not hasattr(self, 'self.centroid_snrs_ra_dec') or self.centroid_snrs_ra_dec is None:
            self.find_centroid_transits_snr(period=self.best_per, ftt=self.best_ftt,
                                            nsig_outliers_0_pos=Nsig_outliers_0_pos_tests,
                                            nsig_outliers_0_neg=Nsig_outliers_0_neg_tests,
                                            save_data_to_file=save_byproducts_to_file)
        self.centroid_snr_2 = self.centroid_snrs_ra_dec[0]**2 + self.centroid_snrs_ra_dec[1]**2
        # self.centroid_snr_2 = self.centroid_snrs_ra_dec[0]**2 * np.cos(self.dec)**2 + self.centroid_snrs_ra_dec[1]**2
        # -> we did not use the spherical basis geometry to make the length element. The reason is that we sum the
        # snr, which will already include its impact. Say we are on the equator, then the snr is the same for both
        # directions. If we are on the pole, then the snr is less for ra because the noise is larger. So the effect
        # of that cosine is taken into account by the snr of ra.
        # this is statistically not really correct, but we at least know what are the snrs we are talking about.
        # we expect the centroid to deviate even if the planet is real. if it deviates in the incorrect direction wrt
        # the neighboring stars, then it is bad.
        if self.centroid_snr_2 > threshold_snr_2:
            self.is_centroid_bad = True
        else:
            self.is_centroid_bad = False

    # noinspection PyProtectedMember
    def find_centroid_transits_snr(self, period=None, ftt=None, nsig_outliers_0_pos=Nsig_outliers_0_pos_tests,
                                   nsig_outliers_0_neg=Nsig_outliers_0_neg_tests,
                                   save_klcs=False, scaling_factor=Scaling_factor_centroid_flux,
                                   save_data_to_file=False):
        if period is None:
            period = self.best_per
        if ftt is None:
            ftt = self.best_ftt
        # times, target_pixel_fluxes = af.get_target_pixel_data_fun(self.target_id, data_folder=self.target_pixels_location)
        times, target_pixel_fluxes, sky_coords, time_masks, aperture_masks = af.get_target_pixel_sky_coords_fun(
            target_id=self.target_id, data_location=self.target_pixels_location,
            remove_zeroth_quarter=True, remove_quarter_17=True, cut_edge_nans=True)
        # Warning! Changing aperture masks to cover the entire stamp and make the centroid actually average over something:
        aperture_masks = [np.ones_like(aperture_masks[q]).astype(bool) for q in range(len(aperture_masks))]
        centroid_x_quarters, centroid_y_quarters = af.get_centroid_series(target_pixel_fluxes=target_pixel_fluxes,
                                                                          sky_coords=sky_coords,
                                                                          time_masks=time_masks, aperture_masks=aperture_masks)
        # Adding the scaling factor to avoid the numerical precision issues:
        centroid_x_quarters = [centroid_x_quarters[q] * scaling_factor for q in range(len(centroid_x_quarters))]
        centroid_y_quarters = [centroid_y_quarters[q] * scaling_factor for q in range(len(centroid_y_quarters))]
        filename_postfixes = [[Connected_stats_centroid_x_filename_postfix, Periodic_stats_centroid_x_filename_postfix],
                              [Connected_stats_centroid_y_filename_postfix, Periodic_stats_centroid_y_filename_postfix]]
        max_coords_deviation_periodic = []
        max_coords_template_periodic = []
        klcs = []
        for i_flux, fluxes in enumerate([centroid_x_quarters, centroid_y_quarters]):
            # making the lightcurve out of this whatever data and doing search:
            klc = spip.get_minimal_processed_folded_klc(target_id=self.target_id, get_data=False,  # target_id is needed for saving name
                                                        times=times, fluxes=fluxes,
                                                        period=period, template_index='all', delete_fulls=False,
                                                        nsig_outliers_0_neg=nsig_outliers_0_neg,
                                                        nsig_outliers_0_pos=nsig_outliers_0_pos,
                                                        data_location=None, output_location=None)
            if save_data_to_file:
                klc.save_connected_stats(filename_postfix=filename_postfixes[i_flux][0],
                                         output_location=self.output_location)
                klc.save_statistics_periodic(output_location=self.output_location,
                                             filename_postfix=filename_postfixes[i_flux][1], specify_periods=True)

            # Finding  correlation size:
            per_ind_left, per_ind_right = klc.get_influence_period_indices(period=period, part_snr_2=0.5)

            max_deviation = []
            max_deviation_template = []
            for in_interl, phases, stats_periodic in [[True, klc._phases_interleaved, klc.statistics_periodic_interleaved],
                                                      [False, klc._phases_not_interleaved, klc.statistics_periodic_not_interleaved]]:
                phase_ind_left, phase_ind_right = klc.get_influence_phase_indices(first_transit_time=ftt, in_interleaved=in_interl,
                                                                                  part_snr_2=0.2)
                # -> we take a big freedom because if it is a glitch of different shape, its ftt may be shifted
                max_pos_dev = np.nanmax(stats_periodic[per_ind_left:per_ind_right,
                                        :, phase_ind_left:phase_ind_right])
                max_neg_dev = np.nanmin(stats_periodic[per_ind_left:per_ind_right,
                                        :, phase_ind_left:phase_ind_right])
                arg_max_pos_dev = np.nanargmax(np.nanmax(stats_periodic[per_ind_left:per_ind_right,
                                               :, phase_ind_left:phase_ind_right], axis=(0, 2)))
                arg_max_neg_dev = np.nanargmin(np.nanmin(stats_periodic[per_ind_left:per_ind_right,
                                               :, phase_ind_left:phase_ind_right], axis=(0, 2)))
                if np.abs(max_pos_dev) > np.abs(max_neg_dev):
                    max_deviation.append(max_pos_dev)
                    if in_interl:
                        max_deviation_template.append(arg_max_pos_dev)
                    else:
                        max_deviation_template.append(arg_max_pos_dev + len(klc.statistics_periodic_interleaved[0]))
                else:
                    max_deviation.append(max_neg_dev)
                    if in_interl:
                        max_deviation_template.append(arg_max_neg_dev)
                    else:
                        max_deviation_template.append(arg_max_neg_dev + len(klc.statistics_periodic_interleaved[0]))

            if np.abs(max_deviation[0]) > np.abs(max_deviation[1]):
                max_coords_deviation_periodic.append(max_deviation[0])
                max_coords_template_periodic.append(max_deviation_template[0])
            else:
                max_coords_deviation_periodic.append(max_deviation[1])
                max_coords_template_periodic.append(max_deviation_template[1])
            if save_klcs:
                klcs.append(copy.deepcopy(klc))
        if save_klcs:
            self.klc_centroid_x = klcs[0]
            self.klc_centroid_y = klcs[1]
        self.centroid_snrs_ra_dec = max_coords_deviation_periodic
        self.centroid_templates_ra_dec = max_coords_template_periodic

    def load_centroid_near_transit_pieces(self, time_wing_days=5*tm.Durations[-1]):
        transit_times = self.summed_transit_times
        seasons = self.transit_seasons
        if transit_times is not None:
            orig_quarters = af.get_orig_quarters(transit_times)
            seasons = np.array(orig_quarters) % 4

        centr_coords_snr_peaks = []
        centr_coords_snr_near_transit_pieces = []
        centr_coords_times_near_transit_pieces = []

        for temp, filename_postfix in [[self.centroid_templates_ra_dec[0], locs.Connected_stats_centroid_x_filename_postfix],
                                       [self.centroid_templates_ra_dec[1], locs.Connected_stats_centroid_y_filename_postfix]]:

            (time_full, statistics_full, stat_masks_full, stat_vars_full, big_hole_masks_full,
             indices_of_templates) = af.get_connected_stats_from_file(
                file_path=None, output_location=self.output_location, target_id=self.target_id,
                filename_postfix=filename_postfix, return_only_stats=True)

            if transit_times is None:
                transit_times = ut.get_transit_times(first_transit_time=self.best_ftt, period=self.best_per,
                                                     max_time=np.nanmax(time_full))
                orig_quarters = af.get_orig_quarters(transit_times)
                seasons = np.array(orig_quarters) % 4

            # Converting to units of SNR:
            statistics_full[stat_vars_full > 0] = statistics_full[stat_vars_full > 0] / (stat_vars_full[stat_vars_full > 0]) ** 0.5

            if indices_of_templates[temp][-1] - indices_of_templates[temp][0] > 1:
                statistic_full = np.nanmin(statistics_full[indices_of_templates[temp][0]: indices_of_templates[temp][-1]], axis=0)
                # -> minimizing over subbing shifts to find the worst
            else:
                statistic_full = statistics_full[indices_of_templates[temp][0]]

            # Since the centroid can be rather bad, we renormalize by the outlier-free total variance
            statistic_full = statistic_full / ut.outlier_free_std(statistic_full, n_iterations=1, nsig=3, renormalize=True)

            times_near_transit_pieces, near_transit_pieces, snr_peaks = af.get_near_transit_pieces_from_full_time(
                transit_times=transit_times, time_full=time_full, data_full=statistic_full,
                time_wing_days=time_wing_days, peak_is_min=True)

            # snr_peaks = []
            # near_transit_pieces = []
            # times_near_transit_pieces = []
            # for trans_tim in transit_times:
            #     ind_0 = np.nanargmin(np.abs(time_full - (trans_tim - time_wing_days)))
            #     ind_1 = np.nanargmin(np.abs(time_full - (trans_tim + time_wing_days)))
            #     near_transit_pieces.append(statistic_full[ind_0:ind_1])
            #     times_near_transit_pieces.append(time_full[ind_0:ind_1])
            #     snr_peaks.append(np.min(statistic_full[ind_0:ind_1]))  # ! Minimization over broad range!

            centr_coords_snr_peaks.append(snr_peaks)
            centr_coords_snr_near_transit_pieces.append(near_transit_pieces)
            centr_coords_times_near_transit_pieces.append(times_near_transit_pieces)

        # --------------------------------------------------------------------------------------------------------------
        # Getting snr squared by aligning times and summing ra and dec
        time_total = np.arange(-time_wing_days, time_wing_days + 0.9 * tm.dt_lc, tm.dt_lc)
        centr_snr2_near_transit_pieces = []
        for i_trans, transit_time in enumerate(transit_times):
            snr2_piece = np.zeros(len(time_total))
            for i_coor in (0, 1):
                time_piece = centr_coords_times_near_transit_pieces[i_coor][i_trans]
                snr_piece = centr_coords_snr_near_transit_pieces[i_coor][i_trans]
                time_start_ind = np.nanargmin(np.abs(time_total - (time_piece[0] - transit_time)))
                len_add = min(len(time_piece), len(time_total) - time_start_ind)
                snr2_piece[time_start_ind: time_start_ind + len_add] += (snr_piece[:len_add]) ** 2
            centr_snr2_near_transit_pieces.append(snr2_piece)
        return (centr_coords_snr_peaks, centr_coords_snr_near_transit_pieces, centr_coords_times_near_transit_pieces,
                centr_snr2_near_transit_pieces, time_total, seasons)

    def load_centroid_periodic_results(self):
        ftt_center = self.best_ftt
        ftts_coords = []
        stats_2_max_periodic_coords = []
        for temp, filename_postfix in [[self.centroid_templates_ra_dec[0], locs.Periodic_stats_centroid_x_filename_postfix],
                                       [self.centroid_templates_ra_dec[1], locs.Periodic_stats_centroid_y_filename_postfix]]:

            filenames = [ff for ff in os.listdir(self.output_location) if filename_postfix in ff]
            if len(filenames) > 0:
                file = os.path.join(self.output_location, filenames[0])
            else:
                raise FileNotFoundError('No ' + filename_postfix + ' file found in ' + self.output_location)
            (periods, ftts_interleaved, ftts_not_interleaved, statistics_periodic_interleaved,
             statistics_periodic_not_interleaved, _, _) = np.load(file, allow_pickle=True)

            if temp < len(statistics_periodic_interleaved):
                stats_2_max_periodic = np.max((statistics_periodic_interleaved[:, temp]) ** 2, axis=0)
                ftts = ftts_interleaved
            else:
                stats_2_max_periodic = np.max((statistics_periodic_not_interleaved[:,
                                               temp - len(statistics_periodic_interleaved[0])]) ** 2, axis=0)
                ftts = ftts_not_interleaved
            ftts_coords.append(ftts)
            stats_2_max_periodic_coords.append(stats_2_max_periodic)

        # INTERLEAVING MAY BE DIFFERENT, so we interpolate to sum:
        if ftts_coords[0][1] - ftts_coords[0][0] > ftts_coords[0][1] - ftts_coords[0][0]:
            ftts_total = ftts_coords[0]
        else:
            ftts_total = ftts_coords[1]
        stats_2_max_periodic_total = np.sum(
            [np.interp(ftts_total, ftts_coords[i], stats_2_max_periodic_coords[i]) for i in range(len(stats_2_max_periodic_coords))],
            axis=0)
        # -> in the veto, we maximize independently, and then sum, so in the veto the result will be bigger
        return ftts_total, stats_2_max_periodic_total, ftts_coords, stats_2_max_periodic_coords, ftt_center

    def check_transit_times_quality_flags(self):
        """
        Opens the fits data and accesses the quality flag series. looks in the vicinity of every summed transit point
        and records all that was there.
        :return:
        :rtype:
        """
        self.get_transit_times_info(from_real_klc=True, data_location=self.data_location, output_location=self.output_location)
        # times, _, _, kepler_quality_masks, _ = af.get_target_pixel_sky_coords_fun(
        #     target_id=self.target_id, data_location=self.target_pixels_location,
        #     remove_zeroth_quarter=True, remove_quarter_17=True, cut_edge_nans=True)
        times, kepler_quality_masks = af.get_kepler_quality_masks(
            target_id=self.target_id, data_location=self.target_pixels_location,
            remove_zeroth_quarter=True, remove_quarter_17=True, cut_edge_nans=True)
        # times = res[0]
        # kepler_quality_masks = res[3]

        concatenated_time = np.concatenate(times)
        concatenated_kepler_quality_mask = np.concatenate(kepler_quality_masks).astype(bool)

        earth_pointing_times = get_earth_pointing_times()
        quarter_start_times, quarter_end_times = get_quarter_start_end_times()

        badness_of_transits = ['' for _ in range(len(self.summed_transit_times))]
        valid_transit_mask = np.ones(len(self.summed_transit_times), dtype=bool)
        valid_transit_mask[np.isnan(self.summed_raw_stats) | np.isnan(self.summed_vars) | np.isclose(self.summed_vars, 0)] = False
        for i in range(len(valid_transit_mask)):
            if not valid_transit_mask[i]:
                badness_of_transits[i] += 'nan '

        for i, transit_time in enumerate(self.summed_transit_times):
            between_quarters = np.array([((transit_time > quarter_end_times[j] - Timing_error_quarter_edge)
                                          & (transit_time < quarter_start_times[j + 1] + Timing_error_quarter_edge))
                                         for j in range(len(quarter_start_times) - 1)]).any()
            if between_quarters:
                badness_of_transits[i] += 'between_quarters '
                valid_transit_mask[i] = False

            index_in_times = np.searchsorted(concatenated_time, transit_time)
            if ((not concatenated_kepler_quality_mask[index_in_times])
                    or (not concatenated_kepler_quality_mask[max(0, index_in_times - 1)])):
                badness_of_transits[i] += 'kepler_quality '
                valid_transit_mask[i] = False

            if np.min(np.abs(earth_pointing_times - transit_time)) < Timing_error_earth_downlink:
                badness_of_transits[i] += 'downlink '
                valid_transit_mask[i] = False

        self.badness_of_transits = badness_of_transits
        self.valid_transit_mask = valid_transit_mask
        self.n_valid_transits = len(np.where(valid_transit_mask)[0])
        self.n_transits = len(self.summed_raw_stats)
        if self.n_valid_transits < 3 or self.n_valid_transits / self.n_transits < Min_fraction_valid_transits:
            self.too_few_valid_transits = True
        else:
            self.too_few_valid_transits = False

    # noinspection DuplicatedCode
    def check_periodicity_contaminants(self, period_error=Period_error, phase_error=Phase_error,
                                       period_year=Period_year, period_quarter=Period_quarter,
                                       period_error_year=Period_error_year,
                                       period_error_quarter=Period_error_quarter,
                                       period_error_year_harm=Period_error_year_harm,
                                       period_error_quarter_harm=Period_error_quarter_harm):
        # Loading known Kepler periodicities of potentially bright contaminants:
        vil_periods, vil_phases, vil_kmags = get_villanova_relevant_periods_phases_kmags()
        periods_koi, ftts_koi, kid_kois = get_strong_koi_periodicities()
        vil_periods = np.array(vil_periods, dtype=float)
        vil_phases = np.array(vil_phases, dtype=float)
        periods_koi = np.array(periods_koi, dtype=float)
        ftts_koi = np.array(ftts_koi, dtype=float)

        # ################################################################################################
        # FILTERING ALL THE BAD PERIODICITIES
        self.bad_periodicity_reasons = ''

        bad_vil = ((vil_periods > self.best_per - period_error) &
                   (vil_periods < self.best_per + period_error) &
                   (((vil_phases > self.best_ftt - phase_error) &
                     (vil_phases < self.best_ftt + phase_error)) |
                    ((vil_phases + vil_periods > self.best_ftt - phase_error) &  # maybe one transit to the left or to the right was missing
                     (vil_phases + vil_periods < self.best_ftt + phase_error)) |
                    ((vil_phases - vil_periods > self.best_ftt - phase_error) &
                     (vil_phases - vil_periods < self.best_ftt + phase_error))
                    ))
        if bad_vil.any():
            self.bad_periodicity_reasons += 'v '
            self.is_bad_periodicity = True

        bad_kois = ((periods_koi > self.best_per - period_error) &
                    (periods_koi < self.best_per + period_error) &
                    (((ftts_koi > self.best_ftt - phase_error) &
                      (ftts_koi < self.best_ftt + phase_error)) |
                     ((ftts_koi + periods_koi > self.best_ftt - phase_error) &  # maybe one transit to the left or to the right was missing
                      (ftts_koi + periods_koi < self.best_ftt + phase_error)) |
                     ((ftts_koi - periods_koi > self.best_ftt - phase_error) &
                      (ftts_koi - periods_koi < self.best_ftt + phase_error))
                     ))
        if bad_kois.any():
            self.bad_periodicity_reasons += 'k '
            self.is_bad_periodicity = True

        bad_year = (np.abs(period_year - self.best_per) < period_error_year)  # check that period is close to year
        if bad_year:
            self.bad_periodicity_reasons += 'y '
            self.is_bad_periodicity = True

        bad_quarter = ((period_quarter > self.best_per - period_error_quarter) &
                       (period_quarter < self.best_per + period_error_quarter))  # check that period is close to quarter
        if bad_quarter:
            self.bad_periodicity_reasons += 'q '
            self.is_bad_periodicity = True

        # ################################################################################################
        # HARMONICS
        if self.is_bad_periodicity is None:
            bad_vil = False
            bad_year = False
            bad_quarter = False
            bad_harms_vil = ''
            bad_harms_year = ''
            bad_harms_quarter = ''
            for harm in [1 / 4, 1 / 3, 1 / 2, 4/3, 2., 3., 4., 5., 6., 7., 8.]:  # warning, 4/3 was not there before
                bad_vil_1 = ((vil_periods / harm > self.best_per - period_error) &
                             (vil_periods / harm < self.best_per + period_error) &
                             (((vil_phases > self.best_ftt - phase_error) &
                               (vil_phases < self.best_ftt + phase_error)) |
                              ((vil_phases + vil_periods > self.best_ftt - phase_error) &  # maybe one transit was missing
                               (vil_phases + vil_periods < self.best_ftt + phase_error)) |
                              ((vil_phases - vil_periods > self.best_ftt - phase_error) &  # maybe one transit was missing
                               (vil_phases - vil_periods < self.best_ftt + phase_error))
                              )).any()

                bad_vil = bad_vil | bad_vil_1
                if bad_vil_1:
                    bad_harms_vil = bad_harms_vil + '' + str(np.round(harm, 2))
                # Check that period is close to harmonics of year
                bad_year_1 = ((period_year / harm > self.best_per - period_error_year_harm) &
                              (period_year / harm < self.best_per + period_error_year_harm))
                bad_year = bad_year | bad_year_1
                if bad_year_1:
                    bad_harms_year = bad_harms_year + '' + str(np.round(harm, 2))
                # Check that period is close to harmonics of quarter
                bad_quarter_1 = ((period_quarter / harm > self.best_per - period_error_quarter_harm) &
                                 (period_quarter / harm < self.best_per + period_error_quarter_harm))

                bad_quarter = bad_quarter | bad_quarter_1
                if bad_quarter_1:
                    bad_harms_quarter = bad_harms_quarter + '' + str(np.round(harm, 2))

            if bad_vil:
                self.bad_periodicity_reasons += ' vh' + str(bad_harms_vil) + ' '
                self.is_bad_periodicity = True
            if bad_year:
                self.bad_periodicity_reasons += ' yh' + str(bad_harms_year) + ' '
                self.is_bad_periodicity = True
            if bad_quarter:
                self.bad_periodicity_reasons += ' qh' + str(bad_harms_quarter)
                # This one we do not want to make the periodicity bad, but we want to still know if it is there

        if self.is_bad_periodicity is None:
            self.is_bad_periodicity = False

    def get_koi_dict(self, koi_dict_file=locs.Koi_dict_file, ttv_dict_file=locs.Ttv_dict_file):
        """
        Opens the corresponding files and reads info about possible KOI of this target and about TTV if there
        is any. Writes the info as fields of the class
        :param koi_dict_file: The local path to the KOI info dictionary file
        :type koi_dict_file: str
        :param ttv_dict_file: The local path to the TTV info file
        :type ttv_dict_file: str
        """
        if koi_dict_file is None:
            koi_dict_file = self.koi_dict_file
        if ttv_dict_file is None:
            ttv_dict_file = self.ttv_dict_file

        self.is_koi, self.koi_dict, self.ttv_dict = get_koi_ttv_dicts(kepler_id=self.target_id,
                                                                      koi_dict_file=koi_dict_file, ttv_dict_file=ttv_dict_file)

    def check_koi_status(self, koi_dict_file=locs.Koi_dict_file, ttv_dict_file=locs.Ttv_dict_file):
        self.get_koi_dict(koi_dict_file=koi_dict_file, ttv_dict_file=ttv_dict_file)
        n_transits = self.n_transits
        transit_overlap_mask = np.zeros(n_transits).astype(bool)
        koi_status = ''
        if len(self.koi_dict) > 0:
            for ind in range(len(self.koi_dict['koi_disposition'])):
                disp = self.koi_dict['koi_disposition'][ind]
                if disp == 'CONFIRMED':
                    koi_status += 'co'
                elif disp == 'CANDIDATE':
                    koi_status += 'ca'
                elif disp == 'FALSE POSITIVE':
                    koi_status += 'fp'
                else:
                    koi_status += 'na'
                per_str = self.koi_dict['koi_period'][ind]
                per_float = float(per_str)
                ftt_float = float(self.koi_dict['koi_time0bk'][ind])
                if per_str == '':
                    per_str = 'nan'
                else:
                    per_str = str(np.round(float(per_str), 1))

                overlap_1 = ut.check_transit_overlap(period=self.best_per, ftt=self.best_ftt,
                                                     period_ref=per_float, ftt_ref=ftt_float, n_transits=n_transits, timing_error=0.08)
                transit_overlap_mask = transit_overlap_mask | overlap_1

                koi_status += per_str
                koi_status += '_' + str(np.sum(overlap_1.astype(int)))
                koi_status += ' '

        self.koi_status = koi_status
        self.n_overlapping_koi_transits = np.sum(transit_overlap_mask.astype(int))
        self.koi_overlap_mask = transit_overlap_mask

    def check_tce_status(self, kepler_tce_table_file=locs.Kepler_tce_table_file):
        # ################################################################################################################
        # TCEs

        """
        '# COLUMN kepid:          KepID\n',
        '# COLUMN tce_plnt_num:   Planet Number\n',
        '# COLUMN tce_period:     Orbital Period [days]\n',
        '# COLUMN tce_time0bk:    Transit Epoch [BKJD]\n',
        '# COLUMN tce_ror:        Planet-Star Radius Ratio\n',
        '# COLUMN tce_duration:   Transit Duration [hrs]\n',
        '# COLUMN tce_depth:      Transit Depth [ppm]\n',
        '# COLUMN tce_model_snr:  Transit Signal-to-Noise (SNR)\n',
        '# COLUMN tce_prad:       Planetary Radius [Earth radii]\n',
        '# COLUMN tce_max_mult_ev: Multiple Event Statistic (MES)\n',
        '# COLUMN boot_fap:       Bootstrap False Alarm Probability\n',
        '# COLUMN boot_mesthresh: Bootstrap MES Threshold\n'
        """
        self.is_tce, self.tce_dict = get_kepler_tce_dict(kepler_id=self.target_id, kepler_tce_table_file=kepler_tce_table_file)

        tce_overlap_mask = np.zeros(self.n_transits).astype(bool)
        tce_status = ''
        if self.is_tce:
            for i_tce in range(len(self.tce_dict['tce_period'])):
                per_str = self.tce_dict['tce_period'][i_tce]
                if per_str == '':
                    per_str = 'nan'
                    per_float = np.nan
                    ftt_float = np.nan
                else:
                    per_float = float(per_str)
                    ftt_float = float(self.tce_dict['tce_time0bk'][i_tce])
                    per_str = str(np.round(float(per_str), 1))

                tce_status += per_str

                if per_float is not np.nan:
                    overlap_1 = ut.check_transit_overlap(period=self.best_per, ftt=self.best_ftt, period_ref=per_float,
                                                         ftt_ref=ftt_float, n_transits=self.n_transits, timing_error=0.08)
                    tce_overlap_mask |= overlap_1
                    tce_status += '_' + str(np.sum(overlap_1.astype(int)))

                tce_status += ' '
        self.kepler_tces = tce_status
        self.kepler_tce_overlap_mask = tce_overlap_mask

    # TODO: make overlap mask with all the bright villanova and koi events to prevent leakage

    def estimate_biased_rp(self):
        if self.summed_raw_stats is None or self.summed_vars is None or self.mean_flux is None:
            self.get_transit_times_info()
        # ATTENTION! It uses the raw statistics after the SES veto, so some transits may be missed, and the estimate may be biased.
        # if self.best_ppms_no_filt is None or self.best_ppm_no_filt is None or self.best_ppms_stds_no_filt is None:
        #     self.get_summed_values_no_filters(nsig_outliers_0_pos=nsig_outliers_0_pos,
        #                                       nsig_outliers_0_neg=nsig_outliers_0_neg)
        if len(np.where(self.valid_transit_mask)[0]) == 0:
            self.mean_transit_amplitude = np.nan
            self.mean_transit_ppm = np.nan
            self.mean_transit_ppm_std = np.nan
            self.transit_amplitudes = np.nan
            self.transit_amplitude_stds = np.nan
            self.transit_normalized_amplitudes = np.nan
            self.transit_normalized_amplitude_stds = np.nan
            self.transit_ppms = np.nan
            self.transit_ppm_stds = np.nan

            self.rp_to_rs = np.nan
            self.rp_to_rs_std = np.nan
            self.rp_to_r_earth = np.nan
            self.rp_to_r_earth_std = np.nan
        else:
            tb = tm.TemplateBank(templates=None, shifts=None, template_index='all', load_essentials=True)
            ti_sh = tb.indices_of_templates[self.best_temp][0]
            templ_max = np.max(np.abs(tb.templates_shifts[ti_sh]))
            (ppm, ppm_std, ppm_variability, ppms, ppms_stds,
             ampl, ampl_std, ampl_variability, ampls, ampls_stds,
             normalized_ampls, normalized_ampls_stds) = af.estimate_ampls_ppms_from_stats(
                summed_raw_stats=self.summed_raw_stats, summed_stat_vars=self.summed_vars, templ_max=templ_max,
                flux_trend=self.mean_flux, trends_std=None, valid_transit_mask=self.valid_transit_mask)

            self.mean_transit_ppm = ppm
            self.mean_transit_ppm_std = ppm_std
            self.mean_ppm_variability = ppm_variability
            self.transit_ppms = ppms
            self.transit_ppm_stds = ppms_stds

            self.mean_transit_amplitude = ampl
            self.transit_amplitudes = ampls
            self.transit_amplitude_stds = ampls_stds
            self.transit_normalized_amplitudes = normalized_ampls
            self.transit_normalized_amplitude_stds = normalized_ampls_stds

            # self.mean_transit_ppm_std = (np.nansum(self.transit_ppm_stds[self.valid_transit_mask] ** 2
            #                                        ) / len(self.transit_ppm_stds[self.valid_transit_mask]) ** 2) ** 0.5
            # self.mean_two_noises_model_ppm_std = ut.get_mean_and_std_of_mean_two_sigmas_model(
            #     self.transit_ppm_stds[self.valid_transit_mask])[1]

        ppm_factor = 1e3
        self.rp_to_rs = - np.sign(self.mean_transit_ppm) * (np.abs(self.mean_transit_ppm)) ** 0.5 / ppm_factor
        # -> Radius estimation is subjected to big error and bias! The template contains depth factor at least,
        # and the depth factor can be different even if the same template triggered. And of course, all the
        # templates have different depth factors for their parameters.
        self.rp_to_rs_std = self.mean_transit_ppm_std / (2 * np.abs(self.mean_transit_ppm) ** 0.5) / ppm_factor
        n_samp = len(self.transit_ppm_stds[self.valid_transit_mask])
        sum_ppm = np.sum(self.transit_ppm_stds[self.valid_transit_mask])
        sum_var_ppm = np.sum(self.transit_ppm_stds[self.valid_transit_mask] ** 2)
        # self.rp_to_rs_std = 1 / (2 * self.rp_to_rs) * (np.sum(self.transit_ppm_stds[self.valid_transit_mask] ** 2)
        #                                                / len(self.transit_ppm_stds[self.valid_transit_mask])) ** 0.5 / 1e3 ???
        self.rp_to_r_earth = self.rp_to_rs * self.stellar_radius * R_sun_R_e
        self.rp_to_r_earth_std = self.rp_to_rs_std * self.stellar_radius * R_sun_R_e
        self.rp_to_r_earth_std_conservative = ((1 / (4 * n_samp) * 1 / sum_ppm * sum_var_ppm) ** 0.5 / ppm_factor
                                               * self.stellar_radius * R_sun_R_e)

    def veto_fold_residuals(self, fn_fold_veto=Fn_fold_veto, ignore_moving_template_shape=True,
                            save_byproducts_to_file=False):
        self.is_bad_fold_residuals = False
        klc = spip.KeplerLightCurve(target_id=self.target_id, get_data=True, data_location=self.data_location,
                                    output_location=self.output_location, m_1_3_r_s=self.m_1_3_r_s,
                                    cut_edge_nans=True, remove_zeroth_quarter=True, remove_quarter_17=True,
                                    simulated_star=False)
        self.n_quarters = len(klc.quarters)
        klc.load_conditioned_data()
        indices_of_templates = tm.Indices_of_templates
        whitened_templates_td = copy.deepcopy(klc.whitened_templates_td)
        dt_days = klc.dt_days
        period_fold = self.best_per
        dt_steps_high_pass = klc.nperseg_psd

        # ------------------------------------------------------------------------------------------------------------
        # FOLDING WHITENED FLUX AND BINNING
        binwid = klc.dt_days  # min(self.templates_durations_hours[temp_number] / 24 / 40, 0.1)
        bins_time, bins_flux, bins_counts, folds_times, folds_fluxes = af.fold_fluxes(
            times=klc.times, fluxes=klc.whitened_fluxes_td, period=period_fold, first_time=self.best_ftt,
            masks=klc.big_hole_masks, binwid=binwid, time_fold=period_fold,
            dt=klc.dt_days, means=None, template=None, return_std=False,
            normalize_by_counts=False)  # np.array(self.klc.templates[self.klc.indices_of_templates[temp_number][0]]))
        flux_normalized_vars = np.zeros(len(bins_flux))
        flux_normalized_vars[bins_counts > 0] = bins_flux[bins_counts > 0] / bins_counts[bins_counts > 0] ** 0.5
        # -> this will have the same variance for all the bins

        # ------------------------------------------------------------------------------------------------------------
        # GETTING THE BEST TEMPLATE INDEX
        # The best template index in the fold may not be the best template index from the search due to different binning,
        # corrections, veto etc. So we have to find the best template index in the folded data.
        n_time_templ = int(np.round(period_fold / dt_days))
        if ignore_moving_template_shape:
            folded_templates_rolled = None
            averaged_templates, time_templates = np.array([af.get_average_whitened_template(
                whitened_templates_td=whitened_templates_td, temp_number=tn, n_time_templ=n_time_templ,
                dt_days=dt_days, indices_of_templates=indices_of_templates)
                for tn in range(len(indices_of_templates))]).transpose(1, 0, 2)
            # Interpolating the best averaged template
            averaged_templates = np.array([np.interp(bins_time, time_templates[i], averaged_templates[i])
                                           for i in range(len(averaged_templates))])
            averaged_templates = np.array([- temp_interp / np.linalg.norm(temp_interp) for temp_interp in averaged_templates])
            del time_templates
            mf_scores, ampl_scores = np.array([af.mf_high_pass(data=flux_normalized_vars, template=temp_interp, template_rolled=None,
                                                               dt=binwid, dt_pass=dt_steps_high_pass * dt_days,
                                                               renorm_psd_drift=True)
                                               for temp_interp in averaged_templates]).transpose(1, 0, 2)
            mf_scores_peaks = np.max(mf_scores, axis=-1)
            rolling_args = np.argmax(mf_scores, axis=-1)
            ampl_scores_peaks = np.array([ampl_scores[i][rolling_arg] for i, rolling_arg in enumerate(rolling_args)])
        else:
            averaged_templates = None
            res = [af.matched_filter_folded_fluxes(
                template_number=template_number, whitened_templates_td=whitened_templates_td, n_time_templ=n_time_templ, binwid=binwid,
                indices_of_templates=indices_of_templates, bins_time=bins_time, bins_counts=bins_counts,
                flux_normalized_vars=flux_normalized_vars, dt_steps_high_pass=dt_steps_high_pass, dt_days=dt_days, renorm_psd_drift=True)
                for template_number in range(len(indices_of_templates))]  # ).transpose(1, 0, 2)

            mf_scores_peaks = np.array([r[0] for r in res])
            ampl_scores_peaks = np.array([r[1] for r in res])
            rolling_args = np.array([r[2] for r in res])
            folded_templates_rolled = np.array([r[3] for r in res])

        rolling_vals_peaks = np.array([rolling_arg if rolling_arg <= len(flux_normalized_vars) / 2
                                       else len(flux_normalized_vars) - rolling_arg for rolling_arg in rolling_args])

        max_dist_from_detected_peak_bins = 5  # how far away the pipeline peak can be from the folding peak for the best template
        if rolling_vals_peaks[np.argmax(mf_scores_peaks)] > max_dist_from_detected_peak_bins:
            self.is_bad_fold_residuals = True  # There is some largest peak in the wrong place
            self.template_and_stat_of_fold_contaminant = (np.argmax(mf_scores_peaks), np.max(mf_scores_peaks))
        self.best_temp_folded = np.argmax(mf_scores_peaks * (rolling_vals_peaks <= max_dist_from_detected_peak_bins).astype(float))
        temp_number = self.best_temp_folded
        amp_best_fit = ampl_scores_peaks[temp_number]
        # ------------------------------------------------------------------------------------------------------------
        # CALCULATING THE MATCHED FILTER SCORE ON THE BINNED WHITENED FLUX
        # Matched-filtering of the folded binned flux with the templates corresponding to their expected values.
        # Takes the original `whitened_templates_td` that were used fot the search, in their original time resolution,
        # averages them over sub-bin shifts, and interpolates them (not integrates, which is in general wrong!)
        # to the binned time `bins_time`. The binned template is normalized by the `bins_counts`, as well as the
        # binned flux `bins_flux`, so that the expectations are consistent, and the variances are uniform.
        # Then, the binned flux `bins_flux` is matched-filtered with the binned and the normalized templates
        # The folded template is proportional to the counts, but the matched filtering should be done
        # on the whitened template, and the standard deviation is proportional to the square root of counts.
        # Important: template multiplied by counts is not shift-invariant because counts are different
        # in every bin. Therefore, convolution cannot be used.
        # ------------------------------------------------------------------------------------------------------------
        # MATCHED_FILTERING OF THE RESIDUALS TO FIND SUSPICIOUS DEVIATIONS
        # THE TEMPLATES DO NOT HAVE TO BE OUR TEMPLATE BANK, THEY CAN BE WHATEVER
        if ignore_moving_template_shape:
            rolling_arg = rolling_args[temp_number]
            # GETTING THE RESIDUALS
            residuals_flux_binned = flux_normalized_vars - amp_best_fit * np.roll(averaged_templates[temp_number], rolling_arg)
            # Renormalizing to account for blueness:
            residuals_flux_binned = af.renorm_noise_high_pass(data=residuals_flux_binned, dt=binwid, dt_pass=dt_steps_high_pass * dt_days)
            mf_scores_resid = np.array([af.mf_high_pass(data=residuals_flux_binned, template=temp_interp, template_rolled=None,
                                                        dt=binwid, dt_pass=dt_steps_high_pass * dt_days,
                                                        renorm_psd_drift=False)[0]
                                        for temp_interp in averaged_templates])
            # -> We do not renormalize the PSD drift here because it will nto make the situation worse for normal triggers,
            # but may make it better for bad triggers. Sometimes it will also make it worse for bad triggers though...
            # The thing is that usually, the data misses flux at the low frequencies where the templates
            # have most power, therefore the incorrectness makes the statistic smaller. Therefore good triggers will not
            # turn to outliers. Bad triggers may be missed now, it is true. But from the other side, there are bad triggers
            # which just have all the fold bad, not only a couple of outliers. And when they are renormalized, they
            # create very good-looking histogram. Their bad places are not outliers, therefore calculating the
            # outlier-free variance does not help it.
        else:
            # GETTING THE RESIDUALS
            residuals_flux_binned = flux_normalized_vars - amp_best_fit * folded_templates_rolled[temp_number]
            # Renormalizing to account for blueness:
            residuals_flux_binned = af.renorm_noise_high_pass(data=residuals_flux_binned, dt=binwid, dt_pass=dt_steps_high_pass * dt_days)
            mf_scores_resid = np.array([af.matched_filter_folded_fluxes(
                template_number=template_number, whitened_templates_td=whitened_templates_td, n_time_templ=n_time_templ, binwid=binwid,
                indices_of_templates=indices_of_templates, bins_time=bins_time, bins_counts=bins_counts,
                flux_normalized_vars=residuals_flux_binned, dt_steps_high_pass=dt_steps_high_pass,
                dt_days=dt_days, renorm_psd_drift=True)[0]
                for template_number in range(len(indices_of_templates))])
        # ------------------------------------------------------------------------------------------------------------
        # Calculating p-values:
        p_val_resid = 1 - (1 - chi2(df=1, scale=1).sf(np.max(residuals_flux_binned ** 2))) ** len(residuals_flux_binned)
        p_val_mf = 1. - (1 - chi2(df=1, scale=1).sf(np.max(mf_scores_resid ** 2))) ** len(mf_scores_resid.flatten())
        self.p_value_fold_residuals = p_val_resid
        self.p_value_mf_fold_residuals = p_val_mf
        self.fn_fold_veto = fn_fold_veto
        if self.p_value_mf_fold_residuals < fn_fold_veto:  # or self.p_value_fold_residuals < fn_fold_veto:
            self.is_bad_fold_residuals = True

        # ------------------------------------------------------------------------------------------------------------
        # Additional: looking at the Fourier transform of the residuals
        # because if there is some very short periodicity, it may be missed by the matched filter, although it should not
        if (np.max(np.abs(np.fft.rfft(residuals_flux_binned))**2) >
                len(residuals_flux_binned) * chi2(df=1).isf(1 / len(residuals_flux_binned) * fn_fold_veto)):
            self.is_bad_fold_residuals_fft = True
            self.is_bad_fold_residuals = True
        else:
            self.is_bad_fold_residuals_fft = False

        # ------------------------------------------------------------------------------------------------------------
        if save_byproducts_to_file:
            file_path = af.get_saving_file_path(target_id=self.target_id, output_location=self.output_location,
                                                filename_postfix=locs.Fold_residuals_flux_binned_filename_postfix,
                                                insert_after_postfix=None,
                                                file_format='npy')
            np.save(file_path, residuals_flux_binned, allow_pickle=True)
            file_path = af.get_saving_file_path(target_id=self.target_id, output_location=self.output_location,
                                                filename_postfix=locs.Mf_scores_residuals_filename_postfix,
                                                insert_after_postfix=None,
                                                file_format='npy')
            np.save(file_path, mf_scores_resid, allow_pickle=True)

    def estimate_transit_duration(self, in_folded_template=True):
        if in_folded_template and self.best_temp_folded is not None:
            template_number = self.best_temp_folded
        else:
            template_number = self.best_temp
        self.transit_duration_hours = tm.Durations[template_number] * 24.  # in hours

    def get_target_gaia_table(self, file_path=Gaia_trigger_file):
        self.gaia_target_table = read_gaia_table_for_target_id(self.target_id, file_path=file_path)

    def get_neighbors_info(self, neighbors_file_path=None, neighbors_file_name=None,
                           neighbors_folder=Neighbors_folder, search_radius_pix=Neighbors_search_radius_pix):
        self.neighbors_table = read_neighbor_info(self.target_id, neighbors_file_path=neighbors_file_path,
                                                  neighbors_file_name=neighbors_file_name, neighbors_folder=neighbors_folder,
                                                  search_radius_pix=search_radius_pix)

    def run_all_checks_exclusive(self, print_message_to_history=True,
                                 pre_filtering_history_targets_file=locs.Pre_filtering_history_targets_file,
                                 save_byproducts_to_file=False):
        self.is_good_pre_filtering = False
        self.get_transit_times_info()
        self.veto_short_periodicity()
        self.check_transit_times_quality_flags()
        self.check_periodicity_contaminants()
        self.check_koi_status()
        self.check_tce_status()
        self.seasonal_test()
        if (not self.too_few_valid_transits) and (not self.is_bad_periodicity) and (not self.is_bad_seasons):
            self.veto_fold_residuals(save_byproducts_to_file=save_byproducts_to_file,
                                     ignore_moving_template_shape=True)  # ignoring moving template shape makes it quick
            if not self.is_bad_fold_residuals:
                self.test_target_pixels_amplitudes(save_byproducts_to_file=save_byproducts_to_file)
                if not self.is_bad_bg_pixels:
                    self.test_centroid_transits_snr(save_byproducts_to_file=save_byproducts_to_file)
                    if not self.is_centroid_bad:
                        self.estimate_transit_duration(in_folded_template=True)
                        self.estimate_biased_rp()
                        self.is_good_pre_filtering = True

        if print_message_to_history:
            # Print message to the history file:
            message = ''
            if self.too_few_valid_transits:
                message += 'too_few_valid_transits '
            elif self.is_bad_periodicity:
                message += 'is_bad_periodicity '
            elif self.is_bad_seasons:
                message += 'not is_bad_seasons '
            elif self.is_bad_fold_residuals:
                message += 'is_bad_fold_residuals '
            elif self.is_bad_bg_pixels:
                message += 'is_bad_bg_pixels '
            elif self.is_centroid_bad:
                message += 'is_centroid_bad '
            elif self.is_good_pre_filtering:
                message += 'is_good_pre_filtering '
            with open(pre_filtering_history_targets_file, 'a') as fil:
                print(self.target_id, message, file=fil)
