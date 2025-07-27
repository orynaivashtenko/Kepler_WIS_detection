#!/usr/bin/env python3

import copy
import os

import numpy as np

from template_bank.template_module import Durations

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
from scipy.stats import chi2, norm
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
        self.dt_steps_high_pass = None
        self.dt_days = None

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
        self.gaia_trigger_file = None
        self.neighbors_folder = None

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
        self.injections_period_range = None
        self.slides_period_range = None

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
        self.transit_probability = None
        self.prior_planets_in_period_range_one_star = None
        self.transiting_planets_of_radii_in_per_range_one_star = None
        self.transit_and_range_probability = None
        self.transiting_planets_in_relevant_range = None

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
        self.overlapping_kois = None
        self.overlapping_villanovas = None
        self.neighboring_kois = None

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
        self.transit_duration_hours = None

        # Vetting attributes:
        self.too_few_valid_transits = None
        self.star_pixels_ppms = None
        self.star_pixels_stds_of_ppms = None
        self.star_bg_pixels_inclusion_masks = None
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
        self.p_value_fold_residuals_time = None
        self.p_value_fold_residuals_fft = None
        self.p_value_mf_fold_residuals = None
        self.fn_fold_veto = None  # the parameter used in the veto, not the result of the veto!
        self.is_bad_fold_residuals = None
        self.best_temp_folded = None
        self.duration_folded_hours = None
        self.amp_best_fit_folded = None
        self.template_and_stat_of_fold_contaminant = None
        self.is_bad_fold_residuals_fft = None
        self.veto_fold_parts_results = None
        self.p_value_veto_fold_parts = None  # minimal
        self.veto_fold_parts_amplitudes = None
        self.is_bad_fold_residuals_parts = None
        self.is_bad_fold_parts_time = None
        self.is_bad_fold_parts_fourier = None
        self.fn_parts_veto = None
        self.p_value_veto_fold_parts_fourier = None  # minimal renorm
        self.p_values_veto_fold_parts_fourier = None
        self.n_parts_veto_fold_parts_fourier = None
        self.scores_ampls_even_odd = None
        self.p_value_even_odd_ampl = None
        self.is_bad_even_odd_ampl = None
        self.max_season_count = None
        self.fraction_same_season = None
        self.is_bad_seasons = None
        self.is_good_pre_filtering = None
        # Short periodicity search results attributes
        self.best_short_int_score = None
        self.best_short_stat = None
        self.best_short_per = None
        self.best_short_ftt = None
        self.best_short_temp = None
        self.best_short_gauss_stat = None
        self.is_period_short = None

        # Tables:
        self.neighbors_table = None
        self.gaia_target_table = None

        # Search attributes:
        self.is_koi_planet_masked = None
        self.was_notched = None
        self.failures = None
        self.n_slides = None
        self.is_too_short = None
        self.is_bad_fft = None

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
        if from_real_klc:
            if data_location is None:
                data_location = self.data_location
            if output_location is None:
                output_location = self.output_location
            if template is None:
                template = int(self.best_temp)
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
        self.fraction_same_season = self.max_season_count / len(transit_seasons_list)
        if self.fraction_same_season > threshold_fraction_same_season:
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

        if os.path.exists(file_path):
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
        :type output_location:
        :return:
        """
        if output_location is None:
            output_location = self.output_location
        dict_to_pickle = {}
        for key in self.__dict__:
            value = self.__dict__[key]
            # We do not want to pickle klc light curve or whatever else.
            # Also, we do not want to pickle None because when we unpickle, we may overwrite not None by None
            if isinstance(value, (str, float, int, bool, np.bool_, np.float32, np.float64, np.complex64, np.complex128,
                                  np.integer, np.ndarray, list, dict, tuple)):
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

        file_path_unpickling = af.get_saving_file_path(output_location=output_location, target_id=self.target_id,
                                                       filename_postfix=pickle_tce_small_attributes_filename_postfix,
                                                       file_format='pickle')
        if os.path.exists(file_path_unpickling):
            with open(file_path_unpickling, 'rb') as pickle_file:
                dict1 = pickle.load(pickle_file)
                self.__dict__.update(dict1.items())

        if not overwrite_location_parameters:
            for key in Location_attributes + ['target_pixels_location']:
                if key in self.__dict__.keys():
                    self.__dict__[key] = dict_of_old_locations[key]

        if self.best_temp is not None:
            self.best_temp = int(self.best_temp)

    def pickle_all_attributes(self, output_location=None,
                              pickle_all_attributes_filename_postfix=Pickle_tce_all_attributes_filename_postfix):
        """
        Pickles all attributes which are of type str, float, int, np.float, np.integer, np.ndarray, list, dict
        Not really used because if we have attributes like klc, we do not want to pickle them, it will take too much space.

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
                ok_to_print = True
                value = self.__dict__[key]
                if isinstance(value, (str, float, int, bool, np.bool_, np.float32, np.float64, np.complex64, np.complex128,
                                      np.integer, np.ndarray, list, dict)) or value is None:
                    if value is not None:
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
        # WARNING: there is a method with the sma name in target_class, and it is doing something somewhat different!
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
                self.best_temp = int(return_array[spip.Best_per_results_index_template][ind_max])
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
        # if self.best_short_int_score > self.best_int_score:
        #     self.is_period_short = True
        # else:
        #     self.is_period_short = False
        # if self.best_per < minimal_period:
        #     self.is_period_short = True
        # else:  # TODO! THIS IS A BUG!!!! IT CHANGES ALL THEM TO TRUE EVEN IF THE SHORT PER SEARCH VETTED THEMMMMMMMMM
        #     self.is_period_short = False
        if (self.best_short_int_score > self.best_int_score) or (self.best_per < minimal_period):
            self.is_period_short = True
        else:
            self.is_period_short = False

    def get_trigger_star_bg_ppms(self, target_pixels_folder=None,
                                 part_of_flux_peak_threshold=Part_of_flux_peak_threshold,
                                 number_of_pixels_cut=Number_of_pixels_cut, part_of_cumsum_2=Part_of_cumsum_2_targ_pix,
                                 part_of_light=Part_of_light, period=None, template_index=None,  # first_transit_time=None,
                                 nsig_outliers_0_neg=Nsig_outliers_0_neg_tests,
                                 nsig_outliers_0_pos=Nsig_outliers_0_pos_tests,
                                 save_klcs=False, save_data_to_file=False):
        if target_pixels_folder is None:
            target_pixels_folder = self.target_pixels_location
        if period is None:
            period = self.best_per
        if template_index is None:
            template_index = int(self.best_temp)
        # if first_transit_time is None:
        #     first_transit_time = self.best_ftt

        target_pixels_folder = af.get_correct_tid_folder(target_id=self.target_id, location=target_pixels_folder)
        pixel_times, target_pixel_fluxes = af.get_target_pixel_data_fun(self.target_id, data_folder=target_pixels_folder,
                                                                        remove_zeroth_quarter=True,
                                                                        remove_quarter_17=True)
        star_part_sap_fluxes, bg_part_sap_fluxes, inclusion_masks = af.get_splitted_saps_from_pixels(
            target_pixel_fluxes, part_of_peak=part_of_flux_peak_threshold, number_of_pixels_cut=number_of_pixels_cut,
            part_of_cumsum_2=part_of_cumsum_2, part_of_light=part_of_light, return_inclusion_masks=True)
        self.star_bg_pixels_inclusion_masks = inclusion_masks
        filename_postfixes = [[Connected_stats_star_pixels_filename_postfix, Periodic_stats_star_pixels_filename_postfix],
                              [Connected_stats_bg_pixels_filename_postfix, Periodic_stats_bg_pixels_filename_postfix]]

        ppms_and_stds_star_and_bg = []
        klcs = []
        for i_flux, fluxes in enumerate([star_part_sap_fluxes, bg_part_sap_fluxes]):
            # making the lightcurve out of this whatever data and doing search:
            klc = spip.get_minimal_processed_folded_klc(target_id=self.target_id,
                                                        search_periodicity=True, get_data=False,  # target_id is needed for saving name
                                                        times=pixel_times, fluxes=copy.deepcopy(fluxes),
                                                        period=period, delta_period=0.2, template_index=template_index,
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
                # is_interleaved = True
                # phases = klc._phases_interleaved
                time = klc.time_interleaved
                # statistics_periodic = klc.statistics_periodic_interleaved
                statistics = klc.statistics_interleaved
                stat_vars = klc.stat_vars_interleaved
            else:
                # is_interleaved = False
                # phases = klc._phases_not_interleaved
                time = klc.time_not_interleaved
                # statistics_periodic = klc.statistics_periodic_not_interleaved
                statistics = klc.statistics_not_interleaved
                stat_vars = klc.stat_vars_not_interleaved

            summed_transit_points = [np.nanargmin(np.abs(time - self.summed_transit_times[i]))
                                     for i in range(len(self.summed_transit_times))]
            # # ---------------------------------------------------------------------------------------------------------------
            # # The following was used before to give some minimixation freedom, but it may lead to incorrect number of transits!
            # # ---------------------------------------------------------------------------------------------------------------
            # # Finding  correlation size:
            # period_index_left, period_index_right = klc.get_influence_period_indices(period=period, part_snr_2=0.5)
            # phase_index_left, phase_index_right = klc.get_influence_phase_indices(first_transit_time=first_transit_time,
            #                                                                       in_interleaved=is_interleaved, part_snr_2=0.3)
            #
            # # Allow the actual minimum to be a little bit shifted:
            # # Is it okay that it is independent in star and bg? I think so because maybe the actual minimum in the BG was a bit off,
            # # and the star caught it in a different time due to its processed. We are interested in what happened in
            # # the background in the worst case.
            # # Redefining the best period index with a small freedom:
            # period_index = period_index_left + np.argmin(np.nanmin(statistics_periodic[period_index_left:period_index_right, 0,
            #                                                        phase_index_left:phase_index_right], axis=-1))
            # # Warning: we are taking minima here assuming that the statistics are not squared,
            # # and that the deviation is actually negative.
            # # redefining the best phase index with a small freedom:
            # phase_index = phase_index_left + np.argmin(statistics_periodic[period_index][0][
            #                                            phase_index_left:phase_index_right])
            # # -> allow the actual minimum to be a little bit shifted
            # # !!! This may lead to different number of transit times!
            # period_new = klc.periods[period_index]
            # first_transit_time_new = phases[phase_index] + klc.tmin
            # expected_transit_times = ut.get_transit_times(first_transit_time=first_transit_time_new, period=period_new, max_time=klc.tmax)
            # # -> I am not sure tmin does not miss some nan in the beginning
            # summed_transit_points = [np.nanargmin(np.abs(time - expected_transit_times[i])) for i in range(len(expected_transit_times))]
            # # summed_transit_times = np.array(time[summed_transit_points])
            #
            # # ---------------------------------------------------------------------------------------------------------------

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
                                      part_of_flux_peak_threshold=Part_of_flux_peak_threshold,
                                      number_of_pixels_cut=Number_of_pixels_cut, part_of_cumsum_2=Part_of_cumsum_2_targ_pix,
                                      part_of_light=Part_of_light, save_byproducts_to_file=False):
        if (self.star_pixels_ppms is None or self.star_pixels_stds_of_ppms is None
                or self.bg_pixels_ppms is None or self.bg_pixels_stds_of_ppms is None):
            self.get_trigger_star_bg_ppms(
                target_pixels_folder=self.target_pixels_location,
                part_of_flux_peak_threshold=part_of_flux_peak_threshold,
                number_of_pixels_cut=number_of_pixels_cut,
                part_of_cumsum_2=part_of_cumsum_2, part_of_light=part_of_light,
                save_data_to_file=save_byproducts_to_file)

        # First, see what happened in the transits because they can happen to be nans if the transit in one of them is too deep:
        self.star_valid_transit_mask = (~np.isnan(self.star_pixels_ppms)) & (~np.isnan(self.star_pixels_stds_of_ppms)
                                                                             & (self.star_pixels_stds_of_ppms > 0))
        self.bg_valid_transit_mask = (~np.isnan(self.bg_pixels_ppms)) & (~np.isnan(self.bg_pixels_stds_of_ppms)
                                                                         & (self.bg_pixels_stds_of_ppms > 0))
        valid_star_bg_transit_mask = self.star_valid_transit_mask & self.bg_valid_transit_mask
        if len(np.where(valid_star_bg_transit_mask)[0]) < 1:
            self.is_bad_bg_pixels = True
            self.p_value_mean_bg_pixels = np.nan
            self.p_value_worst_bg_pixels = np.nan
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
            else:
                self.p_value_worst_bg_pixels = 1.

    def test_centroid_transits_snr(self, threshold_snr_2=Threshold_centroid_snr_2,
                                   save_byproducts_to_file=False):
        if not hasattr(self, 'centroid_snrs_ra_dec') or self.centroid_snrs_ra_dec is None:
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
        # max_coords_deviation_periodic = []
        # max_coords_template_periodic = []
        klcs = []
        statistics_periodic_interleaved_coords = []
        statistics_periodic_not_interleaved_coords = []
        for i_flux, fluxes in enumerate([centroid_x_quarters, centroid_y_quarters]):
            # making the lightcurve out of this whatever data and doing search:
            klc = spip.get_minimal_processed_folded_klc(target_id=self.target_id, get_data=False,  # target_id is needed for saving name
                                                        times=times, fluxes=fluxes,
                                                        period=period, delta_period=0.2,
                                                        template_index='all', delete_fulls=False,
                                                        nsig_outliers_0_neg=nsig_outliers_0_neg,
                                                        nsig_outliers_0_pos=nsig_outliers_0_pos,
                                                        data_location=None, output_location=None)
            if save_data_to_file:
                klc.save_connected_stats(filename_postfix=filename_postfixes[i_flux][0],
                                         output_location=self.output_location)
                klc.save_statistics_periodic(output_location=self.output_location,
                                             filename_postfix=filename_postfixes[i_flux][1], specify_periods=True)

            statistics_periodic_interleaved_coords.append(klc.statistics_periodic_interleaved)
            statistics_periodic_not_interleaved_coords.append(klc.statistics_periodic_not_interleaved)
            if save_klcs:
                klcs.append(copy.deepcopy(klc))
        if save_klcs:
            self.klc_centroid_x = klcs[0]
            self.klc_centroid_y = klcs[1]

        # Finding the maxima: selecting trigger-adjacent chunks, maximizing every coordinate separately over the
        # template, then summing snr squared, and then maximizing over period and phase

        # Finding  correlation size:
        per_ind_left, per_ind_right = klc.get_influence_period_indices(period=period, part_snr_2=0.5)
        phase_ind_left, phase_ind_right = klc.get_influence_phase_indices(first_transit_time=ftt, in_interleaved=True,
                                                                          part_snr_2=0.2)
        snr_dev_tot_max_interl, max_temps_interl, values_at_max_interl = self._get_maximization_of_periodic_stat(
            stats_selected_periodic_coords=[stats[per_ind_left:per_ind_right, :, phase_ind_left:phase_ind_right]
                                            for stats in statistics_periodic_interleaved_coords])
        phase_ind_left, phase_ind_right = klc.get_influence_phase_indices(first_transit_time=ftt, in_interleaved=False,
                                                                          part_snr_2=0.2)
        snr_dev_tot_max_not_interl, max_temps_not_interl, values_at_max_not_interl = self._get_maximization_of_periodic_stat(
            stats_selected_periodic_coords=[stats[per_ind_left:per_ind_right, :, phase_ind_left:phase_ind_right]
                                            for stats in statistics_periodic_not_interleaved_coords])
        if snr_dev_tot_max_interl > snr_dev_tot_max_not_interl:
            self.centroid_snrs_ra_dec = [values_at_max_interl[0], values_at_max_interl[1]]
            self.centroid_templates_ra_dec = max_temps_interl
        else:
            self.centroid_snrs_ra_dec = [values_at_max_not_interl[0], values_at_max_not_interl[1]]
            self.centroid_templates_ra_dec = [t + len(statistics_periodic_interleaved_coords[0][0]) for t in max_temps_not_interl]

    @staticmethod
    def _get_maximization_of_periodic_stat(stats_selected_periodic_coords):
        """  stats_selected_periodic_coords = [stats[per_ind_left:per_ind_right, :, phase_ind_left:phase_ind_right]
                              for stats in stats_periodic_coords]"""
        snr2_devs_periodic = [stats**2 for stats in stats_selected_periodic_coords]
        for i in range(len(snr2_devs_periodic)):
            snr2_devs_periodic[i][np.isnan(snr2_devs_periodic[i])] = - np.inf
        # -> if something is full of nans, we replace it with minus inf for the ease of computations
        # In this case, it will have zero maximal template index, we define it to be so:
        max_temps_per_point = [np.nanargmax(stats, axis=1) for stats in snr2_devs_periodic]
        snr2_devs_periodic_max_over_temps = [np.nanmax(stats, axis=1) for stats in snr2_devs_periodic]
        # Now we replace back minus infs with nans:
        for i in range(len(snr2_devs_periodic_max_over_temps)):
            snr2_devs_periodic_max_over_temps[i][snr2_devs_periodic_max_over_temps[i] == -np.inf] = np.nan
        snr2_devs_max_tot = snr2_devs_periodic_max_over_temps[0] + snr2_devs_periodic_max_over_temps[1]
        p_ind_max = np.nanargmax(np.nanmax(snr2_devs_max_tot, axis=-1))
        ph_ind_max = np.nanargmax(snr2_devs_max_tot[p_ind_max], axis=-1)
        snr_dev_tot_max = np.nanmax(snr2_devs_max_tot)
        max_temps = [max_temps_per_point[i][p_ind_max, ph_ind_max] for i in range(2)]
        values_at_max = [stats_selected_periodic_coords[i][p_ind_max, max_temps[i], ph_ind_max] for i in range(2)]
        return snr_dev_tot_max, max_temps, values_at_max

    def load_centroid_near_transit_pieces(self, time_wing_days=5*tm.Durations[-1]):
        transit_times = self.summed_transit_times
        seasons = self.transit_seasons
        if transit_times is not None:
            orig_quarters = af.get_orig_quarters(transit_times)
            seasons = np.array(orig_quarters) % 4

        centr_coords_snr_peaks = []
        centr_coords_snr_near_transit_pieces = []
        centr_coords_times_near_transit_pieces = []
        centr_coords_masks_near_transit_pieces = []

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

            # Converting to units of SNR and minimizing over the template sub-bin shifts
            statistic_full, stat_mask_full = af.get_shift_reduced_full_stat_snr(
                statistics_full=statistics_full, stat_vars_full=stat_vars_full, stat_masks_full=stat_masks_full, return_mask=True,
                temp_index=temp, indices_of_templates=indices_of_templates, shift_reduction='minimize', renormalize=True)
            # -> Since the centroid can be rather bad, we renormalize by the outlier-free total variance

            times_near_transit_pieces, near_transit_pieces, mask_near_transit_pieces, snr_peaks = (
                af.get_near_transit_pieces_from_full_time(
                    transit_times=transit_times, time_full=time_full, data_full=statistic_full,
                    mask_full=stat_mask_full, time_wing_days=time_wing_days, peak_is_min=True, return_mask=True))

            centr_coords_snr_peaks.append(snr_peaks)
            centr_coords_snr_near_transit_pieces.append(near_transit_pieces)
            centr_coords_times_near_transit_pieces.append(times_near_transit_pieces)
            centr_coords_masks_near_transit_pieces.append(mask_near_transit_pieces)

        # --------------------------------------------------------------------------------------------------------------
        # Getting snr squared by aligning times and summing ra and dec
        time_total = np.arange(-time_wing_days, time_wing_days + 0.9 * tm.dt_lc, tm.dt_lc)
        centr_snr2_near_transit_pieces = []
        for i_trans, transit_time in enumerate(transit_times):
            snr2_piece = np.zeros(len(time_total))
            for i_coor in (0, 1):
                time_piece = centr_coords_times_near_transit_pieces[i_coor][i_trans]
                snr_piece = centr_coords_snr_near_transit_pieces[i_coor][i_trans]
                mask_piece = (centr_coords_masks_near_transit_pieces[i_coor][i_trans]).astype(bool)
                time_start_ind = np.nanargmin(np.abs(time_total - (time_piece[0] - transit_time)))
                len_add = min(len(time_piece), len(time_total) - time_start_ind)
                snr2_addition = (snr_piece[:len_add]) ** 2
                mask_addition = mask_piece[:len_add]
                snr2_addition[np.isnan(snr2_addition)] = 0.
                snr2_addition[~mask_addition] = 0.
                snr2_piece[time_start_ind: time_start_ind + len_add] += snr2_addition
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

            # # #
            if temp < len(statistics_periodic_interleaved[0]):
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
        times, kepler_quality_masks = af.get_kepler_quality_masks_from_tpf(target_id=self.target_id,
                                                                           data_location=self.target_pixels_location,
                                                                           remove_zeroth_quarter=True, remove_quarter_17=True,
                                                                           cut_edge_nans=True)
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
            skip_next = False
            if index_in_times == len(concatenated_time):
                if transit_time - np.nanmax(concatenated_time) > 3 * tm.dt_lc:
                    badness_of_transits[i] += 'out_of_range '
                    valid_transit_mask[i] = False
                    skip_next = True
                else:
                    index_in_times = len(concatenated_time) - 1

            if not skip_next:
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

    def check_neighboring_koi(self, radius_closeness_arcsec=Pixel_scale * Neighbors_search_radius_pix):
        periods_koi, ftts_koi, kid_kois, ras_kois, decs_kois, kepmags_kois = get_strong_koi_periodicities(
            snr_model_trheshold=1, koi_max_sngle_ev=1, koi_kepmag=None, return_coords=True, return_magnitudes=True)
        d_ra = np.abs(ras_kois - self.ra)
        d_dec = np.abs(decs_kois - self.dec)
        dist_sky_degrees = np.sqrt(d_ra ** 2 * np.cos(self.dec / 180 * np.pi) ** 2 + d_dec ** 2)
        inds_sky_close = np.where(dist_sky_degrees * 3600 < radius_closeness_arcsec)[0]
        periods_koi = periods_koi[inds_sky_close]
        ftts_koi = ftts_koi[inds_sky_close]
        kid_kois = kid_kois[inds_sky_close]
        ras_kois = ras_kois[inds_sky_close]
        decs_kois = decs_kois[inds_sky_close]
        kepmags_kois = kepmags_kois[inds_sky_close]
        inds_select = [i for i in range(len(kid_kois)) if kid_kois[i].rjust(9, '0') != self.target_id]
        periods_koi = periods_koi[inds_select]
        ftts_koi = ftts_koi[inds_select]
        kid_kois = kid_kois[inds_select]
        ras_kois = ras_kois[inds_select]
        decs_kois = decs_kois[inds_select]
        kepmags_kois = kepmags_kois[inds_select]
        self.neighboring_kois = {'period': periods_koi, 'ftt': ftts_koi, 'target_id': kid_kois,
                                 'ra': ras_kois, 'dec': decs_kois, 'kepmag': kepmags_kois}

    def check_other_transits_overlap(self, overlap_timing_error=Overlap_timing_error,
                                     radius_closeness_arcsec=Pixel_scale * Overlapping_transits_check_radius_pixels):
        # Loading known Kepler periodicities of bright close potential contaminants:
        periods_koi, ftts_koi, kid_kois, ras_kois, decs_kois = get_strong_koi_periodicities(
            snr_model_trheshold=30, koi_max_sngle_ev=6, koi_kepmag=self.kepmag + 1., return_coords=True)
        d_ra = np.abs(ras_kois - self.ra)
        d_dec = np.abs(decs_kois - self.dec)
        dist_sky_degrees = np.sqrt(d_ra ** 2 * np.cos(self.dec / 180 * np.pi) ** 2 + d_dec ** 2)
        inds_sky_close = np.where(dist_sky_degrees * 3600 < radius_closeness_arcsec)[0]
        periods_koi = periods_koi[inds_sky_close]
        ftts_koi = ftts_koi[inds_sky_close]
        kid_kois = kid_kois[inds_sky_close]
        dist_sky_pixels_koi = (dist_sky_degrees * 3600 / Pixel_scale)[inds_sky_close]

        # ################################################################################################
        # Loading known Villanova periodicities of close potential contaminants:
        villanova_table = get_villanova_table()
        vil_periods = np.array(villanova_table['period'], dtype=float)
        vil_phases = np.array(villanova_table['t0'], dtype=float)
        # vil_kmags = np.array(villanova_table['kepmag'], dtype=float)
        vil_kids = np.array(villanova_table['target_id'], dtype=str)
        vil_ras = np.array(villanova_table['ra'], dtype=float)
        vil_decs = np.array(villanova_table['dec'], dtype=float)
        d_ra = np.abs(vil_ras - self.ra)
        d_dec = np.abs(vil_decs - self.dec)
        dist_sky_degrees = np.sqrt(d_ra ** 2 * np.cos(self.dec / 180 * np.pi) ** 2 + d_dec ** 2)
        inds_sky_close = np.where(dist_sky_degrees * 3600 < radius_closeness_arcsec)[0]
        vil_periods = vil_periods[inds_sky_close]
        vil_phases = vil_phases[inds_sky_close]
        vil_kids = vil_kids[inds_sky_close]
        dist_sky_pixels_vil = (dist_sky_degrees * 3600 / Pixel_scale)[inds_sky_close]

        n_transits = self.n_transits
        overlapping_kois = ['' for _ in range(n_transits)]
        for i in range(len(kid_kois)):
            tid_koi = kid_kois[i].rjust(9, '0')
            per_str = str(periods_koi[i])
            if per_str == '':
                per_float = np.nan
                ftt_float = np.nan
            else:
                per_float = float(per_str)
                ftt_float = float(ftts_koi[i])
            dist_koi_pix = dist_sky_pixels_koi[i]

            overlap_1 = ut.check_transit_overlap(period=self.best_per, ftt=self.best_ftt,
                                                 period_ref=per_float, ftt_ref=ftt_float, n_transits=n_transits,
                                                 timing_error=overlap_timing_error)
            if tid_koi != self.target_id:
                overl_indices = np.where(overlap_1)[0]
                if len(overl_indices) >= 3 or (len(overl_indices) >= 2 and n_transits <= 5):
                    for index in overl_indices:
                        overlapping_kois[index] += str(tid_koi) + '_' + str(
                            int(dist_koi_pix)) + '_' + str(np.round(per_float, 1)) + ' '
        self.overlapping_kois = overlapping_kois

        overlapping_villanovas = ['' for _ in range(n_transits)]
        for i in range(len(vil_periods)):
            per_str = str(vil_periods[i])
            if per_str == '':
                per_float = np.nan
                ftt_float = np.nan
            else:
                per_float = float(per_str)
                ftt_float = float(vil_phases[i])
            dist_vil_pix = dist_sky_pixels_vil[i]

            overlap_1 = ut.check_transit_overlap(period=self.best_per, ftt=self.best_ftt,
                                                 period_ref=per_float, ftt_ref=ftt_float, n_transits=n_transits,
                                                 timing_error=overlap_timing_error)
            overl_indices = np.where(overlap_1)[0]
            if len(overl_indices) >= 3 or (len(overl_indices) >= 2 and n_transits <= 5):
                for index in overl_indices:
                    overlapping_villanovas[index] += str(vil_kids[i]) + '_' + str(
                        int(dist_vil_pix)) + '_' + str(np.round(per_float, 1)) + ' '
        self.overlapping_villanovas = overlapping_villanovas

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

    def check_koi_status(self, koi_dict_file=locs.Koi_dict_file, ttv_dict_file=locs.Ttv_dict_file,
                         overlap_timing_error=Overlap_timing_error):
        self.get_koi_dict(koi_dict_file=koi_dict_file, ttv_dict_file=ttv_dict_file)
        n_transits = self.n_transits
        transit_overlap_mask = np.zeros(n_transits).astype(bool)
        koi_status = ''
        if len(self.koi_dict) > 0:
            for ind in range(len(self.koi_dict['koi_disposition'])):
                disp = self.koi_dict['koi_disposition'][ind]
                if disp == 'CONFIRMED':
                    disp_short = 'co'
                elif disp == 'CANDIDATE':
                    disp_short = 'ca'
                elif disp == 'FALSE POSITIVE':
                    disp_short = 'fp'
                else:
                    disp_short = 'na'
                koi_status += disp_short
                per_str = self.koi_dict['koi_period'][ind]
                if per_str == '':
                    per_str = 'nan'
                    per_float = np.nan
                    ftt_float = np.nan
                else:
                    per_float = float(per_str)
                    per_str = str(np.round(per_float, 1))
                    ftt_float = float(self.koi_dict['koi_time0bk'][ind])

                overlap_1 = ut.check_transit_overlap(period=self.best_per, ftt=self.best_ftt,
                                                     period_ref=per_float, ftt_ref=ftt_float,
                                                     n_transits=n_transits, timing_error=overlap_timing_error)
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
            self.transit_amplitudes = np.nan * np.ones(len(self.summed_transit_times))
            self.transit_amplitude_stds = np.nan * np.ones(len(self.summed_transit_times))
            self.transit_normalized_amplitudes = np.nan * np.ones(len(self.summed_transit_times))
            self.transit_normalized_amplitude_stds = np.nan * np.ones(len(self.summed_transit_times))
            self.transit_ppms = np.nan * np.ones(len(self.summed_transit_times))
            self.transit_ppm_stds = np.nan * np.ones(len(self.summed_transit_times))

            self.rp_to_rs = np.nan
            self.rp_to_rs_std = np.nan
            self.rp_to_r_earth = np.nan
            self.rp_to_r_earth_std = np.nan
            self.rp_to_r_earth_std_conservative = np.nan
        else:
            tb = tm.TemplateBank(templates=None, shifts=None, template_index='all', load_essentials=True)
            ti_sh = tb.indices_of_templates[int(self.best_temp)][0]
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
            self.rp_to_r_earth = self.rp_to_rs * self.stellar_radius * R_sun_R_e
            self.rp_to_r_earth_std = self.rp_to_rs_std * self.stellar_radius * R_sun_R_e
            valid_ppm_stds = self.transit_ppm_stds[self.valid_transit_mask]
            if not hasattr(valid_ppm_stds, '__len__'):
                n_samp = 1
            else:
                n_samp = len(self.transit_ppm_stds[self.valid_transit_mask])
            sum_ppm = np.sum(self.transit_ppm_stds[self.valid_transit_mask])
            sum_var_ppm = np.sum(self.transit_ppm_stds[self.valid_transit_mask] ** 2)
            # self.rp_to_rs_std = 1 / (2 * self.rp_to_rs) * (np.sum(self.transit_ppm_stds[self.valid_transit_mask] ** 2)
            #                                                / len(self.transit_ppm_stds[self.valid_transit_mask])) ** 0.5 / 1e3 ???
            self.rp_to_r_earth_std_conservative = ((1 / (4 * n_samp) * 1 / sum_ppm * sum_var_ppm) ** 0.5 / ppm_factor
                                                   * self.stellar_radius * R_sun_R_e)

    def veto_fold_residuals(self, fn_fold_veto=Fn_fold_veto, ignore_moving_template_shape=True,
                            save_byproducts_to_file=False):
        self.is_bad_fold_residuals = False
        klc = spip.KeplerLightCurve(target_id=self.target_id, get_data=True, data_location=self.data_location,
                                    output_location=self.output_location, m_1_3_r_s=self.m_1_3_r_s,
                                    cut_edge_nans=True, remove_zeroth_quarter=True, remove_quarter_17=True,
                                    simulated_red_star=False, simulated_white_star=False)
        self.n_quarters = len(klc.quarters)
        klc.load_conditioned_data()
        indices_of_templates = tm.Indices_of_templates
        whitened_templates_td = copy.deepcopy(klc.whitened_templates_td)
        dt_days = klc.dt_days
        period_fold = self.best_per
        dt_steps_high_pass = klc.nperseg_psd
        self.dt_steps_high_pass = dt_steps_high_pass
        self.dt_days = dt_days

        # ------------------------------------------------------------------------------------------------------------
        # FOLDING WHITENED FLUX AND BINNING
        binwid = klc.dt_days  # min(self.templates_durations_hours[temp_number] / 24 / 40, 0.1)
        bins_time, bins_flux, bins_counts, folds_times, folds_fluxes = af.fold_fluxes(
            times=klc.times, fluxes=klc.whitened_fluxes_td, period=period_fold, first_time=self.best_ftt,
            masks=klc.big_hole_masks, binwid=binwid, dt=klc.dt_days, means=None,
            template=None, return_std=False, normalize_by_counts=False)
        # np.array(self.klc.templates[self.klc.indices_of_templates[temp_number][0]]))
        flux_normalized_vars = np.zeros(len(bins_flux))
        flux_normalized_vars[bins_counts > 0] = bins_flux[bins_counts > 0] / bins_counts[bins_counts > 0] ** 0.5
        # -> this will have the same variance for all the bins
        # Renormalizing to account for blueness:
        # flux_normalized_vars = af.renorm_noise_by_high_freq_tail(data=flux_normalized_vars, dt=binwid,
        #                                                           dt_pass=dt_steps_high_pass * dt_days)
        flux_normalized_vars = af.filter_data_high_pass(data=flux_normalized_vars, dt=binwid, dt_pass=dt_steps_high_pass * dt_days,
                                                        in_fourier=False, return_cutoff_index=False, renormalize_to_tail=True)
        # -> this renormalization makes sure that the high frequencies behave in a white-noise way, and the low frequencies are
        # removed because we do not expect to have signal there after detrending. The low frequencies
        # will be skipped in the matched filtering because they are not supposed to be meaningful,
        # we do not resolve them.
        # Otherwise, if we plot the time-domain image accounting for all the frequencies, it will be seen as if
        # the template has too much power. However, the matched filter will not see it, because it will zero the low frequencies.

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
            averaged_templates = np.array([af.filter_data_high_pass(
                data=temp, dt=binwid, dt_pass=dt_steps_high_pass * dt_days, in_fourier=False, return_cutoff_index=False,
                renormalize_to_tail=False) for temp in averaged_templates])
            del time_templates
            mf_scores, ampl_scores = np.array([af.mf_high_pass(data=flux_normalized_vars, template=temp_interp, template_rolled=None,
                                                               dt=binwid, dt_pass=dt_steps_high_pass * dt_days,
                                                               renorm_psd_drift=True)
                                               for temp_interp in averaged_templates]).transpose(1, 0, 2)
            mf_scores_peaks = np.max(mf_scores, axis=-1)
            rolling_args = np.argmax(mf_scores, axis=-1)
            ampl_scores_peaks = np.array([ampl_scores[i][rolling_arg] for i, rolling_arg in enumerate(rolling_args)])
        else:
            # Warning: it is better not to use this option!
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
        self.duration_folded_hours = Durations[self.best_temp_folded] * 24.
        temp_number = self.best_temp_folded
        amp_best_fit = ampl_scores_peaks[temp_number]
        self.amp_best_fit_folded = amp_best_fit
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
            wh_template_best = np.roll(averaged_templates[temp_number], rolling_arg)
            residuals_flux_binned = flux_normalized_vars - amp_best_fit * wh_template_best
            # -> this thing has an excess mismatch at the low frequencies because the matched filtering was done at the
            # high frequencies, and the template thinks that there is a continuation at the low frequencies. The data though
            # is blued, so the power at the low frequencies was killed.

            mf_scores_resid = np.array([af.mf_high_pass(data=residuals_flux_binned, template=temp_interp, template_rolled=None,
                                                        dt=binwid, dt_pass=dt_steps_high_pass * dt_days,
                                                        renorm_psd_drift=False)[0]
                                        for temp_interp in averaged_templates])
            # -> We do not renormalize the PSD drift here because it will nto make the situation worse for normal triggers,
            # but may make it better for bad triggers. Sometimes it will also make it worse for bad triggers though...
            # The thing is that usually, the data misses flux at the low frequencies where the templates
            # have most power, therefore the incorrectness makes the statistic smaller. Therefore, good triggers will not
            # turn to outliers. Bad triggers may be missed now, it is true. But from the other side, there are bad triggers
            # which just have all the fold bad, not only a couple of outliers. And when they are renormalized, they
            # create very good-looking histogram. Their bad places are not outliers, therefore calculating the
            # outlier-free variance does not help it.
        else:
            # GETTING THE RESIDUALS
            wh_template_best = folded_templates_rolled[temp_number]
            residuals_flux_binned = flux_normalized_vars - amp_best_fit * wh_template_best
            mf_scores_resid = np.array([af.matched_filter_folded_fluxes(
                template_number=template_number, whitened_templates_td=whitened_templates_td, n_time_templ=n_time_templ, binwid=binwid,
                indices_of_templates=indices_of_templates, bins_time=bins_time, bins_counts=bins_counts,
                flux_normalized_vars=residuals_flux_binned, dt_steps_high_pass=dt_steps_high_pass,
                dt_days=dt_days, renorm_psd_drift=True)[0]
                for template_number in range(len(indices_of_templates))])
        # ------------------------------------------------------------------------------------------------------------
        # Calculating p-values:
        # p_val_resid = 1 - (1 - chi2(df=1, scale=1).sf(np.max(residuals_flux_binned ** 2))) ** len(residuals_flux_binned)
        # We do not want to compare the low frequencies because there data is too blue, so it will not match any template:
        f_flux_high_pass, cutoff_index = af.filter_data_high_pass(data=residuals_flux_binned, dt=binwid,
                                                                  dt_pass=dt_steps_high_pass * dt_days, in_fourier=True,
                                                                  return_cutoff_index=True, renormalize_to_tail=False)
        # ------------------------------------------------------------------------------------------------------------
        # Looking at the real domain of the residuals.
        # We renormalize by the std, although it is low-passed, so it is not expected to have unit variance. We do it
        # because we just want to make sure there are no some weird peaks, and for comparison it is handier to have
        # unit variance. For a more statistically correct detection we have the matched-filtering p-value. We note that
        # only the matched-filtering one and the fourier residuals are used to vet strictly, because they are the ones
        # that make statistical sense.

        chi_values_real = np.fft.irfft(f_flux_high_pass)
        chi_values_real = chi_values_real / ut.outlier_free_std(chi_values_real, nsig=3, n_iterations=2, renormalize=True)
        chi2_values_real = chi_values_real ** 2
        p_val_resid_r = 1 - (1 - chi2(df=1, scale=1).sf(np.max(chi2_values_real))) ** len(chi2_values_real)
        self.p_value_fold_residuals_time = p_val_resid_r
        # ------------------------------------------------------------------------------------------------------------
        # Looking at the Fourier domain of the residuals.
        # Because if there is some very short periodicity, it may be missed by the matched filter, although it should not
        # if (np.max(np.abs(np.fft.rfft(residuals_flux_binned))**2) >
        #         len(residuals_flux_binned) * chi2(df=1).isf(1 / len(residuals_flux_binned) * fn_fold_veto)):
        chi2_values_fft = (np.abs(f_flux_high_pass) ** 2 / len(residuals_flux_binned) * 2)[cutoff_index:]
        p_val_resid_f = 1 - (1 - chi2(df=2, scale=1).sf(np.max(chi2_values_fft))) ** len(chi2_values_fft)
        self.p_value_fold_residuals_fft = p_val_resid_f
        # These things are supposed to be distributed as chi2 with df=2, except for the values lower than the cutoff which are zero
        # ------------------------------------------------------------------------------------------------------------
        # Looking at the matched filter scores
        p_val_mf = 1. - (1 - chi2(df=1, scale=1).sf(np.max(mf_scores_resid ** 2))) ** len(mf_scores_resid.flatten())
        self.p_value_mf_fold_residuals = p_val_mf
        # ------------------------------------------------------------------------------------------------------------
        # Vetting (real-domain residuals not used, see above)
        self.fn_fold_veto = fn_fold_veto
        if self.p_value_mf_fold_residuals < fn_fold_veto:
            self.is_bad_fold_residuals = True
        if self.p_value_fold_residuals_fft < fn_fold_veto:
            self.is_bad_fold_residuals_fft = True
            self.is_bad_fold_residuals = True
        else:
            self.is_bad_fold_residuals_fft = False

        # ------------------------------------------------------------------------------------------------------------
        # Warning: the saving is still done for the fourier-domain residuals because time-domain are meaningless because
        # of the blueness of the data and non-blueness of the template.
        if save_byproducts_to_file:
            file_path = af.get_saving_file_path(target_id=self.target_id, output_location=self.output_location,
                                                filename_postfix=locs.Fold_residuals_flux_fd_chi2_filename_postfix,
                                                insert_after_postfix=None,
                                                file_format='npy')
            np.save(file_path, chi2_values_fft, allow_pickle=True)
            file_path = af.get_saving_file_path(target_id=self.target_id, output_location=self.output_location,
                                                filename_postfix=locs.Mf_scores_residuals_filename_postfix,
                                                insert_after_postfix=None,
                                                file_format='npy')
            np.save(file_path, mf_scores_resid, allow_pickle=True)
            file_path = af.get_saving_file_path(target_id=self.target_id, output_location=self.output_location,
                                                filename_postfix=locs.Folded_binned_flux_template_filename_postfix,
                                                insert_after_postfix=None,
                                                file_format='npy')
            np.save(file_path, np.array([flux_normalized_vars, wh_template_best]), allow_pickle=True)
            file_path = af.get_saving_file_path(target_id=self.target_id, output_location=self.output_location,
                                                filename_postfix=locs.Fold_residuals_flux_td_chi_filename_postfix,
                                                insert_after_postfix=None,
                                                file_format='npy')
            np.save(file_path, chi_values_real, allow_pickle=True)

        # ------------------------------------------------------------------------------------------------------------
        # Veto parts
        self.veto_fold_parts_time(fn_parts_veto=fn_fold_veto)
        self.veto_fold_parts_fourier(fn_parts_veto=Fn_fold_veto)
        if self.is_bad_fold_parts_time or self.is_bad_fold_residuals_fft:
            self.is_bad_fold_residuals_parts = True

    def veto_fold_parts_time(self, fn_parts_veto=Fn_parts_veto, min_part_bins=Min_len_part_bins_veto_parts_fold):
        file_path = af.get_saving_file_path(target_id=self.target_id, output_location=self.output_location,
                                            filename_postfix=locs.Folded_binned_flux_template_filename_postfix,
                                            insert_after_postfix=None,
                                            file_format='npy')
        self.fn_parts_veto = fn_parts_veto
        # flux_normalized = None
        # wh_template_best = None
        if os.path.exists(file_path):
            flux_normalized, wh_template_best = np.load(file_path, allow_pickle=True)
        else:
            raise ValueError('The file with the folded binned flux and the template was not found. Run the `veto_fold_residuals` first.')
        # ------------------------------------------------------------------------------------------------------------
        dt_days = self.dt_days
        if self.dt_days is None:
            dt_days = tm.dt_lc
        binwid = dt_days
        dt_steps_high_pass = self.dt_steps_high_pass
        if dt_steps_high_pass is None:
            dt_steps_high_pass = Nperseg_psd

        temp_duration_bins = int(tm.Durations[self.best_temp_folded] / dt_days)
        # ------------------------------------------------------------------------------------------------------------
        # Veto parts time
        max_n_parts = max(int(temp_duration_bins / min_part_bins), 2)
        max_n_parts = min(max_n_parts, 5)
        self.veto_fold_parts_results = []
        self.veto_fold_parts_amplitudes = []
        # flux_wh_td = np.nan
        # template_wh_td = np.nan
        for n_parts in range(2, max_n_parts + 1):
            total_pval, worst_pval, ampl_full, ampl_std_full, ampls_parts, ampls_stds_parts = af.get_veto_scores_parts_fold_high_pass(
                flux_wh_td=np.array(flux_normalized), template_wh_td=np.array(wh_template_best), n_parts=n_parts,
                dt_steps_high_pass=dt_steps_high_pass, dt_days=dt_days, binwid=binwid, return_amplitudes=True,
                return_vectors=False, min_bins_part=min_part_bins)

            self.veto_fold_parts_results.append([n_parts, total_pval, worst_pval])
            self.veto_fold_parts_amplitudes.append([ampl_full, ampl_std_full, ampls_parts, ampls_stds_parts])
        total_pvals = [self.veto_fold_parts_results[i][1] for i in range(len(self.veto_fold_parts_results))]
        self.p_value_veto_fold_parts = np.min(total_pvals)
        if np.min(total_pvals) < fn_parts_veto:
            self.is_bad_fold_parts_time = True
        else:
            self.is_bad_fold_parts_time = False

    def veto_fold_parts_fourier(self, fn_parts_veto=Fn_parts_veto, wind_len_overlap=64, n_parts_max=3):
        self.fn_parts_veto = fn_parts_veto
        dt_days = self.dt_days
        if dt_days is None:
            dt_days = tm.dt_lc
        binwid = dt_days
        dt_steps_high_pass = self.dt_steps_high_pass
        if dt_steps_high_pass is None:
            dt_steps_high_pass = Nperseg_psd
        # ------------------------------------------------------------------------------------------------------------
        # Getting the folded binned flux and the template:
        file_path = af.get_saving_file_path(target_id=self.target_id, output_location=self.output_location,
                                            filename_postfix=locs.Folded_binned_flux_template_filename_postfix,
                                            insert_after_postfix=None, file_format='npy')
        # flux_normalized = None
        # wh_template_best = None
        if os.path.exists(file_path):
            flux_normalized, wh_template_best = np.load(file_path, allow_pickle=True)
        else:
            raise ValueError(
                'The file with the folded binned flux and the template was not found. Run the `veto_fold_residuals` first.')
        # ------------------------------------------------------------------------------------------------------------
        # Preparing the Fourier-domain high-passed folded flux and template. The high-pass is
        # done because there is no meaningful power in the detrended frequencies, and we do not want the code to think
        # that there is a disagreement between the data and the template there.
        argm_shift = np.argmin(wh_template_best)  # Attention! the template is negative here!
        wh_template_best = np.roll(wh_template_best, -argm_shift)
        flux_normalized = np.roll(flux_normalized, -argm_shift)
        indices_transits = [0]

        flux_wh_fd = af.filter_data_high_pass(data=np.array(flux_normalized), dt=binwid, dt_pass=dt_steps_high_pass * dt_days,
                                              in_fourier=True, return_cutoff_index=False, renormalize_to_tail=False)
        template_wh_fd_full = af.filter_data_high_pass(data=np.array(wh_template_best), dt=binwid, dt_pass=dt_steps_high_pass * dt_days,
                                                       in_fourier=True, return_cutoff_index=False, renormalize_to_tail=False)
        # template_wh_fd_full = template_wh_fd_full / np.linalg.norm(np.fft.irfft(template_wh_fd_full))
        # ------------------------------------------------------------------------------------------------------------
        # Preparing the window for making the template parts in the Fourier domain overlap. We
        # want them to overlap because if they do nt overlap but still sum to the original template, then
        # they need to be too sharp, which means that they will be too long-range in the time-domain. And
        # this we do not want because we want to measure the effect just near the transit, and not some
        # problematic or data-lacking places far away.
        wind_len = wind_len_overlap  # must be even!
        half_wind_len = wind_len // 2
        window = 0.5 * (1 - np.cos(0.5 * 2 * np.pi * np.arange(wind_len) / wind_len))  # Hann window
        window = 1. - window / np.max(window)
        # ------------------------------------------------------------------------------------------------------------
        ind_0 = np.where(np.abs(template_wh_fd_full) > 0)[0][0]  # The beginning of the high-pass
        # Starting with the maximal number of parts and decreasing them to make each one at least as wide
        # as to support the window. The templates which are wide in the time domain, mey not have
        # enough coverage in the Fourier to have more than one part. From the other side, they will have
        # more parts in the time domain.
        n_parts = n_parts_max
        cs = np.cumsum(template_wh_fd_full ** 2)
        chi2_temp_indices = [ind_0] + list(np.searchsorted(cs / cs[-1], (np.arange(n_parts) / n_parts)[1:])) + [
            len(template_wh_fd_full)]
        while ((chi2_temp_indices - np.roll(chi2_temp_indices, 1))[
               1:] < wind_len).any():  # want every piece to be at least to support window
            n_parts -= 1
            cs = np.cumsum(template_wh_fd_full ** 2)
            chi2_temp_indices = [0] + list(np.searchsorted(cs / cs[-1], (np.arange(n_parts) / n_parts)[1:])) + [
                len(template_wh_fd_full)]
        n_parts = len(chi2_temp_indices) - 1
        self.n_parts_veto_fold_parts_fourier = n_parts

        # If the template does not have enough spread to support more than one part, then this veto cannot vet it:
        if n_parts <= 1:
            self.is_bad_fold_parts_fourier = False
            self.p_values_veto_fold_parts_fourier = [np.nan, np.nan]
        else:
            # ------------------------------------------------------------------------------------------------------------
            templates_parts = np.zeros((n_parts, len(template_wh_fd_full)), dtype=np.complex64)  # here it is all in the fourier domain
            i = 0
            templates_parts[i][chi2_temp_indices[i]: chi2_temp_indices[i + 1] + half_wind_len
                               ] = template_wh_fd_full[chi2_temp_indices[i]: chi2_temp_indices[i + 1] + half_wind_len]
            templates_parts[i][chi2_temp_indices[i + 1] - half_wind_len: chi2_temp_indices[i + 1] + half_wind_len] *= window

            for i in range(1, n_parts - 1):
                templates_parts[i][chi2_temp_indices[i] - half_wind_len: chi2_temp_indices[i + 1] + half_wind_len
                                   ] = template_wh_fd_full[chi2_temp_indices[i] - half_wind_len: chi2_temp_indices[i + 1] + half_wind_len]
                templates_parts[i][chi2_temp_indices[i + 1] - half_wind_len: chi2_temp_indices[i + 1] + half_wind_len] *= window
                templates_parts[i][chi2_temp_indices[i] - half_wind_len: chi2_temp_indices[i] + half_wind_len] *= (1 - window)
            i = n_parts - 1
            templates_parts[i][chi2_temp_indices[i] - half_wind_len: chi2_temp_indices[i + 1]
                               ] = template_wh_fd_full[chi2_temp_indices[i] - half_wind_len: chi2_temp_indices[i + 1]]
            templates_parts[i][chi2_temp_indices[i] - half_wind_len: chi2_temp_indices[i] + half_wind_len] *= (1 - window)
            # ------------------------------------------------------------------------------------------------------------
            templates_parts_td = np.fft.irfft(templates_parts, axis=1)
            templ_prod_matrix = np.zeros((n_parts, n_parts))
            for i in range(n_parts):
                for j in range(n_parts):
                    prodct = templates_parts_td[i] @ templates_parts_td[j]
                    templ_prod_matrix[i, j] = prodct
                    if j != i:
                        templ_prod_matrix[j, i] = prodct
            templ_prod_matrix_inv = np.linalg.inv(templ_prod_matrix)
            # ------------------------------------------------------------------------------------------------------------
            stats_parts = np.zeros_like(templates_parts_td)
            stats_parts_renorm = np.zeros_like(templates_parts_td)
            for i in range(n_parts):
                stat_part = np.real(np.fft.irfft(np.conjugate(templates_parts[i]) * flux_wh_fd))
                # since the noise may be too blue, the distribution may not be with unit variance.
                # -> so we force it to have the same variance as it should be according to the norm
                # ampls[i] = stat / (templ_prod_matrix[i, i])
                stat_renorm = (stat_part / ut.outlier_free_std(stat_part, n_iterations=2, nsig=3, renormalize=True)
                               * (templ_prod_matrix[i, i])**0.5)
                stats_parts_renorm[i] = stat_renorm
                stats_parts[i] = stat_part

            stat_full = np.real(np.fft.irfft(np.conjugate(template_wh_fd_full) * flux_wh_fd))
            # ampl_full = stat_full[0] / np.linalg.norm(wh_template_best) ** 2
            # stat_full = stat_full / ut.outlier_free_std(stat_full, n_iterations=2, nsig=3, renormalize=True)
            stat_full_renorm = stat_full / ut.outlier_free_std(stat_full, n_iterations=2, nsig=3, renormalize=True)
            stat_full = stat_full / np.linalg.norm(wh_template_best) ** 2
            # -> the template norm was 1, so no need to multiply by it
            scores_parts = np.array([stats_parts[:, j] @ templ_prod_matrix_inv @ stats_parts[:, j]
                                     for j in range(len(stats_parts[0]))])
            scores_parts_renorm = np.array([stats_parts_renorm[:, j] @ templ_prod_matrix_inv @ stats_parts_renorm[:, j]
                                            for j in range(len(stats_parts_renorm[0]))])
            # -> this has distr chi2(n_parts), and it has var of trace of unity matrix, which is n_parts
            # ------------------------------------------------------------------------------------------------------------
            # scores = scores - stat_full ** 2
            scores = scores_parts[indices_transits] - (stat_full[indices_transits]) ** 2
            scores_renorm = scores_parts_renorm[indices_transits] - (stat_full_renorm[indices_transits]) ** 2
            # -> In reality, it is only one number because we only check one transit. Therefore, we take [0] index further on.
            # We already renormalized the stuff, so the scale should be 1 here.
            p_value = chi2.sf(scores, loc=0, scale=1., df=n_parts - 1)[0]
            p_value_renorm = chi2.sf(scores_renorm, loc=0, scale=1., df=n_parts - 1)[0]
            self.p_values_veto_fold_parts_fourier = [p_value, p_value_renorm]
            self.p_value_veto_fold_parts_fourier = p_value_renorm
            self.is_bad_fold_parts_fourier = p_value_renorm < fn_parts_veto

    def veto_even_odd(self, fn=Fn_fold_veto, save_byproducts_to_file=False):
        klc = spip.KeplerLightCurve(target_id=self.target_id, get_data=True, data_location=self.data_location,
                                    output_location=self.output_location, m_1_3_r_s=self.m_1_3_r_s,
                                    cut_edge_nans=True, remove_zeroth_quarter=True, remove_quarter_17=True,
                                    simulated_red_star=False, simulated_white_star=False)
        self.n_quarters = len(klc.quarters)
        klc.load_conditioned_data()
        indices_of_templates = tm.Indices_of_templates
        whitened_templates_td = copy.deepcopy(klc.whitened_templates_td)
        dt_days = klc.dt_days
        period_fold = 2 * self.best_per
        time_wing_days = self.best_per / 2
        dt_steps_high_pass = klc.nperseg_psd
        self.dt_steps_high_pass = dt_steps_high_pass
        self.dt_days = dt_days

        if np.array([len(klc.whitened_fluxes_td[q]) == klc.support
                     for q in range(len(klc.whitened_fluxes_td))]).all():
            unpad = True
            zero_padding_ends = klc.zero_padding_ends
        else:
            unpad = False
            zero_padding_ends = None

        # ------------------------------------------------------------------------------------------------------------
        even_transit_times = self.summed_transit_times[0::2]
        odd_transit_times = self.summed_transit_times[1::2]
        self.scores_ampls_even_odd = []
        saving_array = []
        for transit_times in [even_transit_times, odd_transit_times]:
            times_near_transit_pieces_wh_flux, wh_fluxes_near_transit_pieces, masks_near_transit_pieces_flux, _ = (
                (af.get_near_transit_pieces_of_whitened_flux(
                    transit_times=transit_times, times=klc.times, whitened_fluxes_td=klc.whitened_fluxes_td,
                    masks=klc.big_hole_masks, time_wing_days=time_wing_days,
                    unpad=unpad, zero_padding_ends=zero_padding_ends, dt_days=klc.dt_days)))

            if len(times_near_transit_pieces_wh_flux) > 0:
                bins_time, bins_flux, bins_counts, folds_times, folds_fluxes, bins_std = af.fold_fluxes(
                    times=copy.deepcopy(times_near_transit_pieces_wh_flux),
                    fluxes=copy.deepcopy(wh_fluxes_near_transit_pieces),
                    period=period_fold,
                    first_time=transit_times[0],
                    masks=copy.deepcopy(masks_near_transit_pieces_flux),
                    binwid=dt_days, dt=dt_days, means=None, template=None, return_std=True,
                    normalize_by_counts=False)
                # Cutting the bins to the time wing
                selection = np.abs(bins_time) < time_wing_days
                bins_time = bins_time[selection]
                bins_flux = bins_flux[selection]
                bins_counts = bins_counts[selection]
                # bins_std = bins_std[selection]
                flux_normalized_vars = np.zeros(len(bins_flux))
                flux_normalized_vars[bins_counts > 0] = bins_flux[bins_counts > 0] / bins_counts[bins_counts > 0] ** 0.5

                n_time_templ = int(np.round(period_fold / dt_days))
                averaged_template, time_template = af.get_average_whitened_template(
                    whitened_templates_td=whitened_templates_td, temp_number=self.best_temp_folded, n_time_templ=n_time_templ,
                    dt_days=dt_days, indices_of_templates=indices_of_templates)
                # Interpolating the best averaged template
                averaged_template = np.interp(bins_time, time_template, averaged_template)
                averaged_template = - averaged_template / np.linalg.norm(averaged_template)
                averaged_template = af.filter_data_high_pass(averaged_template, dt=dt_days, dt_pass=dt_steps_high_pass * dt_days,
                                                             in_fourier=False, return_cutoff_index=False, renormalize_to_tail=False)
                # See `veto_fold_residuals` for explanation
                flux_normalized_vars = af.filter_data_high_pass(data=flux_normalized_vars, dt=dt_days, dt_pass=dt_steps_high_pass * dt_days,
                                                                in_fourier=False, return_cutoff_index=False, renormalize_to_tail=True)
                mf_scores, ampl_scores, denom = af.mf_high_pass(data=flux_normalized_vars, template=averaged_template,
                                                                template_rolled=None, dt=dt_days, dt_pass=dt_steps_high_pass * dt_days,
                                                                renorm_psd_drift=True, return_denom=True)
                mf_score_peak = np.max(mf_scores, axis=-1)
                rolling_arg = np.argmax(mf_scores, axis=-1)
                ampl_score_peak = ampl_scores[rolling_arg]
                # ampl_score_std = ampl_score_peak / mf_score_peak
                ampl_score_std = 1 / denom
                self.scores_ampls_even_odd.append([mf_score_peak, ampl_score_peak, ampl_score_std, rolling_arg])
                saving_array.append([flux_normalized_vars, averaged_template * ampl_score_peak, bins_time, bins_counts])
            else:
                self.scores_ampls_even_odd.append([np.nan, np.nan, np.nan, 0])
        if save_byproducts_to_file:
            file_path = af.get_saving_file_path(target_id=self.target_id, output_location=self.output_location,
                                                filename_postfix=locs.Folded_binned_wh_flux_model_even_odd_filename_postfix,
                                                insert_after_postfix=None, file_format='npy')
            np.save(file_path, np.array(saving_array), allow_pickle=True)

        if (not np.isnan(self.scores_ampls_even_odd[0][0])) and (not np.isnan(self.scores_ampls_even_odd[1][0])):
            d_ampl = np.abs(self.scores_ampls_even_odd[0][1] - self.scores_ampls_even_odd[1][1])
            d_ampl_std = (self.scores_ampls_even_odd[0][2] ** 2 + self.scores_ampls_even_odd[1][2] ** 2) ** 0.5
            score_ampl_diff = d_ampl / d_ampl_std
            self.p_value_even_odd_ampl = norm.sf(score_ampl_diff, scale=1, loc=0) * 2
            if self.p_value_even_odd_ampl < fn:
                self.is_bad_even_odd_ampl = True
            else:
                self.is_bad_even_odd_ampl = False

    def estimate_transit_duration(self, in_folded_template=True):
        if in_folded_template and self.best_temp_folded is not None:
            template_number = int(self.best_temp_folded)
        else:
            template_number = int(self.best_temp)
        self.transit_duration_hours = tm.Durations[template_number] * 24.  # in hours

    def get_target_gaia_table(self, file_path=Gaia_trigger_table_file):
        self.gaia_trigger_file = file_path
        self.gaia_target_table = read_gaia_table_for_target_id(self.target_id, file_path=file_path)

    def get_neighbors_info(self, neighbors_file_path=None, neighbors_file_name=None,
                           neighbors_folder=Neighbors_folder, search_radius_pix=Neighbors_search_radius_pix):
        self.neighbors_folder = neighbors_folder
        self.neighbors_table = read_neighbor_info(self.target_id, neighbors_file_path=neighbors_file_path,
                                                  neighbors_file_name=neighbors_file_name, neighbors_folder=neighbors_folder,
                                                  search_radius_pix=search_radius_pix)

    def run_all_checks_exclusive(self, print_message_to_history=True,
                                 pre_filtering_history_targets_file=locs.Pre_filtering_history_targets_file,
                                 save_byproducts_to_file=True, produce_plots=False, save_plots=False,
                                 plots_output_location=locs.Triggers_plots_location, no_show=False,
                                 update_pickling=True, enforce_all_checks=False, include_veto_short_periodicity=True):
        folding_residuals_test_done = False
        bg_star_pixels_test_done = False
        centroid_test_done = False
        veto_parts_test_done = False
        even_odd_test_done = False
        transit_info_done = False
        enf_all = enforce_all_checks
        self.get_transit_times_info()
        if include_veto_short_periodicity:
            self.veto_short_periodicity()
        else:
            self.is_period_short = False  # TODO change to none; why is it false?
        self.check_transit_times_quality_flags()
        self.check_periodicity_contaminants()
        self.check_other_transits_overlap()
        self.check_koi_status()
        self.check_tce_status()
        self.seasonal_test()
        transits_were_bad = self.too_few_valid_transits or self.is_bad_periodicity or self.is_bad_seasons
        if (not transits_were_bad) or enf_all:
            self.veto_fold_residuals(save_byproducts_to_file=save_byproducts_to_file,
                                     ignore_moving_template_shape=True)  # ignoring moving template shape makes it quick
            folding_residuals_test_done = True
            veto_parts_test_done = True
            if (not self.is_bad_fold_residuals) and (not self.is_bad_fold_residuals_parts) or enf_all:
                self.test_target_pixels_amplitudes(save_byproducts_to_file=save_byproducts_to_file)
                bg_star_pixels_test_done = True
                if (not self.is_bad_bg_pixels) or enf_all:
                    self.test_centroid_transits_snr(save_byproducts_to_file=save_byproducts_to_file)
                    centroid_test_done = True
                    if (not self.is_centroid_bad) or enf_all:
                        self.veto_even_odd(save_byproducts_to_file=save_byproducts_to_file)
                        even_odd_test_done = True
                        if (not self.is_bad_even_odd_ampl) or enf_all:
                            self.estimate_transit_duration(in_folded_template=True)
                            self.estimate_biased_rp()
                            transit_info_done = True
        self.is_good_pre_filtering = False
        if ((not transits_were_bad)
                and (not self.is_period_short)  # Warning: this was added on Nov 24 2024, after search run 12
                and (not self.is_bad_fold_residuals)
                and (not self.is_bad_bg_pixels)
                and (not self.is_bad_fold_residuals_parts)
                and (not self.is_bad_even_odd_ampl)):
            self.is_good_pre_filtering = True

        if update_pickling:
            self.pickle_small_attributes()
            self.print_dict_file()

        if produce_plots:
            from plotting_functions import plot_all_postprocessing
            plot_all_postprocessing(self, save_fig=save_plots, no_show=no_show,
                                    plots_output_location=plots_output_location,
                                    fluxes_and_stats=True,
                                    folding_residuals=folding_residuals_test_done,
                                    bg_star_pixels=bg_star_pixels_test_done,
                                    centroid_plots=centroid_test_done,
                                    veto_parts_plot=veto_parts_test_done,
                                    even_odd=even_odd_test_done,
                                    transit_info=transit_info_done)

        if print_message_to_history:
            # Print message to the history file:
            message = ''
            if self.is_good_pre_filtering:
                message += 'is_good_pre_filtering '
            else:
                message += 'not is_good_pre_filtering '
            if self.too_few_valid_transits:
                message += 'too_few_valid_transits '
            elif self.is_bad_periodicity:
                message += 'is_bad_periodicity '
            elif self.is_period_short:  # Warning: this was added on Nov 24 2024, after search run 12
                message += 'is_period_short '
            elif self.is_bad_seasons:
                message += 'is_bad_seasons '
            elif self.is_bad_fold_residuals:
                message += 'is_bad_fold_residuals '
            elif self.is_bad_fold_residuals_parts:
                if self.is_bad_fold_parts_time:
                    message += 'is_bad_fold_parts_time '
                if self.is_bad_fold_parts_fourier:
                    message += 'is_bad_fold_parts_fourier '
            elif self.is_bad_bg_pixels:
                message += 'is_bad_bg_pixels '
            elif self.is_centroid_bad:
                message += 'is_centroid_bad '
            elif self.is_bad_even_odd_ampl:
                message += 'is_bad_even_odd_ampl '

            with open(pre_filtering_history_targets_file, 'a') as fil:
                print(self.target_id, message, file=fil)
