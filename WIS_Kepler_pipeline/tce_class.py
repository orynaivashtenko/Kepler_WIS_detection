import pickle
import os
import numpy as np
from sys import getsizeof

import target_class
# import pipeline_parameters as pp
from tce_parameters import *
import auxiliary_functions as af


class KeplerTCE(target_class.KeplerTarget):
    def __init__(self, target_id='003097222', data_location=pp.Data_location, output_location=pp.Output_location,
                 target_pixels_location=Target_pixels_location,
                 load_pickled_attributes=False,
                 gaia_kepler_file_path=Gaia_kepler_file,
                 koi_dict_file=Koi_dict_file,
                 ttv_dict_file=Ttv_dict_file,
                 villanova_dict_file=Villanova_dict_file,
                 overwrite_parameters_by_unpickle=False):
        super().__init__(target_id=target_id, data_location=data_location, output_location=output_location,
                         # mes_candidate=pp.Mes_candidate_search,
                         # p_min_search=pp.P_min_search, p_max_search=pp.P_max_search, p_chunks_search=pp.P_chunks_search,
                         load_pickled_attributes=load_pickled_attributes,
                         get_transit_and_binary_info=True,
                         gaia_kepler_file_path=gaia_kepler_file_path,
                         koi_dict_file=koi_dict_file,
                         ttv_dict_file=ttv_dict_file,
                         villanova_dict_file=villanova_dict_file,
                         overwrite_parameters_by_unpickle=overwrite_parameters_by_unpickle)

        target_pixels_location = af.get_correct_tid_folder(target_id, target_pixels_location)

        self.target_pixels_location = target_pixels_location
        self.star_part_ppms = None
        self.star_part_stds_of_ppms = None
        self.bg_part_ppms = None
        self.bg_part_stds_of_ppms = None
        self.centroid_snrs_transits = None
        self.centroid_stds_transits = None
        self.centroid_templs_transits = None

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
            if isinstance(value, (str, float, int, np.float, np.integer, np.ndarray, list, dict)):
                if (((not hasattr(value, '__len__')) or len(value) < 1000) and getsizeof(value) < 80128) or isinstance(value, str):
                    # size in bytes of array 100*100
                    dict_to_pickle[key] = value

        with open(os.path.join(output_location, self.target_id + '_' + pickle_small_attributes_filename_postfix), 'wb') as pickle_file:
            pickle.dump(dict_to_pickle, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

    def unpickle_small_attributes(self, output_location=None,
                                  pickle_small_attributes_filename_postfix=Pickle_tce_small_attributes_filename_postfix,
                                  unpickle_target_attributes=False,
                                  pickle_target_small_attributes_filename_postfix=pp.Pickle_target_small_attributes_filename_postfix
                                  ):
        if unpickle_target_attributes:
            super().pickle_small_attributes(output_location=output_location,
                                            pickle_small_attributes_filename_postfix=pickle_target_small_attributes_filename_postfix)
        if output_location is None:
            output_location = self.output_location
        with open(os.path.join(output_location, self.target_id + '_' + pickle_small_attributes_filename_postfix), 'rb') as pickle_file:
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
            if isinstance(value, (str, float, int, np.float, np.integer, np.ndarray, list, dict)):
                ok_to_print = True
                if isinstance(value, (str, float, int, np.float, np.integer, np.ndarray, list, dict)):
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

    def get_trigger_star_bg_ppms(self, target_pixels_folder=pp.Target_pixels_location,
                                 number_of_pixels_cut=None, part_of_cumsum_2=0.85, part_of_light=None):
        return af.get_trigger_star_bg_ppms(target_id=self.target_id,
                                           best_first_transit_time=self.best_first_transit_time,
                                           best_per=self.best_per, best_temp=self.best_temp,
                                           target_pixels_folder=target_pixels_folder,
                                           number_of_pixels_cut=number_of_pixels_cut,
                                           part_of_cumsum_2=part_of_cumsum_2, part_of_light=part_of_light)

    def test_target_pixels_amplitudes(self):
        star_part_ppms, star_part_stds_of_ppms, bg_part_ppms, bg_part_stds_of_ppms = self.get_trigger_star_bg_ppms(
                            target_pixels_folder=self.target_pixels_location,
                            number_of_pixels_cut=None, part_of_cumsum_2=Part_of_cumsum_2_targ_pix, part_of_light=None)
        self.star_part_ppms = star_part_ppms
        self.star_part_stds_of_ppms = star_part_stds_of_ppms
        self.bg_part_ppms = bg_part_ppms
        self.bg_part_stds_of_ppms = bg_part_stds_of_ppms

    def find_centroid_transits_snr(self, nsig_outliers_0_pos=5, nsig_outliers_0_neg=15):
        summed_transit_times = self.summed_transit_times
        times, target_pixel_fluxes = af.get_target_pixel_data_fun(self.target_id, data_folder=self.target_pixels_location)
        centroids = af.get_centroid_series(target_pixel_fluxes)
        centroid_x_quarters = [centroids[q][0] for q in range(len(centroids))]
        centroid_y_quarters = [centroids[q][1] for q in range(len(centroids))]

        cent_stats_x, cent_stds_x, cent_templs_x = af.find_snr_at_transit_times(signal_quarters=centroid_x_quarters, time_quarters=times,
                                                                                summed_transit_times=summed_transit_times,
                                                                                nsig_outliers_0_pos=nsig_outliers_0_pos,
                                                                                nsig_outliers_0_neg=nsig_outliers_0_neg)
        cent_stats_y, cent_stds_y, cent_templs_y = af.find_snr_at_transit_times(signal_quarters=centroid_y_quarters, time_quarters=times,
                                                                                summed_transit_times=summed_transit_times,
                                                                                nsig_outliers_0_pos=nsig_outliers_0_pos,
                                                                                nsig_outliers_0_neg=nsig_outliers_0_neg)
        self.centroid_snrs_transits = np.array([[cent_stats_x[i], cent_stats_y[i]] for i in range(len(cent_stats_x))])
        self.centroid_stds_transits = np.array([[cent_stds_x[i], cent_stds_y[i]] for i in range(len(cent_stds_x))])
        self.centroid_templs_transits = np.array([[cent_templs_x[i], cent_templs_y[i]] for i in range(len(cent_templs_x))])
