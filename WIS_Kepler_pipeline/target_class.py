#!/usr/bin/env python3
# import warnings

import os
os.environ["OMP_NUM_THREADS"] = "1"  #
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"  #
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
from sys import getsizeof
from astropy.io import fits
import pickle
import search_pipeline as spip
import template_bank.template_module as tm
from simulations import lightcurve_simulation as sim
import auxiliary_functions as af
import utils as ut
import time
from pipeline_parameters import *  # all the parameters, physical constants, occurrence table
# from file_operations import report_exception
# import warnings
# warnings.filterwarnings("ignore")
np.set_printoptions(threshold=1000, linewidth=10000)

# TODO: talk to Barak about saving one best thing per star. It can be for example that the best thing is wrong, but there is one a bit lower
# which is good. It would make sense to save everything which is above a certain snr!
# Or maybe we can say that for now, we want to find at least something. If we want that there is something in this star, then it is worth
# looking at it closer in the future and save more points
"""

Future improvements:
* make it possible to run multiprocessing from inside the functions
* put indents in the docstring in the description
* learn to use json
* reorganize methods, e.g. not to initialize templates
* maybe put template generation to the init
* figure out what is their utils.import_matplotlib()


How to organize the project:
    * When saving new files and the old ones exist, make an option to rename the old one and keep it or to overwrite it
    * When appending something to the savings file, make sure whether you need to clean it in the beginning.
    * All the parameters should be put to the params file, and they can be modified by params.param1 = value_1 after imported.    
      
                
"""

"""
class P1:
    def __init__(self, **kw):
        self.__dict__.update(kw)
"""

if not os.path.isdir(Data_location):
    os.mkdir(Data_location)

if not os.path.isdir(Output_location):
    os.mkdir(Output_location)


######################################################################################################################################


class KeplerTarget(object):
    star_attributes = ['stellar_radius', 'stellar_mass', 'kepmag', 'phot_g_mean_mag', 'parallax',
                       'gaia_abs_gmag', 'gaia_b_r', 'teff', 'ra', 'dec', 'm_1_3_r_s']

    def __init__(self, target_id='003097222', data_location=Data_location, output_location=Output_location,
                 load_pickled_attributes=False,
                 get_transit_and_binary_info=True,
                 gaia_kepler_file_path=Gaia_kepler_file,
                 koi_dict_file=Koi_dict_file,
                 ttv_dict_file=Ttv_dict_file,
                 villanova_dict_file=Villanova_dict_file,
                 overwrite_parameters_by_unpickle=False,
                 allow_output_folder_creation=True):
        self.target_id = target_id.rjust(9, '0')

        data_location = af.get_correct_tid_folder(target_id, data_location)
        output_location = af.get_correct_tid_folder(target_id, output_location,
                                                    allow_creation=allow_output_folder_creation)

        self.data_location = data_location
        self.output_location = output_location
        self.gaia_kepler_file = gaia_kepler_file_path
        self.koi_dict_file = koi_dict_file
        self.ttv_dict_file = ttv_dict_file
        self.villanova_dict_file = villanova_dict_file

        self.stellar_radius = None
        self.stellar_mass = None
        self.kepmag = None
        self.phot_g_mean_mag = None
        self.parallax = None
        self.phot_g_mean_mag = None
        self.gaia_abs_gmag = None
        self.gaia_b_r = None
        self.teff = None
        self.ra = None
        self.dec = None
        self.ruwe = None
        self.gaia_nconfp = None
        # self_gaia_if_planet = None
        self.gaia_non_single_star_flag = None
        self.gaia_variable_star_flag = None
        self.m_1_3_r_s = None  # M**(1/3)*R[in M_sun and R_sun]

        self.number_of_quarters = None
        self.is_bad_fft = None
        self.was_notched = None
        self.is_koi = None
        self.is_koi_planet_masked = None
        self.koi_dict = None
        self.ttv_dict = None
        self.is_villanova_binary = None
        self.villanova_dict = None

        self.periods_in_chunks = None
        self.per_phases_per_chunk = None  # per_phases_per_chunk
        self.duty_cycle = None
        self.p_min = None  # p_min_search
        self.p_max = None  # p_max_search
        self.klc = None
        self.conditioning_ses_time = None  # time of the main search conditioning and ses in seconds
        self.periodicity_search_time = None  # time of the main run periodicity search (mes) in seconds

        self.templates_durations_hours = tm.Durations * 24.

        self.best_stat = None
        self.best_per = None
        self.best_temp = None
        self.best_ftt = None
        self.best_gauss_stat = None
        self.best_integral_score = None
        self.best_g_stat = None
        self.best_g_per = None
        self.best_g_temp = None
        self.best_g_ftt = None
        self.best_g_gauss_stat = None
        self.best_g_integral_score = None

        self.summed_transit_times = None  # times of values that were summed to produce the best trigger
        self.summed_raw_stats = None  # statistics in units not of snr that were summed to produce the best trigger
        self.summed_trends = None
        self.summed_vars = None
        self.summed_corrections = None
        self.summed_gauss_stats = None

        self.is_threshold_crossing = False
        self.per_wing_injection_and_slides = None
        self.transit_prob_factor = None
        self.search_prob_factor = None
        self.transiting_planets_in_search = None
        self.log10_best_period_edges = None
        self.log2_rp_edges = None
        self.planets_of_radii_in_per_range_one_star = None
        self.injection_par_list = None

        # Injections attributes

        self.per_inj_success = None
        self.a_inj_success = None
        self.inc_inj_success = None
        self.ecc_inj_success = None
        self.w_inj_success = None
        self.rp_inj_success = None
        self.first_transit_time_inj_success = None
        self.retrieved_stat_success = None
        self.retrieved_period_success = None
        self.retrieved_first_transit_time_success = None
        self.retrieved_gauss_stat_success = None
        self.retrieved_integral_score_success = None
        self.retrieved_g_stat_success = None
        self.retrieved_g_period_success = None
        self.retrieved_g_first_transit_time_success = None
        self.retrieved_g_gauss_stat_success = None
        self.retrieved_g_integral_score_success = None

        self.per_inj_bad = None
        self.a_inj_bad = None
        self.inc_inj_bad = None
        self.ecc_inj_bad = None
        self.w_inj_bad = None
        self.rp_inj_bad = None
        self.first_transit_time_inj_bad = None
        self.retrieved_stat_bad = None
        self.retrieved_period_bad = None
        self.retrieved_first_transit_time_bad = None
        self.retrieved_gauss_stat_bad = None
        self.retrieved_integral_score_bad = None
        self.retrieved_g_stat_bad = None
        self.retrieved_g_period_bad = None
        self.retrieved_g_first_transit_time_bad = None
        self.retrieved_g_gauss_stat_bad = None
        self.retrieved_g_integral_score_bad = None

        self.number_of_success_injections = None
        self.number_of_injections = None

        self.stats_slides = None
        self.pers_slides = None
        self.first_times_slides = None
        self.gauss_stats_slides = None
        self.int_scores_slides = None
        self.stats_g_slides = None
        self.pers_g_slides = None
        self.first_times_g_slides = None
        self.gauss_stats_g_slides = None
        self.int_scores_g_slides = None

        self.n_slides = None
        self.bins_stat_histogram = None
        self.bins_stat_histogram_inj = None
        self.bins_stat_histogram_bg = None
        self.distr_stat_h0 = None
        self.distr_stat_inj_all = None
        self.distr_stat_inj_success = None
        self.extrapolation_coefs_stat_h0_hist = None
        self.prob_density_h0_trigger = None
        self.false_alarm_prob_trigger = None
        self.prob_density_h1_trigger = None
        self.prob_density_planet_not_noise = None

        self.bins_gauss_stat_histogram = None
        self.bins_gauss_stat_histogram_inj = None
        self.bins_gauss_stat_histogram_bg = None
        self.distr_gauss_stat_h0 = None
        self.distr_gauss_stat_inj_all = None
        self.distr_gauss_stat_inj_success = None

        # Elements of estimations:
        # Non-score-corrected values:
        self.best_raw_stats_no_filt = None
        self.best_stat_vars_no_filt = None
        self.best_trends_no_filt = None
        self.best_ampls_no_filt = None  # amplitudes of different transits
        self.best_ampls_stds_no_filt = None
        self.best_ampl_no_filt = None  # average depth of the transit in parts per million
        self.best_norm_ampls_no_filt = None
        self.best_norm_ampls_stds_no_filt = None
        self.best_ppms_no_filt = None  # depths of different transits in parts per million
        self.best_ppm_no_filt = None  # average amplitude of the transit, coefficient that multiplies the best template
        self.best_ppms_stds_no_filt = None
        self.best_temp_depth_no_filt = None

        self.rp_to_rs = None  # planetary radius in units of stellar radius, estimate from ppm
        self.rp_to_r_earth = None  # planetary radius in units of Earth radius, estimate from ppm
        self.rp_to_rs_std = None  # std due to the dispersion of measurements in diff.transits (not including measurement error)
        self.rp_to_r_earth_std = None

        self.failures = ''  # if something bad happens, we will write it here
        self.errors = ''
        # TODO: this field did not work in the catalog!!!
        self.is_good_for_processing = True

        if load_pickled_attributes:
            self.unpickle_small_attributes()
            if not overwrite_parameters_by_unpickle:
                self.data_location = data_location
                self.output_location = output_location
                self.gaia_kepler_file = gaia_kepler_file_path
                self.koi_dict_file = koi_dict_file
                self.ttv_dict_file = ttv_dict_file
                self.villanova_dict_file = villanova_dict_file

        else:
            if self.gaia_kepler_file is not None:
                self.get_fields_from_gaia(gaia_file_path=self.gaia_kepler_file)
            if get_transit_and_binary_info:
                self.get_koi_and_ttv_fields(koi_dict_file=self.koi_dict_file, ttv_dict_file=self.ttv_dict_file)
                self.get_villanova_fields(villanova_dict_file=self.villanova_dict_file)

        if self.is_villanova_binary:
            self.is_good_for_processing = False
            self.failures += 'villanova; '

    # noinspection DuplicatedCode
    def pickle_small_attributes(self, output_location=None,
                                pickle_small_attributes_filename_postfix=Pickle_target_small_attributes_filename_postfix):
        """
        Pickles all attributes which are of type str, float, int, np.float, np.integer, np.ndarray, list, dict
        TODO: we should avoid pickling too long arrays
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
            if isinstance(value, (str, float, int, np.float32, np.integer, np.ndarray, list, dict)):  # float is np.float64 but not 32!
                if (((not hasattr(value, '__len__')) or len(value) < 1000) and getsizeof(value) < 80128) or isinstance(value, str):
                    # size in bytes of array 100*100
                    dict_to_pickle[key] = value
        file_path = af.get_saving_file_path(output_location=output_location, target_id=self.target_id,
                                            filename_postfix=pickle_small_attributes_filename_postfix, file_format='pickle')
        with open(file_path, 'wb') as pickle_file:
            pickle.dump(dict_to_pickle, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

    def unpickle_small_attributes(self, output_location=None,
                                  pickle_small_attributes_filename_postfix=Pickle_target_small_attributes_filename_postfix):
        if output_location is None:
            output_location = self.output_location
        file_path = af.get_saving_file_path(output_location=output_location, target_id=self.target_id,
                                            filename_postfix=pickle_small_attributes_filename_postfix, file_format='pickle')
        with open(file_path, 'rb') as pickle_file:
            dict1 = pickle.load(pickle_file)
            self.__dict__.update(dict1.items())

    def get_fields_from_gaia(self, gaia_file_path=None):
        if gaia_file_path is None:
            gaia_file_path = self.gaia_kepler_file
        gaia_file = fits.open(gaia_file_path)
        indices = np.where(gaia_file[1].data['kepid'] == int(self.target_id))[0]
        if len(indices) == 0:
            self.failures += ' Target_id not in Gaia;'
            self.is_good_for_processing = False
        else:
            index = indices[0]  # inefficient search of target_id
            self.stellar_radius = gaia_file[1].data['radius'][index]  # in solar radii
            self.stellar_mass = gaia_file[1].data['mass'][index]  # in solar masses
            self.kepmag = gaia_file[1].data['kepmag'][index]  # kepler magnitude
            self.phot_g_mean_mag = gaia_file[1].data['phot_g_mean_mag'][index]
            self.parallax = (gaia_file[1].data['parallax'])[index]  # mas # parallaxes (needed for absolute magnitude)
            self.phot_g_mean_mag = (gaia_file[1].data['phot_g_mean_mag'])[index]  # photometric g magnitudes
            if self.parallax < 0:
                self.failures += ' Parallax value negative;'
                self.gaia_abs_gmag = None
            else:
                self.gaia_abs_gmag = self.phot_g_mean_mag + 5. * (
                        np.log10(self.parallax / 1.e3) + 1.)  # calculating absolute magnitude from Bailer-Jones distances
            self.gaia_b_r = (gaia_file[1].data['bp_rp'])[index]  # color
            self.teff = (gaia_file[1].data['teff'])[index]  # temperature in Kelvin
            self.ra = gaia_file[1].data['ra'][index]  # right ascension, degrees
            self.dec = gaia_file[1].data['dec'][index]  # declination, degrees

            self.ruwe = gaia_file[1].data['ruwe'][index]  # renormalized unit weight error.
            self.gaia_nconfp = gaia_file[1].data['nconfp'][index]  # number of confirmed planets
            # In DR2 ruwe<1.2 were formally considered as single stars
            # if_planet = gaia_file[1].data['planet?'][index]
            self.gaia_non_single_star_flag = gaia_file[1].data['non_single_star'][index]
            self.gaia_variable_star_flag = gaia_file[1].data['phot_variable_flag'][index]

            if self.stellar_mass is None or self.stellar_radius is None or np.isnan(self.stellar_mass) or np.isnan(self.stellar_radius):
                self.failures += 'Unable to generate correct injections because of invalid stellar radius or mass;'
                m_1_3_r_s = 1.
            else:
                m_1_3_r_s = self.stellar_mass ** (1 / 3) / self.stellar_radius
            self.m_1_3_r_s = m_1_3_r_s  # M**(1/3)*R[in M_sun and R_sun]

            for key in self.star_attributes:
                try:
                    self.__dict__[key] = float(self.__dict__[key])
                except Exception as e:
                    self.failures += key + ' error ' + str(e) + '; '
                    self.__dict__[key] = None
            # todo: add fields with units and descriptions. astropy.table can be helpful with that
        gaia_file.close()

    def get_fields_from_info_file(self, gaia_file_path=None):
        """
        WHAT IS THIS FUNCTION?
        :param gaia_file_path:
        :type gaia_file_path:
        :return:
        :rtype:
        """
        # if gaia_file_path is None:
        #     gaia_file_path = self.gaia_kepler_file
        pass

    def get_koi_and_ttv_fields(self, koi_dict_file=None, ttv_dict_file=None):
        """

        :param koi_dict_file:
        :type koi_dict_file:
        :param ttv_dict_file:
        :type ttv_dict_file:
        :return:
        :rtype:
        """
        if koi_dict_file is None:
            koi_dict_file = self.koi_dict_file
        if ttv_dict_file is None:
            ttv_dict_file = self.ttv_dict_file
        self.is_koi = False
        with open(koi_dict_file, 'rb') as koi_catalog_dict_file:
            koi_info_dict = pickle.load(koi_catalog_dict_file)
        if self.target_id in koi_info_dict['kepid']:
            self.is_koi = True
            self.is_koi_planet_masked = False
            self.koi_dict = {}
            # ind = koi_info_dict['kepid'].index(self.target_id)
            inds = np.where(np.array(koi_info_dict['kepid']) == self.target_id)[0]
            for ind in inds:
                for key in koi_info_dict:
                    if key != 'kepid':
                        if key not in self.koi_dict:
                            self.koi_dict[key] = []
                        self.koi_dict[key].append(koi_info_dict[key][ind])
            with open(ttv_dict_file, 'rb') as ttv_catalog_dict_file:
                ttv_info_dict = pickle.load(ttv_catalog_dict_file)
            self.ttv_dict = {}
            for koi_name in self.koi_dict['kepoi_name']:
                koi_inds = np.where((np.array(ttv_info_dict['KOI']) == str(float(koi_name[1:])))
                                    & (np.array(ttv_info_dict['kepid']) == self.target_id))[0]
                if len(koi_inds) > 0:
                    for key in ttv_info_dict:
                        if key != 'kepid':
                            if key not in self.ttv_dict:
                                self.ttv_dict[key] = []
                            self.ttv_dict[key].append(np.array(ttv_info_dict[key])[koi_inds])
            # Units of Kepler times : unit = 'BJD - 2454833'
            # Units in the TTV catalog :  BJD-2454900
            if 'T' in self.ttv_dict.keys():  # it may be not there if the object is koi but not ttv
                for j in range(len(self.ttv_dict['T'])):
                    self.ttv_dict['T'][j] = self.ttv_dict['T'][j].astype(float) + 2454900 - 2454833

    def get_villanova_fields(self, villanova_dict_file=None):
        """
        Keys are 'KIC', 'period', 'period_err', 'bjd0', 'bjd0_err', 'pwidth', 'swidth', 'sep', 'morph'.

        * 'period' - period of the eclipsing binary signal in days
        * 'period_err' - period error in days
        * 'bjd0' time of eclipse, with the convention such that the primary (deeper) eclipse occurs at phase 0.
        Values are shown as BJD-2400000
        * 'bjd0_err'
        * 'pwidth' - width of primary eclipse in phase
        * 'swidth' - width of secondary eclipse in phase, -1 if depth less than 3*sigma
        * 'sep' - separation between primary and secondary eclipse (sec - pri), 1 if secondary depth less than 3*sigma
        * 'morph' - morphology value (between 0-1) assigned to system to designate the class of the system.
        Morphology of the system: values of 0–0.5 are predominantly assigned to detached systems, 0.5– 0.7 to semi-detached
        systems, 0.7–0.8 to contact binaries, and 0.8–1 to ellipsoidal variables, while heartbeat stars do not have an assigned value.

        http://keplerebs.villanova.edu/about#FileFormats

        :param villanova_dict_file:
        :type villanova_dict_file: str
        :return:
        :rtype:
        """
        if villanova_dict_file is None:
            villanova_dict_file = self.villanova_dict_file
        self.is_villanova_binary = False
        with open(villanova_dict_file, 'rb') as villanova_catalog_dict_file:
            villanova_dict = pickle.load(villanova_catalog_dict_file)
        if self.target_id in villanova_dict['KIC']:
            ind = villanova_dict['KIC'].index(self.target_id)
            self.is_villanova_binary = True
            self.villanova_dict = {}
            for key in villanova_dict.keys():
                self.villanova_dict[key] = villanova_dict[key][ind]

    def __str__(self):
        return f"{self.target_id}"

    # noinspection DuplicatedCode
    def print_dict_file(self, output_location=None, dict_txt_filename_postfix=Dict_target_txt_filename_postfix):
        """
        Prints only small attributes
        :param output_location:
        :type output_location:
        :param dict_txt_filename_postfix:
        :type dict_txt_filename_postfix:
        :return:
        :rtype:
        """
        if output_location is None:
            output_location = self.output_location
        dict_to_print = {}
        for key in self.__dict__:
            value = self.__dict__[key]
            if isinstance(value, (str, float, int, np.float32, np.integer, np.ndarray, list, dict)):  # float is np.float64 but not 32!
                ok_to_print = True
                if hasattr(value, '__len__') and len(value) > 0:
                    if isinstance(value, dict) and hasattr(value[list(value.keys())[0]], '__len__'):
                        ok_to_print = False  # TODO: maybe make the selection process look at all the dict keys
                    elif ((not isinstance(value, dict)) and (not isinstance(value, str))
                          and hasattr(value[0], '__len__')):
                        ok_to_print = False  # not printing nested lists
                    elif (not isinstance(value, str)) and len(value) > 40:
                        # roughly to be able to set summed values for planets with period 30
                        ok_to_print = False
                if key == 'errors' or key == 'failures':
                    ok_to_print = True
                if ok_to_print:
                    dict_to_print[key] = value
        file_path = af.get_saving_file_path(output_location=output_location, target_id=self.target_id,
                                            filename_postfix=dict_txt_filename_postfix, file_format='txt')
        with open(file_path, 'w') as fil:
            for key in dict_to_print:
                print(key, ":", dict_to_print[key], file=fil)

    def print_summary(self, to_file=True, file=None, summary_txt_filename_postfix=Summary_target_txt_filename_postfix):
        if file is None:
            file = af.get_saving_file_path(output_location=self.output_location, target_id=self.target_id,
                                           filename_postfix=summary_txt_filename_postfix, file_format='txt')
        if to_file:
            with open(file, 'w') as fil1:
                for attr in self.star_attributes:
                    print(attr, ':', self.__dict__[attr], file=fil1)
                print('', file=fil1)

                print('best_stat', ':', self.best_stat, file=fil1)
                print('best_per', ':', self.best_per, file=fil1)
                print('best_ftt', ':', self.best_ftt, file=fil1)
                print('best_gauss_stat', ':', self.best_gauss_stat, file=fil1)
                print('best_int_score', ':', self.best_integral_score, file=fil1)
                print('', file=fil1)
                print('best_g_stat', ':', self.best_g_stat, file=fil1)
                print('best_g_per', ':', self.best_g_per, file=fil1)
                print('best_g_ftt', ':', self.best_g_ftt, file=fil1)
                print('best_g_gauss_stat', ':', self.best_g_gauss_stat, file=fil1)
                print('best_g_int_score', ':', self.best_g_integral_score, file=fil1)
                print('', file=fil1)
        else:
            for attr in self.star_attributes:
                print(attr, ':', self.__dict__[attr])
            print('')

            print('best_stat', ':', self.best_stat)
            print('best_per', ':', self.best_per)
            print('best_ftt', ':', self.best_ftt)
            print('best_gauss_stat', ':', self.best_gauss_stat)
            print('best_int_score', ':', self.best_integral_score)
            print('')
            print('best_g_stat', ':', self.best_g_stat)
            print('best_g_per', ':', self.best_g_per)
            print('best_g_ftt', ':', self.best_g_ftt)
            print('best_g_gauss_stat', ':', self.best_g_gauss_stat)
            print('best_g_int_score', ':', self.best_g_integral_score)
            print('')

    def clean_output_directory(self):
        for ff in os.listdir(self.output_location):
            os.remove(os.path.join(self.output_location, ff))

    def initialize_kepler_light_curve(self, data_location=None, output_location=None,
                                      remove_zeroth_quarter=True, remove_quarter_17=True):
        if data_location is None:
            data_location = self.data_location
        if output_location is None:
            output_location = self.output_location
        self.klc = spip.KeplerLightCurve(target_id=self.target_id, get_data=True, data_location=data_location,
                                         output_location=output_location, m_1_3_r_s=self.m_1_3_r_s,
                                         remove_zeroth_quarter=remove_zeroth_quarter, remove_quarter_17=remove_quarter_17)  # Q17 is removed
        self.number_of_quarters = len(self.klc.quarters)

    def bad_fft_check(self, amp_max_bad_fft=300, dif_max_bad_fft=11):
        """
        Checks whether there are very high peaks in the fft of the original fluxes
        :param amp_max_bad_fft:
        :type amp_max_bad_fft:
        :param dif_max_bad_fft:
        :type dif_max_bad_fft:
        :return:
        :rtype:
        """
        # filtering bad_ffts WEXAC
        is_good = True
        for q in self.klc.quarters:
            lc = np.array(self.klc.orig_fluxes[q] - np.nanmean(self.klc.orig_fluxes[q]))
            if len(lc) > 2000:
                lc[np.where(np.isnan(lc))] = np.nanmean(lc)
                is_good_1 = not af.is_bad_fft(np.abs(np.fft.rfft(lc)), amp_max=amp_max_bad_fft, dif_max=dif_max_bad_fft)
                is_good &= is_good_1
        if is_good:
            self.is_bad_fft = False
        else:
            self.is_bad_fft = True

    def check_detectable_radius(self, stellar_radius, snr_thresh=Stat_candidate_search**0.5,
                                rp_re_max=Rp_re_max, rp_rs_max=Rp_rs_max):
        units_of_re_to_units_of_r_star_factor = 1 / (stellar_radius * R_sun_R_e)
        rp_rs = units_of_re_to_units_of_r_star_factor * rp_re_max
        rp_rs = min(rp_rs, rp_rs_max)

        n_transits_max = len(self.klc.quarters) * 2
        snr_max = (np.max([np.mean([np.linalg.norm(self.klc.whitened_templates_td[kk][q])
                                   for q in self.klc.quarters], axis=0) / np.max(self.klc.tb.templates_shifts[kk])
                          for kk in range(len(self.klc.whitened_templates_td))])
                   * self.klc.mean_total * n_transits_max ** 0.5 * (rp_rs ** 2))
        if snr_max < snr_thresh:
            self.failures = self.failures + ' too_large_noise; '
            self.is_bad_fft = True

    def get_detectable_radius_flux(self, flux, period, depth_factor=1., snr_min=4.5, snr_max_1=8, snr_max_full=40, stellar_radius=None,
                                   conditioned_data_filename_postfix=spip.Default_conditioned_data_filename_postfix,
                                   units='star'):
        """
        TODO: remind myself what this function does
        :param flux:
        :type flux:
        :param period:
        :type period:
        :param depth_factor:
        :type depth_factor:
        :param snr_min:
        :type snr_min:
        :param snr_max_1:
        :type snr_max_1:
        :param snr_max_full:
        :type snr_max_full:
        :param stellar_radius:
        :type stellar_radius:
        :param conditioned_data_filename_postfix:
        :type conditioned_data_filename_postfix:
        :param units:
        :type units:
        :return:
        :rtype:
        """
        if stellar_radius is None:
            stellar_radius = self.stellar_radius
        wf_fds = self.load_rel_wf_fds(conditioned_data_filename_postfix=conditioned_data_filename_postfix)
        # -> relative wf_fd, mean flux already taken into account
        n_quarters = len(wf_fds)
        n_transits_max = np.ceil(n_quarters * 90. / period)
        wf_fd = np.mean(np.array(wf_fds)**2, axis=0) ** 0.5
        support = 2 * (len(wf_fd) - 1)

        flux = np.roll(flux, -np.argmax(flux))
        flux = flux / np.nanmax(flux)
        flux = np.concatenate((flux[:len(flux)//2], np.zeros(support - len(flux)), flux[len(flux)//2:])) * depth_factor
        flux = np.fft.irfft(np.fft.rfft(flux) * wf_fd)
        # snr_max = np.linalg.norm(flux) * n_transits_max ** 0.5 * (rp_rs ** 2)
        rp_rs_min = (snr_min / (np.linalg.norm(flux) * n_transits_max ** 0.5)) ** 0.5
        rp_rs_max = min((snr_max_1 / np.linalg.norm(flux)) ** 0.5, (snr_max_full / (np.linalg.norm(flux) * n_transits_max ** 0.5)) ** 0.5)
        res = rp_rs_min, rp_rs_max
        if units == 'earth':
            units_of_re_to_units_of_r_star_factor = 1 / (stellar_radius * R_sun_R_e)
            # rp_rs = units_of_re_to_units_of_r_star_factor * rp_re
            rp_re_min = rp_rs_min / units_of_re_to_units_of_r_star_factor
            rp_re_max = rp_rs_max / units_of_re_to_units_of_r_star_factor
            res = rp_re_min, rp_re_max
        return res

    def mask_koi_planet(self, modify_self_klc=True, return_durations_transit_times=False, include_candidates=False):
        # TODO why do we mask the planet, and not subtract? We could also subtract villanova ephemeris.
        # TODO: include occultation (secondary transit)
        assert self.is_koi, 'The target is not KOI, so the KOI planet cannot be masked!'
        # Take all the transit times with TTV, take transit duration, take fractional TDV and mask.
        durations_and_times = []
        for i, koi_name in enumerate(self.koi_dict['kepoi_name']):
            # In Run 8, we will only mask confirmed or false positives, we will leave the candidates.
            if ((self.koi_dict['koi_disposition'][i] == 'CONFIRMED' and self.koi_dict['koi_pdisposition'][i] == 'CANDIDATE')
                    or (self.koi_dict['koi_disposition'][i] == 'FALSE POSITIVE'
                        and self.koi_dict['koi_pdisposition'][i] == 'FALSE POSITIVE')
                    or include_candidates):
                transit_durations = None  # just initialization, no meaning
                transit_times = None
                was_in_ttv = False
                something_bad = False
                if self.ttv_dict is not None and 'T' in self.ttv_dict and 'KOI' in self.ttv_dict:
                    for j in range(len(self.ttv_dict['KOI'])):
                        if self.ttv_dict['KOI'][j][0] == str(float(koi_name[1:])):
                            transit_times = np.array(self.ttv_dict['T'][j], dtype=float)  # this is including TTVs
                            transit_durations = np.ones(len(transit_times)) * float(
                                self.koi_dict['koi_duration'][i]) / 24  # we need it in days, and it was in hours
                            tdv_array = np.array(self.ttv_dict['TDV'][j], dtype=float)
                            tdv_array[np.isnan(tdv_array)] = 0.
                            transit_durations = transit_durations * (1. + tdv_array)  # add TDVs
                            was_in_ttv = True
                            break
                if not was_in_ttv:
                    if self.koi_dict['koi_num_transits'][i] == '':
                        self.failures = self.failures + ' koi_num_transits was empty!'
                    else:
                        transit_times = np.array([float(self.koi_dict['koi_time0bk'][i]) + float(self.koi_dict['koi_period'][i]) * nn
                                                  for nn in range(int(self.koi_dict['koi_num_transits'][i]))])
                        transit_durations = np.ones(len(transit_times)) * float(
                            self.koi_dict['koi_duration'][i]) / 24  # we need it in days, and it was in hours
                        something_bad = True
                if self.klc is not None and modify_self_klc and not something_bad:
                    self.klc.mask_ephemeris(durations=transit_durations, transit_times=transit_times)
                    self.is_koi_planet_masked = True
                if return_durations_transit_times and not something_bad:
                    durations_and_times.append([transit_durations, transit_times])
        if return_durations_transit_times:
            return durations_and_times

    def mask_villanova_eclipses(self):
        """
        Masking Villanova binaries is not implemented yet for reasons described in the comments.
        Keys are 'KIC', 'period', 'period_err', 'bjd0', 'bjd0_err', 'pwidth', 'swidth', 'sep', 'morph'.

        :return:
        :rtype:
        """
        raise AssertionError('Masking Villanova binaries is not implemented yet for reasons described in the documentation!')
        # We are not doing this for very short-period and not detached binaries, because there are too many times to mask.
        # We are not doing this for large periods, although they are detached. For too many of them, the reported separation is 1:
        # if separation is 1, then the depth of the secondary eclipse is less than 3 sigma and they do not report it, but for us this is
        # not convenient because we will detect this secondary eclipse, and not the planet itself.
        # We cannot mask the secondary eclipse because it is not specified where it is.
        # TODO: work this out when we detect more than one planet per star.
        # TODO: why do we mask the planet, and not subtract? We could also subtract villanova ephemeris.

    def run_ses_search(self, conditioned_data_filename_postfix=Conditioned_data_filename_postfix,
                       mask_koi_planet=True, save_conditioned_data=False, save_connected_stats=True,
                       connected_stats_filename_postfix=Connected_statistic_data_filename_postfix,
                       save_no_veto_no_score_cor_stats=True, stop_if_bad=True,
                       no_veto_no_score_cor_filename_postfix=No_veto_no_score_cor_stats_filename_postfix):
        """

        :param no_veto_no_score_cor_filename_postfix:
        :type no_veto_no_score_cor_filename_postfix:
        :param save_no_veto_no_score_cor_stats:
        :type save_no_veto_no_score_cor_stats:
        :param connected_stats_filename_postfix:
        :type connected_stats_filename_postfix:
        :param conditioned_data_filename_postfix:
        :type conditioned_data_filename_postfix:
        :param mask_koi_planet:
        :type mask_koi_planet:
        :param save_conditioned_data:
        :type save_conditioned_data:
        :param stop_if_bad:
        :type stop_if_bad:
        :param save_connected_stats:
        :type save_connected_stats:
        :return:
        :rtype:
        """

        t1 = time.time()
        self.klc = spip.KeplerLightCurve(target_id=self.target_id, get_data=True, data_location=self.data_location,
                                         output_location=self.output_location, m_1_3_r_s=self.m_1_3_r_s)  # Q17 is removed
        self.bad_fft_check()  # Made on orig_fluxes
        if self.is_koi and mask_koi_planet:
            self.mask_koi_planet(modify_self_klc=True, return_durations_transit_times=False, include_candidates=False)
            # TODO: think whom we mask
            # # if there are only candidates, it will not mask
            # # An alternative way:
            # transit_durations_and_times = self.mask_koi_planet(modify_self_klc=False, return_durations_transit_times=True,
            #                                                    include_candidates=False)
            # if len(transit_durations_and_times) > 0:  # if there are only candidates, it will not mask
            #     for transit_durations, transit_times in transit_durations_and_times:
            #         self.klc.mask_ephemeris(durations=transit_durations, transit_times=transit_times)
        self.klc.condition_data()  # In outlier mask, positive and negative nsig_outliers are the same because we run inversion after.
        self.was_notched = self.klc.was_notched
        self.klc.condition_templates()
        self.check_detectable_radius(stellar_radius=self.stellar_radius)
        if (not self.is_bad_fft) or (not stop_if_bad):
            if save_conditioned_data:
                self.klc.save_conditioned_data(filename_postfix=conditioned_data_filename_postfix)
            self.klc.detect_single_transits(gaussianize=True)
            if save_no_veto_no_score_cor_stats:
                self.klc.save_no_veto_no_score_cor_stats(filename_postfix=no_veto_no_score_cor_filename_postfix)
            self.klc.connect_quarters()
            if save_connected_stats:
                self.klc.save_connected_stats(filename_postfix=connected_stats_filename_postfix)
            self.klc.interleave_shifts(delete_fulls=True)

            t2 = time.time()
            self.conditioning_ses_time = t2 - t1

    def get_connected_klc(self, reinitialize=True, conditioned_data_filename_postfix=Conditioned_data_filename_postfix,
                          connected_statistic_data_filename_postfix=Connected_statistic_data_filename_postfix,
                          mask_koi_planet=True, save_connected_stats=True,
                          connected_stats_filename_postfix=Connected_statistic_data_filename_postfix):
        """

        :param reinitialize:
        :type reinitialize:
        :param conditioned_data_filename_postfix:
        :type conditioned_data_filename_postfix:
        :param connected_statistic_data_filename_postfix:
        :type connected_statistic_data_filename_postfix:
        :param mask_koi_planet:
        :type mask_koi_planet: bool or None
        :param save_connected_stats:
        :type save_connected_stats:
        :param connected_stats_filename_postfix:
        :type connected_stats_filename_postfix:
        :return:
        :rtype:
        """
        if (self.klc is None) or reinitialize:
            self.klc = spip.KeplerLightCurve(target_id=self.target_id, get_data=True, data_location=self.data_location,
                                             output_location=self.output_location, m_1_3_r_s=self.m_1_3_r_s)  # Q17 is removed
            self.bad_fft_check()  # Made on orig_fluxes
            if self.is_koi and (mask_koi_planet or (self.is_koi_planet_masked and mask_koi_planet is None)):
                # the latter is from the pickled attributes, meaning
                # that the saved connected data is probably also with the masked koi planet
                self.mask_koi_planet(modify_self_klc=True, return_durations_transit_times=False, include_candidates=False)
                # # if there are only candidates, it will not mask

        file_path_connected_data = af.get_saving_file_path(output_location=self.output_location, target_id=self.target_id,
                                                           filename_postfix=connected_statistic_data_filename_postfix, file_format='npy')
        file_path_conditioned_data = af.get_saving_file_path(output_location=self.output_location, target_id=self.target_id,
                                                             filename_postfix=conditioned_data_filename_postfix, file_format='npy')
        if self.klc.statistics_full is None or not hasattr(self.klc, 'statistics_full'):
            if os.path.exists(file_path_connected_data):
                if not hasattr(self, "errors"):
                    self.errors = ''
                try:
                    self.klc.load_connected_stats(file_path=file_path_connected_data)
                except (EOFError, IOError, OSError, ValueError) as error:
                    self.errors += str(type(error)) + ' ' + str(error) + '\n'
                    success = False
                    rep_err = ''
                    np.random.seed(None)
                    # -> it can be that another process is trying to open the same file, so we will wait
                    for _ in range(3):
                        time.sleep(1. + np.random.rand() * 3.)
                        try:
                            self.klc.load_connected_stats(file_path=file_path_connected_data)
                            success = True
                        except (EOFError, IOError, OSError, ValueError) as error:
                            rep_err = error
                        if success:
                            break
                    if not success:
                        raise IOError(str(type(rep_err)) + " " + str(rep_err))
            else:
                if self.klc.statistics_raw is None:
                    if os.path.exists(file_path_conditioned_data):
                        self.klc.load_conditioned_data(file=file_path_conditioned_data)
                        self.klc.initialize_templates()
                    else:
                        self.klc.condition_data()
                        self.klc.condition_templates()
                    self.klc.detect_single_transits(gaussianize=True)
                self.klc.connect_quarters()
                if save_connected_stats:
                    self.klc.save_connected_stats(filename_postfix=connected_stats_filename_postfix)

    def set_perodicity_search_parameters(self, p_min=P_min_search, p_max=P_max_search, per_phases_per_chunk=Per_phases_per_chunk,
                                         duty_cycle=spip.Duty_cycle, force_overwrite=False):
        if self.p_min is None or force_overwrite:
            self.p_min = p_min
        if self.p_max is None or force_overwrite:
            self.p_max = p_max
        if self.duty_cycle is None or force_overwrite:
            self.duty_cycle = duty_cycle
        if self.per_phases_per_chunk is None or force_overwrite:
            self.per_phases_per_chunk = per_phases_per_chunk

    def run_periodicity_search(self, p_min=P_min_search, p_max=P_max_search, duty_cycle=spip.Duty_cycle,
                               per_phases_per_chunk=Per_phases_per_chunk,
                               best_periodic_scores_filename_postfix=Best_periodic_scores_filename_postfix,
                               best_periodic_scores_file_dir=None, overlap_period_bins=spip.P_bins_overlap):

        """
        :param per_phases_per_chunk:
        :type per_phases_per_chunk:
        :param best_periodic_scores_filename_postfix:
        :type best_periodic_scores_filename_postfix:
        :param p_min:
        :type p_min:
        :param p_max:
        :type p_max:
        :param duty_cycle:
        :type duty_cycle:
        :param best_periodic_scores_file_dir:
        :type best_periodic_scores_file_dir:
        :param overlap_period_bins: when creating period chunks, extend each chunk from the two sides by
            this number of bins for them to overlap. is needed for coherent score integral in order for
            the maximum not to be on the edge
        :type overlap_period_bins: int
        :return:
        :rtype:
        """
        self.set_perodicity_search_parameters(p_min=p_min, p_max=p_max, per_phases_per_chunk=per_phases_per_chunk,
                                         duty_cycle=duty_cycle, force_overwrite=True)

        if self.klc is None or not hasattr(self.klc, 'statistics_full') or self.klc.statistics_full is None:
            self.get_connected_klc(reinitialize=False, mask_koi_planet=None)
        self.klc.delete_irrelevant()  # ATTENTION: we introduced this because of unclear memory issues

        self.periods_in_chunks = self.klc.make_periods_chunks(p_min=self.p_min, p_max=self.p_max,
                                                              per_phases_per_chunk=self.per_phases_per_chunk,
                                                              duty_cycle=duty_cycle, dt=self.klc.dt_days, t_mission=self.klc.T_mission)
        t1 = time.time()
        best_candidates_concat = []
        for pers in self.periods_in_chunks:
            self.klc.periodicity_search(pmin=None, pmax=None, duty_cycle=None, periods=pers,
                                        delete_full_stats=True, force_interleave_shifts=False, delete_periodic_interleaved=True,
                                        save_uncorrected_periodic=False, delete_non_marginalized=False,  # delete_non_maximized=False,
                                        overlap_period_bins=overlap_period_bins)
            self.klc.get_best_trigger(apply_veto_mes=True, in_gaussianized=False)
            self.klc.get_best_trigger(apply_veto_mes=True, in_gaussianized=True)
            self.klc.make_best_periodic_scores()
            best_candidates_concat.append(np.array(self.klc.best_periodic_scores_data, dtype=object))
            self.klc.delete_mes()
        if best_periodic_scores_file_dir is not None:
            np.save(best_periodic_scores_file_dir, np.array(best_candidates_concat, dtype=object), allow_pickle=True)
        else:
            file_path = af.get_saving_file_path(output_location=self.output_location, target_id=self.target_id,
                                                filename_postfix=best_periodic_scores_filename_postfix, file_format='npy')
            np.save(file_path, np.array(best_candidates_concat, dtype=object), allow_pickle=True)

        self._find_best_trigger_from_per_chunks(best_candidates_concat=best_candidates_concat)
        t2 = time.time()
        self.periodicity_search_time = t2 - t1

    def _find_best_trigger_from_per_chunks(self, best_candidates_concat=None):
        self.best_integral_score = -np.inf
        for best_candidates in best_candidates_concat:  # if there is a big MES score, further processing
            scormax = best_candidates[spip.Best_per_results_index_best_integral][0]
            if scormax > self.best_integral_score:
                self.best_integral_score = scormax
                self.best_stat = best_candidates[spip.Best_per_results_index_stats][0]
                self.best_per = best_candidates[spip.Best_per_results_index_per][0]
                self.best_ftt = best_candidates[spip.Best_per_results_index_ftt][0]
                self.best_gauss_stat = best_candidates[spip.Best_per_results_index_stats_gauss][0]
                self.best_temp = best_candidates[spip.Best_per_results_index_template][0]

        self.best_g_integral_score = -np.inf
        for best_candidates in best_candidates_concat:  # if there is a big MES score, further processing
            scormax = best_candidates[spip.Best_per_results_index_best_integral][1]
            if scormax > self.best_g_integral_score:
                self.best_g_integral_score = scormax
                self.best_g_stat = best_candidates[spip.Best_per_results_index_stats][1]
                self.best_g_per = best_candidates[spip.Best_per_results_index_per][1]
                self.best_g_ftt = best_candidates[spip.Best_per_results_index_ftt][1]
                self.best_g_gauss_stat = best_candidates[spip.Best_per_results_index_stats_gauss][1]
                self.best_g_temp = best_candidates[spip.Best_per_results_index_template][1]

    def run_main_search(self, p_min=P_min_search, p_max=P_max_search,
                        per_phases_per_chunk=Per_phases_per_chunk,
                        candidate_threshold=Stat_candidate_search,
                        mask_koi_planet=True,
                        save_to_threshold_crossing_list=True,
                        save_conditioned_data=True,
                        save_no_veto_no_score_cor_stats=False,
                        save_connected_stats=True,
                        conditioned_data_filename_postfix=Conditioned_data_filename_postfix,
                        best_periodic_scores_filename_postfix=Best_periodic_scores_filename_postfix,
                        no_veto_no_score_cor_stats_filename_postfix=No_veto_no_score_cor_stats_filename_postfix,
                        connected_stats_data_filename_postfix=Connected_statistic_data_filename_postfix,
                        threshold_crossing_list_file=None,
                        threshold_crossing_list_location=Threshold_crossing_lists_folder,
                        threshold_crossing_list_filename=Threshold_crossing_list_filename,
                        clean_dir_if_not_threshold_crossing=False, overlap_period_bins=spip.P_bins_overlap):

        """
        :param p_min:
        :type p_min:
        :param p_max:
        :type p_max:
        :param per_phases_per_chunk:
        :type per_phases_per_chunk:
        :param mask_koi_planet:
        :type mask_koi_planet:
        :param candidate_threshold:
        :type candidate_threshold:
        :param save_to_threshold_crossing_list:
        :type save_to_threshold_crossing_list:
        :param save_conditioned_data:
        :type save_conditioned_data:
        :param save_no_veto_no_score_cor_stats:
        :type save_no_veto_no_score_cor_stats:
        :param save_connected_stats:
        :type save_connected_stats:
        :param best_periodic_scores_filename_postfix:
        :type best_periodic_scores_filename_postfix:
        :param conditioned_data_filename_postfix:
        :type conditioned_data_filename_postfix:
        :param no_veto_no_score_cor_stats_filename_postfix:
        :type no_veto_no_score_cor_stats_filename_postfix:
        :param connected_stats_data_filename_postfix:
        :type connected_stats_data_filename_postfix:
        :param threshold_crossing_list_file:
        :type threshold_crossing_list_file:
        :param threshold_crossing_list_location:
        :type threshold_crossing_list_location:
        :param threshold_crossing_list_filename:
        :type threshold_crossing_list_filename:
        :param clean_dir_if_not_threshold_crossing:
        :type clean_dir_if_not_threshold_crossing:
        :param overlap_period_bins: when creating period chunks, extend each chunk from the two sides by
            this number of bins for them to overlap. is needed for coherent score integral in order for
            the maximum not to be on the edge
        :type overlap_period_bins: int
        :return:
        :rtype:
        """
        if (threshold_crossing_list_file is None) and (threshold_crossing_list_location is not None):
            threshold_crossing_list_file = os.path.join(threshold_crossing_list_location, threshold_crossing_list_filename)

        self.run_ses_search(save_conditioned_data=save_conditioned_data,
                            conditioned_data_filename_postfix=conditioned_data_filename_postfix,
                            save_no_veto_no_score_cor_stats=save_no_veto_no_score_cor_stats,
                            no_veto_no_score_cor_filename_postfix=no_veto_no_score_cor_stats_filename_postfix,
                            save_connected_stats=save_connected_stats, stop_if_bad=True,
                            connected_stats_filename_postfix=connected_stats_data_filename_postfix,
                            mask_koi_planet=mask_koi_planet)
        if not self.is_bad_fft:
            self.run_periodicity_search(p_min=p_min, p_max=p_max, per_phases_per_chunk=per_phases_per_chunk,
                                        best_periodic_scores_filename_postfix=best_periodic_scores_filename_postfix,
                                        overlap_period_bins=overlap_period_bins)

            if (self.best_stat >= candidate_threshold) or (self.best_g_gauss_stat >= candidate_threshold):
                self.is_threshold_crossing = True
                if save_to_threshold_crossing_list:
                    with open(threshold_crossing_list_file, 'a') as fil1:  # will save here these good targets
                        print(self.target_id, self.best_stat, self.best_per, self.best_ftt, self.best_gauss_stat, self.best_integral_score,
                              self.best_g_stat, self.best_g_per, self.best_g_ftt, self.best_g_gauss_stat, self.best_g_integral_score,
                              sep=' ', end='\n', file=fil1)

            if (not self.is_threshold_crossing) and clean_dir_if_not_threshold_crossing:
                # delete conditioned data because we have no space on the disk!
                for ff in os.listdir(self.output_location):
                    os.remove(os.path.join(self.output_location, ff))

        self.pickle_small_attributes()
        self.print_dict_file()

    def run_timeslides(self, n_slides=N_slides_chunk, central_period=None, mask_koi_planet=True,
                       n_chunks_around_best=1, per_wind_slides=None,
                       per_wing_slides_relative=Per_wing_injection_and_slides_relative, use_period_chunks=False,
                       use_shuffling_pieces=True, only_if_threshold_crossing=False, overlap_period_bins=spip.P_bins_overlap,
                       slides_res_filename_postfix=Slides_res_filename_postfix):
        """

        :param n_slides:
        :type n_slides: int
        :param central_period:
        :type central_period:
        :param mask_koi_planet: whether to mask the KOI ephemeris if the object is a KOI
        :type mask_koi_planet: bool
        :param n_chunks_around_best:
        :type n_chunks_around_best:
        :param per_wind_slides:
        :type per_wind_slides:
        :param per_wing_slides_relative:
        :type per_wing_slides_relative:
        :param use_period_chunks: whether to slice periods to chunks to reduce the RAM consumed
        :type use_period_chunks: bool
        :param use_shuffling_pieces: whether to cut quarters into pieces and reshuffle them while preserving the total duration of the data
        :type use_shuffling_pieces: bool
        :param only_if_threshold_crossing:
        :type only_if_threshold_crossing:
        :param overlap_period_bins: when creating period chunks, extend each chunk from the two sides by
            this number of bins for them to overlap. is needed for coherent score integral in order for
            the maximum not to be on the edge
        :type overlap_period_bins: int
        :param slides_res_filename_postfix:
        :type slides_res_filename_postfix:
        :return:
        :rtype:
        """
        if self.is_threshold_crossing or not only_if_threshold_crossing:
            self.get_connected_klc(reinitialize=True, mask_koi_planet=mask_koi_planet)
            # ATTENTION: we introduced this because of unclear memory issues
            self.klc.delete_irrelevant()
            # In case some parameters were not set, replace them with the defaults:
            self.set_perodicity_search_parameters(self, force_overwrite=False)

            if central_period is None:
                period = self.best_per
            else:
                period = central_period

            if n_chunks_around_best is not None:
                if self.periods_in_chunks is None:
                    periods_in_chunks = self.klc.make_periods_chunks(p_min=self.p_min, p_max=self.p_max, duty_cycle=self.duty_cycle,
                                                                     per_phases_per_chunk=self.per_phases_per_chunk,
                                                                     dt=self.klc.dt_days, t_mission=self.klc.T_mission)
                else:
                    periods_in_chunks = self.periods_in_chunks
                closest_chunk, closest_chunk_index = ut.find_closest_chunk(periods_in_chunks, period, overlap_bins=0)
                ind_in_chunk = np.nanargmin(np.abs(np.array(closest_chunk) - period))
                if ind_in_chunk < len(closest_chunk) / 2:
                    chunk_inds_choose = (closest_chunk_index + np.arange(-(n_chunks_around_best // 2),
                                                                        (n_chunks_around_best + 1) // 2)).astype(int)
                else:
                    chunk_inds_choose = (closest_chunk_index + np.arange(-(n_chunks_around_best - 1) // 2,
                                                                        (n_chunks_around_best + 2) // 2)).astype(int)
                period_chunks = [periods_in_chunks[i] for i in chunk_inds_choose]

            else:
                if per_wind_slides is not None:
                    self.per_wing_injection_and_slides = per_wind_slides
                if self.per_wing_injection_and_slides is None:
                    self.per_wing_injection_and_slides = per_wing_slides_relative * period
                    # days to the left and the same to the right #limited by memory
                p_max = period + self.per_wing_injection_and_slides
                p_min = period - self.per_wing_injection_and_slides
                # we do not need period chunks because we only have one template and only 20 days of periods.

                if use_period_chunks:
                    period_chunks = self.klc.make_periods_chunks(per_phases_per_chunk=self.per_phases_per_chunk, duty_cycle=self.duty_cycle,
                                                                 p_min=p_min, p_max=p_max, dt=None, t_mission=None)
                else:
                    periods = self.klc.make_periods_array(p_min=p_min, p_max=p_max)
                    period_chunks = [periods]
            # HISTOGRAM DATA WILL BE COLLECTED HERE
            stats = []
            pers = []
            first_times = []
            gauss_stats = []
            int_scores = []
            stats_g = []
            pers_g = []
            first_times_g = []
            gauss_stats_g = []
            int_scores_g = []
            for i_slides in range(n_slides):
                np.random.seed(None)
                new_seed = np.random.randint(int(2**30))
                if not use_shuffling_pieces:
                    self.klc.time_slides(random_seed=new_seed)
                    self.klc.connect_quarters()
                    self.klc.interleave_shifts(delete_fulls=True)
                else:
                    self.klc.scramble_full_stats(max_hole=spip.Max_hole_pieces_scrampling, max_piece=spip.Max_piece_scrambling,
                                                 min_piece=spip.Min_piece_scrambling, random_seed=new_seed)
                    self.klc.interleave_shifts(delete_fulls=False)  # do not delete fulls to then make re-scrambling

                # best_candidates_concat = []
                stats_chunk = []
                pers_chunk = []
                first_times_chunk = []
                gauss_stats_chunk = []
                int_scores_chunk = []
                stats_g_chunk = []
                pers_g_chunk = []
                first_times_g_chunk = []
                gauss_stats_g_chunk = []
                int_scores_g_chunk = []
                for pers_summing in period_chunks:
                    try:
                        self.klc.periodicity_search(pmin=None, pmax=None, periods=pers_summing, delete_full_stats=False,
                                                    force_interleave_shifts=True,  # very important: new slides!
                                                    delete_periodic_interleaved=True, save_uncorrected_periodic=False,
                                                    delete_non_marginalized=False, overlap_period_bins=overlap_period_bins)
                        self.klc.get_best_trigger(apply_veto_mes=True, in_gaussianized=False)
                        self.klc.get_best_trigger(apply_veto_mes=True, in_gaussianized=True)
                        self.klc.make_best_periodic_scores()
                    except Exception as exc:
                        raise exc
                        # report_exception(exception=exc, file_to_append='/home/labs/barakz/orynai/run_9/weird_errors',
                        #                  title=self.target_id, info='seed '+str(new_seed))
                    for list_ch, list_g_ch, s_ind in [[stats_chunk, stats_g_chunk, spip.Best_per_results_index_stats],
                                                      [pers_chunk, pers_g_chunk, spip.Best_per_results_index_per],
                                                      [first_times_chunk, first_times_g_chunk, spip.Best_per_results_index_ftt],
                                                      [gauss_stats_chunk, gauss_stats_g_chunk, spip.Best_per_results_index_stats_gauss],
                                                      [int_scores_chunk, int_scores_g_chunk, spip.Best_per_results_index_best_integral]]:
                        list_ch.append(self.klc.best_periodic_scores_data[s_ind][0])
                        list_g_ch.append(self.klc.best_periodic_scores_data[s_ind][1])
                best_ind_chunk = np.argmax(int_scores_chunk)  # (np.argsort(stats_chunk)[::-1])[:n_best_saving]
                best_g_ind_chunk = np.argmax(int_scores_g_chunk)  # (np.argsort(gauss_stats_g_chunk)[::-1])[:n_best_saving]
                for list_full, list_full_g, list_ch, list_ch_g in [[stats, stats_g, stats_chunk, stats_g_chunk],
                                                                   [pers, pers_g, pers_chunk, pers_g_chunk],
                                                                   [first_times, first_times_g, first_times_chunk, first_times_g_chunk],
                                                                   [gauss_stats, gauss_stats_g, gauss_stats_chunk, gauss_stats_g_chunk],
                                                                   [int_scores, int_scores_g, int_scores_chunk, int_scores_g_chunk]]:
                    list_full.append(list_ch[best_ind_chunk])  # += list(np.array(list_ch)[best_indices_chunk])
                    list_full_g.append(list_ch_g[best_g_ind_chunk])

            new_res_array = np.vstack((stats, pers, first_times, gauss_stats, int_scores,
                                       stats_g, pers_g, first_times_g, gauss_stats_g, int_scores_g)).transpose()

            filename = [filn for filn in os.listdir(self.output_location)
                        if self.target_id + '_' + slides_res_filename_postfix + '_' in filn]
            if len(filename) > 0:
                filename = filename[0]
                n_slides_0 = int(filename[len(self.target_id + '_' + slides_res_filename_postfix + '_'): filename.index('.')])
                n_slides += n_slides_0
                hist_0 = np.load(os.path.join(self.output_location, filename), allow_pickle=True)
                new_res_array = np.vstack((hist_0, new_res_array))
                os.remove(os.path.join(self.output_location, filename))
            np.save(os.path.join(self.output_location, self.target_id + '_' + slides_res_filename_postfix + '_' + str(n_slides)),
                    new_res_array, allow_pickle=True)

            # fls = os.listdir(self.output_location)
            # for filename_postfix, data_list in [[slides_stats_filename_postfix, stats],
            #                                     [slides_pers_filename_postfix, pers],
            #                                     [slides_first_times_filename_postfix, first_times],
            #                                     [slides_gauss_stats_filename_postfix, gauss_stats],
            #                                     [slides_gauss_stats_filename_postfix, gauss_stats]]:
            #     for fl in fls:
            #         if self.target_id + '_' + filename_postfix + '_' in fl:
            #             if filename_postfix == slides_stats_filename_postfix:
            #                 n_slides_0 = int(fl[len(self.target_id + '_' + filename_postfix + '_'): fl.index('.')])
            #                 n_slides += n_slides_0
            #             hist0 = list(np.load(os.path.join(self.output_location, fl), allow_pickle=True))
            #             data_list += hist0
            #             # very important that += is here and not data_list = data_list + hist0 !!!
            #             # the second option would not append to the old object but create a new object, so that wouldn't work!
            #             os.remove(os.path.join(self.output_location, fl))
            #     np.save(os.path.join(self.output_location, self.target_id + '_' + filename_postfix + '_' + str(n_slides)),
            #             np.array(data_list), allow_pickle=True)

    def load_rel_wf_fds(self, conditioned_data_filename_postfix=spip.Default_conditioned_data_filename_postfix):
        """
        Load quarters-mean whitening filter in Fourier domain in units relative to the mean flux level.
        In other words, part of flux in time-domain that is taken by this or that component of noise.
        :param conditioned_data_filename_postfix:
        :type conditioned_data_filename_postfix:
        :return:
        :rtype:
        """
        path = self.output_location
        file = af.get_saving_file_path(output_location=path, target_id=self.target_id,
                                       filename_postfix=conditioned_data_filename_postfix, file_format='npy')
        conditioned_data = np.load(file, allow_pickle=True)
        support = int(conditioned_data[0][10])
        n_quarters = int(np.round(len(conditioned_data[0]) / support))
        quarters = np.arange(n_quarters)
        # len_whitened_templates_td = int(conditioned_data[0][11])
        mean_total = float(conditioned_data[0][13])
        # wf_tds = []
        wf_fds = []
        ind_last = 0
        for _ in quarters:
            wftd1 = np.array(conditioned_data[4][ind_last: ind_last + support]).astype(float)
            # wf_tds.append(wftd1)
            wf_fds.append(np.abs(np.fft.rfft(wftd1)))  # np.fft.ifftshift(wftd1)
            ind_last += support
        return np.array(wf_fds) * mean_total

    def generate_injection_parameters(
            self, n_injections=N_injections_total, add_to_existing_file=False,
            per_wing_injection_relative=Per_wing_injection_and_slides_relative,
            injection_parameters_filename_postfix=Injection_parameters_filename_postfix,
            # mean_psd_filename_postfix=spip.Default_psd_for_injections_filename_postfix,
            all_injection_parameters_filename_postfix=All_injection_parameters_filename_postfix,
            d_ecc=0.3,
            n_w=5,  #
            d_inc=0.5,  # this is in degrees! and often until 85 is fine
            u=(0.47, 0.23),
            snr_min_injected=5.,
            n_per=50,
            n_rp=30):
        """
        TODO REWRITE THIS!
        TODO # make sure we really obtain the correct rate, meaning that when we exclude some duration from the bank,
        TODO # it should decrease the overall rate. The full rate of Zhu and Dong include all the planetary parameters,
        TODO # therefore when we use only part of them, our rate should not sum up to the rate of Zhu and Dong.


        without a good reason for that, and then the template factor will heavily suppress the result. Maybe I need to account for
        the specific expected snr of this transit to smear the templates much more. Or I need to also inject other shit, but this
        is even more complicated.

        :param n_injections: Number of injections to generate in the end. The inner number of generations will be defined by the
        ranges of parameters. `n_injections` is just how many will be returned in the end for the best template
        :type n_injections:
        :param per_wing_injection_relative: days to the left and the same to the right used in the time slides in units of period
        :type per_wing_injection_relative: float
        :param n_per: Number of periods within the period wings to generate. Not evenly, they will be sampled randomly.
        :type n_per: int
        :param n_rp: How many radii is tried per one parameter set
        :type n_rp: int
        :param d_ecc: Eccentricity step when calculating how many eccentricities to generate. Needed for uniformity of eccentricity
        density generation to make sure that when the range of possible orbital is smaller, we generate fewer of them.
        :type d_ecc: float
        :param n_w: Number of periastron angles to generate in each loop. Since this is uniform and with no bounds,
        it does not really matter how many of them we produce
        :type n_w: int
        :param d_inc: Inclination step when calculating how many inclinations to generate. Needed for uniformity of inclination
        density generation to make sure that when the range of possible transit inclinations is smaller, we generate fewer of them.
        This is in degrees! and often until 85 is fine
        :type d_inc: float, this is in degrees! and often until 85 is fine
        :param u: Limb darkening coefficients. nothing is too sensitive to them, so they are just fixed
        :type u: [float, float]
        :param snr_min_injected: full snr, not single-transit
        :type snr_min_injected: float
        :param add_to_existing_file:
        :type add_to_existing_file:
        :param injection_parameters_filename_postfix:
        :type injection_parameters_filename_postfix:
        :param all_injection_parameters_filename_postfix:
        :type all_injection_parameters_filename_postfix:
        :return:
        :rtype:

        (DONE): DO NOT FORGET THE BUG AND LOOK AT THE TRANSIT TIME AND NOT THE BADLY DEFINED PHASE!
        (DONE): THINK ABOUT THE REPETITIONS: WHEN YOU GENERATE ONE P_INJ, YOU USE IT FOR ALL THE OTHER PARAMETERS,
        WHICH IS NOT REALLY FAIR. HOWEVER , IF YOU CHANGE THE THING EVERY TIME, THEN YOU WILL GIVE NOT PROPORTIONAL
        WEIGHTS TO SMALL INCLINATIONS AND BIG INCLINATIONS, LET'S SAY. SO EITHER YOU NEED TO GENERATE MANY OF THEM AND REJECT
        THE NON-TRANSITING ONES, OR SOMETHING ELSE.............OR IT IS OKAY THAT THEY ARE DISCRETE
        :return:
        """
        if self.stellar_mass is None or self.stellar_radius is None or np.isnan(self.stellar_mass) or np.isnan(self.stellar_radius):
            self.failures += 'Unable to generate correct injections because of invalid stellar radius or mass;'

        file_path = af.get_saving_file_path(output_location=self.output_location, target_id=self.target_id,
                                            filename_postfix=injection_parameters_filename_postfix, file_format='txt')
        if os.path.exists(file_path) and not add_to_existing_file:
            with open(file_path, 'w'):
                pass  # will delete the contents of the file

        # templs_prepared_for_cos_fd = np.conj(np.load(os.path.join(Default_template_bank_folder,
        #                                                           Default_templates_fd_for_injections_filename), allow_pickle=True))
        # template_support = 2 * (len(templs_prepared_for_cos_fd[0]) - 1)
        # dt = tm.dt_lc
        # freqs = np.fft.rfftfreq(template_support, d=dt)
        # effective_wf_fd = sim.get_saved_wf_fd(freqs=freqs, dt=tm.dt_lc)
        # # whitened, normalized, fourier domain, conjugated

        # The parity of the bank is odd, but the length 256 is even, so some tricks with time
        # t = tm.time(t_tot=(template_support + 1) * tm.dt_lc, parity='odd')[1:]
        t = tm.time(t_tot=(256 + 1) * tm.dt_lc, parity='odd')[1:]
        # # -> make it conj for the ease of cosine
        # # The following is needed to calculate the correct cosines from the vectors in the Fourier domain. Only works for even length
        # cosine_coeffs = np.ones(len(templs_prepared_for_cos_fd[0])) * 2
        # cosine_coeffs[0] = 1.
        # cosine_coeffs[-1] = 1.
        # cosine_coeffs = cosine_coeffs / template_support
        # templs_prepared_for_cos_fd = templs_prepared_for_cos_fd * cosine_coeffs

        # ##########################################################################################
        # INCORRECT ###############################################################################
        # ##########################################################################################
        if self.per_wing_injection_and_slides is None:
            self.per_wing_injection_and_slides = per_wing_injection_relative * self.best_per

        # ####################################################################
        # All the things with the occurrence rates and densities:
        log2_rp_re_edges, log10_period_edges, occurrences_table = get_occurrence_table()  # Per one star
        units_of_re_to_units_of_r_star_factor = 1 / (self.stellar_radius * R_sun_R_e)
        log10_best_per_low = np.log10(self.best_per - self.per_wing_injection_and_slides)
        log10_best_per_up = np.log10(self.best_per + self.per_wing_injection_and_slides)
        # log10_period_edges = np.log10(period_edges)
        self.log10_best_period_edges = (log10_best_per_low, log10_best_per_up)
        log2_rp_edges = np.log2(units_of_re_to_units_of_r_star_factor * 2. ** log2_rp_re_edges)

        # We do not know occurrence priors for periods > 400 days, so just use the same ones as for 400 days
        # Number of planets of all radii in log day (per 100 stars):
        # For other periods, it will be checked right now:
        row_upper = len(occurrences_table) - 1
        for i in range(len(log10_period_edges) - 1):
            if (log10_best_per_up > log10_period_edges[i]) and (log10_best_per_up <= log10_period_edges[i + 1]):
                row_upper = i
                break
        row_lower = len(occurrences_table) - 1
        for i in range(len(log10_period_edges) - 1):
            if (log10_best_per_low > log10_period_edges[i]) and (log10_best_per_low <= log10_period_edges[i + 1]):
                row_lower = i
                break
        # planets per one star in this period range for different radii bins:
        # (if the period range covered several bins), we take weighted sum. we do not do any smoothing for now, just take bins
        # to be uniform (they are log bins)
        planets_of_radii_in_per_range_one_star = np.zeros(len(occurrences_table[-1]))
        for i in range(row_lower, row_upper + 1):
            row_density = occurrences_table[i] / (log10_period_edges[i + 1] - log10_period_edges[i])
            if i == len(occurrences_table) - 1 and log10_best_per_up > log10_period_edges[len(log10_period_edges) - 1]:
                up_edge = log10_best_per_up
            else:
                up_edge = min(log10_best_per_up, log10_period_edges[i + 1])
            if i == len(occurrences_table) - 1 and log10_best_per_low > log10_period_edges[len(log10_period_edges) - 1]:
                low_edge = log10_best_per_low
            else:
                low_edge = max(log10_best_per_low, log10_period_edges[i])
            planets_of_radii_in_per_range_one_star += row_density * (up_edge - low_edge)
        self.planets_of_radii_in_per_range_one_star = planets_of_radii_in_per_range_one_star
        rp_numbers = (np.ceil(planets_of_radii_in_per_range_one_star /
                              np.sum(planets_of_radii_in_per_range_one_star) * n_rp)).astype(int)
        rp_numbers_total = np.sum(rp_numbers)

        # ####################################################################

        # Taking into account the probability to transit.
        # for example, d_ecc means that every simulation accounts for the parameter space of the size d_ecc, and this will
        # only be made once. The allowed parameter space e_allowed will be covered by the proper number of d_ecc.
        # Applying the factor to every planet makes sure we count not one planet but e_allowed/e_possible planets per
        # simulated star. In the end, this will be multiplied by the number of planets existing around stars. With the
        # factor, we get the number of planets existing, transiting and observed. The distinction between transiting
        # and observed exists because we observe not all the durations etc.
        ecc_factor = d_ecc / 1.
        w_factor = 1. / n_w
        i_factor = d_inc / 90.
        transit_prob_factor = ecc_factor * w_factor * i_factor

        # effective_planet_count = [0. for _ in range(tm.N_templates)]
        # templates_parameter_sets = [[] for _ in range(tm.N_templates)]
        injection_parameter_sets = []
        # injection_durations = []
        n_star_simulations = 0
        n_star_transiting = 0
        # ##########################################################################################
        # INCORRECT PERIODS########################################################################
        # ##########################################################################################
        for period in np.random.uniform(self.best_per - self.per_wing_injection_and_slides,
                                        self.best_per + self.per_wing_injection_and_slides, n_per):
            # Attention, this is a linear uniform distribution, although we are using logarithmic occurrences.
            a = (G / (4 * np.pi ** 2) * (period * 24 * 60 * 60) ** 2) ** (1 / 3) * self.m_1_3_r_s * (
                    M_sun ** (1 / 3) / R_sun)  # 3rd Kepler law #in units of R
            # a = (G * M /(4*np.pi**2) * (per*24*60*60)**2)**(1/3) / R #3rd Kepler law #in units of R
            # r_min = a*(1-e) -> 2*r_star=a*(1-e_max)
            n_star_simulations += rp_numbers_total
            e_max = min(1 - 2 / a, 0.97)  # for some reason, 0.98 gets computationally stuck in barman sometimes, not clear why
            if e_max < 0:
                e_range = [0.]
            else:
                # n_eccs = max(1, int(np.round(e_max / d_ecc)))
                # e_range = np.linspace(0, e_max, n_eccs, endpoint=False)
                e_range = np.arange(0, e_max, d_ecc)
                if len(e_range) == 0:
                    e_range = [0.]
                else:
                    e_range += np.random.uniform(0, d_ecc, len(e_range))
                    e_range = e_range[e_range < e_max]
            for ecc in e_range:
                ws = np.linspace(0, 360, n_w, endpoint=False)
                ws += np.random.uniform(0, (ws[1] - ws[0]), len(ws))
                for w in ws:
                    # i_max = np.arccos(1/a)*180/np.pi
                    i_max = np.arccos(1 / a * (1 + ecc * np.cos((w-90) / 180 * np.pi)) / (1 - ecc ** 2)) * 180 / np.pi
                    incls = np.arange(90, i_max, -d_inc)  # there will be added a random, so it is ok that the final point is not included
                    if len(incls) > 1:
                        incls -= np.random.uniform(0, d_inc, len(incls))
                        incls = incls[incls >= i_max]
                    for inc in incls:
                        rp = 0.02  # assume the shape does not depend on the planet radius, although it is not exactly correct
                        flux_obs = 1. - tm.flux_par(t=t, params=None, per=period, rp=rp, a=a, inc=inc, ecc=ecc, w=w, u1=u[0], u2=u[1], t0=0,
                                                    supersample_factor=20, exp_time=tm.dt_lc)
                        # we built the bank to find only certain durations
                        len_zeros = len(np.where(np.isclose(flux_obs, 0))[0])  # to make sure transit is not wider than the time
                        dur = tm.templ_dur_hmax_fun(flux_obs) * tm.dt_lc
                        if np.linalg.norm(flux_obs) > 0:
                            n_star_transiting += rp_numbers_total
                        if (np.linalg.norm(flux_obs) > 0) and (tm.dt_lc * 2.1 <= dur) and (dur <= 1.5) and len_zeros >= 4:
                            # # -> these were the requirements for the bank on the duration
                            depth_factor = np.max(flux_obs) / rp ** 2
                            rp_rs_min, rp_rs_max = self.get_detectable_radius_flux(flux=flux_obs, period=period, depth_factor=depth_factor,
                                                                                   snr_min=snr_min_injected,
                                                                                   stellar_radius=self.stellar_radius, units='star')
                            # min_rp_re.append(rp_re_min)
                            rps = []
                            for i in range(len(log2_rp_edges) - 1):
                                rps = rps + list(2. ** (np.random.uniform(log2_rp_edges[i], log2_rp_edges[i + 1], rp_numbers[i])))
                            np.random.shuffle(rps)
                            rps = np.array(rps)
                            rps = rps[rps >= rp_rs_min]
                            rps = rps[rps <= rp_rs_max]
                            # -> smaller radii will just give very small statistics, so it will not be important for the density at trigger
                            for rp in rps:
                                injection_parameter_sets.append([period, a, inc, ecc, w, u[0], u[1], rp, dur])

        # Saving the injection parameters for all the templates, although we will only need for one template.
        # Note that the number of injections is different for different templates, and this reflects the template prior.
        # The number of injections in the run will be then renormalized to match the required number.
        file_path = af.get_saving_file_path(output_location=self.output_location, target_id=self.target_id,
                                            filename_postfix=all_injection_parameters_filename_postfix, file_format='npy')
        np.save(file_path, np.array(injection_parameter_sets, dtype=float), allow_pickle=True)
        search_prob_factor = transit_prob_factor * len(injection_parameter_sets) / n_star_simulations
        self.transit_prob_factor = n_star_transiting / n_star_simulations
        self.search_prob_factor = search_prob_factor
        self.transiting_planets_in_search = np.sum(self.planets_of_radii_in_per_range_one_star) * self.search_prob_factor
        # planets in universe for these periods per star
        # from them transiting
        # from them participating in the search
        # effective number of planets in search (for histogram normalization)
        selection_indices = np.random.choice(np.arange(len(injection_parameter_sets)), n_injections,
                                             replace=True if len(injection_parameter_sets) < n_injections else False)
        self.injection_par_list = list(np.array(injection_parameter_sets, dtype=object)[selection_indices])

        self.save_injection_parameters()
        self.pickle_small_attributes()

    def save_injection_parameters(self, injection_parameters_filename_postfix=Injection_parameters_filename_postfix):
        """
        :param injection_parameters_filename_postfix:
        :type injection_parameters_filename_postfix:
        :return:
        :rtype:
        """
        # We wanted to just save parameters inside the generate_injection_parameters function because anyway we do not create attribute
        # of injection parameters sets because we will probably not use it simultaneously. But we decided that maybe we will.
        file_path = af.get_saving_file_path(output_location=self.output_location, target_id=self.target_id,
                                            filename_postfix=injection_parameters_filename_postfix, file_format='txt')
        with open(file_path, 'w') as fil1:
            print('# period[days], a/R, inc[deg 90 or less], ecc, w[deg], u_0, u_1, rp/R, dur_hmax_days', file=fil1)
            for i in range(len(self.injection_par_list)):
                print(*self.injection_par_list[i], file=fil1)

    def delete_injection_parameters_file(self, injection_parameters_filename_postfix=Injection_parameters_filename_postfix):
        file_path = af.get_saving_file_path(output_location=self.output_location, target_id=self.target_id,
                                            filename_postfix=injection_parameters_filename_postfix, file_format='txt')
        if os.path.exists(file_path):
            os.remove(file_path)

    def load_injection_parameters(self, injection_parameters_index=0,
                                  injection_parameters_filename_postfix=Injection_parameters_filename_postfix):
        file_path = af.get_saving_file_path(output_location=self.output_location, target_id=self.target_id,
                                            filename_postfix=injection_parameters_filename_postfix, file_format='txt')

        try:
            with open(file_path, 'r') as fil:
                lines = fil.readlines()
        except (EOFError, IOError, OSError, ValueError) as error:
            np.random.seed(None)
            success = False
            rep_err = error
            # -> it can be that another process is trying to open the same file, so we will wait
            for _ in range(3):
                time.sleep(1. + np.random.rand() * 3.)
                try:
                    with open(file_path, 'r') as fil:
                        lines = fil.readlines()
                    success = True
                except (EOFError, IOError, OSError, ValueError) as error:
                    rep_err = error
                if success:
                    break
            if not success:
                raise IOError(str(type(rep_err)) + " " + str(rep_err))

        # with open(file_path, 'r') as fil:
        #     lines = fil.readlines()
        lines = [line for line in lines if (line.strip()[0] != '#' and len(line.strip()) > 3)]

        injection_params = lines[injection_parameters_index].split()
        per_inj = float(injection_params[0])
        a_inj = float(injection_params[1])
        inc_inj = float(injection_params[2])
        ecc_inj = float(injection_params[3])
        w_inj = float(injection_params[4])
        u1_inj = float(injection_params[5])
        u2_inj = float(injection_params[6])
        rp_inj = float(injection_params[7])
        dur_inj = float(injection_params[8])
        # ' a/R, inc[deg from 90 down], ecc, w[deg], u_0, u_1, rp/R, dur_hmax_days'
        return per_inj, a_inj, inc_inj, ecc_inj, w_inj, u1_inj, u2_inj, rp_inj, dur_inj

    def run_injection_search(self, injection_load_parameters_index=0, per_wing_detection=Per_wing_injection_detection,
                             injections_filename_postfix=Injections_filename_postfix,
                             overlap_period_bins=spip.P_bins_overlap):  # , n_best_saving=N_best_saving_slides):
        """
        :param injections_filename_postfix:
        :type injections_filename_postfix:
        :param injection_load_parameters_index:
        :type injection_load_parameters_index:
        :param per_wing_detection: days to the left and the same to the right in search
        :type per_wing_detection: float
        :param overlap_period_bins: when creating period chunks, extend each chunk from the two sides by
            this number of bins for them to overlap. is needed for coherent score integral in order for
            the maximum not to be on the edge
        :type overlap_period_bins: int
        :return:
        :rtype:
        """
        # sim_per_job = 20  # with this, 10000 injections in 500 jobs should finish during roughly an hour
        # templates_durations_days = self.templates_durations_hours / 24

        per_inj, a_inj, inc_inj, ecc_inj, w_inj, u1_inj, u2_inj, rp_inj, dur_inj = self.load_injection_parameters(
            injection_parameters_index=injection_load_parameters_index)
        # This period range is for injection, not for search:

        phase_inj = np.random.uniform(0, per_inj)

        tms, flxs = af.get_data_fun(target_id=self.target_id, location=self.data_location)
        first_transit_time_inj = np.nanmin(tms[0]) + phase_inj
        flxs_inj = sim.inject_planet_to_data(times=tms, fluxes=flxs, period=per_inj, first_transit_center=first_transit_time_inj,
                                             rp=rp_inj, a=a_inj, inc=inc_inj, ecc=ecc_inj, w=w_inj, u1=u1_inj, u2=u2_inj,
                                             supersample_factor=20, return_template=False)
        p_max = per_inj + per_wing_detection
        p_min = per_inj - per_wing_detection

        self.klc = spip.KeplerLightCurve(target_id=self.target_id, times=tms, fluxes=flxs_inj, get_data=False,
                                         data_location=self.data_location, m_1_3_r_s=self.m_1_3_r_s)
        if self.is_koi:  # masking koi planets
            self.mask_koi_planet(modify_self_klc=True, return_durations_transit_times=False, include_candidates=False)
        # TODO SPECIFY AROUND WHICH PERIOD TO RUN!
        self.klc.mask_ephemeris(period=self.best_per, durations=2*dur_inj, transit_times=self.best_ftt)
        # Mask the suspected real transit. Warning, can kill some of injected signal
        self.klc.condition_data()
        self.klc.condition_templates()

        self.klc.detect_single_transits(apply_veto=True, gaussianize=True)
        self.klc.connect_quarters()
        self.klc.interleave_shifts(delete_fulls=True)
        self.klc.periodicity_search(pmin=p_min, pmax=p_max, overlap_period_bins=overlap_period_bins, force_interleave_shifts=False)
        # self.klc.save_best_candidates(n_best=n_best_saving, save_to_file=False)
        self.klc.get_best_trigger(apply_veto_mes=True, in_gaussianized=False)
        self.klc.get_best_trigger(apply_veto_mes=True, in_gaussianized=True)
        retrieved_stat = self.klc.best_stat
        retrieved_period = self.klc.best_period
        retrieved_first_transit_time = self.klc.best_ftt
        retrieved_gauss_stat = self.klc.best_stat_gaussianized
        retrieved_integral_score = self.klc.best_integral_score
        retrieved_g_stat = self.klc.best_g_stat
        retrieved_g_period = self.klc.best_g_period
        retrieved_g_first_transit_time = self.klc.best_g_ftt
        retrieved_g_gauss_stat = self.klc.best_g_stat_gaussianized
        retrieved_g_integral_score = self.klc.best_g_integral_score

        file_path = af.get_saving_file_path(output_location=self.output_location, target_id=self.target_id,
                                            filename_postfix=injections_filename_postfix, file_format='txt')

        try:
            with open(file_path, 'a') as fil1:
                print(per_inj, a_inj, inc_inj, ecc_inj, w_inj, u1_inj, u2_inj, rp_inj, dur_inj, first_transit_time_inj,
                      retrieved_stat, retrieved_period, retrieved_first_transit_time, retrieved_gauss_stat, retrieved_integral_score,
                      retrieved_g_stat, retrieved_g_period, retrieved_g_first_transit_time, retrieved_g_gauss_stat,
                      retrieved_g_integral_score,
                      sep=' ', end='\n', file=fil1)
        except (EOFError, IOError, OSError, ValueError) as error:
            np.random.seed(None)
            success = False
            rep_err = error
            # -> it can be that another process is trying to open the same file, so we will wait
            for _ in range(3):
                time.sleep(1. + np.random.rand() * 3.)
                try:
                    with open(file_path, 'a') as fil1:
                        print(per_inj, a_inj, inc_inj, ecc_inj, w_inj, u1_inj, u2_inj, rp_inj, dur_inj, first_transit_time_inj,
                              retrieved_stat, retrieved_period, retrieved_first_transit_time, retrieved_gauss_stat,
                              retrieved_integral_score,
                              retrieved_g_stat, retrieved_g_period, retrieved_g_first_transit_time, retrieved_g_gauss_stat,
                              retrieved_g_integral_score,
                              sep=' ', end='\n', file=fil1)
                    success = True
                except (EOFError, IOError, OSError, ValueError) as error:
                    rep_err = error
                if success:
                    break
            if not success:
                raise IOError(str(type(rep_err)) + " " + str(rep_err))

        # with open(file_path, 'a') as fil1:
        #     print(per_inj, a_inj, inc_inj, ecc_inj, w_inj, u1_inj, u2_inj, rp_inj, dur_inj,  first_transit_time_inj,
        #           retrieved_stat, retrieved_period, retrieved_first_transit_time, retrieved_gauss_stat, retrieved_integral_score,
        #           retrieved_g_stat, retrieved_g_period, retrieved_g_first_transit_time,
        #           retrieved_g_gauss_stat, retrieved_g_integral_score,
        #           sep=' ', end='\n', file=fil1)

    def retrieve_injections(self, max_phase_mismatch_to_dur=Max_phase_mismatch_to_dur, file_path=None,
                            injections_filename_postfix=Injections_filename_postfix):

        for status in ['success', 'bad']:
            self.__dict__['per_inj_' + status] = []
            self.__dict__['a_inj_' + status] = []
            self.__dict__['inc_inj_' + status] = []
            self.__dict__['ecc_inj_' + status] = []
            self.__dict__['w_inj_' + status] = []
            self.__dict__['rp_inj_' + status] = []
            self.__dict__['dur_inj_' + status] = []
            self.__dict__['first_transit_time_inj_' + status] = []
            self.__dict__['retrieved_stat_' + status] = []
            self.__dict__['retrieved_period_' + status] = []
            self.__dict__['retrieved_first_transit_time_' + status] = []
            self.__dict__['retrieved_gauss_stat_' + status] = []
            self.__dict__['retrieved_integral_score_' + status] = []
            self.__dict__['retrieved_g_stat_' + status] = []
            self.__dict__['retrieved_g_period_' + status] = []
            self.__dict__['retrieved_g_first_transit_time_' + status] = []
            self.__dict__['retrieved_g_gauss_stat_' + status] = []
            self.__dict__['retrieved_g_integral_score_' + status] = []
        if file_path is None:
            file_path = af.get_saving_file_path(output_location=self.output_location, target_id=self.target_id,
                                            filename_postfix=injections_filename_postfix, file_format='txt')
        with open(file_path, 'r') as fil1:
            inj_lines = fil1.readlines()

        for inj_line0 in inj_lines:
            inj_line = inj_line0.strip()
            inj_line = inj_line.strip('\x00')
            if (len(inj_line) > 5) and (inj_line[0] != '#'):
                per_inj, a_inj, inc_inj, ecc_inj, w_inj, u1_inj, u2_inj, rp_inj, dur_inj, first_transit_time_inj, \
                    retrieved_stat, retrieved_period, retrieved_first_transit_time, retrieved_gauss_stat, retrieved_integral_score, \
                    retrieved_g_stat, retrieved_g_period, retrieved_g_first_transit_time, retrieved_g_gauss_stat, \
                    retrieved_g_integral_score = inj_line.split()
                per_inj = float(per_inj)
                a_inj = float(a_inj)
                inc_inj = float(inc_inj)
                ecc_inj = float(ecc_inj)
                w_inj = float(w_inj)
                # u1_inj = float(u1_inj)
                # u2_inj = float(u2_inj)
                rp_inj = float(rp_inj)
                dur_inj = float(dur_inj)
                first_transit_time_inj = float(first_transit_time_inj)
                retrieved_stat = float(retrieved_stat)
                retrieved_period = float(retrieved_period)
                retrieved_first_transit_time = float(retrieved_first_transit_time)
                retrieved_gauss_stat = float(retrieved_gauss_stat)
                retrieved_integral_score = float(retrieved_integral_score)
                retrieved_g_stat = float(retrieved_g_stat)
                retrieved_g_period = float(retrieved_g_period)
                retrieved_g_first_transit_time = float(retrieved_g_first_transit_time)
                retrieved_g_gauss_stat = float(retrieved_g_gauss_stat)
                retrieved_g_integral_score = float(retrieved_g_integral_score)

                max_phase_mismatch = 2 * dur_inj * max_phase_mismatch_to_dur
                if np.abs(first_transit_time_inj % per_inj
                          - float(retrieved_first_transit_time) % per_inj) > max_phase_mismatch:  # _rel * per_inj:
                    status = 'bad'
                else:
                    status = 'success'

                self.__dict__['per_inj_' + status].append(per_inj)
                self.__dict__['a_inj_' + status].append(a_inj)
                self.__dict__['inc_inj_' + status].append(inc_inj)
                self.__dict__['ecc_inj_' + status].append(ecc_inj)
                self.__dict__['w_inj_' + status].append(w_inj)
                self.__dict__['rp_inj_' + status].append(rp_inj)
                self.__dict__['dur_inj_' + status].append(dur_inj)
                self.__dict__['first_transit_time_inj_' + status].append(first_transit_time_inj)
                self.__dict__['retrieved_stat_' + status].append(retrieved_stat)
                self.__dict__['retrieved_period_' + status].append(retrieved_period)
                self.__dict__['retrieved_first_transit_time_' + status].append(retrieved_first_transit_time)
                self.__dict__['retrieved_gauss_stat_' + status].append(retrieved_gauss_stat)
                self.__dict__['retrieved_integral_score_' + status].append(retrieved_integral_score)
                self.__dict__['retrieved_g_stat_' + status].append(retrieved_g_stat)
                self.__dict__['retrieved_g_period_' + status].append(retrieved_g_period)
                self.__dict__['retrieved_g_first_transit_time_' + status].append(retrieved_g_first_transit_time)
                self.__dict__['retrieved_g_gauss_stat_' + status].append(retrieved_g_gauss_stat)
                self.__dict__['retrieved_g_integral_score_' + status].append(retrieved_g_integral_score)

        self.number_of_success_injections = len(self.per_inj_success)
        self.number_of_injections = len(self.per_inj_bad) + len(self.per_inj_success)

    def _get_bin_min_bin_max(self, gaussianized=False):
        if gaussianized:
            retrieved_stat_success = self.retrieved_gauss_stat_success
            retrieved_stat_bad = self.retrieved_gauss_stat_bad
            stats_slides = self.gauss_stats_slides
            best_stat = self.best_gauss_stat
        else:
            retrieved_stat_success = self.retrieved_stat_success
            retrieved_stat_bad = self.retrieved_stat_bad
            stats_slides = self.stats_slides
            best_stat = self.best_stat

        if best_stat is not None:
            trig_val = best_stat
        else:
            trig_val = np.nan

        if (retrieved_stat_bad is not None) and (retrieved_stat_success is not None):
            conc_inj_stat = np.concatenate((np.abs(retrieved_stat_success), np.abs(retrieved_stat_bad)))
            inj_min = np.nanmin(conc_inj_stat[np.abs(conc_inj_stat) != np.inf])
            inj_max = np.nanmax(conc_inj_stat[np.abs(conc_inj_stat) != np.inf])
        else:
            inj_min = np.nan
            inj_max = np.nan

        if stats_slides is not None:
            slides_min = np.nanmin(stats_slides[np.abs(stats_slides) != np.inf])
            slides_max = np.nanmax(stats_slides[np.abs(stats_slides) != np.inf])
        else:
            slides_min = np.nan
            slides_max = np.nan

        bin_max = np.nanmax([inj_max, slides_max, trig_val])
        bin_min = np.nanmin([inj_min, slides_min, trig_val])
        return bin_min, bin_max

    def retrieve_background(self):
        fil_slides = [ff for ff in os.listdir(self.output_location) if
                      ((str(self.target_id) in ff) and (Slides_res_filename_postfix in ff))][0]
        res_slides = np.load(os.path.join(self.output_location, fil_slides), allow_pickle=True)
        self.stats_slides = res_slides[:, 0]
        self.pers_slides = res_slides[:, 1]
        self.first_times_slides = res_slides[:, 2]
        self.gauss_stats_slides = res_slides[:, 3]
        self.int_scores_slides = res_slides[:, 4]
        self.stats_g_slides = res_slides[:, 5]
        self.pers_g_slides = res_slides[:, 6]
        self.first_times_g_slides = res_slides[:, 7]
        self.gauss_stats_g_slides = res_slides[:, 8]
        self.int_scores_g_slides = res_slides[:, 9]

        self.n_slides = len(res_slides)  # we now save one per star, so it is len!
        # self.n_slides = int(
        #     fil_slides[len(self.target_id + '_' + Slides_stats_filename_postfix + '_'): fil_slides.index('.')])

    def make_histogram_bins(self):
        if self.stats_slides is None:
            self.retrieve_background()
        if self.retrieved_stat_success is None:
            self.retrieve_injections()
        bin_min, bin_max = self._get_bin_min_bin_max()
        self.bins_stat_histogram = np.arange(bin_min, bin_max, Histogram_stats_binwid)
        bin_min, bin_max = self._get_bin_min_bin_max(gaussianized=True)
        self.bins_gauss_stat_histogram = np.arange(bin_min, bin_max, Histogram_stats_binwid)

    def make_background_histogram(self, bins=None, bin_min=None, bin_max=None):
        if self.stats_slides is None:
            self.retrieve_background()

        if bins is not None:
            self.bins_stat_histogram_bg = bins
            self.bins_gauss_stat_histogram_bg = bins
        else:
            if self.bins_stat_histogram is not None:
                self.bins_stat_histogram_bg = self.bins_stat_histogram
            else:
                bin_mi, bin_ma = self._get_bin_min_bin_max()
                if bin_min is None:
                    bin_min = bin_mi
                if bin_max is None:
                    bin_max = bin_ma
                self.bins_stat_histogram_bg = np.arange(bin_min, bin_max, Histogram_stats_binwid)
            if self.bins_gauss_stat_histogram is not None:
                self.bins_gauss_stat_histogram_bg = self.bins_gauss_stat_histogram
            else:
                bin_mi, bin_ma = self._get_bin_min_bin_max(gaussianized=True)
                if bin_min is None:
                    bin_min = bin_mi
                if bin_max is None:
                    bin_max = bin_ma
                self.bins_gauss_stat_histogram_bg = np.arange(bin_min, bin_max, Histogram_stats_binwid)

        self.distr_stat_h0 = np.histogram(self.stats_slides, bins=self.bins_stat_histogram_bg)[0]
        self.distr_stat_h0 = self.distr_stat_h0 / self.n_slides
        self.distr_gauss_stat_h0 = np.histogram(self.gauss_stats_slides, bins=self.bins_gauss_stat_histogram_bg)[0]
        self.distr_gauss_stat_h0 = self.distr_gauss_stat_h0 / self.n_slides

    def make_injections_histogram(self, bins=None, bin_min=None, bin_max=None):
        # TODO!
        if self.retrieved_stat_success is None:
            self.retrieve_injections()

        if bins is not None:
            self.bins_stat_histogram_inj = bins
            self.bins_gauss_stat_histogram_inj = bins
        else:
            if self.bins_stat_histogram is not None:
                self.bins_stat_histogram_inj = self.bins_stat_histogram
            else:
                bin_mi, bin_ma = self._get_bin_min_bin_max()
                if bin_min is None:
                    bin_min = bin_mi
                if bin_max is None:
                    bin_max = bin_ma
                self.bins_stat_histogram_inj = np.arange(bin_min, bin_max, Histogram_stats_binwid)

            if self.bins_gauss_stat_histogram is not None:
                self.bins_gauss_stat_histogram_inj = self.bins_gauss_stat_histogram
            else:
                bin_mi, bin_ma = self._get_bin_min_bin_max(gaussianized=True)
                if bin_min is None:
                    bin_min = bin_mi
                if bin_max is None:
                    bin_max = bin_ma
                self.bins_gauss_stat_histogram_inj = np.arange(bin_min, bin_max, Histogram_stats_binwid)

        self.distr_stat_inj_all = np.histogram(np.array(self.retrieved_stat_bad + self.retrieved_stat_success),
                                               bins=self.bins_stat_histogram_inj)[0]
        self.distr_stat_inj_all = (self.distr_stat_inj_all / self.number_of_injections * self.search_prob_factor)
        self.distr_stat_inj_success = np.histogram(np.array(self.retrieved_stat_success), bins=self.bins_stat_histogram_inj)[0]
        self.distr_stat_inj_success = (self.distr_stat_inj_success / self.number_of_injections * self.search_prob_factor)

        # same for gaussianized
        self.distr_gauss_stat_inj_all = np.histogram(np.array(self.retrieved_gauss_stat_bad + self.retrieved_gauss_stat_success),
                                               bins=self.bins_gauss_stat_histogram_inj)[0]
        self.distr_gauss_stat_inj_all = (self.distr_gauss_stat_inj_all / self.number_of_injections * self.search_prob_factor)
        self.distr_gauss_stat_inj_success = np.histogram(np.array(self.retrieved_gauss_stat_success),
                                                         bins=self.bins_gauss_stat_histogram_inj)[0]
        self.distr_gauss_stat_inj_success = (self.distr_gauss_stat_inj_success / self.number_of_injections * self.search_prob_factor)

    def calculate_significance(self, remake_histograms=True):
        # TODO!
        if remake_histograms or self.bins_stat_histogram is None:
            self.make_histogram_bins()
            self.make_injections_histogram(bins=self.bins_stat_histogram)
            self.make_background_histogram(bins=self.bins_stat_histogram)
        # MAKE EXTRAPOLATION
        # Determine intersection with horizontal axis in log scale:
        self.extrapolation_coefs_stat_h0_hist = af.get_histogram_extrapolation_coefs_in_log(
            self.distr_stat_h0, self.bins_stat_histogram)  # attention: extrapolating only the direct bg
        # self.extrapolation_coefs_stat_h0_hist[1] is supposed to be negative because the trend is decreasing
        # TODO: SELECT STATISTIC!
        self.prob_density_h0_trigger = np.exp(self.best_stat * self.extrapolation_coefs_stat_h0_hist[1]
                                              + self.extrapolation_coefs_stat_h0_hist[0])
        # Follows from the integration of the non-log distribution over MES[SNR^2]
        self.false_alarm_prob_trigger = (1 / np.abs(self.extrapolation_coefs_stat_h0_hist[1])
                                         * np.exp(self.extrapolation_coefs_stat_h0_hist[0] + self.extrapolation_coefs_stat_h0_hist[
                                                                                                            1] * self.best_stat))
        bin_inds_close_trigger = np.sort(np.argsort(np.abs(self.bins_stat_histogram - self.best_stat))[:2])  # find edges of this bin
        self.prob_density_h1_trigger = self.distr_stat_inj_all[bin_inds_close_trigger[0]]  # use the left edge index to access the bin
        self.prob_density_planet_not_noise = self.prob_density_h1_trigger / (self.prob_density_h1_trigger + self.prob_density_h0_trigger)

    def analyze_injections_and_background(self, max_phase_mismatch_to_dur=Max_phase_mismatch_to_dur):
        """
        :param max_phase_mismatch_to_dur:
        :type max_phase_mismatch_to_dur:
        :return:
        :rtype:
        """
        self.retrieve_background()
        self.retrieve_injections(max_phase_mismatch_to_dur=max_phase_mismatch_to_dur)
        self.make_histogram_bins()
        self.make_injections_histogram(bins=self.bins_stat_histogram)
        self.make_background_histogram(bins=self.bins_stat_histogram)
        self.calculate_significance(remake_histograms=False)

    def get_ses_for_postprocessing(self, no_filters=True, nsig_outliers_0_pos=5, nsig_outliers_0_neg=15,
                                   load_no_veto_no_score_cor=True,
                                   filename_postfix=No_veto_no_score_cor_stats_filename_postfix,
                                   template_index='all', apply_score_correction=True, apply_veto=False):
        # TODO!!!! WHAT IS THIS FUNCTION AND MAKE SURE IT IS NOT USED
        if load_no_veto_no_score_cor:
            file_path = af.get_saving_file_path(output_location=self.output_location, target_id=self.target_id,
                                                filename_postfix=filename_postfix, file_format='npy')
            self.klc = spip.KeplerLightCurve(target_id=self.target_id, get_data=True, data_location=self.data_location,
                                             m_1_3_r_s=self.m_1_3_r_s)
            self.klc.subtract_means()
            self.klc.initialize_templates(template_index=template_index)
            self.klc.stats_raw_no_veto_no_score_cor, self.klc.stat_moving_vars = np.load(file_path, allow_pickle=True)
            if template_index != 'all':
                if not hasattr(template_index, '__len__'):
                    template_index = [template_index]
                indices = [kk for k in template_index for kk in tm.Indices_of_templates[k]]
                self.klc.stats_raw_no_veto_no_score_cor = self.klc.stats_raw_no_veto_no_score_cor[indices]
            self.klc.statistics_raw = self.klc.stats_raw_no_veto_no_score_cor
        else:
            if no_filters:
                self.klc = spip.KeplerLightCurve(target_id=self.target_id, get_data=True, data_location=self.data_location,
                                                 m_1_3_r_s=self.m_1_3_r_s)
                self.klc.condition_data(nsig_outliers_0=nsig_outliers_0_neg, nsig_outliers_0_pos=nsig_outliers_0_pos,
                                        nsig_outliers_0_neg=nsig_outliers_0_neg,
                                        nsig_outliers_w=nsig_outliers_0_neg, nsig_outliers_w_pos=nsig_outliers_0_pos,
                                        nsig_outliers_w_neg=nsig_outliers_0_neg)
                self.klc.condition_templates(template_index=template_index)
                self.klc.detect_single_transits(apply_veto=False, gaussianize=False)
            else:
                self.klc = spip.KeplerLightCurve(target_id=self.target_id, get_data=True, data_location=self.data_location,
                                                 m_1_3_r_s=self.m_1_3_r_s)
                file_path = af.get_saving_file_path(output_location=self.output_location, target_id=self.target_id,
                                                    filename_postfix=Conditioned_data_filename_postfix, file_format='npy')
                if not os.path.exists(file_path):
                    self.klc.condition_data()  # IN OUTLIERS MASK, THE POS AND THE NEG NSIG ARE EQUAL BEC. WE NEED TO INVERT FAIRLY
                    self.klc.condition_templates(template_index=template_index)
                else:
                    file_path = af.get_saving_file_path(output_location=self.output_location, target_id=self.target_id,
                                                        filename_postfix=Conditioned_data_filename_postfix, file_format='npy')
                    self.klc.load_conditioned_data(file=file_path)
                self.klc.initialize_templates(template_index=template_index)
                self.klc.detect_single_transits(apply_veto=apply_veto, gaussianize=apply_score_correction)

    def get_summed_values_no_filters(self, nsig_outliers_0_pos=5, nsig_outliers_0_neg=15):
        pass
        # self.get_ses_for_postprocessing(no_filters=True, nsig_outliers_0_pos=nsig_outliers_0_pos, nsig_outliers_0_neg=nsig_outliers_0_neg,
        #                                 load_no_veto_no_score_cor=True,
        #                                 template_index=self.best_temp, apply_score_correction=False, apply_veto=False)
        # ...

    def estimate_rp(self, nsig_outliers_0_pos=5, nsig_outliers_0_neg=15):
        # TODO WRITE!
        if self.best_ppms_no_filt is None or self.best_ppm_no_filt is None or self.best_ppms_stds_no_filt is None:
            self.get_summed_values_no_filters(nsig_outliers_0_pos=nsig_outliers_0_pos,
                                              nsig_outliers_0_neg=nsig_outliers_0_neg)
        # ampl, ppm, ampls, ampls_stds, normalized_ampls, normalized_ampls_stds, ppms, ppms_stds = \
        #     estimate_ampls_ppms_from_stats(summed_raw_stats, summed_trends, summed_stat_vars, templ_max, trends_stds=None)
        #
        self.rp_to_rs = (self.best_ppm_no_filt / 1e6) ** 0.5  # the radius estimation is subjected to big error and bias!
        self.rp_to_rs_std = (np.nanvar(np.abs(self.best_ppms_no_filt) ** 0.5)
                             + np.nanmean(self.best_ppms_stds_no_filt) / np.abs(self.best_ppm_no_filt)) ** 0.5 / 1e3
        self.rp_to_r_earth = self.rp_to_rs * self.stellar_radius * R_sun_R_e
        self.rp_to_r_earth_std = self.rp_to_rs_std * self.stellar_radius * R_sun_R_e


