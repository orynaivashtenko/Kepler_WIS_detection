#!/usr/bin/env python3

# TODO: change the names in such a way that target id is not the last, because it does not give any understanding what
#  is running in the jobname

from sys import getsizeof
from astropy.io import fits
import pickle
import search_pipeline as spip
import templates.template_module as tm
from simulations import lightcurve_simulation as sim
import auxiliary_functions as af
import utils as ut
import time
import importlib
from pipeline_parameters import *  # all the parameters, physical constants, occurrence table
from templates.template_module import get_templates_durations_from_file
from auxiliary_functions import fold_fluxes
from utils import pad_by_zeros
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 11, 'font.serif': 'Times New Roman', 'lines.linewidth': 2})
Def_col = plt.rcParams['axes.prop_cycle'].by_key()['color']

importlib.reload(spip)
np.set_printoptions(threshold=1000, linewidth=10000)

# import warnings
# warnings.filterwarnings("ignore")

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
                       'gaia_abs_gmag', 'gaia_b_r', 'teff', 'ra', 'dec']

    def __init__(self, target_id='003097222', data_location=Data_location, output_location=Output_location,
                 mes_candidate=Mes_candidate_search,
                 p_min_search=P_min_search, p_max_search=P_max_search, p_chunks_search=P_chunks_search,
                 load_pickled_attributes=False,
                 get_transit_and_binary_info=True,
                 gaia_kepler_file_path=Gaia_kepler_file,
                 koi_dict_file=Koi_dict_file,
                 ttv_dict_file=Ttv_dict_file,
                 villanova_dict_file=Villanova_dict_file,
                 overwrite_parameters_by_unpickle=False):
        self.target_id = target_id.rjust(9, '0')

        data_location = af.get_correct_tid_folder(target_id, data_location)
        output_location = af.get_correct_tid_folder(target_id, output_location)
        # I changed the format of data_location, now all the formats are the same: General_folder/target_id[:4]/target_id

        # # The standard of the data location is that it ends with target_id[:4] because our data storage is organized this way,
        # # and then the code is accessing the particular target_id inside.
        # if os.path.split(data_location)[-1] != target_id[:4]:
        #     if os.path.split(os.path.split(data_location)[0])[1] == target_id[:4]:  # if it is above, return there
        #         data_location = os.path.split(data_location)[0]
        #     elif target_id[:4] in os.listdir(data_location):  # if it is inside, go there
        #         data_location = os.path.join(data_location, target_id[:4])
        #
        # # The output is organized in the same way, but we are operating with the target_id folder already, and if it is not created,
        # # we need to create it here
        # if os.path.split(output_location)[-1] != target_id:
        #     if target_id[:4] in os.listdir(output_location):
        #         output_location = os.path.join(output_location, target_id[:4])
        #     if target_id not in os.listdir(output_location):
        #         os.mkdir(os.path.join(output_location, target_id))
        #     output_location = os.path.join(output_location, target_id)

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
        self.koi_dict = None
        self.ttv_dict = None
        self.is_villanova_binary = None
        self.villanova_dict = None

        self.per_edges = None
        self.p_chunks = p_chunks_search
        self.p_min = p_min_search
        self.p_max = p_max_search
        self.klc = None
        self.conditioning_ses_time = None  # time of the main search conditioning and ses in seconds
        self.periodicity_search_time = None  # time of the main run periodicity search (mes) in seconds

        self.mes_candidate = mes_candidate
        # self.best_trigger_attributes = ['best_stat', 'best_per', 'best_temp', 'best_phase', 'best_first_transit_time', 'best_ppm',
        #                                'best_ampl', 'summed_raw_stats', 'summed_trends', 'summed_vars', 'ampls', 'ppms',
        #                                'best_stat_inv', 'best_per_inv', 'best_temp_inv', 'best_phase_inv', 'best_first_transit_time_inv',
        #                                'best_ppm_inv', 'best_ampl_inv', 'summed_raw_stats_inv', 'summed_trends_inv', 'summed_vars_inv',
        #                                'ampls_inv', 'ppms_inv']
        # for attr in self.best_trigger_attributes:
        #     self.__dict__[attr] = None
        self.templates_durations_hours = None

        self.best_stat = None
        self.best_per = None
        self.best_temp = None
        # self.best_phase = None
        self.best_first_transit_time = None
        self.summed_transit_times = None  # times of values that were summed to produce the best trigger
        self.summed_raw_stats = None  # statistics in units not of snr that were summed to produce the best trigger
        self.summed_trends = None
        self.summed_vars = None

        self.best_stat_inv = None
        self.best_per_inv = None
        self.best_temp_inv = None
        # self.best_phase_inv = None
        self.best_first_transit_time_inv = None
        # self.best_ppm_inv = None
        # self.best_ampl_inv = None
        self.summed_transit_times_inv = None
        self.summed_raw_stats_inv = None
        self.summed_trends_inv = None
        self.summed_vars_inv = None
        self.summed_stat_denoms = None
        # self.ampls_inv = None
        # self.ppms_inv = None

        self.is_threshold_crossing = False
        self.per_wing_injection_and_slides = None
        self.best_temp_injection_par_list = None
        self.template_prior = None
        self.planets_in_best_period_range_best_template_per_star = None
        self.log10_best_period_edges = None

        self.per_inj_success = None
        self.a_inj_success = None
        self.inc_inj_success = None
        self.ecc_inj_success = None
        self.w_inj_success = None
        self.rp_inj_success = None
        self.first_transit_time_inj_success = None
        self.retrieved_mes_success = None
        self.retrieved_first_transit_time_success = None

        self.per_inj_bad = None
        self.a_inj_bad = None
        self.inc_inj_bad = None
        self.ecc_inj_bad = None
        self.w_inj_bad = None
        self.rp_inj_bad = None
        self.first_transit_time_inj_bad = None
        self.retrieved_mes_bad = None
        self.retrieved_first_transit_time_bad = None

        self.number_of_success_injections = None
        self.number_of_injections = None

        self.slides_mes2_h0 = None
        self.slides_inv_mes2_h0 = None

        self.n_slides = None
        self.bins_mes2_histogram = None
        self.bins_mes2_histogram_inj = None
        self.bins_mes2_histogram_bg = None
        self.distr_mes2_h0_direct = None
        self.distr_mes2_h0_inv = None
        self.distr_mes2_h0 = None
        self.distr_mes2_inj_all = None
        self.distr_mes2_inj_success = None
        self.extrapolation_coefs_mes2_h0_hist = None
        self.prob_density_h0_trigger = None
        self.false_alarm_prob_trigger = None
        self.prob_density_h1_trigger = None
        self.prob_density_planet_not_noise = None

        # Elements of estimations:
        # Non-score-corrected values:
        self.best_raw_stats = None
        self.best_stat_denoms = None
        self.best_trends = None
        self.best_ampls = None  # amplitudes of different transits
        self.best_ampls_stds = None
        self.best_ampl = None  # average depth of the transit in parts per million
        self.best_norm_ampls = None
        self.best_norm_ampls_stds = None
        self.best_ppms = None  # depths of different transits in parts per million
        self.best_ppm = None  # average amplitude of the transit, coefficient that multiplies the best template
        self.best_ppms_stds = None
        self.best_temp_depth = None

        self.best_raw_stats_inv = None
        self.best_stat_denoms_inv = None
        self.best_trends_inv = None
        self.best_ampls_inv = None  # amplitudes of different transits
        self.best_ampls_stds_inv = None
        self.best_ampl_inv = None  # average depth of the transit in parts per million
        self.best_norm_ampls_inv = None
        self.best_norm_ampls_stds_inv = None
        self.best_ppms_inv = None  # depths of different transits in parts per million
        self.best_ppm_inv = None  # average amplitude of the transit, coefficient that multiplies the best template
        self.best_ppms_stds_inv = None
        self.best_temp_depth_inv = None

        # Things with freedom for ttvs and templates, probably useless:
        self.optim_raw_stats = None
        self.optim_stat_denoms = None
        self.optim_trends = None
        self.optim_templs = None
        self.optim_ampls = None
        self.optim_ampl = None
        self.optim_ppms = None
        self.optim_ppm = None
        self.ttvs_of_optim = None
        self.rp_to_rs = None  # planetary radius in units of stellar radius, estimate from ppm
        self.rp_to_r_earth = None  # planetary radius in units of Earth radius, estimate from ppm
        self.rp_to_rs_std = None  # std due to the dispersion of measurements in diff.transits (not including measurement error)
        self.rp_to_r_earth_std = None

        self.failures = ''  # if something bad happens, we will write it here
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
                self.mes_candidate = mes_candidate
                self.p_chunks = p_chunks_search
                self.p_min = p_min_search
                self.p_max = p_max_search
        else:
            if self.gaia_kepler_file is not None:
                self.get_fields_from_gaia(gaia_file_path=self.gaia_kepler_file)
            if get_transit_and_binary_info:
                self.get_koi_and_ttv_fields(koi_dict_file=self.koi_dict_file, ttv_dict_file=self.ttv_dict_file)
                self.get_villanova_fields(villanova_dict_file=self.villanova_dict_file)

        if self.is_villanova_binary:
            self.is_good_for_processing = False

    # setattr(instance, attr_name_str, attr_value)  # how to set attributes outside
    # dir(instance)  # get everything

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
            if isinstance(value, (str, float, int, np.float, np.integer, np.ndarray, list, dict)):
                if (((not hasattr(value, '__len__')) or len(value) < 1000) and getsizeof(value) < 80128) or isinstance(value, str):
                    # size in bytes of array 100*100
                    dict_to_pickle[key] = value

        with open(os.path.join(output_location, self.target_id + '_' + pickle_small_attributes_filename_postfix), 'wb') as pickle_file:
            pickle.dump(dict_to_pickle, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

    def unpickle_small_attributes(self, output_location=None,
                                  pickle_small_attributes_filename_postfix=Pickle_target_small_attributes_filename_postfix):
        if output_location is None:
            output_location = self.output_location
        with open(os.path.join(output_location, self.target_id + '_' + pickle_small_attributes_filename_postfix), 'rb') as pickle_file:
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

            for key in self.star_attributes:
                try:
                    self.__dict__[key] = float(self.__dict__[key])
                except Exception as e:
                    self.failures += key + ' error ' + str(e) + '; '
                    self.__dict__[key] = None
            # todo: add fields with units and descriptions. astropy.table can be helpful with that
        gaia_file.close()

    def get_fields_from_info_file(self, gaia_file_path=None):
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
        # numpy.savetxt()
        if output_location is None:
            output_location = self.output_location
        dict_to_print = {}
        for key in self.__dict__:
            value = self.__dict__[key]
            if isinstance(value, (str, float, int, np.float, np.integer, np.ndarray, list, dict)):
                ok_to_print = True
                if isinstance(value, (str, float, int, np.float, np.integer, np.ndarray, list, dict)):
                    if hasattr(value, '__len__') and len(value) > 0:
                        if isinstance(value, dict) and hasattr(value[list(value.keys())[0]], '__len__'):
                            ok_to_print = False  # TODO: maybe make the selection process look at all the dict keys
                        elif ((not isinstance(value, dict)) and (not isinstance(value, str))
                              and (len(value) > 0 and hasattr(value[0], '__len__'))):
                            ok_to_print = False  # not printing nested lists
                        elif (not isinstance(value, str)) and len(value) > 50:
                            # roughly to be able to set summed values for planets with period 30
                            ok_to_print = False
                if ok_to_print:
                    dict_to_print[key] = value
        with open(os.path.join(output_location, self.target_id + '_' + dict_txt_filename_postfix), 'w') as fil:
            for key in dict_to_print:
                print(key, ":", dict_to_print[key], file=fil)

    def print_summary(self, to_file=True, file=None, summary_txt_filename_postfix=Summary_target_txt_filename_postfix):
        if file is None:
            file = os.path.join(self.output_location, self.target_id + '_' + summary_txt_filename_postfix)
        if to_file:
            with open(file, 'w') as fil1:
                for attr in self.star_attributes:
                    print(attr, ':', self.__dict__[attr], file=fil1)
                print('', file=fil1)

                print('best_stat', ':', self.best_stat, file=fil1)
                print('best_per', ':', self.best_per, file=fil1)
                print('best_temp', ':', self.best_temp, file=fil1)
                print('best_first_transit_time', ':', self.best_first_transit_time, file=fil1)
                # if self.best_ppm is not None:
                #     print('best_ppm', ':', self.best_ppm, file=fil1)
                print('', file=fil1)

                print('best_stat_inv', ':', self.best_stat_inv, file=fil1)
                print('best_per_inv', ':', self.best_per_inv, file=fil1)
                print('best_temp_inv', ':', self.best_temp_inv, file=fil1)
                print('best_first_transit_time_inv', ':', self.best_first_transit_time_inv, file=fil1)
                # if self.best_ppm_inv is not None:
                #     print('best_ppm_inv', ':', self.best_ppm_inv, file=fil1)
                print('', file=fil1)
        else:
            for attr in self.star_attributes:
                print(attr, ':', self.__dict__[attr])
            print('')

            print('best_stat', ':', self.best_stat)
            print('best_per', ':', self.best_per)
            print('best_temp', ':', self.best_temp)
            print('best_first_transit_time', ':', self.best_first_transit_time)
            # if self.best_ppm is not None:
            #     print('best_ppm', ':', self.best_ppm)
            print('')

            print('best_stat_inv', ':', self.best_stat_inv)
            print('best_per_inv', ':', self.best_per_inv)
            print('best_temp_inv', ':', self.best_temp_inv)
            print('best_first_transit_time_inv', ':', self.best_first_transit_time_inv)
            # if self.best_ppm_inv is not None:
            #     print('best_ppm_inv', ':', self.best_ppm_inv)
            print('')

    def clean_output_directory(self):
        for ff in os.listdir(self.output_location):
            os.remove(os.path.join(self.output_location, ff))

    def initialize_kepler_light_curve(self, scramble=False, std_time_scrambling=spip.Std_time_scrambling,
                                      data_location=None, output_location=None,
                                      remove_zeroth_quarter=True, remove_quarter_17=True,
                                      use_rfft=spip.Use_rfft):
        if data_location is None:
            data_location = self.data_location
        if output_location is None:
            output_location = self.output_location
        self.klc = spip.KeplerLightCurve(target_id=self.target_id, get_data=True, data_location=data_location,
                                         output_location=output_location,
                                         scramble=scramble, std_time_scrambling=std_time_scrambling,
                                         remove_zeroth_quarter=remove_zeroth_quarter, remove_quarter_17=remove_quarter_17,  # Q17 is removed
                                         use_rfft=use_rfft)
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

    def mask_koi_planet(self, modify_self_klc=True, return_durations_transit_times=False):
        # TODO: ATTENTION: we are also masking candidates!!!!!!! not only confirmed and FP!!!!!
        # TODO why do we mask the planet, and not subtract? We could also subtract villanova ephemeris.
        # TODO: include occultation (secondary transit)
        assert self.is_koi, 'The target is not KOI, so the KOI planet cannot be masked!'
        # Take all the transit times with TTV, take transit duration, take fractional TDV and mask.
        durations_and_times = []
        for i, koi_name in enumerate(self.koi_dict['kepoi_name']):
            transit_durations = None  # just initialization, no meaning
            transit_times = None
            was_in_ttv = False
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
                transit_times = [float(self.koi_dict['koi_time0bk'][i]) + float(self.koi_dict['koi_period'][i]) * nn
                                 for nn in range(int(self.koi_dict['koi_num_transits'][i]))]
                # TODO: -> invalid literal for int() with base 10: ''
                transit_durations = np.ones(len(transit_times)) * float(
                    self.koi_dict['koi_duration'][i]) / 24  # we need it in days, and it was in hours

            if self.klc is not None and modify_self_klc:
                self.klc.mask_ephemeris(durations=transit_durations, transit_times=transit_times)
            if return_durations_transit_times:
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
                       mask_koi_planet=True, save_conditioned_data=True):
        """

        :param save_conditioned_data:
        :type save_conditioned_data:
        :param conditioned_data_filename_postfix:
        :type conditioned_data_filename_postfix:
        :param mask_koi_planet:
        :type mask_koi_planet: s
        :return:
        :rtype:
        """

        t1 = time.time()
        self.klc = spip.KeplerLightCurve(target_id=self.target_id, get_data=True, data_location=self.data_location,
                                         output_location=self.output_location)  # Q17 is removed
        self.bad_fft_check()  # Made on orig_fluxes
        if self.is_koi and mask_koi_planet:
            self.mask_koi_planet(modify_self_klc=True, return_durations_transit_times=False)
        self.klc.condition_data()  # In outlier mask, positive and negative nsig_outliers are the same because we run inversion after.
        self.was_notched = self.klc.was_notched
        self.klc.condition_templates()
        if save_conditioned_data:
            self.klc.save_conditioned_data(filename_postfix=conditioned_data_filename_postfix)
        self.klc.detect_single_transits()
        self.klc.connect_quarters()
        self.klc.interleave_shifts()

        t2 = time.time()
        self.conditioning_ses_time = t2 - t1

    def run_periodicity_search(self, p_min=None, p_max=None, duty_cycle=spip.Duty_cycle,
                               p_chunks=None, mes_candidate=None,
                               conditioned_data_filename_postfix=Conditioned_data_filename_postfix,
                               best_periodic_scores_filename_postfix=Best_periodic_scores_filename_postfix,
                               best_inverted_periodic_scores_filename_postfix=Best_inverted_periodic_scores_filename_postfix,
                               best_periodic_scores_filename=None,
                               best_inverted_periodic_scores_filename=None
                               ):
        """
        :param conditioned_data_filename_postfix:
        :type conditioned_data_filename_postfix:
        :param best_periodic_scores_filename_postfix:
        :type best_periodic_scores_filename_postfix:
        :param best_inverted_periodic_scores_filename_postfix:
        :type best_inverted_periodic_scores_filename_postfix:
        :param p_min:
        :type p_min:
        :param p_max:
        :type p_max:
        :param duty_cycle:
        :type duty_cycle:
        :param p_chunks:
        :type p_chunks:
        :param mes_candidate:
        :type mes_candidate:
        :param best_periodic_scores_filename:
        :type best_periodic_scores_filename:
        :param best_inverted_periodic_scores_filename:
        :type best_inverted_periodic_scores_filename:
        :return:
        :rtype:
        """

        if mes_candidate is not None:
            self.mes_candidate = mes_candidate
        if p_min is not None:
            self.p_min = p_min
        if p_max is not None:
            self.p_max = p_max
        if p_chunks is not None:
            self.p_chunks = p_chunks

        if self.p_chunks is not None:
            self.per_edges = [(self.p_min + (self.p_max - self.p_min) * i / self.p_chunks) for i in range(self.p_chunks + 1)]
            # round was there (up)
            self.per_edges = [[self.per_edges[i], self.per_edges[i + 1]] for i in range(len(self.per_edges) - 1)]
        else:
            self.per_edges = [[self.p_min, self.p_max]]

        if self.klc.statistics_interleaved is None:
            self.klc.connect_quarters()
            self.klc.interleave_shifts()
        elif self.klc.statistics_snr is None:
            self.klc.load_conditioned_data(filename_postfix=conditioned_data_filename_postfix)
            self.klc.initialize_templates()
            self.klc.detect_single_transits()
            self.klc.connect_quarters()
            self.klc.interleave_shifts()

        t1 = time.time()

        best_candidates_concat = []
        best_inv_candidates_concat = []
        for pp in self.per_edges:
            self.klc.sum_periods(pmin=pp[0], pmax=pp[1], duty_cycle=duty_cycle)
            # self.klc.veto_mes()  # we are not doing it at this stage
            self.klc.save_best_candidates(save_to_file=False)
            best_candidates_concat.append(np.array(self.klc.best_periodic_scores_data, dtype=object))
            best_inv_candidates_concat.append(np.array(self.klc.best_inverted_periodic_scores_data, dtype=object))
            self.klc.delete_mes()
        if best_periodic_scores_filename is not None:
            np.save(best_periodic_scores_filename,
                    np.array(best_candidates_concat, dtype=object), allow_pickle=True)
        else:
            np.save(os.path.join(self.output_location, self.target_id + '_' + best_periodic_scores_filename_postfix),
                    np.array(best_candidates_concat, dtype=object), allow_pickle=True)
        if best_inverted_periodic_scores_filename is not None:
            np.save(best_inverted_periodic_scores_filename,
                    np.array(best_inv_candidates_concat, dtype=object), allow_pickle=True)
        else:
            np.save(os.path.join(self.output_location, self.target_id + '_' + best_inverted_periodic_scores_filename_postfix),
                    np.array(best_inv_candidates_concat, dtype=object), allow_pickle=True)

        t2 = time.time()
        self.periodicity_search_time = t2 - t1

        # Check for threshold crossings, and if found, then launch time slides
        self.best_stat = 0
        for best_candidates in best_candidates_concat:  # if there is a big MES score, further processing
            argm = np.argmin(best_candidates[spip.Search_results_index_stat])
            ti, i = argm // len(best_candidates[spip.Search_results_index_stat][0]
                                ), argm % len(best_candidates[spip.Search_results_index_stat][0])
            # the structure is (25,10), where 10 is just 10 best per each template
            smin = best_candidates[spip.Search_results_index_stat][ti][i]  # minimal statistic

            if smin < self.best_stat:
                self.best_stat = smin
                self.best_per = best_candidates[spip.Search_results_index_per][ti][i]
                self.best_temp = ti
                self.best_first_transit_time = best_candidates[spip.Search_results_index_sum_tr_times][ti][i][0]
                self.summed_transit_times = best_candidates[spip.Search_results_index_sum_tr_times][ti][i]
                self.summed_raw_stats = np.array(best_candidates[spip.Search_results_index_sum_stats][ti][i])
                self.summed_trends = np.array(best_candidates[spip.Search_results_index_sum_trends][ti][i])
                self.summed_vars = np.array(best_candidates[spip.Search_results_index_sum_vars][ti][i])
                self.summed_vars[self.summed_vars == 0] = np.nan

        self.best_stat_inv = 0
        for best_candidates in best_inv_candidates_concat:  # if there is a big MES score, further processing
            argm = np.argmax(best_candidates[spip.Search_results_index_stat])  # we took inverted best scores collection, they are positive
            ti, i = argm // len(best_candidates[spip.Search_results_index_stat][0]
                                ), argm % len(best_candidates[spip.Search_results_index_stat][0])
            # the structure is (25,10), where 10 is just 10 best per each template
            smax = best_candidates[spip.Search_results_index_stat][ti][i]  # maximal statistic
            if smax > self.best_stat_inv:
                self.best_stat_inv = smax
                self.best_per_inv = best_candidates[spip.Search_results_index_per][ti][i]
                self.best_temp_inv = ti
                self.best_first_transit_time_inv = best_candidates[spip.Search_results_index_sum_tr_times][ti][i][0]
                self.summed_transit_times_inv = best_candidates[spip.Search_results_index_sum_tr_times][ti][i]
                self.summed_raw_stats_inv = np.array(best_candidates[spip.Search_results_index_sum_stats][ti][i])
                self.summed_trends_inv = np.array(best_candidates[spip.Search_results_index_sum_trends][ti][i])
                self.summed_vars_inv = np.array(best_candidates[spip.Search_results_index_sum_vars][ti][i])
                self.summed_vars_inv[self.summed_vars_inv == 0] = np.nan

    # TODO: add parsing parameters!
    def run_main_search(self, p_min=None, p_max=None,
                        p_chunks=None, mes_candidate=None,
                        mask_koi_planet=True,
                        save_to_threshold_crossing_list=True,
                        conditioned_data_filename_postfix=Conditioned_data_filename_postfix,
                        best_periodic_scores_filename_postfix=Best_periodic_scores_filename_postfix,
                        best_inverted_periodic_scores_filename_postfix=Best_inverted_periodic_scores_filename_postfix,
                        threshold_crossing_list_file=None, threshold_inv_crossing_list_file=None,
                        threshold_crossing_list_location=Threshold_crossing_lists_folder,
                        threshold_inv_crossing_list_location=Threshold_crossing_lists_folder,
                        threshold_crossing_list_filename=Threshold_crossing_list_filename,
                        threshold_inv_crossing_list_filename=Threshold_inv_crossing_list_filename,
                        clean_dir_if_not_threshold_crossing=False):
        """

        :param best_periodic_scores_filename_postfix:
        :type best_periodic_scores_filename_postfix:
        :param best_inverted_periodic_scores_filename_postfix:
        :type best_inverted_periodic_scores_filename_postfix:
        :param p_min:
        :type p_min:
        :param p_max:
        :type p_max:
        :param p_chunks:
        :type p_chunks:
        :param mask_koi_planet:
        :type mask_koi_planet:
        :param mes_candidate:
        :type mes_candidate:
        :param save_to_threshold_crossing_list:
        :type save_to_threshold_crossing_list:
        :param conditioned_data_filename_postfix:
        :type conditioned_data_filename_postfix:
        :param threshold_crossing_list_file:
        :type threshold_crossing_list_file:
        :param threshold_inv_crossing_list_file:
        :type threshold_inv_crossing_list_file:
        :param threshold_crossing_list_location:
        :type threshold_crossing_list_location:
        :param threshold_inv_crossing_list_location:
        :type threshold_inv_crossing_list_location:
        :param threshold_crossing_list_filename:
        :type threshold_crossing_list_filename:
        :param threshold_inv_crossing_list_filename:
        :type threshold_inv_crossing_list_filename:
        :param clean_dir_if_not_threshold_crossing:
        :type clean_dir_if_not_threshold_crossing:
        :return:
        :rtype:
        """
        if (threshold_crossing_list_file is None) and (threshold_crossing_list_location is not None):
            threshold_crossing_list_file = os.path.join(threshold_crossing_list_location, threshold_crossing_list_filename)

        if (threshold_inv_crossing_list_file is None) and (threshold_inv_crossing_list_location is not None):
            threshold_inv_crossing_list_file = os.path.join(threshold_inv_crossing_list_location, threshold_inv_crossing_list_filename)

        self.run_ses_search(conditioned_data_filename_postfix=conditioned_data_filename_postfix, mask_koi_planet=mask_koi_planet)
        self.run_periodicity_search(p_min=p_min, p_max=p_max,
                                    p_chunks=p_chunks, mes_candidate=mes_candidate,
                                    conditioned_data_filename_postfix=conditioned_data_filename_postfix,
                                    best_periodic_scores_filename_postfix=best_periodic_scores_filename_postfix,
                                    best_inverted_periodic_scores_filename_postfix=best_inverted_periodic_scores_filename_postfix)

        if self.best_stat ** 2 >= self.mes_candidate ** 2:
            self.is_threshold_crossing = True
            if save_to_threshold_crossing_list:
                with open(threshold_crossing_list_file, 'a') as fil1:  # will save here these good targets
                    print(self.target_id, self.best_stat, self.best_per, self.best_temp,  # self.best_phase,
                          self.best_first_transit_time,  # self.best_ampl, self.best_ppm,
                          sep=' ', end='\n', file=fil1)

        if self.best_stat_inv ** 2 > self.mes_candidate ** 2:
            self.is_threshold_crossing = True
            if save_to_threshold_crossing_list:
                with open(threshold_inv_crossing_list_file, 'a') as fil1:  # will save here these good targets
                    print(self.target_id, self.best_stat_inv, self.best_per_inv, self.best_temp_inv,  # self.best_phase_inv,
                          self.best_first_transit_time_inv,  # self.best_ampl_inv, self.best_ppm_inv,
                          sep=' ', end='\n', file=fil1)

        if (not self.is_threshold_crossing) and clean_dir_if_not_threshold_crossing:
            # delete conditioned data because we have no space on the disk!
            for ff in os.listdir(self.output_location):
                os.remove(os.path.join(self.output_location, ff))

        self.pickle_small_attributes()
        self.print_dict_file()

    def run_timeslides(self, n_slides=N_slides_chunk,
                       per_wing_slides_relative=Per_wing_injection_and_slides_relative,
                       mask_koi_planet=True,
                       slides_mes_inv_filename_postfix=Slides_mes_inv_filename_postfix,
                       slides_mes_filename_postfix=Slides_mes_filename_postfix,
                       slides_pers_inv_filename_postfix=Slides_pers_inv_filename_postfix,
                       slides_pers_filename_postfix=Slides_pers_filename_postfix,
                       slides_first_times_inv_filename_postfix=Slides_first_times_inv_filename_postfix,
                       slides_first_times_filename_postfix=Slides_first_times_filename_postfix,
                       use_period_chunks=False, n_periods_per_chunk=N_periods_per_chunk,
                       n_per_temp=N_per_temp_saving_slides,
                       use_shuffling_pieces=False):
        """
        :param slides_first_times_filename_postfix:
        :type slides_first_times_filename_postfix:
        :param slides_first_times_inv_filename_postfix:
        :type slides_first_times_inv_filename_postfix:
        :param slides_pers_filename_postfix:
        :type slides_pers_filename_postfix:
        :param slides_pers_inv_filename_postfix:
        :type slides_pers_inv_filename_postfix:
        :param slides_mes_filename_postfix:
        :type slides_mes_filename_postfix:
        :param slides_mes_inv_filename_postfix:
        :type slides_mes_inv_filename_postfix:
        :param mask_koi_planet: whether to mask the KOI ephemeris if the object is a KOI
        :type mask_koi_planet: bool
        :param per_wing_slides_relative:
        :type per_wing_slides_relative:
        :param n_slides: WILL TAKE THIS NUMBER OF NORMAL AND OF INVERTED
        :type n_slides: int
        :param use_period_chunks: whether to slice periods to chunks to reduce the RAM consummed
        :type use_period_chunks: bool
        :param n_periods_per_chunk: how many period to process at once
        :type n_periods_per_chunk: int
        :param n_per_temp: how many best candidates per template to save
        :type n_per_temp: int
        :param use_shuffling_pieces: whether to cut quarters into pieces and reshuffle them while preserving the total duration of the data
        :type use_shuffling_pieces: bool
        :return:
        :rtype:
        """
        if self.per_wing_injection_and_slides is None:
            self.per_wing_injection_and_slides = per_wing_slides_relative * self.best_per
            # days to the left and the same to the right #limited by memory
        p_max = self.best_per + self.per_wing_injection_and_slides
        p_min = self.best_per - self.per_wing_injection_and_slides
        # we do not need period chunks because we only have one template and only 20 days of periods.

        klc = spip.KeplerLightCurve(target_id=self.target_id, get_data=True, data_location=self.data_location,
                                    output_location=self.output_location)
        if self.target_id + '_' + Conditioned_data_filename_postfix + '.npy' not in os.listdir(self.output_location):
            if self.is_koi and mask_koi_planet:
                transit_durations, transit_times = self.mask_koi_planet(modify_self_klc=False, return_durations_transit_times=True)
                klc.mask_ephemeris(durations=transit_durations, transit_times=transit_times)
            klc.condition_data()  # IN OUTLIERS MASK, THE POSITIVE AND THE NEGATIVE NSIG ARE EQUAL BECAUSE WE NEED TO INVERT FAIRLY.
            klc.condition_templates()
            klc.save_conditioned_data(filename_postfix=Conditioned_data_filename_postfix)
        klc.load_conditioned_data(file=os.path.join(self.output_location,
                                                    self.target_id + '_' + Conditioned_data_filename_postfix + '.npy'))
        klc.whitened_templates_fd = klc.whitened_templates_fd[
                                    klc.indices_of_templates[self.best_temp][0]:klc.indices_of_templates[self.best_temp][-1]]
        klc.whitened_templates_td = klc.whitened_templates_td[
                                    klc.indices_of_templates[self.best_temp][0]:klc.indices_of_templates[self.best_temp][-1]]
        klc.initialize_templates(template_ind=self.best_temp)
        klc.detect_single_transits(apply_score_correction=True)

        # ATTENTION: we introduced this because of unclear memory issues
        klc.delete_irrelevant()

        # HISTOGRAM DATA WILL BE COLLECTED HERE
        mess = []
        mess_inv = []
        pers = []
        pers_inv = []
        first_times = []
        first_times_inv = []

        if not use_period_chunks:
            for i_slides in range(n_slides):
                if not use_shuffling_pieces:
                    klc.time_slides()
                    klc.connect_quarters()
                else:
                    klc.connect_quarters()  # this will every time destroy the effect of scrambling
                    klc.scramble_full_stats(max_hole=spip.Max_hole_pieces_scrampling, max_piece=spip.Max_piece_scrambling,
                                            min_piece=spip.Min_piece_scrambling)
                klc.interleave_shifts()
                klc.sum_periods(pmin=p_min, pmax=p_max)
                # klc.veto_mes()  # we are not doing it at this stage
                klc.save_best_candidates(n_per_temp=n_per_temp, save_to_file=False)
                mess = mess + list(klc.best_periodic_scores_data[spip.Search_results_index_stat][0])
                pers = pers + list(klc.best_periodic_scores_data[spip.Search_results_index_per][0])
                first_times = first_times + [klc.best_periodic_scores_data[spip.Search_results_index_sum_tr_times][0][i][0]
                                             for i in range(len(klc.best_periodic_scores_data[spip.Search_results_index_sum_tr_times][0]))]
                mess_inv = mess_inv + list(klc.best_inverted_periodic_scores_data[spip.Search_results_index_stat][0])
                pers_inv = pers_inv + list(klc.best_inverted_periodic_scores_data[spip.Search_results_index_per][0])
                first_times_inv = first_times_inv + [klc.best_inverted_periodic_scores_data[spip.Search_results_index_sum_tr_times][0][i][0]
                                     for i in range(len(klc.best_inverted_periodic_scores_data[spip.Search_results_index_sum_tr_times][0]))]
                klc.delete_mes()
        else:
            periods = klc.make_periods_array(pmin=p_min, pmax=p_max)
            period_chunks = [periods[i * n_periods_per_chunk: min((i+1) * n_periods_per_chunk, len(periods))]
                             for i in range(int(np.ceil(len(periods) / n_periods_per_chunk))-1)]
            mess = []
            mess_inv = []
            pers = []
            pers_inv = []
            first_times = []
            first_times_inv = []
            for i_slides in range(n_slides):
                if not use_shuffling_pieces:
                    klc.time_slides()
                    klc.connect_quarters()
                else:
                    klc.connect_quarters()  # this will every time destroy the effect of scrambling
                    klc.scramble_full_stats(max_hole=spip.Max_hole_pieces_scrampling, max_piece=spip.Max_piece_scrambling,
                                            min_piece=spip.Min_piece_scrambling)
                klc.interleave_shifts()

                best_candidates_concat = []
                best_inv_candidates_concat = []
                mess_temp = []
                pers_temp = []
                first_times_temp = []
                mess_inv_temp = []
                pers_inv_temp = []
                first_times_inv_temp = []
                for pers_summing in period_chunks:
                    klc.sum_periods(periods=pers_summing)
                    # klc.veto_mes()  # we are not doing it at this stage
                    klc.save_best_candidates(n_per_temp=n_per_temp, save_to_file=False)
                    best_candidates_concat.append(np.array(klc.best_periodic_scores_data, dtype=object))
                    best_inv_candidates_concat.append(np.array(klc.best_inverted_periodic_scores_data, dtype=object))

                    mess_temp = mess_temp + list(klc.best_periodic_scores_data[spip.Search_results_index_stat][0])
                    pers_temp = pers_temp + list(klc.best_periodic_scores_data[spip.Search_results_index_per][0])
                    first_times_temp = first_times_temp + [klc.best_periodic_scores_data[spip.Search_results_index_sum_tr_times][0][i][0]
                                             for i in range(len(klc.best_periodic_scores_data[spip.Search_results_index_sum_tr_times][0]))]
                    mess_inv_temp = mess_inv_temp + list(klc.best_inverted_periodic_scores_data[spip.Search_results_index_stat][0])
                    pers_inv_temp = pers_inv_temp + list(klc.best_inverted_periodic_scores_data[spip.Search_results_index_per][0])
                    first_times_inv_temp = first_times_inv_temp + [
                                    klc.best_inverted_periodic_scores_data[spip.Search_results_index_sum_tr_times][0][i][0]
                                    for i in range(len(klc.best_inverted_periodic_scores_data[spip.Search_results_index_sum_tr_times][0]))]
                    klc.delete_mes()
                best_indices_temp = np.argsort(mess_temp)[:n_per_temp]
                best_indices_inv_temp = (np.argsort(mess_inv_temp)[::-1])[:n_per_temp]

                mess = mess + list(np.array(mess_temp)[best_indices_temp])
                mess_inv = mess_inv + list(np.array(mess_inv_temp)[best_indices_inv_temp])
                pers = pers + list(np.array(pers_temp)[best_indices_temp])
                pers_inv = pers_inv + list(np.array(pers_inv_temp)[best_indices_inv_temp])
                first_times = first_times + list(np.array(first_times_temp)[best_indices_temp])
                first_times_inv = first_times_inv + list(np.array(first_times_inv_temp)[best_indices_inv_temp])

        fls = os.listdir(self.output_location)

        for fl in fls:
            if self.target_id + '_' + slides_mes_inv_filename_postfix + '_' in fl:
                n_slides_0 = int(fl[len(self.target_id + '_' + slides_mes_inv_filename_postfix + '_'): fl.index('.')])
                n_slides += n_slides_0
                hist0 = list(np.load(os.path.join(self.output_location, fl), allow_pickle=True))
                mess_inv = mess_inv + hist0
                os.remove(os.path.join(self.output_location, fl))
        np.save(os.path.join(self.output_location, self.target_id + '_' + slides_mes_inv_filename_postfix + '_' + str(n_slides)),
                mess_inv, allow_pickle=True)

        for fl in fls:
            if self.target_id + '_' + slides_mes_filename_postfix + '_' in fl:
                hist0 = list(np.load(os.path.join(self.output_location, fl), allow_pickle=True))
                mess = mess + hist0
                os.remove(os.path.join(self.output_location, fl))
        np.save(os.path.join(self.output_location, self.target_id + '_' + slides_mes_filename_postfix + '_' + str(n_slides)),
                mess, allow_pickle=True)

        for fl in fls:
            if self.target_id + '_' + slides_pers_inv_filename_postfix + '_' in fl:
                hist0 = list(np.load(os.path.join(self.output_location, fl), allow_pickle=True))
                pers_inv = pers_inv + hist0
                os.remove(os.path.join(self.output_location, fl))
        np.save(os.path.join(self.output_location, self.target_id + '_' + slides_pers_inv_filename_postfix + '_' + str(n_slides)),
                pers_inv, allow_pickle=True)

        for fl in fls:
            if self.target_id + '_' + slides_pers_filename_postfix + '_' in fl:
                hist0 = list(np.load(os.path.join(self.output_location, fl), allow_pickle=True))
                pers = pers + hist0
                os.remove(os.path.join(self.output_location, fl))
        np.save(os.path.join(self.output_location, self.target_id + '_' + slides_pers_filename_postfix + '_' + str(n_slides)),
                pers, allow_pickle=True)

        for fl in fls:
            if self.target_id + '_' + slides_first_times_inv_filename_postfix + '_' in fl:
                hist0 = list(np.load(os.path.join(self.output_location, fl), allow_pickle=True))
                first_times_inv = first_times_inv + hist0
                os.remove(os.path.join(self.output_location, fl))
        np.save(os.path.join(self.output_location, self.target_id + '_' + slides_first_times_inv_filename_postfix + '_' + str(n_slides)),
                first_times_inv, allow_pickle=True)

        for fl in fls:
            if self.target_id + '_' + slides_first_times_filename_postfix + '_' in fl:
                hist0 = list(np.load(os.path.join(self.output_location, fl), allow_pickle=True))
                first_times = first_times + hist0
                os.remove(os.path.join(self.output_location, fl))
        np.save(os.path.join(self.output_location, self.target_id + '_' + slides_first_times_filename_postfix + '_' + str(n_slides)),
                first_times, allow_pickle=True)

        ##############################################################################################################
        # This code was commented out because i want to write the new file as quickly as i can after deleting the previous one.
        # for fl in fls:
        #     if self.target_id + '_' + slides_mes_inv_filename_postfix + '_' in fl:
        #         n_slides_0 = int(fl[len(self.target_id + '_' + slides_mes_inv_filename_postfix + '_'): fl.index('.')])
        #         n_slides += n_slides_0
        #         hist0 = list(np.load(os.path.join(self.output_location, fl), allow_pickle=True))
        #         mess_inv = mess_inv + hist0
        #         os.remove(os.path.join(self.output_location, fl))
        #     elif self.target_id + '_' + slides_mes_filename_postfix + '_' in fl:
        #         hist0 = list(np.load(os.path.join(self.output_location, fl), allow_pickle=True))
        #         mess = mess + hist0
        #         os.remove(os.path.join(self.output_location, fl))
        #     elif self.target_id + '_' + slides_pers_inv_filename_postfix + '_' in fl:
        #         hist0 = list(np.load(os.path.join(self.output_location, fl), allow_pickle=True))
        #         pers_inv = pers_inv + hist0
        #         os.remove(os.path.join(self.output_location, fl))
        #     elif self.target_id + '_' + slides_pers_filename_postfix + '_' in fl:
        #         hist0 = list(np.load(os.path.join(self.output_location, fl), allow_pickle=True))
        #         pers = pers + hist0
        #         os.remove(os.path.join(self.output_location, fl))
        #     elif self.target_id + '_' + slides_first_times_inv_filename_postfix + '_' in fl:
        #         hist0 = list(np.load(os.path.join(self.output_location, fl), allow_pickle=True))
        #         first_times_inv = first_times_inv + hist0
        #         os.remove(os.path.join(self.output_location, fl))
        #     elif self.target_id + '_' + slides_first_times_filename_postfix + '_' in fl:
        #         hist0 = list(np.load(os.path.join(self.output_location, fl), allow_pickle=True))
        #         first_times = first_times + hist0
        #         os.remove(os.path.join(self.output_location, fl))
        #
        # np.save(os.path.join(self.output_location, self.target_id + '_' + slides_mes_filename_postfix + '_' + str(n_slides)),
        #         mess, allow_pickle=True)
        # np.save(os.path.join(self.output_location, self.target_id + '_' + slides_pers_filename_postfix + '_' + str(n_slides)),
        #         pers, allow_pickle=True)
        # np.save(os.path.join(self.output_location, self.target_id + '_' + slides_first_times_filename_postfix + '_' + str(n_slides)),
        #         first_times, allow_pickle=True)
        # np.save(os.path.join(self.output_location, self.target_id + '_' + slides_mes_inv_filename_postfix + '_' + str(n_slides)),
        #         mess_inv, allow_pickle=True)
        # np.save(os.path.join(self.output_location, self.target_id + '_' + slides_pers_inv_filename_postfix + '_' + str(n_slides)),
        #         pers_inv, allow_pickle=True)
        # np.save(os.path.join(self.output_location, self.target_id + '_' + slides_first_times_inv_filename_postfix + '_' + str(n_slides)),
        #         first_times_inv, allow_pickle=True)

    def generate_injection_parameters(
            self, n_injections=N_injections_total, add_to_existing_file=False,
            per_wing_injection_relative=Per_wing_injection_and_slides_relative,
            injection_parameters_filename_postfix=Injection_parameters_filename_postfix,
            all_templates_injection_parameters_filename_postfix=All_templates_injection_parameters_filename_postfix,
            d_ecc=0.2,
            n_w=5,  #
            d_inc=0.2,
            u=(0.47, 0.23),
            snr=7,
            n_per=50
    ):
        """

        :param n_injections: Number of injections to generate in the end. The inner number of generations will be defined by the
        ranges of parameters. `n_injections` is just how many will be returned in the end for the best template
        :type n_injections:
        :param per_wing_injection_relative: days to the left and the same to the right used in the time slides in units of period
        :type per_wing_injection_relative: float
        :param n_per: Number of periods within the period wings to generate. Not evenly, they will be sampled randomly.
        :type n_per: int
        :param d_ecc: Eccentricity step when calculating how many eccentricities to generate. Needed for uniformity of eccentricity
        density generation to make sure that when the range of possible orbital is smaller, we generate fewer of them.
        :type d_ecc: float
        :param n_w: Number of periastron angles to generate in each loop. Since this is uniform and with no bounds,
        it does not really matter how many of them we produce
        :type n_w: int
        :param d_inc: Inclination step when calculating how many inclinations to generate. Needed for uniformity of inclination
        density generation to make sure that when the range of possible transit inclinations is smaller, we generate fewer of them.
        :type d_inc: float
        :param u: Limb darkening coefficients. nothing is too sensitive to them, so they are just fixed
        :type u: [float, float]
        :param snr: Signal to noise of the generated injections. It is just for leakage between the templates, not very important.
        :type snr: float
        :param add_to_existing_file:
        :type add_to_existing_file:
        :param injection_parameters_filename_postfix:
        :type injection_parameters_filename_postfix:
        :param all_templates_injection_parameters_filename_postfix:
        :type all_templates_injection_parameters_filename_postfix:
        :return:
        :rtype:

        (DONE): DO NOT FORGET THE BUG AND LOOK AT THE TRANSIT TIME AND NOT THE BADLY DEFINED PHASE!
        (DONE): THINK ABOUT THE REPETITIONS: WHEN YOU GENERATE ONE P_INJ, YOU USE IT FOR ALL THE OTHER PARAMETERS,
        WHICH IS NOT REALLY FAIR. HOWEVER , IF YOU CHANGE THE THING EVERY TIME, THEN YOU WILL GIVE NOT PROPORTIONAL
        WEIGHTS TO SMALL INCLINATIONS AND BIG INCLINATIONS, LET'S SAY. SO EITHER YOU NEED TO GENERATE MANY OF THEM AND REJECT
        THE NON-TRANSITING ONES, OR SOMETHING ELSE.............OR IT IS OKAY THAT THEY ARE DISCRETE
        :return:
        """
        # TODO: generating this for only one best template is not correct if the template prior is not flat! (what??)
        if self.stellar_mass is None or self.stellar_radius is None or np.isnan(self.stellar_mass) or np.isnan(self.stellar_radius):
            self.failures += 'Unable to generate correct injections because of invalid stellar radius or mass;'
            m_1_3_r_s = 1.
        else:
            m_1_3_r_s = self.stellar_mass ** (1 / 3) / self.stellar_radius
        self.m_1_3_r_s = m_1_3_r_s  # M**(1/3)*R[in M_sun and R_sun]
        if (self.target_id + '_' + injection_parameters_filename_postfix in os.listdir(self.output_location)) and not add_to_existing_file:
            with open(os.path.join(self.output_location, self.target_id + '_' + injection_parameters_filename_postfix), 'w'):
                pass  # will delete the contents of the file

        effective_wf_fd = np.load(os.path.join(os.path.dirname(__file__), 'templates',
                                               'effective_whitening_filter_fourier_domain.npy'), allow_pickle=True)
        temp_bank_whit_norm = np.load(os.path.join(os.path.dirname(__file__), 'templates',
                                                   'temp_bank_whit_norm.npy'), allow_pickle=True)
        # TODO: I am not sure about this whitening anymore!
        template_shifts = af.read_quarters_from_file(spip.Shifts_template_bank_file, dimension=2)
        n_templates = len(template_shifts)  # it is 25, not 45
        actual_template_indices = [j for j in range(len(template_shifts)) for _ in range(len(template_shifts[j]))]
        # actual_template_indices = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 11, 12,
        #                            13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
        assert (len(actual_template_indices) == len(temp_bank_whit_norm)), 'Lengths of the bank and its indices should match!'
        t_length = (len(effective_wf_fd) - 1) * 2 * tm.dt_lc  # len like that because it was with rfft #0.02043368 is long cadence in days
        t = tm.time(T=t_length + tm.dt_lc)

        if self.per_wing_injection_and_slides is None:
            self.per_wing_injection_and_slides = per_wing_injection_relative * self.best_per

        templates_parameter_sets = [[] for _ in range(n_templates)]
        for period in np.random.uniform(self.best_per - self.per_wing_injection_and_slides,
                                        self.best_per + self.per_wing_injection_and_slides, n_per):
            # Attention, this is a linear uniform distribution, although we are using logarithmic occurrences.
            a = (G / (4 * np.pi ** 2) * (period * 24 * 60 * 60) ** 2) ** (1 / 3) * m_1_3_r_s * (
                    M_sun ** (1 / 3) / R_sun)  # 3rd Kepler law #in units of R
            # a = (G * M /(4*np.pi**2) * (per*24*60*60)**2)**(1/3) / R #3rd Kepler law #in units of R
            # r_min = a*(1-e) -> 2*r_star=a*(1-e_max)
            e_max = min(1 - 2 / a, 0.97)  # for some reason, 0.98 gets computationally stuck in barman sometimes, not clear why
            if e_max < 0:
                e_range = [0.]
            else:
                n_eccs = max(1, int(np.round(e_max / d_ecc)))
                e_range = np.random.uniform(0, e_max, n_eccs)
                if len(e_range) == 0:
                    e_range = [0.]
            for ecc in e_range:
                ws = np.random.uniform(0, 360, n_w)
                for w in ws:
                    # i_max = np.arccos(1/a)*180/np.pi
                    i_max = np.arccos(1 / a * (1 + ecc * np.cos(w / 180 * np.pi)) / (1 - ecc ** 2)) * 180 / np.pi
                    n_incls = max(1, int(np.round((90 - i_max) / d_inc)))
                    for inc in np.random.uniform(i_max, 90, n_incls):
                        rp = 0.02  # assume the shape does not depend on the planet radius, although it is not exactly correct
                        flux_obs = 1. - tm.flux_par(t=t, params=None, per=period, rp=rp, a=a, inc=inc, ecc=ecc, w=w, u1=u[0], u2=u[1], t0=0,
                                                    supersample_factor=20, exp_time=tm.dt_lc)
                        norm = np.linalg.norm(flux_obs)
                        if norm > 0:
                            wh_flux_obs = np.fft.rfft(np.fft.ifftshift(flux_obs)) * effective_wf_fd
                            wh_flux_obs = wh_flux_obs / np.linalg.norm(wh_flux_obs)
                            if snr is not None:
                                wh_flux_obs = snr * wh_flux_obs + np.fft.rfft(np.random.normal(0, 1, (len(effective_wf_fd) - 1) * 2)) / (
                                        (len(effective_wf_fd) - 1) * 2) ** 0.5
                                wh_flux_obs = wh_flux_obs / np.linalg.norm(wh_flux_obs)
                            cos_angles = [wh_flux_obs @ temp_bank_whit_norm[j] for j in range(len(temp_bank_whit_norm))]
                            # TODO This is in Fourier rfft, how does this work on Earth?
                            args = np.argsort(cos_angles)
                            ind_1 = actual_template_indices[args[-1]]
                            ind_2 = actual_template_indices[args[-2]]
                            templates_parameter_sets[ind_1].append([period, a, inc, ecc, w, u[0], u[1]])
                            if ind_2 != ind_1:
                                templates_parameter_sets[ind_2].append([period, a, inc, ecc, w, u[0], u[1]])
        # Saving the injection parameters for all the templates, although we will only need for one template.
        # Note that the number of injections is different for different templates, and this reflects the template prior.
        # The number of injections in the run will be then renormalized to match the required number.
        np.save(os.path.join(self.output_location, self.target_id + '_' + all_templates_injection_parameters_filename_postfix),
                np.array(templates_parameter_sets, dtype=object), allow_pickle=True)
        self.template_prior = np.array([len(templates_parameter_sets[k]) for k in range(len(templates_parameter_sets))]
                                       ) / np.sum([len(templates_parameter_sets[i]) for i in range(len(templates_parameter_sets))])
        template_factor = len(templates_parameter_sets[self.best_temp]) / np.sum([len(templates_parameter_sets[i])
                                                                                  for i in range(len(templates_parameter_sets))])
        # template_factor = self.template_prior[self.best_temp]

        # All the things with the occurrence rates and densities:
        log10_rp_edges, period_edges, occurrences_table = get_occurrence_table(self.stellar_radius)  # Per one star
        log10_best_per_low = np.log10(self.best_per - self.per_wing_injection_and_slides)
        log10_best_per_up = np.log10(self.best_per + self.per_wing_injection_and_slides)
        log10_period_edges = np.log10(period_edges)
        self.log10_best_period_edges = (log10_best_per_low, log10_best_per_up)

        rp_occurrences = occurrences_table[-1]
        # We do not know occurrence priors for periods > 400 days, so just use the same ones as for 400 days
        # Number of planets of all radii in log day (per 100 stars):
        planets_per_log10_day = np.sum(rp_occurrences) / (log10_period_edges[-1] - log10_period_edges[-2])
        # For other periods, it will be checked right now:
        for i in range(len(period_edges) - 1):
            if (self.best_per > period_edges[i]) and (self.best_per <= period_edges[i + 1]):
                rp_occurrences = occurrences_table[i]
                planets_per_log10_day = np.sum(rp_occurrences) / (log10_period_edges[i + 1] - log10_period_edges[i])
                break
        # TODO: We do not interpolate:
        # do not account for the case when the value is near the edge of the bin, amd the density is estimated from the two Bins

        # Number of planets of all radii expected in the considered period range (per 1 star)
        planets_in_per_range = planets_per_log10_day * (log10_best_per_up - log10_best_per_low)
        # Number of planets of this template of all radii of slides periods, per one star:
        self.planets_in_best_period_range_best_template_per_star = planets_in_per_range * template_factor
        # TODO: MAYBE IT IS BETTER TO MEASURE THE RADIUS OF THE TRIGGER AND TAKE THE OCCURRENCE RATE FOR THIS RADIUS?!

        n_mult = int(np.ceil(n_injections / np.sum(rp_occurrences)))  # factor to multiply the occurrences to get the generation numbers
        rp_numbers = (np.ceil(np.array(rp_occurrences) * n_mult)).astype(int)  # numbers of rp to generate
        rps = []
        for i in range(len(log10_rp_edges) - 1):
            rps = rps + list(np.exp(np.random.uniform(log10_rp_edges[i], log10_rp_edges[i + 1], rp_numbers[i])))
        np.random.shuffle(rps)

        best_temp_par_list_0 = templates_parameter_sets[self.best_temp]
        # select indices of the parameters to match the number (maybe with repetitions, but it is not important because there
        # will be random radii and phases)
        selection_indices = np.random.choice(np.arange(len(best_temp_par_list_0)), n_injections, replace=True)
        self.best_temp_injection_par_list = list(np.array(best_temp_par_list_0, dtype=object)[selection_indices])
        self.best_temp_injection_par_list = [list(self.best_temp_injection_par_list[i]) + [rps[i]] for i in range(
            len(self.best_temp_injection_par_list))]

        self.save_injection_parameters()
        self.pickle_small_attributes()  # self.best_temp_injection_par_list will not be pickled

    def save_injection_parameters(self, injection_parameters_filename_postfix=Injection_parameters_filename_postfix):
        """
        :param injection_parameters_filename_postfix:
        :type injection_parameters_filename_postfix:
        :return:
        :rtype:
        """
        # We wanted to just save parameters inside the generate_injection_parameters function because anyway we do not create attribute
        # of injection parameters sets because we will probably not use it simultaneously. But we decided that maybe we will.
        with open(os.path.join(self.output_location, self.target_id + '_' + injection_parameters_filename_postfix), 'w') as fil1:
            print('# period[days], a/R, inc[deg 90 or less], ecc, w[deg], u_0, u_1, rp/R', file=fil1)
            for i in range(len(self.best_temp_injection_par_list)):
                print(*self.best_temp_injection_par_list[i], file=fil1)

    def delete_injection_parameters_file(self, injection_parameters_filename_postfix=Injection_parameters_filename_postfix):
        if self.target_id + '_' + injection_parameters_filename_postfix in os.listdir(self.output_location):
            os.remove(os.path.join(self.output_location, self.target_id + '_' + injection_parameters_filename_postfix))

    def load_injection_parameters(self, injection_parameters_index=0,
                                  injection_parameters_filename_postfix=Injection_parameters_filename_postfix):
        with open(os.path.join(self.output_location, self.target_id + '_' + injection_parameters_filename_postfix), 'r') as fil:
            lines = fil.readlines()
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
            # ' a/R, inc[deg from 90 down], ecc, w[deg], u_0, u_1, rp/R, phase'
        return per_inj, a_inj, inc_inj, ecc_inj, w_inj, u1_inj, u2_inj, rp_inj

    def run_injection_search(self, injection_load_parameters_index=0, per_wing_detection=Per_wing_injection_detection,
                             injections_filename_postfix=Injections_filename_postfix, n_per_temp=N_per_temp_saving_slides):
        """
        TODO: add the thing where it reads the fields of the class to make it autonomous
        :param injections_filename_postfix:
        :type injections_filename_postfix:
        :param injection_load_parameters_index:
        :type injection_load_parameters_index:
        :param per_wing_detection: days to the left and the same to the right in search
        :type per_wing_detection: float
        :return:
        :rtype:
        """
        # sim_per_job = 20  # with this, 10000 injections in 500 jobs should finish during roughly an hour

        templates_durations_days = get_templates_durations_from_file() / 24

        per_inj, a_inj, inc_inj, ecc_inj, w_inj, u1_inj, u2_inj, rp_inj = self.load_injection_parameters(
            injection_parameters_index=injection_load_parameters_index)
        # This period range is for injection, not for search:

        phase_inj = np.random.uniform(0, per_inj)

        tms, flxs = af.get_data_fun(target_id=self.target_id,
                                    location=self.data_location)
        first_transit_time_inj = np.nanmin(tms[0]) + phase_inj
        flxs_inj = sim.generate_injection(times=tms, fluxes=flxs, get_data=False, target_id=self.target_id,
                                          data_location='', nperseg=128,
                                          period=per_inj, template_num=None, transit_duration=None,
                                          first_transit_center=first_transit_time_inj,
                                          use_physical_parameters=True, rp=rp_inj, a=a_inj, inc=inc_inj, ecc=ecc_inj, w=w_inj,
                                          u1=u1_inj, u2=u2_inj,
                                          supersampling=True, integration_points=20, normalize_by_level=True, detlen=500,
                                          snr=None, depth_ppm=None, depth_by_rp=True,
                                          return_template=False, return_wfs=False, return_zero_padding_ends=False)

        p_max = per_inj + per_wing_detection
        p_min = per_inj - per_wing_detection

        klc = spip.KeplerLightCurve(target_id=self.target_id, times=tms, fluxes=flxs_inj, get_data=False,
                                    data_location=self.data_location)

        klc.condition_data()  # IN OUTLIERS MASK, THE POSITIVE AND THE NEGATIVE NSIG ARE EQUAL BECAUSE WE NEED TO INVERT FAIRLY.
        klc.condition_templates(template_ind=self.best_temp)
        # Mask the suspected real transit. Warning, can kill some of injected signal
        klc.mask_ephemeris(period=self.best_per, durations=templates_durations_days[self.best_temp],
                           transit_times=self.best_first_transit_time)

        klc.detect_single_transits(apply_score_correction=True)
        klc.connect_quarters()
        klc.interleave_shifts()
        klc.sum_periods(pmin=p_min, pmax=p_max)
        # klc.veto_mes()  # we are not doing it at this stage
        klc.save_best_candidates(n_per_temp=n_per_temp, save_to_file=False)
        # TODO: replace indices for them not to be hard-coded!
        retrieved_mes = klc.best_periodic_scores_data[spip.Search_results_index_stat][0][0]
        # retrieved_phase = klc.best_periodic_scores_data[3][0][0]  # wrt the beginning of Kepler time
        retrieved_first_transit_time = klc.best_periodic_scores_data[spip.Search_results_index_sum_tr_times][0][0][0]

        with open(os.path.join(self.output_location, self.target_id + '_' + injections_filename_postfix), 'a') as fil1:
            print(per_inj, a_inj, inc_inj, ecc_inj, w_inj, u1_inj, u2_inj, rp_inj, first_transit_time_inj, retrieved_mes,
                  retrieved_first_transit_time, sep=' ', end='\n', file=fil1)

    def retrieve_injections(self, max_phase_mismatch_rel=Max_phase_mismatch_rel):
        if self.best_stat is None:
            self.unpickle_small_attributes()

        for status in ['success', 'bad']:
            self.__dict__['per_inj_' + status] = []
            self.__dict__['a_inj_' + status] = []
            self.__dict__['inc_inj_' + status] = []
            self.__dict__['ecc_inj_' + status] = []
            self.__dict__['w_inj_' + status] = []
            self.__dict__['rp_inj_' + status] = []
            self.__dict__['first_transit_time_inj_' + status] = []
            self.__dict__['retrieved_mes_' + status] = []
            self.__dict__['retrieved_first_transit_time_' + status] = []

        with open(os.path.join(self.output_location, self.target_id + '_' + Injections_filename_postfix), 'r') as fil1:
            inj_lines = fil1.readlines()

        for inj_line0 in inj_lines:
            inj_line = inj_line0.strip()
            inj_line = inj_line.strip('\x00')
            if (len(inj_line) > 5) and (inj_line[0] != '#'):
                per_inj, a_inj, inc_inj, ecc_inj, w_inj, u1_inj, u2_inj, rp_inj, first_transit_time_inj, retrieved_mes, \
                   retrieved_first_transit_time = inj_line.split()
                per_inj = float(per_inj)
                a_inj = float(a_inj)
                inc_inj = float(inc_inj)
                ecc_inj = float(ecc_inj)
                w_inj = float(w_inj)
                # u1_inj = float(u1_inj)
                # u2_inj = float(u2_inj)
                rp_inj = float(rp_inj)
                first_transit_time_inj = float(first_transit_time_inj)
                retrieved_mes = float(retrieved_mes)
                retrieved_first_transit_time = float(retrieved_first_transit_time)
                if np.abs(first_transit_time_inj % per_inj
                          - float(retrieved_first_transit_time) % per_inj) > max_phase_mismatch_rel * per_inj:
                    status = 'bad'
                else:
                    status = 'success'

                self.__dict__['per_inj_' + status].append(per_inj)
                self.__dict__['a_inj_' + status].append(a_inj)
                self.__dict__['inc_inj_' + status].append(inc_inj)
                self.__dict__['ecc_inj_' + status].append(ecc_inj)
                self.__dict__['w_inj_' + status].append(w_inj)
                self.__dict__['rp_inj_' + status].append(rp_inj)
                self.__dict__['first_transit_time_inj_' + status].append(first_transit_time_inj)
                self.__dict__['retrieved_mes_' + status].append(retrieved_mes)
                self.__dict__['retrieved_first_transit_time_' + status].append(retrieved_first_transit_time)

                self.number_of_success_injections = len(self.per_inj_success)
                self.number_of_injections = len(self.per_inj_bad) + len(self.per_inj_success)

    def _get_bin_min_bin_max(self):
        if self.best_stat is not None:
            trig_val = self.best_stat ** 2
        else:
            trig_val = np.nan

        if (self.retrieved_mes_bad is not None) and (self.retrieved_mes_success is not None):
            conc_inj_stat2 = np.concatenate((np.abs(self.retrieved_mes_success) ** 2, np.abs(self.retrieved_mes_bad) ** 2))
            inj_min = np.nanmin(conc_inj_stat2)
            inj_max = np.nanmax(conc_inj_stat2)
        else:
            inj_min = np.nan
            inj_max = np.nan

        if self.slides_mes2_h0 is not None:
            slides_min = np.nanmin(self.slides_mes2_h0)
            slides_max = np.nanmax(self.slides_mes2_h0)
        else:
            slides_min = np.nan
            slides_max = np.nan

        if self.slides_inv_mes2_h0 is not None:
            slides_inv_min = np.nanmin(self.slides_inv_mes2_h0)
            slides_inv_max = np.nanmax(self.slides_inv_mes2_h0)
        else:
            slides_inv_min = np.nan
            slides_inv_max = np.nan

        bin_max = np.nanmax([inj_max, slides_max, slides_inv_max, trig_val])
        bin_min = np.nanmin([inj_min, slides_min, slides_inv_min, trig_val])

        # if (self.retrieved_mes_bad is not None) and (self.retrieved_mes_success is not None):
        #     bin_min = min([np.nanmin(np.concatenate((np.abs(self.retrieved_mes_success) ** 2, np.abs(self.retrieved_mes_bad) ** 2))),
        #                    np.nanmin(self.slides_mes2_h0), np.nanmin(self.slides_inv_mes2_h0), self.best_stat ** 2])
        # else:
        #     bin_min = min([np.nanmin(self.slides_mes2_h0), np.nanmin(self.slides_inv_mes2_h0)])
        #
        # if (self.retrieved_mes_bad is not None) and (self.retrieved_mes_success is not None):
        #     bin_max = max([np.nanmax(np.concatenate((np.abs(self.retrieved_mes_success) ** 2, np.abs(self.retrieved_mes_bad) ** 2))),
        #                    np.nanmax(self.slides_mes2_h0), np.nanmax(self.slides_inv_mes2_h0), self.best_stat ** 2])
        # else:
        #     bin_max = max([np.nanmax(self.slides_mes2_h0), np.nanmax(self.slides_inv_mes2_h0), self.best_stat ** 2])
        return bin_min, bin_max

    def retrieve_background(self):
        fil_slides_h0_mes = [ff for ff in os.listdir(self.output_location) if
                             ((str(self.target_id) in ff) and (Slides_mes_filename_postfix in ff) and (
                                     Slides_mes_inv_filename_postfix not in ff))][0]
        fil_slides_h0_inv_mes = [ff for ff in os.listdir(self.output_location) if
                                 ((str(self.target_id) in ff) and (Slides_mes_inv_filename_postfix in ff))][0]

        self.slides_mes2_h0 = np.load(os.path.join(self.output_location, fil_slides_h0_mes), allow_pickle=True) ** 2
        self.slides_inv_mes2_h0 = np.load(os.path.join(self.output_location, fil_slides_h0_inv_mes), allow_pickle=True) ** 2

        self.n_slides = int(
            fil_slides_h0_mes[len(self.target_id + '_' + Slides_mes_filename_postfix + '_'): fil_slides_h0_mes.index('.')])

    def make_histogram_bins(self):
        if self.slides_mes2_h0 is None:
            self.retrieve_background()
        if self.retrieved_mes_success is None:
            self.retrieve_injections()
        bin_min, bin_max = self._get_bin_min_bin_max()
        self.bins_mes2_histogram = np.arange(bin_min, bin_max, Histogram_mes2_binwid)

    def make_background_histogram(self, bins=None, bin_min=None, bin_max=None):
        if self.slides_mes2_h0 is None:
            self.retrieve_background()

        if bins is not None:
            self.bins_mes2_histogram_bg = bins
        else:
            if self.bins_mes2_histogram is not None:
                self.bins_mes2_histogram_bg = self.bins_mes2_histogram
            else:
                bin_mi, bin_ma = self._get_bin_min_bin_max()
                if bin_min is None:
                    bin_min = bin_mi
                if bin_max is None:
                    bin_max = bin_ma
                self.bins_mes2_histogram_bg = np.arange(bin_min, bin_max, Histogram_mes2_binwid)

        self.distr_mes2_h0_direct = np.histogram(self.slides_mes2_h0, bins=self.bins_mes2_histogram_bg)[0]
        self.distr_mes2_h0_direct = self.distr_mes2_h0_direct / self.n_slides
        self.distr_mes2_h0_inv = np.histogram(self.slides_inv_mes2_h0, bins=self.bins_mes2_histogram_bg)[0]
        self.distr_mes2_h0_inv = self.distr_mes2_h0_inv / self.n_slides
        self.distr_mes2_h0 = (self.distr_mes2_h0_direct + self.distr_mes2_h0_inv) / 2

    def make_injections_histogram(self, bins=None, bin_min=None, bin_max=None):
        if self.retrieved_mes_success is None:
            self.retrieve_injections()

        if bins is not None:
            self.bins_mes2_histogram_inj = bins
        else:
            if self.bins_mes2_histogram is not None:
                self.bins_mes2_histogram_inj = self.bins_mes2_histogram
            else:
                bin_mi, bin_ma = self._get_bin_min_bin_max()
                if bin_min is None:
                    bin_min = bin_mi
                if bin_max is None:
                    bin_max = bin_ma
                self.bins_mes2_histogram_inj = np.arange(bin_min, bin_max, Histogram_mes2_binwid)

        self.distr_mes2_inj_all = np.histogram(np.array(self.retrieved_mes_bad + self.retrieved_mes_success) ** 2,
                                               bins=self.bins_mes2_histogram_inj)[0]
        self.distr_mes2_inj_all = (self.distr_mes2_inj_all / self.number_of_injections
                                   * self.planets_in_best_period_range_best_template_per_star)
        self.distr_mes2_inj_success = np.histogram(np.array(self.retrieved_mes_success) ** 2, bins=self.bins_mes2_histogram_inj)[0]
        self.distr_mes2_inj_success = (self.distr_mes2_inj_success / self.number_of_injections
                                       * self.planets_in_best_period_range_best_template_per_star)

    def calculate_significance(self, remake_histograms=True):
        if remake_histograms or self.bins_mes2_histogram is None:
            self.make_histogram_bins()
            self.make_injections_histogram(bins=self.bins_mes2_histogram)
            self.make_background_histogram(bins=self.bins_mes2_histogram)
        # MAKE EXTRAPOLATION
        # Determine intersection with horizontal axis in log scale:
        self.extrapolation_coefs_mes2_h0_hist = af.get_histogram_extrapolation_coefs_in_log(
            self.distr_mes2_h0_direct, self.bins_mes2_histogram)  # attention: extrapolating only the direct bg
        # self.extrapolation_coefs_mes2_h0_hist[1] is supposed to be negative because the trend is decreasing
        self.prob_density_h0_trigger = np.exp((self.best_stat ** 2) * self.extrapolation_coefs_mes2_h0_hist[1]
                                              + self.extrapolation_coefs_mes2_h0_hist[0])
        # Follows from the integration of the non-log distribution over MES[SNR^2]
        self.false_alarm_prob_trigger = (1 / np.abs(self.extrapolation_coefs_mes2_h0_hist[1])
                                         * np.exp(self.extrapolation_coefs_mes2_h0_hist[0] + self.extrapolation_coefs_mes2_h0_hist[
                                            1] * self.best_stat ** 2))
        bin_inds_close_trigger = np.sort(np.argsort(np.abs(self.bins_mes2_histogram - self.best_stat ** 2))[:2])  # find edges of this bin
        self.prob_density_h1_trigger = self.distr_mes2_inj_all[bin_inds_close_trigger[0]]  # use the left edge index to access the bin
        self.prob_density_planet_not_noise = self.prob_density_h1_trigger / (self.prob_density_h1_trigger + self.prob_density_h0_trigger)

    def analyze_injections_and_background(self, max_phase_mismatch_rel=Max_phase_mismatch_rel):
        """
        :param max_phase_mismatch_rel:
        :type max_phase_mismatch_rel:
        :return:
        :rtype:
        """
        self.retrieve_background()
        self.retrieve_injections(max_phase_mismatch_rel=max_phase_mismatch_rel)
        self.make_histogram_bins()
        self.make_injections_histogram(bins=self.bins_mes2_histogram)
        self.make_background_histogram(bins=self.bins_mes2_histogram)
        self.calculate_significance(remake_histograms=False)

    def get_ses_no_filters(self, nsig_outliers_0_pos=5, nsig_outliers_0_neg=15):
        klc = spip.KeplerLightCurve(target_id=self.target_id, get_data=True, data_location=self.data_location)
        klc.condition_data(nsig_outliers_0=nsig_outliers_0_neg, nsig_outliers_0_pos=nsig_outliers_0_pos,
                           nsig_outliers_0_neg=nsig_outliers_0_neg,
                           nsig_outliers_w=nsig_outliers_0_neg, nsig_outliers_w_pos=nsig_outliers_0_pos,
                           nsig_outliers_w_neg=nsig_outliers_0_neg)
        klc.condition_templates()
        klc.detect_single_transits(apply_score_correction=False, apply_veto=False)
        return klc

    def get_summed_values_no_filters_freedom(self, nsig_outliers_0_pos=5, nsig_outliers_0_neg=15, ttv_hours_allowed=5):
        """
        similar to `get_summed_values_no_filters`, but allow for ttv and template variation
        :param nsig_outliers_0_pos:
        :type nsig_outliers_0_pos:
        :param nsig_outliers_0_neg:
        :type nsig_outliers_0_neg:
        :param ttv_hours_allowed:
        :type ttv_hours_allowed:
        :return:
        :rtype:
        """
        klc = self.get_ses_no_filters(nsig_outliers_0_pos=nsig_outliers_0_pos, nsig_outliers_0_neg=nsig_outliers_0_neg)
        ttv_half_bins_allowed = int(np.round(ttv_hours_allowed / klc.dt_days))
        best_temp_sh_inds = []
        best_quarters = []
        best_time_inds = []
        ttvs_of_bests = []
        for i, tt in enumerate(self.summed_transit_times):
            best_stat = 0.
            best_templ_sh_ind = None
            best_quarter = None
            best_time_ind = None
            ttv_to_max = None
            for q in klc.quarters:
                if klc.times[q][~np.isnan(klc.times[q])][0] <= tt <= klc.times[q][~np.isnan(klc.times[q])][-1]:
                    ind_tr = np.nanargmin(np.abs(klc.times[q] - tt))
                    for k in range(len(klc.statistics_snr)):
                        ti0 = max(0, ind_tr - ttv_half_bins_allowed)
                        rel_shift = np.nanargmin(klc.statistics_snr[k][q][ti0:ind_tr + ttv_half_bins_allowed])
                        time_index = ti0 + rel_shift
                        # -> allow the actual minimum to be a little shifted
                        stat_here = klc.statistics_snr[k][q][time_index]
                        if np.abs(stat_here) > np.abs(best_stat):
                            best_stat = stat_here
                            best_templ_sh_ind = k
                            best_quarter = q
                            ttv_to_max = (rel_shift - ttv_half_bins_allowed) * klc.dt_days
                            best_time_ind = time_index
                            # best_time = klc.times[q][time_index]
                            #
            if best_time_ind is not None:  # it can be None if the transit is between the quarters or something like that
                best_temp_sh_inds.append(best_templ_sh_ind)
                best_quarters.append(best_quarter)
                best_time_inds.append(best_time_ind)
                ttvs_of_bests.append(ttv_to_max)
            else:
                best_temp_sh_inds.append(np.nan)
                best_quarters.append(np.nan)
                best_time_inds.append(np.nan)
                ttvs_of_bests.append(np.nan)

        best_raw_stats = []
        best_stat_denoms = []
        best_trends = []
        best_templs = []
        optim_templ_depths = []
        for i in range(len(best_temp_sh_inds)):
            if not np.isnan(best_temp_sh_inds[i]):
                best_raw_stats.append(klc.statistics_raw_cor[best_temp_sh_inds[i]][best_quarters[i]][best_time_inds[i]])
                best_stat_denoms.append(klc.stat_denoms[best_temp_sh_inds[i]][best_quarters[i]])
                best_trends.append(klc.trends_orig_fluxes[best_quarters[i]][best_time_inds[i]])
                best_templs.append(klc.templates_of_indices[best_temp_sh_inds[i]])
                optim_templ_depths.append(np.max(np.abs(klc.templates[best_temp_sh_inds[i]])))
            else:
                best_raw_stats.append(np.nan)
                best_stat_denoms.append(np.nan)
                best_trends.append(np.nan)
                best_templs.append(np.nan)
                optim_templ_depths.append(np.nan)
        self.optim_raw_stats = np.array(best_raw_stats)
        self.optim_stat_denoms = np.array(best_stat_denoms)
        self.optim_trends = np.array(best_trends)
        self.optim_templs = np.array(best_templs)
        self.ttvs_of_optim = np.array(ttvs_of_bests)
        optim_templ_depths = np.array(optim_templ_depths)

        # templates_of_indices = [k for k in range(len(klc.indices_of_templates))
        #                         for _ in range(klc.indices_of_templates[k][0], klc.indices_of_templates[k][1])]
        # self.optim_templs = [klc.templates_of_indices[templ_sh_ind] if not np.isnan(templ_sh_ind) else np.nan
        #                      for templ_sh_ind in best_temp_sh_inds]

        self.optim_ampl, self.optim_ppm, self.optim_ampls, ampls_stds, normalized_ampls, \
            normalized_ampls_stds, self.optim_ppms, ppms_stds = \
            af.estimate_ampls_ppms_from_stats(summed_raw_stats=self.optim_raw_stats, summed_trends=self.optim_trends,
                                              summed_stat_denoms=self.optim_stat_denoms, templ_max=optim_templ_depths, trends_stds=None)

    def get_summed_values_no_filters(self, nsig_outliers_0_pos=5, nsig_outliers_0_neg=15):
        klc = self.get_ses_no_filters(nsig_outliers_0_pos=nsig_outliers_0_pos, nsig_outliers_0_neg=nsig_outliers_0_neg)
        best_temps_sh = np.arange(klc.indices_of_templates[self.best_temp][0], klc.indices_of_templates[self.best_temp][-1])
        best_raw_stats = []
        best_stat_denoms = []
        best_trends = []
        optim_templ_depths = []
        for i, tt in enumerate(self.summed_transit_times):
            best_stat = 0.
            best_time_ind = None
            best_temp_ind = None
            best_quarter = None
            for q in klc.quarters:
                if klc.times[q][~np.isnan(klc.times[q])][0] <= tt <= klc.times[q][~np.isnan(klc.times[q])][-1]:
                    ind_tr = np.nanargmin(np.abs(klc.times[q] - tt))
                    for k in best_temps_sh:
                        ti0 = max(0, ind_tr - 1)
                        ti1 = min(ind_tr + 1, len(klc.statistics_raw_cor[k][q]))
                        rel_shift = np.nanargmin(klc.statistics_raw_cor[k][q][ti0:ti1])
                        time_index = ti0 + rel_shift
                        stat_here = klc.statistics_raw_cor[k][q][time_index]
                        if stat_here < best_stat:
                            best_stat = stat_here
                            best_time_ind = time_index
                            best_temp_ind = k
                            best_quarter = q
            if best_time_ind is not None:  # it can be None if the transit is between the quarters or something like that
                best_raw_stats.append(klc.statistics_raw_cor[best_temp_ind][best_quarter][best_time_ind])
                best_stat_denoms.append(klc.stat_denoms[best_temp_ind][best_quarter])
                best_trends.append(klc.trends_orig_fluxes[best_quarter][best_time_ind])
                optim_templ_depths.append(np.max(np.abs(klc.templates[best_temp_ind])))
            else:
                best_raw_stats.append(np.nan)
                best_stat_denoms.append(np.nan)
                best_trends.append(np.nan)
                optim_templ_depths.append(np.nan)

        self.best_raw_stats = np.array(best_raw_stats)
        self.best_stat_denoms = np.array(best_stat_denoms)
        self.best_trends = np.array(best_trends)
        temp_depth = np.max(np.abs(klc.templates[best_temps_sh[0]]))
        self.best_temp_depth = temp_depth

        ampl, ppm, ampls, ampls_stds, normalized_ampls, normalized_ampls_stds, ppms, ppms_stds = \
            af.estimate_ampls_ppms_from_stats(summed_raw_stats=self.best_raw_stats,
                                              summed_trends=self.best_trends, summed_stat_denoms=self.best_stat_denoms,
                                              templ_max=temp_depth, trends_stds=None)

        self.best_ampls = ampls
        self.best_ampls_stds = ampls_stds
        self.best_ampl = ampl
        self.best_norm_ampls = normalized_ampls
        self.best_norm_ampls_stds = normalized_ampls_stds
        self.best_ppms = ppms
        self.best_ppm = ppm
        self.best_ppms_stds = ppms_stds

    def estimate_rp(self, nsig_outliers_0_pos=5, nsig_outliers_0_neg=15):
        if self.best_ppms is None or self.best_ppm is None or self.best_ppms_stds is None:
            self.get_summed_values_no_filters(nsig_outliers_0_pos=nsig_outliers_0_pos,
                                              nsig_outliers_0_neg=nsig_outliers_0_neg)
        # ampl, ppm, ampls, ampls_stds, normalized_ampls, normalized_ampls_stds, ppms, ppms_stds = \
        #     estimate_ampls_ppms_from_stats(summed_raw_stats, summed_trends, summed_stat_denoms, templ_max, trends_stds=None)
        #
        self.rp_to_rs = (self.best_ppm / 1e6) ** 0.5
        self.rp_to_rs_std = (np.nanvar(np.abs(self.best_ppms) ** 0.5)
                             + np.nanmean(self.best_ppms_stds ** 2) / np.abs(self.best_ppm)) ** 0.5 / 1e3
        self.rp_to_r_earth = self.rp_to_rs * self.stellar_radius * R_sun_R_e
        self.rp_to_r_earth_std = self.rp_to_rs_std * self.stellar_radius * R_sun_R_e

    def copy_output_to_marvin(self, marvin_path=Home_output_storage_folder):
        os.system('scp -r ' + self.output_location + ' ' + marvin_path + '/' + self.target_id[:4])

    def plot_fluxes(self, ax=None, show=True, figsize=(15, 5), clrs=((0.5, 0.5, 1), (0.2, 0.7, 0.5))):
        self.get_light_curve_for_plotting_ses()
        temp_n = self.best_temp  # because we initialized templates with only one template
        temp_n_sh = self.klc.indices_of_templates[temp_n][0]
        # temp_n_inv = self.best_temp_inv
        # temp_n_sh_inv = self.klc.indices_of_templates[temp_n_inv][0]

        tms = []
        flxs = []
        for tim in self.summed_transit_times:
            for q in self.klc.quarters:
                if (tim >= self.klc.times[q][0]) and (tim <= self.klc.times[q][-1]):
                    ind = np.argmin(np.abs(self.klc.times[q] - tim))
                    tms.append(self.klc.times[q][ind])
                    flxs.append(self.klc.fluxes[q][ind])
        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.gca()
        for q in self.klc.quarters:
            ax.plot(self.klc.times[q], self.klc.fluxes[q], color=clrs[q % 2], linewidth=1)
        ax.plot(tms, flxs, 'o', color='k', label='summed raw', markersize=4)
        for q in self.klc.quarters:
            vm = self.klc.veto_masks[temp_n_sh][q].astype(bool)
            ax.plot(self.klc.times[q][~vm], self.klc.fluxes[q][~vm], 'o', color='r', markersize=2)
        q = self.klc.quarters[-1]
        vm = self.klc.veto_masks[temp_n_sh][q].astype(bool)
        ax.plot(self.klc.times[q][~vm], self.klc.fluxes[q][~vm], 'o', color='r', markersize=2, label='vetted statistics')  # just for label
        ax.set_xlabel('time, days')
        ax.set_ylabel('fluxes, mean subtracted')
        ax.set_title('KOI' + self.target_id + ' conditioned fluxes\n' + 'period: ' + str(np.round(self.best_per, 1))
                     + ', duration: ' + str(self.templates_durations_hours[self.best_temp]) + ' hours')
        ax.legend(loc='upper right')
        if show:
            plt.show()

    def plot_whitened_fluxes(self, ax=None, show=True, figsize=(15, 5), clrs=((0.5, 0.5, 1), (0.2, 0.7, 0.5))):
        self.get_light_curve_for_plotting_ses()
        temp_n = self.best_temp  # because we initialized templates with only one template
        temp_n_sh = self.klc.indices_of_templates[temp_n][0]
        # temp_n_inv = self.best_temp_inv
        # temp_n_sh_inv = self.klc.indices_of_templates[temp_n_inv][0]

        tms = []
        flxs = []
        for tim in self.summed_transit_times:
            for q in self.klc.quarters:
                if (tim >= self.klc.times[q][0]) and (tim <= self.klc.times[q][-1]):
                    ind = np.argmin(np.abs(self.klc.times[q] - tim))
                    tms.append(self.klc.times[q][ind])
                    flxs.append(self.klc.whitened_fluxes_td[q][ind])
        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.gca()
        for q in self.klc.quarters:
            ax.plot(self.klc.times[q], self.klc.whitened_fluxes_td[q], color=clrs[q % 2], linewidth=1)
        ax.plot(tms, flxs, 'o', color='k', label='summed raw', markersize=4)
        for q in self.klc.quarters:
            vm = self.klc.veto_masks[temp_n_sh][q].astype(bool)
            ax.plot(self.klc.times[q][~vm], self.klc.whitened_fluxes_td[q][~vm], 'o', color='r', markersize=2)
        q = self.klc.quarters[-1]
        vm = self.klc.veto_masks[temp_n_sh][q].astype(bool)
        ax.plot(self.klc.times[q][~vm], self.klc.whitened_fluxes_td[q][~vm], 'o', color='r', markersize=2,
                label='vetted statistics')  # just for label
        ax.set_xlabel('time, days')
        ax.set_ylabel('whitened fluxes')
        ax.set_title('KOI' + self.target_id + ' whitened fluxes\n' + 'period: ' + str(np.round(self.best_per, 1))
                     + ', duration: ' + str(self.templates_durations_hours[self.best_temp]) + ' hours')
        ax.legend(loc='upper right')
        if show:
            plt.show()

    def get_light_curve_for_plotting_ses(self):
        if self.templates_durations_hours is None:
            self.templates_durations_hours = get_templates_durations_from_file()
        if self.best_stat is None:
            self.unpickle_small_attributes()
        if self.klc is None:
            self.klc = spip.KeplerLightCurve(target_id=self.target_id, get_data=True, data_location=self.data_location)
            if self.target_id + '_' + Conditioned_data_filename_postfix + '.npy' not in os.listdir(self.output_location):
                self.klc.condition_data()  # IN OUTLIERS MASK, THE POSITIVE AND THE NEGATIVE NSIG ARE EQUAL BEC. WE NEED TO INVERT FAIRLY
                self.klc.condition_templates()
            else:
                self.klc.load_conditioned_data(file=os.path.join(self.output_location,
                                                                 self.target_id + '_' + Conditioned_data_filename_postfix + '.npy'))
            self.klc.initialize_templates()  # template_ind=targ.best_temp # we need two templates, therefore we have to initialize all
            # TODO: write the template initialization in such a way that we could pass many template indices
            # TODO: (but if you do this, make sure to remove the other usages that need all the templates)
            self.klc.detect_single_transits(apply_score_correction=True)

    def plot_ses_raw(self, ax=None, figsize=(15, 5), clrs=((0.5, 0.5, 1), (0.2, 0.7, 0.5))):
        self.get_light_curve_for_plotting_ses()
        temp_n = self.best_temp  # because we initialized templates with only one template
        temp_n_sh = self.klc.indices_of_templates[temp_n][0]
        # temp_n_inv = self.best_temp_inv
        # temp_n_sh_inv = self.klc.indices_of_templates[temp_n_inv][0]

        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.gca()
        for q in self.klc.quarters:
            ax.plot(self.klc.times[q], self.klc.statistics_raw_cor[temp_n_sh][q], color=clrs[q % 2], linewidth=1)
        ax.plot(self.summed_transit_times, self.summed_raw_stats, 'o', color='k', label='summed raw', markersize=4)
        for q in self.klc.quarters:
            vm = self.klc.veto_masks[temp_n_sh][q].astype(bool)
            ax.plot(self.klc.times[q][~vm], self.klc.statistics_raw_cor[temp_n_sh][q][~vm], 'o', color='r', markersize=2)
        ax.plot([], [], 'o', color='r', markersize=2, label='vetted statistics')
        ax.set_xlabel('Time, days')
        ax.set_ylabel('Statistics raw')
        ax.set_title('KOI' + self.target_id + ' SES raw\n' + 'period: ' + str(np.round(self.best_per, 1))
                     + ', duration: ' + str(self.templates_durations_hours[self.best_temp]) + ' hours')
        ax.legend(loc='upper right')

    def plot_ses_raw_inv(self, ax=None, figsize=(15, 5), clrs=((0.5, 0.5, 1), (0.2, 0.7, 0.5))):
        self.get_light_curve_for_plotting_ses()
        # temp_n = self.best_temp  # because we initialized templates with only one template
        # temp_n_sh = self.klc.indices_of_templates[temp_n][0]
        temp_n_inv = self.best_temp_inv
        temp_n_sh_inv = self.klc.indices_of_templates[temp_n_inv][0]

        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.gca()
        for q in self.klc.quarters:
            ax.plot(self.klc.times[q], -self.klc.statistics_raw_cor[temp_n_sh_inv][q], color=clrs[q % 2], linewidth=1)
        ax.plot(self.summed_transit_times_inv, self.summed_raw_stats_inv, 'o', color='k', label='summed raw inverted', markersize=4)
        for q in self.klc.quarters:
            vm = self.klc.veto_masks[temp_n_sh_inv][q].astype(bool)
            ax.plot(self.klc.times[q][~vm], -self.klc.statistics_raw_cor[temp_n_sh_inv][q][~vm], 'o', color='r', markersize=2)
        ax.plot([], [], 'o', color='r', markersize=2, label='vetted statistics')
        ax.set_xlabel('Time, days')
        ax.set_ylabel('Statistics raw inverted')
        ax.set_title('KOI' + self.target_id + ' SES raw inverted\n' + 'period: ' + str(np.round(self.best_per_inv, 1))
                     + ', duration: ' + self.templates_durations_hours[self.best_temp_inv] + ' hours')
        ax.legend(loc='upper right')

    def plot_ses_snr(self, ax=None, figsize=(15, 5), clrs=((0.5, 0.5, 1), (0.2, 0.7, 0.5))):
        self.get_light_curve_for_plotting_ses()
        temp_n = self.best_temp  # because we initialized templates with only one template
        temp_n_sh = self.klc.indices_of_templates[temp_n][0]
        # temp_n_inv = self.best_temp_inv
        # temp_n_sh_inv = self.klc.indices_of_templates[temp_n_inv][0]

        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.gca()
        for q in self.klc.quarters:
            ax.plot(self.klc.times[q], self.klc.statistics_snr[temp_n_sh][q], color=clrs[q % 2], linewidth=1)
        ax.plot(self.summed_transit_times, self.summed_raw_stats / self.summed_vars ** 0.5, 'o', color='k', label='summed raw / std',
                markersize=4)
        for q in self.klc.quarters:
            vm = self.klc.veto_masks[temp_n_sh][q].astype(bool)
            ax.plot(self.klc.times[q][~vm], self.klc.statistics_snr[temp_n_sh][q][~vm], 'o', color='r', markersize=2)
        ax.plot([], [], 'o', color='r', markersize=2, label='vetted statistics')
        ax.set_xlabel('Time, days')
        ax.set_ylabel('Statistics snr')
        ax.set_title('KOI' + self.target_id + ' SES SNR\n' + 'period: ' + str(np.round(self.best_per, 1))
                     + ', duration: ' + str(self.templates_durations_hours[self.best_temp]) + ' hours')
        ax.legend(loc='upper right')

    def plot_ses_snr_inv(self, ax=None, figsize=(15, 5), clrs=((0.5, 0.5, 1), (0.2, 0.7, 0.5))):
        self.get_light_curve_for_plotting_ses()
        # temp_n = self.best_temp  # because we initialized templates with only one template
        # temp_n_sh = self.klc.indices_of_templates[temp_n][0]
        temp_n_inv = self.best_temp_inv
        temp_n_sh_inv = self.klc.indices_of_templates[temp_n_inv][0]

        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.gca()
        for q in self.klc.quarters:
            ax.plot(self.klc.times[q], -self.klc.statistics_snr[temp_n_sh_inv][q], color=clrs[q % 2], linewidth=1)
        ax.plot(self.summed_transit_times_inv, self.summed_raw_stats_inv / self.summed_vars_inv ** 0.5, 'o', color='k',
                label='summed raw / std', markersize=4)
        for q in self.klc.quarters:
            vm = self.klc.veto_masks[temp_n_sh_inv][q].astype(bool)
            ax.plot(self.klc.times[q][~vm], -self.klc.statistics_snr[temp_n_sh_inv][q][~vm], 'o', color='r', markersize=2)
        ax.plot([], [], 'o', color='r', markersize=2,
                label='vetted statistics')
        ax.set_xlabel('Time, days')
        ax.set_ylabel('Statistics snr inverted')
        ax.set_title('KOI' + self.target_id + ' SES SNR inverted\n' + 'period: ' + str(np.round(self.best_per_inv, 1))
                     + ', duration: ' + self.templates_durations_hours[self.best_temp_inv] + ' hours')
        ax.legend(loc='upper right')

    def plot_folded_red_flux(self, ax=None, figsize=(8, 6), plot_twice=False):
        # PLOT FOLD RED FLUX
        self.get_light_curve_for_plotting_ses()
        binwid = min(self.templates_durations_hours[self.best_temp] / 24 / 5, 0.1)
        means = [np.nanmean(self.klc.orig_fluxes[q]) for q in self.klc.quarters]

        bins_time, bins_flux, bins_counts, folds_times, folds_fluxes, time_template, template_cropped, bins_std = fold_fluxes(
            times=self.klc.times, fluxes=self.klc.fluxes, period=self.best_per, first_time=self.summed_transit_times[0],
            masks=self.klc.big_hole_masks,
            binwid=binwid, time_fold=self.best_per,  # self.templates_durations_hours[self.best_temp] / 24 * 20,
            dt=self.klc.dt_days,
            means=means,
            template=np.array(self.klc.templates[self.klc.indices_of_templates[self.best_temp][0]]),
            return_std=True)
        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.gca()
        for i in range(len(folds_times)):
            ax.plot(folds_times[i], folds_fluxes[i], 'o', markersize=1, color=(0.5, 0.5, 0.5))
        ax.errorbar(bins_time[bins_counts > 0], bins_flux[bins_counts > 0], yerr=bins_std[bins_counts > 0], xerr=None,
                    markersize=2.5, fmt='o', markerfacecolor='k', markeredgecolor='k',
                    ecolor='k', elinewidth=1.2, capsize=None, barsabove=False, lolims=False, uplims=False,
                    xlolims=False, xuplims=False, errorevery=1, capthick=None,
                    label='binning, bin ' + ut.fstr(np.round(binwid, 3)) + ' days')
        # ax.plot(bins_time, bins_flux, '-o', markersize=3, color='b', linewidth=1,
        #         label='binning, bin ' + ut.fstr(np.round(binwid, 3)) + ' days')
        ax.plot(time_template, 1 - template_cropped * np.abs(np.nanmean(
            self.best_ampls[self.summed_trends > 0] / self.summed_trends[self.summed_trends > 0])),
                color='r', linewidth=1,
                label=r'template $\cdot$ amplitude,'+'\nduration: ' + self.templates_durations_hours[self.best_temp] + ' hours')

        if plot_twice:
            for i in range(len(folds_times)):
                ax.plot(folds_times[i] + self.best_per, folds_fluxes[i], 'o', markersize=1, color=(0.5, 0.5, 0.5))
            ax.errorbar(bins_time[bins_counts > 0] + self.best_per, bins_flux[bins_counts > 0], yerr=bins_std[bins_counts > 0], xerr=None,
                        markersize=2.5, fmt='o', markerfacecolor='k', markeredgecolor='k',
                        ecolor='k', elinewidth=1.2, capsize=None, barsabove=False, lolims=False, uplims=False,
                        xlolims=False, xuplims=False, errorevery=1, capthick=None)
            ax.plot(time_template + self.best_per, 1 - template_cropped * np.abs(np.nanmean(
                self.best_ampls[self.summed_trends > 0] / self.summed_trends[self.summed_trends > 0])),
                    color='r', linewidth=1)

        maxdev = np.nanmax(
            [np.nanmax(np.abs(bins_flux - 1)), np.nanmax([np.nanmax(np.abs(folds_fluxes[i] - 1)) for i in range(len(folds_fluxes))])])
        ax.set_ylim(1 - maxdev * 1.01, 1 + maxdev * 1.01)  # make the limits symmetric
        ax.set_xlim(-self.best_per / 2, self.best_per / 2)
        if plot_twice:
            ax.set_xlim(-self.best_per / 2, self.best_per / 2 + self.best_per)
            ax.axvline(self.best_per / 2, color=(0.5, 0.5, 0.5), linewidth=1)
        ax.legend(loc='upper right')
        ax.set_title('KOI' + self.target_id + ' fold, ' + str(len(self.summed_raw_stats)) + ' transits')
        ax.set_xlabel('Time from the mid transit, days')
        ax.set_ylabel('Flux normalized by mean')

    def plot_folded_red_flux_inv(self, ax=None, figsize=(8, 6)):
        # PLOT FOLD RED FLUX
        self.get_light_curve_for_plotting_ses()
        binwid_inv = min(self.templates_durations_hours[self.best_temp_inv] / 24 / 5, 0.1)
        bins_time, bins_flux, bins_counts, folds_times, folds_fluxes, time_template, template_cropped, bins_std = fold_fluxes(
            times=self.klc.times, fluxes=[-fl for fl in self.klc.fluxes], period=self.best_per_inv,
            first_time=self.summed_transit_times_inv[0],
            masks=self.klc.big_hole_masks,
            binwid=binwid_inv, time_fold=self.best_per_inv,  # self.templates_durations_hours[self.best_temp_inv] / 24 * 10,
            dt=self.klc.dt_days,
            means=[np.nanmean(self.klc.orig_fluxes[q]) for q in self.klc.quarters],
            template=np.array(self.klc.templates[self.klc.indices_of_templates[self.best_temp_inv][0]]),
            return_std=True)
        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.gca()
        for i in range(len(folds_times)):
            ax.plot(folds_times[i], folds_fluxes[i], 'o', markersize=1, color=(0.5, 0.5, 0.5))
        ax.errorbar(bins_time[bins_counts > 0], bins_flux[bins_counts > 0], yerr=bins_std[bins_counts > 0], xerr=None,
                    markersize=2.5, fmt='o', markerfacecolor='k', markeredgecolor='k',
                    ecolor='k', elinewidth=1.2, capsize=None, barsabove=False, lolims=False, uplims=False,
                    xlolims=False, xuplims=False, errorevery=1, capthick=None,
                    label='binning, bin ' + ut.fstr(np.round(binwid_inv, 3)) + ' days')
        # ax.plot(bins_time, bins_flux, '-o', markersize=3, color='b', linewidth=1, label='binning, bin ' + ut.fstr(binwid_inv) + ' days')
        ax.plot(time_template, 1 - template_cropped * np.abs(np.nanmean(
            self.best_ampls_inv[self.summed_trends_inv > 0] / self.summed_trends_inv[self.summed_trends_inv > 0])),
                color='r', linewidth=1,
                label=r'template $\cdot$ amplitude,'+'\nduration: ' + self.templates_durations_hours[self.best_temp_inv] + ' hours')
        maxdev = np.nanmax(
            [np.nanmax(np.abs(bins_flux - 1)), np.nanmax([np.nanmax(np.abs(folds_fluxes[i] - 1)) for i in range(len(folds_fluxes))])])
        ax.set_ylim(1 - maxdev * 1.01, 1 + maxdev * 1.01)  # make the limits symmetric
        ax.set_xlim(-self.best_per_inv / 2, self.best_per_inv / 2)
        ax.legend(loc='upper right')
        ax.set_title('KOI' + self.target_id + ' inverted fold, ' + str(len(self.summed_raw_stats_inv)) + ' transits')
        ax.set_xlabel('Time from the mid transit, days')
        ax.set_ylabel('Flux normalized by mean')

    def plot_folded_whitened_flux(self, ax=None, figsize=(8, 6), plot_twice=False):
        self.get_light_curve_for_plotting_ses()
        binwid = min(self.templates_durations_hours[self.best_temp] / 24 / 5, 0.1)
        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.gca()
        bins_time, bins_flux, bins_counts, folds_times, folds_fluxes, bins_std = fold_fluxes(
            times=self.klc.times, fluxes=self.klc.whitened_fluxes_td, period=self.best_per, first_time=self.summed_transit_times[0],
            masks=self.klc.big_hole_masks,
            binwid=binwid, time_fold=self.best_per,  # self.templates_durations_hours[self.best_temp] / 24 * 10,
            dt=self.klc.dt_days, means=None,
            template=None, return_std=True)  # np.array(self.klc.templates[self.klc.indices_of_templates[self.best_temp][0]]))
        n_time_templ = int(np.round(self.best_per / self.klc.dt_days))
        temps = self.klc.whitened_templates_td[self.klc.indices_of_templates[self.best_temp][0]]
        for q in range(len(temps)):
            if len(temps[q]) < n_time_templ:
                temps[q] = pad_by_zeros(temps[q], n_time_templ, in_the_center=True)
        temp_best_1 = np.array(np.mean([temps[q][:n_time_templ // 2] for q in self.klc.quarters], axis=0))
        temp_best_2 = np.array(np.mean([temps[q][-n_time_templ // 2:] for q in self.klc.quarters], axis=0))
        temp_best = np.concatenate((temp_best_2, temp_best_1))
        tem_center = np.nanargmax(np.abs(temp_best))
        time_template = np.concatenate((np.arange(tem_center) - tem_center, np.arange(len(temp_best) - tem_center))) * self.klc.dt_days

        for i in range(len(folds_times)):
            ax.plot(folds_times[i], folds_fluxes[i], 'o', markersize=1, color=(0.5, 0.5, 0.5))
        ax.errorbar(bins_time[bins_counts > 0], bins_flux[bins_counts > 0], yerr=bins_std[bins_counts > 0], xerr=None,
                    markersize=2.5, fmt='o', markerfacecolor='k', markeredgecolor='k',
                    ecolor='k', elinewidth=1.2, capsize=None, barsabove=False, lolims=False, uplims=False,
                    xlolims=False, xuplims=False, errorevery=1, capthick=None,
                    label='binning, bin ' + ut.fstr(binwid) + ' days')
        # ax.plot(bins_time, bins_flux, '-o', markersize=3, color='b', linewidth=1, label='binning, bin ' + ut.fstr(binwid) + ' days')
        ax.plot(time_template, -temp_best * self.best_ampl, color='r', linewidth=1,
                label='average whitened template,'+'\ntransit duration: ' + self.templates_durations_hours[self.best_temp] + ' hours')

        if plot_twice:
            for i in range(len(folds_times)):
                ax.plot(folds_times[i] + self.best_per, folds_fluxes[i], 'o', markersize=1, color=(0.5, 0.5, 0.5))
            ax.errorbar(bins_time[bins_counts > 0] + self.best_per, bins_flux[bins_counts > 0], yerr=bins_std[bins_counts > 0], xerr=None,
                        markersize=2.5, fmt='o', markerfacecolor='k', markeredgecolor='k',
                        ecolor='k', elinewidth=1.2, capsize=None, barsabove=False, lolims=False, uplims=False,
                        xlolims=False, xuplims=False, errorevery=1, capthick=None)
            ax.plot(time_template + self.best_per, -temp_best * self.best_ampl, color='r', linewidth=1)

        maxdev = np.nanmax(
            [np.nanmax(np.abs(bins_flux)), np.nanmax([np.nanmax(np.abs(folds_fluxes[i])) for i in range(len(folds_fluxes))])])
        ax.set_ylim(-maxdev * 1.01, maxdev * 1.01)  # make the limits symmetric
        ax.set_xlim(-self.best_per / 2, self.best_per / 2)
        if plot_twice:
            ax.set_xlim(-self.best_per / 2, self.best_per / 2 + self.best_per)
            ax.axvline(self.best_per / 2, color=(0.5, 0.5, 0.5), linewidth=1)
        ax.set_xlabel('Time from the mid transit, days')
        ax.set_ylabel('Whitened flux')
        ax.legend(loc='upper right')
        ax.set_title('KOI' + self.target_id + ' whitened fold, ' + str(len(self.summed_raw_stats)) + ' transits')

    def plot_folded_whitened_flux_inv(self, ax=None, figsize=(8, 6)):
        self.get_light_curve_for_plotting_ses()
        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.gca()
        binwid_inv = min(self.templates_durations_hours[self.best_temp_inv] / 24 / 5, 0.1)
        bins_time, bins_flux, bins_counts, folds_times, folds_fluxes, time_template, template_cropped, bins_std = fold_fluxes(
            times=self.klc.times, fluxes=self.klc.whitened_fluxes_td, period=self.best_per,
            first_time=self.summed_transit_times_inv[0],
            masks=self.klc.big_hole_masks,
            binwid=binwid_inv, time_fold=self.best_per_inv,  # self.templates_durations_hours[self.best_temp] / 24 * 10,
            dt=self.klc.dt_days, means=None,
            template=np.array(self.klc.templates[self.klc.indices_of_templates[self.best_temp_inv][0]]))
        n_time_templ = len(time_template)
        temps = self.klc.whitened_templates_td[self.klc.indices_of_templates[self.best_temp_inv][0]]
        for q in range(len(temps)):
            if len(temps[q]) < n_time_templ:
                temps[q] = pad_by_zeros(temps[q], n_time_templ, in_the_center=True)
        temp_best_1 = np.array(np.mean([temps[q][:n_time_templ // 2] for q in self.klc.quarters], axis=0))
        temp_best_2 = np.array(np.mean([temps[q][-n_time_templ // 2:] for q in self.klc.quarters], axis=0))
        temp_best = np.concatenate((temp_best_2, temp_best_1))
        # temp_best_1 = np.array(
        #     np.mean([self.klc.whitened_templates_td[self.klc.indices_of_templates[self.best_temp_inv][0]][q][:n_time_templ // 2]
        #              for q in self.klc.quarters], axis=0))
        # temp_best_2 = np.array(
        #     np.mean([self.klc.whitened_templates_td[self.klc.indices_of_templates[self.best_temp_inv][0]][q][-n_time_templ // 2:]
        #              for q in self.klc.quarters],
        #             axis=0))
        # temp_best = np.concatenate((temp_best_2, temp_best_1))
        for i in range(len(folds_times)):
            ax.plot(folds_times[i], folds_fluxes[i], 'o', markersize=1, color=(0.5, 0.5, 0.5))
        ax.errorbar(bins_time[bins_counts > 0], bins_flux[bins_counts > 0], yerr=bins_std[bins_counts > 0], xerr=None,
                    markersize=2.5, fmt='o', markerfacecolor='k', markeredgecolor='k',
                    ecolor='k', elinewidth=1.2, capsize=None, barsabove=False, lolims=False, uplims=False,
                    xlolims=False, xuplims=False, errorevery=1, capthick=None,
                    label='binning, bin ' + ut.fstr(binwid_inv) + ' days')
        # ax.plot(bins_time, bins_flux, '-o', markersize=3, color='b', linewidth=1, label='binning, bin ' + ut.fstr(binwid_inv) + ' days')
        ax.plot(time_template, -temp_best * self.best_ampl_inv, color='r', linewidth=1,
                label='average whitened template,'+'\ntransit duration: ' + self.templates_durations_hours[self.best_temp_inv] + ' hours')
        maxdev = np.nanmax(
            [np.nanmax(np.abs(bins_flux)), np.nanmax([np.nanmax(np.abs(folds_fluxes[i])) for i in range(len(folds_fluxes))])])
        ax.set_ylim(-maxdev * 1.01, maxdev * 1.01)  # make the limits symmetric
        ax.set_xlim(-self.best_per_inv / 2, self.best_per_inv / 2)
        ax.set_xlabel('Time from the mid transit, days')
        ax.set_ylabel('Whitened flux')
        ax.legend(loc='upper right')
        ax.set_title('KOI' + self.target_id + ' inverted whitened fold, ' + str(len(self.summed_raw_stats_inv)) + ' transits')

    def plot_folded_stat_snr(self, ax=None, figsize=(8, 6), plot_twice=False):
        self.get_light_curve_for_plotting_ses()
        time_full, trend_orig_flux_full = af.connect_quarters_fun(self.klc.times, self.klc.trends_orig_fluxes,
                                                                  zero_padding_ends=None,
                                                                  is_data_cut=~self.klc.is_zero_padded,
                                                                  is_time_cut=~self.klc.is_zero_padded,
                                                                  dt=self.klc.dt_days, fill_gaps_by_nan=False)
        stats_snr = self.klc.statistics_snr[
                    self.klc.indices_of_templates[self.best_temp][0]:self.klc.indices_of_templates[self.best_temp][1]]
        masks_temp = self.klc.stat_masks[self.klc.indices_of_templates[self.best_temp][0]:self.klc.indices_of_templates[self.best_temp][1]]
        stats_snr_full = np.array([af.connect_quarters_fun(self.klc.times, stats_snr[k],
                                                           zero_padding_ends=None, is_data_cut=~self.klc.is_zero_padded,
                                                           is_time_cut=~self.klc.is_zero_padded, dt=self.klc.dt_days,
                                                           fill_gaps_by_nan=False)[1]
                                   for k in range(len(stats_snr))])
        stat_masks_full = np.array([(af.connect_quarters_fun(self.klc.times, masks_temp[k],
                                                             zero_padding_ends=None, is_data_cut=~self.klc.is_zero_padded,
                                                             is_time_cut=~self.klc.is_zero_padded, dt=self.klc.dt_days,
                                                             fill_gaps_by_nan=False)[1]).astype(bool)
                                    for k in range(len(stats_snr))])
        if len(stats_snr_full) > 1:
            time_interleaved = [np.arange(len(trend_orig_flux_full)) * self.klc.dt_days / self.klc.n_shifts + time_full[0]]
            statistics_interleaved = [np.array(stats_snr_full.T.flatten())]
            masks_interleaved = [np.array(stat_masks_full.T.flatten())]
        else:
            time_interleaved = [time_full]
            statistics_interleaved = stats_snr_full
            masks_interleaved = stat_masks_full

        binwid = min(self.templates_durations_hours[self.best_temp] / 24 / 5, 0.1)
        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.gca()
        bins_time, bins_flux, bins_counts, folds_times, folds_fluxes, bins_std = fold_fluxes(
            times=time_interleaved, fluxes=statistics_interleaved, period=self.best_per, first_time=self.summed_transit_times[0],
            masks=masks_interleaved,
            binwid=binwid, time_fold=self.best_per,  # self.templates_durations_hours[self.best_temp] / 24 * 10,
            dt=self.klc.dt_days, means=None,
            template=None, return_std=True)  # np.array(self.klc.templates[self.klc.indices_of_templates[self.best_temp][0]]))

        n_time_templ = int(np.round(self.best_per / self.klc.dt_days))
        temps = self.klc.whitened_templates_td[self.klc.indices_of_templates[self.best_temp][0]]
        for q in range(len(temps)):
            if len(temps[q]) < n_time_templ:
                temps[q] = ut.pad_by_zeros(temps[q], n_time_templ, in_the_center=True)
        temp_best_1 = np.array(np.mean([temps[q][:n_time_templ // 2] for q in self.klc.quarters], axis=0))
        temp_best_2 = np.array(np.mean([temps[q][-n_time_templ // 2:] for q in self.klc.quarters], axis=0))
        temp_best = np.concatenate((temp_best_2, temp_best_1))
        temp_best = temp_best / np.linalg.norm(temp_best)
        tem_center = np.nanargmax(np.abs(temp_best))
        temp_best = np.fft.irfft(np.abs(np.fft.rfft(temp_best)) ** 2)  # for statistic, templete is convolution
        temp_best = np.roll(temp_best, tem_center)
        tem_center = np.nanargmax(np.abs(temp_best))
        time_template = np.concatenate((np.arange(tem_center) - tem_center, np.arange(len(temp_best) - tem_center))) * self.klc.dt_days

        for i in range(len(folds_times)):
            ax.plot(folds_times[i], folds_fluxes[i], 'o', markersize=0.6, color=(0.6, 0.6, 0.6))
        ax.errorbar(bins_time[bins_counts > 0], bins_flux[bins_counts > 0], yerr=bins_std[bins_counts > 0], xerr=None,
                    markersize=2.5, fmt='o', markerfacecolor='k', markeredgecolor='k',
                    ecolor='k', elinewidth=1.2, capsize=None, barsabove=False, lolims=False, uplims=False,
                    xlolims=False, xuplims=False, errorevery=1, capthick=None,
                    label='binning, bin ' + ut.fstr(binwid) + ' days')
        ax.plot(time_template, temp_best * self.best_stat / (len(self.summed_raw_stats)) ** 0.5, color='r', linewidth=1,
                label='average per-transit snr')

        if plot_twice:
            for i in range(len(folds_times)):
                ax.plot(folds_times[i] + self.best_per, folds_fluxes[i], 'o', markersize=0.6, color=(0.6, 0.6, 0.6))
            ax.errorbar(bins_time[bins_counts > 0] + self.best_per, bins_flux[bins_counts > 0], yerr=bins_std[bins_counts > 0], xerr=None,
                        markersize=2.5, fmt='o', markerfacecolor='k', markeredgecolor='k',
                        ecolor='k', elinewidth=1.2, capsize=None, barsabove=False, lolims=False, uplims=False,
                        xlolims=False, xuplims=False, errorevery=1, capthick=None)
            ax.plot(time_template + self.best_per, temp_best * self.best_stat / (len(self.summed_raw_stats)) ** 0.5, color='r', linewidth=1)
        maxdev = np.nanmax(
            [np.nanmax(np.abs(bins_flux)), np.nanmax([np.nanmax(np.abs(folds_fluxes[i])) for i in range(len(folds_fluxes))])])
        ax.set_ylim(-maxdev * 1.01, maxdev * 1.01)  # make the limits symmetric
        ax.set_xlim(-self.best_per / 2, self.best_per / 2)
        if plot_twice:
            ax.set_xlim(-self.best_per / 2, self.best_per / 2 + self.best_per)
            ax.axvline(self.best_per / 2, color=(0.5, 0.5, 0.5), linewidth=1)
        ax.set_xlabel('Time from the mid transit, days')
        ax.set_title('Statistic [snr]', fontsize=14)
        ax.legend(loc='upper right')
        ax.set_title('KOI' + self.target_id + ' statistic snr fold, ' + str(len(self.summed_raw_stats)) + ' transits')

    def plot_background(self, ax=None, figsize=(10, 5), fixed_limits=True, limits=None, plot_extrapolation=True,
                        fontsize_legend=None):
        # PLOT BACKGROUND AND INJECTION-RECOVERY
        if self.prob_density_planet_not_noise is None:
            self.analyze_injections_and_background()
        bins = self.bins_mes2_histogram
        if fixed_limits:
            if limits is not None:
                min_plot_mes2 = limits[0]
                max_plot_mes2 = limits[1]
            else:
                min_plot_mes2 = Default_min_plot_mes2
                max_plot_mes2 = Default_max_plot_mes2
        else:
            min_plot_mes2 = 0.5 * bins[0]
            max_plot_mes2 = 1.1 * bins[-1]
        # UNIFORMLY FORMATTED PLOT
        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.gca()

        ax.stairs(self.distr_mes2_inj_all, bins, label='all injections', linewidth=2)  # after density correction factor, get per one star
        ax.stairs(self.distr_mes2_inj_success, bins, label='success injections',
                  linewidth=2)  # after density correction factor, get per one star
        # Divide by the number of the timeslides, get effectively per one star:
        ax.stairs(self.distr_mes2_h0_direct, bins, label='background direct H0', linewidth=2)
        ax.stairs(self.distr_mes2_h0_inv, bins, label='background inv H0', linewidth=2)
        ax.stairs(self.distr_mes2_h0, bins, label='background total H0', linewidth=2)

        ax.axvline(self.best_stat ** 2, color='k', label='trigger, snr=' + str(np.round(np.abs(self.best_stat), 1)) +
                                                         ',\n  period=' + str(np.round(self.best_per, 1)) + ' d')

        # TODO: this is a bootstrap, you need to make a function for this:
        # log10_rp_edges, period_edges, occurrences_table = get_occurrence_table(self.stellar_radius)  # Per one star
        log10_best_per_low = np.log10(self.best_per - self.per_wing_injection_and_slides)
        log10_best_per_up = np.log10(self.best_per + self.per_wing_injection_and_slides)
        # log10_period_edges = np.log10(period_edges)
        self.log10_best_period_edges = (log10_best_per_low, log10_best_per_up)
        if self.templates_durations_hours is None:
            self.templates_durations_hours = get_templates_durations_from_file()

        ax.set_title('KOI' + self.target_id + ', periods ['
                     + str(np.round(10 ** (self.log10_best_period_edges[0]), 1)) + '-'
                     + str(np.round(10 ** (self.log10_best_period_edges[1]), 1))
                     + '] days, duration=' + str(self.templates_durations_hours[self.best_temp]) + ' hours', fontsize=13)
        ax.set_xlabel('Recovered MES [$SNR^2$]')
        ax.set_ylabel('Counts per effective one star \n per period bin per template')
        ax.set_yscale('log')
        ax.set_xlim(min_plot_mes2, max_plot_mes2)

        if plot_extrapolation:
            # MAKE EXTRAPOLATION
            coefs = self.extrapolation_coefs_mes2_h0_hist
            # determine intersection with horizontal axis in log scale:
            bin_extra_max = (np.log(np.nanmin(self.distr_mes2_h0_direct[self.distr_mes2_h0_direct != 0])) - coefs[0]) / coefs[1]
            bin_extra_max = np.nanmax([(self.best_stat ** 2) + 1, bin_extra_max + 1])
            bin_extra_min = bins[np.nanargmax(self.distr_mes2_h0_direct) + 1]
            bin_min_distr = bins[:-1][((~np.isnan(self.distr_mes2_h0_direct)) & (self.distr_mes2_h0_direct > 0))][0]
            # bin_extra_min = int((bin_extra_min + bin_extra_max) / 2)
            bin_extra_min = bin_extra_min + int(np.abs(bin_extra_min - bin_min_distr) / 2)
            bins_extra = np.linspace(bin_extra_min, bin_extra_max, 10)
            ax.plot(bins_extra, np.exp(bins_extra * coefs[1] + coefs[0]), '--',
                    label='extrapolation, FAP=' + f"{self.false_alarm_prob_trigger:.2e}")
        if fontsize_legend is not None:
            ax.legend(loc='upper right', fontsize=fontsize_legend)
        else:
            ax.legend(loc='upper right')
