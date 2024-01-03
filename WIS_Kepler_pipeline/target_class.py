#!/usr/bin/env python3
# import warnings
# TODO: change the names in such a way that target id is not the last, because it does not give any understanding what
#  is running in the jobname

from sys import getsizeof
from astropy.io import fits
import pickle
import search_pipeline as spip
import template_bank.template_module as tm
from simulations import lightcurve_simulation as sim
import auxiliary_functions as af
import utils as ut
import time
import importlib
from pipeline_parameters import *  # all the parameters, physical constants, occurrence table
from auxiliary_functions import fold_fluxes
from utils import pad_by_zeros
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 11, 'font.serif': 'Times New Roman', 'lines.linewidth': 2})
Def_col = plt.rcParams['axes.prop_cycle'].by_key()['color']

importlib.reload(spip)
np.set_printoptions(threshold=1000, linewidth=10000)

# import warnings
# warnings.filterwarnings("ignore")

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
        self.koi_dict = None
        self.ttv_dict = None
        self.is_villanova_binary = None
        self.villanova_dict = None

        self.periods_in_chunks = None
        self.per_phases_per_chunk = None  # per_phases_per_chunk
        self.p_min = None  # p_min_search
        self.p_max = None  # p_max_search
        self.klc = None
        self.conditioning_ses_time = None  # time of the main search conditioning and ses in seconds
        self.periodicity_search_time = None  # time of the main run periodicity search (mes) in seconds

        # self.candidate_threshold = candidate_threshold
        # self.best_trigger_attributes = ['best_stat', 'best_per', 'best_temp', 'best_phase', 'best_ftt', 'best_ppm',
        #                                'best_ampl', 'summed_raw_stats', 'summed_trends', 'summed_vars', 'ampls', 'ppms']
        # for attr in self.best_trigger_attributes:
        #     self.__dict__[attr] = None
        self.templates_durations_hours = tm.Durations * 24.

        # self.best_score = None
        self.best_stat = None
        self.best_per = None
        self.best_temp = None
        self.best_ftt = None
        self.best_stat_gauss = None
        self.best_integral_score = None
        # self.best_g_score = None
        self.best_g_stat = None
        self.best_g_per = None
        self.best_g_temp = None
        self.best_g_ftt = None
        self.best_g_stat_gauss = None
        self.best_g_integral_score = None

        self.summed_transit_times = None  # times of values that were summed to produce the best trigger
        self.summed_raw_stats = None  # statistics in units not of snr that were summed to produce the best trigger
        self.summed_trends = None
        self.summed_vars = None
        self.summed_corrections = None
        self.summed_gauss_stats = None

        self.is_threshold_crossing = False
        self.per_wing_injection_and_slides = None
        self.best_temp_injection_par_list = None
        # self.template_prior = None
        self.effective_planets_per_template_per_star = None
        self.planets_in_best_period_range_best_template_per_star = None
        self.log10_best_period_edges = None
        self.log2_rp_edges = None
        self.planets_of_radii_in_per_range_best_template_per_star = None

        # Injections attributes

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
        self.slides_gauss_mes2_h0 = None
        self.slides_coh_scores_h0 = None

        self.n_slides = None
        self.bins_mes2_histogram = None
        self.bins_mes2_histogram_inj = None
        self.bins_mes2_histogram_bg = None
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

        # Things with freedom for ttvs and templates, probably useless:
        # self.optim_raw_stats = None
        # self.optim_stat_vars = None
        # self.optim_trends = None
        # self.optim_templs = None
        # self.optim_ampls = None
        # self.optim_ampl = None
        # self.optim_ppms = None
        # self.optim_ppm = None
        # self.ttvs_of_optim = None
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
                # self.candidate_threshold = candidate_threshold
                # self.per_phases_per_chunk = per_phases_per_chunk
                # self.p_min = p_min_search
                # self.p_max = p_max_search
        else:
            if self.gaia_kepler_file is not None:
                self.get_fields_from_gaia(gaia_file_path=self.gaia_kepler_file)
            if get_transit_and_binary_info:
                self.get_koi_and_ttv_fields(koi_dict_file=self.koi_dict_file, ttv_dict_file=self.ttv_dict_file)
                self.get_villanova_fields(villanova_dict_file=self.villanova_dict_file)

        if self.is_villanova_binary:
            self.is_good_for_processing = False
            self.failures += 'villanova; '

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
                print('best_stat_gauss', ':', self.best_stat_gauss, file=fil1)
                print('best_int_score', ':', self.best_integral_score, file=fil1)
                print('', file=fil1)
                print('best_g_stat', ':', self.best_g_stat, file=fil1)
                print('best_g_per', ':', self.best_g_per, file=fil1)
                print('best_g_ftt', ':', self.best_g_ftt, file=fil1)
                print('best_g_stat_gauss', ':', self.best_g_stat_gauss, file=fil1)
                print('best_g_int_score', ':', self.best_g_integral_score, file=fil1)
                print('', file=fil1)
        else:
            for attr in self.star_attributes:
                print(attr, ':', self.__dict__[attr])
            print('')

            print('best_stat', ':', self.best_stat)
            print('best_per', ':', self.best_per)
            print('best_temp', ':', self.best_temp)
            print('best_ftt', ':', self.best_ftt)
            # if self.best_ppm is not None:
            #     print('best_ppm', ':', self.best_ppm)
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

    def get_detectable_radius_flux(self, flux, period, depth_factor=1., snr_min=5, stellar_radius=None,
                                   conditioned_data_filename_postfix=spip.Default_conditioned_data_filename_postfix):
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

        units_of_re_to_units_of_r_star_factor = 1 / (stellar_radius * R_sun_R_e)
        # rp_rs = units_of_re_to_units_of_r_star_factor * rp_re
        rp_re_min = rp_rs_min / units_of_re_to_units_of_r_star_factor
        return rp_re_min

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
        if (self.klc is None) or reinitialize:
            self.klc = spip.KeplerLightCurve(target_id=self.target_id, get_data=True, data_location=self.data_location,
                                             output_location=self.output_location, m_1_3_r_s=self.m_1_3_r_s)  # Q17 is removed
            self.bad_fft_check()  # Made on orig_fluxes
            if self.is_koi and mask_koi_planet:
                self.mask_koi_planet(modify_self_klc=True, return_durations_transit_times=False, include_candidates=False)
                # # if there are only candidates, it will not mask

        file_path_connected_data = af.get_saving_file_path(output_location=self.output_location, target_id=self.target_id,
                                                           filename_postfix=connected_statistic_data_filename_postfix, file_format='npy')
        file_path_conditioned_data = af.get_saving_file_path(output_location=self.output_location, target_id=self.target_id,
                                                             filename_postfix=conditioned_data_filename_postfix, file_format='npy')
        if self.klc.statistics_full is None or not hasattr(self.klc, 'statistics_full'):
            if os.path.exists(file_path_connected_data):
                self.klc.load_connected_stats(file_path=file_path_connected_data)
            else:
                if self.klc.statistics_raw is None:
                    if os.path.exists(file_path_conditioned_data):
                        self.klc.load_conditioned_data(file=file_path_conditioned_data)
                        self.klc.initialize_templates()
                    else:
                        self.klc.condition_data()
                        # self.klc.save_conditioned_data(file=file_path_connected_data)
                        self.klc.condition_templates()
                    self.klc.detect_single_transits(gaussianize=True)
                self.klc.connect_quarters()
            if save_connected_stats:
                self.klc.save_connected_stats(filename_postfix=connected_stats_filename_postfix)

    def run_periodicity_search(self, p_min=None, p_max=None, duty_cycle=spip.Duty_cycle,
                               per_phases_per_chunk=Per_phases_per_chunk,
                               conditioned_data_filename_postfix=Conditioned_data_filename_postfix,
                               best_periodic_scores_filename_postfix=Best_periodic_scores_filename_postfix,
                               best_periodic_scores_file_dir=None,
                               n_best_saving=spip.N_best_saving):

        """
        :param per_phases_per_chunk:
        :type per_phases_per_chunk:
        :param conditioned_data_filename_postfix:
        :type conditioned_data_filename_postfix:
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
        :param n_best_saving:
        :type n_best_saving:
        :return:
        :rtype:
        """

        # if candidate_threshold is not None:
        #     self.candidate_threshold = candidate_threshold
        self.p_min = p_min
        self.p_max = p_max
        self.per_phases_per_chunk = per_phases_per_chunk

        if self.per_phases_per_chunk is not None:
            per_edges = self.klc.make_periods_edges(pmin=self.p_min, pmax=self.p_max,
                                                    per_phases_per_chunk=per_phases_per_chunk, duty_cycle=duty_cycle)

        else:
            per_edges = [[self.p_min, self.p_max]]

        periods = self.klc.make_periods_array(pmin=self.p_min, pmax=self.p_max, duty_cycle=duty_cycle)
        i_last = 0
        self.periods_in_chunks = []
        for edge in per_edges:
            end_edge = np.searchsorted(periods, edge[-1])
            self.periods_in_chunks.append(periods[i_last:end_edge + 1])
            i_last = end_edge + 1

        if self.klc.statistics_interleaved is None:
            if self.klc.statistics_full is None or not hasattr(self.klc, 'statistics_full'):
                file_dir = af.get_saving_file_path(output_location=self.output_location, target_id=self.target_id,
                                                   filename_postfix=Connected_statistic_data_filename_postfix, file_format='npy')
                if os.path.exists(file_dir):
                    self.klc.load_connected_stats(file_path=file_dir)
                elif self.klc.statistics_snr is None:
                    self.klc.load_conditioned_data(filename_postfix=conditioned_data_filename_postfix)
                    self.klc.initialize_templates()
                    self.klc.detect_single_transits(gaussianize=True)
                self.klc.connect_quarters()
            self.klc.interleave_shifts(delete_fulls=True)

        t1 = time.time()

        best_candidates_concat = []
        for pers in self.periods_in_chunks:
            self.klc.periodicity_search(pmin=None, pmax=None, duty_cycle=None, periods=pers,
                                        delete_full_stats=True, delete_periodic_interleaved=True,
                                        save_uncorrected_periodic=False, delete_non_marginalized=True, delete_non_maximized=False)
            # self.klc.save_best_candidates(save_to_file=False, n_best=n_best_saving)
            self.klc.get_best_trigger(make_best_periodic_scores=True)
            best_candidates_concat.append(np.array(self.klc.best_periodic_scores_data, dtype=object))
            self.klc.delete_mes()
        if best_periodic_scores_file_dir is not None:
            np.save(best_periodic_scores_file_dir, np.array(best_candidates_concat, dtype=object), allow_pickle=True)
        else:
            file_path = af.get_saving_file_path(output_location=self.output_location, target_id=self.target_id,
                                                filename_postfix=best_periodic_scores_filename_postfix, file_format='npy')
            np.save(file_path, np.array(best_candidates_concat, dtype=object), allow_pickle=True)

        t2 = time.time()
        self.periodicity_search_time = t2 - t1
        # Check for threshold crossings, and if found, then launch time slides
        self.best_integral_score = -np.inf
        for best_candidates in best_candidates_concat:  # if there is a big MES score, further processing
            scormax = best_candidates[spip.Best_per_results_index_best_integral][0]
            if scormax > self.best_integral_score:
                self.best_integral_score = scormax
                self.best_stat = best_candidates[spip.Best_per_results_index_stats][0]
                self.best_per = best_candidates[spip.Best_per_results_index_per][0]
                self.best_ftt = best_candidates[spip.Best_per_results_index_ftt][0]
                self.best_stat_gauss = best_candidates[spip.Best_per_results_index_stats_gauss][0]

        self.best_g_integral_score = -np.inf
        for best_candidates in best_candidates_concat:  # if there is a big MES score, further processing
            scormax = best_candidates[spip.Best_per_results_index_best_integral][1]
            if scormax > self.best_g_integral_score:
                self.best_g_integral_score = scormax
                self.best_g_stat = best_candidates[spip.Best_per_results_index_stats][1]
                self.best_g_per = best_candidates[spip.Best_per_results_index_per][1]
                self.best_g_ftt = best_candidates[spip.Best_per_results_index_ftt][1]
                self.best_g_stat_gauss = best_candidates[spip.Best_per_results_index_stats_gauss][1]

    def find_best_trigger(self, best_candidates_concat=None):
        # Check for threshold crossings, and if found, then launch time slides
        self.best_integral_score = -np.inf
        for best_candidates in best_candidates_concat:  # if there is a big MES score, further processing
            scormax = best_candidates[spip.Best_per_results_index_best_integral][0]
            if scormax > self.best_integral_score:
                self.best_integral_score = scormax
                self.best_stat = best_candidates[spip.Best_per_results_index_stats][0]
                self.best_per = best_candidates[spip.Best_per_results_index_per][0]
                self.best_ftt = best_candidates[spip.Best_per_results_index_ftt][0]
                self.best_stat_gauss = best_candidates[spip.Best_per_results_index_stats_gauss][0]

        self.best_g_integral_score = -np.inf
        for best_candidates in best_candidates_concat:  # if there is a big MES score, further processing
            scormax = best_candidates[spip.Best_per_results_index_best_integral][1]
            if scormax > self.best_g_integral_score:
                self.best_g_integral_score = scormax
                self.best_g_stat = best_candidates[spip.Best_per_results_index_stats][1]
                self.best_g_per = best_candidates[spip.Best_per_results_index_per][1]
                self.best_g_ftt = best_candidates[spip.Best_per_results_index_ftt][1]
                self.best_g_stat_gauss = best_candidates[spip.Best_per_results_index_stats_gauss][1]
        # self.best_score = -np.inf
        # for best_candidates in best_candidates_concat:  # if there is a big MES score, further processing
        #     scormax = best_candidates[spip.Best_per_results_index_stats][0]
        #     if scormax > self.best_score:
        #         self.best_stat = scormax
        #         self.best_per = best_candidates[spip.Best_per_results_index_per][0]
        #         self.best_ftt = best_candidates[spip.Best_per_results_index_ftt][0]
        #         self.best_stat_gauss = best_candidates[spip.Best_per_results_index_stats_gauss][0]

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
                        threshold_crossing_list_file=None,
                        threshold_crossing_list_location=Threshold_crossing_lists_folder,
                        threshold_crossing_list_filename=Threshold_crossing_list_filename,
                        clean_dir_if_not_threshold_crossing=False):

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
        :param threshold_crossing_list_file:
        :type threshold_crossing_list_file:
        :param threshold_crossing_list_location:
        :type threshold_crossing_list_location:
        :param threshold_crossing_list_filename:
        :type threshold_crossing_list_filename:
        :param clean_dir_if_not_threshold_crossing:
        :type clean_dir_if_not_threshold_crossing:
        :return:
        :rtype:
        """
        if (threshold_crossing_list_file is None) and (threshold_crossing_list_location is not None):
            threshold_crossing_list_file = os.path.join(threshold_crossing_list_location, threshold_crossing_list_filename)

        self.run_ses_search(save_conditioned_data=save_conditioned_data,
                            conditioned_data_filename_postfix=conditioned_data_filename_postfix,
                            save_no_veto_no_score_cor_stats=save_no_veto_no_score_cor_stats,
                            no_veto_no_score_cor_filename_postfix=No_veto_no_score_cor_stats_filename_postfix,
                            save_connected_stats=save_connected_stats, stop_if_bad=True,
                            connected_stats_filename_postfix=Connected_statistic_data_filename_postfix,
                            mask_koi_planet=mask_koi_planet)
        if not self.is_bad_fft:
            self.run_periodicity_search(p_min=p_min, p_max=p_max, per_phases_per_chunk=per_phases_per_chunk,
                                        conditioned_data_filename_postfix=conditioned_data_filename_postfix,
                                        best_periodic_scores_filename_postfix=best_periodic_scores_filename_postfix)

            if (self.best_stat >= candidate_threshold) or (self.best_g_stat_gauss >= candidate_threshold):
                self.is_threshold_crossing = True
                if save_to_threshold_crossing_list:
                    with open(threshold_crossing_list_file, 'a') as fil1:  # will save here these good targets
                        print(self.target_id, self.best_stat, self.best_per, self.best_ftt, self.best_stat_gauss, self.best_integral_score,
                              self.best_g_stat, self.best_g_per, self.best_g_ftt, self.best_g_stat_gauss, self.best_g_integral_score,
                              sep=' ', end='\n', file=fil1)

            if (not self.is_threshold_crossing) and clean_dir_if_not_threshold_crossing:
                # delete conditioned data because we have no space on the disk!
                for ff in os.listdir(self.output_location):
                    os.remove(os.path.join(self.output_location, ff))

        self.pickle_small_attributes()
        self.print_dict_file()

    def run_timeslides(self, n_slides=N_slides_chunk, mask_koi_planet=True,
                       per_wing_slides_relative=Per_wing_injection_and_slides_relative,
                       slides_res_filename_postfix=Slides_res_filename_postfix,
                       # slides_stats_filename_postfix=Slides_stats_filename_postfix,
                       # slides_pers_filename_postfix=Slides_pers_filename_postfix,
                       # slides_first_times_filename_postfix=Slides_first_times_filename_postfix,
                       # slides_gauss_stats_filename_postfix=Slides_gauss_stats_filename_postfix,
                       # slides_int_scores_filename_postfix=Slides_int_scores_filename_postfix,
                       use_period_chunks=False, per_phases_per_chunk=Per_phases_per_chunk,
                       n_best_saving=N_best_saving_slides,
                       use_shuffling_pieces=True, only_if_threshold_crossing=False):
        """
        :param slides_res_filename_postfix:
        :type slides_res_filename_postfix:
        :param only_if_threshold_crossing:
        :type only_if_threshold_crossing:
        :param mask_koi_planet: whether to mask the KOI ephemeris if the object is a KOI
        :type mask_koi_planet: bool
        :param per_wing_slides_relative:
        :type per_wing_slides_relative:
        :param n_slides:
        :type n_slides: int
        :param use_period_chunks: whether to slice periods to chunks to reduce the RAM consummed
        :type use_period_chunks: bool
        :param per_phases_per_chunk: how many period-phases to process at once
        :type per_phases_per_chunk: int
        :param n_best_saving: how many best candidates per template to save
        :type n_best_saving: int
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
        if self.is_threshold_crossing or not only_if_threshold_crossing:
            self.get_connected_klc(reinitialize=True, mask_koi_planet=mask_koi_planet)
            # ATTENTION: we introduced this because of unclear memory issues
            self.klc.delete_irrelevant()

            if use_period_chunks:
                per_edges = self.klc.make_periods_edges(pmin=p_min, pmax=p_max, per_phases_per_chunk=per_phases_per_chunk)
                period_chunks = [self.klc.make_periods_array(pmin=per_edges[i][0], pmax=per_edges[i][1]) for i in range(len(per_edges))]
            else:
                periods = self.klc.make_periods_array(pmin=p_min, pmax=p_max)
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
                if not use_shuffling_pieces:
                    self.klc.time_slides()
                    self.klc.connect_quarters()
                    self.klc.interleave_shifts(delete_fulls=True)
                else:
                    self.klc.scramble_full_stats(max_hole=spip.Max_hole_pieces_scrampling, max_piece=spip.Max_piece_scrambling,
                                                 min_piece=spip.Min_piece_scrambling)
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
                    self.klc.periodicity_search(pmin=None, pmax=None, periods=pers_summing, delete_full_stats=False,
                                                delete_periodic_interleaved=True, save_uncorrected_periodic=False,
                                                delete_non_marginalized=True, delete_non_maximized=False)
                    # self.klc.save_best_candidates(n_best=n_best_saving, save_to_file=False)
                    self.klc.get_best_trigger(make_best_periodic_scores=True)
                    # best_candidates_concat.append(np.array(klc.best_periodic_scores_data, dtype=object))
                    for list_ch, list_g_ch, s_ind in [[stats_chunk, stats_g_chunk, spip.Best_per_results_index_stats],
                                                      [pers_chunk, pers_g_chunk, spip.Best_per_results_index_per],
                                                      [first_times_chunk, first_times_g_chunk, spip.Best_per_results_index_ftt],
                                                      [gauss_stats_chunk, gauss_stats_g_chunk, spip.Best_per_results_index_stats_gauss],
                                                      [int_scores_chunk, int_scores_g_chunk, spip.Best_per_results_index_best_integral]]:
                        list_ch.append(self.klc.best_periodic_scores_data[s_ind][0])
                        list_g_ch.append(self.klc.best_periodic_scores_data[s_ind][1])
                best_ind_chunk = np.argmax(stats_chunk)  # (np.argsort(stats_chunk)[::-1])[:n_best_saving]
                best_g_ind_chunk = np.argmax(gauss_stats_g_chunk)  # (np.argsort(gauss_stats_g_chunk)[::-1])[:n_best_saving]
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
            #             # very important that += is here and not data_list = data_lixt + hist0 !!!
            #             # the second option would not append to the old onject but create a new object, so that wouldn't work!
            #             os.remove(os.path.join(self.output_location, fl))
            #     np.save(os.path.join(self.output_location, self.target_id + '_' + filename_postfix + '_' + str(n_slides)),
            #             np.array(data_list), allow_pickle=True)

    def load_rel_wf_fds(self, conditioned_data_filename_postfix=spip.Default_conditioned_data_filename_postfix):
        path = self.output_location
        file = af.get_saving_file_path(output_location=path, target_id=self.target_id,
                                       filename_postfix=conditioned_data_filename_postfix, file_format='npy')
        conditioned_data = np.load(file, allow_pickle=True)
        support = int(conditioned_data[0][10])
        n_quarters = int(np.round(len(conditioned_data[0]) / support))
        quarters = np.arange(n_quarters)
        len_whitened_templates_td = int(conditioned_data[0][11])
        mean_total = float(conditioned_data[0][13])
        #wf_tds = []
        wf_fds = []
        ind_last = 0
        for q in quarters:
            wftd1 = np.array(conditioned_data[4][ind_last: ind_last + support]).astype(float)
            #wf_tds.append(wftd1)
            wf_fds.append(np.abs(np.fft.rfft(wftd1)))  # np.fft.ifftshift(wftd1)
        return np.array(wf_fds) * mean_total

    def generate_injection_parameters(
            self, n_injections=N_injections_total, add_to_existing_file=False,
            per_wing_injection_relative=Per_wing_injection_and_slides_relative,
            injection_parameters_filename_postfix=Injection_parameters_filename_postfix,
            mean_psd_filename_postfix=spip.Default_psd_for_injections_filename_postfix,
            all_templates_injection_parameters_filename_postfix=All_templates_injection_parameters_filename_postfix,
            delta_snr_from_best_templ=spip.Delta_snr_from_best_veto,
            d_ecc=0.1,
            n_w=5,  #
            d_inc=0.02,  # this is in degrees! and often until 85 is fine
            u=(0.47, 0.23),
            snr=4.,  # something closer to reality to describe smearing to other templates consistently
            n_per=200):
        """
        TODO: write it after I implemented the sum over all the templates
        without a good reason for that, and then the template factor will heavily suppress the result. Maybe I need to account for
        the specific expected snr of this transit to smear the templates much more. Or I need to also inject other shit, but this
        is even more complicated.

        :param delta_snr_from_best_templ:
        :type delta_snr_from_best_templ:
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
        if self.stellar_mass is None or self.stellar_radius is None or np.isnan(self.stellar_mass) or np.isnan(self.stellar_radius):
            self.failures += 'Unable to generate correct injections because of invalid stellar radius or mass;'

        file_path = af.get_saving_file_path(output_location=self.output_location, target_id=self.target_id,
                                            filename_postfix=injection_parameters_filename_postfix, file_format='txt')
        if os.path.exists(file_path) and not add_to_existing_file:
            with open(file_path, 'w'):
                pass  # will delete the contents of the file

        templs_prepared_for_cos_fd = np.conj(np.load(os.path.join(Default_template_bank_folder,
                                                                  Default_templates_fd_for_injections_filename), allow_pickle=True))
        template_support = 2 * (len(templs_prepared_for_cos_fd[0]) - 1)
        dt = tm.dt_lc
        freqs = np.fft.rfftfreq(template_support, d=dt)
        effective_wf_fd = sim.get_saved_wf_fd(freqs=freqs, dt=tm.dt_lc)
        # whitened, normalized, fourier domain, conjugated

        # The parity of the bank is odd, but the length 256 is even, so some tricks with time
        t = tm.time(t_tot=(template_support+1) * tm.dt_lc, parity='odd')[1:]
        # -> make it conj for the ease of cosine
        # The following is needed to calculate the correct cosines from the vectors in the Fourier domain. Only works for even length
        cosine_coeffs = np.ones(len(templs_prepared_for_cos_fd[0])) * 2
        cosine_coeffs[0] = 1.
        cosine_coeffs[-1] = 1.
        cosine_coeffs = cosine_coeffs / template_support
        templs_prepared_for_cos_fd = templs_prepared_for_cos_fd * cosine_coeffs

        if self.per_wing_injection_and_slides is None:
            self.per_wing_injection_and_slides = per_wing_injection_relative * self.best_per

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
        dur_bins = []
        depth_factors = []
        n_star_simulations = 0
        for period in np.random.uniform(self.best_per - self.per_wing_injection_and_slides,
                                        self.best_per + self.per_wing_injection_and_slides, n_per):
            # Attention, this is a linear uniform distribution, although we are using logarithmic occurrences.
            a = (G / (4 * np.pi ** 2) * (period * 24 * 60 * 60) ** 2) ** (1 / 3) * self.m_1_3_r_s * (
                    M_sun ** (1 / 3) / R_sun)  # 3rd Kepler law #in units of R
            # a = (G * M /(4*np.pi**2) * (per*24*60*60)**2)**(1/3) / R #3rd Kepler law #in units of R
            # r_min = a*(1-e) -> 2*r_star=a*(1-e_max)
            n_star_simulations += 1
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
                        if (np.linalg.norm(flux_obs) > 0) and (tm.dt_lc * 2.1 <= dur) and (dur <= 1.5) and len_zeros >= 4:
                            injection_parameter_sets.append([period, a, inc, ecc, w, u[0], u[1]])
                            #get_detectable_radius_flux
                            # # -> these were the requirements for the bank on the duration
                            # flux_obs = np.roll(flux_obs, -np.argmax(flux_obs))  # will out the peak to the left to align everything
                            # # ignore the maximization over shifts, assume it is negligible
                            # wh_flux_obs = np.fft.rfft(flux_obs) * effective_wf_fd
                            # wh_flux_obs = wh_flux_obs / np.linalg.norm(np.fft.irfft(wh_flux_obs))
                            # # wh_flux_obs = wh_flux_obs / np.abs(wh_flux_obs) * cosine_coeffs**0.5  # we do not need unit norm of the noise
                            # snrs_or_cos_angles = np.array([np.real(wh_flux_obs @ templs_prepared_for_cos_fd[j])
                            #                                for j in range(len(templs_prepared_for_cos_fd))])
                            # # snrs_or_cos_angles = snr_1 * np.array(snrs_or_cos_angles)
                            # max_snr_temps = np.max(snrs_or_cos_angles)
                            # inds = np.where((snrs_or_cos_angles > 0) & (snrs_or_cos_angles >= max_snr_temps - delta_snr_from_best_templ))[0]
                            # if len(inds) > 0:
                            #     inds = tm.Templates_of_indices[inds]
                            #     inds = np.array(list(set(inds)))  # remove repetitions
                            #     for ind in inds:
                            #         templates_parameter_sets[ind].append([period, a, inc, ecc, w, u[0], u[1]])
        # Saving the injection parameters for all the templates, although we will only need for one template.
        # Note that the number of injections is different for different templates, and this reflects the template prior.
        # The number of injections in the run will be then renormalized to match the required number.
        file_path = af.get_saving_file_path(output_location=self.output_location, target_id=self.target_id,
                                            filename_postfix=all_templates_injection_parameters_filename_postfix, file_format='npy')
        np.save(file_path, np.array(templates_parameter_sets, dtype=object), allow_pickle=True)
        transit_prob_factor = transit_prob_factor / n_star_simulations  # we divide it by the number of stars that we
        # simulated to make it effective number of planets that can transit and contribute to this template per star
        self.effective_planets_per_template_per_star = transit_prob_factor * np.array([len(templates_parameter_sets[k])
                                                                                       for k in range(len(templates_parameter_sets))])
        # self.template_prior = np.array([len(templates_parameter_sets[k]) for k in range(len(templates_parameter_sets))])
        # self.template_prior = self.template_prior / np.sum(self.template_prior)
        # template_factor = self.template_prior[self.best_temp]
        template_factor = self.effective_planets_per_template_per_star[self.best_temp]
        if template_factor == 0.:
            self.failures += 'zero_occurrence_rate_for_best_template'
        else:
            # ##################################### new code maybe will be here ######################################
            # # TODO: We do not interpolate:
            # # do not account for the case when the value is near the edge of the bin, amd the density is estimated from the two Bins
            # log10_best_per_low = np.log10(self.best_per - self.per_wing_injection_and_slides)
            # log10_best_per_up = np.log10(self.best_per + self.per_wing_injection_and_slides)
            # self.log10_best_period_edges = (log10_best_per_low, log10_best_per_up)
            # get_occurrence_rate(stellar_radius=, per_days=, rp_rearth=None, rp_rstar=)
            # #####################################################################################

            # ##################################### new code ######################################
            # All the things with the occurrence rates and densities:
            log2_rp_re_edges, log10_period_edges, occurrences_table = get_occurrence_table()  # Per one star
            units_of_re_to_units_of_r_star_factor = 1 / (self.stellar_radius * R_sun_R_e)
            log10_best_per_low = np.log10(self.best_per - self.per_wing_injection_and_slides)
            log10_best_per_up = np.log10(self.best_per + self.per_wing_injection_and_slides)
            # log10_period_edges = np.log10(period_edges)
            self.log10_best_period_edges = (log10_best_per_low, log10_best_per_up)

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

            # Number of planets of all radii expected in the considered period range (per 1 star). (sum over radii)
            # planets_in_per_range_one_star = np.sum(planets_of_radii_in_per_range_one_star)
            # Number of planets of this template of all radii of slides periods, per one star:
            # self.planets_in_best_period_range_best_template_per_star = planets_in_per_range_one_star * template_factor
            planets_of_radii_in_per_range_best_template_per_star = (planets_of_radii_in_per_range_one_star
                                                                    * self.effective_planets_per_template_per_star)

            ##########################
            # Some radii will give too small snr so that it makes no sense to even run them
            snr_full = np.abs(self.best_stat)
            n_transits = len(self.summed_raw_stats)  # it is assuming that all the transits are present. If
            # some transits are missed
            snr_1_min = max(0, snr_full / n_transits ** 0.5 - 2.)
            file_path = af.get_saving_file_path(output_location=self.output_location, target_id=self.target_id,
                                                filename_postfix=mean_psd_filename_postfix, file_format='npy')
            mean_rel_psd = np.load(file_path, allow_pickle=True)
            rel_psd_support = 2 * (len(mean_rel_psd) - 1)
            # best_template = tm.TemplateBank(templates=None, shifts=None, template_index='all',
            #                                 dt=tm.dt_lc, support=rel_psd_support, load_essentials=True).templates[self.best_temp][0]
            m_1_3_r_s_1, period_1, a_1, ecc_1, w_1, inc_1, u1_1, u2_1 = tm.get_default_templates_params()[self.best_temp]
            rp_1 = 0.02
            flux_best = 1. - tm.flux_par(t=tm.time(t_tot=(rel_psd_support+1) * tm.dt_lc, parity='odd')[1:], params=None,
                                         per=period_1, rp=rp_1, a=a_1, inc=inc_1, ecc=ecc_1, w=w_1, u1=u1_1, u2=u2_1, t0=0,
                                         supersample_factor=20, exp_time=tm.dt_lc)
            depth_factor = np.max(flux_best) / rp_1**2
            flux_best = np.roll(flux_best, -np.argmax(flux_best))
            flux_best = flux_best / np.max(flux_best) * depth_factor  # pseudo-normalized flux that just needs radius
            # snr_1_exp = rp_trial**2 * np.sum(np.fft.irfft(np.fft.rfft(flux_best) / mean_rel_psd**0.5)**2)**0.5
            rp_min = (snr_1_min / np.sum(np.fft.irfft(np.fft.rfft(flux_best) / mean_rel_psd ** 0.5) ** 2) ** 0.5)**0.5
            log_2_rp_min = np.log2(rp_min)
            ###################################
            # here there should be factor of how many planets per radii which have big enough snr
            log2_rp_edges = np.log2(units_of_re_to_units_of_r_star_factor * 2. ** log2_rp_re_edges)
            if log_2_rp_min > log2_rp_edges[0]:
                column_lower = 0
                value_in_column_lower = planets_of_radii_in_per_range_best_template_per_star[0]
                for i in range(len(log2_rp_edges) - 1):
                    if (log_2_rp_min > log2_rp_edges[i]) and (log_2_rp_min <= log2_rp_edges[i + 1]):
                        column_lower = i
                        value_in_column_lower = (planets_of_radii_in_per_range_best_template_per_star[i] /
                                                 (log2_rp_edges[i + 1] - log2_rp_edges[i]) *
                                                 (log2_rp_edges[i + 1] - log_2_rp_min))
                        break
                self.log2_rp_edges = np.array([log_2_rp_min] + list(log2_rp_edges[column_lower + 1:]))
                self.planets_of_radii_in_per_range_best_template_per_star \
                    = np.array([value_in_column_lower]
                               + list(planets_of_radii_in_per_range_best_template_per_star[column_lower+1:]))
            else:
                self.log2_rp_edges = log2_rp_edges
                self.planets_of_radii_in_per_range_best_template_per_star = planets_of_radii_in_per_range_best_template_per_star
            self.planets_in_best_period_range_best_template_per_star = np.sum(self.planets_of_radii_in_per_range_best_template_per_star)

            # number of injections which will be made for every radii bin
            rp_numbers = (np.ceil(self.planets_of_radii_in_per_range_best_template_per_star /
                          np.sum(self.planets_of_radii_in_per_range_best_template_per_star) * n_injections)).astype(int)

            rps = []
            for i in range(len(self.log2_rp_edges) - 1):
                rps = rps + 2. ** (np.random.uniform(self.log2_rp_edges[i], self.log2_rp_edges[i + 1], rp_numbers[i]))
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
        file_path = af.get_saving_file_path(output_location=self.output_location, target_id=self.target_id,
                                            filename_postfix=injection_parameters_filename_postfix, file_format='txt')
        with open(file_path, 'w') as fil1:
            print('# period[days], a/R, inc[deg 90 or less], ecc, w[deg], u_0, u_1, rp/R', file=fil1)
            for i in range(len(self.best_temp_injection_par_list)):
                print(*self.best_temp_injection_par_list[i], file=fil1)

    def delete_injection_parameters_file(self, injection_parameters_filename_postfix=Injection_parameters_filename_postfix):
        file_path = af.get_saving_file_path(output_location=self.output_location, target_id=self.target_id,
                                            filename_postfix=injection_parameters_filename_postfix, file_format='txt')
        if os.path.exists(file_path):
            os.remove(file_path)

    def load_injection_parameters(self, injection_parameters_index=0,
                                  injection_parameters_filename_postfix=Injection_parameters_filename_postfix):
        file_path = af.get_saving_file_path(output_location=self.output_location, target_id=self.target_id,
                                            filename_postfix=injection_parameters_filename_postfix, file_format='txt')
        with open(file_path, 'r') as fil:
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
                             injections_filename_postfix=Injections_filename_postfix, n_best_saving=N_best_saving_slides):
        """
        :param injections_filename_postfix:
        :type injections_filename_postfix:
        :param injection_load_parameters_index:
        :type injection_load_parameters_index:
        :param per_wing_detection: days to the left and the same to the right in search
        :type per_wing_detection: float
        :param n_best_saving:
        :type n_best_saving:
        :return:
        :rtype:
        """
        # sim_per_job = 20  # with this, 10000 injections in 500 jobs should finish during roughly an hour

        templates_durations_days = self.templates_durations_hours / 24
        pass
        # TODO!!!
        # per_inj, a_inj, inc_inj, ecc_inj, w_inj, u1_inj, u2_inj, rp_inj = self.load_injection_parameters(
        #     injection_parameters_index=injection_load_parameters_index)
        # # This period range is for injection, not for search:
        #
        # phase_inj = np.random.uniform(0, per_inj)
        #
        # tms, flxs = af.get_data_fun(target_id=self.target_id,
        #                             location=self.data_location)
        # first_transit_time_inj = np.nanmin(tms[0]) + phase_inj
        # flxs_inj = sim.inject_planet_to_data(times=tms, fluxes=flxs, period=per_inj, first_transit_center=first_transit_time_inj,
        #                                      rp=rp_inj, a=a_inj, inc=inc_inj, ecc=ecc_inj, w=w_inj, u1=u1_inj, u2=u2_inj,
        #                                      supersample_factor=20, return_template=False)
        # p_max = per_inj + per_wing_detection
        # p_min = per_inj - per_wing_detection
        #
        # self.klc = spip.KeplerLightCurve(target_id=self.target_id, times=tms, fluxes=flxs_inj, get_data=False,
        #                                  data_location=self.data_location, m_1_3_r_s=self.m_1_3_r_s)
        # if self.is_koi:  # masking koi planets
        #     self.mask_koi_planet(modify_self_klc=True, return_durations_transit_times=False, include_candidates=False)
        # self.klc.condition_data()  # IN OUTLIERS MASK, THE POSITIVE AND THE NEGATIVE NSIG ARE EQUAL BECAUSE WE NEED TO INVERT FAIRLY.
        # self.klc.condition_templates()
        # # Mask the suspected real transit. Warning, can kill some of injected signal
        # self.klc.mask_ephemeris(period=self.best_per, durations=templates_durations_days[self.best_temp],
        #                         transit_times=self.best_ftt)
        # self.klc.detect_single_transits(gaussianize=True)
        # self.klc.connect_quarters()
        # self.klc.interleave_shifts(delete_fulls=True)
        # self.klc.sum_periods(pmin=p_min, pmax=p_max)
        # # klc.veto_mes()  # we are not doing it at this stage
        # self.klc.save_best_candidates(n_best=n_best_saving, save_to_file=False)
        # retrieved_mes = self.klc.best_periodic_scores_data[spip.Search_results_index_stat][0][0]
        # retrieved_coh_score = self.klc.best_periodic_scores_data[spip.Search_results_index_coh_score][0][0]
        # retrieved_gauss_mes = self.klc.best_periodic_scores_data[spip.Search_results_index_gauss_stat][0][0]
        # # retrieved_phase = klc.best_periodic_scores_data[3][0][0]  # wrt the beginning of Kepler time
        # retrieved_first_transit_time = self.klc.best_periodic_scores_data[spip.Search_results_index_sum_tr_times][0][0][0]
        # retrieved_period = self.klc.best_periodic_scores_data[spip.Search_results_index_per][0][0]
        # file_path = af.get_saving_file_path(output_location=self.output_location, target_id=self.target_id,
        #                                     filename_postfix=injections_filename_postfix, file_format='txt')
        # with open(file_path, 'a') as fil1:
        #     print(per_inj, a_inj, inc_inj, ecc_inj, w_inj,  # u1_inj, u2_inj,
        #           rp_inj, first_transit_time_inj, retrieved_coh_score,
        #           retrieved_first_transit_time, retrieved_period,
        #           retrieved_mes, retrieved_gauss_mes, sep=' ', end='\n', file=fil1)

    def retrieve_injections(self, max_phase_mismatch_to_dur=Max_phase_mismatch_to_dur,
                            injections_filename_postfix=Injections_filename_postfix):
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
            self.__dict__['retrieved_coh_score_' + status] = []
            self.__dict__['retrieved_gauss_mes_' + status] = []
            self.__dict__['retrieved_first_transit_time_' + status] = []
            self.__dict__['retrieved_period_' + status] = []
        file_path = af.get_saving_file_path(output_location=self.output_location, target_id=self.target_id,
                                            filename_postfix=injections_filename_postfix, file_format='txt')
        with open(file_path, 'r') as fil1:
            inj_lines = fil1.readlines()

        for inj_line0 in inj_lines:
            inj_line = inj_line0.strip()
            inj_line = inj_line.strip('\x00')
            if (len(inj_line) > 5) and (inj_line[0] != '#'):
                per_inj, a_inj, inc_inj, ecc_inj, w_inj, rp_inj, first_transit_time_inj, retrieved_coh_score, \
                    retrieved_first_transit_time, retrieved_period, retrieved_mes, retrieved_gauss_mes = inj_line.split()
                per_inj = float(per_inj)
                a_inj = float(a_inj)
                inc_inj = float(inc_inj)
                ecc_inj = float(ecc_inj)
                w_inj = float(w_inj)
                # u1_inj = float(u1_inj)
                # u2_inj = float(u2_inj)
                rp_inj = float(rp_inj)
                first_transit_time_inj = float(first_transit_time_inj)
                retrieved_coh_score = float(retrieved_coh_score)
                retrieved_gauss_mes = float(retrieved_gauss_mes)
                retrieved_mes = float(retrieved_mes)
                retrieved_first_transit_time = float(retrieved_first_transit_time)
                retrieved_period = float(retrieved_period)
                max_phase_mismatch = tm.Durations[self.best_temp] * max_phase_mismatch_to_dur
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
                self.__dict__['first_transit_time_inj_' + status].append(first_transit_time_inj)
                self.__dict__['retrieved_coh_score_' + status].append(retrieved_coh_score)
                self.__dict__['retrieved_gauss_mes_' + status].append(retrieved_gauss_mes)
                self.__dict__['retrieved_mes_' + status].append(retrieved_mes)
                self.__dict__['retrieved_first_transit_time_' + status].append(retrieved_first_transit_time)
                self.__dict__['retrieved_period_' + status].append(retrieved_period)

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

        bin_max = np.nanmax([inj_max, slides_max, trig_val])
        bin_min = np.nanmin([inj_min, slides_min, trig_val])
        return bin_min, bin_max

    def retrieve_background(self):
        fil_slides_h0_stats = [ff for ff in os.listdir(self.output_location) if
                               ((str(self.target_id) in ff) and (Slides_stats_filename_postfix in ff))][0]
        fil_slides_h0_gauss_mes = [ff for ff in os.listdir(self.output_location) if
                                   ((str(self.target_id) in ff) and (Slides_gauss_stats_filename_postfix in ff))][0]

        self.slides_mes2_h0 = np.load(os.path.join(self.output_location, fil_slides_h0_stats), allow_pickle=True) ** 2
        self.slides_gauss_mes2_h0 = np.load(os.path.join(self.output_location, fil_slides_h0_gauss_mes), allow_pickle=True) ** 2

        self.n_slides = int(
            fil_slides_h0_stats[len(self.target_id + '_' + Slides_stats_filename_postfix + '_'): fil_slides_h0_stats.index('.')])

    def make_histogram_bins(self):
        if self.slides_mes2_h0 is None:
            self.retrieve_background()
        if self.retrieved_mes_success is None:
            self.retrieve_injections()
        bin_min, bin_max = self._get_bin_min_bin_max()
        self.bins_mes2_histogram = np.arange(bin_min, bin_max, Histogram_stats_binwid)

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
                self.bins_mes2_histogram_bg = np.arange(bin_min, bin_max, Histogram_stats_binwid)

        self.distr_mes2_h0 = np.histogram(self.slides_mes2_h0, bins=self.bins_mes2_histogram_bg)[0]
        self.distr_mes2_h0 = self.distr_mes2_h0 / self.n_slides

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
                self.bins_mes2_histogram_inj = np.arange(bin_min, bin_max, Histogram_stats_binwid)

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
            self.distr_mes2_h0, self.bins_mes2_histogram)  # attention: extrapolating only the direct bg
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
        self.make_injections_histogram(bins=self.bins_mes2_histogram)
        self.make_background_histogram(bins=self.bins_mes2_histogram)
        self.calculate_significance(remake_histograms=False)

    def get_ses_for_postprocessing(self, no_filters=True, nsig_outliers_0_pos=5, nsig_outliers_0_neg=15,
                                   load_no_veto_no_score_cor=True,
                                   filename_postfix=No_veto_no_score_cor_stats_filename_postfix,
                                   template_index='all', apply_score_correction=True, apply_veto=False):
        # TODO!!!! WHAT IS THIS FUNTION AND MAKE SURE IT IS NOT USED
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
                    self.klc.condition_data()  # IN OUTLIERS MASK, THE POSITIV AND THE NEGATIV NSIG ARE EQUAL BEC. WE NEED TO INVERT FAIRLY
                    self.klc.condition_templates(template_index=template_index)
                else:
                    file_path = af.get_saving_file_path(output_location=self.output_location, target_id=self.target_id,
                                                        filename_postfix=Conditioned_data_filename_postfix, file_format='npy')
                    self.klc.load_conditioned_data(file=file_path)
                self.klc.initialize_templates(template_index=template_index)
                # template_ind=targ.best_temp # we need two templates, therefore we have to initialize all
                self.klc.detect_single_transits(apply_veto=apply_veto, gaussianize=apply_score_correction)

    # def get_summed_values_no_filters_freedom(self, nsig_outliers_0_pos=5, nsig_outliers_0_neg=15, ttv_hours_allowed=5):
    #     """
    #     DO NOT USE THIS!
    #     similar to `get_summed_values_no_filters`, but allow for ttv and template variation
    #     :param nsig_outliers_0_pos:
    #     :type nsig_outliers_0_pos:
    #     :param nsig_outliers_0_neg:
    #     :type nsig_outliers_0_neg:
    #     :param ttv_hours_allowed:
    #     :type ttv_hours_allowed:
    #     :return:
    #     :rtype:
    #     """
    #     klc = self.get_ses_no_filters(nsig_outliers_0_pos=nsig_outliers_0_pos, nsig_outliers_0_neg=nsig_outliers_0_neg)
    #     ttv_half_bins_allowed = int(np.round(ttv_hours_allowed / klc.dt_days))
    #     best_temp_sh_inds = []
    #     best_quarters = []
    #     best_time_inds = []
    #     ttvs_of_bests = []
    #     for i, tt in enumerate(self.summed_transit_times):
    #         best_stat = 0.
    #         best_templ_sh_ind = None
    #         best_quarter = None
    #         best_time_ind = None
    #         ttv_to_max = None
    #         for q in klc.quarters:
    #             if klc.times[q][~np.isnan(klc.times[q])][0] <= tt <= klc.times[q][~np.isnan(klc.times[q])][-1]:
    #                 ind_tr = np.nanargmin(np.abs(klc.times[q] - tt))
    #                 for k in range(len(klc.statistics_snr)):
    #                     ti0 = max(0, ind_tr - ttv_half_bins_allowed)
    #                     rel_shift = np.nanargmin(klc.statistics_snr[k][q][ti0:ind_tr + ttv_half_bins_allowed])
    #                     time_index = ti0 + rel_shift
    #                     # -> allow the actual minimum to be a little shifted
    #                     stat_here = klc.statistics_snr[k][q][time_index]
    #                     if np.abs(stat_here) > np.abs(best_stat):
    #                         best_stat = stat_here
    #                         best_templ_sh_ind = k
    #                         best_quarter = q
    #                         ttv_to_max = (rel_shift - ttv_half_bins_allowed) * klc.dt_days
    #                         best_time_ind = time_index
    #                         # best_time = klc.times[q][time_index]
    #                         #
    #         if best_time_ind is not None:  # it can be None if the transit is between the quarters or something like that
    #             best_temp_sh_inds.append(best_templ_sh_ind)
    #             best_quarters.append(best_quarter)
    #             best_time_inds.append(best_time_ind)
    #             ttvs_of_bests.append(ttv_to_max)
    #         else:
    #             best_temp_sh_inds.append(np.nan)
    #             best_quarters.append(np.nan)
    #             best_time_inds.append(np.nan)
    #             ttvs_of_bests.append(np.nan)
    #
    #     best_raw_stats = []
    #     best_stat_vars_tot = []
    #     best_trends = []
    #     best_templs = []
    #     optim_templ_depths = []
    #     for i in range(len(best_temp_sh_inds)):
    #         if not np.isnan(best_temp_sh_inds[i]):
    #             best_raw_stats.append(klc.statistics_raw[best_temp_sh_inds[i]][best_quarters[i]][best_time_inds[i]])
    #             best_stat_vars_tot.append(klc.stat_denoms[best_temp_sh_inds[i]][best_quarters[i]])
    #             best_trends.append(klc.trends_orig_fluxes[best_quarters[i]][best_time_inds[i]])
    #             best_templs.append(klc.tb.templates_of_indices[best_temp_sh_inds[i]])
    #             optim_templ_depths.append(np.max(np.abs(klc.tb.templates_shifts[best_temp_sh_inds[i]])))
    #         else:
    #             best_raw_stats.append(np.nan)
    #             best_stat_vars_tot.append(np.nan)
    #             best_trends.append(np.nan)
    #             best_templs.append(np.nan)
    #             optim_templ_depths.append(np.nan)
    #     self.optim_raw_stats = np.array(best_raw_stats)
    #     self.optim_stat_denoms = np.array(best_stat_vars_tot)
    #     self.optim_trends = np.array(best_trends)
    #     self.optim_templs = np.array(best_templs)
    #     self.ttvs_of_optim = np.array(ttvs_of_bests)
    #     optim_templ_depths = np.array(optim_templ_depths)
    #
    #     # templates_of_indices = [k for k in range(len(klc.indices_of_templates))
    #     #                         for _ in range(klc.indices_of_templates[k][0], klc.indices_of_templates[k][1])]
    #     # self.optim_templs = [klc.templates_of_indices[templ_sh_ind] if not np.isnan(templ_sh_ind) else np.nan
    #     #                      for templ_sh_ind in best_temp_sh_inds]
    #
    #     self.optim_ampl, self.optim_ppm, self.optim_ampls, ampls_stds, normalized_ampls, \
    #         normalized_ampls_stds, self.optim_ppms, ppms_stds = \
    #         af.estimate_ampls_ppms_from_stats(summed_raw_stats=self.optim_raw_stats, summed_trends=self.optim_trends,
    #                                           summed_stat_denoms=self.optim_stat_denoms, templ_max=optim_templ_depths, trends_stds=None)

    def get_summed_values_no_filters(self, nsig_outliers_0_pos=5, nsig_outliers_0_neg=15):
        # TODO: revision
        self.get_ses_for_postprocessing(no_filters=True, nsig_outliers_0_pos=nsig_outliers_0_pos, nsig_outliers_0_neg=nsig_outliers_0_neg,
                                        load_no_veto_no_score_cor=True,
                                        template_index=self.best_temp, apply_score_correction=False, apply_veto=False)
        klc = self.klc
        best_raw_stats = []
        best_stat_vars = []
        best_trends = []
        optim_templ_depths = []
        for i, tt in enumerate(self.summed_transit_times):
            best_stat = 0.
            best_time_ind = None
            best_temp_sh_ind = None
            best_quarter = None
            for q in klc.quarters:
                if klc.times[q][~np.isnan(klc.times[q])][0] <= tt <= klc.times[q][~np.isnan(klc.times[q])][-1]:
                    ind_tr = np.nanargmin(np.abs(klc.times[q] - tt))
                    for k in range(len(klc.statistics_raw)):  # we only initialized one template here
                        ti0 = max(0, ind_tr - 1)
                        ti1 = min(ind_tr + 1, len(klc.statistics_raw[k][q]))
                        rel_shift = np.nanargmin(klc.statistics_raw[k][q][ti0:ti1])
                        time_index = ti0 + rel_shift
                        stat_here = klc.statistics_raw[k][q][time_index]
                        if stat_here < best_stat:
                            best_stat = stat_here
                            best_time_ind = time_index
                            best_temp_sh_ind = k
                            best_quarter = q
            if best_time_ind is not None:  # it can be None if the transit is between the quarters or something like that
                best_raw_stats.append(klc.statistics_raw[best_temp_sh_ind][best_quarter][best_time_ind])
                best_stat_vars.append(klc.stat_moving_vars[best_temp_sh_ind][best_quarter][best_time_ind])
                best_trends.append(klc.trends_orig_fluxes[best_quarter][best_time_ind])
                optim_templ_depths.append(np.max(np.abs(klc.tb.templates_shifts[best_temp_sh_ind])))
            else:
                best_raw_stats.append(np.nan)
                best_stat_vars.append(np.nan)
                best_trends.append(np.nan)
                optim_templ_depths.append(np.nan)

        self.best_raw_stats_no_filt = np.array(best_raw_stats)
        self.best_stat_vars_no_filt = np.array(best_stat_vars)
        self.best_trends_no_filt = np.array(best_trends)
        temp_depth = np.max(np.abs(klc.tb.templates_shifts[0]))
        self.best_temp_depth_no_filt = temp_depth

        ampl, ppm, ampls, ampls_stds, normalized_ampls, normalized_ampls_stds, ppms, ppms_stds = \
            af.estimate_ampls_ppms_from_stats(summed_raw_stats=self.best_raw_stats_no_filt,
                                              summed_trends=self.best_trends_no_filt, summed_stat_vars=self.best_stat_vars_no_filt,
                                              templ_max=temp_depth, trends_stds=None)

        self.best_ampls_no_filt = ampls
        self.best_ampls_stds_no_filt = ampls_stds
        self.best_ampl_no_filt = ampl
        self.best_norm_ampls_no_filt = normalized_ampls
        self.best_norm_ampls_stds_no_filt = normalized_ampls_stds
        self.best_ppms_no_filt = ppms
        self.best_ppm_no_filt = ppm
        self.best_ppms_stds_no_filt = ppms_stds

    def estimate_rp(self, nsig_outliers_0_pos=5, nsig_outliers_0_neg=15):
        if self.best_ppms_no_filt is None or self.best_ppm_no_filt is None or self.best_ppms_stds_no_filt is None:
            self.get_summed_values_no_filters(nsig_outliers_0_pos=nsig_outliers_0_pos,
                                              nsig_outliers_0_neg=nsig_outliers_0_neg)
        # ampl, ppm, ampls, ampls_stds, normalized_ampls, normalized_ampls_stds, ppms, ppms_stds = \
        #     estimate_ampls_ppms_from_stats(summed_raw_stats, summed_trends, summed_stat_vars, templ_max, trends_stds=None)
        #
        self.rp_to_rs = (self.best_ppm_no_filt / 1e6) ** 0.5  # the radius estimation is subjected to big error and bias!
        self.rp_to_rs_std = (np.nanvar(np.abs(self.best_ppms_no_filt) ** 0.5)
                             + np.nanmean(self.best_ppms_stds_no_filt ** 2) / np.abs(self.best_ppm_no_filt)) ** 0.5 / 1e3
        self.rp_to_r_earth = self.rp_to_rs * self.stellar_radius * R_sun_R_e
        self.rp_to_r_earth_std = self.rp_to_rs_std * self.stellar_radius * R_sun_R_e

    def copy_output_to_marvin(self, marvin_path=Home_output_storage_folder):
        os.system('scp -r ' + self.output_location + ' ' + marvin_path + '/' + self.target_id[:4])

    def plot_fluxes(self, ax=None, show=True, figsize=(15, 5), clrs=((0.5, 0.5, 1), (0.2, 0.7, 0.5))):
        if self.klc is None or self.klc.veto_masks is None:
            self.get_ses_for_postprocessing(no_filters=False, nsig_outliers_0_pos=5, nsig_outliers_0_neg=15,
                                            template_index=self.best_temp, apply_score_correction=True, apply_veto=True)
        temp_n_sh = 0  # self.klc.indices_of_templates[self.best_temp][0]
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
        if self.klc is None or self.klc.veto_masks is None:
            self.get_ses_for_postprocessing(no_filters=False, nsig_outliers_0_pos=5, nsig_outliers_0_neg=15,
                                            template_index=self.best_temp, apply_score_correction=True, apply_veto=True)
        temp_n_sh = 0  # self.klc.indices_of_templates[self.best_temp][0]
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

    def plot_ses_raw(self, ax=None, figsize=(15, 5), clrs=((0.5, 0.5, 1), (0.2, 0.7, 0.5))):
        if self.klc is None or self.klc.veto_masks is None:
            self.get_ses_for_postprocessing(no_filters=False, nsig_outliers_0_pos=5, nsig_outliers_0_neg=15,
                                            template_index=self.best_temp, apply_score_correction=True, apply_veto=True)
        temp_n_sh = 0  # self.klc.indices_of_templates[self.best_temp][0]
        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.gca()
        for q in self.klc.quarters:
            ax.plot(self.klc.times[q], self.klc.statistics_raw[temp_n_sh][q], color=clrs[q % 2], linewidth=1)
        ax.plot(self.summed_transit_times, self.summed_raw_stats, 'o', color='k', label='summed raw', markersize=4)
        for q in self.klc.quarters:
            vm = self.klc.veto_masks[temp_n_sh][q].astype(bool)
            ax.plot(self.klc.times[q][~vm], self.klc.statistics_raw[temp_n_sh][q][~vm], 'o', color='r', markersize=2)
        ax.plot([], [], 'o', color='r', markersize=2, label='vetted statistics')
        ax.set_xlabel('Time, days')
        ax.set_ylabel('Statistics raw')
        ax.set_title('KOI' + self.target_id + ' SES raw\n' + 'period: ' + str(np.round(self.best_per, 1))
                     + ', duration: ' + str(self.templates_durations_hours[self.best_temp]) + ' hours')
        ax.legend(loc='upper right')

    def plot_ses_snr(self, ax=None, figsize=(15, 5), clrs=((0.5, 0.5, 1), (0.2, 0.7, 0.5))):
        if self.klc is None or self.klc.veto_masks is None:
            self.get_ses_for_postprocessing(no_filters=False, nsig_outliers_0_pos=5, nsig_outliers_0_neg=15,
                                            template_index=self.best_temp, apply_score_correction=True, apply_veto=True)
        temp_n_sh = 0  # self.klc.indices_of_templates[self.best_temp][0]
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

    def plot_folded_red_flux(self, ax=None, figsize=(8, 6), plot_twice=False):
        # PLOT FOLD RED FLUX
        if self.klc is None or self.klc.veto_masks is None:
            self.get_ses_for_postprocessing(no_filters=False, nsig_outliers_0_pos=5, nsig_outliers_0_neg=15,
                                            template_index=self.best_temp, apply_score_correction=True, apply_veto=True)
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
            self.best_ampls_no_filt[self.summed_trends > 0] / self.summed_trends[self.summed_trends > 0])),
                color='r', linewidth=1,
                label=r'template $\cdot$ amplitude,' + '\nduration: ' + self.templates_durations_hours[self.best_temp] + ' hours')

        if plot_twice:
            for i in range(len(folds_times)):
                ax.plot(folds_times[i] + self.best_per, folds_fluxes[i], 'o', markersize=1, color=(0.5, 0.5, 0.5))
            ax.errorbar(bins_time[bins_counts > 0] + self.best_per, bins_flux[bins_counts > 0], yerr=bins_std[bins_counts > 0], xerr=None,
                        markersize=2.5, fmt='o', markerfacecolor='k', markeredgecolor='k',
                        ecolor='k', elinewidth=1.2, capsize=None, barsabove=False, lolims=False, uplims=False,
                        xlolims=False, xuplims=False, errorevery=1, capthick=None)
            ax.plot(time_template + self.best_per, 1 - template_cropped * np.abs(np.nanmean(
                self.best_ampls_no_filt[self.summed_trends > 0] / self.summed_trends[self.summed_trends > 0])),
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

    def plot_folded_whitened_flux(self, ax=None, figsize=(8, 6), plot_twice=False):
        if self.klc is None or self.klc.veto_masks is None:
            self.get_ses_for_postprocessing(no_filters=False, nsig_outliers_0_pos=5, nsig_outliers_0_neg=15,
                                            template_index=self.best_temp, apply_score_correction=True, apply_veto=True)
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
        ax.plot(time_template, -temp_best * self.best_ampl_no_filt, color='r', linewidth=1,
                label='average whitened template,' + '\ntransit duration: ' + self.templates_durations_hours[self.best_temp] + ' hours')

        if plot_twice:
            for i in range(len(folds_times)):
                ax.plot(folds_times[i] + self.best_per, folds_fluxes[i], 'o', markersize=1, color=(0.5, 0.5, 0.5))
            ax.errorbar(bins_time[bins_counts > 0] + self.best_per, bins_flux[bins_counts > 0], yerr=bins_std[bins_counts > 0], xerr=None,
                        markersize=2.5, fmt='o', markerfacecolor='k', markeredgecolor='k',
                        ecolor='k', elinewidth=1.2, capsize=None, barsabove=False, lolims=False, uplims=False,
                        xlolims=False, xuplims=False, errorevery=1, capthick=None)
            ax.plot(time_template + self.best_per, -temp_best * self.best_ampl_no_filt, color='r', linewidth=1)

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

    def plot_folded_stat_snr(self, ax=None, figsize=(8, 6), plot_twice=False):
        if self.klc is None or self.klc.veto_masks is None:
            self.get_ses_for_postprocessing(no_filters=False, nsig_outliers_0_pos=5, nsig_outliers_0_neg=15,
                                            template_index=self.best_temp, apply_score_correction=True, apply_veto=True)
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
                min_plot_mes2 = Default_min_plot_stat
                max_plot_mes2 = Default_max_plot_stat
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
        ax.stairs(self.distr_mes2_h0, bins, label='background H0', linewidth=2)

        ax.axvline(self.best_stat ** 2, color='k', label='trigger, snr=' + str(np.round(np.abs(self.best_stat), 1)) +
                                                         ',\n  period=' + str(np.round(self.best_per, 1)) + ' d')

        # TODO: this is a bootstrap, you need to make a function for this:
        # log10_rp_edges, period_edges, occurrences_table = get_occurrence_table(self.stellar_radius)  # Per one star
        log10_best_per_low = np.log10(self.best_per - self.per_wing_injection_and_slides)
        log10_best_per_up = np.log10(self.best_per + self.per_wing_injection_and_slides)
        # log10_period_edges = np.log10(period_edges)
        self.log10_best_period_edges = (log10_best_per_low, log10_best_per_up)

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
            bin_extra_max = (np.log(np.nanmin(self.distr_mes2_h0[self.distr_mes2_h0 != 0])) - coefs[0]) / coefs[1]
            bin_extra_max = np.nanmax([(self.best_stat ** 2) + 1, bin_extra_max + 1])
            bin_extra_min = bins[np.nanargmax(self.distr_mes2_h0) + 1]
            bin_min_distr = bins[:-1][((~np.isnan(self.distr_mes2_h0)) & (self.distr_mes2_h0 > 0))][0]
            # bin_extra_min = int((bin_extra_min + bin_extra_max) / 2)
            bin_extra_min = bin_extra_min + int(np.abs(bin_extra_min - bin_min_distr) / 2)
            bins_extra = np.linspace(bin_extra_min, bin_extra_max, 10)
            ax.plot(bins_extra, np.exp(bins_extra * coefs[1] + coefs[0]), '--',
                    label='extrapolation, FAP=' + f"{self.false_alarm_prob_trigger:.2e}")
        if fontsize_legend is not None:
            ax.legend(loc='upper right', fontsize=fontsize_legend)
        else:
            ax.legend(loc='upper right')
