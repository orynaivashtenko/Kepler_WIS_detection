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
from file_operations import report_exception, Locker
# import warnings
# warnings.filterwarnings("ignore")
np.set_printoptions(threshold=1000, linewidth=10000)

# TODO: write documentation
# TODO: injections paradox
# TODO: look-elsewhere in the FAP
# TODO: think about a project for bad ffts

"""

Future improvements:
* search more than one planet per star. It would make sense to save everything which is above a certain snr
-> for now, we are only comparing hypotheses nothing vs one planet
* learn to use json
* reorganize methods, e.g. not to initialize templates
* maybe put template generation to the init

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

"""
How to read the documantation:
consult the description
if something is unclear, check the vocabulary
check in many places
"""

"""
VOCABULARY:
KOI - Kepler Object of Interest, meaning something that is a potential planet according to Kepler
TTV - Transit timing variation
quarter - 90-day pieces to which Kepler light curve is split. Every quarter has different location on the detector, contamination, PSD.
SES - 
Raw SES - 
SES SNR - 
MES - 
Main search (run) - 
Condidioning - 
SES search - 
Periodicity search - 
Time shuffling (Timeslides) - 
Injections - 
Background distribution - 
Probability density units

TIME CONVENTIONS
Kepler time, rejected quarters
Mission duration
First transit time - convention
Best-gaussianized sense

Naming notations
Saving formats
"""

"""
SEARCH STRATEGY DESCRIPTION

name postfix notation
periodicity search - chunks
per_phases_per_chunk
duty_cycle
"""

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
                 limb_darkening_array_file=Limb_darkening_array_file,
                 overwrite_location_parameters_by_unpickle=False,
                 allow_output_folder_creation=True,
                 allow_data_folder_creation=True,
                 problems_folder=Problems_folder,
                 clean_problems_file=False):
        """
        Initializes the KeplerTarget object which is used to run the planet search, the statistical significance
        evaluation and some estimations for Kepler stars.
        :param target_id: Kepler ID of the target. A 9-digit number, where some first digits may be zero.
            May be specified as integer or as string, should only contain digits.
        :type target_id: str
        :param data_location: Local path to the folder where data files are stored. The target_id will
            be searched in there. If light curves are not found in the folder, they will be downloaded.
            If folder is not found, it will be created.
        :type data_location: str
        :param output_location: Directory where the output is to be saved. The target ID named folder will be created.
        :type output_location: str
        :param load_pickled_attributes: Whether to load the class attributes which were saved in the `output_location`
        :type load_pickled_attributes: bool
        :param get_transit_and_binary_info: Whether to get the target info about existing planets and binary
            companions from the Kepler catalogs.
        :type get_transit_and_binary_info: bool
        :param gaia_kepler_file_path: The local path to the Gaia-Kepler catalog file
        :type gaia_kepler_file_path: str
        :param koi_dict_file: The local path to the KOI info dictionary file
        :type koi_dict_file: str
        :param ttv_dict_file: The local path to the TTV info file
        :type ttv_dict_file: str
        :param villanova_dict_file: The local path to the Villanova Kepler eclipsing binary info file
        :type villanova_dict_file: str
        :param overwrite_location_parameters_by_unpickle: Whether to use the locations parameters saved in the
            output directory from the previous runs
        :type overwrite_location_parameters_by_unpickle: bool
        :param allow_output_folder_creation: Whether to allow creation of the new output target-ID-named folder if
            the existing one was not found
        :type allow_output_folder_creation: bool
        :param allow_data_folder_creation: Whether to allow creation of the new data target-ID-named folder if
            the existing one was not found
        :type allow_data_folder_creation: bool
        """
        self.target_id = target_id.rjust(9, '0')
        """Kepler target ID. Formatted to be a 9-digit string"""

        # ############################################################################################################
        # Obtaining the locations of the input and the output:
        data_location = af.get_correct_tid_folder(target_id, data_location,
                                                  allow_creation=allow_data_folder_creation)
        output_location = af.get_correct_tid_folder(target_id, output_location,
                                                    allow_creation=allow_output_folder_creation)
        self.data_location = data_location
        """Where Kepler data is saved. If not there, it will be downloaded"""
        self.output_location = output_location
        """Where the output will be saved. A target_id named folder will be made"""
        if problems_folder is None:
            problems_folder = self.output_location
        self.problems_file_location = os.path.join(problems_folder, target_id + '_' + Problems_target_txt_filename_postfix)
        if clean_problems_file:
            self.clean_problems_file()
        """File where issues in the processing will be written"""

        # ############################################################################################################
        # Locations of other files required for the processing:
        self.gaia_kepler_file = gaia_kepler_file_path
        """Where the GAIA-Kepler fun file is saved"""
        self.koi_dict_file = koi_dict_file
        """Where the dictionary of the known KOIs is saved"""
        self.ttv_dict_file = ttv_dict_file
        """Where the known TTV info file is saved"""
        self.villanova_dict_file = villanova_dict_file
        """Where the Villanova Kepler eclipsing binary catalog file is saved"""
        self.limb_darkening_array_file = limb_darkening_array_file
        """Where the limb darkening coefficients table is saved"""

        # ############################################################################################################
        # Target properties from the catalogs:
        self.stellar_radius = None
        """Stellar radius in units of solar radii. Should be obtained from GAIA if exists"""
        self.stellar_mass = None
        """Stellar mass in units of solar masses. Should be obtained from GAIA if exists"""
        self.kepmag = None
        """Stellar Kepler magnitude. Should be obtained from GAIA if exists"""
        self.phot_g_mean_mag = None
        """Stellar photometric G magnitude. Should be obtained from GAIA if exists"""
        self.parallax = None
        """Stellar parallax. Should be obtained from GAIA if exists"""
        self.gaia_abs_gmag = None
        """Stellar absolute G magnitude. Will be calculated from the photometric, if defined"""
        self.gaia_b_r = None
        """Stellar color. Should be obtained from GAIA if exists"""
        self.teff = None
        """Stellar effective temperature in Kelvins. Should be obtained from GAIA if exists"""
        self.metallicity = None
        """Stellar [Fe/H], which is used as [Me/H]"""
        self.logg = None
        """Stellar log g from Gaia"""
        self.ra = None
        """Stellar right ascention coordinate. Should be obtained from GAIA if exists"""
        self.dec = None
        """Stellar declination coordinate. Should be obtained from GAIA if exists"""
        self.ruwe = None
        """GAIA re-normalised unit weight error (RUWE). Should be obtained from GAIA if exists"""
        self.gaia_nconfp = None
        """GAIA number of confirmed planets. Should be obtained from GAIA if exists"""
        self.gaia_non_single_star_flag = None
        """GAIA flag of non-single star. Should be obtained from GAIA if exists"""
        self.gaia_variable_star_flag = None
        """GAIA flag of variable star. Should be obtained from GAIA if exists"""
        self.m_1_3_r_s = None  # M**(1/3)*R[in M_sun and R_sun]
        """Mass of the star in units of solar mass divided by the radius of the star to the power 1/3"""
        self.limb_darkening_coefs = None
        """u1, u2, the quadratic law limb darkening coefficients, approximately defined from 
           stellar temperature, metallicity and log gravity"""

        # ############################################################################################################
        # Data-processing-related facts:
        self.number_of_quarters = None
        """Number of quarters in the Kepler data"""
        self.is_bad_fft = None
        """Whether the Fourier-domain data looks bad, as checked by our function. E.g. contains very large peaks"""
        self.was_notched = None
        """Whether our data conditioning had to apply the notch filter to frequency peaks of this data"""
        self.is_koi = None
        """Whether this star is a Kepler Object of Interest"""
        self.is_koi_planet_masked = None
        """Whether our data processing pipeline masked the KOI ephemeris"""
        self.koi_dict = None
        """Dictionary with info about KOIs in this target star"""
        self.ttv_dict = None
        """Dictionary with info about TTVs in this target star"""
        self.is_villanova_binary = None
        """Whether the processed star appears in the Villanova Kepler eclipsing binary catalog"""
        self.villanova_dict = None
        """Dictionary with info about eclipsing binary properties of this target star"""

        # ############################################################################################################
        # Parameters that will be created by our search pipeline:
        self.periods_in_chunks = None
        """Chunks of period grid used in the periodicity search - see Search Strategy Description"""
        self.per_phases_per_chunk = None
        """Number of period-phase grid in one periods chunk used in the periodicity search - see Search Strategy Description"""
        self.duty_cycle = None
        """Duty cycle in the periodicity search, in days - see Search Strategy Description"""
        self.p_min = None
        """Minimal period used in the search"""
        self.p_max = None
        """Maximal period used in the search"""
        self.klc = None
        """KeplerLightCurve object used for the search - consult `search_pipeline.py`"""
        self.conditioning_ses_time = None
        """Time of the main search conditioning and ses in seconds"""
        self.periodicity_search_time = None
        """Time of the main run periodicity search (mes) in seconds"""
        self.templates_durations_hours = tm.Durations * 24.

        # ############################################################################################################
        # Best trigger results of our search pipeline:
        self.best_stat = None
        """Best score in the search pipeline - see `search_pipeline.py`"""
        self.best_per = None
        """Period in days corresponding to the best score in the search pipeline - see `search_pipeline.py`"""
        self.best_temp = None
        """Template index corresponding to the best score in the search pipeline - see `search_pipeline.py`"""
        self.best_ftt = None
        """First transit time corresponding to the best score in the search pipeline - see `search_pipeline.py`"""
        self.best_gauss_stat = None
        """Gaussianized statistic corresponding to the best score in the search pipeline - see `search_pipeline.py`"""
        self.best_integral_score = None
        """Integral coherent score corresponding to the best score in the search pipeline - see `search_pipeline.py`"""
        self.best_g_stat = None
        """Staitstic corresponding to the best gaussianized score in the search pipeline - see `search_pipeline.py`"""
        self.best_g_per = None
        """Period in days corresponding to the best gaussianized score in the search pipeline - see `search_pipeline.py`"""
        self.best_g_temp = None
        """Template index corresponding to the best gaussianized score in the search pipeline - see `search_pipeline.py`"""
        self.best_g_ftt = None
        """First transit time corresponding to the best gaussianized score in the search pipeline - see `search_pipeline.py`"""
        self.best_g_gauss_stat = None
        """Best gaussianized score in the search pipeline - see `search_pipeline.py`"""
        self.best_g_integral_score = None
        """Integral coherent score corresponding to the best gaussianized score in the search pipeline - see `search_pipeline.py`"""
        self.summed_transit_times = None  # times of values that were summed to produce the best trigger
        """Approximate times when the individual transits of the best trigger happened"""
        self.summed_raw_stats = None  # statistics in units not of snr that were summed to produce the best trigger
        """Single-event matched-filtering scores corresponding to the approximate transit times"""
        self.summed_vars = None
        """Variance of the single-event matched-filtering scores corresponding to the approximate transit times"""
        self.summed_corrections = None
        """Log single-event distribution corrections corresponding to the approximate transit times"""
        self.summed_gauss_stats = None
        """Gaussianized single-event statistics corresponding to the approximate transit times"""

        # ############################################################################################################
        # After the best trigger:
        self.is_threshold_crossing = False
        """Whether the best trigger crossed the pre-defined detection threshold"""

        # ############################################################################################################
        # Injection parameters:
        self.per_wing_injection_and_slides = None
        """Wing of the periods window that will be used to make injections and shuffle timeslides. 
           Is used when the range is specifically required."""
        self.injections_period_range = None
        """Minimal and maximal period of the range that will be injected."""
        self.transit_probability = None
        """Geometric probability to transit. Given that there is a planet, what is the probability that it transits"""
        self.prior_planets_in_period_range_one_star = None
        """Number of planets in the injection period range per one star"""
        self.transiting_planets_of_radii_in_per_range_one_star = None
        """Number of planets in the injection period range per one star that exist and transit"""
        self.transit_and_range_probability = None
        """Probability to transit and to have a radius in the relevant range given that there is a planet of this period"""
        self.transiting_planets_in_relevant_range = None
        """Number of planets in the injection period range in the radii range relevant (snr-wise) for the trigger 
            per one star that exist and transit"""
        self.injection_par_list = None
        """List of parameters that will be injected. List of [period, a, inc, ecc, w, u[0], u[1], rp, dur]"""
        self.prior_radii_distribution = None
        """Probability distribution of the planetary radii in this period range"""
        self.log2_rp_edges = None
        """Edges of bins for probability distribution of the planetary radii"""

        # ############################################################################################################
        # Injections attributes
        self.per_inj_success = None
        """List of injeccted periods of injections which were successfully detected"""
        self.a_inj_success = None
        """List of injected semi-major axes in units of stellar radii of injections which were successfully detected"""
        self.inc_inj_success = None
        """List of injected inclinations in degrees of injections which were successfully detected"""
        self.ecc_inj_success = None
        """List of injected eccentricities of injections which were successfully detected"""
        self.w_inj_success = None
        """List of injected periastron arguments in degrees of injections which were successfully detected"""
        self.rp_inj_success = None
        """List of injected planetary radii in units of stellar radius of injections which were successfully detected"""
        self.first_transit_time_inj_success = None
        """List of injected first transit times of injections which were successfully detected"""
        self.retrieved_stat_success = None
        """List of detected statistics of injections which were successfully detected"""
        self.retrieved_period_success = None
        """List of detected periods in days of injections which were successfully detected"""
        self.retrieved_first_transit_time_success = None
        """List of detected first transit times of injections which were successfully detected"""
        self.retrieved_gauss_stat_success = None
        """List of detected gaussianized statistics of injections which were successfully detected"""
        self.retrieved_integral_score_success = None
        """List of detected integral coherent scores of injections which were successfully detected"""
        self.retrieved_g_stat_success = None
        """List of detected (in the best gaussianised sense) of ststistics of injections which were successfully detected"""
        self.retrieved_g_period_success = None
        """List of detected (in the best gaussianised sense) of periods of injections which were successfully detected"""
        self.retrieved_g_first_transit_time_success = None
        """List of detected (in the best gaussianised sense) of first transit times of injections which were successfully detected"""
        self.retrieved_g_gauss_stat_success = None
        """List of detected (in the best gaussianised sense) of gaussianized statistics of injections which were successfully detected"""
        self.retrieved_g_integral_score_success = None
        """List of detected (in the best gaussianised sense) on integral coherent scores of injections which were successfully detected"""

        self.per_inj_bad = None
        """List of injeccted periods of injections which were missed by the search"""
        self.a_inj_bad = None
        """List of injected semi-major axes in units of stellar radii of injections which were missed by the search"""
        self.inc_inj_bad = None
        """List of injected inclinations in degrees of injections which were missed by the search"""
        self.ecc_inj_bad = None
        """List of injected eccentricities of injections which were missed by the search"""
        self.w_inj_bad = None
        """List of injected periastron arguments in degrees of injections which were missed by the search"""
        self.rp_inj_bad = None
        """List of injected planetary radii in units of stellar radius of injections which were missed by the search"""
        self.first_transit_time_inj_bad = None
        """List of injected first transit times of injections which were missed by the search"""
        self.retrieved_stat_bad = None
        """List of detected statistics of injections which were missed by the search"""
        self.retrieved_period_bad = None
        """List of detected periods in days of injections which were missed by the search"""
        self.retrieved_first_transit_time_bad = None
        """List of detected first transit times of injections which were missed by the search"""
        self.retrieved_gauss_stat_bad = None
        """List of detected gaussianized statistics of injections which were missed by the search"""
        self.retrieved_integral_score_bad = None
        """List of detected integral coherent scores of injections which were missed by the search"""
        self.retrieved_g_stat_bad = None
        """List of detected (in the best gaussianised sense) of ststistics of injections which were missed by the search"""
        self.retrieved_g_period_bad = None
        """List of detected (in the best gaussianised sense) of periods of injections which were missed by the search"""
        self.retrieved_g_first_transit_time_bad = None
        """List of detected (in the best gaussianised sense) of first transit times of injections which were missed by the search"""
        self.retrieved_g_gauss_stat_bad = None
        """List of detected (in the best gaussianised sense) of gaussianized statistics of injections which were missed by the search"""
        self.retrieved_g_integral_score_bad = None
        """List of detected (in the best gaussianised sense) on integral coherent scores of injections which were missed by the search"""

        self.number_of_success_injections = None
        """Total number of injections that were successfully detected"""
        self.number_of_injections = None
        """Total number of injections that were made"""

        # ############################################################################################################
        # Time-scrambling attributes
        self.stats_slides = None
        """List of best statistics that were found in the time-scrambling run"""
        self.pers_slides = None
        """List of best periods in days that were found in the time-scrambling run"""
        self.first_times_slides = None
        """List of best first transit times that were found in the time-scrambling run"""
        self.gauss_stats_slides = None
        """List of best gaussianized statistics that were found in the time-scrambling run"""
        self.int_scores_slides = None
        """List of best integral corerent scores that were found in the time-scrambling run"""
        self.stats_g_slides = None
        """List of statistics (in the best-gaussianized sense) of that were found in the time-scrambling run"""
        self.pers_g_slides = None
        """List of periods (in the best-gaussianized sense) that were found in the time-scrambling run"""
        self.first_times_g_slides = None
        """List of first transit times (in the best-gaussianized sense) that were found in the time-scrambling run"""
        self.gauss_stats_g_slides = None
        """List of gaussianized statistics (in the best-gaussianized sense) that were found in the time-scrambling run"""
        self.int_scores_g_slides = None
        """List of integral corerent scores (in the best-gaussianized sense) that were found in the time-scrambling run"""

        self.n_slides = None
        """Total number of time-scramblings made"""
        self.bins_stat_histogram = None
        """Bins used to build the distribution of the statistics of the time-scrambling and injections"""
        self.bins_stat_histogram_inj = None
        """Bins used to build the distribution of the statistics of the injections"""
        self.bins_stat_histogram_bg = None
        """Bins used to build the distribution of the statistics of the time-scrambling"""
        self.distr_stat_h0 = None
        """Histogram of distribution of the statistics of the time-scrambling"""
        self.distr_stat_inj_all = None
        """Histogram of distribution of the statistics of the injections (successful and missed together)"""
        self.distr_stat_inj_success = None
        """Histogram of distribution of the statistics of the successful injections"""
        # self.extrapolation_coefs_stat_h0_hist = None  # TODO
        self.prob_density_h0_trigger = None
        """Probability density (see units in description) of the background at the location of the trigger"""
        self.false_alarm_prob_trigger = None
        """False-alarm probability of the trigger estimated from the time-scrambling"""  # TODO look-elsewhere
        self.prob_density_h1_trigger = None
        """Probability density (see units in description) of the expected planets at the location of the trigger"""
        self.prob_density_planet_not_noise = None  # TODO not good name density
        """Probability of the trigger to originate from the planet and not from the noise"""

        # Gaussianized statistic (still for the location of the best non-gaussianized score):
        self.bins_gauss_stat_histogram = None
        """Bins of the gaussianized statistic used to build the distribution of the statistics of the time-scrambling and injections"""
        self.bins_gauss_stat_histogram_inj = None
        """Bins of the gaussianized statistic used to build the distribution of the statistics of the injections"""
        self.bins_gauss_stat_histogram_bg = None
        """Bins of the gaussianized statistic used to build the distribution of the statistics of the time-scrambling"""
        self.distr_gauss_stat_h0 = None
        """Histogram of distribution of the gaussianized statistics of the time-scrambling"""
        self.distr_gauss_stat_inj_all = None
        """Histogram of distribution of the gaussianized statistics of the all (success and bad) injections"""
        self.distr_gauss_stat_inj_success = None
        """Histogram of distribution of the gaussianized statistics of the successful injections"""

        # ############################################################################################################
        # Elements of estimations:  # TODO
        # Non-score-corrected values:
        self.best_raw_stats_no_filt = None
        self.best_stat_vars_no_filt = None
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
        """Records problems in data or processing which appeared during the run"""
        self.is_good_for_processing = True

        if load_pickled_attributes:
            self.unpickle_small_attributes(overwrite_location_parameters=overwrite_location_parameters_by_unpickle)
        else:
            if self.gaia_kepler_file is not None:
                self.get_fields_from_gaia(gaia_file_path=self.gaia_kepler_file)
            if get_transit_and_binary_info:
                self.get_koi_and_ttv_fields(koi_dict_file=self.koi_dict_file, ttv_dict_file=self.ttv_dict_file)
                self.get_villanova_fields(villanova_dict_file=self.villanova_dict_file)

        if self.is_villanova_binary:
            self.is_good_for_processing = False
            self.failures += 'villanova; '

    def clean_problems_file(self):
        if os.path.exists(self.problems_file_location):
            os.remove(self.problems_file_location)

    # noinspection DuplicatedCode
    def pickle_small_attributes(self, output_location=None,
                                pickle_small_attributes_filename_postfix=Pickle_target_small_attributes_filename_postfix):
        """
        Pickles all attributes which are of type str, float, int, np.float, np.integer, np.ndarray, list, dict
        :param pickle_small_attributes_filename_postfix:
        :type pickle_small_attributes_filename_postfix:
        :param output_location: The directory in which the output of this target is saved
        :type output_location: str
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
                                  pickle_small_attributes_filename_postfix=Pickle_target_small_attributes_filename_postfix,
                                  overwrite_location_parameters=False):
        """
        Load attributes of the class which were previously pickled by the method `self.pickle_small_attributes`.
        :param output_location: The folder in which the output of this target is saved
        :type output_location: str
        :param pickle_small_attributes_filename_postfix:
        :type pickle_small_attributes_filename_postfix:
        :param overwrite_location_parameters:
        :type overwrite_location_parameters:
        """
        if output_location is None:
            output_location = self.output_location
        file_path = af.get_saving_file_path(output_location=output_location, target_id=self.target_id,
                                            filename_postfix=pickle_small_attributes_filename_postfix, file_format='pickle')

        old_output_location = self.output_location
        old_data_location = self.data_location
        old_gaia_kepler_file_path = self.gaia_kepler_file
        old_koi_dict_file = self.koi_dict_file
        old_ttv_dict_file = self.ttv_dict_file
        old_villanova_dict_file = self.villanova_dict_file
        old_limb_darkening_array_file = self.limb_darkening_array_file

        with open(file_path, 'rb') as pickle_file:
            dict1 = pickle.load(pickle_file)
            self.__dict__.update(dict1.items())

        if not overwrite_location_parameters:
            self.data_location = old_data_location
            self.output_location = old_output_location
            self.gaia_kepler_file = old_gaia_kepler_file_path
            self.koi_dict_file = old_koi_dict_file
            self.ttv_dict_file = old_ttv_dict_file
            self.villanova_dict_file = old_villanova_dict_file
            self.limb_darkening_array_file = old_limb_darkening_array_file

    def get_fields_from_gaia(self, gaia_file_path=None):
        """
        Opens the GAIA-Kepler crossmatch info file and gets some target parameters
        :param gaia_file_path: Local path to the file
        :type gaia_file_path: str
        """
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
            self.metallicity = (gaia_file[1].data['feh'])[index]  # stellar metallicity, [Fe/H]
            self.logg = (gaia_file[1].data['logg'])[index]  # stellar log gravity
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
            self.limb_darkening_coefs = tm.get_limb_darkening_coefs(teff=self.teff, logg=self.logg, metallicity=self.metallicity,
                                                                    limb_darkening_array_file=self.limb_darkening_array_file)

            for key in self.star_attributes:
                try:
                    self.__dict__[key] = float(self.__dict__[key])
                except Exception as e:
                    report_exception(exception=e, file_to_append=self.problems_file_location, title='', info='')
                    self.failures += key + ' error ' + str(e) + '; '
                    self.__dict__[key] = None
            # todo: add fields with units and descriptions. astropy.table can be helpful with that
        gaia_file.close()

    def get_koi_and_ttv_fields(self, koi_dict_file=None, ttv_dict_file=None):
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
        Opens the Villanova Kepler eclipsing binary info file and loads the corresponding fields.
        
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

        :param villanova_dict_file: The local path to the Villanova Kepler eclipsing binary info file
        :type villanova_dict_file: str
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
        Prints some fields from the class dictionary in a txt file to read easily. Prints only small attributes,
        see the definition pf smallness inside the function.
        :param output_location: The folder in which the output of this target is saved
        :type output_location: str
        :param dict_txt_filename_postfix: The postfix to add to the target ID in the filename. See postfix notation.
        :type dict_txt_filename_postfix: str
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
                        ok_to_print = False
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
        """
        Prints the summary file of the search summarizing what are the best trigger parameters
        :param to_file: Whether to print the summary to the file (or to the system output)
        :type to_file: bool
        :param file: Path to the file in which to print the data. Overrides `summary_txt_filename_postfix`
        :type file: str
        :param summary_txt_filename_postfix: Name postfix (see postfix notation) added to the target ID
            in the filename. The file will be saved in the `self.output_location`. Overriden by `file`.
        :type summary_txt_filename_postfix: str
        """
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

    def print_problem(self, text, file=None):
        if file is None:
            file = self.problems_file_location
        with open(file, 'w') as fil1:
            print(text, file=fil1)
            print('\n\n', file=fil1)

    def clean_output_directory(self, leave_periodic_stats=True,
                               best_periodic_scores_filename_postfix=Best_periodic_scores_filename_postfix):
        for ff in os.listdir(self.output_location):
            if (not leave_periodic_stats) or (not (best_periodic_scores_filename_postfix in ff)):
                os.remove(os.path.join(self.output_location, ff))

    def initialize_kepler_light_curve(self, data_location=None, output_location=None,
                                      remove_zeroth_quarter=True, remove_quarter_17=True):
        """
        Initializes an instance of `search_pipeline.KeplerLightCurve` class as an attribute of this class instance.
        KeplerLightCurve is used for data conditioning and search.
        See the description in the `search_pipeline.py`
        :param data_location: Local path to the folder where data files are stored. The target_id will
            be searched in there. If light curves are not found in the folder, they will be downloaded.
            If folder is not found, it will be created.
        :type data_location: str
        :param output_location: The directory in which the output of this target is saved.
             The target ID named folder will be created.
        :type output_location: str
        :param remove_zeroth_quarter: Whether to remove quarter 0. This quarter is small and often bad,
            not intended for science.
        :type remove_zeroth_quarter: bool
        :param remove_quarter_17: Whether to remove quarter 17.
            This quarter is small, there is a hole, and there might be glitch. We remove it
        :type remove_quarter_17: bool
        """
        if data_location is None:
            data_location = self.data_location
        if output_location is None:
            output_location = self.output_location
        self.klc = spip.KeplerLightCurve(target_id=self.target_id, get_data=True, data_location=data_location,
                                         output_location=output_location, m_1_3_r_s=self.m_1_3_r_s,
                                         remove_zeroth_quarter=remove_zeroth_quarter, remove_quarter_17=remove_quarter_17)  # Q17 is removed
        self.number_of_quarters = len(self.klc.quarters)

    def bad_fft_check(self, amp_max_bad_fft=Amp_max_bad_fft, dif_max_bad_fft=Dif_max_bad_fft):
        """
        Checks whether there are very high peaks in the fft of the original fluxes
        :param amp_max_bad_fft: Limit amplitude of the fft maximum (above it the maximum is bad)
        :type amp_max_bad_fft: float
        :param dif_max_bad_fft: Maximal difference between the value at point and the surrounding. Bigger is bad.
        :type dif_max_bad_fft: float
        :return:
        :rtype:
        """
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

    def check_detectable_radius(self, stellar_radius=None, snr_thresh=Stat_candidate_search**0.5-1,
                                rp_re_max=Rp_re_max, rp_rs_max=Rp_rs_max):
        """
        Finds (very approximately) whether there can be detectable planets in a target with
        the average noise properties of this target, assuming that the threshold SNR is `snr_thresh`.
        Assuming that we search planets up to radius `rp_re_max` in units of Earth radii and
        up to `rp_rs_max` in units of stellar radii. That is, we are not interested in brown dwarfs.
        The function takes the average noise properties of the star, checks what is the radius
        corresponding to the thrashold SNR and sees whether this radius is too large or not.
        Writes the results in theattributes of the class.
        :param stellar_radius: Radius of the star in units of solar radii.
        :type stellar_radius: float
        :param snr_thresh: Thre
        :type snr_thresh:
        :param rp_re_max:
        :type rp_re_max:
        :param rp_rs_max:
        :type rp_rs_max:
        """
        if stellar_radius is None:
            stellar_radius = self.stellar_radius
        units_of_re_to_units_of_r_star_factor = 1 / (stellar_radius * R_sun_R_e)
        rp_rs = units_of_re_to_units_of_r_star_factor * rp_re_max
        rp_rs = min(rp_rs, rp_rs_max)

        n_transits_max = len(self.klc.quarters) * 2
        snr_max = (np.max([np.mean([np.linalg.norm(self.klc.whitened_templates_td[kk][q])
                                   for q in self.klc.quarters], axis=0) / np.max(self.klc.tb.templates_shifts[kk])
                          for kk in range(len(self.klc.whitened_templates_td))])
                   * self.klc.mean_total * n_transits_max ** 0.5 * (rp_rs ** 2))
        # -> for an exact conversion between the snr and the radius, look dedicated functions
        if snr_max < snr_thresh:
            self.failures = self.failures + ' too_large_noise; '
            self.is_bad_fft = True

    def get_predicted_snr_flux(self, flux_model, period, depth_factor=1., rp_rs=0.02, wf_fd=None, n_quarters=16,
                               include_detrending=True, nperseg_psd=128,
                               conditioned_data_filename_postfix=spip.Default_conditioned_data_filename_postfix):
        """
        Takes the mean flux model and the period, uploads the noise properties of the star. Checks what would be the signal to noise
        for different planetary radii and selects the minimal and maximal radii accounting for the detection threshold and
        the punishment of too strong signals in the pipeline. Other radii are supposed to be not interesting for the search
        because they will never populate the range of the histogram which surrounds the potential triggers that we are aiming for.
        :param flux_model: Model of the transit, mean flux as function of time.
        :type flux_model: numpy.ndarray 1d of float
        :param period: Period of the supposed planet, days. Defines the number of transits.
        :type period: float
        :param depth_factor: Ratio of the maximal dimming anf the (planet_radius / stellar_radius)**2. Shows how important is the
            limb darkening and the inclination in the minimal attainable dimming. Often this can be significant.
        :type depth_factor: float
        :param rp_rs: Radius of the planet divided by the radius of the star
        :type rp_rs: float
        :param wf_fd: Average whitening filter in Fourier domain
        :type wf_fd: numpy.ndarray 1d
        :param n_quarters:  Number of 90-days wuarters expected in the data. Is important for the number of transits that can hapen
        :type n_quarters: int
        :param include_detrending: Whether to include the fact that in the detection pipeline, the lowest frequencies
            will be detrended, and all the SNR there will be lost.
        :type include_detrending: bool
        :param nperseg_psd: Slice size of the PSD measurement which defines the lowest frequency which will be resolved by
            the PSD, whereas lower frequencies will be detrended.
        :type nperseg_psd: int
        :param conditioned_data_filename_postfix: Text in the filename added to the target ID. See the naming notation
        :type conditioned_data_filename_postfix: str
        :return: The expected SNR for this planet
        :rtype: tuple(float, float)
        """
        if wf_fd is None:
            wf_fds = self.load_rel_wf_fds(conditioned_data_filename_postfix=conditioned_data_filename_postfix)
            # -> relative wf_fd, mean flux already taken into account
            n_quarters = len(wf_fds)
            wf_fd = np.mean(np.array(wf_fds)**2, axis=0) ** 0.5
        if include_detrending:
            support = 2 * (len(wf_fd) - 1)
            freqs_wf_fd = np.fft.rfftfreq(support, d=tm.dt_lc)
            freq_psd_resolution = 1 / (nperseg_psd * tm.dt_lc)
            wf_fd[:np.searchsorted(freqs_wf_fd, freq_psd_resolution)] = 0.
        n_transits_est = np.round(n_quarters * 90. / period)
        flux_model = flux_model / np.nanmax(flux_model) * depth_factor
        flux_model = ut.whiten_vector(flux_model, wf_fd=wf_fd)
        snr_exp = np.linalg.norm(flux_model) * n_transits_est ** 0.5 * (rp_rs ** 2)
        return snr_exp

    def get_rp_inj_from_snr(self, snr_req, per_inj=300., m_1_3_r_s=None, inc_inj=90., ecc_inj=0, w_inj=90., u1_inj=None, u2_inj=None,
                            nperseg_psd=128, include_detrending=True, n_lost_transits=0,
                            conditioned_data_filename_postfix=spip.Default_conditioned_data_filename_postfix):
        if m_1_3_r_s is None:
            m_1_3_r_s = self.m_1_3_r_s
        if u1_inj is None or u2_inj is None:
            u1_inj, u2_inj = self.limb_darkening_coefs
        if u1_inj is None or u2_inj is None:
            u1_inj, u2_inj = Default_limb_darkening_coefs
        wf_fds = self.load_rel_wf_fds(conditioned_data_filename_postfix=conditioned_data_filename_postfix)
        n_quarters = len(wf_fds) - n_lost_transits
        wf_fd = np.mean(np.array(wf_fds) ** 2, axis=0) ** 0.5
        if include_detrending:
            support = 2 * (len(wf_fd) - 1)
            freqs_wf_fd = np.fft.rfftfreq(support, d=tm.dt_lc)
            freq_psd_resolution = 1 / (nperseg_psd * tm.dt_lc)
            wf_fd[:np.searchsorted(freqs_wf_fd, freq_psd_resolution)] = 0.
        n_transits = np.round(n_quarters * 90. / per_inj)
        rp_inj = tm.get_rp_inj(snr_req=snr_req, wf_fd=wf_fd, n_transits=n_transits, per_inj=per_inj, m_1_3_r_s=m_1_3_r_s, inc_inj=inc_inj,
                               ecc_inj=ecc_inj, w_inj=w_inj, u1_inj=u1_inj, u2_inj=u2_inj, nperseg_psd=nperseg_psd,
                               include_detrending=include_detrending, dt=tm.dt_lc)
        return rp_inj

    def get_detectable_radius_flux(self, flux_model, period, depth_factor=1., snr_min=4.5, snr_max_1=8, snr_max_full=40,
                                   max_n_lost_transits=None, min_survived_transits=3,
                                   min_relative_survived_transits=spip.Min_required_nonzero_transits_fraction,
                                   stellar_radius=None, units='star', wf_fd=None, include_detrending=True, nperseg_psd=spip.Nperseg_psd,
                                   n_quarters=16, conditioned_data_filename_postfix=spip.Default_conditioned_data_filename_postfix):
        """
        Takes the mean flux model and the period, uploads the noise properties of the star. Checks what would be the signal to noise
        for different planetary radii and selects the minimal and maximal radii accounting for the detection threshold and
        the punishment of too strong signals in the pipeline. Other radii are supposed to be not interesting for the search
        because they will never populate the range of the histogram which surrounds the potential triggers that we are aiming for.
        :type nperseg_psd: int
        :param flux_model: Model of the transit, mean flux as function of time.
        :type flux_model: numpy.ndarray 1d of float
        :param period: Period of the supposed planet, days. Defines the number of transits.
        :type period: float
        :param depth_factor: Ratio of the maximal dimming anf the (planet_radius / stellar_radius)**2. Shows how important is the
            limb darkening and the inclination in the minimal attainable dimming. Often this can be significant.
        :type depth_factor: float
        :param snr_min: Minimal SNR for the full many-transit data assumed to be detectable. We take extra safety assuming there
            can be dispersion and error, and take this lower.
        :type snr_min: float
        :param snr_max_1: Maximal SNR which can be allocated to one transit. Higher will be cut as outlier or obtain a too big
            punishment from the non-gaussianity.
        :type snr_max_1: float
        :param snr_max_full: Maximal SNR which can be allocated to one transit. Higher is assumed to be outside of our pieline
            operation range, so that it will never influence any density in the range of our possible triggers.
        :type snr_max_full: float
        :param stellar_radius: Radius of the star in units of solar radii.
        :type stellar_radius: float
        :param conditioned_data_filename_postfix: Text in the filename added to the target ID. See the naming notation
        :type conditioned_data_filename_postfix: str
        :param units: 'star' or 'earth'. Whether to report the result in units of stellar radii or earch radii.
        :type units: str
        :param max_n_lost_transits: Maximal number of transits which is allowed to be lost in holes
        :type max_n_lost_transits: int or None
        :param min_survived_transits: Minimal transits that need to survive, not falling on the holes
        :type min_survived_transits: int or None
        :param min_relative_survived_transits: Minimal fraction of transits that need to survive, not falling on the holes
        :type min_relative_survived_transits: float or None
        :param wf_fd: Average whitening filter in Fourier domain
        :type wf_fd: numpy.ndarray 1d
        :param n_quarters:  Number of 90-days wuarters expected in the data. Is important for the number of transits that can hapen
        :type n_quarters: int
        :param include_detrending: Whether to include the fact that in the detection pipeline, the lowest frequencies
            will be detrended, and all the SNR there will be lost.
        :type include_detrending: bool
        :param nperseg_psd: Slice size of the PSD measurement which defines the lowest frequency which will be resolved by
            the PSD, whereas lower frequencies will be detrended.

        :return: rps_res, the resulting minimal and maximal considered radii in the specified units
        :rtype: tuple(float, float)
        """
        if stellar_radius is None:
            stellar_radius = self.stellar_radius
        if wf_fd is None:
            wf_fds = self.load_rel_wf_fds(conditioned_data_filename_postfix=conditioned_data_filename_postfix)
            # -> relative wf_fd, mean flux already taken into account
            n_quarters = len(wf_fds)
            wf_fd = np.mean(np.array(wf_fds)**2, axis=0) ** 0.5
        support = 2 * (len(wf_fd) - 1)
        if include_detrending:
            freqs_wf_fd = np.fft.rfftfreq(support, d=tm.dt_lc)
            freq_psd_resolution = 1 / (nperseg_psd * tm.dt_lc)
            wf_fd[:np.searchsorted(freqs_wf_fd, freq_psd_resolution)] = 0.

        n_transits_max = np.ceil(n_quarters * 90. / period)
        n_transits_min = np.floor(n_quarters * 90. / period)
        n_transits_min_initial = n_transits_min
        # Some number of transits can be lost due to holes, so this is how many are left in the worst case.
        if min_relative_survived_transits is not None:
            min_survived_transits_from_relative = int(np.round(min_relative_survived_transits * n_transits_min_initial))
            n_transits_min = max(n_transits_min, min_survived_transits_from_relative)
        if max_n_lost_transits is not None:
            n_transits_min = max(n_transits_min, n_transits_min_initial - max_n_lost_transits)
        if min_survived_transits is not None:
            n_transits_min = max(n_transits_min, min_survived_transits)

        flux_model = flux_model / np.nanmax(flux_model) * depth_factor
        flux_model = ut.whiten_vector(flux_model, wf_fd=wf_fd)

        # snr_max = np.linalg.norm(flux) * n_transits_max ** 0.5 * (rp_rs ** 2)
        rp_rs_min = (snr_min / (np.linalg.norm(flux_model) * n_transits_max ** 0.5)) ** 0.5
        if snr_max_full is not None:
            rp_rs_max = (snr_max_full / (np.linalg.norm(flux_model) * n_transits_min ** 0.5)) ** 0.5
            if snr_max_1 is not None:
                rp_rs_max = min((snr_max_1 / np.linalg.norm(flux_model)) ** 0.5, rp_rs_max)
        else:
            rp_rs_max = (snr_max_1 / np.linalg.norm(flux_model)) ** 0.5
        rps_res = rp_rs_min, rp_rs_max
        if units == 'earth':
            units_of_re_to_units_of_r_star_factor = 1 / (stellar_radius * R_sun_R_e)
            # rp_rs = units_of_re_to_units_of_r_star_factor * rp_re
            rp_re_min = rp_rs_min / units_of_re_to_units_of_r_star_factor
            rp_re_max = rp_rs_max / units_of_re_to_units_of_r_star_factor
            rps_res = rp_re_min, rp_re_max
        return rps_res

    def mask_koi_planet(self, modify_self_klc=True, return_durations_transit_times=False, include_candidates=False):
        """
        Take the information of the KIO of this target that are saved in the class instance fields. Find the times and the
        durations of transits of these planets. If `modify_self_klc` is True, then in the KeplerLightCurve attribute
        `self.klc` modufy the masks to exclude the transits from the search.
        :param modify_self_klc: Whether to modify the masks in `self.klc` to mask the KOI transits
        :type modify_self_klc: bool
        :param return_durations_transit_times: Whether to return the durations and the transit times of KOI planets.
        :type return_durations_transit_times: bool
        :param include_candidates: Whether to include in the process the Kepler planets with status 'Candidate',
            as opposed to 'Confirmed'.
        :type include_candidates: bool
        :return: durations_and_times, durations and the transit times of KOI planets of this target
        :rtype: list of list(numpy.ndarray 1d, numpy.ndarray 1d)
        """
        # TODO We could also subtract villanova ephemeris.
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
        Search for the single-event statistics. Is done throught the KeplerLightCurve instance, so see more detalis
        in `search_pipeline.py`. Includes the following:
        - Getting the data, checking whether it is good enough for the search, masking possible KOI
        - Conditioning data and templates
        - Checking the detectable radii in this noise, and stop the search if it is too bad
        - Running the matched-filtering for single-transit templates
        - Vetting statistics, assessing the distribution non-gaussianity
        - Connecting the quarters into cintinuous data ready for the peruiodicity search
        - Saving the processed data for future use
        The next stage after this would be to run the periodicity search.

        :param save_conditioned_data: Whether to save conditioned data
        :type save_conditioned_data: bool
        :param conditioned_data_filename_postfix: Text added to the target ID in the name of the file saving conditioned data.
            See naming notations.
        :type conditioned_data_filename_postfix: str
        :param save_connected_stats: Whether to save statistics with connected quarters which can be used later
            for periodicity search.
        :type save_connected_stats: bool
        :param connected_stats_filename_postfix: Text added to the target ID in the name of the file saving
            statistics with connected quarters. See naming notations.
        :type connected_stats_filename_postfix:
        :param save_no_veto_no_score_cor_stats: Whether to save statistics which did not undergo veto or score correction.
        :type save_no_veto_no_score_cor_stats: bool
        :param no_veto_no_score_cor_filename_postfix: Text added to the target ID in the name of the file saving
            statistics which did not undergo veto or score correction. See naming notations.
        :type no_veto_no_score_cor_filename_postfix: str
        :param mask_koi_planet: Whether to mask Kepler planets
        :type mask_koi_planet: bool
        :param stop_if_bad: Whether to stop the search if the target was revealed to have a too bad PSD or if it was
            understood that the radii of the planets that can be detected in this search are too big.
        :type stop_if_bad: bool
        """
        t1 = time.time()
        self.klc = spip.KeplerLightCurve(target_id=self.target_id, get_data=True, data_location=self.data_location,
                                         output_location=self.output_location, m_1_3_r_s=self.m_1_3_r_s)  # Q17 is removed
        self.bad_fft_check()  # Made on orig_fluxes
        if self.is_koi and mask_koi_planet:
            self.mask_koi_planet(modify_self_klc=True, return_durations_transit_times=False, include_candidates=False)
            # Attention: we do not mask candidates
        self.klc.condition_data()  # In outlier mask, positive and negative nsig_outliers are the same because we run inversion after.
        self.was_notched = self.klc.was_notched
        self.klc.condition_templates()
        self.check_detectable_radius(stellar_radius=self.stellar_radius)
        if self.is_bad_fft:
            self.print_problem('is bad fft')
        if self.is_koi:
            self.print_problem('is koi')
        if self.was_notched:
            self.print_problem('was notched')
        if (not self.is_bad_fft) or (not stop_if_bad):
            if save_conditioned_data:
                self.klc.save_conditioned_data(filename_postfix=conditioned_data_filename_postfix)
            self.klc.detect_single_transits(apply_veto=True, gaussianize=True, save_no_veto_no_score_cor_stats_as_attr=True)
            if save_no_veto_no_score_cor_stats:
                self.klc.save_no_veto_no_score_cor_stats(filename_postfix=no_veto_no_score_cor_filename_postfix)
            self.klc.connect_quarters()
            if save_connected_stats:
                self.klc.save_connected_stats(filename_postfix=connected_stats_filename_postfix)
            self.klc.delete_irrelevant()  # For memory
            self.klc.interleave_shifts(delete_fulls=True)

            t2 = time.time()
            self.conditioning_ses_time = t2 - t1

    def get_connected_klc(self, reinitialize=True, conditioned_data_filename_postfix=Conditioned_data_filename_postfix,
                          connected_statistic_data_filename_postfix=Connected_statistic_data_filename_postfix,
                          mask_koi_planet=True, save_connected_stats=True,
                          connected_stats_filename_postfix=Connected_statistic_data_filename_postfix):
        """
        In the KeplerLightCurve instance attribute (`self.klc`) of this class, load the precomputed statistics
        (or compute if there are no precomputed) with connected quarters which can be used in the periodicity
        search. See more detalis about the statistics in `search_pipeline.py`. If this method is executed, you
        will get `self.klc` with ready single-transit statistics at which the periodicity search can be run.
        These statistics also can be time-scrambled.
        :param reinitialize: Whether to reinstantiate `self.klc` even if it already existed before.
        :type reinitialize: bool
        :param conditioned_data_filename_postfix: Text added to the target ID in the name of the file saving conditioned data.
            See naming notations. This is needed in the case that the connected statistics file does not exixt. Then
            the conditioned data can be loaded and processed to make it all faster.
        :type conditioned_data_filename_postfix: str
        :param connected_statistic_data_filename_postfix: Text added to the target ID in the name of the file saving
            conditioned data. See naming notations.
        :type connected_statistic_data_filename_postfix: str
        :param mask_koi_planet: Whether to mask Kepler planets in the case `self.klc` is created or reinstantiated.
        :type mask_koi_planet: bool or None
        :param save_connected_stats: Whether to save connected statistics in the case they were calculated now.
        :type save_connected_stats: bool
        :param connected_stats_filename_postfix: Text added to the target ID in the name of the file saving connected data.
            See naming notations.
        :type connected_stats_filename_postfix: str
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
                try:
                    with Locker(file_path_connected_data):
                        self.klc.load_connected_stats(file_path=file_path_connected_data)
                except Exception as exc:
                    report_exception(exception=exc, file_to_append=self.problems_file_location,
                                     title='statistics_full file', info='')
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

    def set_periodicity_search_parameters(self, p_min=P_min_search, p_max=P_max_search, per_phases_per_chunk=Per_phases_per_chunk,
                                          duty_cycle=spip.Duty_cycle, force_overwrite=False):
        """
        Sets the parameters for the periodicity search as attributes of the class for the future use.
        :param p_min: Minimal period in the search.
        :type p_min: float
        :param p_max: Maximal period in the search.
        :type p_max: float
        :param per_phases_per_chunk: The total number of period-phase bins which should be in one chunk.
            Is set by memory estimation. Number of phases is different for every period, therefore chunks will have
            different number of periods in them.
        :type per_phases_per_chunk: int
        :param duty_cycle: Maximal accumulated deviation due to the period error in the end of the mission.
            In units of days.
        :type duty_cycle: float
        :param force_overwrite: Whether to overwrite the class fields if they already exist
        :type force_overwrite: bool
        """
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
        Runs the search for periodically repeating transits basing on the single-event statistic search done earlier
        on the KeplerLightCurve instance attribute, `self.klc` (see in `search_pipeline.py`). Includes the following:
        - Splitting the period range specified by the user into chunks which can be fit into the RAM limitations.
            One star is supposed to occupy one core with limited memory and runtime. The period range is first
            built automatically to cover uniformly the periods range. It makes sure that planet of any periodicity
            inside this range can get a sufficient SNR to be detected. Then this period range is split automatically
            into chunks which have the same number of period-phase cells. The memory requirement is set by the size
            of the array with all the period and phase bins. Each period has different number of phases, therefore the
            shapes of the chunks can be different, but they will have approximately the same total number of elements.
            Every chunk also has approximately the same look-elsewhere weight.
            Then every chunk is run separately in the periodicity search.
        - Runs the periodicity folding of the statistics and other arrays for every chunk.
        - Performs marginalization over the templates and corrects for the ses distribution non-gaussianity.
        - From each of the chunks, selects the best trigger, applying the transit-consistency veto. Checks whether the
            best trigger is consistent with having transits of the same depth, as opposed to being dominated by one
            peak and having the rest to be zero.
        - Compares the best triggers of all the chunks accounting for the priors. Selects the best trigger per star.
        See details in the periodicity search description of `search_pipeline.py`.
        :param p_min: Minimal period in the search.
        :type p_min: float
        :param p_max: Maximal period in the search.
        :type p_max: float
        :param duty_cycle: Maximal accumulated deviation due to the period error in the end of the mission.
            In units of days.
        :type duty_cycle: float
        :param per_phases_per_chunk: The total number of period-phase bins which should be in one chunk.
            Is set by memory estimation. Number of phases is different for every period, therefore chunks will have
            different number of periods in them.
        :type per_phases_per_chunk: int
        :param best_periodic_scores_filename_postfix: Text added to the target ID in the name of the file saving
            the best periodic scores found for this target. See naming notations.
        :type best_periodic_scores_filename_postfix: str
        :param best_periodic_scores_file_dir: Local folder in which to put the file with saved data. if None,
            `self.output_location` will be used.
        :type best_periodic_scores_file_dir: str
        :param overlap_period_bins: When creating period chunks, extend each chunk from the two sides by
            this number of bins for them to overlap. is needed for coherent score integral in order for
            the maximum not to be on the edge
        :type overlap_period_bins: int
        """
        self.set_periodicity_search_parameters(p_min=p_min, p_max=p_max, per_phases_per_chunk=per_phases_per_chunk, duty_cycle=duty_cycle,
                                               force_overwrite=True)

        if self.klc is None or not hasattr(self.klc, 'statistics_full') or self.klc.statistics_full is None:
            self.get_connected_klc(reinitialize=False, mask_koi_planet=None)
        self.klc.delete_irrelevant()  # For memory

        self.periods_in_chunks = self.klc.make_periods_chunks(p_min=self.p_min, p_max=self.p_max,
                                                              per_phases_per_chunk=self.per_phases_per_chunk,
                                                              duty_cycle=duty_cycle, dt=self.klc.dt_days, t_mission=self.klc.T_mission)
        t1 = time.time()
        best_candidates_concat = []
        for pers in self.periods_in_chunks:
            try:
                self.klc.periodicity_search(p_min=None, p_max=None, duty_cycle=None, periods=pers,
                                            delete_full_stats=True, force_interleave_shifts=False, delete_periodic_interleaved=True,
                                            save_uncorrected_periodic=False, delete_non_marginalized=False,  # delete_non_maximized=False,
                                            overlap_period_bins=overlap_period_bins)
                self.klc.get_best_trigger(apply_veto_mes=True, in_gaussianized=False)
                self.klc.get_best_trigger(apply_veto_mes=True, in_gaussianized=True)
                self.klc.make_best_periodic_scores()
                best_candidates_concat.append(np.array(self.klc.best_periodic_scores_data, dtype=object))
            except Exception as exc:
                report_exception(exception=exc, file_to_append=self.problems_file_location,
                                 title='pers=' + str(pers[0]) + '-' + str(pers[1]), info='')
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
        """
        Compares the scores of several chunks and sets the best results as class attributes
        :param best_candidates_concat: The properly formatted list of the best cancicates from every chunk
        :type best_candidates_concat: list of lists of length 2
        """
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
                        clean_dir_if_not_threshold_crossing=False, overlap_period_bins=spip.P_bins_overlap,
                        clean_problems_file=True):

        """
        Runs the ses-search and the periodicity search. See `self.run_ses_search` and `run_periodicity_search`.
        Saves the trigger to the all stars triggers file.
        :param p_min: Minimal period in the search.
        :type p_min: float
        :param p_max: Maximal period in the search.
        :type p_max: float
        :param per_phases_per_chunk: The total number of period-phase bins which should be in one chunk.
            Is set by memory estimation. Number of phases is different for every period, therefore chunks will have
            different number of periods in them.
        :type per_phases_per_chunk: int
        :param best_periodic_scores_filename_postfix: Text added to the target ID in the name of the file saving
            the best periodic scores found for this target. See naming notations.
        :type best_periodic_scores_filename_postfix: str
        :param overlap_period_bins: When creating period chunks, extend each chunk from the two sides by
            this number of bins for them to overlap. is needed for coherent score integral in order for
            the maximum not to be on the edge
        :type overlap_period_bins: int
        :param mask_koi_planet: Whether to mask Kepler planets
        :type mask_koi_planet: bool
        :param candidate_threshold: Threshold in units of prior-aware SNR squared of the threshold above which trigger
            is considered for time-slides and injections.
        :type candidate_threshold: float
        :param save_to_threshold_crossing_list: Whether to append the result to the list of all triggers for all stars.
        :type save_to_threshold_crossing_list: bool
        :param threshold_crossing_list_file: Full path to the file saving the list of all triggers for all stars.
            Overrides `threshold_crossing_list_location` and `threshold_crossing_list_filename`.
        :type threshold_crossing_list_file: str
        :param threshold_crossing_list_location: Path to the folder wn which file with `threshold_crossing_list_filename`
            is to be found or created. Overriden by `threshold_crossing_list_file`.
        :type threshold_crossing_list_location:
        :param threshold_crossing_list_filename: Name (no path) of the file with the list of all triggers for all stars.
            Overriden by `threshold_crossing_list_file`.
        :type threshold_crossing_list_filename: str
        :param save_conditioned_data: Whether to save conditioned date for future use
        :type save_conditioned_data: bool
        :param save_no_veto_no_score_cor_stats: Whether to save statistics which did not undergo veto or score correction.
        :type save_no_veto_no_score_cor_stats: bool
        :param save_connected_stats:  Whether to save statistics with connected quarters which can be used later
            for timeslides periodicity search.
        :type save_connected_stats: bool
        :param best_periodic_scores_filename_postfix: Text added to the target ID in the name of the file saving
            the best scores from the periodicity search. See naming notations.
        :type best_periodic_scores_filename_postfix: str
        :param conditioned_data_filename_postfix: Text added to the target ID in the name of the file saving conditioned data.
            See naming notations.
        :type conditioned_data_filename_postfix: str
        :param no_veto_no_score_cor_stats_filename_postfix: Text added to the target ID in the name of the file saving
            statistics which did not undergo veto or score correction. See naming notations.
        :type no_veto_no_score_cor_stats_filename_postfix: str
        :param connected_stats_data_filename_postfix: Text added to the target ID in the name of the file saving
            statistics with connected quarters. See naming notations.
        :type connected_stats_data_filename_postfix: str
        :param clean_dir_if_not_threshold_crossing: Whether to delete all non-summary data from the output folder if the
            best trigger is not threshold-crossing. Is done for storage memory purposes.
        :type clean_dir_if_not_threshold_crossing: bool
        :param overlap_period_bins: when creating period chunks, extend each chunk from the two sides by
            this number of bins for them to overlap. is needed for coherent score integral in order for
            the maximum not to be on the edge
        :type overlap_period_bins: int
        :param clean_problems_file: Whether to clean before the search the file that reports the exceptions
            and the problems of the search
        :type clean_problems_file: bool
        """
        if (threshold_crossing_list_file is None) and (threshold_crossing_list_location is not None):
            threshold_crossing_list_file = os.path.join(threshold_crossing_list_location, threshold_crossing_list_filename)

        if clean_problems_file:
            self.clean_problems_file()

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
                self.clean_output_directory(leave_periodic_stats=True)

        self.pickle_small_attributes()
        self.print_dict_file()

    def run_timeslides(self, n_slides=N_slides_chunk, central_period=None, mask_koi_planet=True,
                       n_chunks_around_best=1, per_wing_slides=None,
                       per_wing_slides_relative=Per_wing_injection_and_slides_relative, use_period_chunks=False,
                       use_shuffling_pieces=True, only_if_threshold_crossing=False, overlap_period_bins=spip.P_bins_overlap,
                       slides_res_filename_postfix=Slides_res_filename_postfix):
        """
        Shuffles the full statistics (see inside the methods how it is done), running the periodicity search
        on the shuffled SES in the specified pariod range, and saving the best scores.
        The timeslides are usually run to estimate the background density in the location of the trigger from
        the main search, therefore the periods of the timeslides search are close to the best period from the
        main search. This method selects the adjacent periods and runs the shuffled search for them.
        There are three ways to organize the period grid:
        - Just run on one grid
        - Split the grid into chunks
        - Select the corresponding chunks from the chunked grid of the main search

        :param n_slides: Number of times to shuffle and run the perodicity search
        :type n_slides: int
        :param central_period: The period in days around which the timeslides will run. If None, `self.best_per` will be used.
        :type central_period: float
        :param mask_koi_planet: Whether to mask the KOI ephemeris if the object is a KOI
        :type mask_koi_planet: bool
        :param n_chunks_around_best: How many period chunks from the main search around the best period to use for
            the periodicity search in the shuffled data. Overrides `per_wing_slides` and `per_wing_slides_relative`.
        :type n_chunks_around_best: int
        :param per_wing_slides: Wing in days to take to the left and to the right from the central period to build
            the period grid. Overrides `per_wing_slides_relative`. Overriden by `n_chunks_around_best`.
        :type per_wing_slides: float
        :param per_wing_slides_relative: Relative wing to take to the left and to the right from the central period
            to build the period grid. Meaning that the central period will be multiplitd by this number to get the
            value in days. Thet is, the size of the wing will be proportional to the central period.
            Overriden by `n_chunks_around_best` and `per_wing_slides`.
        :type per_wing_slides_relative: float
        :param use_period_chunks: Whether to slice periods to chunks to reduce the RAM consumed. Used if
            `n_chunks_around_best` is None. Otherwise, the chunks from the main search are used.
        :type use_period_chunks: bool
        :param use_shuffling_pieces: Whether to cut quarters into pieces and reshuffle them while preserving the
            total duration of the data. Alternative is time-slides, where every quarter just gets a random positive shift
            of the time axis, so that the total duration of the data is not preserved. This can create scores with too
            large number of transits which is not representative of the main search.
        :type use_shuffling_pieces: bool
        :param only_if_threshold_crossing: Only run the time-shuffling if the main search provided a threshold-crossing
            best trigger.
        :type only_if_threshold_crossing: bool
        :param overlap_period_bins: When creating period chunks, extend each chunk from the two sides by
            this number of bins for them to overlap. is needed for coherent score integral in order for
            the maximum not to be on the edge.
        :type overlap_period_bins: int
        :param slides_res_filename_postfix: Text added to the target ID in the name of the file saving the results.
            See naming notations.
        :type slides_res_filename_postfix: str
        """
        if self.is_threshold_crossing or not only_if_threshold_crossing:
            self.get_connected_klc(reinitialize=True, mask_koi_planet=mask_koi_planet)
            self.klc.delete_irrelevant()  # For memory
            # In case some parameters were not set, replace them with the defaults:
            self.set_periodicity_search_parameters(force_overwrite=False)

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
                if per_wing_slides is not None:
                    self.per_wing_injection_and_slides = per_wing_slides
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
                                                 min_piece=spip.Min_piece_scrambling, random_seed=new_seed,
                                                 regenerate_pieces=False)  # we run many jobs, they will have different pieces
                    self.klc.interleave_shifts(delete_fulls=False)  # do not delete fulls to then make re-scrambling

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
                        self.klc.periodicity_search(p_min=None, p_max=None, periods=pers_summing, delete_full_stats=False,
                                                    force_interleave_shifts=True,  # very important: new slides!
                                                    delete_periodic_interleaved=True, save_uncorrected_periodic=False,
                                                    delete_non_marginalized=False, overlap_period_bins=overlap_period_bins)
                        self.klc.get_best_trigger(apply_veto_mes=True, in_gaussianized=False)
                        self.klc.get_best_trigger(apply_veto_mes=True, in_gaussianized=True)
                        self.klc.make_best_periodic_scores()
                        for list_ch, list_g_ch, s_ind in [[stats_chunk, stats_g_chunk, spip.Best_per_results_index_stats],
                                                          [pers_chunk, pers_g_chunk, spip.Best_per_results_index_per],
                                                          [first_times_chunk, first_times_g_chunk, spip.Best_per_results_index_ftt],
                                                          [gauss_stats_chunk, gauss_stats_g_chunk, spip.Best_per_results_index_stats_gauss],
                                                          [int_scores_chunk, int_scores_g_chunk, spip.Best_per_results_index_best_integral]
                                                          ]:
                            list_ch.append(self.klc.best_periodic_scores_data[s_ind][0])
                            list_g_ch.append(self.klc.best_periodic_scores_data[s_ind][1])
                    except Exception as exc:
                        report_exception(exception=exc, file_to_append=self.problems_file_location,
                                         title='slides pers='+str(pers_summing[0])+'-'+str(pers_summing[1]), info='seed='+str(new_seed))
                    self.klc.delete_mes()
                if len(int_scores_chunk) > 0:  # If everything failed, there will be no result
                    best_ind_chunk = np.argmax(int_scores_chunk)
                    best_g_ind_chunk = np.argmax(int_scores_g_chunk)
                    for list_full, list_full_g, list_ch, list_ch_g in [[stats, stats_g, stats_chunk, stats_g_chunk],
                                                                       [pers, pers_g, pers_chunk, pers_g_chunk],
                                                                       [first_times, first_times_g, first_times_chunk, first_times_g_chunk],
                                                                       [gauss_stats, gauss_stats_g, gauss_stats_chunk, gauss_stats_g_chunk],
                                                                       [int_scores, int_scores_g, int_scores_chunk, int_scores_g_chunk]]:
                        list_full.append(list_ch[best_ind_chunk])
                        list_full_g.append(list_ch_g[best_g_ind_chunk])

            new_res_array = np.vstack((stats, pers, first_times, gauss_stats, int_scores,
                                       stats_g, pers_g, first_times_g, gauss_stats_g, int_scores_g)).transpose()

            file_path = af.get_saving_file_path(output_location=self.output_location, target_id=self.target_id,
                                                filename_postfix=slides_res_filename_postfix, file_format='npy')
            try:
                with Locker(file_path):
                    if os.path.exists(file_path):
                        hist_slides = np.load(file_path, allow_pickle=True)
                        hist_slides = np.vstack((hist_slides, new_res_array))
                    else:
                        hist_slides = new_res_array
                    n_slides = len(hist_slides)
                    np.save(file_path, hist_slides, allow_pickle=True)
                    with open(os.path.join(self.output_location, 'number_of_slides.txt'), 'w') as fil1:
                        print(n_slides, file=fil1)
            except Exception as exc:
                report_exception(exception=exc, file_to_append=self.problems_file_location,
                                 title='saving timeslides', info='')

    def load_rel_wf_fds(self, conditioned_data_filename_postfix=spip.Default_conditioned_data_filename_postfix):
        """
        Load quarters-mean whitening filter in Fourier domain in units relative to the mean flux level.
        In other words, part of flux in time-domain that is taken by this or that component of noise.
        :param conditioned_data_filename_postfix: ext added to the target ID in the name of the file saving conditioned data.
            See naming notations.
        :type conditioned_data_filename_postfix: str
        """
        path = self.output_location
        file = af.get_saving_file_path(output_location=path, target_id=self.target_id,
                                       filename_postfix=conditioned_data_filename_postfix, file_format='npy')
        conditioned_data = np.load(file, allow_pickle=True)
        support = int(conditioned_data[0][10])
        n_quarters = int(np.round(len(conditioned_data[0]) / support))
        quarters = np.arange(n_quarters)
        mean_total = float(conditioned_data[0][13])
        wf_fds = []
        ind_last = 0
        for _ in quarters:
            wftd1 = np.array(conditioned_data[4][ind_last: ind_last + support]).astype(float)
            wf_fds.append(np.abs(np.fft.rfft(wftd1)))  # np.fft.ifftshift(wftd1)
            ind_last += support
        return np.array(wf_fds) * mean_total

    def generate_injection_parameters(
            self, n_injections=N_injections_total, add_to_existing_file=False,
            central_period=None, use_special_per_wing=False,
            per_wing_injection_relative=Per_wing_injection_and_slides_relative,
            injection_parameters_filename_postfix=Injection_parameters_filename_postfix,
            conditioned_data_filename_postfix=spip.Default_conditioned_data_filename_postfix,
            n_per=20,
            n_ecc=60,  # Probability to transit at a fixed eccentricity depends strongly on eccentricity
            n_w=20,
            n_cos_incl=1000,  # This divides the full circle  # should be at least few semi-major axes
            n_rp=30,
            u=None,
            snr_min_injected=5.,
            snr_max_1_injected=8,
            snr_max_full_injected=None,
            rp_generation=0.02):  # Planetary radius used to generate the flux (we neglect shape dependence on radius)
        if self.stellar_mass is None or self.stellar_radius is None or np.isnan(self.stellar_mass) or np.isnan(self.stellar_radius):
            self.failures += 'Unable to generate correct injections because of invalid stellar radius or mass;'

        if u is None:
            u = self.limb_darkening_coefs
        if u is None:
            u = Default_limb_darkening_coefs
        file_path = af.get_saving_file_path(output_location=self.output_location, target_id=self.target_id,
                                            filename_postfix=injection_parameters_filename_postfix, file_format='txt')
        if os.path.exists(file_path) and not add_to_existing_file:
            with open(file_path, 'w'):
                pass  # will delete the contents of the file

        if central_period is None:
            period = self.best_per
        else:
            period = central_period

        self.set_periodicity_search_parameters(force_overwrite=False)
        if not use_special_per_wing:
            if self.periods_in_chunks is None:
                if not hasattr(self, 'klc') or self.klc is None:
                    self.initialize_kepler_light_curve()
                periods_in_chunks = self.klc.make_periods_chunks(p_min=self.p_min, p_max=self.p_max, duty_cycle=self.duty_cycle,
                                                                 per_phases_per_chunk=self.per_phases_per_chunk,
                                                                 dt=self.klc.dt_days, t_mission=self.klc.T_mission)
            else:
                periods_in_chunks = self.periods_in_chunks
            closest_chunk, closest_chunk_index = ut.find_closest_chunk(periods_in_chunks, period, overlap_bins=0)
            p_min_inj, p_max_inj = closest_chunk[0], closest_chunk[-1]
        else:
            p_max_inj = period * (1 + per_wing_injection_relative)
            p_min_inj = period * (1 - per_wing_injection_relative)

        # ################################################################################################
        # OCCURRENCE RATES
        # All the things with the occurrence rates and densities:
        log2_rp_re_edges, log10_period_edges, occurrences_table = get_occurrence_table()  # Per one star
        units_of_re_to_units_of_r_star_factor = 1 / (self.stellar_radius * R_sun_R_e)
        log10_best_per_low = np.log10(p_min_inj)
        log10_best_per_up = np.log10(p_max_inj)
        # self.log10_best_period_edges = (log10_best_per_low, log10_best_per_up)
        log2_rp_edges = np.log2(units_of_re_to_units_of_r_star_factor * 2. ** log2_rp_re_edges)

        # FINDING ROWS IN THE OCCURRENCE TABLE THAT SPAN OUR PERIOD RANGE:
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
        # CALCULATING EXPECTED NUMBER OF PLANETS IN OUR PERIOD RANGE:
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

        self.prior_planets_in_period_range_one_star = np.sum(planets_of_radii_in_per_range_one_star)
        self.prior_radii_distribution = planets_of_radii_in_per_range_one_star / np.sum(planets_of_radii_in_per_range_one_star)
        self.injections_period_range = (10. ** log10_best_per_low, 10. ** log10_best_per_up)

        rp_numbers = (np.ceil(planets_of_radii_in_per_range_one_star /
                              np.sum(planets_of_radii_in_per_range_one_star) * n_rp)).astype(int)
        rp_numbers_total = np.sum(rp_numbers)  # how many of each radius to generate

        t = tm.time(t_tot=(256 + 1) * tm.dt_lc, parity='odd')[1:]  # Making an even points number but odd parity

        # ################################################################################################
        # GENERATING PLANET PARAMETERS, SEE WHO TRANSITS AND WHO HAS REASONABLE RADII
        wf_fds = self.load_rel_wf_fds(conditioned_data_filename_postfix=conditioned_data_filename_postfix)
        # -> relative wf_fd, mean flux already taken into account
        n_quarters = len(wf_fds)
        wf_fd = np.mean(np.array(wf_fds) ** 2, axis=0) ** 0.5

        # Calculating the geometric probability to transit.
        # It is sort of Monte-Carlo integration where every random point comes with its measure.
        # The total sum of the points with the measure will give the part of the parameter space
        # in which the planet transits (the sphere surface part and the eccentricity range).
        ecc_measure = 1. / n_ecc
        w_measure = 1. / n_w
        cos_i_measure = 1. / n_cos_incl
        transit_prob_measure = ecc_measure * w_measure * cos_i_measure

        injection_parameter_sets = []
        count_orbit_simulations = 0
        count_orbit_transiting = 0
        count_orbit_transiting_relevant_radii = 0
        for period in np.random.uniform(p_min_inj, p_max_inj, n_per):
            # Semi major axis of the orbit in units of R_star
            a = (G / (4 * np.pi ** 2) * (period * 24 * 60 * 60) ** 2) ** (1 / 3) * self.m_1_3_r_s * (M_sun ** (1 / 3) / R_sun)
            e_max = min(1 - 2 / a, 0.97)
            assert e_max > 0, 'Semi-major axis is too small or stellar radius is too large!'
            # The eccentricity distribution is normalized to unity in the allowed range.
            # There can be no planets that clash the star, so it is uniform in the allowed range.
            e_range = ut.get_uniform_random_sequence(0, e_max, n_ecc)
            e_range = e_range[e_range <= e_max]
            for ecc in e_range:
                w_range = ut.get_uniform_random_sequence(0, 360, n_w)
                for w in w_range:
                    cos_i_max = 1 / a * (1 + ecc * np.cos((w - 90) / 180 * np.pi)) / (1 - ecc ** 2)  # ) * 180 / np.pi
                    cos_i_range = ut.get_uniform_random_sequence(0, 1, n_cos_incl)
                    cos_i_range = cos_i_range[cos_i_range <= cos_i_max]
                    incls = np.arccos(cos_i_range) * 180 / np.pi
                    for inc in incls:
                        flux_obs = 1. - tm.flux_par(t=t, params=None, per=period, rp=rp_generation, a=a, inc=inc, ecc=ecc, w=w, u1=u[0],
                                                    u2=u[1], t0=0, supersample_factor=20, exp_time=tm.dt_lc)
                        dur = tm.templ_dur_hmax_fun(flux_obs) * tm.dt_lc
                        count_orbit_simulations += 1
                        if np.linalg.norm(flux_obs) > 0:
                            count_orbit_transiting += 1
                            depth_factor = np.max(flux_obs) / rp_generation ** 2
                            rp_rs_min, rp_rs_max = self.get_detectable_radius_flux(flux_model=flux_obs, period=period,
                                                                                   depth_factor=depth_factor, snr_min=snr_min_injected,
                                                                                   n_quarters=n_quarters, wf_fd=wf_fd,
                                                                                   snr_max_1=snr_max_1_injected,
                                                                                   snr_max_full=snr_max_full_injected,
                                                                                   stellar_radius=self.stellar_radius, units='star',
                                                                                   include_detrending=True, nperseg_psd=spip.Nperseg_psd)
                            rps = []
                            for i in range(len(log2_rp_edges) - 1):
                                log2_rp_range = (log2_rp_edges[i] + (log2_rp_edges[i + 1] - log2_rp_edges[i])
                                                 * ut.get_uniform_random_sequence(0, 1, rp_numbers[i]))
                                rps = rps + list(2. ** log2_rp_range)
                            np.random.shuffle(rps)
                            rps = np.array(rps)
                            rps = rps[rps >= rp_rs_min]
                            rps = rps[rps <= rp_rs_max]
                            count_orbit_transiting_relevant_radii += len(rps)  # will be divided by rp_numbers_total
                            for rp in rps:
                                injection_parameter_sets.append([period, a, inc, ecc, w, u[0], u[1], rp, dur])
        # Probability to transit given a planet of a given period:
        self.transit_probability = count_orbit_transiting / n_per * transit_prob_measure
        # Number of planets per star in the considered period range with any radius:
        self.transiting_planets_of_radii_in_per_range_one_star = self.prior_planets_in_period_range_one_star * self.transit_probability
        # Probability to transit and be in the relevant snr range given a planet of a given period:
        self.transit_and_range_probability = count_orbit_transiting_relevant_radii / rp_numbers_total / n_per * transit_prob_measure
        # Number of planets per star in the considered period range with radii that provide relevant SNR:
        self.transiting_planets_in_relevant_range = self.prior_planets_in_period_range_one_star * self.transit_and_range_probability
        selection_indices = np.random.choice(np.arange(len(injection_parameter_sets)), n_injections,
                                             replace=True if len(injection_parameter_sets) < n_injections else False)
        self.injection_par_list = list(np.array(injection_parameter_sets, dtype=object)[selection_indices])

        self.save_injection_parameters()
        self.pickle_small_attributes()

    def save_injection_parameters(self, injection_parameters_filename_postfix=Injection_parameters_filename_postfix):
        """
        Saves the generated injection parameters to the specified file to then load them and use the injection-recovery.
        :param injection_parameters_filename_postfix: Text added to the target ID in the name of the file saving
            the injection parameters. See naming notations.
        :type injection_parameters_filename_postfix: str
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
        """
        Removes the file with the generated injection parameters.
        :param injection_parameters_filename_postfix: Text added to the target ID in the name of the file saving
            the injection parameters. See naming notations.
        :type injection_parameters_filename_postfix: str
        """
        file_path = af.get_saving_file_path(output_location=self.output_location, target_id=self.target_id,
                                            filename_postfix=injection_parameters_filename_postfix, file_format='txt')
        if os.path.exists(file_path):
            os.remove(file_path)

    def load_injection_parameters(self, injection_parameters_index=0,
                                  injection_parameters_filename_postfix=Injection_parameters_filename_postfix):
        """
        Loads a specific line of the list of the injection parameters previously generated with
        `self.generate_injection_parameters()` and saved with `self.save_injection_parameters()`.
        :param injection_parameters_index: Index of the line from the list to load
        :type injection_parameters_index: int
        :param injection_parameters_filename_postfix: Text added to the target ID in the name of the file saving
            the injection parameters. See naming notations.
        :type injection_parameters_filename_postfix: str
        """
        file_path = af.get_saving_file_path(output_location=self.output_location, target_id=self.target_id,
                                            filename_postfix=injection_parameters_filename_postfix, file_format='txt')

        try:
            with open(file_path, 'r') as fil:
                lines = fil.readlines()
        except Exception as exc:
            report_exception(exception=exc, file_to_append=self.problems_file_location,
                             title='loading injection_parameters file', info='')

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

    def run_injection_search(self, injection_load_parameters_index=0,
                             injections_filename_postfix=Injections_filename_postfix,
                             one_chunk_around_injection=True,
                             per_wing_detection=Per_wing_injection_detection,
                             overlap_period_bins=spip.P_bins_overlap):
        """
        Loads a specific line of the list of the injection parameters previously generated with
        `self.generate_injection_parameters()` and saved with `self.save_injection_parameters()`.
        Masks the trigger found in the main search, as well as possible KOIs if there are any for this target.
        Injects the corresponding planet of the specified parameterts to data and runs the entire search from scratch.
        In the periodicity search, uses the period grid piece around the trigger.
        The goal is just to check whether the phase is recovered and what is the SNR. The look-elsewhere effect
        etc are not relevant.
        :param injections_filename_postfix: Text added to the target ID in the name of the file saving
            the injection-recovery results. See naming notations.
        :type injections_filename_postfix: str
        :param injection_load_parameters_index: Index of the line from the list to load
        :type injection_load_parameters_index: int
        :param one_chunk_around_injection: In the periodicity search, take only one period chunk from the
            main search around the central period used for the injection.
        :type one_chunk_around_injection: bool
        :param per_wing_detection: Days to the left and the same to the right in search
        :type per_wing_detection: float
        :param overlap_period_bins: when creating period chunks, extend each chunk from the two sides by
            this number of bins for them to overlap. is needed for coherent score integral in order for
            the maximum not to be on the edge
        :type overlap_period_bins: int
        """

        per_inj, a_inj, inc_inj, ecc_inj, w_inj, u1_inj, u2_inj, rp_inj, dur_inj = self.load_injection_parameters(
            injection_parameters_index=injection_load_parameters_index)
        # This period range is for injection, not for search:

        phase_inj = np.random.uniform(0, per_inj)

        tms, flxs = af.get_data_fun(target_id=self.target_id, location=self.data_location)
        first_transit_time_inj = np.nanmin(tms[0]) + phase_inj
        flxs_inj = sim.inject_planet_to_data(times=tms, fluxes=flxs, period=per_inj, first_transit_center=first_transit_time_inj,
                                             rp=rp_inj, a=a_inj, inc=inc_inj, ecc=ecc_inj, w=w_inj, u1=u1_inj, u2=u2_inj,
                                             supersample_factor=20, return_template=False)

        if one_chunk_around_injection:
            if self.periods_in_chunks is None:
                periods_in_chunks = self.klc.make_periods_chunks(p_min=self.p_min, p_max=self.p_max, duty_cycle=self.duty_cycle,
                                                                 per_phases_per_chunk=self.per_phases_per_chunk,
                                                                 dt=self.klc.dt_days, t_mission=self.klc.T_mission)
            else:
                periods_in_chunks = self.periods_in_chunks
            closest_chunk, closest_chunk_index = ut.find_closest_chunk(periods_in_chunks, per_inj, overlap_bins=0)
            periods_to_search = closest_chunk
        else:
            p_max = per_inj + per_wing_detection
            p_min = per_inj - per_wing_detection
            periods_to_search = self.klc.make_periods_array(p_min=p_min, p_max=p_max, dp=None, duty_cycle=spip.Duty_cycle, t_mission=None)

        self.klc = spip.KeplerLightCurve(target_id=self.target_id, times=tms, fluxes=flxs_inj, get_data=False,
                                         data_location=self.data_location, m_1_3_r_s=self.m_1_3_r_s)
        if self.is_koi:  # Masking koi planets
            self.mask_koi_planet(modify_self_klc=True, return_durations_transit_times=False, include_candidates=False)
        masked_dur = 2. * tm.Durations[self.best_temp]
        self.klc.mask_ephemeris(period=self.best_per, durations=masked_dur, transit_times=self.best_ftt)
        # Mask the suspected real transit. Warning, can kill some of injected signal
        self.klc.condition_data()
        self.klc.condition_templates()
        self.klc.detect_single_transits(apply_veto=True, gaussianize=True)
        self.klc.connect_quarters()
        self.klc.delete_irrelevant()  # For memory
        self.klc.interleave_shifts(delete_fulls=True)
        try:
            self.klc.periodicity_search(p_min=None, p_max=None, periods=periods_to_search,
                                        force_interleave_shifts=False, delete_periodic_interleaved=True,
                                        save_uncorrected_periodic=False, delete_non_marginalized=False,
                                        overlap_period_bins=overlap_period_bins)
            self.klc.get_best_trigger(apply_veto_mes=True, in_gaussianized=False)
            self.klc.get_best_trigger(apply_veto_mes=True, in_gaussianized=True)
            self.klc.make_best_periodic_scores()
        except Exception as exc:
            report_exception(exception=exc, file_to_append=self.problems_file_location,
                             title='injections, index='+str(injection_load_parameters_index), info='phase_inj='+str(phase_inj))
        self.klc.delete_mes()
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
            with Locker(file_path):
                with open(file_path, 'a') as fil1:
                    print(per_inj, a_inj, inc_inj, ecc_inj, w_inj, u1_inj, u2_inj, rp_inj, dur_inj, first_transit_time_inj,
                          retrieved_stat, retrieved_period, retrieved_first_transit_time, retrieved_gauss_stat, retrieved_integral_score,
                          retrieved_g_stat, retrieved_g_period, retrieved_g_first_transit_time, retrieved_g_gauss_stat,
                          retrieved_g_integral_score,
                          sep=' ', end='\n', file=fil1)
        except Exception as exc:
            report_exception(exception=exc, file_to_append=self.problems_file_location,
                             title='printing to injections file', info='')

    def retrieve_injections(self, max_phase_mismatch_to_dur=Max_phase_mismatch_to_dur, file_path=None,
                            injections_filename_postfix=Injections_filename_postfix):
        """
        Open the file in which the injection-recovery results are saved and save the statistics of these results
        as attributes in the class.
        :param max_phase_mismatch_to_dur: Maximal phase (first transit time) mismatch between the injected ephemeris
             and the recovered best trigger for it to be considered recovered. Defined with respect to the duration
             of the injected transit.
        :type max_phase_mismatch_to_dur: float
        :param file_path: Local path to the file where the injection-recovery info is stored. Should be a file
            created by `run_injection_search`.
        :type file_path: str
        :param injections_filename_postfix: Text added to the target ID in the name of the file saving
            the injection-recovery results. See naming notations.
        :type injections_filename_postfix: str
        """

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
                per_inj, a_inj, inc_inj, ecc_inj, w_inj, u1_inj, u2_inj, rp_inj, dur_inj, ftt_inj, \
                    retrieved_stat, retrieved_period, retrieved_first_transit_time, retrieved_gauss_stat, retrieved_integral_score, \
                    retrieved_g_stat, retrieved_g_period, retrieved_g_first_transit_time, retrieved_g_gauss_stat, \
                    retrieved_g_integral_score = inj_line.split()
                per_inj = float(per_inj)
                a_inj = float(a_inj)
                inc_inj = float(inc_inj)
                ecc_inj = float(ecc_inj)
                w_inj = float(w_inj)
                rp_inj = float(rp_inj)
                dur_inj = float(dur_inj)
                ftt_inj = float(ftt_inj)  # first transit time
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

                max_phase_mismatch = dur_inj * max_phase_mismatch_to_dur
                if min(np.abs(ftt_inj - retrieved_first_transit_time) % per_inj,
                       (per_inj - np.abs(ftt_inj - retrieved_first_transit_time)) % per_inj) > max_phase_mismatch:
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
                self.__dict__['first_transit_time_inj_' + status].append(ftt_inj)
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

    def retrieve_background(self, slides_res_filename_postfix=Slides_res_filename_postfix):
        """
        Opens the file where the time-shuffling runs results were saved, retrieves these data
        and stores as sttributes.
        :param slides_res_filename_postfix: Text added to the target ID in the name of the file saving
            the time-shuffled search results. See naming notations.
        :type slides_res_filename_postfix: str
        """
        fil_slides = [ff for ff in os.listdir(self.output_location) if
                      ((str(self.target_id) in ff) and (slides_res_filename_postfix in ff))][0]
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

    def _get_bin_min_bin_max(self, in_gaussianized=False):
        """
        Gets the boundaties for the histograms of the time shuffling and the injection-recovery runs
        from the obtained statistics.
        :param in_gaussianized: Whether to look at the gaussianized statistics or not.
        :type in_gaussianized: bool
        :return:
        :rtype:
        """
        if in_gaussianized:
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

    def make_histogram_bins(self, bin_width=Histogram_stats_binwid, bin_min=None, bin_max=None):
        """
        Create the bins array over which the distributions of the background and the injection-recovery
        statistics (snr squared) will be plotted.
        :param bin_width: The width of the bin in snr squared that should be used
        :type bin_width: float
        :param bin_min: Minimal bin value. if None, will be measured
        :type bin_min: float
        :param bin_max: Maximal bin value. if None, will be measured
        :type bin_max: float
        """
        if self.stats_slides is None:
            self.retrieve_background()
        if self.retrieved_stat_success is None:
            self.retrieve_injections()
        if bin_min is None or bin_max is None:
            bin_min, bin_max = self._get_bin_min_bin_max()
        self.bins_stat_histogram = np.arange(bin_min, bin_max, bin_width)
        bin_min, bin_max = self._get_bin_min_bin_max(in_gaussianized=True)
        self.bins_gauss_stat_histogram = np.arange(bin_min, bin_max, bin_width)

    def make_background_histogram(self, bins=None, bin_min=None, bin_max=None, bin_width=Histogram_stats_binwid):
        """
        Create a histogram of the background (from time-shuffled runs) distribution of the statistic scores.
        Saves bins and counts as attributes.
        :param bins: Bin edges to use in the histogram. If None, will create from `bin_min', `bin_max', `bin_width`.
        :type bins: None or numpy.ndarray of float
        :param bin_min: Minimal snr squared in the bins array. if None, will be determined from the distribution.
        :type bin_min: None or float
        :param bin_max: Maximal snr squared in the bins array. if None, will be determined from the distribution.
        :type bin_max: None or float
        :param bin_width: The width of the bin in snr squared that should be used
        :type bin_width: float
        """
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
                self.bins_stat_histogram_bg = np.arange(bin_min, bin_max, bin_width)
            if self.bins_gauss_stat_histogram is not None:
                self.bins_gauss_stat_histogram_bg = self.bins_gauss_stat_histogram
            else:
                bin_mi, bin_ma = self._get_bin_min_bin_max(in_gaussianized=True)
                if bin_min is None:
                    bin_min = bin_mi
                if bin_max is None:
                    bin_max = bin_ma
                self.bins_gauss_stat_histogram_bg = np.arange(bin_min, bin_max, bin_width)

        self.distr_stat_h0 = np.histogram(self.stats_slides, bins=self.bins_stat_histogram_bg)[0]
        self.distr_stat_h0 = self.distr_stat_h0 / self.n_slides
        if self.transiting_planets_in_relevant_range is not None:
            self.distr_stat_h0 = self.distr_stat_h0 * (1 - self.transiting_planets_in_relevant_range)  # the rate of not having a planet
        self.distr_gauss_stat_h0 = np.histogram(self.gauss_stats_slides, bins=self.bins_gauss_stat_histogram_bg)[0]
        self.distr_gauss_stat_h0 = self.distr_gauss_stat_h0 / self.n_slides
        if self.transiting_planets_in_relevant_range is not None:
            self.distr_gauss_stat_h0 = self.distr_gauss_stat_h0 * (1 - self.transiting_planets_in_relevant_range)
            # -> the rate of not having a planet

    def make_injections_histogram(self, bins=None, bin_min=None, bin_max=None):
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
                bin_mi, bin_ma = self._get_bin_min_bin_max(in_gaussianized=True)
                if bin_min is None:
                    bin_min = bin_mi
                if bin_max is None:
                    bin_max = bin_ma
                self.bins_gauss_stat_histogram_inj = np.arange(bin_min, bin_max, Histogram_stats_binwid)

        self.distr_stat_inj_all = np.histogram(np.array(self.retrieved_stat_bad + self.retrieved_stat_success),
                                               bins=self.bins_stat_histogram_inj, density=False)[0]
        self.distr_stat_inj_all = (self.distr_stat_inj_all / self.number_of_injections * self.transiting_planets_in_relevant_range)
        self.distr_stat_inj_success = np.histogram(np.array(self.retrieved_stat_success), bins=self.bins_stat_histogram_inj)[0]
        self.distr_stat_inj_success = (self.distr_stat_inj_success / self.number_of_injections * self.transiting_planets_in_relevant_range)

        # same for gaussianized
        self.distr_gauss_stat_inj_all = np.histogram(np.array(self.retrieved_gauss_stat_bad + self.retrieved_gauss_stat_success),
                                                     bins=self.bins_gauss_stat_histogram_inj)[0]
        self.distr_gauss_stat_inj_all = (self.distr_gauss_stat_inj_all / self.number_of_injections
                                         * self.transiting_planets_in_relevant_range)
        self.distr_gauss_stat_inj_success = np.histogram(np.array(self.retrieved_gauss_stat_success),
                                                         bins=self.bins_gauss_stat_histogram_inj)[0]
        self.distr_gauss_stat_inj_success = (self.distr_gauss_stat_inj_success / self.number_of_injections
                                             * self.transiting_planets_in_relevant_range)

    def calculate_significance(self, remake_histograms=True):
        # TODO!
        if remake_histograms or self.bins_stat_histogram is None:
            self.make_histogram_bins()
            self.make_injections_histogram(bins=self.bins_stat_histogram)
            self.make_background_histogram(bins=self.bins_stat_histogram)
        # MAKE EXTRAPOLATION
        # Determine intersection with horizontal axis in log scale:
        # self.extrapolation_coefs_stat_h0_hist = af.get_histogram_extrapolation_coefs_in_log(
        #     self.distr_stat_h0, self.bins_stat_histogram)  # attention: extrapolating only the direct bg
        # self.extrapolation_coefs_stat_h0_hist[1] is supposed to be negative because the trend is decreasing
        # TODO: SELECT STATISTIC!
        # self.prob_density_h0_trigger = np.exp(self.best_stat * self.extrapolation_coefs_stat_h0_hist[1]
        #                                       + self.extrapolation_coefs_stat_h0_hist[0])
        # # Follows from the integration of the non-log distribution over MES[SNR^2]
        # self.false_alarm_prob_trigger = (1 / np.abs(self.extrapolation_coefs_stat_h0_hist[1])
        #                                  * np.exp(self.extrapolation_coefs_stat_h0_hist[0] + self.extrapolation_coefs_stat_h0_hist[
        #                                                                                                     1] * self.best_stat))
        # bin_inds_close_trigger = np.sort(np.argsort(np.abs(self.bins_stat_histogram - self.best_stat))[:2])  # find edges of this bin
        # self.prob_density_h1_trigger = self.distr_stat_inj_all[bin_inds_close_trigger[0]]  # use the left edge index to access the bin
        # self.prob_density_planet_not_noise = self.prob_density_h1_trigger / (self.prob_density_h1_trigger + self.prob_density_h0_trigger)

    def analyze_injections_and_background(self, max_phase_mismatch_to_dur=Max_phase_mismatch_to_dur,
                                          bins=None, bin_min=None, bin_max=None, bin_width=Histogram_stats_binwid):
        """
        Retrieves the results of the time-shuffled and injection-recovery runs in the corresponding files,
        makes histograms of them, finds the distribution density and estimates the statistical significance
        of the trigger basing on that.
        :param max_phase_mismatch_to_dur: Maximal phase (first transit time) mismatch between the injected ephemeris
             and the recovered best trigger for it to be considered recovered. Defined with respect to the duration
             of the injected transit.
        :type max_phase_mismatch_to_dur: float
        :param bins: Bin edges to use in the histogram. If None, will create from `bin_min', `bin_max', `bin_width`.
        :type bins: None or numpy.ndarray of float
        :param bin_min: Minimal snr squared in the bins array. if None, will be determined from the distribution.
        :type bin_min: None or float
        :param bin_max: Maximal snr squared in the bins array. if None, will be determined from the distribution.
        :type bin_max: None or float
        :param bin_width: The width of the bin in snr squared that should be used
        :type bin_width: float
        :param bin_width: The width of the bin in snr squared that should be used
        :type bin_width: float
        """
        self.retrieve_background()
        self.retrieve_injections(max_phase_mismatch_to_dur=max_phase_mismatch_to_dur)
        if bins is None:
            self.make_histogram_bins(bin_min=bin_min, bin_max=bin_max, bin_width=bin_width)
        else:
            self.bins_stat_histogram = bins
            self.bins_gauss_stat_histogram = bins
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
        #     estimate_ampls_ppms_from_stats(summed_raw_stats, summed_stat_vars, templ_max, trends_stds=None)
        #
        self.rp_to_rs = (self.best_ppm_no_filt / 1e6) ** 0.5  # the radius estimation is subjected to big error and bias!
        self.rp_to_rs_std = (np.nanvar(np.abs(self.best_ppms_no_filt) ** 0.5)
                             + np.nanmean(self.best_ppms_stds_no_filt) / np.abs(self.best_ppm_no_filt)) ** 0.5 / 1e3
        self.rp_to_r_earth = self.rp_to_rs * self.stellar_radius * R_sun_R_e
        self.rp_to_r_earth_std = self.rp_to_rs_std * self.stellar_radius * R_sun_R_e
