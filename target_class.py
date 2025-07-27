#!/usr/bin/env python3

"""
How to read the documentation:
- consult the description
- if something is unclear, check the vocabulary
- check in many places

VOCABULARY:
KOI - Kepler Object of Interest, meaning something that is a potential planet according to Kepler
TTV - Transit timing variation
quarter - 90-day pieces to which Kepler light curve is split. Every quarter has different location on the detector, contamination, PSD.
SES -
Raw SES -
SES SNR -
MES -
Main search (run) -
Conditioning -
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


SEARCH STRATEGY DESCRIPTION

name postfix notation
periodicity search - chunks
Saving formats
"""

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
from template_bank.template_bank_generation_parameters import E_crit
from simulations import lightcurve_simulation as sim
import auxiliary_functions as af
import utils as ut
import time
from pipeline_parameters import *  # all the parameters, physical constants, occurrence table
from file_operations import report_exception, Locker, print_problem
import trigger_functions as tc
from catalog_functions import get_koi_ttv_dicts
np.set_printoptions(threshold=1000, linewidth=10000)

######################################################################################################################################


class KeplerTarget(object):
    star_attributes = Star_attributes

    def __init__(self, target_id='003097222', data_location=Data_location, output_location=Output_location,
                 load_pickled_attributes=False,
                 get_transit_and_binary_info=True,
                 gaia_kepler_file_path=Gaia_kepler_file,
                 koi_dict_file=Koi_dict_file,
                 ttv_dict_file=Ttv_dict_file,
                 villanova_dict_file=Villanova_dict_file,
                 kill_villanova_binary=True,
                 limb_darkening_array_file=Limb_darkening_array_file,
                 overwrite_location_parameters_by_unpickle=False,
                 allow_output_folder_creation=True,
                 allow_data_folder_creation=True,
                 problems_folder=Problems_folder,
                 add_prefix_to_problems_folder=True,
                 clean_problems_file=False,
                 simulated_white_star=False, simulated_red_star=False,
                 plots_location=None,
                 allow_plots_folder_creation=True):
        """
        Initializes the KeplerTarget object which is used to run the planet search, the statistical significance
        evaluation and some estimations for Kepler stars.
        :param target_id: Kepler ID of the target. A 9-digit number, where some first digits may be zero.
            May be specified as integer or as string, should only contain digits.
        :type target_id: str
        :param data_location: Local path to the folder where data files are stored. The target_id will
            be searched in there. If light curves are not found in the folder, they will be downloaded.
            If folder is not found, it will be created.
        :type data_location: str or LiteralString or bytes or None
        :param output_location: Directory where the output is to be saved. The target ID named folder will be created.
        :type output_location: str or LiteralString or bytes or None
        :param load_pickled_attributes: Whether to load the class attributes which were saved in the `output_location`
        :type load_pickled_attributes: bool
        :param get_transit_and_binary_info: Whether to get the target info about existing planets and binary
            companions from the Kepler catalogs.
        :type get_transit_and_binary_info: bool
        :param gaia_kepler_file_path: The local path to the Gaia-Kepler catalog file
        :type gaia_kepler_file_path: str or LiteralString or bytes or None
        :param koi_dict_file: The local path to the KOI info dictionary file
        :type koi_dict_file: str or LiteralString or bytes or None
        :param ttv_dict_file: The local path to the TTV info file
        :type ttv_dict_file: str or LiteralString or bytes or None
        :param villanova_dict_file: The local path to the Villanova Kepler eclipsing binary info file
        :type villanova_dict_file: str or LiteralString or bytes or None
        :param overwrite_location_parameters_by_unpickle: Whether to use the locations parameters saved in the
            output directory from the previous runs
        :type overwrite_location_parameters_by_unpickle: bool
        :param allow_output_folder_creation: Whether to allow creation of the new output target-ID-named folder if
            the existing one was not found
        :type allow_output_folder_creation: bool
        :param allow_data_folder_creation: Whether to allow creation of the new data target-ID-named folder if
            the existing one was not found
        :type allow_data_folder_creation: bool
        :param simulated_white_star: Whether to take the simulated white star data and not a real Kepler one
        :type simulated_white_star: bool
        :param simulated_red_star: Whether to generate the simulated red star data and not a real Kepler one
        :type simulated_red_star: bool
        :param plots_location:
        :type plots_location: str or LiteralString or bytes or None
        :param allow_plots_folder_creation:
        :type allow_plots_folder_creation: bool
        """
        self.target_id = target_id.rjust(9, '0')
        """Kepler target ID. Formatted to be a 9-digit string"""
        self.simulated_star = 'white' if simulated_white_star else 'red' if simulated_red_star else False
        """Whether the star is simulated and not taken from the real Kepler data"""

        # ############################################################################################################
        # Obtaining the locations of the input and the output:
        if data_location is not None:
            data_location = af.get_correct_tid_folder(target_id, data_location,
                                                      allow_creation=allow_data_folder_creation)
        if output_location is not None:
            output_location = af.get_correct_tid_folder(target_id, output_location,
                                                        allow_creation=allow_output_folder_creation)
        self.data_location = data_location
        """Where Kepler data is saved. If not there, it will be downloaded"""
        self.output_location = output_location
        """Where the output will be saved. A target_id named folder will be made"""
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
        if problems_folder is not None:
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
        """Stellar right ascension coordinate. Should be obtained from GAIA if exists"""
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
        self.is_too_short = None
        """The data is too short to search even for thr shortest period."""
        self.is_bad_fft = None
        """Whether the light curve is too bad to search for planets. The Fourier-domain data looks bad, 
           as checked by our functions. E.g. contains very large peaks"""
        self.snr_max_detectable_radius = None
        """The maximal score that can be detected for the planet of the maximal radius considered,
           see `check_detectable_radius` method"""
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
        self.n_quarters = None
        """Number of quarters in the Kepler data that we process"""

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
        self.best_candidates_concat = None
        """Saves the best trigger from every periodicity search chunk. See description in the code"""
        self.best_best_candidates_concat_index = None
        """Index of the best trigger chunk in `self.best_candidates_concat`"""
        self.best_g_best_candidates_concat_index = None
        """Index of the best gaussianized trigger chunk in `self.best_candidates_concat`"""
        self.log_baseline_int_score_rate = None
        """Value of integral score which is subtracted from all the integral scores. Subtraction removes the
           arbitrary units of rate and makes the scores relative. It is like having the statistic with 
           punishment for prior and parameter space measure. 
           The definition of the punishment is such that it vanishes at the m_1_3_r_s=1
           and the period chunk corresponding to the lowest period (30 days usually). 
           This rate is a product of `self.log_baseline_int_score_rate_density` 
           and `self.log_baseline_phase_space_measure`. 
           It also makes scores comparable between different stars with different m_1_3_r_s."""
        self.baseline_rate_period = None
        """Period used fo compute `self.log_baseline_int_score_rate`"""
        self.log_baseline_int_score_rate_density = None
        """Log density of occurrence per d_period and d_first_transit time. Occurrence is for planets of all radii 
           summed."""
        self.dp_0 = None
        """Period step in the reference chunk (smallest-period chunk with vanishing baseline density correction).
           Is used in `self.log_baseline_phase_space_measure` which is needed for `self.log_baseline_int_score_rate`
        """
        self.d_ftt_0 = None
        """Phase step in the reference chunk (smallest-period chunk with vanishing baseline density correction).
           Is used in `self.log_baseline_phase_space_measure` which is needed for `self.log_baseline_int_score_rate`
        """
        self.log_baseline_phase_space_measure = None
        """Is equal to np.log(`self.dp_0` * `self.d_ftt_0`), is used in `self.log_baseline_int_score_rate`. 
        """
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
        self.best_int_score = None
        """Integral coherent score corresponding to the best score in the search pipeline - see `search_pipeline.py`.
           Note that the `self.log_baseline_int_score_rate` was subtracted"""
        self.best_g_stat = None
        """Statistic corresponding to the best gaussianized score in the search pipeline - see `search_pipeline.py`"""
        self.best_g_per = None
        """Period in days corresponding to the best gaussianized score in the search pipeline - see `search_pipeline.py`"""
        self.best_g_temp = None
        """Template index corresponding to the best gaussianized score in the search pipeline - see `search_pipeline.py`"""
        self.best_g_ftt = None
        """First transit time corresponding to the best gaussianized score in the search pipeline - see `search_pipeline.py`"""
        self.best_g_gauss_stat = None
        """Best gaussianized score in the search pipeline - see `search_pipeline.py`"""
        self.best_g_int_score = None
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
        self.mean_flux = None
        """Mean flux of the light curve, all the quarters taken together"""

        # ############################################################################################################
        # After the best trigger:
        self.is_threshold_crossing = None
        """Whether the best trigger crossed the pre-defined detection threshold"""
        self.is_threshold_g_crossing = None
        """Whether the best_gaussian trigger crossed the pre-defined detection threshold"""
        self.triggers = None
        """If multiple triggers are saved, this will keep al the results that triggered in the
        non-gaussianized statistics"""
        self.g_triggers = None
        """If multiple triggers are saved, this will keep al the results that triggered in the
           gaussianized statistics"""
        # ############################################################################################################
        # Injection parameters:
        self.per_wing_injection_and_slides = None
        """Wing of the periods window that will be used to make injections and shuffle timeslides. 
           Is used when the range is specifically required."""
        self.slides_period_range = None
        self.injections_period_range = None
        """Minimal and maximal period of the range that will be injected."""
        self.transit_probability = None
        """Geometric probability to transit. Given that there is a planet, what is the probability that it transits"""
        self.prior_planets_in_period_range_one_star = None
        """Number of planets in the injection period range per one star"""
        self.transiting_planets_of_radii_in_per_range_one_star = None
        """Number of planets in the injection period range per one star that exist and transit"""
        self.transiting_planets_of_radii_in_per_range_one_star_poisson = None
        """Poisson probability to have al least one planet in the dataset if the poisson rate is
           `self.transiting_planets_of_radii_in_per_range_one_star`. It is needed because we are looking for one maximal trigger
           per star, therefore if there happens to be more than one planet, it does not change the detection.
           Technically, however, the rate is so small that this number and the rate itself are equal."""
        self.transit_and_range_probability = None
        """Probability to transit and to have a radius in the relevant range given that there is a planet of this period"""
        self.transiting_planets_in_relevant_range = None
        """Number of planets in the injection period range in the radii range relevant (snr-wise) for the trigger 
            per one star that exist and transit"""
        self.transiting_planets_in_relevant_range_poisson = None
        """Poisson probability to have al least one planet in the dataset if the poisson rate is
           `self.transiting_planets_in_relevant_range`. It is needed because we are looking for one maximal trigger
           per star, therefore if there happens to be more than one planet, it does not change the detection.
           Technically, however, the rate is so small that this number and the rate itself are equal."""
        self.injection_par_list = None
        """List of parameters that will be injected. List of [period, a, inc, ecc, w, u[0], u[1], rp, dur]"""
        self.prior_radii_distribution = None
        """Probability distribution of the planetary radii in this period range"""
        self.log2_rp_edges = None
        """Edges of bins for probability distribution of the planetary radii"""

        # ############################################################################################################
        # Injections attributes
        self.per_inj_success = None
        """List of injected periods of injections which were successfully detected"""
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
        self.dur_inj_success = None
        """Approximate duration of the injected transit of injections which were successfully detected"""
        self.retrieved_stat_success = None
        """List of detected statistics of injections which were successfully detected"""
        self.retrieved_period_success = None
        """List of detected periods in days of injections which were successfully detected"""
        self.retrieved_first_transit_time_success = None
        """List of detected first transit times of injections which were successfully detected"""
        self.retrieved_template_success = None
        """List of detected templates of injections which were successfully detected"""
        self.retrieved_gauss_stat_success = None
        """List of detected gaussianized statistics of injections which were successfully detected"""
        self.retrieved_int_score_success = None
        """List of detected integral scores of injections which were successfully detected"""
        self.retrieved_g_stat_success = None
        """List of detected (in the best gaussianized sense) of statistics of injections which were successfully detected"""
        self.retrieved_g_period_success = None
        """List of detected (in the best gaussianized sense) of periods of injections which were successfully detected"""
        self.retrieved_g_first_transit_time_success = None
        """List of detected (in the best gaussianized sense) of first transit times of injections which were successfully detected"""
        self.retrieved_g_gauss_stat_success = None
        """List of detected (in the best gaussianized sense) of gaussianized statistics of injections which were successfully detected"""
        self.retrieved_g_int_score_success = None
        """List of detected (in the best gaussianized sense) on integral coherent scores of injections which were successfully detected"""
        self.retrieved_g_template_success = None
        """List of detected (in the best gaussianized sense) templates of injections which were successfully detected"""

        self.per_inj_bad = None
        """List of injected periods of injections which were missed by the search"""
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
        self.dur_inj_bad = None
        """Approximate duration of the injected transit of injections which were missed by the search"""
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
        self.retrieved_int_score_bad = None
        """List of detected integral scores of injections which were missed by the search"""
        self.retrieved_template_bad = None
        """List of detected templates of injections which were missed by the search"""
        self.retrieved_g_stat_bad = None
        """List of detected (in the best gaussianized sense) of statistics of injections which were missed by the search"""
        self.retrieved_g_period_bad = None
        """List of detected (in the best gaussianized sense) of periods of injections which were missed by the search"""
        self.retrieved_g_first_transit_time_bad = None
        """List of detected (in the best gaussianized sense) of first transit times of injections which were missed by the search"""
        self.retrieved_g_gauss_stat_bad = None
        """List of detected (in the best gaussianized sense) of gaussianized statistics of injections which were missed by the search"""
        self.retrieved_g_int_score_bad = None
        """List of detected (in the best gaussianized sense) of integral scores of injections which were missed by the search"""
        self.retrieved_g_template_bad = None
        """List of detected (in the best gaussianized sense) templates of injections which were missed by the search"""

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
        """List of best integral coherent scores that were found in the time-scrambling run"""
        self.stats_g_slides = None
        """List of statistics (in the best-gaussianized sense) of that were found in the time-scrambling run"""
        self.pers_g_slides = None
        """List of periods (in the best-gaussianized sense) that were found in the time-scrambling run"""
        self.first_times_g_slides = None
        """List of first transit times (in the best-gaussianized sense) that were found in the time-scrambling run"""
        self.gauss_stats_g_slides = None
        """List of gaussianized statistics (in the best-gaussianized sense) that were found in the time-scrambling run"""
        self.int_scores_g_slides = None
        """List of integral coherent scores (in the best-gaussianized sense) that were found in the time-scrambling run"""
        self.n_slides = None
        """Total number of time-scramblings made"""
        self.n_injections = None
        """Total number of injections made"""
        self.n_success_injections = None

        # Time-scrambling clean attributes
        self.stats_slides_clean = None
        """List of best statistics that were found in the time-scrambling run"""
        self.pers_slides_clean = None
        """List of best periods in days that were found in the time-scrambling run"""
        self.first_times_slides_clean = None
        """List of best first transit times that were found in the time-scrambling run"""
        self.gauss_stats_slides_clean = None
        """List of best gaussianized statistics that were found in the time-scrambling run"""
        self.int_scores_slides_clean = None
        """List of best integral coherent scores that were found in the time-scrambling run"""
        self.stats_g_slides_clean = None
        """List of statistics (in the best-gaussianized sense) of that were found in the time-scrambling run"""
        self.pers_g_slides_clean = None
        """List of periods (in the best-gaussianized sense) that were found in the time-scrambling run"""
        self.first_times_g_slides_clean = None
        """List of first transit times (in the best-gaussianized sense) that were found in the time-scrambling run"""
        self.gauss_stats_g_slides_clean = None
        """List of gaussianized statistics (in the best-gaussianized sense) that were found in the time-scrambling run"""
        self.int_scores_g_slides_clean = None
        """List of integral coherent scores (in the best-gaussianized sense) that were found in the time-scrambling run"""
        self.n_slides_clean = None
        """Total number of time-scramblings made"""

        # Significance evaluation attributes:
        # Integral:
        self.bins_int_score_histogram = None
        """Bins used to build the distribution of the integral score of the time-scrambling and injections"""
        self.bins_int_score_histogram_inj = None
        """Bins used to build the distribution of the integral score of the injections"""
        self.bins_int_score_histogram_bg = None
        """Bins used to build the distribution of the integral score of the time-scrambling"""
        self.bins_int_score_histogram_bg_clean = None
        """Bins used to build the distribution of the integral score of the time-scrambling with removed trigger"""
        self.distr_int_score_h0 = None
        """Histogram of distribution of the integral score of the time-scrambling"""
        self.distr_int_score_h0_clean = None
        """Histogram of distribution of the integral score of the time-scrambling with removed trigger"""
        self.distr_int_score_inj_all = None
        """Histogram of distribution of the integral score of the injections (successful and missed together)"""
        self.distr_int_score_inj_success = None
        """Histogram of distribution of the integral score of the successful injections"""
        self.prob_density_h0_trigger = None
        """Probability density (see units in description) of the background at the location of the trigger (integral score)"""
        self.prob_density_h0_clean_trigger = None
        """Probability density (see units in description) of the background with removed trigger at the location 
           of the trigger (integral score)"""
        self.false_alarm_prob_trigger = None
        """False-alarm probability of the trigger estimated from the time-scrambling (integral score)"""
        self.prob_density_h1_trigger = None
        """Probability density (see units in description) of the expected planets at the location of the trigger (integral score)"""
        self.prob_density_planet_not_noise = None
        """Probability of the trigger to originate from the planet and not from the noise (integral score)"""
        self.extrapolation_coefs_int_score_h0_hist = None
        """Pair of coefficients describing the linear extrapolation of the background distribution of the integral score
           up to the location of the trigger"""
        self.extrapolation_coefs_int_score_h0_clean_hist = None
        """Pair of coefficients describing the linear extrapolation of the background distribution with the removed trigger
           of the integral score up to the location of the trigger"""

        # Statistic
        self.bins_stat_histogram = None
        """Bins used to build the distribution of the statistics of the time-scrambling and injections"""
        self.bins_stat_histogram_inj = None
        """Bins used to build the distribution of the statistics of the injections"""
        self.bins_stat_histogram_bg = None
        """Bins used to build the distribution of the statistics of the time-scrambling"""
        self.bins_stat_histogram_bg_clean = None
        """Bins used to build the distribution of the statistics of the time-scrambling with removed trigger"""
        self.distr_stat_h0 = None
        """Histogram of distribution of the statistics of the time-scrambling"""
        self.distr_stat_h0_clean = None
        """Histogram of distribution of the statistics of the time-scrambling with removed trigger"""
        self.distr_stat_inj_all = None
        """Histogram of distribution of the statistics of the injections (successful and missed together)"""
        self.distr_stat_inj_success = None
        """Histogram of distribution of the statistics of the successful injections"""
        self.background_normalization_coef = None
        """What the integral of the background distribution is equal to. Takes care of scores too low to be included
           abd the prior background probability"""
        self.extrapolation_coefs_stat_h0_hist = None
        """Pair of coefficients describing the linear extrapolation of the background distribution of the statistic
           up to the location of the trigger"""
        self.extrapolation_coefs_stat_h0_hist_clean = None
        """Pair of coefficients describing the linear extrapolation of the background distribution with the removed trigger
           of the statistic up to the location of the trigger"""
        self.extrapolation_coef_int_score_h0_hist = None
        """Coefficient describing the chi2(1) distribution of the extrapolation of the background distribution of the integral score"""
        self.extrapolation_limit_int_score_h0_hist = None
        """Minimal score used in the extrapolation of the background distribution of the integral score"""
        self.extrapolation_amplitude_int_score_h0_hist = None
        """Normalization which should multiply the scaled chi2(1) distribution of the extrapolation of the background"""
        self.extrapolation_amplitude_int_score_clean_h0_hist = None
        self.extrapolation_coef_int_score_clean_h0_hist = None
        self.extrapolation_limit_int_score_clean_h0_hist = None
        # Gaussianized statistic (still for the location of the best non-gaussianized score):
        self.bins_gauss_stat_histogram = None
        """Bins of the gaussianized statistic used to build the distribution of the statistics of the time-scrambling and injections"""
        self.bins_gauss_stat_histogram_inj = None
        """Bins of the gaussianized statistic used to build the distribution of the statistics of the injections"""
        self.bins_gauss_stat_histogram_bg = None
        """Bins of the gaussianized statistic used to build the distribution of the statistics of the time-scrambling"""
        self.bins_gauss_stat_histogram_bg_clean = None
        """Bins of the gaussianized statistic used to build the distribution of the statistics of the time-scrambling 
           with removed trigger"""
        self.distr_gauss_stat_h0 = None
        """Histogram of distribution of the gaussianized statistics of the time-scrambling"""
        self.distr_gauss_stat_h0_clean = None
        """Histogram of distribution of the gaussianized statistics of the time-scrambling with removed trigger"""
        self.distr_gauss_stat_inj_all = None
        """Histogram of distribution of the gaussianized statistics of the all (success and bad) injections"""
        self.distr_gauss_stat_inj_success = None
        """Histogram of distribution of the gaussianized statistics of the successful injections"""

        # ############################################################################################################
        self.failures = ''  # if something bad happens, we will write it here
        """Records problems in data or processing which appeared during the run"""
        self.is_good_for_processing = True
        self.ses_search_failed = False

        if load_pickled_attributes:
            self.unpickle_small_attributes(overwrite_location_parameters=overwrite_location_parameters_by_unpickle)
        else:
            if self.simulated_star is False:
                if self.gaia_kepler_file is not None:
                    self.get_fields_from_gaia(gaia_file_path=self.gaia_kepler_file)
                if get_transit_and_binary_info:
                    self.get_koi_and_ttv_fields(koi_dict_file=self.koi_dict_file, ttv_dict_file=self.ttv_dict_file)
                    self.get_villanova_fields(villanova_dict_file=self.villanova_dict_file)
            else:
                self.is_good_for_processing = True
                self.stellar_radius = 1.
                self.stellar_mass = 1.
                self.gaia_nconfp = 0.
                self.m_1_3_r_s = 1.
                self.limb_darkening_coefs = Default_limb_darkening_coefs
                self.is_koi = False
                self.is_villanova_binary = False

        if kill_villanova_binary:
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
        dict_of_old_locations = {}
        for key in Location_attributes:
            dict_of_old_locations[key] = self.__dict__[key]

        # old_output_location = self.output_location
        # old_data_location = self.data_location
        # old_gaia_kepler_file_path = self.gaia_kepler_file
        # old_koi_dict_file = self.koi_dict_file
        # old_ttv_dict_file = self.ttv_dict_file
        # old_villanova_dict_file = self.villanova_dict_file
        # old_limb_darkening_array_file = self.limb_darkening_array_file

        if os.path.exists(file_path):
            with open(file_path, 'rb') as pickle_file:
                dict1 = pickle.load(pickle_file)
                self.__dict__.update(dict1.items())

            if not overwrite_location_parameters:
                for key in Location_attributes:
                    self.__dict__[key] = dict_of_old_locations[key]
                # self.data_location = old_data_location
                # self.output_location = old_output_location
                # self.gaia_kepler_file = old_gaia_kepler_file_path
                # self.koi_dict_file = old_koi_dict_file
                # self.ttv_dict_file = old_ttv_dict_file
                # self.villanova_dict_file = old_villanova_dict_file
                # self.limb_darkening_array_file = old_limb_darkening_array_file

    def get_fields_from_gaia(self, gaia_file_path=None):
        """
        Opens the GAIA-Kepler cross-match info file and gets some target parameters
        :param gaia_file_path: Local path to the file
        :type gaia_file_path: str
        """
        if gaia_file_path is None:
            gaia_file_path = self.gaia_kepler_file
        gaia_file = fits.open(gaia_file_path)
        indices = np.where(gaia_file[1].data['kepid'] == int(self.target_id))[0]
        if len(indices) == 0:
            self.failures += ' Target_id not in Gaia; '
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
                self.failures += ' Parallax value negative; '
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
                self.failures += 'Unable to generate correct injections because of invalid stellar radius or mass; '
                m_1_3_r_s = 1.
            else:
                m_1_3_r_s = self.stellar_mass ** (1 / 3) / self.stellar_radius
            self.m_1_3_r_s = m_1_3_r_s  # M**(1/3)*R[in M_sun and R_sun]
            self.limb_darkening_coefs = tm.get_limb_darkening_coefs(teff=self.teff, logg=self.logg, metallicity=self.metallicity,
                                                                    limb_darkening_array_file=self.limb_darkening_array_file)

            for key in self.star_attributes:
                try:
                    if self.__dict__[key] is not None:
                        self.__dict__[key] = float(self.__dict__[key])
                except Exception as e:
                    report_exception(exception=e, file_to_append=self.problems_file_location, title='', info='')
                    self.failures += key + ' error ' + str(e) + '; '
                    self.__dict__[key] = None
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

        self.is_koi, self.koi_dict, self.ttv_dict = get_koi_ttv_dicts(kepler_id=self.target_id,
                                                                      koi_dict_file=koi_dict_file, ttv_dict_file=ttv_dict_file)
        self.is_koi_planet_masked = False
        # self.is_koi = False
        # with open(koi_dict_file, 'rb') as koi_catalog_dict_file:
        #     koi_info_dict = pickle.load(koi_catalog_dict_file)
        # if self.target_id in koi_info_dict['kepid']:
        #     self.is_koi = True
        #     self.is_koi_planet_masked = False
        #     self.koi_dict = {}
        #     # ind = koi_info_dict['kepid'].index(self.target_id)
        #     inds = np.where(np.array(koi_info_dict['kepid']) == self.target_id)[0]
        #     for ind in inds:
        #         for key in koi_info_dict:
        #             if key != 'kepid':
        #                 if key not in self.koi_dict:
        #                     self.koi_dict[key] = []
        #                 self.koi_dict[key].append(koi_info_dict[key][ind])
        #     with open(ttv_dict_file, 'rb') as ttv_catalog_dict_file:
        #         ttv_info_dict = pickle.load(ttv_catalog_dict_file)
        #     self.ttv_dict = {}
        #     for koi_name in self.koi_dict['kepoi_name']:
        #         koi_inds = np.where((np.array(ttv_info_dict['KOI']) == str(float(koi_name[1:])))
        #                             & (np.array(ttv_info_dict['kepid']) == self.target_id))[0]
        #         if len(koi_inds) > 0:
        #             for key in ttv_info_dict:
        #                 if key != 'kepid':
        #                     if key not in self.ttv_dict:
        #                         self.ttv_dict[key] = []
        #                     self.ttv_dict[key].append(np.array(ttv_info_dict[key])[koi_inds])
        #     # Units of Kepler times : unit = 'BJD - 2454833'
        #     # Units in the TTV catalog :  BJD-2454900
        #     if 'T' in self.ttv_dict.keys():  # it may be not there if the object is koi but not ttv
        #         for j in range(len(self.ttv_dict['T'])):
        #             self.ttv_dict['T'][j] = self.ttv_dict['T'][j].astype(float) + 2454900 - 2454833

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
        fields_not_to_print = ['gaia_kepler_file',
                               'koi_dict_file',
                               'ttv_dict_file',
                               'villanova_dict_file',
                               'limb_darkening_array_file']
        if output_location is None:
            output_location = self.output_location
        dict_to_print = {}
        for key in self.__dict__:
            ok_to_print = True
            if key not in fields_not_to_print:
                value = self.__dict__[key]
                if isinstance(value, (str, float, int, bool, np.bool_, np.float32, np.float64, np.complex64, np.complex128,
                                      np.integer, np.ndarray, list, dict, tuple)):
                    # -> float is np.float64 but not 32!
                    if hasattr(value, '__len__') and len(value) > 0:
                        if isinstance(value, dict) and hasattr(value[list(value.keys())[0]], '__len__'):
                            ok_to_print = False
                        elif ((not isinstance(value, dict)) and (not isinstance(value, str))
                              and hasattr(value[0], '__len__')):
                            ok_to_print = False  # not printing nested lists
                        elif (not isinstance(value, str)) and len(value) > 50:
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

    # noinspection PyTypeChecker
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

                print('best_int_score', ':', self.best_int_score, file=fil1)
                print('best_stat', ':', self.best_stat, file=fil1)
                print('best_per', ':', self.best_per, file=fil1)
                print('best_ftt', ':', self.best_ftt, file=fil1)
                print('best_gauss_stat', ':', self.best_gauss_stat, file=fil1)
                print('best_temp', ':', self.best_temp, file=fil1)

                print('', file=fil1)
                print('best_g_int_score', ':', self.best_g_int_score, file=fil1)
                print('best_g_stat', ':', self.best_g_stat, file=fil1)
                print('best_g_per', ':', self.best_g_per, file=fil1)
                print('best_g_ftt', ':', self.best_g_ftt, file=fil1)
                print('best_g_gauss_stat', ':', self.best_g_gauss_stat, file=fil1)
                print('best_g_temp', ':', self.best_g_temp, file=fil1)
                print('', file=fil1)
        else:
            for attr in self.star_attributes:
                print(attr, ':', self.__dict__[attr])
            print('')

            print('best_int_score', ':', self.best_int_score)
            print('best_stat', ':', self.best_stat)
            print('best_per', ':', self.best_per)
            print('best_ftt', ':', self.best_ftt)
            print('best_gauss_stat', ':', self.best_gauss_stat)
            print('best_temp', ':', self.best_temp)

            print('')
            print('best_g_int_score', ':', self.best_g_int_score)
            print('best_g_stat', ':', self.best_g_stat)
            print('best_g_per', ':', self.best_g_per)
            print('best_g_ftt', ':', self.best_g_ftt)
            print('best_g_gauss_stat', ':', self.best_g_gauss_stat)
            print('best_g_temp', ':', self.best_g_temp)
            print('')

    def print_problem(self, text, file=None):
        if file is None:
            file = self.problems_file_location
        print_problem(text=text, file=file, add_time=True)

    def clean_output_directory(self, leave_periodic_stats=True,
                               best_periodic_scores_filename_postfix=Best_periodic_scores_filename_postfix):
        for ff in os.listdir(self.output_location):
            if (not leave_periodic_stats) or (not (best_periodic_scores_filename_postfix in ff)):
                os.remove(os.path.join(self.output_location, ff))

    def initialize_kepler_light_curve(self, data_location=None, output_location=None,
                                      remove_zeroth_quarter=True, remove_quarter_17=True, cut_edge_nans=True):
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
        :param cut_edge_nans: Whether to remove from the edges of the data entries which are numpy.nan
        :type cut_edge_nans: bool
        """
        if data_location is None:
            data_location = self.data_location
        if output_location is None:
            output_location = self.output_location
        self.klc = spip.KeplerLightCurve(target_id=self.target_id, get_data=True, data_location=data_location,
                                         output_location=output_location, m_1_3_r_s=self.m_1_3_r_s,
                                         cut_edge_nans=cut_edge_nans,
                                         remove_zeroth_quarter=remove_zeroth_quarter, remove_quarter_17=remove_quarter_17,
                                         simulated_red_star=(self.simulated_star == 'red'),
                                         simulated_white_star=(self.simulated_star == 'white'))
        self.n_quarters = len(self.klc.quarters)

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
            self.is_good_for_processing = False

    def check_detectable_radius(self, stellar_radius=None, snr_thresh=Score_candidate_search ** 0.5 - 1,
                                rp_re_max=Rp_re_max, rp_rs_max=Rp_rs_max, per_min_candidate=Per_min_candidate):
        """
        Finds (very approximately) whether there can be detectable planets in a target with
        the average noise properties of this target, assuming that the threshold SNR is `snr_thresh`.
        Assuming that we search planets up to radius `rp_re_max` in units of Earth radii and
        up to `rp_rs_max` in units of stellar radii. That is, we are not interested in brown dwarfs.
        The function takes the average noise properties of the star, checks what is the radius
        corresponding to the threshold SNR and sees whether this radius is too large or not.
        Writes the results in the attributes of the class.
        :param per_min_candidate:
        :type per_min_candidate:
        :param stellar_radius: Radius of the star in units of solar radii.
        :type stellar_radius: float
        :param snr_thresh: Threshold of snr
        :type snr_thresh: float
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

        n_transits_max = int(np.ceil(len(self.klc.quarters) * 90 / per_min_candidate))
        snr_max = (np.nanmax([np.mean([np.linalg.norm(self.klc.whitened_templates_td[kk][q])
                                       for q in self.klc.quarters], axis=0) / np.nanmax(self.klc.tb.templates_shifts[kk])
                              for kk in range(len(self.klc.whitened_templates_td))])
                   * self.klc.mean_total * n_transits_max ** 0.5 * (rp_rs ** 2))
        # -> for an exact conversion between the snr and the radius, look dedicated functions
        self.snr_max_detectable_radius = snr_max
        if snr_max < snr_thresh:
            self.failures = self.failures + ' too_large_noise; '
            self.is_bad_fft = True
            self.is_good_for_processing = False

    def get_predicted_snr_flux(self, flux_model, period, depth_factor=1., rp_rs=0.02, wf_fd=None, n_quarters=16,
                               include_detrending=True, nperseg_psd=128,
                               conditioned_data_filename_postfix=spip.Conditioned_data_filename_postfix):
        """
        Takes the mean flux model and the period, uploads the noise properties of the star. Checks what would be the signal-to-noise
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
        :param n_quarters:  Number of 90-days quarters expected in the data. Is important for the number of transits that can happen
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
            wf_fd = np.mean(np.array(wf_fds) ** 2, axis=0) ** 0.5
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
                            conditioned_data_filename_postfix=spip.Conditioned_data_filename_postfix):
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
        n_transits = int(np.round(n_quarters * 90. / per_inj))
        rp_inj = tm.get_rp_inj(snr_req=snr_req, wf_fd=wf_fd, n_transits=n_transits, per_inj=per_inj, m_1_3_r_s=m_1_3_r_s, inc_inj=inc_inj,
                               ecc_inj=ecc_inj, w_inj=w_inj, u1_inj=u1_inj, u2_inj=u2_inj, nperseg_psd=nperseg_psd,
                               include_detrending=include_detrending, dt=tm.dt_lc)
        return rp_inj

    def get_detectable_radius_flux(self, flux_model, period, depth_factor=1., snr_min=4.5, snr_max_1=8, snr_max_full=40,
                                   max_n_lost_transits=None, min_survived_transits=3,
                                   min_relative_survived_transits=spip.Min_required_nonzero_transits_fraction,
                                   stellar_radius=None, units='star', wf_fd=None, include_detrending=True, nperseg_psd=spip.Nperseg_psd,
                                   n_quarters=16, conditioned_data_filename_postfix=spip.Conditioned_data_filename_postfix):
        """
        Takes the mean flux model and the period, uploads the noise properties of the star. Checks what would be the signal-to-noise
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
        :param snr_max_full: Maximal SNR which can be allocated to one transit. Higher is assumed to be outside our pipeline
            operation range, so that it will never influence any density in the range of our possible triggers.
        :type snr_max_full: float
        :param stellar_radius: Radius of the star in units of solar radii.
        :type stellar_radius: float
        :param conditioned_data_filename_postfix: Text in the filename added to the target ID. See the naming notation
        :type conditioned_data_filename_postfix: str
        :param units: 'star' or 'earth'. Whether to report the result in units of stellar radii or earth radii.
        :type units: str
        :param max_n_lost_transits: Maximal number of transits which is allowed to be lost in holes
        :type max_n_lost_transits: int or None
        :param min_survived_transits: Minimal transits that need to survive, not falling on the holes
        :type min_survived_transits: int or None
        :param min_relative_survived_transits: Minimal fraction of transits that need to survive, not falling on the holes
        :type min_relative_survived_transits: float or None
        :param wf_fd: Average whitening filter in Fourier domain
        :type wf_fd: numpy.ndarray 1d
        :param n_quarters:  Number of 90-days quarters expected in the data. Is important for the number of transits that can happen
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
            wf_fd = np.mean(np.array(wf_fds) ** 2, axis=0) ** 0.5

        rps_res = af.get_detectable_radius_flux(flux_model=flux_model, period=period,
                                                depth_factor=depth_factor, snr_min=snr_min,
                                                snr_max_1=snr_max_1, snr_max_full=snr_max_full,
                                                max_n_lost_transits=max_n_lost_transits,
                                                min_survived_transits=min_survived_transits,
                                                min_relative_survived_transits=min_relative_survived_transits,
                                                stellar_radius=stellar_radius, units=units,
                                                wf_fd=wf_fd, include_detrending=include_detrending, nperseg_psd=nperseg_psd,
                                                n_quarters=n_quarters)
        return rps_res

    def get_rp_min_max(self, snr_max_full_injected, snr_min_full_injected,
                       snr_max_1_injected=spip.Max_abs_stat_veto,
                       period_inj=365.25, a_inj=None, u_inj=None,
                       data_location=None, output_location=None,
                       nsig_outliers_0=spip.Nsig_outliers_0,
                       nsig_outliers_0_pos=spip.Nsig_outliers_0_pos,
                       nsig_outliers_0_neg=spip.Nsig_outliers_0_neg,
                       nsig_outliers_psd=spip.Nsig_outliers_psd,
                       nperseg_psd=spip.Nperseg_psd,
                       npsdsmin=spip.Npsdsmin,
                       max_small_hole=spip.Max_small_hole,
                       detr_length_outliers=spip.Detr_length_outliers,
                       reject_outliers_psd=spip.Reject_outliers_psd,
                       tukey_steps_len=spip.Tukey_steps_len_psd,
                       save_low_trends=False):
        # --------------------------------------------------------------------------------------------
        # GETTING THE PSD
        target_id = self.target_id
        if (data_location is not None) and (os.path.basename(data_location) != target_id):
            data_location = af.get_correct_tid_folder(target_id, data_location, allow_creation=False)
        else:
            data_location = self.data_location
        if (output_location is not None) and (os.path.basename(output_location) != target_id):
            output_location = af.get_correct_tid_folder(target_id, output_location, allow_creation=False)
        else:
            output_location = self.output_location

        klc = spip.KeplerLightCurve(target_id=target_id,
                                    data_location=data_location,
                                    output_location=output_location,
                                    allow_data_folder_creation=False, allow_output_folder_creation=False)

        klc.nperseg_psd = nperseg_psd
        klc.subtract_means(save_means=True, masks=None)
        klc.mask_flux_outliers(nsig=nsig_outliers_0, nsig_positive=nsig_outliers_0_pos,
                               nsig_negative=nsig_outliers_0_neg,
                               nsig_clean=nsig_outliers_psd, detr_length_outliers=detr_length_outliers,
                               in_whitened_fluxes=False, modify_clean_masks=True)
        klc.subtract_means(save_means=True, masks=klc.masks)
        klc.normalize_by_means()
        klc.make_big_hole_masks(max_small_hole=max_small_hole, exclude_individual_points=True)
        klc.interpolate_fluxes(interpolate_clean_fluxes=True)
        klc.subtract_low_trends(nperseg=klc.nperseg_psd, save_low_trends=save_low_trends)
        klc.fill_additional_outliers(nsig=nsig_outliers_0, nsig_positive=nsig_outliers_0_pos,
                                     nsig_negative=nsig_outliers_0_neg,
                                     nsig_clean=nsig_outliers_psd, detr_length_outliers=detr_length_outliers,
                                     in_whitened_fluxes=False, modify_clean_masks=True,
                                     interpolate_clean_fluxes=True)  # again after piecewise detrending
        klc.prepare_for_psd(nperseg=klc.nperseg_psd, npsdsmin=npsdsmin, max_small_hole=max_small_hole)
        klc.pad_by_zeros()  # Needed to avoid crosstalk in whitening filter. And we want the end zeros to be actual zeros
        klc.measure_psds(nperseg=klc.nperseg_psd, tukey_steps_len=tukey_steps_len, reject_outliers=reject_outliers_psd)

        # Getting parameters needed for estimation:
        n_quarters = len(klc.quarters)
        wf_fds = np.array(klc.wf_fds) * klc.mean_total  # -> relative wf_fd, take mean flux into account
        wf_fd = np.mean(np.array(np.abs(wf_fds)) ** 2, axis=0) ** 0.5

        # --------------------------------------------------------------------------------------------
        # Getting flux
        inc_inj = 90.
        ecc_inj = 0.
        rp_generation = 0.02
        w_inj = 90.
        if u_inj is None:
            u = self.limb_darkening_coefs
            if u is None:
                u = Default_limb_darkening_coefs
        else:
            u = u_inj

        if a_inj is None:
            a = get_a_from_p(period=period_inj, m_1_3_r_s=self.m_1_3_r_s)
        else:
            a = a_inj

        t = tm.time(t_tot=(256 + 1) * tm.dt_lc, parity='odd')[1:]  # Making an even points number but odd parity
        flux_obs = 1. - tm.flux_par(t=t, params=None, per=period_inj, rp=rp_generation, a=a, inc=inc_inj, ecc=ecc_inj, w=w_inj, u1=u[0],
                                    u2=u[1], t0=0, supersample_factor=20, exp_time=tm.dt_lc)

        # dur = tm.templ_dur_fun(flux_obs) * tm.dt_lc
        depth_factor = np.nanmax(flux_obs) / rp_generation ** 2
        rp_rs_min, rp_rs_max = self.get_detectable_radius_flux(flux_model=flux_obs, period=period_inj,
                                                               depth_factor=depth_factor, snr_min=snr_min_full_injected,
                                                               n_quarters=n_quarters, wf_fd=wf_fd,
                                                               snr_max_1=snr_max_1_injected,
                                                               snr_max_full=snr_max_full_injected,
                                                               stellar_radius=self.stellar_radius, units='star',
                                                               include_detrending=True, nperseg_psd=spip.Nperseg_psd)
        del klc
        return rp_rs_min, rp_rs_max

    def mask_koi_planet(self, modify_self_klc=True, return_durations_transit_times=False,
                        mask_candidates=Mask_kepler_candidates,
                        mask_false_positives=Mask_kepler_false_positives,
                        mask_bright_candidates=Mask_kepler_bright_candidates,
                        bright_candidate_mes=Kepler_bright_candidate_mes,
                        bright_candidate_ses=Kepler_bright_candidate_ses,
                        avoid_period=None, avoid_time_first=None, dp_avoid=0.002, dt_avoid=0.04):
        """
        Take the information of the KIO of this target that are saved in the class instance fields. Find the times and the
        durations of transits of these planets. If `modify_self_klc` is True, then in the KeplerLightCurve attribute
        `self.klc` modify the masks to exclude the transits from the search.
        :param modify_self_klc: Whether to modify the masks in `self.klc` to mask the KOI transits
        :type modify_self_klc: bool
        :param return_durations_transit_times: Whether to return the durations and the transit times of KOI planets.
        :type return_durations_transit_times: bool
        :param mask_candidates: Whether to include in the process the Kepler planets with status 'Candidate',
            as opposed to 'Confirmed'.
        :type mask_candidates: bool
        :param mask_false_positives: Whether to include Kepler official false positives. Warning: sometimes, there is
            no ephemeris provided for them!
        :type mask_false_positives: bool
        :param mask_bright_candidates:
        :type mask_bright_candidates:
        :param bright_candidate_mes:
        :type bright_candidate_mes:
        :param bright_candidate_ses:
        :type bright_candidate_ses:
        :param avoid_period: If a signal that should be masked coincides with `avoid_period` within `dp_avoid`,
            and `avoid_time_first` within `dt_avoid`, it will not be masked.
        :type avoid_period: float, days
        :param avoid_time_first: If a signal that should be masked coincides with `avoid_period` within `dp_avoid`,
            and `avoid_time_first` within `dt_avoid`, it will not be masked.
        :type avoid_time_first: float, days
        :param dp_avoid: If a signal that should be masked coincides with `avoid_period` within `dp_avoid`,
            and `avoid_time_first` within `dt_avoid`, it will not be masked.
        :type dp_avoid: float, days
        :param dt_avoid: If a signal that should be masked coincides with `avoid_period` within `dp_avoid`,
            and `avoid_time_first` within `dt_avoid`, it will not be masked.
        :type dt_avoid: float, days
        :return: durations_and_times, durations and the transit times of KOI planets of this target
        :rtype: list of list(numpy.ndarray 1d, numpy.ndarray 1d)

        """
        assert self.is_koi, 'The target is not KOI, so the KOI planet cannot be masked!'
        # Take all the transit times with TTV, take transit duration, take fractional TDV and mask.
        durations_and_times = []
        for i, koi_name in enumerate(self.koi_dict['kepoi_name']):
            # Deciding whether to mask:
            to_mask = False
            if self.koi_dict['koi_disposition'][i] == 'CONFIRMED':
                to_mask = True
            elif mask_false_positives:
                if self.koi_dict['koi_disposition'][i] == 'FALSE POSITIVE' or self.koi_dict['koi_pdisposition'][i] == 'FALSE POSITIVE':
                    to_mask = True
            elif mask_candidates:
                if self.koi_dict['koi_disposition'][i] == 'CANDIDATE' or self.koi_dict['koi_pdisposition'][i] == 'CANDIDATE':
                    to_mask = True
            elif mask_bright_candidates:
                if self.koi_dict['koi_disposition'][i] == 'CANDIDATE' or self.koi_dict['koi_pdisposition'][i] == 'CANDIDATE':
                    max_mul = self.koi_dict['koi_max_mult_ev'][i]
                    try:
                        max_mul = float(max_mul)
                    except ValueError:
                        max_mul = 0.

                    max_sin = self.koi_dict['koi_max_sngle_ev'][i]
                    try:
                        max_sin = float(max_sin)
                    except ValueError:
                        max_sin = 0.

                    if max_mul >= bright_candidate_mes or max_sin >= bright_candidate_ses:
                        to_mask = True
            if avoid_period is not None:
                avoid_period = float(avoid_period)
                try:
                    per_koi = float(self.koi_dict['koi_period'][i])
                except ValueError:
                    per_koi = None
                try:
                    ftt_koi = float(self.koi_dict['koi_time0bk'][i])
                except ValueError:
                    ftt_koi = None
                if per_koi is not None and ftt_koi is not None:
                    if avoid_time_first is not None:
                        avoid_time_first = float(avoid_time_first)
                        if ((abs(per_koi - avoid_period) <= dp_avoid) and
                                ((abs(ftt_koi - avoid_time_first) % per_koi) <= dt_avoid)):
                            to_mask = False
                    else:
                        if abs(per_koi - avoid_period) <= dp_avoid:
                            to_mask = False
            # If decided that we need to mask:
            if to_mask:
                transit_durations = None  # just initialization, no meaning
                transit_times = None
                was_in_ttv = False
                if self.ttv_dict is not None and 'T' in self.ttv_dict and 'KOI' in self.ttv_dict:
                    for j in range(len(self.ttv_dict['KOI'])):
                        if self.ttv_dict['KOI'][j][0] == str(float(koi_name[1:])):
                            transit_times = np.array(self.ttv_dict['T'][j], dtype=float)  # this is including TTVs
                            transit_durations = np.ones(len(transit_times)) * float(
                                self.koi_dict['koi_duration'][i]) / 24. * 1.2  # we need it in days, and it was in hours
                            # -> multiplied by some safety factor because sometimes it does not cover all the transit
                            tdv_array = np.array(self.ttv_dict['TDV'][j], dtype=float)
                            tdv_array[np.isnan(tdv_array)] = 0.
                            transit_durations = transit_durations * (1. + tdv_array)  # add TDVs
                            was_in_ttv = True
                            break
                if not was_in_ttv:
                    time_first = float(self.koi_dict['koi_time0bk'][i])
                    period = float(self.koi_dict['koi_period'][i])
                    # if self.koi_dict['koi_num_transits'][i] == '':
                    if hasattr(self, 'klc') and hasattr(self.klc, 'times'):
                        time_last = np.nanmax(self.klc.times[-1])
                    else:
                        time_last = spip.Default_end_time
                    n_transits = int(np.floor((time_last - time_first) / period)) + 1
                    # WE DO NOT USE N_TRANSITS BECAUSE IT IS ONLY VALID TRANSITS! (RUN_13)
                    # else:
                    #     n_transits = int(self.koi_dict['koi_num_transits'][i])
                    transit_times = np.array([time_first + period * nn for nn in range(n_transits)])
                    transit_durations = np.ones(len(transit_times)) * float(
                        self.koi_dict['koi_duration'][i]) / 24. * 1.2  # we need it in days, and it was in hours
                    # -> multiplied by some safety factor because sometimes it does not cover all the transit
                if self.klc is not None and modify_self_klc:
                    self.klc.mask_ephemeris(durations=transit_durations, transit_times=transit_times)
                    self.is_koi_planet_masked = True
                if return_durations_transit_times:
                    durations_and_times.append([transit_durations, transit_times])
        if return_durations_transit_times:
            return durations_and_times

    def run_ses_search(self, min_quarters_to_process=Min_quarters_to_process,
                       conditioned_data_filename_postfix=Conditioned_data_filename_postfix,
                       mask_koi_planet=True, save_conditioned_data=False, save_connected_stats=True,
                       connected_stats_filename_postfix=Connected_statistic_data_filename_postfix,
                       file_path_connected_stats=None,
                       save_no_veto_no_score_cor_stats=False, stop_if_bad=True,
                       no_veto_no_score_cor_filename_postfix=No_veto_no_score_cor_stats_filename_postfix,
                       delete_full_stats=True):
        """
        Search for the single-event statistics. Is done throughout the KeplerLightCurve instance, so see more details
        in `search_pipeline.py`. Includes the following:
        - Getting the data, checking whether it is good enough for the search, masking possible KOI
        - Conditioning data and templates
        - Checking the detectable radii in this noise, and stop the search if it is too bad
        - Running the matched-filtering for single-transit templates
        - Vetting statistics, assessing the distribution non-gaussianity
        - Connecting the quarters into continuous data ready for the periodicity search
        - Saving the processed data for future use
        The next stage after this would be to run the periodicity search.

        :param file_path_connected_stats:
        :type file_path_connected_stats:
        :param min_quarters_to_process: Minimal number of quarters that is required to process the data. If
            there are fewer quarters, the search will stop.
        :type min_quarters_to_process: int
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
        :param delete_full_stats:
        :type delete_full_stats:
        """
        t1 = time.time()
        self.klc = spip.KeplerLightCurve(target_id=self.target_id, get_data=True, data_location=self.data_location,
                                         output_location=self.output_location, m_1_3_r_s=self.m_1_3_r_s,
                                         simulated_red_star=(self.simulated_star == 'red'),
                                         simulated_white_star=(self.simulated_star == 'white'))
        self.n_quarters = len(self.klc.quarters)
        if len(self.klc.quarters) < 6:
            self.print_problem('has only ' + str(len(self.klc.quarters)) + ' quarters')

        if len(self.klc.quarters) < min_quarters_to_process:
            self.is_too_short = True  # too few quarters to search for a 50-day period
            self.is_good_for_processing = False
        else:
            self.is_too_short = False
            self.bad_fft_check()  # Made on orig_fluxes
            if (not self.is_bad_fft) or (not stop_if_bad):
                if self.is_koi and mask_koi_planet:
                    self.mask_koi_planet(modify_self_klc=True, return_durations_transit_times=False,
                                         mask_candidates=Mask_kepler_candidates,
                                         mask_false_positives=Mask_kepler_false_positives,
                                         mask_bright_candidates=Mask_kepler_bright_candidates,
                                         bright_candidate_mes=Kepler_bright_candidate_mes,
                                         bright_candidate_ses=Kepler_bright_candidate_ses)
                    # Attention: we do not mask candidates
                self.klc.condition_data()
                self.was_notched = self.klc.was_notched
                self.klc.condition_templates()
                self.check_detectable_radius(stellar_radius=self.stellar_radius)
                if self.is_bad_fft:
                    self.print_problem('is bad fft')
                if self.is_koi:
                    self.print_problem('is koi')
                if self.was_notched:
                    self.print_problem('was notched')
                if (not self.is_too_short) and (not self.is_bad_fft or (not stop_if_bad)):
                    if save_conditioned_data:
                        self.klc.save_conditioned_data(filename_postfix=conditioned_data_filename_postfix)
                    self.klc.detect_single_transits(apply_veto=True, gaussianize=True, save_no_veto_no_score_cor_stats_as_attr=True)
                    if save_no_veto_no_score_cor_stats:
                        self.klc.save_no_veto_no_score_cor_stats(filename_postfix=no_veto_no_score_cor_filename_postfix)
                    self.klc.connect_quarters()
                    if save_connected_stats:
                        self.klc.save_connected_stats(filename_postfix=connected_stats_filename_postfix,
                                                      file_path=file_path_connected_stats)
                    self.klc.interleave_shifts(delete_fulls=delete_full_stats)

                    t2 = time.time()
                    self.conditioning_ses_time = t2 - t1

    def get_connected_klc(self, reinitialize=True, allow_loading=True,
                          conditioned_data_filename_postfix=Conditioned_data_filename_postfix,
                          file_path_connected_data=None,
                          connected_stats_filename_postfix=Connected_statistic_data_filename_postfix,
                          file_path_conditioned_data=None,
                          mask_koi_planet=True, period_avoid_masking=None, ftt_avoid_masking=None,
                          save_connected_stats=True, save_conditioned_data=True,
                          use_locker=True):
        """
        In the KeplerLightCurve instance attribute (`self.klc`) of this class, load the precomputed statistics
        (or compute if there are no precomputed) with connected quarters which can be used in the periodicity
        search. See more details about the statistics in `search_pipeline.py`. If this method is executed, you
        will get `self.klc` with ready single-transit statistics at which the periodicity search can be run.
        These statistics also can be time-scrambled.
        :param use_locker:
        :type use_locker:
        :param allow_loading:
        :type allow_loading:
        :param file_path_connected_data:
        :type file_path_connected_data:
        :param file_path_conditioned_data:
        :type file_path_conditioned_data:
        :param period_avoid_masking:
        :type period_avoid_masking:
        :param ftt_avoid_masking:
        :type ftt_avoid_masking:
        :param reinitialize: Whether to re-instantiate `self.klc` even if it already existed before.
        :type reinitialize: bool
        :param conditioned_data_filename_postfix: Text added to the target ID in the name of the file saving conditioned data.
            See naming notations. This is needed in the case that the connected statistics file does not exist. Then
            the conditioned data can be loaded and processed to make it all faster.
        :type conditioned_data_filename_postfix: str
        :param mask_koi_planet: Whether to mask Kepler planets in the case `self.klc` is created or re-instantiated.
        :type mask_koi_planet: bool or None
        :param save_connected_stats: Whether to save connected statistics in the case they were calculated now.
        :type save_connected_stats: bool
        :param save_conditioned_data: Whether to save conditioned data in the case they were calculated now.
        :type save_conditioned_data: bool
        :param connected_stats_filename_postfix: Text added to the target ID in the name of the file saving connected data.
            See naming notations.
        :type connected_stats_filename_postfix: str
        """
        if (self.klc is None) or reinitialize:
            self.klc = spip.KeplerLightCurve(target_id=self.target_id, get_data=True, data_location=self.data_location,
                                             output_location=self.output_location, m_1_3_r_s=self.m_1_3_r_s,
                                             simulated_red_star=(self.simulated_star == 'red'),
                                             simulated_white_star=(self.simulated_star == 'white'))  # Q17 is removed
            self.bad_fft_check()  # Made on orig_fluxes
            if self.is_koi and (mask_koi_planet or (self.is_koi_planet_masked and mask_koi_planet is None)):
                # the latter is from the pickled attributes, meaning
                # that the saved connected data is probably also with the masked koi planet
                self.mask_koi_planet(modify_self_klc=True, return_durations_transit_times=False,
                                     mask_candidates=Mask_kepler_candidates,
                                     mask_false_positives=Mask_kepler_false_positives,
                                     mask_bright_candidates=Mask_kepler_bright_candidates,
                                     bright_candidate_mes=Kepler_bright_candidate_mes,
                                     bright_candidate_ses=Kepler_bright_candidate_ses,
                                     avoid_period=period_avoid_masking, avoid_time_first=ftt_avoid_masking)
                # # if there are only candidates, it will not mask
        if period_avoid_masking is not None:
            insert_after_postfix_period = str(int(np.round(period_avoid_masking)))
        else:
            insert_after_postfix_period = None
        if file_path_connected_data is None:
            file_path_connected_data = af.get_saving_file_path(output_location=self.output_location, target_id=self.target_id,
                                                               filename_postfix=connected_stats_filename_postfix,
                                                               insert_after_postfix=None,
                                                               insert_integer_periods=insert_after_postfix_period,
                                                               file_format='npy')
        if file_path_conditioned_data is None:
            file_path_conditioned_data = af.get_saving_file_path(output_location=self.output_location, target_id=self.target_id,
                                                                 filename_postfix=conditioned_data_filename_postfix,
                                                                 insert_after_postfix=None,
                                                                 insert_integer_periods=insert_after_postfix_period,
                                                                 file_format='npy')
        if self.klc.statistics_full is None or not hasattr(self.klc, 'statistics_full'):
            success = False
            if allow_loading and os.path.exists(file_path_connected_data) and os.path.getsize(file_path_connected_data) > 0:
                try:
                    if use_locker:
                        with Locker(file_path_connected_data):
                            self.klc.load_connected_stats(file_path=file_path_connected_data)
                            success = True
                    else:
                        self.klc.load_connected_stats(file_path=file_path_connected_data)
                        success = True
                except Exception as exc:
                    success = False
                    report_exception(exception=exc, file_to_append=self.problems_file_location,
                                     title='statistics_full file', info='')
            if not success:
                if self.klc.statistics_raw is None:
                    # Try to load conditioned data if the file exists and is not empty:
                    if allow_loading and os.path.exists(file_path_conditioned_data) and os.path.getsize(file_path_conditioned_data) > 0:
                        try:
                            self.klc.load_conditioned_data(file=file_path_conditioned_data)
                            self.klc.initialize_templates()
                            success = True
                        except (EOFError, pickle.UnpicklingError):
                            success = False
                    if not success:
                        self.klc.condition_data()
                        self.was_notched = self.klc.was_notched
                        self.klc.condition_templates()
                        if save_conditioned_data:
                            self.klc.save_conditioned_data(filename_postfix=conditioned_data_filename_postfix)
                    self.klc.detect_single_transits(apply_veto=True, gaussianize=True)
                    self.klc.connect_quarters()
                    if save_connected_stats:
                        self.klc.save_connected_stats(filename_postfix=connected_stats_filename_postfix,
                                                      file_path=file_path_connected_data)

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

    def calculate_baseline_integral_score_rate(self, baseline_rate_period=None):
        """
        Calculates the baseline integral score rate that is used to re-normalise the integral score and make it have the
        form of statistic with punishment. This is the rate for
        :return:
        :rtype:
        """
        if self.periods_in_chunks is None:
            self.set_periodicity_search_parameters(force_overwrite=False)
            self.periods_in_chunks = self.klc.make_periods_chunks(p_min=self.p_min, p_max=self.p_max,
                                                                  per_phases_per_chunk=self.per_phases_per_chunk,
                                                                  duty_cycle=self.duty_cycle, dt=self.klc.dt_days,
                                                                  t_mission=self.klc.t_mission)

        if baseline_rate_period is None:
            baseline_rate_period = self.periods_in_chunks[0][0]
            self.dp_0 = self.periods_in_chunks[0][1] - self.periods_in_chunks[0][0]
        else:
            per_chunk, _ = ut.find_closest_chunk(self.periods_in_chunks, baseline_rate_period, overlap_bins=0)
            baseline_rate_period = per_chunk[0]
            self.dp_0 = per_chunk[1] - per_chunk[0]

        self.baseline_rate_period = baseline_rate_period
        self.d_ftt_0 = tm.dt_lc  # Should be taken from KeplerLightCurve._phases, but hopefully, this is correct
        # Preparing densities for renormalizing the integral score
        self.log_baseline_int_score_rate_density = np.log(tm.get_number_of_transiting_planets_density_per_dp(
            baseline_rate_period, m_1_3_r_s=1., dp=1.) / baseline_rate_period)
        # -> the function gives density per dp, and dividing it by the period, we get it also for dt.

        self.log_baseline_phase_space_measure = np.log(self.dp_0 * self.d_ftt_0)
        self.log_baseline_int_score_rate = (self.log_baseline_int_score_rate_density
                                            + self.log_baseline_phase_space_measure)

    def run_periodicity_search(self, p_min=P_min_search, p_max=P_max_search, duty_cycle=spip.Duty_cycle,
                               per_phases_per_chunk=Per_phases_per_chunk,
                               best_periodic_scores_filename_postfix=Best_periodic_scores_filename_postfix,
                               best_periodic_scores_file_dir=None, overlap_period_bins=spip.P_bins_overlap,
                               min_required_nonzero_transits_number=spip.Min_required_nonzero_transits_number,
                               min_required_nonzero_transits_fraction=spip.Min_required_nonzero_transits_fraction,
                               baseline_rate_period=Baseline_rate_period,
                               allow_getting_connected_klc=False,
                               print_no_trigger_found=True):

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
        :param min_required_nonzero_transits_number: Minimal required number of valid transits for a trigger. Will be used
            in the pre-filtering before the periodicity search and in the veto_mes
        :type min_required_nonzero_transits_number: int
        :param min_required_nonzero_transits_fraction: Minimal required fraction of valid transits with respect to the total
         number of transits for a trigger. Will be used in the pre-filtering before the periodicity search and in the veto_mes
        :type min_required_nonzero_transits_fraction: float
        :param baseline_rate_period:
        :type baseline_rate_period:
        :param allow_getting_connected_klc:
        :type allow_getting_connected_klc:
        """
        if ((self.klc is None or not hasattr(self.klc, 'statistics_full') or self.klc.statistics_full is None)
                and allow_getting_connected_klc):
            self.get_connected_klc(reinitialize=False, mask_koi_planet=None)
        p_max = min(self.klc.t_mission / 2, p_max)
        self.set_periodicity_search_parameters(p_min=p_min, p_max=p_max, per_phases_per_chunk=per_phases_per_chunk, duty_cycle=duty_cycle,
                                               force_overwrite=True)
        self.klc.delete_irrelevant()  # For memory

        self.periods_in_chunks = self.klc.make_periods_chunks(p_min=self.p_min, p_max=self.p_max,
                                                              per_phases_per_chunk=self.per_phases_per_chunk,
                                                              duty_cycle=self.duty_cycle, dt=self.klc.dt_days, t_mission=self.klc.t_mission)
        self.calculate_baseline_integral_score_rate(baseline_rate_period=baseline_rate_period)
        best_candidates_concat, failures = self.search_periodicity_over_chunks(
            period_chunks=self.periods_in_chunks,
            force_interleave_shifts=False,
            min_required_nonzero_transits_number=min_required_nonzero_transits_number,
            min_required_nonzero_transits_fraction=min_required_nonzero_transits_fraction,
            overlap_period_bins=overlap_period_bins,
            report_exceptions=True, problem_description='',
            return_failures=True, save_search_time=True, save_veto_masks=True, delete_full_stats=True,
            print_no_trigger_found=print_no_trigger_found)
        self.best_candidates_concat = best_candidates_concat
        self.failures += failures
        if best_periodic_scores_file_dir is not None:
            np.save(best_periodic_scores_file_dir, np.array(best_candidates_concat, dtype=object), allow_pickle=True)
        else:
            file_path = af.get_saving_file_path(output_location=self.output_location, target_id=self.target_id,
                                                filename_postfix=best_periodic_scores_filename_postfix, file_format='npy')
            np.save(file_path, np.array(best_candidates_concat, dtype=object), allow_pickle=True)

    def search_periodicity_over_chunks(self, period_chunks=None,
                                       force_interleave_shifts=False,
                                       min_required_nonzero_transits_number=spip.Min_required_nonzero_transits_number,
                                       min_required_nonzero_transits_fraction=spip.Min_required_nonzero_transits_fraction,
                                       overlap_period_bins=spip.P_bins_overlap,
                                       return_failures=False, save_search_time=False,
                                       save_veto_masks=False,
                                       delete_full_stats=False,
                                       delete_periodic_interleaved=True,
                                       save_uncorrected_periodic=False,
                                       delete_non_marginalized=False,
                                       delete_mes=True,
                                       report_exceptions=False, problem_description='', exception_info='',
                                       print_no_trigger_found=True):
        if period_chunks is None:
            period_chunks = self.periods_in_chunks

        failures = ''
        t1 = time.time()
        best_candidates_concat = []

        for pers in period_chunks:
            has_enough_transits = self.klc.if_has_potentially_enough_transits(
                periods_to_check=(pers[0], pers[-1]),
                min_required_nonzero_transits_number=min_required_nonzero_transits_number,
                min_required_nonzero_transits_fraction=min_required_nonzero_transits_fraction)
            if has_enough_transits:
                try:
                    self.klc.periodicity_search(p_min=None, p_max=None, periods=pers,
                                                delete_full_stats=delete_full_stats, force_interleave_shifts=force_interleave_shifts,
                                                delete_periodic_interleaved=delete_periodic_interleaved,
                                                save_uncorrected_periodic=save_uncorrected_periodic,
                                                delete_non_marginalized=delete_non_marginalized,
                                                overlap_period_bins=overlap_period_bins)
                    self.klc.get_best_trigger(apply_veto_mes=True, in_gaussianized=False, save_veto_masks=save_veto_masks,
                                              min_required_nonzero_transits_number=min_required_nonzero_transits_number,
                                              min_required_nonzero_transits_fraction=min_required_nonzero_transits_fraction,
                                              log_baseline_integral_score_rate=self.log_baseline_int_score_rate)
                    self.klc.get_best_trigger(apply_veto_mes=True, in_gaussianized=True, save_veto_masks=save_veto_masks,
                                              min_required_nonzero_transits_number=min_required_nonzero_transits_number,
                                              min_required_nonzero_transits_fraction=min_required_nonzero_transits_fraction,
                                              log_baseline_integral_score_rate=self.log_baseline_int_score_rate)
                    self.klc.make_best_periodic_scores()
                    if print_no_trigger_found and np.isnan(self.klc.best_period):
                        self.print_problem(text=problem_description + 'No trigger found after veto MES for pers='
                                           + str(pers[0]) + '-' + str(pers[-1]),
                                           file=self.problems_file_location)
                    best_candidates_concat.append(np.array(self.klc.best_periodic_scores_data, dtype=object))
                except Exception as exc:
                    if report_exceptions:
                        report_exception(exception=exc, file_to_append=self.problems_file_location,
                                         title=problem_description + 'pers=' + str(pers[0]) + '-' + str(pers[-1]), info=exception_info)
                if delete_mes:
                    self.klc.delete_mes()
            else:
                failures += 'pers=[' + str(pers[0]) + '-' + str(pers[0]) + '] : Too small possible non-zero transits fraction;\n'
        t2 = time.time()
        if save_search_time:
            self.periodicity_search_time = t2 - t1
        if not return_failures:
            return best_candidates_concat
        else:
            return best_candidates_concat, failures

    def retrieve_best_candidates_concat(self, best_periodic_scores_filename_postfix=Best_periodic_scores_filename_postfix):
        file_path = af.get_saving_file_path(output_location=self.output_location, target_id=self.target_id,
                                            filename_postfix=best_periodic_scores_filename_postfix, file_format='npy')
        self.best_candidates_concat = np.load(file_path, allow_pickle=True)

    def save_all_triggers_from_per_chunks(self,
                                          save_to_threshold_crossing_list=False,
                                          threshold_crossing_list_multiple_triggers_file=None,
                                          threshold_g_crossing_list_multiple_triggers_file=None,
                                          threshold_crossing_list_multiple_triggers_filename=None,
                                          threshold_g_crossing_list_multiple_triggers_filename=None,
                                          threshold_crossing_list_location=None,
                                          best_periodic_scores=None,
                                          best_periodic_scores_filename_postfix=Best_periodic_scores_filename_postfix):
        """
        Compares the scores of several chunks and sets the best results as class attributes
        :param best_periodic_scores: The properly formatted list of the best candidates from every chunk
        :type best_periodic_scores: list of lists of length 2
        :param save_to_threshold_crossing_list:
        :type save_to_threshold_crossing_list:
        :param threshold_crossing_list_multiple_triggers_file:
        :type threshold_crossing_list_multiple_triggers_file:
        :param threshold_g_crossing_list_multiple_triggers_file:
        :type threshold_g_crossing_list_multiple_triggers_file:
        :param threshold_crossing_list_multiple_triggers_filename:
        :type threshold_crossing_list_multiple_triggers_filename:
        :param threshold_g_crossing_list_multiple_triggers_filename:
        :type threshold_g_crossing_list_multiple_triggers_filename:
        :param threshold_crossing_list_location:
        :type threshold_crossing_list_location:
        :param best_periodic_scores_filename_postfix:
        :type best_periodic_scores_filename_postfix:
        """
        if (threshold_crossing_list_multiple_triggers_file is None) and (threshold_crossing_list_location is not None):
            threshold_crossing_list_multiple_triggers_file = os.path.join(threshold_crossing_list_location,
                                                                          threshold_crossing_list_multiple_triggers_filename)
        if (threshold_g_crossing_list_multiple_triggers_file is None) and (threshold_crossing_list_location is not None):
            threshold_g_crossing_list_multiple_triggers_file = os.path.join(threshold_crossing_list_location,
                                                                            threshold_g_crossing_list_multiple_triggers_filename)

        if best_periodic_scores is None:
            if self.best_candidates_concat is None:
                self.retrieve_best_candidates_concat(best_periodic_scores_filename_postfix=best_periodic_scores_filename_postfix)
            best_periodic_scores = self.best_candidates_concat
        if len(best_periodic_scores) > 0:
            ind_gauss = 0
            stats_res = np.array(
                [best_periodic_scores[i][spip.Best_per_results_index_stats][ind_gauss] for i in range(len(best_periodic_scores))])
            periods_res = np.array(
                [best_periodic_scores[i][spip.Best_per_results_index_per][ind_gauss] for i in range(len(best_periodic_scores))])
            integral_scores_res = np.array(
                [best_periodic_scores[i][spip.Best_per_results_index_best_integral][ind_gauss] for i in range(len(best_periodic_scores))])
            ftts_res = np.array(
                [best_periodic_scores[i][spip.Best_per_results_index_ftt][ind_gauss] for i in range(len(best_periodic_scores))])
            stats_gauss_res = np.array(
                [best_periodic_scores[i][spip.Best_per_results_index_stats_gauss][ind_gauss] for i in range(len(best_periodic_scores))])
            max_templs_res = np.array(
                [best_periodic_scores[i][spip.Best_per_results_index_template][ind_gauss] for i in range(len(best_periodic_scores))])

            ind_gauss = 1
            g_stats_res = np.array(
                [best_periodic_scores[i][spip.Best_per_results_index_stats][ind_gauss] for i in range(len(best_periodic_scores))])
            g_periods_res = np.array(
                [best_periodic_scores[i][spip.Best_per_results_index_per][ind_gauss] for i in range(len(best_periodic_scores))])
            g_int_scores_res = np.array(
                [best_periodic_scores[i][spip.Best_per_results_index_best_integral][ind_gauss] for i in range(len(best_periodic_scores))])
            g_ftts_res = np.array(
                [best_periodic_scores[i][spip.Best_per_results_index_ftt][ind_gauss] for i in range(len(best_periodic_scores))])
            g_stats_gauss_res = np.array(
                [best_periodic_scores[i][spip.Best_per_results_index_stats_gauss][ind_gauss] for i in range(len(best_periodic_scores))])
            g_max_templs_res = np.array(
                [best_periodic_scores[i][spip.Best_per_results_index_template][ind_gauss] for i in range(len(best_periodic_scores))])

            # Determining TCEs among [chunks maxima in non-gaussianized sense]
            trigger_mask = integral_scores_res >= Score_candidate_search
            triggering_int_scores = integral_scores_res[trigger_mask]
            triggering_stats = stats_res[trigger_mask]
            triggering_periods = periods_res[trigger_mask]
            triggering_ftts = ftts_res[trigger_mask]
            triggering_stats_gauss = stats_gauss_res[trigger_mask]
            triggering_max_templs = max_templs_res[trigger_mask]

            self.triggers = [[] for _ in range(spip.Best_per_results_length)]
            self.triggers[spip.Best_per_results_index_best_integral] = triggering_int_scores
            self.triggers[spip.Best_per_results_index_stats] = triggering_stats
            self.triggers[spip.Best_per_results_index_per] = triggering_periods
            self.triggers[spip.Best_per_results_index_ftt] = triggering_ftts
            self.triggers[spip.Best_per_results_index_stats_gauss] = triggering_stats_gauss
            self.triggers[spip.Best_per_results_index_template] = triggering_max_templs

            if save_to_threshold_crossing_list:
                with open(threshold_crossing_list_multiple_triggers_file, 'a') as fil1:  # will save here these good targets
                    for i in range(len(triggering_int_scores)):
                        triggering_int_score = triggering_int_scores[i]
                        triggering_stat = triggering_stats[i]
                        triggering_period = triggering_periods[i]
                        triggering_ftt = triggering_ftts[i]
                        triggering_max_templ = triggering_max_templs[i]
                        triggering_stat_gauss = triggering_stats_gauss[i]
                        tc.print_trigger_line(target_id=self.target_id, stat=triggering_stat, per=triggering_period, ftt=triggering_ftt,
                                              gauss_stat=triggering_stat_gauss, integral_score=triggering_int_score,
                                              template=triggering_max_templ, file=fil1, sep=' ', end='\n')

            # Determining TCEs among [chunks maxima in gaussianized sense]
            # SAME FOR GAUSS BUT A DIFFERENT ORDER OF SAVING: GAUSS STAT GOES FIRST< AND THE INTEGRAL WAS ALSO MADE ON GAUSS
            g_trigger_mask = g_int_scores_res >= Score_candidate_search
            g_triggering_int_scores = g_int_scores_res[g_trigger_mask]
            g_triggering_stats = g_stats_res[g_trigger_mask]
            g_triggering_periods = g_periods_res[g_trigger_mask]
            g_triggering_ftts = g_ftts_res[g_trigger_mask]
            g_triggering_stats_gauss = g_stats_gauss_res[g_trigger_mask]
            g_triggering_max_templs = g_max_templs_res[g_trigger_mask]

            self.g_triggers = [[] for _ in range(spip.Best_per_results_length)]
            self.g_triggers[spip.Best_per_results_index_best_integral] = g_triggering_int_scores
            self.g_triggers[spip.Best_per_results_index_stats] = g_triggering_stats
            self.g_triggers[spip.Best_per_results_index_per] = g_triggering_periods
            self.g_triggers[spip.Best_per_results_index_ftt] = g_triggering_ftts
            self.g_triggers[spip.Best_per_results_index_stats_gauss] = g_triggering_stats_gauss
            self.g_triggers[spip.Best_per_results_index_template] = g_triggering_max_templs

            if save_to_threshold_crossing_list:
                with open(threshold_g_crossing_list_multiple_triggers_file, 'a') as fil1:  # will save here these good targets
                    for i in range(len(g_triggering_int_scores)):
                        g_triggering_int_score = g_triggering_int_scores[i]
                        g_triggering_stat = g_triggering_stats[i]
                        g_triggering_period = g_triggering_periods[i]
                        g_triggering_ftt = g_triggering_ftts[i]
                        g_triggering_max_templ = g_triggering_max_templs[i]
                        g_triggering_stat_gauss = g_triggering_stats_gauss[i]
                        tc.print_trigger_line(target_id=self.target_id, stat=g_triggering_stat, per=g_triggering_period,
                                              ftt=g_triggering_ftt,
                                              gauss_stat=g_triggering_stat_gauss, integral_score=g_triggering_int_score,
                                              template=g_triggering_max_templ, file=fil1, sep=' ', end='\n')
        else:
            self.triggers = [[] for _ in range(spip.Best_per_results_length)]
            self.g_triggers = [[] for _ in range(spip.Best_per_results_length)]

    def find_best_trigger_from_per_chunks(self, best_candidates_concat=(),
                                          delta_integral_close=Delta_integral_close, delta_stat_close=Delta_stat_close):
        """
        Compares the scores of several chunks and sets the best results as class attributes
        The `close` parameters are for the maximal score selection. Sometimes the maximal integral corresponds
        to the chunk that does not capture the true peak. But the integral does capture the true peak because of the
        period overlapping. We check whether in the neighboring chunk within `Delta_period_close` there is a score which is no less
        than the maximum minus `Delta_integral_close` and gives the gain in statistic more than `Delta_stat_close`. If
        there is such a score, it is selected as a maximum.
        :param delta_integral_close:
        :type delta_integral_close:
        :param delta_stat_close:
        :type delta_stat_close:
        :param best_candidates_concat: The properly formatted list of the best candidates from every chunk
        :type best_candidates_concat: list of lists of length 2
        """
        self.best_best_candidates_concat_index = None
        self.best_g_best_candidates_concat_index = None
        if len(best_candidates_concat) > 0:
            self.best_int_score = -np.inf
            ii_last = None
            for ii, best_candidates in enumerate(best_candidates_concat):  # if there is a big MES score, further processing
                scormax = best_candidates[spip.Best_per_results_index_best_integral][0]
                stat = best_candidates[spip.Best_per_results_index_stats][0]
                assign = False
                if not np.isnan(scormax):
                    if (self.best_int_score is None or self.best_stat is None
                            or np.isnan(self.best_int_score) or np.isnan(self.best_stat) or (self.best_int_score == -np.inf)):
                        assign = True
                    else:
                        delta_integral = scormax - self.best_int_score
                        if ii_last is not None and (ii - ii_last == 1):
                            # -> if we are only one chunk away from the previous peak, then it can be that
                            # one of the chunks contains the true peak, but the integral is in the other chunk captured it from the
                            # overlap, but does not point to the peak itself.
                            # see get_best_int_score_and_stat_index for the compact implementation of the selection
                            delta_stat = stat - self.best_stat
                            if delta_integral >= - delta_stat * (delta_integral_close / delta_stat_close):
                                assign = True
                        elif delta_integral >= 0:
                            assign = True
                if assign:
                    self.best_int_score = scormax
                    self.best_stat = best_candidates[spip.Best_per_results_index_stats][0]
                    self.best_per = best_candidates[spip.Best_per_results_index_per][0]
                    self.best_ftt = best_candidates[spip.Best_per_results_index_ftt][0]
                    self.best_gauss_stat = best_candidates[spip.Best_per_results_index_stats_gauss][0]
                    if not np.isnan(best_candidates[spip.Best_per_results_index_template][0]):
                        self.best_temp = int(best_candidates[spip.Best_per_results_index_template][0])
                    else:
                        self.best_temp = np.nan
                    self.best_best_candidates_concat_index = ii
                    ii_last = ii
            if self.best_int_score == -np.inf:
                self.best_int_score = np.nan
                self.best_stat = np.nan
                self.best_per = np.nan
                self.best_ftt = np.nan
                self.best_gauss_stat = np.nan
                self.best_temp = np.nan

            # Same for g_trigger:
            self.best_g_int_score = -np.inf
            ii_last = None
            for ii, best_candidates in enumerate(best_candidates_concat):  # if there is a big MES score, further processing
                scormax = best_candidates[spip.Best_per_results_index_best_integral][1]
                stat = best_candidates[spip.Best_per_results_index_stats][1]
                assign = False
                if not np.isnan(scormax):
                    if (self.best_g_int_score is None or self.best_g_stat is None
                            or np.isnan(self.best_g_int_score) or np.isnan(self.best_g_stat) or (self.best_g_int_score == -np.inf)):
                        assign = True
                    else:
                        delta_integral = scormax - self.best_g_int_score
                        if ii_last is not None and (ii - ii_last == 1):
                            delta_stat = stat - self.best_g_stat
                            if delta_integral >= - delta_stat * (delta_integral_close / delta_stat_close):
                                assign = True
                        elif delta_integral >= 0:
                            assign = True
                if assign:
                    self.best_g_int_score = scormax
                    self.best_g_stat = best_candidates[spip.Best_per_results_index_stats][1]
                    self.best_g_per = best_candidates[spip.Best_per_results_index_per][1]
                    self.best_g_ftt = best_candidates[spip.Best_per_results_index_ftt][1]
                    self.best_g_gauss_stat = best_candidates[spip.Best_per_results_index_stats_gauss][1]
                    if not np.isnan(best_candidates[spip.Best_per_results_index_template][1]):
                        self.best_g_temp = int(best_candidates[spip.Best_per_results_index_template][1])
                    else:
                        self.best_g_temp = np.nan
                    self.best_g_best_candidates_concat_index = ii
                    ii_last = ii
            if self.best_g_int_score == -np.inf:
                self.best_g_int_score = np.nan
                self.best_g_stat = np.nan
                self.best_g_per = np.nan
                self.best_g_ftt = np.nan
                self.best_g_gauss_stat = np.nan
                self.best_g_temp = np
        else:
            self.best_int_score = np.nan
            self.best_stat = np.nan
            self.best_per = np.nan
            self.best_ftt = np.nan
            self.best_gauss_stat = np.nan
            self.best_temp = np.nan
            self.best_g_int_score = np.nan
            self.best_g_stat = np.nan
            self.best_g_per = np.nan
            self.best_g_ftt = np.nan
            self.best_g_gauss_stat = np.nan
            self.best_g_temp = np.nan

    def save_best_candidate(self, candidate_threshold=Score_candidate_search,
                            period_threshold=None,  # Per_min_candidate  #* not doing period limitation here, will do in post-process
                            threshold_crossing_list_file=None,
                            threshold_crossing_list_location=None, threshold_crossing_list_filename=None,
                            threshold_g_crossing_list_file=None, threshold_g_crossing_list_filename=None,
                            save_to_threshold_crossing_list=False, best_candidates_concat=None):
        if best_candidates_concat is None:
            best_candidates_concat = self.best_candidates_concat
        self.find_best_trigger_from_per_chunks(best_candidates_concat=best_candidates_concat)
        if ((self.best_int_score is not None) and (self.best_stat is not None)
                and (not np.isnan(self.best_int_score)) and (not np.isnan(self.best_stat))
                and (self.best_per is not None) and (not np.isnan(self.best_per))
                and ((period_threshold is None) or (self.best_per >= period_threshold))
                and self.best_int_score >= candidate_threshold):
            self.is_threshold_crossing = True
            if save_to_threshold_crossing_list:
                if (threshold_crossing_list_file is None) and (threshold_crossing_list_location is not None):
                    threshold_crossing_list_file = os.path.join(threshold_crossing_list_location, threshold_crossing_list_filename)
                with open(threshold_crossing_list_file, 'a') as fil1:  # will save here these good targets
                    tc.print_trigger_line(target_id=self.target_id, integral_score=self.best_int_score, stat=self.best_stat,
                                          per=self.best_per, ftt=self.best_ftt, gauss_stat=self.best_gauss_stat, template=self.best_temp,
                                          file=fil1, sep=' ', end='\n')
        else:
            self.is_threshold_crossing = False
        # ----------------------------------------------------------------------------------------------------------------
        # Separately for gaussianized:
        if ((self.best_g_int_score is not None) and (self.best_g_gauss_stat is not None)
                and (not np.isnan(self.best_g_int_score)) and (not np.isnan(self.best_g_gauss_stat))
                and (self.best_g_per is not None) and (not np.isnan(self.best_g_per))
                and ((period_threshold is None) or (self.best_g_per >= period_threshold))
                and self.best_g_int_score >= candidate_threshold):
            self.is_threshold_g_crossing = True
            if save_to_threshold_crossing_list:
                if (threshold_g_crossing_list_file is None) and (threshold_crossing_list_location is not None):
                    threshold_g_crossing_list_file = os.path.join(threshold_crossing_list_location, threshold_g_crossing_list_filename)
                with open(threshold_g_crossing_list_file, 'a') as fil1:  # will save here these good targets
                    tc.print_trigger_line(target_id=self.target_id, integral_score=self.best_g_int_score, stat=self.best_g_stat,
                                          per=self.best_g_per, ftt=self.best_g_ftt, gauss_stat=self.best_g_gauss_stat,
                                          template=self.best_g_temp, file=fil1, sep=' ', end='\n')
        else:
            self.is_threshold_g_crossing = False

    def run_main_search(self, p_min=P_min_search, p_max=P_max_search,
                        per_phases_per_chunk=Per_phases_per_chunk,
                        candidate_threshold=Score_candidate_search,
                        min_quarters_to_process=Min_quarters_to_process,
                        mask_koi_planet=True,
                        save_conditioned_data=True,
                        save_no_veto_no_score_cor_stats=False,
                        save_connected_stats=True,
                        conditioned_data_filename_postfix=Conditioned_data_filename_postfix,
                        best_periodic_scores_filename_postfix=Best_periodic_scores_filename_postfix,
                        no_veto_no_score_cor_stats_filename_postfix=No_veto_no_score_cor_stats_filename_postfix,
                        connected_stats_data_filename_postfix=Connected_statistic_data_filename_postfix,
                        save_to_threshold_crossing_list=True,
                        threshold_crossing_list_file=Threshold_crossing_list_file,
                        threshold_g_crossing_list_file=Threshold_g_crossing_list_file,
                        threshold_crossing_list_location=Trigger_lists_folder,
                        threshold_crossing_list_filename=Threshold_crossing_list_filename,
                        threshold_g_crossing_list_filename=Threshold_g_crossing_list_filename,
                        save_multiple_triggers=False,
                        threshold_crossing_list_multiple_triggers_filename=Threshold_crossing_list_multiple_triggers_filename,
                        threshold_g_crossing_list_multiple_triggers_filename=Threshold_g_crossing_list_multiple_triggers_filename,
                        clean_dir_if_not_threshold_crossing=False, overlap_period_bins=spip.P_bins_overlap,
                        clean_problems_file=True, stop_if_bad=True,
                        seeds_scrambling_run_file=Seeds_scrambling_run_file,
                        scramble=False, seed_scramble=None, get_new_seed=True, allow_load_connected_klc=False,
                        connected_statistic_data_filename_postfix=Connected_statistic_data_filename_postfix,
                        print_message_to_history=True,
                        search_history_targets_file=Search_history_targets_file):

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
        :param min_quarters_to_process: Minimal number of quarters that is required to process the data. If
            there are fewer quarters, the search will stop.
        :type min_quarters_to_process: int
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
        :param threshold_g_crossing_list_file:
        :type threshold_g_crossing_list_file:
        :param threshold_g_crossing_list_filename:
        :type threshold_g_crossing_list_filename:
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
        :param save_multiple_triggers:
        :type save_multiple_triggers:
        :param threshold_crossing_list_multiple_triggers_filename:
        :type threshold_crossing_list_multiple_triggers_filename:
        :param threshold_g_crossing_list_multiple_triggers_filename:
        :type threshold_g_crossing_list_multiple_triggers_filename:
        :param clean_problems_file: Whether to clean before the search the file that reports the exceptions
            and the problems of the search
        :type clean_problems_file: bool
        :param stop_if_bad: Whether to stop the search if the data is too poor quality or too short
        :type stop_if_bad: bool
        :param scramble:  Whether to time-slide the connected data before the periodicity search
        :type scramble: bool
        :param seed_scramble:
        :type seed_scramble: int or None
        :param get_new_seed: Whether to get a new random seed for scrambling the data.
        :type get_new_seed: bool
        :param allow_load_connected_klc: Whether to allow loading the connected klc from the file and not calculating it
        :type allow_load_connected_klc: bool
        :param seeds_scrambling_run_file: Where to save the random seed if scrambling was performed
        :type seeds_scrambling_run_file: str or None
        :param connected_statistic_data_filename_postfix:
        :type connected_statistic_data_filename_postfix:
        :param print_message_to_history:
        :type print_message_to_history:
        :param search_history_targets_file:
        :type search_history_targets_file:

        Parameters
        ----------
        seed_scramble :
        """
        if clean_problems_file:
            self.clean_problems_file()

        try:
            file_path_connected_data = af.get_saving_file_path(output_location=self.output_location, target_id=self.target_id,
                                                               filename_postfix=connected_statistic_data_filename_postfix,
                                                               file_format='npy')
            if allow_load_connected_klc and os.path.exists(file_path_connected_data) and os.path.getsize(file_path_connected_data) > 0:
                self.initialize_kepler_light_curve()
                if len(self.klc.quarters) < min_quarters_to_process:
                    self.is_too_short = True  # too few quarters to search for a 50-day period
                    self.is_good_for_processing = False
                else:
                    self.is_too_short = False
                    self.bad_fft_check()  # Made on orig_fluxes
                if (not self.is_too_short) and (not self.is_bad_fft or not stop_if_bad):
                    self.get_connected_klc(reinitialize=True, mask_koi_planet=mask_koi_planet,
                                           save_connected_stats=False, save_conditioned_data=False)
                # Warning: not all the quality checks are accessible in it
            else:
                self.run_ses_search(save_conditioned_data=save_conditioned_data,
                                    min_quarters_to_process=min_quarters_to_process,
                                    conditioned_data_filename_postfix=conditioned_data_filename_postfix,
                                    save_no_veto_no_score_cor_stats=save_no_veto_no_score_cor_stats,
                                    no_veto_no_score_cor_filename_postfix=no_veto_no_score_cor_stats_filename_postfix,
                                    save_connected_stats=save_connected_stats, stop_if_bad=stop_if_bad,
                                    connected_stats_filename_postfix=connected_stats_data_filename_postfix,
                                    mask_koi_planet=mask_koi_planet,
                                    delete_full_stats=False)
        except Exception as exc:
            report_exception(exception=exc, file_to_append=self.problems_file_location,
                             title='ses_search: ', info='')
            self.ses_search_failed = True

        # If there is enough data, and if the data is good enough, then we run the periodicity search:
        if ((not self.is_too_short) and (not self.is_bad_fft or not stop_if_bad)) and not self.ses_search_failed:
            if scramble:
                np.random.seed(seed_scramble)
                if get_new_seed:
                    seed_scramble = np.random.randint(int(2 ** 28))
                    if self.target_id is not None:
                        seed_scramble = seed_scramble + int(self.target_id)
                if seeds_scrambling_run_file is not None:
                    with open(seeds_scrambling_run_file, 'a') as fil:
                        print(self.target_id, seed_scramble, time.strftime("%Y-%m-%d %H:%M:%S"), file=fil)
                self.klc.scramble_full_stats(max_hole=spip.Max_hole_pieces_scrampling, max_piece=spip.Max_piece_scrambling,
                                             min_piece=spip.Min_piece_scrambling, random_seed=seed_scramble,
                                             regenerate_pieces=False)  # we run many jobs, they will have different pieces
                self.klc.interleave_shifts(delete_fulls=False)  # do not delete fulls to then make re-scrambling

            self.run_periodicity_search(p_min=p_min, p_max=p_max, per_phases_per_chunk=per_phases_per_chunk,
                                        best_periodic_scores_filename_postfix=best_periodic_scores_filename_postfix,
                                        overlap_period_bins=overlap_period_bins)

            self.save_best_candidate(candidate_threshold=candidate_threshold,
                                     threshold_crossing_list_file=threshold_crossing_list_file,
                                     threshold_g_crossing_list_file=threshold_g_crossing_list_file,
                                     threshold_crossing_list_location=threshold_crossing_list_location,
                                     threshold_crossing_list_filename=threshold_crossing_list_filename,
                                     threshold_g_crossing_list_filename=threshold_g_crossing_list_filename,
                                     save_to_threshold_crossing_list=save_to_threshold_crossing_list,
                                     best_candidates_concat=self.best_candidates_concat)

            if save_multiple_triggers:
                self.save_all_triggers_from_per_chunks(
                    save_to_threshold_crossing_list=save_to_threshold_crossing_list,
                    threshold_crossing_list_multiple_triggers_filename=threshold_crossing_list_multiple_triggers_filename,
                    threshold_g_crossing_list_multiple_triggers_filename=threshold_g_crossing_list_multiple_triggers_filename,
                    threshold_crossing_list_location=threshold_crossing_list_location,
                    best_periodic_scores=self.best_candidates_concat)

            if (not self.is_threshold_crossing) and clean_dir_if_not_threshold_crossing:
                # delete conditioned data because we have no space on the disk!
                self.clean_output_directory(leave_periodic_stats=True)

        if not scramble:
            self.pickle_small_attributes()
            self.print_dict_file()

            if print_message_to_history:
                # Print message to the history file:
                message = ''
                if self.is_too_short:
                    message += 'is_too_short '
                elif self.is_bad_fft:
                    message += 'is_bad_fft '
                elif not self.is_good_for_processing:
                    message += 'not is_good_for_processing '
                elif self.ses_search_failed:
                    message += 'ses_search_failed '
                elif not self.is_threshold_crossing:
                    message += 'not is_threshold_crossing, score ' + str(self.best_int_score) + ', per ' + str(self.best_per)
                else:
                    message += 'is_threshold_crossing, score ' + str(self.best_int_score) + ', per ' + str(self.best_per)
                with open(search_history_targets_file, 'a') as fil:
                    print(self.target_id, message, file=fil)

    def run_one_chunk_search(self, period, overlap_period_bins=spip.P_bins_overlap,
                             min_required_nonzero_transits_number=spip.Min_required_nonzero_transits_number,
                             min_required_nonzero_transits_fraction=spip.Min_required_nonzero_transits_fraction,
                             baseline_rate_period=Baseline_rate_period,
                             saving_folder=None, allow_saving_folder_creation=True, add_saving_folder_prefix=False,
                             candidate_threshold=Score_candidate_search,
                             mask_koi_planet=True, avoid_masking_central_period=True,
                             mask_kepler_candidates=Mask_kepler_candidates,
                             mask_kepler_false_positives=Mask_kepler_false_positives,
                             mask_kepler_bright_candidates=Mask_kepler_bright_candidates,
                             veto_tail_statistics=True,
                             veto_extreme_statistics=True,
                             save_connected_stats=True,
                             connected_statistic_data_filename_postfix=Connected_statistic_data_filename_postfix,
                             add_period_to_connected_statistic_data_filename_postfix=False,
                             delete_irrelevant=True,
                             save_veto_masks=False,
                             delete_full_stats=True,
                             delete_periodic_interleaved=True,
                             save_uncorrected_periodic=False,
                             delete_non_marginalized=False,
                             save_best_periodic_scores=True,
                             delete_mes=True,
                             best_periodic_scores_filename_postfix=Best_periodic_scores_filename_postfix,
                             add_period_to_best_periodic_scores_filename_postfix=True,
                             save_to_threshold_crossing_list=True,
                             threshold_crossing_list_file=None,
                             threshold_g_crossing_list_file=None,
                             threshold_crossing_list_location=None,
                             threshold_crossing_list_filename=Threshold_crossing_list_filename,
                             threshold_g_crossing_list_filename=Threshold_g_crossing_list_filename,
                             save_multiple_triggers=False,
                             threshold_crossing_list_multiple_triggers_filename=Threshold_crossing_list_multiple_triggers_filename,
                             threshold_g_crossing_list_multiple_triggers_filename=Threshold_g_crossing_list_multiple_triggers_filename,
                             seeds_scrambling_run_file=None,
                             scramble=False, seed_scramble=None, get_new_seed=True,
                             print_no_trigger_found=False,
                             report_exceptions=True):
        """
        This method is designed to focus on one period around which no KOI will be masked. Is good for
        running on the actual KOIs.
        :param save_connected_stats:
        :type save_connected_stats:
        :param connected_statistic_data_filename_postfix:
        :type connected_statistic_data_filename_postfix:
        :param add_period_to_connected_statistic_data_filename_postfix:
        :type add_period_to_connected_statistic_data_filename_postfix:
        :param avoid_masking_central_period:
        :type avoid_masking_central_period:
        :param add_period_to_best_periodic_scores_filename_postfix:
        :type add_period_to_best_periodic_scores_filename_postfix:
        :param delete_irrelevant:
        :type delete_irrelevant:
        :param period:
        :type period:
        :param overlap_period_bins: Extend each periods chunk from the two sides by
            this number of bins for them to overlap. is needed for coherent score integral in order for
            the maximum not to be on the edge
        :type overlap_period_bins: int
        :param min_required_nonzero_transits_number: Minimal required number of valid transits for a trigger. Will be used
            in the pre-filtering before the periodicity search and in the veto_mes
        :type min_required_nonzero_transits_number: int
        :param min_required_nonzero_transits_fraction: Minimal required fraction of valid transits with respect to the total
         number of transits for a trigger. Will be used in the pre-filtering before the periodicity search and in the veto_mes
        :type min_required_nonzero_transits_fraction: float
        :param baseline_rate_period:
        :type baseline_rate_period:
        :param saving_folder:
        :type saving_folder:
        :param allow_saving_folder_creation:
        :type allow_saving_folder_creation:
        :param add_saving_folder_prefix:
        :type add_saving_folder_prefix:
        :param candidate_threshold:
        :type candidate_threshold:
        :param mask_koi_planet:
        :type mask_koi_planet:
        :param best_periodic_scores_filename_postfix:
        :type best_periodic_scores_filename_postfix:
        :param save_to_threshold_crossing_list:
        :type save_to_threshold_crossing_list:
        :param threshold_crossing_list_file:
        :type threshold_crossing_list_file:
        :param threshold_g_crossing_list_file:
        :type threshold_g_crossing_list_file:
        :param threshold_crossing_list_location:
        :type threshold_crossing_list_location:
        :param threshold_crossing_list_filename:
        :type threshold_crossing_list_filename:
        :param threshold_g_crossing_list_filename:
        :type threshold_g_crossing_list_filename:
        :param save_multiple_triggers:
        :type save_multiple_triggers:
        :param threshold_crossing_list_multiple_triggers_filename:
        :type threshold_crossing_list_multiple_triggers_filename:
        :param threshold_g_crossing_list_multiple_triggers_filename:
        :type threshold_g_crossing_list_multiple_triggers_filename:
        :param seeds_scrambling_run_file:
        :type seeds_scrambling_run_file:
        :param scramble:
        :type scramble:
        :param seed_scramble:
        :type seed_scramble: int or None
        :param get_new_seed: Whether to get a new random seed for scrambling the data.
        :type get_new_seed: bool
        """
        if saving_folder is not None:
            saving_location = af.get_correct_tid_folder(self.target_id, saving_folder,
                                                        allow_creation=allow_saving_folder_creation,
                                                        with_prefix_folder=add_saving_folder_prefix)
        else:
            saving_location = self.output_location

        self.klc = spip.KeplerLightCurve(target_id=self.target_id, get_data=True, data_location=self.data_location,
                                         output_location=self.output_location, m_1_3_r_s=self.m_1_3_r_s,
                                         simulated_red_star=False, simulated_white_star=False)

        if self.periods_in_chunks is None:
            self.set_periodicity_search_parameters(force_overwrite=False)
            periods_in_chunks = self.klc.make_periods_chunks(p_min=self.p_min, p_max=self.p_max, duty_cycle=self.duty_cycle,
                                                             per_phases_per_chunk=self.per_phases_per_chunk,
                                                             dt=self.klc.dt_days, t_mission=self.klc.t_mission)
        else:
            periods_in_chunks = self.periods_in_chunks
        closest_chunk, closest_chunk_index = ut.find_closest_chunk(periods_in_chunks, period, overlap_bins=0)
        periods_to_search = closest_chunk

        # We will not mask the planet around the period of search!
        if self.is_koi and mask_koi_planet:  # Masking koi planets
            if avoid_masking_central_period:
                avoid_period = period
            else:
                avoid_period = None
            self.mask_koi_planet(modify_self_klc=True, return_durations_transit_times=False,
                                 mask_candidates=mask_kepler_candidates,
                                 mask_false_positives=mask_kepler_false_positives,
                                 mask_bright_candidates=mask_kepler_bright_candidates,
                                 bright_candidate_mes=Kepler_bright_candidate_mes,
                                 bright_candidate_ses=Kepler_bright_candidate_ses,
                                 avoid_period=avoid_period, avoid_time_first=None, dp_avoid=0.002, dt_avoid=0.04)

        self.klc.condition_data()
        self.klc.condition_templates()
        self.klc.detect_single_transits(apply_veto=True, gaussianize=True,
                                        veto_tail_statistics=veto_tail_statistics,
                                        veto_extreme_statistics=veto_extreme_statistics)
        self.klc.connect_quarters()

        if add_period_to_connected_statistic_data_filename_postfix or avoid_masking_central_period:
            insert_after_postfix_period = str(int(np.round(period)))
        else:
            insert_after_postfix_period = None

        if save_connected_stats:
            file_path_connected_data = af.get_saving_file_path(output_location=self.output_location, target_id=self.target_id,
                                                               filename_postfix=connected_statistic_data_filename_postfix,
                                                               insert_after_postfix=None,
                                                               insert_integer_periods=insert_after_postfix_period,
                                                               file_format='npy')
            self.klc.save_connected_stats(file_path=file_path_connected_data)

        if delete_irrelevant:
            self.klc.delete_irrelevant()  # For memory
        if scramble:
            np.random.seed(seed_scramble)
            if get_new_seed:
                seed_scramble = np.random.randint(int(2 ** 30))
            if seeds_scrambling_run_file is not None:
                with open(seeds_scrambling_run_file, 'a') as fil:
                    print(self.target_id, seed_scramble, time.strftime("%Y-%m-%d %H:%M:%S"), file=fil)
            self.klc.scramble_full_stats(max_hole=spip.Max_hole_pieces_scrampling, max_piece=spip.Max_piece_scrambling,
                                         min_piece=spip.Min_piece_scrambling, random_seed=seed_scramble,
                                         regenerate_pieces=False)  # we run many jobs, they will have different pieces

        self.klc.interleave_shifts(delete_fulls=delete_full_stats)

        if self.log_baseline_int_score_rate is None:
            self.calculate_baseline_integral_score_rate(baseline_rate_period=baseline_rate_period)

        period_chunks = [periods_to_search]
        best_candidates_concat, failures = self.search_periodicity_over_chunks(
            period_chunks=period_chunks,
            force_interleave_shifts=False,
            delete_periodic_interleaved=delete_periodic_interleaved,
            save_uncorrected_periodic=save_uncorrected_periodic,
            delete_non_marginalized=delete_non_marginalized,
            delete_mes=delete_mes,
            min_required_nonzero_transits_number=min_required_nonzero_transits_number,
            min_required_nonzero_transits_fraction=min_required_nonzero_transits_fraction,
            overlap_period_bins=overlap_period_bins,
            report_exceptions=report_exceptions, exception_info='',
            problem_description='One-chunk search, period=' + str(period) + ': ',
            return_failures=True, save_search_time=False, save_veto_masks=save_veto_masks,
            print_no_trigger_found=print_no_trigger_found)

        if save_best_periodic_scores:
            if add_period_to_best_periodic_scores_filename_postfix:
                insert_after_postfix_period = str(int(np.round(period)))
            else:
                insert_after_postfix_period = None
            file_path = af.get_saving_file_path(output_location=saving_location, target_id=self.target_id,
                                                filename_postfix=best_periodic_scores_filename_postfix,
                                                insert_after_postfix=None,
                                                insert_integer_periods=insert_after_postfix_period,
                                                file_format='npy')
            np.save(file_path, np.array(best_candidates_concat, dtype=object), allow_pickle=True)

        self.save_best_candidate(candidate_threshold=candidate_threshold,
                                 threshold_crossing_list_file=threshold_crossing_list_file,
                                 threshold_g_crossing_list_file=threshold_g_crossing_list_file,
                                 threshold_crossing_list_location=threshold_crossing_list_location,
                                 threshold_crossing_list_filename=threshold_crossing_list_filename,
                                 threshold_g_crossing_list_filename=threshold_g_crossing_list_filename,
                                 save_to_threshold_crossing_list=save_to_threshold_crossing_list,
                                 best_candidates_concat=best_candidates_concat)

        if save_multiple_triggers:
            self.save_all_triggers_from_per_chunks(
                save_to_threshold_crossing_list=save_to_threshold_crossing_list,
                threshold_crossing_list_multiple_triggers_filename=threshold_crossing_list_multiple_triggers_filename,
                threshold_g_crossing_list_multiple_triggers_filename=threshold_g_crossing_list_multiple_triggers_filename,
                threshold_crossing_list_location=threshold_crossing_list_location,
                best_periodic_scores=best_candidates_concat)

    def run_timeslides(self, n_slides=N_slides_chunk, central_period=None, mask_koi_planet=True,
                       n_chunks_around_best=1, per_wing_slides=None,
                       per_wing_slides_relative=Per_wing_injection_and_slides_relative, use_period_chunks=False,
                       use_shuffling_pieces=True, only_if_threshold_crossing=False, overlap_period_bins=spip.P_bins_overlap,
                       slides_res_filename_postfix=Slides_res_filename_postfix,
                       min_required_nonzero_transits_number=spip.Min_required_nonzero_transits_number,
                       min_required_nonzero_transits_fraction=spip.Min_required_nonzero_transits_fraction,
                       clean_trigger=False, trigger_ftt=None, trigger_period=None,
                       allow_loading_data=True, avoid_masking_central_period=False,
                       file_path_connected_stats=None,
                       seed_scramble=None, get_new_seed=True):
        """
        Shuffles the full statistics (see inside the methods how it is done), running the periodicity search
        on the shuffled SES in the specified period range, and saving the best scores.
        The timeslides are usually run to estimate the background density in the location of the trigger from
        the main search, therefore the periods of the timeslides search are close to the best period from the
        main search. This method selects the adjacent periods and runs the shuffled search for them.
        There are three ways to organize the period grid:
        - Just run on one grid
        - Split the grid into chunks
        - Select the corresponding chunks from the chunked grid of the main search

        :param file_path_connected_stats:
        :type file_path_connected_stats:
        :param allow_loading_data:
        :type allow_loading_data:
        :param avoid_masking_central_period:
        :type avoid_masking_central_period:
        :param n_slides: Number of times to shuffle and run the periodicity search
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
            to build the period grid. Meaning that the central period will be multiplied by this number to get the
            value in days. That is, the size of the wing will be proportional to the central period.
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
        :param min_required_nonzero_transits_number: Minimal required number of valid transits for a trigger. Will be used
            in the pre-filtering before the periodicity search and in the veto_mes
        :type min_required_nonzero_transits_number: int
        :param min_required_nonzero_transits_fraction: Minimal required fraction of valid transits with respect to the total
         number of transits for a trigger. Will be used in the pre-filtering before the periodicity search and in the veto_mes
        :type min_required_nonzero_transits_fraction: float
        :param clean_trigger: Whether to mask the trigger before doing the timeslides
        :type clean_trigger: bool
        :param trigger_ftt: The first transit time of the trigger to be masked if `clean_trigger` is True
        :type trigger_ftt: float or None
        :param trigger_period: The period of the trigger to be masked if `clean_trigger` is True
        :type trigger_period: float or None
        :param seed_scramble
        :type seed_scramble: int or None
        :param get_new_seed
        :type get_new_seed: bool
        """
        if self.is_threshold_crossing or not only_if_threshold_crossing:
            if central_period is None:
                period = self.best_per
            else:
                period = central_period

            if avoid_masking_central_period:
                period_avoid_masking = period
            else:
                period_avoid_masking = None
            self.get_connected_klc(reinitialize=True, allow_loading=allow_loading_data,
                                   file_path_connected_data=file_path_connected_stats,
                                   mask_koi_planet=mask_koi_planet, period_avoid_masking=period_avoid_masking, ftt_avoid_masking=None,
                                   save_connected_stats=False, save_conditioned_data=False)
            # -> if the file did not exist, the connected stats will be saved in order not to reconnect again
            # If we want to run the timeslides after masking the trigger:
            if clean_trigger:
                if trigger_ftt is None:
                    trigger_ftt = self.best_ftt
                if trigger_period is None:
                    trigger_period = period

                masked_dur = 1.5 * tm.Durations[-1]
                self.klc.mask_ephemeris(period=trigger_period, durations=masked_dur, transit_times=trigger_ftt,
                                        absolute_phase=None, time_edges_to_mask=None)
                self.klc.mask_ephemeris_connected_quarters(period=trigger_period, durations=masked_dur, transit_times=trigger_ftt,
                                                           absolute_phase=None, time_edges_to_mask=None)

            self.klc.delete_irrelevant()  # For memory
            # In case some parameters were not set, replace them with the defaults:
            # if assert_main_search_periods:
            #     assert (self.p_min is not None) and (self.p_max is not None), 'Minimal and maximal periods were not set in the class!'
            self.set_periodicity_search_parameters(force_overwrite=False)

            if n_chunks_around_best is not None:
                if self.periods_in_chunks is None:
                    periods_in_chunks = self.klc.make_periods_chunks(p_min=self.p_min, p_max=self.p_max, duty_cycle=self.duty_cycle,
                                                                     per_phases_per_chunk=self.per_phases_per_chunk,
                                                                     dt=self.klc.dt_days, t_mission=self.klc.t_mission)
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
                self.slides_period_range = (period_chunks[0][0], period_chunks[-1][-1])
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

            if self.log_baseline_int_score_rate is None:
                self.calculate_baseline_integral_score_rate()

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
            seeds = []
            for i_slides in range(n_slides):
                np.random.seed(seed_scramble)
                if get_new_seed:
                    seed_scramble = np.random.randint(int(2 ** 30))
                if not use_shuffling_pieces:
                    self.klc.time_slides(random_seed=seed_scramble)
                    self.klc.connect_quarters()
                    self.klc.interleave_shifts(delete_fulls=True)
                else:
                    self.klc.scramble_full_stats(max_hole=spip.Max_hole_pieces_scrampling, max_piece=spip.Max_piece_scrambling,
                                                 min_piece=spip.Min_piece_scrambling, random_seed=seed_scramble,
                                                 regenerate_pieces=False)  # we run many jobs, they will have different pieces
                    self.klc.interleave_shifts(delete_fulls=False)  # do not delete fulls to then make re-scrambling

                if clean_trigger:
                    prob_descr = 'Slides: '
                else:
                    prob_descr = 'Slides clean: '
                best_candidates_concat, failures = self.search_periodicity_over_chunks(
                    period_chunks=period_chunks,
                    force_interleave_shifts=True,
                    min_required_nonzero_transits_number=min_required_nonzero_transits_number,
                    min_required_nonzero_transits_fraction=min_required_nonzero_transits_fraction,
                    overlap_period_bins=overlap_period_bins,
                    report_exceptions=True, problem_description=prob_descr, exception_info='seed=' + str(seed_scramble),
                    return_failures=True, save_search_time=False, save_veto_masks=False)

                if len(best_candidates_concat) > 0:
                    if len(best_candidates_concat) == 1:
                        best_periodic_scores_data = best_candidates_concat[0]
                    else:
                        self.find_best_trigger_from_per_chunks(best_candidates_concat=best_candidates_concat)
                        best_periodic_scores_data = best_candidates_concat[self.best_best_candidates_concat_index]

                    seeds.append(seed_scramble)
                    for list_full, list_full_g, s_ind in [[stats, stats_g, spip.Best_per_results_index_stats],
                                                          [pers, pers_g, spip.Best_per_results_index_per],
                                                          [first_times, first_times_g, spip.Best_per_results_index_ftt],
                                                          [gauss_stats, gauss_stats_g, spip.Best_per_results_index_stats_gauss],
                                                          [int_scores, int_scores_g, spip.Best_per_results_index_best_integral]
                                                          ]:
                        list_full.append(best_periodic_scores_data[s_ind][0])
                        list_full_g.append(best_periodic_scores_data[s_ind][1])

            new_res_array = np.vstack((stats, pers, first_times, gauss_stats, int_scores,
                                       stats_g, pers_g, first_times_g, gauss_stats_g, int_scores_g, seeds)).transpose()

            file_path = af.get_saving_file_path(output_location=self.output_location, target_id=self.target_id,
                                                filename_postfix=slides_res_filename_postfix,
                                                insert_after_postfix=None,
                                                insert_integer_periods=str(int(np.round(central_period))),
                                                file_format='npy')
            try:
                with Locker(file_path):
                    if os.path.exists(file_path):
                        hist_slides = np.load(file_path, allow_pickle=True)
                        hist_slides = np.vstack((hist_slides, new_res_array))
                    else:
                        hist_slides = new_res_array
                    n_slides = len(hist_slides)
                    np.save(file_path, hist_slides, allow_pickle=True)
                    if clean_trigger:
                        with open(os.path.join(self.output_location, 'number_of_clean_slides.txt'), 'w') as fil1:
                            print(n_slides, file=fil1)
                    else:
                        with open(os.path.join(self.output_location, 'number_of_slides.txt'), 'w') as fil1:
                            print(n_slides, file=fil1)
            except Exception as exc:
                if clean_trigger:
                    report_exception(exception=exc, file_to_append=self.problems_file_location,
                                     title='saving clean timeslides', info='')
                else:
                    report_exception(exception=exc, file_to_append=self.problems_file_location,
                                     title='saving timeslides', info='')

    def load_rel_wf_fds(self, conditioned_data_file_path=None,
                        conditioned_data_filename_postfix=spip.Conditioned_data_filename_postfix):
        """
        Load quarters-mean whitening filter in Fourier domain in units relative to the mean flux level.
        In other words, part of flux in time-domain that is taken by this or that component of noise.
        :param conditioned_data_file_path:
        :type conditioned_data_file_path:
        :param conditioned_data_filename_postfix: ext added to the target ID in the name of the file saving conditioned data.
            See naming notations.
        :type conditioned_data_filename_postfix: str
        """
        path = self.output_location
        if conditioned_data_file_path is None:
            conditioned_data_file_path = af.get_saving_file_path(output_location=path, target_id=self.target_id,
                                                                 filename_postfix=conditioned_data_filename_postfix, file_format='npy')
        conditioned_data = np.load(conditioned_data_file_path, allow_pickle=True)
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
            central_period=None, use_special_per_wing=False, assert_main_search_periods=True,
            per_wing_injection_relative=Per_wing_injection_and_slides_relative,
            injection_parameters_filename_postfix=Injection_parameters_filename_postfix,
            conditioned_data_file_path=None,
            conditioned_data_filename_postfix=spip.Conditioned_data_filename_postfix,
            n_per=20,
            n_ecc=60,  # Probability to transit at a fixed eccentricity depends strongly on eccentricity
            n_w=20,
            n_cos_incl=1000,  # This divides the full circle  # should be at least few semi-major axes
            n_rp=30,
            u=None,
            snr_min_injected=4.,
            snr_max_1_injected=8,
            snr_max_full_injected=None,
            rp_generation=0.02):  # Planetary radius used to generate the flux (we neglect shape dependence on radius)
        if self.stellar_mass is None or self.stellar_radius is None or np.isnan(self.stellar_mass) or np.isnan(self.stellar_radius):
            self.failures += 'Unable to generate correct injections because of invalid stellar radius or mass; '

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
            focus_period = self.best_per
        else:
            focus_period = central_period

        if assert_main_search_periods:
            assert (self.p_min is not None) and (self.p_max is not None), 'Minimal and maximal periods were not set in the class!'
        self.set_periodicity_search_parameters(force_overwrite=False)
        if not use_special_per_wing:
            if self.periods_in_chunks is None:
                if not hasattr(self, 'klc') or self.klc is None:
                    self.initialize_kepler_light_curve()
                periods_in_chunks = self.klc.make_periods_chunks(p_min=self.p_min, p_max=self.p_max, duty_cycle=self.duty_cycle,
                                                                 per_phases_per_chunk=self.per_phases_per_chunk,
                                                                 dt=self.klc.dt_days, t_mission=self.klc.t_mission)
            else:
                periods_in_chunks = self.periods_in_chunks
            closest_chunk, closest_chunk_index = ut.find_closest_chunk(periods_in_chunks, focus_period, overlap_bins=0)
            p_min_inj, p_max_inj = closest_chunk[0], closest_chunk[-1]
        else:
            p_max_inj = focus_period * (1 + per_wing_injection_relative)
            p_min_inj = focus_period * (1 - per_wing_injection_relative)

        # ################################################################################################
        # OCCURRENCE RATES
        # All the things with the occurrence rates and densities:
        planets_of_radii_in_per_range_one_star, log2_rp_re_edges = tm.get_occurrence_rate_per_chunk(period_min=p_min_inj,
                                                                                                    period_max=p_max_inj)
        units_of_re_to_units_of_r_star_factor = 1 / (self.stellar_radius * R_sun_R_e)
        log2_rp_edges = np.log2(units_of_re_to_units_of_r_star_factor * 2. ** log2_rp_re_edges)
        self.log2_rp_edges = log2_rp_edges

        self.prior_planets_in_period_range_one_star = np.sum(planets_of_radii_in_per_range_one_star)
        self.prior_radii_distribution = planets_of_radii_in_per_range_one_star / np.sum(planets_of_radii_in_per_range_one_star)
        self.injections_period_range = (p_min_inj, p_max_inj)

        rp_numbers = (np.ceil(planets_of_radii_in_per_range_one_star /
                              np.sum(planets_of_radii_in_per_range_one_star) * n_rp)).astype(int)
        rp_numbers_total = np.sum(rp_numbers)  # how many of each radius to generate

        t = tm.time(t_tot=(256 + 1) * tm.dt_lc, parity='odd')[1:]  # Making an even points number but odd parity

        # ################################################################################################
        # GENERATING PLANET PARAMETERS, SEE WHO TRANSITS AND WHO HAS REASONABLE RADII
        wf_fds = self.load_rel_wf_fds(conditioned_data_file_path=conditioned_data_file_path,
                                      conditioned_data_filename_postfix=conditioned_data_filename_postfix)
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

        overshoot_factor = 10.
        n_min_size = n_ecc * n_w * n_rp * (n_cos_incl / 2)
        size_overshoot_factor = max(2.1 * overshoot_factor * n_injections, n_min_size) / n_injections
        # injection_parameter_sets = []
        self.injection_par_list = np.zeros((int(np.ceil(n_injections * size_overshoot_factor)), 10), dtype=np.float64)
        count_orbit_simulations = 0
        count_orbit_transiting = 0
        count_orbit_transiting_relevant_radii = 0
        n_per_actual = 0
        for period in np.concatenate((np.array([focus_period]), np.random.uniform(p_min_inj, p_max_inj, n_per-1))):
            # Semi major axis of the orbit in units of R_star
            # a = (G / (4 * np.pi ** 2) * (period * 24 * 60 * 60) ** 2) ** (1 / 3) * self.m_1_3_r_s * (M_sun ** (1 / 3) / R_sun)
            a = get_a_from_p(period=period, m_1_3_r_s=self.m_1_3_r_s)
            e_max = min(1 - 2 / a, E_crit)
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
                        # dur = tm.templ_dur_hmax_fun(flux_obs) * tm.dt_lc  # This used to be in run_11, but this
                        # is not the function which was used in the template bank generation, so it is unfair
                        dur = tm.templ_dur_fun(flux_obs) * tm.dt_lc
                        count_orbit_simulations += 1
                        if np.linalg.norm(flux_obs) > 0:
                            count_orbit_transiting += 1
                            depth_factor = np.nanmax(flux_obs) / rp_generation ** 2
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
                            # count_orbit_transiting_relevant_radii += len(rps)  # will be divided by rp_numbers_total
                            for rp in rps:
                                phase_inj = np.random.uniform(0, period)
                                self.injection_par_list[count_orbit_transiting_relevant_radii
                                                        ] = [period, a, inc, ecc, w, u[0], u[1], rp, dur, phase_inj]
                                count_orbit_transiting_relevant_radii += 1
                                # injection_parameter_sets.append([period, a, inc, ecc, w, u[0], u[1], rp, dur, phase_inj])
            n_per_actual += 1
            if count_orbit_transiting_relevant_radii > n_injections * overshoot_factor:
                break

        self.injection_par_list = self.injection_par_list[:count_orbit_transiting_relevant_radii]
        # Probability to transit given a planet of a given period:
        self.transit_probability = count_orbit_transiting / n_per_actual * transit_prob_measure
        # Number of planets per star in the considered period range with any radius:
        self.transiting_planets_of_radii_in_per_range_one_star = self.prior_planets_in_period_range_one_star * self.transit_probability
        # Probability to transit and be in the relevant snr range given a planet of a given period:
        self.transit_and_range_probability = count_orbit_transiting_relevant_radii / rp_numbers_total / n_per_actual * transit_prob_measure
        # Number of planets per star in the considered period range with radii that provide relevant SNR:
        self.transiting_planets_in_relevant_range = self.prior_planets_in_period_range_one_star * self.transit_and_range_probability
        selection_indices = np.random.choice(np.arange(len(self.injection_par_list)), n_injections,
                                             replace=True if len(self.injection_par_list) < n_injections else False)
        # self.injection_par_list = list(np.array(injection_parameter_sets, dtype=object)[selection_indices])
        self.injection_par_list = self.injection_par_list[selection_indices]
        self.save_injection_parameters(file_path=file_path, injection_parameters_filename_postfix=None)
        self.pickle_small_attributes()

    def save_injection_parameters(self, file_path=None,
                                  injection_parameters_filename_postfix=Injection_parameters_filename_postfix):
        """
        Saves the generated injection parameters to the specified file to then load them and use the injection-recovery.
        :param file_path:
        :type file_path:
        :param injection_parameters_filename_postfix: Text added to the target ID in the name of the file saving
            the injection parameters. See naming notations.
        :type injection_parameters_filename_postfix: str
        """
        # We wanted to just save parameters inside the generate_injection_parameters function because anyway we do not create attribute
        # of injection parameters sets because we will probably not use it simultaneously. But we decided that maybe we will.
        if file_path is None:
            file_path = af.get_saving_file_path(output_location=self.output_location, target_id=self.target_id,
                                                filename_postfix=injection_parameters_filename_postfix, file_format='txt')
        with open(file_path, 'w') as fil1:
            print('# period[days], a/R, inc[deg 90 or less], ecc, w[deg], u_0, u_1, rp/R, dur_hmax_days', file=fil1)
            for i in range(len(self.injection_par_list)):
                print(*self.injection_par_list[i], file=fil1)

    def load_injection_parameters(self, injection_parameters_index=0,
                                  injection_parameters_filename_postfix=Injection_parameters_filename_postfix,
                                  central_period_for_file=None, insert_after_filename_postfix=None):
        """
        Loads a specific line of the list of the injection parameters previously generated with
        `self.generate_injection_parameters()` and saved with `self.save_injection_parameters()`.

        :param injection_parameters_index: Index of the line from the list to load
        :type injection_parameters_index: int
        :param injection_parameters_filename_postfix: Text added to the target ID in the name of the file saving
            the injection parameters. See naming notations.
        :type injection_parameters_filename_postfix: str
        :param central_period_for_file: central period for injection parameters to indicate in the filename
        :type central_period_for_file: float, str, int, or None
        :param insert_after_filename_postfix: Text to insert after `injection_parameters_filename_postfix`, unrelated to period
        :type insert_after_filename_postfix: str or None
        """
        file_path = af.get_saving_file_path(output_location=self.output_location, target_id=self.target_id,
                                            filename_postfix=injection_parameters_filename_postfix,
                                            insert_integer_periods=central_period_for_file,
                                            insert_after_postfix=insert_after_filename_postfix, file_format='txt')

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
        phase_inj = float(injection_params[9])
        # ' a/R, inc[deg from 90 down], ecc, w[deg], u_0, u_1, rp/R, dur_hmax_days'
        return per_inj, a_inj, inc_inj, ecc_inj, w_inj, u1_inj, u2_inj, rp_inj, dur_inj, phase_inj

    def run_injection_search(self, injection_load_parameters_index=0,
                             injections_filename_postfix=Injections_filename_postfix,
                             one_chunk_around_injection=True,
                             per_wing_detection=Per_wing_injection_detection,
                             overlap_period_bins=spip.P_bins_overlap,
                             period_avoid_masking=None,
                             mask_kepler_candidates=Mask_kepler_candidates,
                             mask_kepler_false_positives=Mask_kepler_false_positives,
                             mask_kepler_bright_candidates=Mask_kepler_bright_candidates,
                             min_required_nonzero_transits_number=spip.Min_required_nonzero_transits_number,
                             min_required_nonzero_transits_fraction=spip.Min_required_nonzero_transits_fraction,
                             baseline_rate_period=Baseline_rate_period):

        """
        Loads a specific line of the list of the injection parameters previously generated with
        `self.generate_injection_parameters()` and saved with `self.save_injection_parameters()`.
        Masks the trigger found in the main search, as well as possible KOIs if there are any for this target.
        Injects the corresponding planet of the specified parameters to data and runs the entire search from scratch.
        In the periodicity search, uses the period grid piece around the trigger.
        The goal is just to check whether the phase is recovered and what is the SNR. The look-elsewhere effect
        etc. are not relevant.
        :param period_avoid_masking:
        :type period_avoid_masking:
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
        :param min_required_nonzero_transits_number: Minimal required number of valid transits for a trigger. Will be used
            in the pre-filtering before the periodicity search and in the veto_mes
        :type min_required_nonzero_transits_number: int
        :param min_required_nonzero_transits_fraction: Minimal required fraction of valid transits with respect to the total
         number of transits for a trigger. Will be used in the pre-filtering before the periodicity search and in the veto_mes
        :type min_required_nonzero_transits_fraction: float
        :param baseline_rate_period:
        :type baseline_rate_period:
        """
        per_inj, a_inj, inc_inj, ecc_inj, w_inj, u1_inj, u2_inj, rp_inj, dur_inj, phase_inj = self.load_injection_parameters(
            injection_parameters_index=injection_load_parameters_index)
        # This period range is for injection, not for search:

        tms, flxs = af.get_data_fun(target_id=self.target_id, location=self.data_location)
        first_transit_time_inj = np.nanmin(tms[0]) + phase_inj
        flxs_inj = sim.inject_planet_to_data(times=tms, fluxes=flxs, period=per_inj, first_transit_center=first_transit_time_inj,
                                             rp=rp_inj, a=a_inj, inc=inc_inj, ecc=ecc_inj, w=w_inj, u1=u1_inj, u2=u2_inj,
                                             supersample_factor=20, return_template=False)

        self.klc = spip.KeplerLightCurve(target_id=self.target_id, times=tms, fluxes=flxs_inj, get_data=False,
                                         data_location=self.data_location, m_1_3_r_s=self.m_1_3_r_s,
                                         output_location=self.output_location,
                                         simulated_red_star=(self.simulated_star == 'red'),
                                         simulated_white_star=(self.simulated_star == 'white'))

        if one_chunk_around_injection:
            if self.periods_in_chunks is None:
                periods_in_chunks = self.klc.make_periods_chunks(p_min=self.p_min, p_max=self.p_max, duty_cycle=self.duty_cycle,
                                                                 per_phases_per_chunk=self.per_phases_per_chunk,
                                                                 dt=self.klc.dt_days, t_mission=self.klc.t_mission)
            else:
                periods_in_chunks = self.periods_in_chunks
            closest_chunk, closest_chunk_index = ut.find_closest_chunk(periods_in_chunks, per_inj, overlap_bins=0)
            periods_to_search = closest_chunk
        else:
            p_max = per_inj + per_wing_detection
            p_min = per_inj - per_wing_detection
            periods_to_search = self.klc.make_periods_array(p_min=p_min, p_max=p_max, dp=None, duty_cycle=self.duty_cycle, t_mission=None)

        if self.is_koi:  # Masking koi planets
            self.mask_koi_planet(modify_self_klc=True, return_durations_transit_times=False,
                                 mask_candidates=mask_kepler_candidates,
                                 mask_false_positives=mask_kepler_false_positives,
                                 mask_bright_candidates=mask_kepler_bright_candidates,
                                 bright_candidate_mes=Kepler_bright_candidate_mes,
                                 bright_candidate_ses=Kepler_bright_candidate_ses,
                                 avoid_period=float(period_avoid_masking) if period_avoid_masking is not None else None,
                                 avoid_time_first=None, dp_avoid=0.002, dt_avoid=0.04)

        masked_dur = 1.5 * tm.Durations[-1]
        self.klc.mask_ephemeris(period=self.best_per, durations=masked_dur, transit_times=self.best_ftt)
        # Mask the suspected real transit. Warning, can kill some of injected signal
        self.klc.condition_data()
        self.klc.condition_templates()
        self.klc.detect_single_transits(apply_veto=True, gaussianize=True)
        self.klc.connect_quarters()
        self.klc.delete_irrelevant()  # For memory
        self.klc.interleave_shifts(delete_fulls=True)

        if self.log_baseline_int_score_rate is None:
            self.calculate_baseline_integral_score_rate(baseline_rate_period=baseline_rate_period)

        period_chunks = [periods_to_search]
        best_candidates_concat, failures = self.search_periodicity_over_chunks(
            period_chunks=period_chunks,
            force_interleave_shifts=False,
            min_required_nonzero_transits_number=min_required_nonzero_transits_number,
            min_required_nonzero_transits_fraction=min_required_nonzero_transits_fraction,
            overlap_period_bins=overlap_period_bins,
            report_exceptions=True, problem_description='Injections, index=' + str(injection_load_parameters_index) + ': ',
            exception_info='phase_inj=' + str(phase_inj),
            return_failures=True, save_search_time=False, save_veto_masks=False)

        if len(best_candidates_concat) > 0:
            if len(best_candidates_concat) == 1:
                best_periodic_scores_data = best_candidates_concat[0]
            else:
                self.find_best_trigger_from_per_chunks(best_candidates_concat=best_candidates_concat)
                best_periodic_scores_data = best_candidates_concat[self.best_best_candidates_concat_index]

            retrieved_stat = best_periodic_scores_data[spip.Best_per_results_index_stats][0]
            retrieved_period = best_periodic_scores_data[spip.Best_per_results_index_per][0]
            retrieved_first_transit_time = best_periodic_scores_data[spip.Best_per_results_index_ftt][0]
            retrieved_gauss_stat = best_periodic_scores_data[spip.Best_per_results_index_stats_gauss][0]
            retrieved_int_score = best_periodic_scores_data[spip.Best_per_results_index_best_integral][0]
            retrieved_temp = best_periodic_scores_data[spip.Best_per_results_index_template][0]
            retrieved_g_stat = best_periodic_scores_data[spip.Best_per_results_index_stats][1]
            retrieved_g_period = best_periodic_scores_data[spip.Best_per_results_index_per][1]
            retrieved_g_first_transit_time = best_periodic_scores_data[spip.Best_per_results_index_ftt][1]
            retrieved_g_gauss_stat = best_periodic_scores_data[spip.Best_per_results_index_stats_gauss][1]
            retrieved_g_int_score = best_periodic_scores_data[spip.Best_per_results_index_best_integral][1]
            retrieved_g_temp = best_periodic_scores_data[spip.Best_per_results_index_template][1]

            file_path = af.get_saving_file_path(output_location=self.output_location, target_id=self.target_id,
                                                filename_postfix=injections_filename_postfix,
                                                insert_after_postfix=None,
                                                insert_integer_periods=None, file_format='txt')
            try:
                with Locker(file_path):
                    with open(file_path, 'a') as fil1:
                        line_inj_par = [per_inj, a_inj, inc_inj, ecc_inj, w_inj, u1_inj, u2_inj, rp_inj, dur_inj, first_transit_time_inj]
                        line_inj_par = [str(e) for e in line_inj_par]
                        line_inj_par = ' '.join(line_inj_par)
                        line_trigger = tc.line_from_trigger_fields(target_id=self.target_id, integral_score=retrieved_int_score,
                                                                   stat=retrieved_stat, per=retrieved_period,
                                                                   ftt=retrieved_first_transit_time, gauss_stat=retrieved_gauss_stat,
                                                                   template=retrieved_temp, sep=' ')
                        line_g_trigger = tc.line_from_trigger_fields(target_id=self.target_id, integral_score=retrieved_g_int_score,
                                                                     stat=retrieved_g_stat, per=retrieved_g_period,
                                                                     ftt=retrieved_g_first_transit_time, gauss_stat=retrieved_g_gauss_stat,
                                                                     template=retrieved_g_temp, sep=' ')
                        print(line_inj_par + ' ' + line_trigger + ' ' + line_g_trigger, file=fil1)
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
            self.__dict__['retrieved_template_' + status] = []
            self.__dict__['retrieved_int_score_' + status] = []
            self.__dict__['retrieved_g_stat_' + status] = []
            self.__dict__['retrieved_g_period_' + status] = []
            self.__dict__['retrieved_g_first_transit_time_' + status] = []
            self.__dict__['retrieved_g_gauss_stat_' + status] = []
            self.__dict__['retrieved_g_int_score_' + status] = []
            self.__dict__['retrieved_g_template_' + status] = []
        if file_path is None:
            file_path = af.get_saving_file_path(output_location=self.output_location, target_id=self.target_id,
                                                filename_postfix=injections_filename_postfix,
                                                insert_integer_periods=None, file_format='txt')
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            with open(file_path, 'r') as fil1:
                inj_lines = fil1.readlines()

            for inj_line0 in inj_lines:
                inj_line = inj_line0.strip()
                inj_line = inj_line.strip('\x00')
                if (len(inj_line) > 5) and (inj_line[0] != '#'):
                    res = inj_line.split(' ')
                    n_inj_pars = 10
                    n_trigger_pars = tc.Trigger_len
                    per_inj, a_inj, inc_inj, ecc_inj, w_inj, u1_inj, u2_inj, rp_inj, dur_inj, ftt_inj = res[:n_inj_pars]
                    (_, retrieved_int_score, retrieved_stat, retrieved_period, retrieved_first_transit_time, retrieved_gauss_stat,
                     retrieved_template) = tc.trigger_fields_from_list(res[n_inj_pars: n_inj_pars + n_trigger_pars])
                    (_, retrieved_g_int_score, retrieved_g_stat, retrieved_g_period, retrieved_g_first_transit_time,
                     retrieved_g_gauss_stat, retrieved_g_template) = tc.trigger_fields_from_list(
                        res[n_inj_pars + n_trigger_pars: n_inj_pars + 2 * n_trigger_pars])
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
                    retrieved_int_score = float(retrieved_int_score)
                    retrieved_g_stat = float(retrieved_g_stat)
                    retrieved_g_period = float(retrieved_g_period)
                    retrieved_g_first_transit_time = float(retrieved_g_first_transit_time)
                    retrieved_g_gauss_stat = float(retrieved_g_gauss_stat)
                    retrieved_g_int_score = float(retrieved_g_int_score)

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
                    self.__dict__['retrieved_int_score_' + status].append(retrieved_int_score)
                    self.__dict__['retrieved_template_' + status].append(retrieved_template)
                    self.__dict__['retrieved_g_stat_' + status].append(retrieved_g_stat)
                    self.__dict__['retrieved_g_period_' + status].append(retrieved_g_period)
                    self.__dict__['retrieved_g_first_transit_time_' + status].append(retrieved_g_first_transit_time)
                    self.__dict__['retrieved_g_gauss_stat_' + status].append(retrieved_g_gauss_stat)
                    self.__dict__['retrieved_g_int_score_' + status].append(retrieved_g_int_score)
                    self.__dict__['retrieved_g_template_' + status].append(retrieved_g_template)

            self.number_of_success_injections = len(self.per_inj_success)
            self.number_of_injections = len(self.per_inj_bad) + len(self.per_inj_success)
        else:
            self.number_of_success_injections = 0
            self.number_of_injections = 0

    def retrieve_background(self, slides_res_filename_postfix=Slides_res_filename_postfix,
                            central_period=None):
        """
        Opens the file where the time-shuffling runs results were saved, retrieves these data
        and stores as attributes.
        :param slides_res_filename_postfix: Text added to the target ID in the name of the file saving
            the time-shuffled search results. See naming notations.
        :type slides_res_filename_postfix: str
        :param central_period: period around which the timeslides were done. There can be several such periods.
        :type central_period: float
        """
        if central_period is not None:
            central_period_name = str(int(np.round(central_period)))
            fil_slides = [ff for ff in os.listdir(self.output_location) if
                          ((str(self.target_id) in ff) and (slides_res_filename_postfix in ff) and ('locker' not in ff)
                           and (central_period_name in ff))]
        else:
            fil_slides = [ff for ff in os.listdir(self.output_location) if
                          ((str(self.target_id) in ff) and (slides_res_filename_postfix in ff) and ('locker' not in ff))]
        if len(fil_slides) == 0:
            raise ValueError('No timeslides file found for the requested parameters!')
        else:
            fil_slides = fil_slides[0]
        res_slides = np.load(os.path.join(self.output_location, fil_slides), allow_pickle=True)
        self.stats_slides = np.array(res_slides[:, 0], dtype=float)
        self.pers_slides = np.array(res_slides[:, 1], dtype=float)
        self.first_times_slides = np.array(res_slides[:, 2], dtype=float)
        self.gauss_stats_slides = np.array(res_slides[:, 3], dtype=float)
        self.int_scores_slides = np.array(res_slides[:, 4], dtype=float)
        self.stats_g_slides = np.array(res_slides[:, 5], dtype=float)
        self.pers_g_slides = np.array(res_slides[:, 6], dtype=float)
        self.first_times_g_slides = np.array(res_slides[:, 7], dtype=float)
        self.gauss_stats_g_slides = np.array(res_slides[:, 8], dtype=float)
        self.int_scores_g_slides = np.array(res_slides[:, 9], dtype=float)

        mask_not_nan = (~np.isnan(self.int_scores_slides)) & (~np.isnan(self.stats_slides))
        self.stats_slides = self.stats_slides[mask_not_nan]
        self.pers_slides = self.pers_slides[mask_not_nan]
        self.first_times_slides = self.first_times_slides[mask_not_nan]
        self.gauss_stats_slides = self.gauss_stats_slides[mask_not_nan]
        self.int_scores_slides = self.int_scores_slides[mask_not_nan]
        mask_not_nan = (~np.isnan(self.int_scores_g_slides)) & (~np.isnan(self.gauss_stats_g_slides))
        self.stats_g_slides = self.stats_g_slides[mask_not_nan]
        self.pers_g_slides = self.pers_g_slides[mask_not_nan]
        self.first_times_g_slides = self.first_times_g_slides[mask_not_nan]
        self.gauss_stats_g_slides = self.gauss_stats_g_slides[mask_not_nan]
        self.int_scores_g_slides = self.int_scores_g_slides[mask_not_nan]

        self.n_slides = len(res_slides)  # we save one trigger per star, so it is len.
        # -> we do it before the nans rejection because those nans just mean that the sore was too low or something. The
        # idea behind p_planet is that we conduct the experiment many times and register everything that we get, so
        # we need to count all the times, including the nan-giving ones.

    def retrieve_background_clean(self, slides_clean_res_filename_postfix=Slides_clean_res_filename_postfix,
                                  central_period=None):
        """
        Opens the file where the time-shuffling runs results were saved, retrieves these data
        and stores as attributes.
        :param slides_clean_res_filename_postfix: Text added to the target ID in the name of the file saving
            the time-shuffled search results. See naming notations.
        :type slides_clean_res_filename_postfix: str
        :param central_period: period around which the timeslides were done. There can be several such periods.
        :type central_period: float
        """
        if central_period is not None:
            central_period_name = str(int(np.round(central_period)))
            fil_slides = [ff for ff in os.listdir(self.output_location) if
                          ((str(self.target_id) in ff) and (slides_clean_res_filename_postfix in ff) and ('locker' not in ff)
                           and (central_period_name in ff))]
        else:
            fil_slides = [ff for ff in os.listdir(self.output_location) if
                          ((str(self.target_id) in ff) and (slides_clean_res_filename_postfix in ff) and ('locker' not in ff))]
        if len(fil_slides) == 0:
            raise ValueError('No timeslides file found for the requested parameters!')
        else:
            fil_slides = fil_slides[0]
        res_slides = np.load(os.path.join(self.output_location, fil_slides), allow_pickle=True)
        self.stats_slides_clean = np.array(res_slides[:, 0], dtype=float)
        self.pers_slides_clean = np.array(res_slides[:, 1], dtype=float)
        self.first_times_slides_clean = np.array(res_slides[:, 2], dtype=float)
        self.gauss_stats_slides_clean = np.array(res_slides[:, 3], dtype=float)
        self.int_scores_slides_clean = np.array(res_slides[:, 4], dtype=float)
        self.stats_g_slides_clean = np.array(res_slides[:, 5], dtype=float)
        self.pers_g_slides_clean = np.array(res_slides[:, 6], dtype=float)
        self.first_times_g_slides_clean = np.array(res_slides[:, 7], dtype=float)
        self.gauss_stats_g_slides_clean = np.array(res_slides[:, 8], dtype=float)
        self.int_scores_g_slides_clean = np.array(res_slides[:, 9], dtype=float)

        mask_not_nan = (~np.isnan(self.int_scores_slides_clean)) & (~np.isnan(self.stats_slides_clean))
        self.stats_slides_clean = self.stats_slides_clean[mask_not_nan]
        self.pers_slides_clean = self.pers_slides_clean[mask_not_nan]
        self.first_times_slides_clean = self.first_times_slides_clean[mask_not_nan]
        self.gauss_stats_slides_clean = self.gauss_stats_slides_clean[mask_not_nan]
        self.int_scores_slides_clean = self.int_scores_slides_clean[mask_not_nan]
        mask_not_nan = (~np.isnan(self.int_scores_g_slides_clean)) & (~np.isnan(self.gauss_stats_g_slides_clean))
        self.stats_g_slides_clean = self.stats_g_slides_clean[mask_not_nan]
        self.pers_g_slides_clean = self.pers_g_slides_clean[mask_not_nan]
        self.first_times_g_slides_clean = self.first_times_g_slides_clean[mask_not_nan]
        self.gauss_stats_g_slides_clean = self.gauss_stats_g_slides_clean[mask_not_nan]
        self.int_scores_g_slides_clean = self.int_scores_g_slides_clean[mask_not_nan]

        self.n_slides_clean = len(res_slides)  # we save one trigger per star, so it is len.
        # -> we do it before the nans rejection because those nans just mean that the sore was too low or something. The
        # idea behind p_planet is that we conduct the experiment many times and register everything that we get, so
        # we need to count all the times, including the nan-giving ones.

    def _get_bin_min_bin_max(self, in_gaussianized=False, use_slides=True, use_injections=True, use_clean_slides=True,
                             use_integral=True):
        """
        # Attention! Here, the trigger used is not Gaussianized!
        Gets the boundaries for the histograms of the time shuffling and the injection-recovery runs
        from the obtained statistics.
        :param in_gaussianized: Whether to look at the gaussianized statistics or not.
        :type in_gaussianized: bool
        :return:
        :rtype:
        """
        if use_integral:
            if use_injections:
                retrieved_score_success = self.retrieved_int_score_success
                retrieved_score_bad = self.retrieved_int_score_bad
            else:
                retrieved_score_success = None
                retrieved_score_bad = None
            if use_slides:
                scores_slides = self.int_scores_slides
            else:
                scores_slides = None
            if use_clean_slides:
                scores_slides_clean = self.int_scores_slides_clean
            else:
                scores_slides_clean = None
            best_score = self.best_int_score
        else:
            if in_gaussianized:
                if use_injections:
                    retrieved_score_success = self.retrieved_gauss_stat_success
                    retrieved_score_bad = self.retrieved_gauss_stat_bad
                else:
                    retrieved_score_success = None
                    retrieved_score_bad = None
                if use_slides:
                    scores_slides = self.gauss_stats_slides
                else:
                    scores_slides = None
                if use_clean_slides:
                    scores_slides_clean = self.gauss_stats_slides_clean
                else:
                    scores_slides_clean = None
                # Attention! Here, the trigger used is not Gaussianized!
                best_score = self.best_gauss_stat
            else:
                if use_injections:
                    retrieved_score_success = self.retrieved_stat_success
                    retrieved_score_bad = self.retrieved_stat_bad
                else:
                    retrieved_score_success = None
                    retrieved_score_bad = None
                if use_slides:
                    scores_slides = self.stats_slides
                else:
                    scores_slides = None
                if use_clean_slides:
                    scores_slides_clean = self.stats_slides_clean
                else:
                    scores_slides_clean = None
                best_score = self.best_stat

        # The following can happen if there are no injections but the injection retrieval function was used.
        # (If only one of them is empty, it means that maybe all the injections were good or bad)
        if ((retrieved_score_success is not None and len(retrieved_score_success) == 0) and
                (retrieved_score_bad is not None and len(retrieved_score_bad) == 0)):
            retrieved_score_success = None
            retrieved_score_bad = None

        if best_score is not None:
            trig_val = best_score
        else:
            trig_val = np.nan

        if (retrieved_score_bad is not None) and (retrieved_score_success is not None):
            conc_inj_stat = np.concatenate((np.abs(retrieved_score_success), np.abs(retrieved_score_bad)))
            inj_min = np.nanmin(conc_inj_stat[np.abs(conc_inj_stat) != np.inf])
            inj_max = np.nanmax(conc_inj_stat[np.abs(conc_inj_stat) != np.inf])
        else:
            inj_min = np.nan
            inj_max = np.nan

        if scores_slides is not None:
            slides_min = np.nanmin(scores_slides[np.abs(scores_slides) != np.inf])
            slides_max = np.nanmax(scores_slides[np.abs(scores_slides) != np.inf])
        else:
            slides_min = np.nan
            slides_max = np.nan

        if scores_slides_clean is not None:
            slides_clean_min = np.nanmin(scores_slides_clean[np.abs(scores_slides_clean) != np.inf])
            slides_clean_max = np.nanmax(scores_slides_clean[np.abs(scores_slides_clean) != np.inf])
        else:
            slides_clean_min = np.nan
            slides_clean_max = np.nan

        bin_max = np.nanmax([inj_max, slides_max, slides_clean_max, trig_val])
        bin_min = np.nanmin([inj_min, slides_min, slides_clean_min, trig_val])
        return bin_min, bin_max

    def make_histogram_bins(self, bin_width=Histogram_mes_binwid, bin_min=None, bin_max=None,
                            use_slides=True, use_injections=True):
        """
        Create the bins array over which the distributions of the background and the injection-recovery
        statistics (snr squared) will be plotted.
        :param bin_width: The width of the bin in snr squared that should be used
        :type bin_width: float
        :param bin_min: Minimal bin value. if None, will be measured
        :type bin_min: float
        :param bin_max: Maximal bin value. if None, will be measured
        :type bin_max: float
        :param use_slides:
        :type use_slides:
        :param use_injections:
        :type use_injections:
        """
        use_clean_slides = use_slides
        if use_slides:
            if self.int_scores_slides is None:
                try:
                    self.retrieve_background()
                except FileNotFoundError:
                    use_slides = False
                try:
                    self.retrieve_background_clean()
                except (ValueError, FileNotFoundError):
                    use_clean_slides = False
        if use_injections:
            if self.retrieved_int_score_success is None:
                try:
                    self.retrieve_injections()
                except FileNotFoundError:
                    use_injections = False
        if bin_min is None or bin_max is None:
            bin_min_stat, bin_max_stat = self._get_bin_min_bin_max(in_gaussianized=False, use_slides=use_slides,
                                                                   use_clean_slides=use_clean_slides, use_injections=use_injections,
                                                                   use_integral=True)
            bin_min_integr, bin_max_integr = self._get_bin_min_bin_max(in_gaussianized=False, use_slides=use_slides,
                                                                       use_clean_slides=use_clean_slides, use_injections=use_injections,
                                                                       use_integral=False)
            bin_min_gauss_stat, bin_max_gauss_stat = self._get_bin_min_bin_max(in_gaussianized=True, use_slides=use_slides,
                                                                               use_clean_slides=use_clean_slides,
                                                                               use_injections=use_injections, use_integral=True)
        else:
            bin_min_stat = bin_min
            bin_max_stat = bin_max
            bin_min_integr = bin_min
            bin_max_integr = bin_max
            bin_min_gauss_stat = bin_min
            bin_max_gauss_stat = bin_max
        self.bins_int_score_histogram = np.arange(bin_min_integr, bin_max_integr, bin_width)
        self.bins_stat_histogram = np.arange(bin_min_stat, bin_max_stat, bin_width)
        self.bins_gauss_stat_histogram = np.arange(bin_min_gauss_stat, bin_max_gauss_stat, bin_width)

    def make_background_histogram(self, bins=None, bin_min=None, bin_max=None, bin_width=Histogram_mes_binwid, central_period=None,
                                  slides_res_filename_postfix=Slides_res_filename_postfix,
                                  slides_clean_res_filename_postfix=Slides_clean_res_filename_postfix,
                                  use_integral=True, calculate_extrapolation=True,
                                  extrapolation_start_score=Extrapolation_start_score):
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
        :param central_period: The period around which we want the timeslides. There may be different instances fo the slides
        :type central_period: float
        :param slides_res_filename_postfix: The postfix of the time-slide file name, it does not contain period.
        :type slides_res_filename_postfix: str
        :param slides_clean_res_filename_postfix: The postfix of the clean time-slide file name, it does not contain period.
        :type slides_clean_res_filename_postfix: str
        :param use_integral: Whether to use integral score or statistic
        :type use_integral: bool
        :param calculate_extrapolation:
        :type calculate_extrapolation:
        :param extrapolation_start_score:
        :type extrapolation_start_score:
        """
        if self.int_scores_slides is None:
            self.retrieve_background(central_period=central_period, slides_res_filename_postfix=slides_res_filename_postfix)
        use_clean = True
        if self.int_scores_slides_clean is None:
            try:
                self.retrieve_background_clean(central_period=central_period,
                                               slides_clean_res_filename_postfix=slides_clean_res_filename_postfix)
            except (ValueError, FileNotFoundError):
                use_clean = False

        # scores_slides, bins_scores_histogram, n_slides, in_gaussianized, in_integral, in_clean:
        if use_integral:
            if use_clean:
                list_options = [
                    [self.int_scores_slides, self.bins_int_score_histogram_bg, self.n_slides, False, True, False],
                    [self.int_scores_slides_clean, self.bins_int_score_histogram_bg, self.n_slides_clean, False, True, True]]
            else:
                list_options = [
                    [self.int_scores_slides, self.bins_int_score_histogram_bg, self.n_slides, False, True, False]]
        else:
            if use_clean:
                list_options = [
                    [self.stats_slides, self.bins_stat_histogram, self.n_slides, False, False, False],
                    [self.stats_slides_clean, self.bins_stat_histogram, self.n_slides_clean, False, False, True],
                    [self.gauss_stats_slides, self.bins_gauss_stat_histogram, self.n_slides, True, False, False],
                    [self.gauss_stats_slides_clean, self.bins_gauss_stat_histogram, self.n_slides_clean, True, False, True]]
            else:
                list_options = [
                    [self.stats_slides, self.bins_stat_histogram, self.n_slides, False, False, False],
                    [self.gauss_stats_slides, self.bins_gauss_stat_histogram, self.n_slides, True, False, False]]

        for scores_slides, bins_scores_histogram, n_slides, in_gaussianized, in_integral, in_clean in list_options:
            if scores_slides is not None:  # If some slides do not exist, do not do anything
                if bins is not None:
                    bins_scores_histogram_bg = bins
                else:
                    if bins_scores_histogram is not None:
                        bins_scores_histogram_bg = bins_scores_histogram
                    else:
                        bin_mi, bin_ma = self._get_bin_min_bin_max(in_gaussianized=in_gaussianized, use_integral=in_integral)
                        if bin_min is None:
                            bin_min = bin_mi
                        if bin_max is None:
                            bin_max = bin_ma
                        bins_scores_histogram_bg = np.arange(bin_min, bin_max, bin_width)

                distr_score_h0 = np.histogram(scores_slides, bins=bins_scores_histogram_bg, density=True)[0]
                # Correcting for the slides that had a too low score in order to make the full histogram integrate to unity
                distr_score_h0 = distr_score_h0 * len(scores_slides) / self.n_slides

                if self.transiting_planets_of_radii_in_per_range_one_star is not None:
                    self.transiting_planets_of_radii_in_per_range_one_star_poisson = ut.get_poisson_at_least(
                        rate=self.transiting_planets_of_radii_in_per_range_one_star, min_number=1)
                    distr_score_h0 = distr_score_h0 * (1 - self.transiting_planets_of_radii_in_per_range_one_star_poisson)
                    self.background_normalization_coef = len(self.stats_slides) / self.n_slides * (
                            1 - self.transiting_planets_of_radii_in_per_range_one_star_poisson)
                # # In this definition, the integral is one (\sum d score * count(score) = 1)
                # if self.transiting_planets_in_relevant_range is not None:
                #     distr_score_h0 = distr_score_h0 * (1 - self.transiting_planets_in_relevant_range)
                #     # -> the rate of not having a planet
                #     # -> This is to make the integral of the injections distribution equal to the number of planets in the
                #     # relevant parameter range per star, and the slides distribution complementary of that.
                #     # This is the expected number of scores from not-planet in one experiment. The sum of the scores from
                #     # planets and from not-planets is unity

                if in_integral:
                    if in_clean:
                        self.distr_int_score_h0_clean = distr_score_h0
                        self.bins_int_score_histogram_bg = bins_scores_histogram_bg
                    else:
                        self.distr_int_score_h0 = distr_score_h0
                        self.bins_int_score_histogram_bg = bins_scores_histogram_bg
                else:
                    if in_gaussianized:
                        if in_clean:
                            self.distr_stat_h0_clean = distr_score_h0
                            self.bins_stat_histogram_bg = bins_scores_histogram_bg
                        else:
                            self.distr_stat_h0 = distr_score_h0
                            self.bins_stat_histogram_bg = bins_scores_histogram_bg
                    else:
                        if in_clean:
                            self.distr_gauss_stat_h0_clean = distr_score_h0
                            self.bins_gauss_stat_histogram_bg = bins_scores_histogram_bg
                        else:
                            self.distr_gauss_stat_h0 = distr_score_h0
                            self.bins_gauss_stat_histogram_bg = bins_scores_histogram_bg
                # ---------------------------------------------------------------------------------------
                # MAKE EXTRAPOLATION.
                if calculate_extrapolation and (not in_clean) and (not in_gaussianized):
                    coef, ampl = af.get_extrapolation_chi2_1_limited(stats2=scores_slides, stat2_limit=extrapolation_start_score)
                    self.extrapolation_coef_int_score_h0_hist = coef
                    self.extrapolation_amplitude_int_score_h0_hist = ampl
                    self.extrapolation_limit_int_score_h0_hist = extrapolation_start_score
                if calculate_extrapolation and in_clean and (not in_gaussianized):
                    coef, ampl_clean = af.get_extrapolation_chi2_1_limited(stats2=scores_slides, stat2_limit=extrapolation_start_score)
                    self.extrapolation_coef_int_score_clean_h0_hist = coef
                    self.extrapolation_amplitude_int_score_clean_h0_hist = ampl_clean
                    self.extrapolation_limit_int_score_clean_h0_hist = extrapolation_start_score

    def make_injections_histogram(self, bins=None, bin_min=None, bin_max=None, use_integral=True):
        if self.retrieved_int_score_success is None:
            self.retrieve_injections()

        if self.number_of_injections > 0:
            if use_integral:
                # retrieved_score_success, retrieved_score_bad, bins_scores_histogram, in_gaussianized, in_integral
                list_options = [[self.retrieved_int_score_success, self.retrieved_int_score_bad, self.bins_int_score_histogram,
                                 False, True]]
            else:
                list_options = [[self.retrieved_stat_success, self.retrieved_stat_bad, self.bins_stat_histogram, False, False],
                                [self.retrieved_gauss_stat_success, self.retrieved_gauss_stat_bad, self.bins_gauss_stat_histogram,
                                 True, False]]
            for retrieved_score_success, retrieved_score_bad, bins_scores_histogram, in_gaussianized, in_integral in list_options:
                if bins is not None:
                    bins_score_histogram_inj = bins
                else:
                    if bins_scores_histogram is not None:
                        bins_score_histogram_inj = bins_scores_histogram
                    else:
                        bin_mi, bin_ma = self._get_bin_min_bin_max(use_integral=in_integral, in_gaussianized=in_gaussianized)
                        if bin_min is None:
                            bin_min = bin_mi
                        if bin_max is None:
                            bin_max = bin_ma
                        bins_score_histogram_inj = np.arange(bin_min, bin_max, Histogram_mes_binwid)

                scores_all = np.array(retrieved_score_bad + retrieved_score_success, dtype=float)
                n_injections = len(scores_all)
                scores_all = scores_all[~np.isnan(scores_all)]
                distr_score_inj_all = np.histogram(scores_all, bins=bins_score_histogram_inj, density=True)[0]
                # Normalization to have integral one (\sum d score * n(score) = 1)

                self.transiting_planets_in_relevant_range_poisson = ut.get_poisson_at_least(
                    rate=self.transiting_planets_in_relevant_range, min_number=1)
                distr_score_inj_all = (distr_score_inj_all * self.transiting_planets_in_relevant_range_poisson)
                # -> This is to make the integral of the distribution equal to the number of planets in the
                # relevant parameter range per star. This is the expected number of scores from planet in one experiment
                # distr_score_inj_all = (distr_score_inj_all / n_injections * self.transiting_planets_in_relevant_range)
                scores_success = np.array(retrieved_score_success, dtype=float)
                n_success_injections = len(scores_success)
                scores_success = scores_success[~np.isnan(scores_success)]
                distr_score_inj_success = np.histogram(scores_success, bins=bins_score_histogram_inj, density=True)[0]
                distr_score_inj_success = (distr_score_inj_success * self.transiting_planets_in_relevant_range_poisson)
                self.n_injections = n_injections
                self.n_success_injections = n_success_injections
                distr_score_inj_success = distr_score_inj_success * n_success_injections / n_injections
                # -> this is to match the total number of planets after we used density

                if in_integral:
                    self.distr_int_score_inj_all = distr_score_inj_all
                    self.distr_int_score_inj_success = distr_score_inj_success
                    self.bins_int_score_histogram_inj = bins_score_histogram_inj
                else:
                    if in_gaussianized:
                        self.distr_gauss_stat_inj_all = distr_score_inj_all
                        self.distr_gauss_stat_inj_success = distr_score_inj_success
                        self.bins_gauss_stat_histogram_inj = bins_score_histogram_inj
                    else:
                        self.distr_stat_inj_all = distr_score_inj_all
                        self.distr_stat_inj_success = distr_score_inj_success
                        self.bins_stat_histogram_inj = bins_score_histogram_inj

    def calculate_significance(self, remake_histograms=True, bin_width=Histogram_mes_binwid):
        if remake_histograms or self.bins_int_score_histogram is None:
            self.make_histogram_bins(bin_width=bin_width, use_slides=True, use_injections=True)
            self.make_injections_histogram(bins=self.bins_int_score_histogram, use_integral=True)
            self.make_background_histogram(bins=self.bins_int_score_histogram, use_integral=True)
        # ---------------------------------------------------------------------------------------
        # MAKE EXTRAPOLATION. Determine intersection with horizontal axis in log scale.

        self.prob_density_h0_trigger = (self.background_normalization_coef * self.extrapolation_amplitude_int_score_h0_hist
                                        * af.chi2_1(x2=self.best_int_score, coef=self.extrapolation_coef_int_score_h0_hist))
        #  stat2_limit=self.extrapolation_limit_int_score_h0_hist))
        false_alarm_prob_trigger = (self.background_normalization_coef * self.extrapolation_amplitude_int_score_h0_hist
                                    * af.sf_asymptotic_chi2_1(stat2=self.best_int_score,
                                                              coef=self.extrapolation_coef_int_score_h0_hist))  # ,
        # stat2_limit=self.extrapolation_limit_int_score_h0_hist))

        # self.extrapolation_coefs_int_score_h0_hist = af.get_histogram_extrapolation_coefs_in_log(
        #     self.distr_int_score_h0, self.bins_int_score_histogram)
        # self.prob_density_h0_trigger = np.exp(self.best_int_score * self.extrapolation_coefs_int_score_h0_hist[1]
        #                                       + self.extrapolation_coefs_int_score_h0_hist[0])
        # false_alarm_prob_trigger = (1 / np.abs(self.extrapolation_coefs_int_score_h0_hist[1])
        #                             * np.exp(self.extrapolation_coefs_int_score_h0_hist[0]
        #                                      + self.extrapolation_coefs_int_score_h0_hist[1] * self.best_int_score))

        self.prob_density_h0_clean_trigger = (self.background_normalization_coef * self.extrapolation_amplitude_int_score_clean_h0_hist
                                              * af.chi2_1(x2=self.best_int_score,
                                                          coef=self.extrapolation_coef_int_score_clean_h0_hist))
        # stat2_limit=self.extrapolation_limit_int_score_clean_h0_hist))
        false_alarm_prob_trigger_clean = (self.background_normalization_coef * self.extrapolation_amplitude_int_score_clean_h0_hist
                                          * af.sf_asymptotic_chi2_1(stat2=self.best_int_score,
                                                                    coef=self.extrapolation_coef_int_score_clean_h0_hist))
        # stat2_limit=
        # self.extrapolation_limit_int_score_clean_h0_hist))

        # self.extrapolation_coefs_int_score_h0_clean_hist = af.get_histogram_extrapolation_coefs_in_log(
        #     self.distr_int_score_h0_clean, self.bins_int_score_histogram)
        # self.prob_density_h0_clean_trigger = np.exp(self.best_int_score * self.extrapolation_coefs_int_score_h0_clean_hist[1]
        #                                             + self.extrapolation_coefs_int_score_h0_clean_hist[0])
        # false_alarm_prob_trigger_clean = (1 / np.abs(self.extrapolation_coefs_int_score_h0_clean_hist[1])
        #                                   * np.exp(self.extrapolation_coefs_int_score_h0_clean_hist[0]
        #                                            + self.extrapolation_coefs_int_score_h0_clean_hist[1] * self.best_int_score))
        # Find edges of this bin:
        bin_ind_close_trigger = np.sort(np.argsort(np.abs(self.bins_int_score_histogram - self.best_int_score))[:2])[0]
        vals_select = self.distr_int_score_inj_success[max(0, bin_ind_close_trigger - 3): min(len(self.distr_int_score_inj_success),
                                                                                              bin_ind_close_trigger + 3)]
        vals_select = vals_select[vals_select != 0]  # remove zeros in case the bins are too narrow and there are holes
        if len(vals_select) == 0:
            vals_select = [0]
        self.prob_density_h1_trigger = np.nanmean(vals_select)
        # use the left edge index to access the bin
        p_planet = ut.calculate_p_planet_statistical(f_p=self.prob_density_h1_trigger, f_s=self.prob_density_h0_trigger,
                                                     f_sc=self.prob_density_h0_clean_trigger, pi_p=None,  # We already multiplied by it
                                                     mu_m_mu_p=1.)  # we take this approximation close to the detection threshold
        self.prob_density_planet_not_noise = p_planet
        # self.prob_density_planet_not_noise = self.prob_density_h1_trigger / (self.prob_density_h1_trigger + self.prob_density_h0_trigger)
        self.false_alarm_prob_trigger = false_alarm_prob_trigger * (1 - p_planet) + false_alarm_prob_trigger_clean * p_planet

        # # ---------------------------------------------------------------------------------------
        # # MAKE EXTRAPOLATION
        # # Determine intersection with horizontal axis in log scale:
        # self.extrapolation_coefs_stat_h0_hist = af.get_histogram_extrapolation_coefs_in_log(
        #     self.distr_stat_h0, self.bins_stat_histogram)  # attention: extrapolating only the direct bg
        # # self.extrapolation_coefs_stat_h0_hist[1] is supposed to be negative because the trend is decreasing
        # self.prob_density_h0_trigger = np.exp(self.best_stat * self.extrapolation_coefs_stat_h0_hist[1]
        #                                       + self.extrapolation_coefs_stat_h0_hist[0])
        # # Follows from the integration of the non-log distribution over MES[SNR^2]
        # self.false_alarm_prob_trigger = (1 / np.abs(self.extrapolation_coefs_stat_h0_hist[1])
        #                                  * np.exp(
        #             self.extrapolation_coefs_stat_h0_hist[0] + self.extrapolation_coefs_stat_h0_hist[1] * self.best_stat))
        # bin_ind_close_trigger = np.sort(np.argsort(np.abs(self.bins_stat_histogram - self.best_stat))[:2])[0]  # find edges of this bin
        # # self.prob_density_h1_trigger = self.distr_stat_inj_all[bin_ind_close_trigger]  # use the left edge index to access the bin
        # self.prob_density_h1_trigger = np.nanmean(self.distr_stat_inj_all[
        #                                           max(0, bin_ind_close_trigger - 1): min(len(self.distr_stat_inj_all),
        #                                                                                  bin_ind_close_trigger + 1)])
        # # use the left edge index to access the bin
        # self.prob_density_planet_not_noise = self.prob_density_h1_trigger / (self.prob_density_h1_trigger + self.prob_density_h0_trigger)

    def analyze_injections_and_background(self, max_phase_mismatch_to_dur=Max_phase_mismatch_to_dur,
                                          bins=None, bin_min=None, bin_max=None, bin_width=Histogram_mes_binwid):
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
            self.make_histogram_bins(bin_min=bin_min, bin_max=bin_max, bin_width=bin_width, use_slides=True, use_injections=True)
        else:
            self.bins_int_score_histogram = bins
        self.make_injections_histogram(bins=self.bins_int_score_histogram, use_integral=True)
        self.make_background_histogram(bins=self.bins_int_score_histogram, use_integral=True)
        self.calculate_significance(remake_histograms=False)

    def get_periodicity_results(self, short_periodicity=True,
                                best_periodic_scores_filename_postfix=Best_periodic_scores_filename_postfix,
                                best_short_periodic_scores_filename_postfix=Best_short_periodic_scores_filename_postfix,
                                return_lists=False, set_attributes=True,
                                delta_stat_close=Delta_stat_close,
                                delta_integral_close=Delta_integral_close):
        return_array = af.get_periodicity_results(target_id=self.target_id, output_location=self.output_location,
                                                  short_periodicity=short_periodicity,
                                                  best_periodic_scores_filename_postfix=best_periodic_scores_filename_postfix,
                                                  best_short_periodic_scores_filename_postfix=best_short_periodic_scores_filename_postfix)
        if set_attributes and not short_periodicity:
            ind_max, _, __ = af.get_best_int_score_and_stat_index(int_scores=return_array[spip.Best_per_results_index_best_integral],
                                                                  stats=return_array[spip.Best_per_results_index_stats],
                                                                  delta_stat_close=delta_stat_close,
                                                                  delta_integral_close=delta_integral_close)
            self.best_int_score = return_array[spip.Best_per_results_index_best_integral][ind_max]
            self.best_stat = return_array[spip.Best_per_results_index_stats][ind_max]
            self.best_per = return_array[spip.Best_per_results_index_per][ind_max]
            self.best_ftt = return_array[spip.Best_per_results_index_ftt][ind_max]
            self.best_temp = return_array[spip.Best_per_results_index_template][ind_max]
            self.best_gauss_stat = return_array[spip.Best_per_results_index_stats_gauss][ind_max]
        if return_lists:
            return return_array

    def fetch_saved_lightcurve(self, initialize_templates=False, condition_templates=False, detect_single_transits=False):
        if detect_single_transits:
            condition_templates = True
        self.unpickle_small_attributes()
        self.initialize_kepler_light_curve()
        self.klc.load_conditioned_data()
        if initialize_templates:
            self.klc.initialize_templates()
        if condition_templates:
            self.klc.condition_templates()
        if detect_single_transits:
            self.klc.detect_single_transits()
        self.klc.load_connected_stats()
        self.klc.interleave_shifts(delete_fulls=False)
        self.klc.get_summed_values(first_transit_time=self.best_ftt, period=self.best_per, template=self.best_temp)


def get_target_for_tce(tce):
    target = KeplerTarget(tce.target_id, data_location=tce.data_location, output_location=tce.output_location,
                          problems_folder=None)
    return target
