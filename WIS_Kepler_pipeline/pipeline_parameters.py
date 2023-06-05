import numpy as np
import os

######################################################################################################################################
# PHYSICAL CONSTANTS AND RATES
# ------------------------------------------------------------------------------------------------------------------------------------

# Physical constants is CGS
R_sun = 6.9634e10  # cm #solar radius
au = 1.495978707e13  # cm #astronomical unit
M_sun = 1.989e33  # g #solar mass
G = 6.67430e-8  # dyn⋅cm2/g2 #gravitational constant
R_sun_R_e = 109.076  # sun radii to Earth radii

_this_file_dirname = os.path.dirname(__file__)
Working_place = None  # On which machine it is running: my office Mac ('Home'), Marvin or WEXAC
# Working_place = 'Home' # 'Marvin'  # 'WEXAC'
if Working_place is None:
    if '/Users/orynaivashtenko/Work/Kepler_Project' in _this_file_dirname:
        Working_place = 'Home'
    elif '/SMALLDATA' in _this_file_dirname or '/home/orynai/' in _this_file_dirname:
        Working_place = 'Marvin'
    elif '/home/labs/barakz' in _this_file_dirname:
        Working_place = 'WEXAC'

del _this_file_dirname


# Way to use the occurrence table:
def get_occurrence_table(stellar_radius):
    """
    Per one star
    :param stellar_radius:
    :type stellar_radius:
    :return:
    :rtype:
    """
    # We are not putting the following occurrence rate info to a separate file because it is easier like this.
    # Generating rp according to the prior
    log_rp_edges = np.log(np.array([0.5, 1, 2, 4, 8, 20]) / stellar_radius / R_sun_R_e)  # This was measured in Earth radii
    period_edges = 10 ** (np.linspace(1, 3, 11))[2:9]  # An exact empirical formula for the values in the table.
    occurrences_table = np.array([
        # zhu and dong, planets per 100 stars (WARNING: WE ONLY DETECT ONE PER STAR, DISTRIBUTION CAN BE DIFFERENT!)
        [4.7, 4.1, 7.2, 0.66, 0.23],  # P = 25.1  - 39.8  days
        [4.2, 4.2, 8.1, 0.58, 0.52],  # P = 39.8  - 63.1  days
        [3.1, 3.1, 6.9, 1.50, 0.44],  # P = 63.0  - 100.0 days
        [4.2, 4.2, 6.2, 1.30, 1.30],  # P = 100.0 - 158.5 days
        [4.9, 4.9, 7.5, 2.40, 1.20],  # P = 158.5 - 251.2 days
        [10.0, 10.0, 6.9, 1.70, 1.70]]  # P = 251.2 - 398.1 days
    ) / 100
    return log_rp_edges, period_edges, occurrences_table


# TARGET CLASS PARAMETERS
P_min_search = 30  # days
P_max_search = 500  # days
P_chunks_search = 360  # will give approx 2GB per core, I increased it when changed duty cycle
Mes_candidate_search = 40**0.5  # mes in units of SNR (not squared)
N_broad_slides = 5  # for all templates and periods
N_deep_slides_total = 100000  # for one template and narrow periods for all the runs
# I thought to make chunk 50000 for one template and narrow periods per one run (about 14 hours) (rough 1 sec per one periodicity search)
# But there was a timeout, so
N_deep_slides_chunk = 2000
N_injections_total = 10000  # total number of all injections (all radii and periods etc.)
N_injections_chunk = 50  # number of injections per job so that it will take 10/3 hours (120/3 sec per one inj detection)
Per_wing_injection_and_slides_relative = 2 / 30  # Range to the left and to the right of the best period in units of period. Will be used
# to generate injections and deep time slides with variable periods around the best one
Per_wing_injection_detection = 0.5  # days. Range to the left and to the right of the injected period used in the injection recovery
Max_phase_mismatch_rel = 0.001  # Maximal relative phase difference for defining successful injections
Default_min_plot_mes2 = 10
Default_max_plot_mes2 = 140

Figsize_fluxes = (15, 5)
Figsize_folds = (8, 6)
Figsize_hist = (8, 6)

Data_location = os.path.join(os.getcwd(), 'light_curves')  # in this directory, a file named after target_id will be created
Output_location = os.path.join(os.getcwd(), 'outputs')  # in this directory, a file named after target_id will be created
Target_pixels_location = os.path.join(os.getcwd(), 'target_pixels')
Threshold_crossing_lists_folder = os.path.join(os.getcwd(), 'outputs')
Run_root_folder = os.path.join(os.getcwd(), 'outputs')
Run_name = 'run_7'

if Working_place == 'Home':
    Gaia_kepler_file = '/Users/orynaivashtenko/Work/Kepler_Project/kepler_dr3_good.fits'
    Koi_dict_file = '/Users/orynaivashtenko/Work/Kepler_Project/KOI_catalog_dict.pickle'
    Ttv_dict_file = '/Users/orynaivashtenko/Work/Kepler_Project/TTV_catalog_dict.pickle'
    Villanova_dict_file = '/Users/orynaivashtenko/Work/Kepler_Project/villanova_catalog_dict.pickle'

if Working_place == 'Marvin':
    Data_location = '/SMALLDATA/KeplerData/Kepler'  # TODO : this does not work properly, not entering the prefix directory
    Output_location = '/SMALLDATA/orynai/' + str(Run_name) + '_output'
    Kepler_files_folder = '/Collaboration-Kepler/KeplerFiles'
    Koi_dict_file = os.path.join(Kepler_files_folder, 'KOI_catalog_dict.pickle')
    Ttv_dict_file = os.path.join(Kepler_files_folder, 'TTV_catalog_dict.pickle')
    Gaia_kepler_file = os.path.join(Kepler_files_folder, 'kepler_dr3_good.fits')
    Villanova_dict_file = os.path.join(Kepler_files_folder, 'villanova_catalog_dict.pickle')
    Threshold_crossing_lists_folder = '/SMALLDATA/orynai/' + str(Run_name)
    Run_root_folder = '/SMALLDATA/orynai/' + str(Run_name)
    Target_pixels_location = '/Collaboration-Kepler/KeplerTargetPixels'

if Working_place == 'WEXAC':
    Data_location = '/home/labs/barakz/Collaboration-Kepler/KeplerData'
    Output_location = '/home/labs/barakz/Collaboration-Kepler/orynai/output_' + str(Run_name)
    Kepler_files_folder = '/home/labs/barakz/Collaboration-Kepler/KeplerFiles'
    Koi_dict_file = os.path.join(Kepler_files_folder, 'KOI_catalog_dict.pickle')
    Ttv_dict_file = os.path.join(Kepler_files_folder, 'TTV_catalog_dict.pickle')
    Gaia_kepler_file = os.path.join(Kepler_files_folder, 'kepler_dr3_good.fits')
    Villanova_dict_file = os.path.join(Kepler_files_folder, 'villanova_catalog_dict.pickle')
    Threshold_crossing_lists_folder = '/home/labs/barakz/orynai/' + str(Run_name)
    Run_root_folder = '/home/labs/barakz/orynai/' + str(Run_name)
    Target_pixels_location = '/home/labs/barakz/Collaboration-Kepler/KeplerTargetPixels'

# Gaia_kepler_file = '/home/orynai/Kepler_Notebooks/kepler_dr3_good.fits'
# Default_koi_file = '/home/orynai/Kepler_Notebooks/cumulative_2022.05.30_11.44.28.csv'
# Default_targets_info_file = '/Users/orynaivashtenko/Work/Kepler_Project/input_stars_R2_and_R3_mass.txt'
# Default_koi_file = '/Users/orynaivashtenko/Work/Kepler_Project/cumulative_2022.05.30_11.44.28.csv'  # used in run_5 for validation
# # # Default_koi_catalog_file = '/Users/orynaivashtenko/Work/Kepler_Project/cumulative_2022.10.18_03.41.17.csv'
# containing koi names and many fields
# Default_ttv_catalog_file = '/Users/orynaivashtenko/Work/Kepler_Project/TTV_Catalog_Table3_O-C.txt'


Home_output_storage_folder = 'orynai@marvin.weizmann.ac.il:/SMALLDATA/orynai/' + str(Run_name) + '_output'


Threshold_crossing_list_filename = 'threshold_crossing_kid_info.txt'
Threshold_inv_crossing_list_filename = 'threshold_inv_crossing_kid_info.txt'

Summary_target_txt_filename_postfix = 'target_summary.txt'
Dict_target_txt_filename_postfix = 'target_class_dict.txt'
Pickle_target_small_attributes_filename_postfix = 'target_class_small_attributes.pickle'
Conditioned_data_filename_postfix = 'conditioned_data'
Best_periodic_scores_filename_postfix = 'best_periodic_scores_all_p_slides'
Best_inverted_periodic_scores_filename_postfix = 'best_inverted_periodic_scores_all_p_slides'
# WARNING incorrect name TODO change in the next run!
Injection_parameters_filename_postfix = 'injection_parameter_sets.txt'
All_templates_injection_parameters_filename_postfix = 'all_templates_injection_parameter_sets'
Injections_filename_postfix = 'injections.txt'
Slides_mes_inv_filename_postfix = 'slides_mes_inversion_slides'
Slides_mes_filename_postfix = 'slides_mes_slides'
Slides_pers_inv_filename_postfix = 'slides_pers_inversion_slides'
Slides_pers_filename_postfix = 'slides_pers_slides'
Slides_first_times_inv_filename_postfix = 'slides_first_times_inversion_slides'
Slides_first_times_filename_postfix = 'slides_first_times_slides'

Histogram_mes2_binwid = 1.  # Width of the bin in the MES[SNR^2] histogram of background and injections
