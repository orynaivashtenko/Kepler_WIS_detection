#!/usr/bin/env python3

import os

os.environ["OMP_NUM_THREADS"] = "1"  #
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"  #
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import numpy as np
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import auxiliary_functions
from locations import Problems_folder
import target_class
import trigger_functions
from pipeline_parameters import Data_location, Output_location, Problems_folder
from astropy.table import Table

p_planet_table_file = '/home/projects/barakz/Collaboration-Kepler/orynai/p_planet_for_p50_run_12_catalog.ecsv'
trig_ids = ['005107348', '006368905', '002994092', '007191646', '008644911', '006186182', '008509442', '001865864', '004912650', '005342061', '004270799', '009071593', '005792202', '009782748',
       '010003058', '009605514', '002310993', '002995392', '003101923', '006934986', '009947389', '005184911', '006387557', '006390122', '008364232', '003646449', '009839821', '009514372',
       '009268546', '007300182', '003558849', '008007174', '005900235', '003236705', '003758499', '008120820', '007829836', '009466429', '001718189', '005856571', '007663405', '008934103',
       '009935983', '008349399', '002141783', '010749059', '010055126', '011446424', '009153824', '008807057', '006677841', '010272442', '006934045', '010089123', '005385304', '010798331',
       '010610039', '010933306', '008264402', '009391817', '011968463', '010810859', '012416987', '012734396', '010467714', '011125797', '011446961', '011450414', '010658177', '011658036',
       '009521317', '011913235', '003101178', '010967168', '012785320', '003122872', '004450844', '009020114']


# CALCULATING SIGNIFICANCES HERE
p_planets_for_target_ids = []
false_alarm_prob_trigger_for_target_ids = []
transit_probability_for_target_ids = []
transiting_planets_of_radii_in_per_range_one_star_for_target_ids = []
transit_and_range_probability_for_target_ids = []
transiting_planets_in_relevant_range_for_target_ids = []

for i, tid in enumerate(trig_ids):
    data_location = auxiliary_functions.get_correct_tid_folder(target_id=tid, location=Data_location,
                                                                 allow_creation=False, with_prefix_folder=True)
    output_location = auxiliary_functions.get_correct_tid_folder(target_id=tid, location=Output_location,
                                                                 allow_creation=False, with_prefix_folder=True)
    target_1 = target_class.KeplerTarget(tid, data_location=data_location,
                                         output_location=output_location, problems_folder=Problems_folder)
    target_1.unpickle_small_attributes()
    if not np.isnan(target_1.best_per) and not np.isnan(target_1.stellar_radius) and target_1.is_good_for_processing:
        target_1.analyze_injections_and_background()
        if (target_1.prob_density_planet_not_noise < 0.1) and (target_1.best_int_score > 60):
            target_1.analyze_injections_and_background(bin_width=3.)
            # -> Sometimes the injection range is so broad that there can be holes between bins
        p_planets_for_target_ids.append(target_1.prob_density_planet_not_noise)
        false_alarm_prob_trigger_for_target_ids.append(target_1.false_alarm_prob_trigger)
        transit_probability_for_target_ids.append(target_1.transit_probability)
        transiting_planets_of_radii_in_per_range_one_star_for_target_ids.append(target_1.transiting_planets_of_radii_in_per_range_one_star)
        transit_and_range_probability_for_target_ids.append(target_1.transit_and_range_probability)
        transiting_planets_in_relevant_range_for_target_ids.append(target_1.transiting_planets_in_relevant_range)
    else:
        p_planets_for_target_ids.append(np.nan)
        false_alarm_prob_trigger_for_target_ids.append(np.nan)
        transit_probability_for_target_ids.append(np.nan)
        transiting_planets_of_radii_in_per_range_one_star_for_target_ids.append(np.nan)
        transit_and_range_probability_for_target_ids.append(np.nan)
        transiting_planets_in_relevant_range_for_target_ids.append(np.nan)

dict_significance = {
    'target_id': list(trig_ids),
    'p_planet': p_planets_for_target_ids,
    'false_alarm_prob': false_alarm_prob_trigger_for_target_ids,
    'transit_probability': transit_probability_for_target_ids,
    'transiting_planets_of_radii_in_per_range_one_star': transiting_planets_of_radii_in_per_range_one_star_for_target_ids,
    'transit_and_range_probability': transit_and_range_probability_for_target_ids,
    'transiting_planets_in_relevant_range': transiting_planets_in_relevant_range_for_target_ids}

table_results = Table(dict_significance)
table_results.write(p_planet_table_file, format='ascii.ecsv', overwrite=True)

# bsub -q physics-short -R "rusage[mem=3.9GB]"  source ~/.bashrc; conda activate planets_env_amd_epyc; /home/projects/barakz/orynai/Kepler_Earths/WIS_Kepler_pipeline/runfiles/serial_p_planet_quick.py





