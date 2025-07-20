#!/usr/bin/env python3

import os
os.environ["OMP_NUM_THREADS"] = "1"  #
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"  #
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import target_class
from pipeline_parameters import *
import search_pipeline as spip
from confirmed_koi_search_create import Kepler_confirmed_koi_output_folder, Kepler_confirmed_koi_problems_folder
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--target_id", type=str, nargs=1, help="KIC number", default="003097222")
parser.add_argument("--period", type=str, nargs=1, help="period, days", default="100")
args = parser.parse_args()
Target_id = args.target_id[0]
Period = float(args.period[0])

output_location = str(os.path.join(Kepler_confirmed_koi_output_folder, Target_id))

targ = target_class.KeplerTarget(target_id=Target_id, output_location=output_location,
                                 problems_folder=Kepler_confirmed_koi_problems_folder,
                                 load_pickled_attributes=False)  # all the arguments are set to the defaults of the class
targ.unpickle_small_attributes()
# ----------------------------------------------------------------------
# We did not save conditioned data, so have to condition now:
targ.klc = spip.KeplerLightCurve(target_id=targ.target_id, get_data=True, data_location=targ.data_location,
                                 output_location=targ.output_location, m_1_3_r_s=targ.m_1_3_r_s,
                                 simulated_red_star=False, simulated_white_star=False)
targ.mask_koi_planet(modify_self_klc=True, return_durations_transit_times=False,
                     mask_candidates=Mask_kepler_candidates,
                     mask_false_positives=Mask_kepler_false_positives,
                     mask_bright_candidates=Mask_kepler_bright_candidates,
                     bright_candidate_mes=Kepler_bright_candidate_mes,
                     bright_candidate_ses=Kepler_bright_candidate_ses)
targ.klc.condition_data()
targ.klc.condition_templates()
targ.klc.save_conditioned_data(path=output_location)
# ----------------------------------------------------------------------
# main_run_output_dir = '/home/projects/barakz/Collaboration-Kepler/orynai/run_12/output_run_12'
# conditioned_data_from_main_run_file_path = os.path.join(
#     main_run_output_dir, Target_id[:4], Target_id, Target_id + '_conditioned_data.npy')
targ.generate_injection_parameters(central_period=Period)  # , conditioned_data_file_path=conditioned_data_from_main_run_file_path
