#!/usr/bin/env python3

# #########################################################################################################################
# THIS FILE CHECKS WHO FROM THE TRIGGERS THAT SHOULD HAVE RUN DID NOT RUN (USUALLY BECAUSE SOME WEXAC ISSUES)
# #########################################################################################################################

import os

core_num = "1"
os.environ["OMP_NUM_THREADS"] = core_num  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = core_num  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = core_num  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = core_num  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = core_num  # export NUMEXPR_NUM_THREADS=6

import numpy as np
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import pipeline_parameters as pp

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--tids_list_file", type=str, nargs=1,
                    help="path to file where to take tids. If empty, will go over all the folders",
                    default='')
parser.add_argument("--not_run_tids_file", type=str, nargs=1, help="path to file where to save non-run tids",
                    default=os.path.join(pp.Threshold_crossing_lists_folder, pp.Tids_that_did_not_run_filename))

args = parser.parse_args()
tids_list_file = args.tids_list_file[0]
not_run_tids_file = args.not_run_tids_file[0]


def there_is_output(output_folder):
    if not os.path.exists(output_folder):
        return False
    else:
        filenames = os.listdir(output_folder)
        target_class_dict_list = [filename for filename in filenames if 'target_class_dict' in filename]
        if len(target_class_dict_list) == 0:
            return False
    return True


Input_folder = pp.Data_location
Output_folder = pp.Output_location

not_run_tids = []
if tids_list_file == '':
    for prefix in os.listdir(Input_folder):
        for tid in os.listdir(os.path.join(Input_folder, prefix)):
            output_folder_tid = os.path.join(Output_folder, prefix, tid)
            if not there_is_output(output_folder_tid):
                not_run_tids.append(tid)
else:
    with open(tids_list_file, 'r') as fil:
        tid_list = fil.readlines()
    tid_list = [line.strip() for line in tid_list]
    for tid in tid_list:
        prefix = tid[:4]
        output_folder_tid = os.path.join(Output_folder, prefix, tid)
        if not there_is_output(output_folder_tid):
            not_run_tids.append(tid)

with open(not_run_tids_file, 'w') as fil:
    for tid in not_run_tids:
        print(tid, file=fil)

# bsub -q short -R "rusage[mem=2GB]" /home/labs/barakz/orynai/Kepler_Earths/WIS_Kepler_pipeline/runfiles/serial_check_who_did_not_run.py
