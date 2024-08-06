#!/usr/bin/env python3

import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import file_operations as fop
from pipeline_parameters import Triggers_for_slides_injections_file, N_slides_total, N_slides_chunk
from runfiles_parameters import *
from trigger_class import Index_tid, Index_per, Index_ftt

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--trigger_list_file", type=str, nargs=1, help="path to file where to take triggers",
                    default=Triggers_for_slides_injections_file)
args = parser.parse_args()
trigger_list_file = args.trigger_list_file
if isinstance(trigger_list_file, list):
    trigger_list_file = trigger_list_file[0]

with open(trigger_list_file, 'r') as tlf:
    trigs = tlf.readlines()
trig_ids = [trig.split()[Index_tid] for trig in trigs]
periods = [float(trig.split()[Index_per]) for trig in trigs]
ftts = [float(trig.split()[Index_ftt]) for trig in trigs]

# # noinspection DuplicatedCode
for ii in range(len(trig_ids)):
    tid = trig_ids[ii]
    per = periods[ii]
    ftt = ftts[ii]
    for i in range(int(N_slides_total // N_slides_chunk)  # how many times we need to repeat timeslides
                   + int(bool(N_slides_total % N_slides_chunk))):  # stupid way to make np.ceil (otherwise numpy import)
        # noinspection DuplicatedCode
        fop.write_job_file(code_file=None,
                           code_filename=Slides_clean_runfile_filename,
                           code_file_dir=Run_files_folder,
                           jobname=Slides_clean_job_filename_prefix + '_' + tid + '.bsub',
                           job_file=None,
                           job_filename=Slides_clean_job_filename_prefix + '_' + tid + '_' + str(i) + '.bsub',
                           job_file_dir=Job_files_folder,
                           queue=Queue_submit,
                           mem_per_job_gb=Mem_per_job_gb,
                           array_indices=None,
                           output_file=None,
                           output_filename=Slides_clean_output_filename_prefix + '_' + tid + '.out',
                           output_file_dir=Output_files_folder,
                           error_file=None,
                           error_filename=Slides_clean_error_filename_prefix + '_' + tid + '.out',
                           error_file_dir=Error_files_folder,
                           output_error_overwrite=False,
                           arg_jobind=False,
                           other_args=[['target_id', tid], ['period', per], ['ftt', ftt]])

# bsub -q short /home/labs/barakz/orynai/Kepler_Earths/WIS_Kepler_pipeline/runfiles/create_jobs_timeslides_clean.py --trigger_list_file /home/labs/barakz/orynai/run_11/triggers_for_slides_injections.txt
