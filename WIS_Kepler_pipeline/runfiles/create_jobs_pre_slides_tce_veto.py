#!/usr/bin/env python3

import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import file_operations as fop
from runfiles_parameters import *
from pipeline_parameters import Threshold_crossing_list_file, Score_candidate_postprocess, Per_min_candidate
from trigger_functions import Index_tid, get_trigger_from_target_id, Indices_dict

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--trigger_list_file", type=str, nargs=1, help="path to file where to take triggers",
                    default=Threshold_crossing_list_file)
args = parser.parse_args()
trigger_list_file = args.trigger_list_file
if isinstance(trigger_list_file, list):
    trigger_list_file = trigger_list_file[0]

with open(trigger_list_file, 'r') as tlf:
    trig_ids = tlf.readlines()
trig_ids = [tid.split()[Index_tid] for tid in trig_ids]
trig_ids = list(set(trig_ids))  # remove duplicates


mem_per_job_gb = '6.9'  # because there are many target pixels, job may need more memory
Queue_submit = 'physics-short'

for tid in trig_ids:
    trigger_main_search = get_trigger_from_target_id(tid, trigger_list_file=trigger_list_file)
    int_score_search = float(trigger_main_search[Indices_dict['int_score']])
    per_search = float(trigger_main_search[Indices_dict['per']])
    if int_score_search >= Score_candidate_postprocess and per_search >= Per_min_candidate:
        job_name = fop.compose_job_name(job_type_prefix=Pre_slides_tce_veto_job_filename_prefix,
                                        target_id=tid, additional_params=())
        # noinspection DuplicatedCode
        fop.write_job_file(code_file=None,
                           code_filename=Pre_slides_tce_veto_code_filename,
                           code_file_dir=Run_files_folder,
                           jobname=job_name,
                           job_file=None,
                           job_filename=job_name + '.bsub',
                           job_file_dir=Job_files_folder,
                           queue=Queue_submit,
                           mem_per_job_gb=mem_per_job_gb,
                           array_indices=None,
                           output_file=None,
                           output_filename=job_name + '.out',
                           output_file_dir=Output_files_folder,
                           error_file=None,
                           error_filename=job_name + '.err',
                           error_file_dir=Error_files_folder,
                           output_error_overwrite=False,
                           arg_jobind=False,
                           other_args=[['target_id', tid], ['job_name', job_name]])

# bsub -q short source ~/.bashrc; conda activate planets_env_amd_epyc; /home/projects/barakz/orynai/Kepler_Earths/WIS_Kepler_pipeline/runfiles/create_jobs_pre_slides_tce_veto.py
