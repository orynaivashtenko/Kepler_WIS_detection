#!/usr/bin/env python3

import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import file_operations as fop
from runfiles_parameters import *
from pipeline_parameters import Threshold_crossing_list_file, Score_candidate_postprocess, Per_min_candidate
from trigger_functions import Index_tid, get_trigger_from_target_id, Indices_dict, load_triggers_unique

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
trig_ids = [tid for tid in trig_ids if (tid.isnumeric() and len(tid) == 9)]


mem_per_job_gb = '6.9'  # because there are many target pixels, job may need more memory
Queue_submit = 'short'
job_type_prefix = Pre_slides_tce_veto_job_prefix
fop.arrange_job_folders(job_type_prefix=job_type_prefix,
                        jobs_folder=Job_files_folder,
                        output_files_folder=Output_files_folder,
                        error_files_folder=Error_files_folder,
                        jobs_monitor_folder=Jobs_monitor_folder,
                        use_last_digit=False, use_last_digit_monitor=False)

job_files_folder = os.path.join(Job_files_folder, job_type_prefix)
output_files_folder = os.path.join(Output_files_folder, job_type_prefix)
error_files_folder = os.path.join(Error_files_folder, job_type_prefix)

target_ids, integral_scores, stats, pers, ftts, gauss_stats, templates = load_triggers_unique(file=trigger_list_file)

for tid in trig_ids:
    trigger_main_search = get_trigger_from_target_id(target_id=tid,
                                                     target_ids=target_ids,
                                                     integral_scores=integral_scores,
                                                     stats=stats,
                                                     pers=pers,
                                                     ftts=ftts,
                                                     gauss_stats=gauss_stats,
                                                     templates=templates)  # trigger_list_file=trigger_list_file
    int_score_search = float(trigger_main_search[Indices_dict['int_score']])
    per_search = float(trigger_main_search[Indices_dict['per']])
    if int_score_search >= Score_candidate_postprocess and per_search >= Per_min_candidate:
        job_name = fop.compose_job_name(job_type_prefix=Pre_slides_tce_veto_job_prefix,
                                        target_id=tid, additional_params=())
        # noinspection DuplicatedCode
        fop.write_job_file(code_file=None,
                           code_filename=Pre_slides_tce_veto_code_filename,
                           code_file_dir=Run_files_folder,
                           jobname=job_name,
                           job_file=None,
                           job_filename=job_name + '.bsub',
                           job_file_dir=job_files_folder,
                           queue=Queue_submit,
                           mem_per_job_gb=mem_per_job_gb,
                           array_indices=None,
                           output_file=None,
                           output_filename=job_name + '.out',
                           output_file_dir=output_files_folder,
                           error_file=None,
                           error_filename=job_name + '.err',
                           error_file_dir=error_files_folder,
                           output_error_overwrite=False,
                           arg_jobind=False,
                           other_args=[['target_id', tid], ['job_name', job_name]])

# bsub -sp 100 -q short source ~/.bashrc; conda activate planets_env_amd_epyc; /home/projects/barakz/orynai/Kepler_Earths/WIS_Kepler_pipeline/runfiles/create_jobs_pre_slides_tce_veto.py
