#!/usr/bin/env python3

import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import file_operations as fop
from pipeline_parameters import Triggers_for_slides_injections_file, N_slides_total, N_slides_chunk
from runfiles_parameters import *
from trigger_functions import Index_tid, Index_per

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
trig_ids_0 = [trig.split()[Index_tid] for trig in trigs]
periods_0 = [float(trig.split()[Index_per]) for trig in trigs]

# Select unique:
trig_ids = []
periods = []
for i_tid, tid in enumerate(trig_ids_0):
    if tid not in trig_ids:
        trig_ids.append(tid)
        periods.append(periods_0[i_tid])


job_type_prefix = Slides_job_prefix
fop.arrange_job_folders(job_type_prefix=job_type_prefix,
                        jobs_folder=Job_files_folder,
                        output_files_folder=Output_files_folder,
                        error_files_folder=Error_files_folder,
                        jobs_monitor_folder=Jobs_monitor_folder,
                        use_last_digit=False)
# -> do not use last digit here because it is only for triggers, and they are not that many

# Folders for this specific type of lob (main search):
job_files_folder = os.path.join(Job_files_folder, job_type_prefix)
output_files_folder = os.path.join(Output_files_folder, job_type_prefix)
error_files_folder = os.path.join(Error_files_folder, job_type_prefix)


# # noinspection DuplicatedCode
for ii in range(len(trig_ids)):
    tid = trig_ids[ii]
    per = periods[ii]
    for i in range(int(N_slides_total // N_slides_chunk)  # how many times we need to repeat timeslides
                   + int(bool(N_slides_total % N_slides_chunk))):  # stupid way to make np.ceil (otherwise numpy import)
        # noinspection DuplicatedCode
        job_name = fop.compose_job_name(job_type_prefix=job_type_prefix,
                                        target_id=tid, additional_params=(str(i),))
        out_name = fop.compose_job_name(job_type_prefix=job_type_prefix,
                                        target_id=tid, additional_params=())
        fop.write_job_file(code_file=None,
                           code_filename=Slides_runfile_filename,
                           code_file_dir=Run_files_folder,
                           jobname=job_name,
                           job_file=None,
                           job_filename=job_name + '.bsub',
                           job_file_dir=job_files_folder,
                           queue=Queue_submit,
                           mem_per_job_gb=Mem_per_job_gb,
                           array_indices=None,
                           output_file=None,
                           output_filename=out_name + '.out',
                           output_file_dir=output_files_folder,
                           error_file=None,
                           error_filename=out_name + '.err',
                           error_file_dir=error_files_folder,
                           output_error_overwrite=False,
                           arg_jobind=False,
                           other_args=[['target_id', tid], ['period', per], ['job_name', job_name]])

# bsub -q short source ~/.bashrc; conda activate planets_env_amd_epyc;  /home/projects/barakz/orynai/Kepler_Earths/WIS_Kepler_pipeline/runfiles/create_jobs_timeslides.py
# bsub -q short source ~/.bashrc; conda activate planets_env_amd_epyc;  /home/projects/barakz/orynai/Kepler_Earths/WIS_Kepler_pipeline/runfiles/create_jobs_timeslides.py --trigger_list_file /home/projects/barakz/orynai/run_11/triggers_for_slides_injections.txt
# bsub -q short source ~/.bashrc; conda activate planets_env_amd_epyc;  /home/projects/barakz/orynai/Kepler_Earths/WIS_Kepler_pipeline/runfiles/create_jobs_timeslides.py --trigger_list_file /home/projects/barakz/orynai/run_11/threshold_crossing_scrambled_triggers_to_slide.txt
# bsub -q short source ~/.bashrc; conda activate planets_env_amd_epyc;  /home/projects/barakz/orynai/Kepler_Earths/WIS_Kepler_pipeline/runfiles/create_jobs_timeslides.py --trigger_list_file /home/projects/barakz/Collaboration-Kepler/orynai/triggers_global_slides_to_slide_rerun.txt