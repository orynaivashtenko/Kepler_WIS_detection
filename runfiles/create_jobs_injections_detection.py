#!/usr/bin/env python3

import os
import sys
# noinspection DuplicatedCode
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import file_operations as fop
from pipeline_parameters import np, Triggers_for_slides_injections_file, N_injections_total, N_injections_chunk, Threshold_crossing_list_file
from runfiles_parameters import *
from trigger_functions import Index_tid, load_triggers_unique
import numpy as np

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--trigger_list_file", type=str, nargs=1, help="path to file where to take triggers",
                    default=Triggers_for_slides_injections_file)
args = parser.parse_args()
trigger_list_file = args.trigger_list_file
if isinstance(trigger_list_file, list):
    trigger_list_file = trigger_list_file[0]

trig_ids_0, integral_scores_0, stats_0, periods_0, ftts_0, gauss_stats_0, templates_0 = load_triggers_unique(
    file=trigger_list_file
)
Mem_per_job_gb_here = '6.9'

# trig_ids = trig_ids_0
#
# # with open(trigger_list_file, 'r') as tlf:
# #     trig_ids = tlf.readlines()
# # trig_ids = [tid.split()[Index_tid] for tid in trig_ids]
# # trig_ids = [tid for tid in trig_ids if (tid.isnumeric() and len(tid) == 9)]  # select only valid target ids
#

# trigger_list_file = Threshold_crossing_list_file
# # !!! end temporary
# trig_ids_0, integral_scores_0, stats_0, periods_0, ftts_0, gauss_stats_0, templates_0 = load_triggers_unique(
#     file=trigger_list_file
# )

# # # !!!!!!!!!!!!
# # !!! temporary change
trig_ids = ['006368905', '008644911', '006390122', '002995392', '003851910', '010610039', '008509442', '004813750', '012416987', '012266099', '004172805', '012458190', '004578306', '005307283', '011192996', '003627455', '003240364', '008545603', '011666874', '004676964', '007762721', '009353679', '012734424', '012349560', '005900235', '009818394', '009580633', '003751118', '012365083', '005856571', '004366323', '006615511', '011822200', '005036804', '009782748', '009833034', '010157328', '003097222', '007663405', '008546484', '009091519', '004168174', '011445869', '005184911', '010580274', '008482582', '011462341', '009947389', '010257370']
periods = []
ftts = []
for tid in trig_ids:
    tid_index = np.where(np.array(trig_ids_0) == tid)[0]
    if len(tid_index) > 0:
        tid_index = tid_index[0]
        periods.append(periods_0[tid_index])
        ftts.append(ftts_0[tid_index])
    else:
        raise ValueError(f"tid {tid} not in original list")


# ATTENTION!!!! REMOVING PREVIOS INJS!

fold = '/home/projects/barakz/Collaboration-Kepler/orynai/run_15/output_run_15/'
for tid in trig_ids:
    tid_folder = os.path.join(fold, tid[:4], tid)
    filnames_tid = os.listdir(tid_folder)
    tce_filenames = [f for f in filnames_tid if '_injections.' in f]
    if len(tce_filenames) > 0:
        for filename in tce_filenames:
            os.remove(os.path.join(tid_folder, filename))



job_type_prefix = Injection_detection_job_prefix
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


for tid in trig_ids:
    for i in np.arange(int(N_injections_total // N_injections_chunk)  # how many times we need to repeat
                       + int(bool(N_injections_total % N_injections_chunk))):
        job_name = fop.compose_job_name(job_type_prefix=job_type_prefix,
                                        target_id=tid, additional_params=(str(i),))
        out_name = fop.compose_job_name(job_type_prefix=job_type_prefix,
                                        target_id=tid, additional_params=())
        fop.write_job_file(code_file=None,
                           code_filename=Injection_detection_runfile_filename,
                           code_file_dir=Run_files_folder,
                           jobname=job_name,
                           job_file=None,
                           job_filename=job_name+'.bsub',
                           job_file_dir=job_files_folder,
                           queue=Queue_submit,
                           mem_per_job_gb=Mem_per_job_gb_here,
                           array_indices=None,
                           output_file=None,
                           output_filename=out_name+'.out',
                           output_file_dir=output_files_folder,
                           error_file=None,
                           error_filename=out_name+'.err',
                           error_file_dir=error_files_folder,
                           output_error_overwrite=False,
                           arg_jobind=False,
                           other_args=[['target_id', tid], ['first_injection_load_parameters_index', i],
                                       ['job_name', job_name]])


# bsub -q short source ~/.bashrc; conda activate planets_env_amd_epyc; /home/projects/barakz/orynai/Kepler_Earths/WIS_Kepler_pipeline/runfiles/create_jobs_injections_detection.py
