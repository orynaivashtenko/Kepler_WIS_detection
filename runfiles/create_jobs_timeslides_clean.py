#!/usr/bin/env python3

import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import file_operations as fop
from pipeline_parameters import Triggers_for_slides_injections_file, N_slides_total, N_slides_chunk, Threshold_crossing_list_file
from runfiles_parameters import *
from trigger_functions import Index_tid, Index_per, Index_ftt, load_triggers_unique
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
trig_ids = trig_ids_0
periods = periods_0
ftts = ftts_0

# #
# # !!! temporary change
# trigger_list_file = Threshold_crossing_list_file
# # !!! end temporary
# trig_ids_0, integral_scores_0, stats_0, periods_0, ftts_0, gauss_stats_0, templates_0 = load_triggers_unique(
#     file=trigger_list_file
# )
# # with open(trigger_list_file, 'r') as tlf:
# #     trigs = tlf.readlines()
# # trig_ids_0 = [trig.split()[Index_tid] for trig in trigs]
# # periods_0 = [float(trig.split()[Index_per]) for trig in trigs]
# # ftts_0 = [float(trig.split()[Index_ftt]) for trig in trigs]
# #
# # # Select unique:
# # trig_ids = []
# # periods = []
# # ftts = []
# # for i_tid, tid in enumerate(trig_ids_0):
# #     if tid not in trig_ids:
# #         if (tid.isnumeric() and len(tid) == 9):
# #             trig_ids.append(tid)
# #             periods.append(periods_0[i_tid])
# #             ftts.append(ftts_0[i_tid])
#
# # !!!!!!!!!!!!!!!!!!
# # !!!!!!!!!!!!!!!!!!
# trig_ids = ['005107348', '006368905', '002994092', '007191646', '008644911', '006186182', '008509442', '001865864', '004912650', '005342061', '004270799', '009071593', '005792202', '009782748',
#        '010003058', '009605514', '002310993', '002995392', '003101923', '006934986', '009947389', '005184911', '006387557', '006390122', '008364232', '003646449', '009839821', '009514372',
#        '009268546', '007300182', '003558849', '008007174', '005900235', '003236705', '003758499', '008120820', '007829836', '009466429', '001718189', '005856571', '007663405', '008934103',
#        '009935983', '008349399', '002141783', '010749059', '010055126', '011446424', '009153824', '008807057', '006677841', '010272442', '006934045', '010089123', '005385304', '010798331',
#        '010610039', '010933306', '008264402', '009391817', '011968463', '010810859', '012416987', '012734396', '010467714', '011125797', '011446961', '011450414', '010658177', '011658036',
#        '009521317', '011913235', '003101178', '010967168', '012785320', '003122872', '004450844', '009020114']
# periods = []
# ftts = []
# for tid in trig_ids:
#     tid_index = np.where(np.array(trig_ids_0) == tid)[0]
#     if len(tid_index) > 0:
#         tid_index = tid_index[0]
#         periods.append(periods_0[tid_index])
#         ftts.append(ftts_0[tid_index])
#     else:
#         raise ValueError(f"tid {tid} not in original list")

job_type_prefix = Slides_clean_job_prefix
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
    ftt = ftts[ii]
    for i in range(int(N_slides_total // N_slides_chunk)  # how many times we need to repeat timeslides
                   + int(bool(N_slides_total % N_slides_chunk))):  # stupid way to make np.ceil (otherwise numpy import)
        # noinspection DuplicatedCode
        job_name = fop.compose_job_name(job_type_prefix=job_type_prefix,
                                        target_id=tid, additional_params=(str(i),))
        out_name = fop.compose_job_name(job_type_prefix=job_type_prefix,
                                        target_id=tid, additional_params=())
        fop.write_job_file(code_file=None,
                           code_filename=Slides_clean_runfile_filename,
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
                           other_args=[['target_id', tid], ['period', per], ['ftt', ftt]])

# bsub -q short source ~/.bashrc; conda activate planets_env_amd_epyc; /home/projects/barakz/orynai/Kepler_Earths/WIS_Kepler_pipeline/runfiles/create_jobs_timeslides_clean.py --trigger_list_file /home/projects/barakz/orynai/run_11/triggers_for_slides_injections.txt
