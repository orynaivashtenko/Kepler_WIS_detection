#!/usr/bin/env python3

import os
import sys
# noinspection DuplicatedCode
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import file_operations as fop
from pipeline_parameters import np, Triggers_for_slides_injections_file, N_injections_total, N_injections_chunk
from runfiles_parameters import *
from trigger_functions import Index_tid

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--trigger_list_file", type=str, nargs=1, help="path to file where to take triggers",
                    default=Triggers_for_slides_injections_file)
args = parser.parse_args()
trigger_list_file = args.trigger_list_file[0]

with open(trigger_list_file, 'r') as tlf:
    trig_ids = tlf.readlines()
trig_ids = [tid.split()[Index_tid] for tid in trig_ids]

for tid in trig_ids:
    # noinspection DuplicatedCode
    fop.write_job_file(code_file=None,
                       code_filename=Plotting_runfile_filename,
                       code_file_dir=Run_files_folder,
                       jobname=Plotting_job_filename_prefix+'_'+tid+'.bsub',
                       job_file=None,
                       job_filename=Plotting_job_filename_prefix+'_'+tid+'.bsub',
                       job_file_dir=Job_files_folder,
                       queue=Queue_submit,
                       mem_per_job_gb=Mem_per_job_gb,
                       array_indices=None,
                       output_file=None,
                       output_filename=Plotting_output_filename_prefix+'_'+tid+'.out',
                       output_file_dir=Output_files_folder,
                       error_file=None,
                       error_filename=Plotting_error_filename_prefix+'_'+tid+'.out',
                       error_file_dir=Error_files_folder,
                       output_error_overwrite=False,
                       arg_jobind=False,
                       other_args=[['target_id', tid]])
#
#     # if len([ff for ff in os.listdir(os.path.join(Output_location_wexac, tid[:4], tid)) if Injections_filename_postfix in ff]) > 0:
#     #     # noinspection DuplicatedCode
#     #     fop.write_job_file(code_file=None,
#     #                        code_filename=Pre_slides_tce_veto_code_filename,
#     #                        code_file_dir=Run_files_folder,
#     #                        jobname=Target_pixels_veto_job_filename_prefix+'_'+tid+'.bsub',
#     #                        job_file=None,
#     #                        job_filename=Target_pixels_veto_job_filename_prefix+'_'+tid+'.bsub',
#     #                        job_file_dir=Job_files_folder,
#     #                        queue=Queue_submit,
#     #                        mem_per_job_gb=Mem_per_job_gb,
#     #                        array_indices=None,
#     #                        output_file=None,
#     #                        output_filename=Target_pixels_veto_output_filename_prefix+'_'+tid+'.out',
#     #                        output_file_dir=Output_files_folder,
#     #                        error_file=None,
#     #                        error_filename=Target_pixels_veto_error_filename_prefix+'_'+tid+'.out',
#     #                        error_file_dir=Error_files_folder,
#     #                        output_error_overwrite=False,
#     #                        arg_jobind=False,
#     #                        other_args=[['target_id', tid]])
#
#
#
# # bsub -q physics-short -sp 100 /home/labs/barakz/orynai/Kepler_Earths/WIS_Kepler_pipeline/runfiles/create_jobs_plotting.py
# # bsub -q physics-short /home/labs/barakz/orynai/Kepler_Earths/WIS_Kepler_pipeline/runfiles/create_jobs_plotting.py
