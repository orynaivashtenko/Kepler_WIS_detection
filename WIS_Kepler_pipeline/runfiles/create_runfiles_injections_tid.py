#!/usr/bin/env python3

import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import file_operations as fop
from pipeline_parameters import *
from runfiles_parameters import *

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--target_ids', nargs='+', default=[])
parser.add_argument("--which", type=str, nargs=1,
                    help="which job to create: parameters, p, or detection, d", default="p")
args = parser.parse_args()
target_ids = args.target_ids
which = args.which[0]

queues = ['risk']

Output_location_wexac = '/home/labs/barakz/Collaboration-Kepler/orynai/output_run_10'
# noinspection DuplicatedCode
for itid, tid in enumerate(target_ids):
    queue = queues[0]  # [itid % 2]
    if which == 'p':
        fop.write_job_file(code_file=None,
                           code_filename=Injection_parameters_runfile_filename,
                           code_file_dir=Run_files_folder,
                           jobname=Injection_parameters_job_filename_prefix+'_'+tid+'.bsub',
                           job_file=None,
                           job_filename=Injection_parameters_job_filename_prefix+'_'+tid+'.bsub',
                           job_file_dir=Job_files_folder,
                           queue=queue,
                           mem_per_job_gb=Mem_per_job_gb,
                           array_indices=None,
                           output_file=None,
                           output_filename=Injection_parameters_output_filename_prefix+'_'+tid+'.out',
                           output_file_dir=Output_files_folder,
                           error_file=None,
                           error_filename=Injection_parameters_error_filename_prefix+'_'+tid+'.out',
                           error_file_dir=Error_files_folder,
                           output_error_overwrite=False,
                           arg_jobind=False,
                           other_args=[['target_id', tid]])
    else:
        for i in np.arange(int(N_injections_total // N_injections_chunk)  # how many times we need to repeat
                           + int(bool(N_injections_total % N_injections_chunk))):
            queue = queues[0]
            fop.write_job_file(code_file=None,
                               code_filename=Injection_detection_runfile_filename,
                               code_file_dir=Run_files_folder,
                               jobname=Injection_detection_job_filename_prefix+'_'+tid+'_'+str(i)+'.bsub',
                               job_file=None,
                               job_filename=Injection_detection_job_filename_prefix+'_'+tid+'_'+str(i)+'.bsub',
                               job_file_dir=Job_files_folder,
                               queue=queue,
                               mem_per_job_gb=Mem_per_job_gb,
                               array_indices=None,
                               output_file=None,
                               output_filename=Injection_detection_output_filename_prefix+'_'+tid+'.out',
                               output_file_dir=Output_files_folder,
                               error_file=None,
                               error_filename=Injection_detection_error_filename_prefix+'_'+tid+'.out',
                               error_file_dir=Error_files_folder,
                               output_error_overwrite=False,
                               arg_jobind=False,
                               other_args=[['target_id', tid], ['first_injection_load_parameters_index', i]])


# bsub -q risk -R "rusage[mem=500MB]" /home/labs/barakz/orynai/Kepler_Earths/WIS_Kepler_pipeline/runfiles/create_runfiles_injections_tid.py.py
