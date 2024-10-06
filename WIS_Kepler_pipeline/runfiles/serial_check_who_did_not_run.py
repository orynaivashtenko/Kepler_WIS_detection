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

# import numpy as np
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import pipeline_parameters as pp
from auxiliary_functions import check_dict_keyword
from file_operations import write_job_file
from runfiles.runfiles_parameters import *

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--tids_list_file", type=str, nargs=1,
                    help="path to file where to take tids. If empty, will go over all the folders",
                    default='')
parser.add_argument("--not_run_tids_file", type=str, nargs=1, help="path to file where to save non-run tids",
                    default=os.path.join(pp.Run_technical_folder, pp.Tids_that_did_not_run_filename))
parser.add_argument("--output_folder", type=str, nargs=1, help="path to the place where outputs of the pipeline are saved",
                    default=pp.Output_location)
parser.add_argument("--keyword_output", type=str, nargs=1, help="path to the place where outputs of the pipeline are saved",
                    default=pp.Dict_target_txt_filename_postfix)

args = parser.parse_args()
tids_list_file = args.tids_list_file
not_run_tids_file = args.not_run_tids_file
Output_folder = args.output_folder
Keyword_output = args.keyword_output

# If defaults are used, they will not be lists. If they are parsed, they will be lists
if isinstance(tids_list_file, list):
    tids_list_file = tids_list_file[0]
if isinstance(not_run_tids_file, list):
    not_run_tids_file = not_run_tids_file[0]
if isinstance(Output_folder, list):
    Output_folder = Output_folder[0]
if isinstance(Keyword_output, list):
    Keyword_output = Keyword_output[0]

Create_job_files = True


def write_the_job(target_id):
    search_runfile_filename = Main_search_runfile_filename
    write_job_file(code_file=None,
                   code_filename=search_runfile_filename,
                   code_file_dir=Run_files_folder,
                   jobname=Main_search_job_filename_prefix + '_' + target_id + '.bsub',
                   job_file=None,
                   job_filename=Main_search_job_filename_prefix + '_' + target_id + '.bsub',
                   job_file_dir=Job_files_folder,
                   queue=Queue_submit,
                   mem_per_job_gb=Mem_per_job_gb,
                   array_indices=None,
                   output_file=None,
                   output_filename=Main_search_output_filename_prefix + '_' + target_id + '.out',
                   output_file_dir=Output_files_folder,
                   error_file=None,
                   error_filename=Main_search_error_filename_prefix + '_' + target_id + '.out',
                   error_file_dir=Error_files_folder,
                   output_error_overwrite=False,
                   arg_jobind=False,
                   other_args=[['target_id', target_id]])


def there_is_output(output_folder):
    if not os.path.exists(output_folder):
        return False
    else:
        filenames = os.listdir(output_folder)
        keyword_list = [filename for filename in filenames if Keyword_output in filename]
        if len(keyword_list) == 0:
            return False
    return True


Input_folder = pp.Data_location

not_run_tids = []
if tids_list_file == '':
    for prefix in os.listdir(Input_folder):
        for tid in os.listdir(os.path.join(Input_folder, prefix)):
            output_folder_tid = os.path.join(Output_folder, prefix, tid)
            if not there_is_output(output_folder_tid):
                add_to_list = True
                # Maybe the thing was bad for processing, need to check it:
                if pp.Dict_target_txt_filename_postfix not in Keyword_output:
                    dict_file = os.path.join(output_folder_tid, tid + '_' + pp.Dict_target_txt_filename_postfix)
                    if os.path.exists(dict_file):
                        is_good_for_processing = check_dict_keyword(target_id=tid, output_folder=output_folder_tid,
                                                                    keyword='is_good_for_processing')
                        if is_good_for_processing == 'False':
                            add_to_list = False
                if add_to_list:
                    not_run_tids.append(tid)
                    if Create_job_files:
                        write_the_job(target_id=tid)
else:
    with open(tids_list_file, 'r') as fil:
        tid_list = fil.readlines()
    tid_list = [line.strip() for line in tid_list]
    for tid in tid_list:
        prefix = tid[:4]
        output_folder_tid = os.path.join(Output_folder, prefix, tid)
        if not there_is_output(output_folder_tid):
            add_to_list = True
            # Maybe the thing was bad for processing, need to check it:
            if pp.Dict_target_txt_filename_postfix not in Keyword_output:
                dict_file = os.path.join(output_folder_tid, tid + '_' + pp.Dict_target_txt_filename_postfix)
                if os.path.exists(dict_file):
                    is_good_for_processing = check_dict_keyword(target_id=tid, output_folder=output_folder_tid,
                                                                keyword='is_good_for_processing')
                    if is_good_for_processing == 'False':
                        add_to_list = False
            if add_to_list:
                not_run_tids.append(tid)
                if Create_job_files:
                    write_the_job(target_id=tid)

with open(not_run_tids_file, 'w') as fil:
    for tid in not_run_tids:
        print(tid, file=fil)

# bsub -q short -R "rusage[mem=2GB]" /home/labs/barakz/orynai/Kepler_Earths/WIS_Kepler_pipeline/runfiles/serial_check_who_did_not_run.py
# bsub -q short -R "rusage[mem=2GB]" /home/labs/barakz/orynai/Kepler_Earths/WIS_Kepler_pipeline/runfiles/serial_check_who_did_not_run.py --tids_list_file /home/labs/barakz/orynai/run_11/selected_tids_run_11_rerun_main_search.txt --not_run_tids_file /home/labs/barakz/orynai/run_11/not_run_tids.txt --output_folder /home/labs/barakz/Collaboration-Kepler/orynai/output_run_11_alternative
# bsub -q short -R "rusage[mem=2GB]" /home/labs/barakz/orynai/Kepler_Earths/WIS_Kepler_pipeline/runfiles/serial_check_who_did_not_run.py --not_run_tids_file /home/labs/barakz/orynai/run_11/not_run_tids_scrambled_run.txt --output_folder /home/labs/barakz/Collaboration-Kepler/orynai/output_run_11 --keyword_output _best_scrambled_periodic_scores_all_p
