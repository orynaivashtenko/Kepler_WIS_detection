#!/usr/bin/env python3

import os
os.environ["OMP_NUM_THREADS"] = "1"  #
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"  #
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import target_class
import datetime
from runfiles_parameters import *
from file_operations import report_job_done
import traceback

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--target_id", type=str, nargs=1, help="KIC number", default="003097222")
parser.add_argument("--job_name", type=str, nargs=1, help="name of the job to report", default="003097222")
args = parser.parse_args()
Target_id = args.target_id[0]
job_name = args.job_name[0]

targ = target_class.KeplerTarget(target_id=Target_id,
                                 load_pickled_attributes=False)  # all the arguments are set to the defaults of the class
targ.clean_output_directory()  # !!! CAREFUL !!!

if not targ.is_good_for_processing:  # is villanova binary or is not in Gaia
    targ.print_dict_file()
else:  # for now, run the search only if it is not a villanova binary
    try:
        targ.run_main_search(save_to_threshold_crossing_list=True,
                             clean_dir_if_not_threshold_crossing=True,
                             save_multiple_triggers=True,
                             print_message_to_history=True)

    except Exception as exc:
        tb = sys.exc_info()
        list_str_exc = traceback.format_exception(exc, value=tb[1], tb=tb[2])
        with open(Error_list_file, 'a') as err_list_fil:
            print(targ.target_id, file=err_list_fil)
            print(datetime.datetime.now(), file=err_list_fil)
            for str_exc in list_str_exc:
                print(str_exc.strip(), file=err_list_fil)
            print('', file=err_list_fil)

report_job_done(job_name=job_name, jobs_monitor_folder=Jobs_monitor_folder)

# bsub -q physics-short /home/projects/barakz/orynai/Kepler_Earths/WIS_Kepler_pipeline/runfiles/create_jobs_main_search_all_stars.py
