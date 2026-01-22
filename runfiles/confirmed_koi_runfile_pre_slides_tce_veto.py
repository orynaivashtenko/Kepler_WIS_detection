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
import tce_class
from pipeline_parameters import Data_location, Target_pixels_location
# from locations import Pickle_tce_small_attributes_filename_postfix
import locations as locs
# from runfiles_parameters import Error_list_file, Jobs_monitor_folder
from file_operations import report_job_done
from confirmed_koi_search_create import (Kepler_confirmed_koi_output_folder, Kepler_confirmed_koi_problems_folder,
                                         Kepler_confirmed_koi_plots_folder, Kepler_confirmed_koi_run_folder)

import traceback
import datetime

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--target_id", type=str, nargs=1, help="KIC number", default="003097222")
parser.add_argument("--job_name", type=str, nargs=1, help="name of the job to report", default="003097222")
args = parser.parse_args()
Target_id = args.target_id[0]
job_name = args.job_name[0]

output_location = str(os.path.join(Kepler_confirmed_koi_output_folder, Target_id))
plots_output_location = Kepler_confirmed_koi_plots_folder
Error_list_file = os.path.join(Kepler_confirmed_koi_run_folder, 'confirmed_koi_pre_slides_tce_veto_errors.txt')

# Only if the score and the period are good enough for postprocessing, do the postprocessing:
try:
    tce = tce_class.WisTCE(target_id=Target_id,
                           output_location=output_location, data_location=Data_location,
                           target_pixels_location=Target_pixels_location,
                           problems_folder=Kepler_confirmed_koi_problems_folder,
                           load_pickled_attributes=False, load_target_attributes=False)
    # ---------------------------------
    prefixes_to_clean = [
        locs.Dict_tce_txt_filename_postfix,
        locs.Pickle_tce_small_attributes_filename_postfix,
        locs.Folded_binned_wh_flux_model_even_odd_filename_postfix,
        locs.Periodic_stats_centroid_y_filename_postfix,
        locs.Connected_stats_centroid_y_filename_postfix,
        locs.Periodic_stats_centroid_x_filename_postfix,
        locs.Connected_stats_centroid_x_filename_postfix,
        locs.Periodic_stats_bg_pixels_filename_postfix,
        locs.Connected_stats_bg_pixels_filename_postfix,
        locs.Periodic_stats_star_pixels_filename_postfix,
        locs.Connected_stats_star_pixels_filename_postfix,
        locs.Fold_residuals_flux_td_chi_filename_postfix,
        locs.Folded_binned_flux_template_filename_postfix,
        locs.Mf_scores_residuals_filename_postfix,
        locs.Fold_residuals_flux_fd_chi2_filename_postfix
    ]
    for ff in os.listdir(tce.output_location):
        for prefix in prefixes_to_clean:
            if prefix in ff:
                os.remove(os.path.join(tce.output_location, ff))
    # Clean the pickle file to make sure we run from scratch:
    # file_path_unpickling = get_saving_file_path(output_location=tce.output_location, target_id=tce.target_id,
    #                                             filename_postfix=Pickle_tce_small_attributes_filename_postfix,
    #                                             file_format='pickle')
    # if os.path.exists(file_path_unpickling):
    #     os.remove(file_path_unpickling)
    # ---------------------------------
    tce.unpickle_small_attributes(load_target_attributes=True, overwrite_existing_by_target=True,
                                  overwrite_location_parameters=False)
    tce.run_all_checks_exclusive(print_message_to_history=False, save_byproducts_to_file=True,
                                 pre_filtering_history_targets_file=None,
                                 produce_plots=True, no_show=True, save_plots=True,
                                 plots_output_location=plots_output_location,
                                 update_pickling=True, enforce_all_checks=True,
                                 include_veto_short_periodicity=False)
    tce.pickle_small_attributes()
    tce.print_dict_file()

    # if tce.is_good_pre_filtering is True:
    #     # copy_trigger_between_files(target_id=Target_id, source_file=pp.Threshold_crossing_list_file,
    #     #                            destination_file=pp.Triggers_for_slides_injections_file)
    #     copy_trigger_between_files(target_id=Target_id, source_file=Threshold_crossing_list_file,
    #                                destination_file=Triggers_prefiltered_file)
except Exception as exc:
    tb = sys.exc_info()
    list_str_exc = traceback.format_exception(exc, value=tb[1], tb=tb[2])
    with open(Error_list_file, 'a') as err_list_fil:
        print(Target_id, file=err_list_fil)
        print(datetime.datetime.now(), file=err_list_fil)
        for str_exc in list_str_exc:
            print(str_exc.strip(), file=err_list_fil)
        print('', file=err_list_fil)
# report_job_done(job_name=job_name, jobs_monitor_folder=Jobs_monitor_folder, str_to_add='')
