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
from pipeline_parameters import (Threshold_crossing_list_file, Triggers_prefiltered_file,
                                 Output_location, Data_location, Target_pixels_location,
                                 Score_candidate_postprocess, Per_min_candidate)
# from locations import Pickle_tce_small_attributes_filename_postfix
import locations as locs
from runfiles_parameters import Error_list_file
from trigger_functions import copy_trigger_between_files, Indices_dict, get_trigger_from_target_id
from auxiliary_functions import get_saving_file_path
import traceback
import datetime

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--target_id", type=str, nargs=1, help="KIC number", default="003097222")
args = parser.parse_args()
Target_id = args.target_id[0]

# Taking the trigger values for the score and the period
trigger_main_search = get_trigger_from_target_id(Target_id, trigger_list_file=Threshold_crossing_list_file)
int_score_search = float(trigger_main_search[Indices_dict['int_score']])
per_search = float(trigger_main_search[Indices_dict['per']])

# Only if the score and the period are good enough for postprocessing, do the postprocessing:
if int_score_search >= Score_candidate_postprocess and per_search >= Per_min_candidate:
    try:
        tce = tce_class.WisTCE(target_id=Target_id,
                               output_location=Output_location, data_location=Data_location,
                               target_pixels_location=Target_pixels_location,
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
        tce.run_all_checks_exclusive(print_message_to_history=True, save_byproducts_to_file=True,
                                     produce_plots=True, no_show=True, save_plots=True,
                                     update_pickling=True, enforce_all_checks=True)
        tce.pickle_small_attributes()
        tce.print_dict_file()

        if tce.is_good_pre_filtering is True:
            # copy_trigger_between_files(target_id=Target_id, source_file=pp.Threshold_crossing_list_file,
            #                            destination_file=pp.Triggers_for_slides_injections_file)
            copy_trigger_between_files(target_id=Target_id, source_file=Threshold_crossing_list_file,
                                       destination_file=Triggers_prefiltered_file)
    except Exception as exc:
        tb = sys.exc_info()
        list_str_exc = traceback.format_exception(exc, value=tb[1], tb=tb[2])
        with open(Error_list_file, 'a') as err_list_fil:
            print(Target_id, file=err_list_fil)
            print(datetime.datetime.now(), file=err_list_fil)
            for str_exc in list_str_exc:
                print(str_exc.strip(), file=err_list_fil)
            print('', file=err_list_fil)
