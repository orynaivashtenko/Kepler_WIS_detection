#!/usr/bin/env python3

import os

# import auxiliary_functions

os.environ["OMP_NUM_THREADS"] = "1"  #
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"  #
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import numpy as np
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import target_class
import search_pipeline as spip
import pipeline_parameters as pp
from file_operations import report_job_done
from runfiles_parameters import Jobs_monitor_folder
from trigger_functions import print_trigger_line  # , get_trigger_from_target_id, Indices_dict, copy_trigger_between_files
from auxiliary_functions import get_saving_file_path
import traceback
from runfiles_parameters import Error_list_file
import datetime

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--target_id", type=str, nargs=1, help="KIC number", default="003097222")
parser.add_argument("--job_name", type=str, nargs=1, help="name of the job to report", default="003097222")
args = parser.parse_args()
Target_id = args.target_id[0]
job_name = args.job_name[0]

Saving_maximal_scores_file = pp.Maximal_short_periodicity_triggers_file
Saving_maximal_g_scores_file = pp.Maximal_short_periodicity_g_triggers_file
P_short_min = pp.P_short_min
P_short_max = pp.P_short_max

try:
    targ = target_class.KeplerTarget(target_id=Target_id,
                                     load_pickled_attributes=False)  # all the arguments are set to the defaults like main search

    # We will have here a simplified version of the search. We do not change anything in the saved parameters after the main search!

    targ.unpickle_small_attributes()
    # Just in case:
    targ.best_stat = None
    targ.best_per = None
    targ.best_temp = None
    targ.best_ftt = None
    targ.best_gauss_stat = None
    targ.best_int_score = None
    targ.best_g_stat = None
    targ.best_g_per = None
    targ.best_g_temp = None
    targ.best_g_ftt = None
    targ.best_g_gauss_stat = None
    targ.best_g_int_score = None

    targ.get_connected_klc(reinitialize=True, mask_koi_planet=True, save_conditioned_data=True, save_connected_stats=True)
    targ.klc.interleave_shifts()

    duty_cycle = spip.Duty_cycle
    per_phases_per_chunk = pp.Per_phases_per_chunk
    overlap_period_bins = spip.P_bins_overlap
    min_required_nonzero_transits_number = spip.Min_required_nonzero_transits_number
    min_required_nonzero_transits_fraction = spip.Min_required_nonzero_transits_fraction

    targ.p_min = P_short_min
    targ.p_max = P_short_max
    targ.periods_in_chunks = targ.klc.make_periods_chunks(p_min=targ.p_min, p_max=targ.p_max,
                                                          per_phases_per_chunk=targ.per_phases_per_chunk,
                                                          duty_cycle=targ.duty_cycle, dt=targ.klc.dt_days, t_mission=targ.klc.t_mission)
    best_candidates_concat, failures = targ.search_periodicity_over_chunks(
        period_chunks=targ.periods_in_chunks,
        force_interleave_shifts=False,
        min_required_nonzero_transits_number=min_required_nonzero_transits_number,
        min_required_nonzero_transits_fraction=min_required_nonzero_transits_fraction, overlap_period_bins=overlap_period_bins,
        report_exceptions=True, problem_description='Short: ',
        return_failures=True, save_search_time=False, save_veto_masks=False)

    file_path = get_saving_file_path(output_location=targ.output_location, target_id=targ.target_id,
                                     filename_postfix=pp.Best_short_periodic_scores_filename_postfix, file_format='npy')
    np.save(file_path, np.array(best_candidates_concat, dtype=object), allow_pickle=True)
    targ.save_best_candidate(candidate_threshold=pp.Score_candidate_search,
                             threshold_crossing_list_file=None,
                             threshold_crossing_list_location=None,
                             threshold_crossing_list_filename=None,
                             threshold_g_crossing_list_filename=None,
                             save_to_threshold_crossing_list=False,
                             best_candidates_concat=best_candidates_concat)

    with open(Saving_maximal_scores_file, 'a') as fil1:  # will save here these good targets
        print_trigger_line(target_id=targ.target_id, integral_score=targ.best_int_score, stat=targ.best_stat, per=targ.best_per,
                           ftt=targ.best_ftt, gauss_stat=targ.best_gauss_stat, template=targ.best_temp,
                           file=fil1, sep=' ', end='\n')

    with open(Saving_maximal_g_scores_file, 'a') as fil1:  # will save here these good targets
        print_trigger_line(target_id=targ.target_id, integral_score=targ.best_g_int_score, stat=targ.best_g_stat, per=targ.best_g_per,
                           ftt=targ.best_g_ftt, gauss_stat=targ.best_g_gauss_stat, template=targ.best_g_temp,
                           file=fil1, sep=' ', end='\n')

    # trigger_main_search = get_trigger_from_target_id(Target_id, trigger_list_file=pp.Threshold_crossing_list_file)
    # int_score_main = trigger_main_search[Indices_dict['int_score']]
    # if int_score_main > targ.best_int_score:
    #     copy_trigger_between_files(target_id=Target_id, source_file=pp.Threshold_crossing_list_file,
    #                                destination_file=pp.Triggers_for_slides_injections_file)
except Exception as exc:
    tb = sys.exc_info()
    list_str_exc = traceback.format_exception(exc, value=tb[1], tb=tb[2])
    with open(Error_list_file, 'a') as err_list_fil:
        print(Target_id, file=err_list_fil)
        print(datetime.datetime.now(), file=err_list_fil)
        for str_exc in list_str_exc:
            print(str_exc.strip(), file=err_list_fil)
        print('', file=err_list_fil)

report_job_done(job_name=job_name, jobs_monitor_folder=Jobs_monitor_folder)
