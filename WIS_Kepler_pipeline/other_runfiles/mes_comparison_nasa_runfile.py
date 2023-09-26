#!/usr/bin/env python3

# import pickle
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import target_class
import pipeline_parameters as pp
# from mes_comparison_nasa_job_creator import Mes_comparison_folder
# import file_operations as fop
from search_pipeline_parameters import Duty_cycle
# import traceback

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--koi_index", type=str, nargs=1, help="index of KOI number from the list", default="0")
args = parser.parse_args()
koi_index = int(args.koi_index[0])

data_folder = '/home/labs/barakz/Collaboration-Kepler/KeplerData'
output_folder = '/home/labs/barakz/Collaboration-Kepler/orynai/known_kois_run_8'
results_list_file = os.path.join(output_folder, 'faint_nasa_kois_results.txt')
Low_mes_file = '/home/labs/barakz/Collaboration-Kepler/orynai/koi_all_mes_less_20.csv'

with open(Low_mes_file, 'r') as fil:
    lines = fil.readlines()

lines = [line for line in lines if len(line.strip()) > 5 and line.strip()[0] != '#' and line.strip()[0].isnumeric()]
line = lines[koi_index]
kepid, koi_disposition, koi_pdisposition, koi_period, koi_time0bk, koi_duration, koi_depth, koi_max_sngle_ev, koi_max_mult_ev, koi_model_snr \
    = line.strip().split(',')
kepid = kepid.rjust(9, '0')
koi_period = float(koi_period) if koi_period != '' else None
koi_time0bk = float(koi_time0bk) if koi_time0bk != '' else None
koi_duration = float(koi_duration) if koi_duration != '' else None
koi_depth = float(koi_depth) if koi_depth != '' else None
koi_max_sngle_ev = float(koi_max_sngle_ev) if koi_max_sngle_ev != '' else None
koi_max_mult_ev = float(koi_max_mult_ev) if koi_max_mult_ev != '' else None
koi_model_snr = float(koi_model_snr) if koi_model_snr != '' else None

if koi_period is not None and koi_time0bk is not None:
    targ = target_class.KeplerTarget(target_id=kepid, data_location=data_folder, output_location=output_folder,
                                     load_pickled_attributes=False)

    targ.run_ses_search(mask_koi_planet=False,
                        save_conditioned_data=True,
                        conditioned_data_filename_postfix=pp.Conditioned_data_filename_postfix,
                        save_no_veto_no_score_cor_stats=True,
                        no_veto_no_score_cor_filename_postfix=pp.No_veto_no_score_cor_stats_filename_postfix,
                        save_connected_stats=True,
                        connected_stats_filename_postfix=pp.Connected_statistic_data_filename_postfix)
    p_min = koi_period - 0.5
    p_max = koi_period + 0.5
    targ.klc.sum_periods(pmin=p_min, pmax=p_max, duty_cycle=Duty_cycle)
    targ.klc.save_best_candidates(save_to_file=True, n_per_temp=20)
    targ.klc.get_best_trigger()

    with open(results_list_file, 'a') as fil1:  # will save here these target_ids outputs
        print(targ.target_id, koi_period,
              koi_time0bk, koi_duration, koi_depth, koi_max_sngle_ev, koi_max_mult_ev, koi_model_snr,
              targ.klc.best_stat, targ.klc.best_per, targ.klc.best_first_transit_time, targ.klc.best_temp,
              targ.klc.summed_raw_stats, targ.klc.summed_vars, targ.klc.summed_trends,
              sep=' ', end='\n', file=fil1)




# if not targ.is_good_for_processing:  # is villanova binary or is not in Gaia
#     targ.print_dict_file()
# else:  # for now, run the search only if it is not a villanova binary
#     targ.run_main_search(save_to_threshold_crossing_list=False,
#                          clean_dir_if_not_threshold_crossing=False,
#                          mask_koi_planet=False,
#                          p_min=Period-1,
#                          p_max=Period+1,
#                          p_chunks=None)


