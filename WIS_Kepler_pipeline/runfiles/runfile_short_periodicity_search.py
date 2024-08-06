#!/usr/bin/env python3

import os
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
from trigger_class import print_trigger_line

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--target_id", type=str, nargs=1, help="KOI number", default="003097222")
args = parser.parse_args()
Target_id = args.target_id[0]

Saving_maximal_scores_file = pp.Maximal_short_periodicity_triggers_file
Saving_maximal_g_scores_file = pp.Maximal_short_periodicity_g_triggers_file
P_short_min = 10
P_short_max = 30

targ = target_class.KeplerTarget(target_id=Target_id,
                                 load_pickled_attributes=False)  # all the arguments are set to the defaults like main search

# We will have here a simplified version of the search. We do not change anything in the saved parameters after the main search!

targ.unpickle_small_attributes()
# targ.initialize_kepler_light_curve()
# targ.klc.load_conditioned_data()
# targ.klc.load_connected_stats()
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

np.save(os.path.join(targ.output_location, targ.target_id + '_best_periodic_scores_'+str(targ.p_min)+'_'+str(targ.p_max)+'.npy'),
        np.array(best_candidates_concat, dtype=object), allow_pickle=True)
targ.save_best_candidate(candidate_threshold=pp.Stat_candidate_search,
                         threshold_crossing_list_file=None,
                         threshold_crossing_list_location=None,
                         threshold_crossing_list_filename=None,
                         threshold_g_crossing_list_filename=None,
                         save_to_threshold_crossing_list=False,
                         best_candidates_concat=best_candidates_concat)

with open(Saving_maximal_scores_file, 'a') as fil1:  # will save here these good targets
    print_trigger_line(target_id=targ.target_id, integral_score=targ.best_integral_score, stat=targ.best_stat, per=targ.best_per,
                       ftt=targ.best_ftt, gauss_stat=targ.best_gauss_stat, template=targ.best_temp,
                       file=fil1, sep=' ', end='\n')

with open(Saving_maximal_g_scores_file, 'a') as fil1:  # will save here these good targets
    print_trigger_line(target_id=targ.target_id, integral_score=targ.best_g_integral_score, stat=targ.best_g_stat, per=targ.best_g_per,
                       ftt=targ.best_g_ftt, gauss_stat=targ.best_g_gauss_stat, template=targ.best_g_temp,
                       file=fil1, sep=' ', end='\n')

# bsub -q physics-short -R "rusage[mem=3.9GB]" /home/labs/barakz/orynai/Kepler_Earths/WIS_Kepler_pipeline/runfiles/runfile_short_periodicity_search.py --target_id
