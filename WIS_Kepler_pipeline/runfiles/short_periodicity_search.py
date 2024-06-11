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

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--target_id", type=str, nargs=1, help="KOI number", default="003097222")
args = parser.parse_args()
Target_id = args.target_id[0]

Saving_maximal_scores_file = '/home/labs/barakz/orynai/run_11/short_periods_veto_maximal_triggers.txt'
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

best_candidates_concat = []
for ii, pers in enumerate(targ.periods_in_chunks):
    has_enough_transits = targ.klc.if_has_potentially_enough_transits(
        periods_to_check=(pers[0], pers[-1]),
        min_required_nonzero_transits_number=min_required_nonzero_transits_number,
        min_required_nonzero_transits_fraction=min_required_nonzero_transits_fraction)
    if has_enough_transits:
        targ.klc.periodicity_search(p_min=None, p_max=None, periods=pers,
                                    delete_full_stats=True, force_interleave_shifts=False, delete_periodic_interleaved=True,
                                    save_uncorrected_periodic=False, delete_non_marginalized=False,
                                    overlap_period_bins=overlap_period_bins)
        targ.klc.get_best_trigger(apply_veto_mes=True, in_gaussianized=False,
                                  min_required_nonzero_transits_number=min_required_nonzero_transits_number,
                                  min_required_nonzero_transits_fraction=min_required_nonzero_transits_fraction,
                                  log_baseline_integral_score_rate=targ.log_baseline_integral_score_rate,
                                  save_veto_masks=False)
        targ.klc.get_best_trigger(apply_veto_mes=True, in_gaussianized=True,
                                  min_required_nonzero_transits_number=min_required_nonzero_transits_number,
                                  min_required_nonzero_transits_fraction=min_required_nonzero_transits_fraction,
                                  log_baseline_integral_score_rate=targ.log_baseline_integral_score_rate,
                                  save_veto_masks=False)
        targ.klc.make_best_periodic_scores()
        if np.isnan(targ.klc.best_period):
            targ.print_problem(text='No trigger found after veto MES for pers=' + str(pers[0]) + '-' + str(pers[-1]),
                               file=targ.problems_file_location)
        best_candidates_concat.append(np.array(targ.klc.best_periodic_scores_data, dtype=object))
np.save(os.path.join(targ.output_location, targ.target_id + '_best_periodic_scores_'+str(targ.p_min)+'_'+str(targ.p_max)+'.npy'),
        np.array(best_candidates_concat, dtype=object), allow_pickle=True)
targ.save_best_candidate(candidate_threshold=pp.Stat_candidate_search,
                         threshold_crossing_list_file=None,
                         threshold_crossing_list_location=None,
                         threshold_crossing_list_filename=None,
                         save_to_threshold_crossing_list=False,
                         best_candidates_concat=best_candidates_concat)

with open(Saving_maximal_scores_file, 'a') as fil1:  # will save here these good targets
    print(targ.target_id, targ.best_stat, targ.best_per, targ.best_ftt, targ.best_gauss_stat, targ.best_integral_score,
          targ.best_g_stat, targ.best_g_per, targ.best_g_ftt, targ.best_g_gauss_stat, targ.best_g_integral_score,
          sep=' ', end='\n', file=fil1)

# bsub -q physics-short -R "rusage[mem=3.9GB]" /home/labs/barakz/orynai/Kepler_Earths/WIS_Kepler_pipeline/runfiles/short_periodicity_search.py --target_id
