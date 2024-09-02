#!/usr/bin/env python3

# TODO work on this
# #########################################################################################################################
# THIS FILE CHECKS THAT THE TRIGGER IS NOT A BRIGHT KOI AND IS NOT TOO SHORT
# #########################################################################################################################

import os

core_num = "1"
os.environ["OMP_NUM_THREADS"] = core_num  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = core_num  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = core_num  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = core_num  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = core_num  # export NUMEXPR_NUM_THREADS=6

import numpy as np
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from pipeline_parameters import Triggers_short_periodicity_filtered_file, Triggers_for_slides_injections_file
from trigger_functions import load_triggers, print_trigger_line

Threshold_integral_score_for_now = 48.

target_ids, integral_scores, stats, pers, ftts, gauss_stats, templates = load_triggers(file=Triggers_short_periodicity_filtered_file)


# ##### CLEANING #########
with open(Triggers_for_slides_injections_file, 'w'):
    pass
# ######################

with open(Triggers_for_slides_injections_file, 'w') as fil:
    for i in range(len(target_ids)):
        if integral_scores[i] >= Threshold_integral_score_for_now:
            if pers[i] >= 50:
                print_trigger_line(target_id=target_ids[i], integral_score=integral_scores[i], stat=stats[i], per=pers[i],
                                   ftt=ftts[i], gauss_stat=gauss_stats[i], template=templates[i],
                                   file=fil, sep=' ', end='\n')

# bsub -q short -R "rusage[mem=2GB]" /home/labs/barakz/orynai/Kepler_Earths/WIS_Kepler_pipeline/runfiles/serial_select_triggers_for_timeslides.py
