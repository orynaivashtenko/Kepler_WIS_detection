#!/usr/bin/env python3

import time
from numpy.random import rand
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import target_class
# from pipeline_parameters import *
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--target_id", type=str, nargs=1, help="KOI number", default="003097222")
args = parser.parse_args()
Target_id = args.target_id[0]

targ = target_class.KeplerTarget(target_id=Target_id,
                                 load_pickled_attributes=False)  # all the arguments are set to the defaults of the class
targ.unpickle_small_attributes()
# The KeplerLightCurve object is created inside the method, no need in initialization.
rt = rand() * 10.  # make it sleep random number of seconds because maybe many jobs will want to access the same file...
time.sleep(rt)
targ.run_timeslides(use_period_chunks=True, use_shuffling_pieces=True)  # ATTENTION: NEW!
