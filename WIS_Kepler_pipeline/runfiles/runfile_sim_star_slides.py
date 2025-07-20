#!/usr/bin/env python3

import time
import numpy.random as npr
import os

from locations import Run_name

os.environ["OMP_NUM_THREADS"] = "1"  #
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"  #
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import target_class
# from pipeline_parameters import *
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--target_id", type=str, nargs=1, help="KIC number", default="000000000")
parser.add_argument("--period", type=str, nargs=1, help="period, days", default="100")
args = parser.parse_args()
Target_id = args.target_id
if isinstance(Target_id, list):
    Target_id = Target_id[0]
Period = args.period
if isinstance(Period, list):
    Period = Period[0]
Period = float(Period)

targ = target_class.KeplerTarget(target_id=Target_id,
                                 simulated_red_star=True,  # Warning: in run_12, it was a saved white star
                                 load_pickled_attributes=False,
                                 data_location=None,
                                 output_location='/home/projects/barakz/Collaboration-Kepler/orynai/simulated_star_' + Run_name)
targ.unpickle_small_attributes()
# The KeplerLightCurve object is created inside the method, no need in initialization.
npr.seed(None)
rt = npr.rand() * 10.  # make it sleep random number of seconds because maybe many jobs will want to access the same file...
time.sleep(rt)
targ.run_timeslides(central_period=Period, use_period_chunks=True, use_shuffling_pieces=True)

# bsub -q medium -sp 100 -R "rusage[mem=3.9G]" -o /home/projects/barakz/Collarboration-Kepler/orynai/simulated_star/outputfile_slides.out -e /home/projects/barakz/Collarboration-Kepler/orynai/simulated_star/errorfile_slides.out /home/projects/barakz/orynai/Kepler_Earths/WIS_Kepler_pipeline/runfiles/runfile_sim_star_slides.py --target_id 000000000 --period 260
# bsub -q risk -sp 100 -R "rusage[mem=3.9G]" -o /home/projects/barakz/Collarboration-Kepler/orynai/simulated_star/outputfile_slides.out -e /home/projects/barakz/Collarboration-Kepler/orynai/simulated_star/errorfile_slides.out /home/projects/barakz/orynai/Kepler_Earths/WIS_Kepler_pipeline/runfiles/runfile_sim_star_slides.py --target_id 000000000 --period 260
# bsub -q risk -R "rusage[mem=3.9G]" /home/projects/barakz/orynai/Kepler_Earths/WIS_Kepler_pipeline/runfiles/runfile_sim_star_slides.py --target_id 000000000 --period 260
# bsub -q medium -R "rusage[mem=3.9G]" /home/projects/barakz/orynai/Kepler_Earths/WIS_Kepler_pipeline/runfiles/runfile_sim_star_slides.py --target_id 000000000 --period 260
# bsub -q short -R "rusage[mem=3.9G]" /home/projects/barakz/orynai/Kepler_Earths/WIS_Kepler_pipeline/runfiles/runfile_sim_star_slides.py --target_id 000000000 --period 260
# bsub -q short -sp 100 -R "rusage[mem=3.9G]" -o /home/projects/barakz/Collaboration-Kepler/orynai/simulated_star_run_12/outputfile_slides.out -e /home/projects/barakz/Collaboration-Kepler/orynai/simulated_star_run_12/errorfile_slides.out /home/projects/barakz/orynai/Kepler_Earths/WIS_Kepler_pipeline/runfiles/runfile_sim_star_slides.py --target_id 000000000 --period 200

