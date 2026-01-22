#!/usr/bin/env python3

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
from pipeline_parameters import Data_location, Output_location, Target_pixels_location, Triggers_for_slides_injections_file
# from trigger_functions import load_triggers
import target_class
# import tce_class
# from catalog_functions import write_trigger_table
# from tce_parameters import Catalog_names_properties_dict, Trigger_table_file, Trigger_table_format
from plotting_functions import plot_all_after_slides_injections, plot_neighbors_maps, update_parameters
# from astropy.table import Table, MaskedColumn
# from locations import Triggers_plots_location
update_parameters()

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--target_id", type=str, nargs=1, help="KIC number", default="003097222")
parser.add_argument("--job_name", type=str, nargs=1, help="name of the job to report", default="")
args = parser.parse_args()

target_id = args.target_id
if isinstance(target_id, list):
    target_id = target_id[0]
job_name = args.job_name
if isinstance(job_name, list):
    job_name = job_name[0]

# ---------------------------------------------------------------------------------------------------------------------
do_analysis = True
# A temporary piece which makes it so that the analysis is not done twice because I thought it will not take so much time
# plots_folder = str(os.path.join(Triggers_plots_location, tid))
# neighbor_maps_files = [f for f in os.listdir(plots_folder) if 'neighbors_maps' in f]
# if len(neighbor_maps_files) > 0:
#     do_analysis = False
# else:
#     do_analysis = True
# ----------------------------------------------


targ = target_class.KeplerTarget(target_id=target_id,
                                 output_location=Output_location, data_location=Data_location,
                                 load_pickled_attributes=False)  # all the arguments are set to the defaults of the class
targ.unpickle_small_attributes()
targ.analyze_injections_and_background()
# targ.pickle_small_attributes()
# targ.print_dict_file()

# # Updating the TCE dict:
# tce = tce_class.WisTCE(target_id=target_id,
#                        output_location=Output_location, data_location=Data_location,
#                        target_pixels_location=Target_pixels_location,
#                        load_pickled_attributes=False, load_target_attributes=False)
# tce.unpickle_small_attributes(load_target_attributes=True, overwrite_existing_by_target=True,
#                               overwrite_location_parameters=False)
# if do_analysis:
#     tce.pickle_small_attributes()
#     tce.print_dict_file()


plot_all_after_slides_injections(target=targ, save_fig=True, no_show=True)
# plot_all_after_postprocessing
# plot_neighbors_maps(tce=tce, save_fig=True, no_show=True)


# bsub -q short -R "rusage[mem=6.9GB]" source ~/.bashrc; conda activate planets_env_amd_epyc; /home/projects/barakz/orynai/Kepler_Earths/WIS_Kepler_pipeline/runfiles/runfile_plotting_results.py
# bsub -q short -sp 100 -R "rusage[mem=6.9GB]" source ~/.bashrc; conda activate planets_env_amd_epyc; /home/projects/barakz/orynai/Kepler_Earths/WIS_Kepler_pipeline/runfiles/runfile_plotting_results.py --target_id 006934045
