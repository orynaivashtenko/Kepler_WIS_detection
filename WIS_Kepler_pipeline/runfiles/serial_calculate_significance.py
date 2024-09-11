#!/usr/bin/env python3

# #########################################################################################################################
# CALCULATING SIGNIFICANCES: P_PLANET, FALSE ALARM ETC
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
from pipeline_parameters import Data_location, Output_location, Target_pixels_location, Triggers_for_slides_injections_file
from trigger_functions import load_triggers
import target_class
import tce_class
from catalog_functions import write_trigger_table
from tce_parameters import Catalog_names_properties_dict, Trigger_table_file, Trigger_table_format
from astropy.table import Table, MaskedColumn

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--trigger_list_file", type=str, nargs=1, help="path to file where to take triggers",
                    default=Triggers_for_slides_injections_file)
args = parser.parse_args()
trigger_list_file = args.trigger_list_file
if isinstance(trigger_list_file, list):
    trigger_list_file = trigger_list_file[0]

target_ids, integral_scores, stats, pers, ftts, gauss_stats, templates = load_triggers(file=trigger_list_file)

# Sort everyone
indices_order = np.argsort(target_ids)
target_ids = target_ids[indices_order]
integral_scores = integral_scores[indices_order]
stats = stats[indices_order]
pers = pers[indices_order]
ftts = ftts[indices_order]
gauss_stats = gauss_stats[indices_order]
templates = templates[indices_order]

# ---------------------------------------------------------------------------------------------------------------------
# STARTING THE CATALOG RIGHT HERE
# ---------------------------------------------------------------------------------------------------------------------
# 1) Initializing an empty table
trigger_table = Table(masked=True)
for i, name in enumerate(Catalog_names_properties_dict.keys()):
    col = MaskedColumn(data=np.empty(0, dtype=Catalog_names_properties_dict[name][1]),
                       name=name,
                       mask=np.empty(0, dtype=bool),
                       # fill_value=Catalog_names_properties_dict[name][x],
                       fill_value=None,
                       dtype=Catalog_names_properties_dict[name][1],
                       description=Catalog_names_properties_dict[name][4],
                       unit=Catalog_names_properties_dict[name][2],
                       format=None, meta=None, copy=False, copy_indices=True)
    trigger_table.add_column(col)
# ---------------------------------------------------------------------------------------------------------------------
for tid in target_ids:
    targ = target_class.KeplerTarget(target_id=tid,
                                     output_location=Output_location, data_location=Data_location,
                                     load_pickled_attributes=False)  # all the arguments are set to the defaults of the class
    targ.unpickle_small_attributes()
    targ.analyze_injections_and_background()
    targ.pickle_small_attributes()
    targ.print_dict_file()

    # Updating the TCE dict:
    tce = tce_class.WisTCE(target_id=tid,
                           output_location=Output_location, data_location=Data_location,
                           target_pixels_location=Target_pixels_location,
                           load_pickled_attributes=False, load_target_attributes=False)
    tce.unpickle_small_attributes(load_target_attributes=True, overwrite_existing_by_target=True,
                                  overwrite_location_parameters=False)
    tce.pickle_small_attributes()
    tce.print_dict_file()

    # Entering the target ino the trigger table:
    trigger_table.add_row()
    for key in Catalog_names_properties_dict.keys():
        if key in tce.__dict__.keys():
            if tce.__dict__[key] is not None:
                trigger_table[key][-1] = tce.__dict__[key]
                trigger_table[key].mask[-1] = False
            else:
                # trigger_table[key][-1] = np.ma.masked
                trigger_table[key].mask[-1] = True

write_trigger_table(trigger_table, file_path=Trigger_table_file, overwrite=True,
                    trigger_table_format=Trigger_table_format, fill_values=None)

# bsub -q short -R "rusage[mem=2GB]" /home/labs/barakz/orynai/Kepler_Earths/WIS_Kepler_pipeline/runfiles/serial_calculate_significance.py
