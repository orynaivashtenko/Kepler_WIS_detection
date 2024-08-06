#!/usr/bin/env python3

import os
os.environ["OMP_NUM_THREADS"] = "1"  #
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"  #
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import sys
import pickle

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import tce_class
# from auxiliary_functions import fold_fluxes
# import search_pipeline as spip
from pipeline_parameters import Target_ids_prefiltered_file, Output_location, Data_location, Target_pixels_location
# from templates.template_module import get_templates_durations_from_file

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 11, 'font.serif': 'Times New Roman', 'lines.linewidth': 2})
def_col = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.ioff()

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--target_id", type=str, nargs=1, help="KOI number", default="003097222")
args = parser.parse_args()
Target_id = args.target_id[0]

tce = tce_class.KeplerTCE(target_id=Target_id, load_pickled_attributes=False,
                          output_location=Output_location, data_location=Data_location,
                          target_pixels_location=Target_pixels_location)
tce.unpickle_small_attributes(unpickle_target_attributes=True, overwrite_location_parameters=False)

tce.run_all_checks_exclusive()
tce.pickle_small_attributes(pickle_target_attributes=False)
tce.print_dict_file()

if tce.is_good_pre_filtering is True:
    with open(Target_ids_prefiltered_file, 'a') as fil:
        print(Target_id, file=fil)



# output_location = tce.output_location
# dict_to_pickle = {}
# for key in ['centroid_snrs_ra_dec', 'centroid_stds_transits', 'centroid_templs_transits']:
#     value = tce.__dict__[key]
#     dict_to_pickle[key] = value
#
# with open(os.path.join(output_location, tce.target_id + '_' + 'centroid_attributes.pickle'), 'wb') as pickle_file:
#     pickle.dump(dict_to_pickle, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

