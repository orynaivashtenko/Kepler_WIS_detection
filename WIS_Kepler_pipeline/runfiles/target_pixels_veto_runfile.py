#!/usr/bin/env python3

import os
import sys
import pickle

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import tce_class
# from auxiliary_functions import fold_fluxes
# import search_pipeline as spip
# from pipeline_parameters import *
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

tce = tce_class.KeplerTCE(target_id=Target_id, load_pickled_attributes=True)

# tce.test_target_pixels_amplitudes()
tce.find_centroid_transits_snr()
tce.estimate_rp()
# tce.test_single_pixels()

tce.pickle_small_attributes(pickle_target_attributes=False)
tce.print_dict_file()

output_location = tce.output_location
dict_to_pickle = {}
for key in ['centroid_snrs_transits', 'centroid_stds_transits', 'centroid_templs_transits']:
    value = tce.__dict__[key]
    dict_to_pickle[key] = value

with open(os.path.join(output_location, tce.target_id + '_' + 'centroid_attributes.pickle'), 'wb') as pickle_file:
    pickle.dump(dict_to_pickle, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

