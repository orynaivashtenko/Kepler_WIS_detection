#!/usr/bin/env python3

import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from locations import (Run_outputs_folder, Output_location, Triggers_plots_location,
                       Trigger_lists_folder, Problems_folder, Data_location)


if not os.path.exists(Run_outputs_folder) or not os.path.isdir(Run_outputs_folder):
    os.makedirs(Run_outputs_folder)
if not os.path.exists(Output_location) or not os.path.isdir(Output_location):
    os.makedirs(Output_location)
if not os.path.exists(Triggers_plots_location) or not os.path.isdir(Triggers_plots_location):
    os.makedirs(Triggers_plots_location)
if not os.path.exists(Trigger_lists_folder) or not os.path.isdir(Trigger_lists_folder):
    os.makedirs(Trigger_lists_folder)
if not os.path.exists(Problems_folder) or not os.path.isdir(Problems_folder):
    os.makedirs(Problems_folder)

for folder in [Trigger_lists_folder, Triggers_plots_location]:
    if not os.path.exists(folder):
        os.makedirs(folder)
    if len(os.listdir(folder)) != 0:
        os.system('rm -r ' + folder + '/*')

for ff in os.listdir(Run_outputs_folder):
    if not os.path.isdir(os.path.join(Run_outputs_folder, ff)):
        os.system('rm ' + os.path.join(Run_outputs_folder, ff))


prefixes = [p for p in os.listdir(Data_location) if len(p) == 4 and os.path.isdir(os.path.join(Data_location, p))]
for big_folder in [Output_location, Problems_folder]:
    for prefix_folder in os.listdir(big_folder):
        if os.path.isdir(os.path.join(big_folder, prefix_folder)):
            for tid_folder in os.listdir(os.path.join(big_folder, prefix_folder)):
                if os.path.isdir(os.path.join(big_folder, prefix_folder, tid_folder)):
                    os.system('rm -rf ' + os.path.join(big_folder, prefix_folder, tid_folder) + '/*')
                else:
                    os.remove(os.path.join(big_folder, prefix_folder, tid_folder))
        else:
            os.remove(os.path.join(big_folder, prefix_folder))

    for prefix in prefixes:
        if not os.path.exists(os.path.join(big_folder, prefix)):
            os.makedirs(os.path.join(big_folder, prefix))

# bsub -q risk -R "rusage[mem=1GB]" /home/projects/barakz/orynai/Kepler_Earths/WIS_Kepler_pipeline/runfiles/prepare_run_outputs_folder.py
