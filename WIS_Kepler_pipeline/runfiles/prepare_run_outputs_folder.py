#!/usr/bin/env python3

import os
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

prefixes = [p for p in os.listdir(Data_location) if len(p) == 4 and os.path.isdir(os.path.join(Data_location, p))]
for prefixes_folder in [Output_location, Problems_folder]:
    for fold in os.listdir(prefixes_folder):
        if os.path.isdir(os.path.join(prefixes_folder, fold)):
            if len(os.listdir(os.path.join(prefixes_folder, fold))) != 0:
                os.system('rm ' + os.path.join(prefixes_folder, fold) + '/*')
        else:
            os.remove(os.path.join(prefixes_folder, fold))
    for prefix in prefixes:
        if not os.path.exists(os.path.join(prefixes_folder, prefix)):
            os.makedirs(os.path.join(prefixes_folder, prefix))

# bsub -q risk -R "rusage[mem=1GB]" /home/labs/barakz/orynai/Kepler_Earths/WIS_Kepler_pipeline/runfiles/prepare_run_outputs_folder.py
