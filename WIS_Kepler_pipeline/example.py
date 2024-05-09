#!/usr/bin/env python3

import os

print(os.path.dirname(os.path.dirname(__file__)))

import target_class
import matplotlib.pyplot as plt
import importlib
importlib.reload(target_class)

Target_id = '010187017'
# Target_id = '005302100'
# '005302100',  # with no planet
# '011853905',  # with planet
# '010417986',  # villanova morph 0.99
# '009084778',  # villanova morph 0.50
# '008700506',  # villanova morph 0.04

targ = target_class.KeplerTarget(target_id=Target_id,
                                 data_location='/Users/orynaivashtenko/Work/Kepler_Project/Intermediate_tasks/Runs_files_not_marvin/Data',
                                 output_location='/Users/orynaivashtenko/Work/Kepler_Project/Intermediate_tasks/Runs_files_not_marvin',
                                 load_pickled_attributes=False)

