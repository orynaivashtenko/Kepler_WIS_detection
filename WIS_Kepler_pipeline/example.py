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
                                 load_pickled_attributes=False,
                                 data_location='/Users/orynaivashtenko/Work/Kepler_Project/Intermediate_tasks/Runs_files_not_marvin/Data',
                                 output_location='/Users/orynaivashtenko/Work/Kepler_Project/Intermediate_tasks/Runs_files_not_marvin',)

# # targ1 = target_class.KeplerTarget(# target_id='005302100',  # with no planet
# #                                   # target_id='011853905',  # with planet
# #                                   target_id='010417986',  # villanova morph 0.99
# #                                   # target_id='009084778',  # villanova morph 0.50
# #                                   # target_id='8700506',  # villanova morph 0.04
# #                                   data_location='/Users/orynaivashtenko/Work/Kepler_Project/Intermediate_tasks/Runs_files_not_marvin/Data',
# #                                   output_location='/Users/orynaivashtenko/Work/Kepler_Project/Intermediate_tasks/Runs_files_not_marvin',
# #                                   # gaia_kepler_file=Gaia_kepler_file,
# #                                   # targets_info_file=Default_targets_info_file,
# #                                   # mes_candidate=Mes_candidate_search,
# #                                   p_min_search=40,
# #                                   p_max_search=41,
# #                                   p_chunks_search=None)
# #
# # # targ1.get_koi_and_ttv_fields()
# # targ1.get_villanova_fields()
# # targ1.initialize_kepler_light_curve()
# # targ1.klc.subtract_means()
#
# targ2 = target_class.KeplerTarget(# target_id='005302100',  # with no planet
#                                   # target_id='011853905',  # with planet
#                                   # target_id='010417986',  # villanova morph 0.99
#                                   target_id='8700506',  # villanova morph 0.04
#                                   # target_id='009084778',  # villanova morph 0.50
#                                   data_location='/Users/orynaivashtenko/Work/Kepler_Project/Intermediate_tasks/Runs_files_not_marvin/Data',
#                                   output_location='/Users/orynaivashtenko/Work/Kepler_Project/Intermediate_tasks/Runs_files_not_marvin',
#                                   # gaia_kepler_file=Gaia_kepler_file,
#                                   # targets_info_file=Default_targets_info_file,
#                                   # mes_candidate=Mes_candidate_search,
#                                   p_min_search=40,
#                                   p_max_search=41,
#                                   p_chunks_search=None)
#
# targ2.get_villanova_fields()
# targ2.initialize_kepler_light_curve()
# targ2.klc.subtract_means()
#
# # print(targ1, targ2)
#
# # plt.figure(figsize=(10, 5))
# # for q in targ1.klc.quarters:
# #     plt.plot(targ1.klc.times[q][targ1.klc.masks[q]], targ1.klc.fluxes[q][targ1.klc.masks[q]])
# # plt.show()
#
# plt.figure(figsize=(10, 5))
# for q in targ2.klc.quarters:
#     plt.plot(targ2.klc.times[q][targ2.klc.masks[q]], targ2.klc.fluxes[q][targ2.klc.masks[q]])
# plt.show()
#
# # targ1.mask_koi_planet()
# # # # targ.klc.load_conditioned_data()
# # # # targ.klc.initialize_templates()
# # # # targ.klc.detect_single_transits()
# # # # targ.run_periodicity_search()
# # # targ.unpickle_small_attributes()
# #
# # for q in targ.klc.quarters:
# #     plt.plot(targ.klc.times[q][targ.klc.masks[q]], targ.klc.fluxes[q][targ.klc.masks[q]], 'o-')
# #
# # for q in targ.klc.quarters:
# #     plt.plot(targ.klc.times[q][~targ.klc.masks[q]], targ.klc.fluxes[q][~targ.klc.masks[q]], 'o-')
#
# # plt.show()
