#!/usr/bin/env python3

import pickle
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import target_class
# from auxiliary_functions import fold_fluxes
# import search_pipeline as spip
from pipeline_parameters import *
from templates.template_module import get_templates_durations_from_file

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 11, 'font.serif': 'Times New Roman', 'lines.linewidth': 2})
def_col = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.ioff()

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--target_id", type=str, nargs=1, help="KOI number", default="003097222")
args = parser.parse_args()
Target_id = args.target_id[0]

templates_durations_hours = get_templates_durations_from_file()

tlim_fold = 10  # length of the half xlimit in the folds plot in units of the template duration

targ = target_class.KeplerTarget(target_id=Target_id, load_pickled_attributes=True)
targ.analyze_injections_and_background()

# This is a temporary measure to correct the mistake in the code that made it calculate not the depth but the amplitude in ppms
###########################################
targ.get_light_curve_for_plotting_ses()

targ.summed_stat_denoms = np.ones(len(targ.summed_transit_times)) * np.inf
for ii, tt in enumerate(targ.summed_transit_times):
    for q in targ.klc.quarters:
        if (tt >= targ.klc.times[q][0]) & (tt <= targ.klc.times[q][-1]):
            targ.summed_stat_denoms[ii] = targ.klc.stat_denoms[targ.klc.indices_of_templates[targ.best_temp][0]][q]

targ.ampls = np.abs(targ.summed_raw_stats / targ.summed_stat_denoms)
not_missed_transit_mask = targ.summed_trends > 0.  # TODO: introduce this concept everywhere
targ.best_ampl = (np.nanmean((targ.ampls[not_missed_transit_mask] / targ.summed_trends[not_missed_transit_mask]))
                  * np.nanmean(targ.summed_trends[not_missed_transit_mask]))
# targ.ppms = (targ.summed_raw_stats / targ.summed_vars / targ.summed_trends
#              * np.max(np.abs(targ.klc.templates[targ.klc.indices_of_templates[targ.best_temp][0]])) * 1e6)
# I am not sure this is correct. It was written, but now I don't know why.
targ.ppms = (targ.summed_raw_stats / targ.summed_stat_denoms / targ.summed_trends
             * np.max(np.abs(targ.klc.templates[targ.klc.indices_of_templates[targ.best_temp][0]])) * 1e6)
targ.best_ppm = np.abs(np.nanmean(targ.ppms[not_missed_transit_mask]))

# same for inv
targ.summed_stat_denoms_inv = np.ones(len(targ.summed_transit_times_inv)) * np.inf
for ii, tt in enumerate(targ.summed_transit_times_inv):
    for q in targ.klc.quarters:
        if (tt >= targ.klc.times[q][0]) & (tt <= targ.klc.times[q][-1]):
            targ.summed_stat_denoms_inv[ii] = targ.klc.stat_denoms[targ.klc.indices_of_templates[targ.best_temp_inv][0]][q]

targ.ampls_inv = np.abs(targ.summed_raw_stats_inv / targ.summed_stat_denoms_inv)
not_missed_transit_mask_inv = targ.summed_trends_inv > 0.  # TODO: introduce this concept everywhere
targ.best_ampl_inv = (np.nanmean((targ.ampls_inv[not_missed_transit_mask_inv] / targ.summed_trends_inv[not_missed_transit_mask_inv]))
                      * np.nanmean(targ.summed_trends_inv[not_missed_transit_mask_inv]))
targ.ppms_inv = (targ.summed_raw_stats_inv / targ.summed_vars_inv / targ.summed_trends_inv
                 * np.max(np.abs(targ.klc.templates[targ.klc.indices_of_templates[targ.best_temp_inv][0]])) * 1e6)
targ.best_ppm_inv = np.abs(np.nanmean(targ.ppms_inv[not_missed_transit_mask_inv]))

###########################################
targ.pickle_small_attributes()
targ.print_dict_file()

fig = plt.figure(figsize=Figsize_hist)
ax = fig.gca()
targ.plot_background(ax=ax)
pickle.dump(fig, open(os.path.join(targ.output_location, str(targ.target_id) + '_background_injection_trigger_extrapolation.png'), 'wb'))
plt.savefig(os.path.join(targ.output_location, str(targ.target_id) + '_background_injection_trigger_extrapolation.png'),
            bbox_inches='tight', dpi=320)
plt.clf()

fig = plt.figure(figsize=Figsize_fluxes)
ax = fig.gca()
targ.plot_fluxes(ax=ax)
pickle.dump(fig, open(os.path.join(targ.output_location, 'conditioned_fluxes.fig.pickle'), 'wb'))
plt.savefig(os.path.join(targ.output_location, str(targ.target_id) + 'conditioned_fluxes.png'), bbox_inches='tight', dpi=320)
plt.clf()

fig = plt.figure(figsize=Figsize_fluxes)
ax = fig.gca()
targ.plot_ses_snr(ax=ax)
pickle.dump(fig, open(os.path.join(targ.output_location, 'ses_snr.fig.pickle'), 'wb'))
plt.savefig(os.path.join(targ.output_location, 'ses_snr.png'), bbox_inches='tight', dpi=320)
plt.clf()

fig = plt.figure(figsize=Figsize_fluxes)
ax = fig.gca()
targ.plot_ses_raw(ax=ax)
pickle.dump(fig, open(os.path.join(targ.output_location, 'ses_raw.fig.pickle'), 'wb'))
plt.savefig(os.path.join(targ.output_location, 'ses_raw.png'), bbox_inches='tight', dpi=320)
plt.clf()

fig = plt.figure(figsize=Figsize_folds)
ax = fig.gca()
targ.plot_folded_red_flux(ax=ax)
ax.set_xlim(-targ.best_per/2, targ.best_per/2)
pickle.dump(fig, open(os.path.join(targ.output_location, 'fold_red.fig.pickle'), 'wb'))
ax.set_xlim(-targ.templates_durations_hours[targ.best_temp] / 24 * tlim_fold,
            targ.templates_durations_hours[targ.best_temp] / 24 * tlim_fold)
plt.savefig(os.path.join(targ.output_location, 'fold_red.png'), bbox_inches='tight', dpi=320)
plt.clf()

fig = plt.figure(figsize=Figsize_folds)
ax = fig.gca()
targ.plot_folded_red_flux_inv(ax=ax)
ax.set_xlim(-targ.best_per_inv/2, targ.best_per_inv/2)
pickle.dump(fig, open(os.path.join(targ.output_location, 'fold_red_inv.fig.pickle'), 'wb'))
ax.set_xlim(-targ.templates_durations_hours[targ.best_temp_inv] / 24 * tlim_fold,
            targ.templates_durations_hours[targ.best_temp_inv] / 24 * tlim_fold)
plt.savefig(os.path.join(targ.output_location, 'fold_red_inv.png'), bbox_inches='tight', dpi=320)
plt.clf()

fig = plt.figure(figsize=Figsize_folds)
ax = fig.gca()
targ.plot_folded_whitened_flux(ax=ax)
ax.set_xlim(-targ.best_per/2, targ.best_per/2)
pickle.dump(fig, open(os.path.join(targ.output_location, 'fold_whitened.fig.pickle'), 'wb'))
ax.set_xlim(-targ.templates_durations_hours[targ.best_temp] / 24 * tlim_fold,
            targ.templates_durations_hours[targ.best_temp] / 24 * tlim_fold)
plt.savefig(os.path.join(targ.output_location, 'fold_whitened.png'), bbox_inches='tight', dpi=320)
plt.clf()

fig = plt.figure(figsize=Figsize_folds)
ax = fig.gca()
targ.plot_folded_whitened_flux_inv(ax=ax)
ax.set_xlim(-targ.best_per_inv/2, targ.best_per_inv/2)
pickle.dump(fig, open(os.path.join(targ.output_location, 'fold_whitened_inv.fig.pickle'), 'wb'))
ax.set_xlim(-targ.templates_durations_hours[targ.best_temp_inv] / 24 * tlim_fold,
            targ.templates_durations_hours[targ.best_temp_inv] / 24 * tlim_fold)
plt.savefig(os.path.join(targ.output_location, 'fold_whitened_inv.png'), bbox_inches='tight', dpi=320)
plt.clf()

# # PLOT SES RAW INV
# fig1 = plt.figure(figsize=(15, 5))
# clrs = [(0.5, 0.5, 1), (0.2, 0.7, 0.5)]
# for q in klc.quarters:
#     plt.plot(klc.times[q], -klc.statistics_raw_cor[temp_n_sh_inv][q], color=clrs[q % 2], linewidth=1)
# plt.plot(targ.summed_transit_times_inv, targ.summed_raw_stats_inv, 'o', color='k', label='summed raw inverted', markersize=4)
# for q in klc.quarters:
#     vm = klc.veto_masks[temp_n_sh_inv][q].astype(bool)
#     plt.plot(klc.times[q][~vm], -klc.statistics_snr[temp_n_sh_inv][q][~vm], 'o', color='r', markersize=2)
# plt.plot(klc.times[q][~vm], -klc.statistics_snr[temp_n_sh_inv][q][~vm], 'o', color='r', markersize=2, label='vetted statistics')
# plt.xlabel('time, days')
# plt.ylabel('statistics raw inverted')
# plt.title('KOI' + targ.target_id + ' SES raw inverted')
# plt.legend(loc='upper right')
# pickle.dump(fig1, open(os.path.join(targ.output_location, 'ses_raw_inv.fig.pickle'), 'wb'))
# plt.savefig(os.path.join(targ.output_location, 'ses_raw_inv.png'), bbox_inches='tight', dpi=320)
# plt.clf()
# plt.close()
#
# # PLOT FOLD RED FLUX INV
# binwid_inv = min(templates_durations_hours[targ.best_temp_inv] / 24 / 5, 0.1)
# bins_time, bins_flux, bins_counts, folds_times, folds_fluxes, time_template, template_cropped = fold_fluxes(
#     times=klc.times, fluxes=[-fl for fl in klc.fluxes], period=targ.best_per_inv, first_time=targ.summed_transit_times_inv[0],
#     masks=klc.big_hole_masks,
#     binwid=binwid_inv, time_fold=templates_durations_hours[targ.best_temp_inv] / 24 * 10, dt=klc.dt_days,
#     means=[np.nanmean(klc.orig_fluxes[q]) for q in klc.quarters],
#     template=np.array(klc.templates[klc.indices_of_templates[targ.best_temp_inv][0]]))
# plt.figure(figsize=(8, 6))
# for i in range(len(folds_times)):
#     plt.plot(folds_times[i], folds_fluxes[i], 'o', markersize=1, color=(0.5, 0.5, 0.5))
# plt.plot(bins_time, bins_flux, '-o', markersize=3, color='b', linewidth=1, label='binning, bin ' + str(binwid_inv) + ' days')
# plt.plot(time_template, 1 - template_cropped * targ.best_ppm / 1e6, color='r', linewidth=1, label=r'template $\cdot$ amplitude')
# plt.legend(loc='upper right')
# plt.title('KOI' + targ.target_id + ' inverted fold, ' + str(len(targ.summed_raw_stats_inv)) + ' transits')
# plt.xlabel('time from the mid transit, days')
# plt.ylabel('flux normalized by mean')
# plt.savefig(os.path.join(targ.output_location, 'fold_red_inv.png'), dpi=320, bbox_inches='tight')
# plt.clf()
# plt.close()
#
# # PLOT FOLD WHITENED FLUX
# bins_time, bins_flux, bins_counts, folds_times, folds_fluxes, time_template, template_cropped = fold_fluxes(
#     times=klc.times, fluxes=klc.whitened_fluxes_td, period=targ.best_per, first_time=targ.summed_transit_times[0], masks=klc.big_hole_masks,
#     binwid=binwid, time_fold=templates_durations_hours[targ.best_temp] / 24 * 10, dt=klc.dt_days, means=None,
#     template=np.array(klc.templates[klc.indices_of_templates[targ.best_temp][0]]))
# N = len(time_template)
# temp_best_1 = np.array(
#     np.mean([klc.whitened_templates_td[klc.indices_of_templates[targ.best_temp][0]][q][:N // 2] for q in klc.quarters], axis=0))
# temp_best_2 = np.array(
#     np.mean([klc.whitened_templates_td[klc.indices_of_templates[targ.best_temp][0]][q][-N // 2:] for q in klc.quarters], axis=0))
# temp_best = np.concatenate((temp_best_2, temp_best_1))
# plt.figure(figsize=(8, 6))
# for i in range(len(folds_times)):
#     plt.plot(folds_times[i], folds_fluxes[i], 'o', markersize=1, color=(0.5, 0.5, 0.5))
# plt.plot(bins_time, bins_flux, '-o', markersize=3, color='b', linewidth=1, label='binning, bin ' + str(binwid) + ' days')
# plt.plot(time_template, -temp_best * targ.best_ampl, color='r', linewidth=1, label='average whitened template')
# plt.xlabel('time from the mid transit, days')
# plt.ylabel('whitened flux')
# plt.legend(loc='upper right')
# plt.title('KOI' + targ.target_id + ' whitened fold, ' + str(len(targ.summed_raw_stats)) + ' transits')
# plt.savefig(os.path.join(targ.output_location, 'fold_whitened.png'), dpi=320, bbox_inches='tight')
# plt.clf()
# plt.close()
#
# # PLOT FOLD WHITENED FLUX INV
# bins_time, bins_flux, bins_counts, folds_times, folds_fluxes, time_template, template_cropped = fold_fluxes(
#     times=klc.times, fluxes=[-fl for fl in klc.whitened_fluxes_td], period=targ.best_per_inv, first_time=targ.summed_transit_times_inv[0],
#     masks=klc.big_hole_masks,
#     binwid=binwid_inv, time_fold=templates_durations_hours[targ.best_temp_inv] / 24 * 10, dt=klc.dt_days, means=None,
#     template=np.array(klc.templates[klc.indices_of_templates[targ.best_temp_inv][0]]))
# N = len(time_template)
# temp_best_1 = np.array(
#     np.mean([klc.whitened_templates_td[klc.indices_of_templates[targ.best_temp_inv][0]][q][:N // 2] for q in klc.quarters], axis=0))
# temp_best_2 = np.array(
#     np.mean([klc.whitened_templates_td[klc.indices_of_templates[targ.best_temp_inv][0]][q][-N // 2:] for q in klc.quarters], axis=0))
# temp_best = np.concatenate((temp_best_2, temp_best_1))
# plt.figure(figsize=(8, 6))
# for i in range(len(folds_times)):
#     plt.plot(folds_times[i], folds_fluxes[i], 'o', markersize=1, color=(0.5, 0.5, 0.5))
# plt.plot(bins_time, bins_flux, '-o', markersize=3, color='b', linewidth=1, label='binning, bin ' + str(binwid_inv) + ' days')
# plt.plot(time_template, -temp_best * targ.best_ampl, color='r', linewidth=1, label='average whitened template')
# plt.xlabel('time from the mid transit, days')
# plt.ylabel('whitened flux')
# plt.legend(loc='upper right')
# plt.title('KOI' + targ.target_id + ' inverted whitened fold, ' + str(len(targ.summed_raw_stats_inv)) + ' transits')
# plt.savefig(os.path.join(targ.output_location, 'fold_whitened_inv.png'), dpi=320, bbox_inches='tight')
# plt.clf()
# plt.close()
#
# # PLOT BACKGROUND AND INJECTION-RECOVERY
# Default_min_bin_mes2 = 10
# Default_max_bin_mes2 = 140
# # Default_n_bin_edges_mes2 = 191
# Default_binwid_mes2 = 1.
#
# targ.analyze_injections_and_background()
# # plot_background(self, ax=None, figsize=(10, 5), fixed_limits=True, limits=None, plot_extrapolation=True)
# self.pickle_small_attributes()
# self.print_dict_file()
#
# Bins = targ.bins_mes2_histogram
#
# # UNIFORMLY FORMATTED PLOT
# plt.figure(figsize=(10, 5))
#
# plt.stairs(targ.distr_mes2_inj_all, Bins, label='all injections', linewidth=2)  # after density correction factor, get per one star
# plt.stairs(targ.distr_mes2_inj_success, Bins, label='success injections', linewidth=2)  # after density correction factor, get per one star
# # Divide by the number of the timeslides, get effectively per one star:
# plt.stairs(targ.distr_mes2_h0_direct, Bins, label='background direct H0', linewidth=2)
# plt.stairs(targ.distr_mes2_h0_inv, Bins, label='background inv H0', linewidth=2)
# plt.stairs(targ.distr_mes2_h0, Bins, label='background total H0', linewidth=2)
#
# plt.axvline(targ.best_stat ** 2, color='k', label='trigger, period=' + str(np.round(targ.best_per, 1)))
# plt.title('KOI' + targ.target_id + ', periods ['
#           + str(np.round(10 ** (targ.log10_best_period_edges[0]), 1)) + '-'
#           + str(np.round(10 ** (targ.log10_best_period_edges[1]), 1))
#           + '] days, duration=' + str(templates_durations_hours[targ.best_temp]) + ' hours', fontsize=13)
# plt.xlabel('recovered MES [$SNR^2$]')
# plt.ylabel('counts per effective one star \n per these periods and template')
# plt.yscale('log')
# plt.xlim(Default_min_bin_mes2, Default_max_bin_mes2)
# plt.legend(fontsize=11)
# plt.savefig(os.path.join(targ.output_location, str(targ.target_id) + '_background_injection_trigger_uniform.png'),
#             bbox_inches='tight', dpi=320)
#
# # FULL-SCALE LIMITS
# plt.xlim(0.5 * Bins[0], 1.1 * Bins[-1])
# plt.savefig(os.path.join(targ.output_location, str(targ.target_id) + '_background_injection_trigger_full.png'),
#             bbox_inches='tight', dpi=320)
#
# # MAKE EXTRAPOLATION
# coefs = targ.extrapolation_coefs_mes2_h0_hist
# bin_extra_max = (np.log(np.nanmin(targ.distr_mes2_h0)) - coefs[0]) / coefs[1]  # determine intersection with horizontal axis in log scale
# bins_extra = np.linspace(Bins[np.nanargmax(targ.distr_mes2_h0) + 1], np.nanmax([(targ.best_stat ** 2) + 1, bin_extra_max + 1]), 10)
# plt.plot(bins_extra, np.exp(bins_extra * coefs[1] + coefs[0]), label='extrapolation, FAP='+f"{targ.false_alarm_prob_trigger:.2e}")
# plt.xlim(0.5 * Bins[0], 1.1 * Bins[-1])
# plt.savefig(os.path.join(targ.output_location, str(targ.target_id) + '_background_injection_trigger_extrapolation.png'),
#             bbox_inches='tight', dpi=320)
# plt.clf()
# plt.close()
