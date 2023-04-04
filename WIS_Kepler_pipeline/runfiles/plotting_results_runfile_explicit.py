#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pickle
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import target_class
from auxiliary_functions import fold_fluxes
import search_pipeline as spip
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

targ = target_class.KeplerTarget(target_id=Target_id,
                                 load_pickled_attributes=False)  # all the arguments are set to the defaults of the class
targ.unpickle_small_attributes()
klc = spip.KeplerLightCurve(target_id=targ.target_id, get_data=True, data_location=targ.data_location)
if targ.target_id + '_' + Conditioned_data_filename_postfix + '.npy' not in os.listdir(targ.output_location):
    klc.condition_data()  # IN OUTLIERS MASK, THE POSITIVE AND THE NEGATIVE NSIG ARE EQUAL BECAUSE WE NEED TO INVERT FAIRLY.
    klc.condition_templates()
else:
    klc.load_conditioned_data(file=os.path.join(targ.output_location,
                                                targ.target_id + '_' + Conditioned_data_filename_postfix + '.npy'))
klc.initialize_templates()  # template_ind=targ.best_temp # we need two templates, therefore we have to initialize all
# TODO: write the template initialization in such a way that we could pass many template indices
klc.detect_single_transits(apply_score_correction=True)

# PLOT FLUXES

temp_n = targ.best_temp  # because we initialized templates with only one template
temp_n_sh = klc.indices_of_templates[temp_n][0]
temp_n_inv = targ.best_temp_inv
temp_n_sh_inv = klc.indices_of_templates[temp_n_inv][0]

tms = []
flxs = []
for tm in targ.summed_transit_times:
    for q in klc.quarters:
        if (tm >= klc.times[q][0]) and (tm <= klc.times[q][-1]):
            ind = np.argmin(np.abs(klc.times[q] - tm))
            tms.append(klc.times[q][ind])
            flxs.append(klc.fluxes[q][ind])

fig1 = plt.figure(figsize=(15, 5))
clrs = [(0.5, 0.5, 1), (0.2, 0.7, 0.5)]
for q in klc.quarters:
    plt.plot(klc.times[q], klc.fluxes[q], color=clrs[q % 2], linewidth=1)
plt.plot(tms, flxs, 'o', color='k', label='summed raw', markersize=4)
for q in klc.quarters:
    vm = klc.veto_masks[temp_n_sh][q].astype(bool)
    plt.plot(klc.times[q][~vm], klc.fluxes[q][~vm], 'o', color='r', markersize=2)
q = klc.quarters[-1]
vm = klc.veto_masks[temp_n_sh][q].astype(bool)
plt.plot(klc.times[q][~vm], klc.fluxes[q][~vm], 'o', color='r', markersize=2, label='vetted statistics')  # just for label
plt.xlabel('time, days')
plt.ylabel('fluxes, mean subtracted')
plt.title('KOI' + targ.target_id + ' conditioned fluxes')
plt.legend(loc='upper right')
pickle.dump(fig1, open(os.path.join(targ.output_location, 'conditioned_fluxes.fig.pickle'), 'wb'))
plt.savefig(os.path.join(targ.output_location, 'conditioned_fluxes.png'), bbox_inches='tight', dpi=320)
plt.clf()
plt.close()

# PLOT SES RAW
fig1 = plt.figure(figsize=(15, 5))
clrs = [(0.5, 0.5, 1), (0.2, 0.7, 0.5)]
for q in klc.quarters:
    plt.plot(klc.times[q], klc.statistics_raw_cor[temp_n_sh][q], color=clrs[q % 2], linewidth=1)
plt.plot(targ.summed_transit_times, targ.summed_raw_stats, 'o', color='k', label='summed raw', markersize=4)
for q in klc.quarters:
    vm = klc.veto_masks[temp_n_sh][q].astype(bool)
    plt.plot(klc.times[q][~vm], klc.statistics_raw_cor[temp_n_sh][q][~vm], 'o', color='r', markersize=2)
plt.plot(klc.times[q][~vm], klc.statistics_raw_cor[temp_n_sh][q][~vm], 'o', color='r', markersize=2, label='vetted statistics')
plt.xlabel('time, days')
plt.ylabel('statistics raw')
plt.title('KOI' + targ.target_id + ' SES raw')
plt.legend(loc='upper right')
pickle.dump(fig1, open(os.path.join(targ.output_location, 'ses_raw.fig.pickle'), 'wb'))
plt.savefig(os.path.join(targ.output_location, 'ses_raw.png'), bbox_inches='tight', dpi=320)
plt.clf()
plt.close()

# PLOT SES RAW INV
fig1 = plt.figure(figsize=(15, 5))
clrs = [(0.5, 0.5, 1), (0.2, 0.7, 0.5)]
for q in klc.quarters:
    plt.plot(klc.times[q], -klc.statistics_raw_cor[temp_n_sh_inv][q], color=clrs[q % 2], linewidth=1)
plt.plot(targ.summed_transit_times_inv, targ.summed_raw_stats_inv, 'o', color='k', label='summed raw inverted', markersize=4)
for q in klc.quarters:
    vm = klc.veto_masks[temp_n_sh_inv][q].astype(bool)
    plt.plot(klc.times[q][~vm], -klc.statistics_snr[temp_n_sh_inv][q][~vm], 'o', color='r', markersize=2)
plt.plot(klc.times[q][~vm], -klc.statistics_snr[temp_n_sh_inv][q][~vm], 'o', color='r', markersize=2, label='vetted statistics')
plt.xlabel('time, days')
plt.ylabel('statistics raw inverted')
plt.title('KOI' + targ.target_id + ' SES raw inverted')
plt.legend(loc='upper right')
pickle.dump(fig1, open(os.path.join(targ.output_location, 'ses_raw_inv.fig.pickle'), 'wb'))
plt.savefig(os.path.join(targ.output_location, 'ses_raw_inv.png'), bbox_inches='tight', dpi=320)
plt.clf()
plt.close()

# PLOT SES SNR
fig1 = plt.figure(figsize=(15, 5))
clrs = [(0.5, 0.5, 1), (0.2, 0.7, 0.5)]
for q in klc.quarters:
    plt.plot(klc.times[q], klc.statistics_snr[temp_n_sh][q], color=clrs[q % 2], linewidth=1)
plt.plot(targ.summed_transit_times, targ.summed_raw_stats / targ.summed_vars ** 0.5, 'o', color='k', label='summed raw / std', markersize=4)
for q in klc.quarters:
    vm = klc.veto_masks[temp_n_sh][q].astype(bool)
    plt.plot(klc.times[q][~vm], klc.statistics_snr[temp_n_sh][q][~vm], 'o', color='r', markersize=2)
plt.plot(klc.times[q][~vm], klc.statistics_snr[temp_n_sh][q][~vm], 'o', color='r', markersize=2, label='vetted statistics')
plt.xlabel('time, days')
plt.ylabel('statistics snr')
plt.title('KOI' + targ.target_id + ' SES SNR')
plt.legend(loc='upper right')
pickle.dump(fig1, open(os.path.join(targ.output_location, 'ses_snr.fig.pickle'), 'wb'))
plt.savefig(os.path.join(targ.output_location, 'ses_snr.png'), bbox_inches='tight', dpi=320)
plt.clf()
plt.close()

# PLOT FOLD RED FLUX
binwid = min(templates_durations_hours[targ.best_temp] / 24 / 5, 0.1)
bins_time, bins_flux, bins_counts, folds_times, folds_fluxes, time_template, template_cropped = fold_fluxes(
    times=klc.times, fluxes=klc.fluxes, period=targ.best_per, first_time=targ.summed_transit_times[0], masks=klc.big_hole_masks,
    binwid=binwid, time_fold=templates_durations_hours[targ.best_temp] / 24 * 10, dt=klc.dt_days,
    means=[np.nanmean(klc.orig_fluxes[q]) for q in klc.quarters],
    template=np.array(klc.templates[klc.indices_of_templates[targ.best_temp][0]]))
plt.figure(figsize=(8, 6))
for i in range(len(folds_times)):
    plt.plot(folds_times[i], folds_fluxes[i], 'o', markersize=1, color=(0.5, 0.5, 0.5))
plt.plot(bins_time, bins_flux, '-o', markersize=3, color='b', linewidth=1, label='binning, bin ' + str(binwid) + ' days')
plt.plot(time_template, 1 - template_cropped * targ.best_ppm / 1e6, color='r', linewidth=1, label=r'template $\cdot$ amplitude')
plt.legend(loc='upper right')
plt.title('KOI' + targ.target_id + ' fold, ' + str(len(targ.summed_raw_stats)) + ' transits')
plt.xlabel('time from the mid transit, days')
plt.ylabel('flux normalized by mean')
plt.savefig(os.path.join(targ.output_location, 'fold_red.png'), dpi=320, bbox_inches='tight')
plt.clf()
plt.close()

# PLOT FOLD RED FLUX INV
binwid_inv = min(templates_durations_hours[targ.best_temp_inv] / 24 / 5, 0.1)
bins_time, bins_flux, bins_counts, folds_times, folds_fluxes, time_template, template_cropped = fold_fluxes(
    times=klc.times, fluxes=[-fl for fl in klc.fluxes], period=targ.best_per_inv, first_time=targ.summed_transit_times_inv[0],
    masks=klc.big_hole_masks,
    binwid=binwid_inv, time_fold=templates_durations_hours[targ.best_temp_inv] / 24 * 10, dt=klc.dt_days,
    means=[np.nanmean(klc.orig_fluxes[q]) for q in klc.quarters],
    template=np.array(klc.templates[klc.indices_of_templates[targ.best_temp_inv][0]]))
plt.figure(figsize=(8, 6))
for i in range(len(folds_times)):
    plt.plot(folds_times[i], folds_fluxes[i], 'o', markersize=1, color=(0.5, 0.5, 0.5))
plt.plot(bins_time, bins_flux, '-o', markersize=3, color='b', linewidth=1, label='binning, bin ' + str(binwid_inv) + ' days')
plt.plot(time_template, 1 - template_cropped * targ.best_ppm / 1e6, color='r', linewidth=1, label=r'template $\cdot$ amplitude')
plt.legend(loc='upper right')
plt.title('KOI' + targ.target_id + ' inverted fold, ' + str(len(targ.summed_raw_stats_inv)) + ' transits')
plt.xlabel('time from the mid transit, days')
plt.ylabel('flux normalized by mean')
plt.savefig(os.path.join(targ.output_location, 'fold_red_inv.png'), dpi=320, bbox_inches='tight')
plt.clf()
plt.close()

# PLOT FOLD WHITENED FLUX
bins_time, bins_flux, bins_counts, folds_times, folds_fluxes, time_template, template_cropped = fold_fluxes(
    times=klc.times, fluxes=klc.whitened_fluxes_td, period=targ.best_per, first_time=targ.summed_transit_times[0], masks=klc.big_hole_masks,
    binwid=binwid, time_fold=templates_durations_hours[targ.best_temp] / 24 * 10, dt=klc.dt_days, means=None,
    template=np.array(klc.templates[klc.indices_of_templates[targ.best_temp][0]]))
N = len(time_template)
temp_best_1 = np.array(
    np.mean([klc.whitened_templates_td[klc.indices_of_templates[targ.best_temp][0]][q][:N // 2] for q in klc.quarters], axis=0))
temp_best_2 = np.array(
    np.mean([klc.whitened_templates_td[klc.indices_of_templates[targ.best_temp][0]][q][-N // 2:] for q in klc.quarters], axis=0))
temp_best = np.concatenate((temp_best_2, temp_best_1))
plt.figure(figsize=(8, 6))
for i in range(len(folds_times)):
    plt.plot(folds_times[i], folds_fluxes[i], 'o', markersize=1, color=(0.5, 0.5, 0.5))
plt.plot(bins_time, bins_flux, '-o', markersize=3, color='b', linewidth=1, label='binning, bin ' + str(binwid) + ' days')
plt.plot(time_template, -temp_best * targ.best_ampl, color='r', linewidth=1, label='average whitened template')
plt.xlabel('time from the mid transit, days')
plt.ylabel('whitened flux')
plt.legend(loc='upper right')
plt.title('KOI' + targ.target_id + ' whitened fold, ' + str(len(targ.summed_raw_stats)) + ' transits')
plt.savefig(os.path.join(targ.output_location, 'fold_whitened.png'), dpi=320, bbox_inches='tight')
plt.clf()
plt.close()

# PLOT FOLD WHITENED FLUX INV
bins_time, bins_flux, bins_counts, folds_times, folds_fluxes, time_template, template_cropped = fold_fluxes(
    times=klc.times, fluxes=[-fl for fl in klc.whitened_fluxes_td], period=targ.best_per_inv, first_time=targ.summed_transit_times_inv[0],
    masks=klc.big_hole_masks,
    binwid=binwid_inv, time_fold=templates_durations_hours[targ.best_temp_inv] / 24 * 10, dt=klc.dt_days, means=None,
    template=np.array(klc.templates[klc.indices_of_templates[targ.best_temp_inv][0]]))
N = len(time_template)
temp_best_1 = np.array(
    np.mean([klc.whitened_templates_td[klc.indices_of_templates[targ.best_temp_inv][0]][q][:N // 2] for q in klc.quarters], axis=0))
temp_best_2 = np.array(
    np.mean([klc.whitened_templates_td[klc.indices_of_templates[targ.best_temp_inv][0]][q][-N // 2:] for q in klc.quarters], axis=0))
temp_best = np.concatenate((temp_best_2, temp_best_1))
plt.figure(figsize=(8, 6))
for i in range(len(folds_times)):
    plt.plot(folds_times[i], folds_fluxes[i], 'o', markersize=1, color=(0.5, 0.5, 0.5))
plt.plot(bins_time, bins_flux, '-o', markersize=3, color='b', linewidth=1, label='binning, bin ' + str(binwid_inv) + ' days')
plt.plot(time_template, -temp_best * targ.best_ampl, color='r', linewidth=1, label='average whitened template')
plt.xlabel('time from the mid transit, days')
plt.ylabel('whitened flux')
plt.legend(loc='upper right')
plt.title('KOI' + targ.target_id + ' inverted whitened fold, ' + str(len(targ.summed_raw_stats_inv)) + ' transits')
plt.savefig(os.path.join(targ.output_location, 'fold_whitened_inv.png'), dpi=320, bbox_inches='tight')
plt.clf()
plt.close()

# PLOT BACKGROUND AND INJECTION-RECOVERY
Default_min_bin_mes2 = 10
Default_max_bin_mes2 = 140
# Default_n_bin_edges_mes2 = 191
Default_binwid_mes2 = 1.

targ.analyze_injections_and_background()
Bins = targ.bins_mes2_histogram

# UNIFORMLY FORMATTED PLOT
plt.figure(figsize=(10, 5))

plt.stairs(targ.distr_mes2_inj_all, Bins, label='all injections', linewidth=2)  # after density correction factor, get per one star
plt.stairs(targ.distr_mes2_inj_success, Bins, label='success injections', linewidth=2)  # after density correction factor, get per one star
# Divide by the number of the timeslides, get effectively per one star:
plt.stairs(targ.distr_mes2_h0_direct, Bins, label='background direct H0', linewidth=2)
plt.stairs(targ.distr_mes2_h0_inv, Bins, label='background inv H0', linewidth=2)
plt.stairs(targ.distr_mes2_h0, Bins, label='background total H0', linewidth=2)

plt.axvline(targ.best_stat ** 2, color='k', label='trigger, period=' + str(np.round(targ.best_per, 1)))
plt.title('KOI' + targ.target_id + ', periods ['
          + str(np.round(10 ** (targ.log10_best_period_edges[0]), 1)) + '-'
          + str(np.round(10 ** (targ.log10_best_period_edges[1]), 1))
          + '] days, duration=' + str(templates_durations_hours[targ.best_temp]) + ' hours', fontsize=13)
plt.xlabel('recovered MES [$SNR^2$]')
plt.ylabel('counts per effective one star \n per these periods and template')
plt.yscale('log')
plt.xlim(Default_min_bin_mes2, Default_max_bin_mes2)
plt.legend(fontsize=11)
plt.savefig(os.path.join(targ.output_location, str(targ.target_id) + '_background_injection_trigger_uniform.png'),
            bbox_inches='tight', dpi=320)

# FULL-SCALE LIMITS
plt.xlim(0.5 * Bins[0], 1.1 * Bins[-1])
plt.savefig(os.path.join(targ.output_location, str(targ.target_id) + '_background_injection_trigger_full.png'),
            bbox_inches='tight', dpi=320)

# MAKE EXTRAPOLATION
coefs = targ.extrapolation_coefs_mes2_h0_hist
bin_extra_max = (np.log(np.nanmin(targ.distr_mes2_h0)) - coefs[0]) / coefs[1]  # determine intersection with horizontal axis in log scale
bins_extra = np.linspace(Bins[np.nanargmax(targ.distr_mes2_h0) + 1], np.nanmax([(targ.best_stat ** 2) + 1, bin_extra_max + 1]), 10)
plt.plot(bins_extra, np.exp(bins_extra * coefs[1] + coefs[0]), label='extrapolation, FAP='+f"{targ.false_alarm_prob_trigger:.2e}")
plt.xlim(0.5 * Bins[0], 1.1 * Bins[-1])
plt.savefig(os.path.join(targ.output_location, str(targ.target_id) + '_background_injection_trigger_extrapolation.png'),
            bbox_inches='tight', dpi=320)
plt.clf()
plt.close()
