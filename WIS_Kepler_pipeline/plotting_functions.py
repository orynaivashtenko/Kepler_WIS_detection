#!/usr/bin/env python3

import os
core_num = "1"
os.environ["OMP_NUM_THREADS"] = core_num
os.environ["OPENBLAS_NUM_THREADS"] = core_num
os.environ["MKL_NUM_THREADS"] = core_num
os.environ["VECLIB_MAXIMUM_THREADS"] = core_num
os.environ["NUMEXPR_NUM_THREADS"] = core_num

import numpy as np
from auxiliary_functions import fold_fluxes, get_average_stat_template, get_average_whitened_template
import utils as ut
import pickle
import pipeline_parameters as pp
# from astropy.io import fits
# import sys
import copy
# import search_pipeline as spip
# import target_class

import matplotlib
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 15, 'font.serif': 'Times New Roman', 'lines.linewidth': 1, 'lines.markersize': 3,
                     'legend.fontsize': 12, 'legend.loc': 'upper right',
                     'figure.figsize': (14, 5), 'figure.autolayout': True})
def_cols = plt.rcParams['axes.prop_cycle'].by_key()['color']
Cols = [(0.5, 0.5, 1), (0.2, 0.7, 0.5)]
# plt.ioff()


def save_figure(fig, as_image=True, as_pickle=False, file=None, folder=None, filename='fig', dpi=320):
    if file is None:
        if folder is None:
            folder = os.getcwd()
        file = os.path.join(folder, filename)
    file_0 = str(file)
    if as_image:
        file = str(file_0)
        if file[-len('.png'):] != '.png':
            file = file + '.png'
        fig.savefig(file, bbox_inches='tight', dpi=dpi)
    if as_pickle:
        file = str(file_0)
        if file[-len('.pickle'):] != '.pickle':
            file = file + '.pickle'
        pickle.dump(fig, open(file, 'wb'))


def get_target_title(self, score=None, per=None, temp=None):
    if score is None:
        score = self.best_int_score
    if per is None:
        per = self.best_per
    if temp is None:
        temp = self.best_temp
    title = 'KOI' + self.target_id
    if score is not None:
        title += ', snr ' + str(np.round(score, 1))
    if per is not None:
        title += ', period ' + str(np.round(per, 1)) + ' d'
    if temp is not None:
        title += ', dur ' + str(np.round(self.templates_durations_hours[temp], 1)) + ' h'
    return title


def get_normalized_values(values, subtract_means=False, normalize=False):
    if subtract_means:
        if normalize:
            values = [(values[q] - np.nanmean(values[q])) / np.nanmean(values[q]) for q in range(len(values))]
        else:
            values = [values[q] - np.nanmean(values[q]) for q in range(len(values))]
    elif normalize:
        values = [values[q] / np.nanmean(values[q]) for q in range(len(values))]
    return values


def get_interleaving_stats(target, temp_number):
    if temp_number is None:
        temp_number = target.best_temp
    klc = target.klc
    if temp_number < len(klc.statistics_interleaved):
        statistics = np.array(klc.statistics_interleaved[temp_number])
        variances = np.array(klc.stat_vars_interleaved[temp_number])
        mask = np.array(klc.stat_masks_interleaved[temp_number])
        corrections = np.array(klc.corrections_interleaved[temp_number])
        statistics_gaussianized = np.array(klc.statistics_gaussianized_interleaved[temp_number])
        time = np.array(klc.time_interleaved)
    else:
        statistics = np.array(klc.statistics_not_interleaved[temp_number - len(klc.statistics_interleaved)])
        variances = np.array(klc.stat_vars_not_interleaved[temp_number - len(klc.statistics_interleaved)])
        mask = np.array(klc.stat_masks_not_interleaved[temp_number - len(klc.statistics_interleaved)])
        corrections = np.array(klc.corrections_not_interleaved[temp_number - len(klc.statistics_interleaved)])
        statistics_gaussianized = np.array(
            klc.statistics_gaussianized_not_interleaved[temp_number - len(klc.statistics_interleaved)])
        time = np.array(klc.time_not_interleaved)
    return time, statistics, variances, mask, corrections, statistics_gaussianized


def plot_points_at_times(times, values, point_times, ax=None, **kwargs):
    tms = []
    vals = []
    for tim in point_times:
        for q in range(len(values)):
            if (tim >= np.nanmin(times[q])) and (tim <= np.nanmax(times[q])):
                ind = np.nanargmin(np.abs(times[q] - tim))
                tms.append(times[q][ind])
                vals.append(values[q][ind])
    if ax is None:
        ax = plt.gca()

    if 's' in kwargs:
        ax.scatter(tms, vals, **kwargs)
    else:
        ax.plot(tms, vals, 'o', **kwargs)


def plot_vlines(times, ax=None, kwargs=(('color', 'k'), ('linewidth', 0.5))):
    kwargs = dict(kwargs)
    if ax is None:
        ax = plt.gca()
    for tim in times:
        ax.axvline(tim, **kwargs)


def plot_many(times, values=None, zero_padding_ends=None, exclusion_masks=None, scatter_masks=None,
              ax=None, label_prefix=None, colors=None, scatter_masks_kwargs=(('color', 'k'), ('markersize', 2)),
              subtract_means=False, normalize=False, edge_gap_time=10, **kwargs):
    times = [np.array(times[i]) for i in range(len(times))]
    values = [np.array(values[i]) for i in range(len(values))]

    scatter_masks_kwargs = dict(scatter_masks_kwargs)
    if values is None:
        values = copy.deepcopy(times)
        lengths = [len(values[q]) for q in range(len(values))]
        times = np.arange(np.sum(lengths))
        cumsum_length = np.concatenate([np.array([0]), np.cumsum(lengths)])
        times = [times[cumsum_length[q]:cumsum_length[q + 1]] for q in range(len(times))]

    if zero_padding_ends is not None:
        times = [tt[zero_padding_ends[q][0]: len(tt) - zero_padding_ends[q][-1]] for q, tt in enumerate(times)]
        values = [vv[zero_padding_ends[q][0]: len(vv) - zero_padding_ends[q][-1]] for q, vv in enumerate(values)]
        if exclusion_masks is not None:
            exclusion_masks = [exclusion_masks[q][zero_padding_ends[q][0]: len(exclusion_masks[q]) - zero_padding_ends[q][-1]] for q in
                               range(len(values))]
        if scatter_masks is not None:
            scatter_masks = [scatter_masks[q][zero_padding_ends[q][0]: len(scatter_masks[q]) - zero_padding_ends[q][-1]] for q in
                             range(len(values))]

    values = get_normalized_values(values, subtract_means=subtract_means, normalize=normalize)

    if colors is not None:
        colors = [colors[q % len(colors)] for q in range(len(times))]

    if ax is None:
        ax = plt.gca()

    for q in range(len(times)):
        if colors is not None:
            kwargs['color'] = colors[q]

        plot_time = np.array(times[q])
        plot_value = np.real(values[q])
        if exclusion_masks is not None:
            plot_time[~exclusion_masks[q]] = np.nan
            plot_value[~exclusion_masks[q]] = np.nan

        ax.plot(plot_time, plot_value, label=f'{label_prefix}:{q}' if label_prefix is not None else None, **kwargs)
        if scatter_masks is not None:
            ax.plot(times[q][~scatter_masks[q]], np.real(values[q])[~scatter_masks[q]], 'o', zorder=100, **scatter_masks_kwargs)
    ax.set_xlim(np.nanmin(times[0]) - edge_gap_time, np.nanmax(times[-1]) + edge_gap_time)


def plot_orig_fluxes(self, ax=None, colors=(Cols[0], Cols[1]), plot_trigger=True,
                     points_kwargs=(('color', 'r'), ('markersize', 3), ('label', 'transits')), **kwargs):
    points_kwargs = dict(points_kwargs)
    if ax is None:
        ax = plt.gca()

    times = [tt for q, tt in enumerate(self.klc.orig_times)]
    fluxes = [vv - np.nanmean(vv) for q, vv in enumerate(self.klc.orig_fluxes)]

    plot_many(times=times, values=fluxes, zero_padding_ends=None, exclusion_masks=None, ax=ax, label_prefix=None, colors=colors,
              subtract_means=False, normalize=False, **kwargs)

    if plot_trigger:
        if self.klc.summed_times is not None:
            plot_vlines(times=self.klc.summed_times)
            plot_points_at_times(times=times, values=fluxes, point_times=self.klc.summed_times, ax=ax, **points_kwargs)

    ax.set_xlabel('time, days')
    ax.set_ylabel('Original fluxes, mean subtracted')

    title = get_target_title(self, score=None, per=None, temp=None)
    ax.set_title(title)
    ax.legend(loc='upper right')


def plot_fluxes(self, ax=None, colors=(Cols[0], Cols[1]), plot_trigger=True,
                points_kwargs=(('color', 'r'), ('markersize', 3), ('label', 'transits')), **kwargs):
    points_kwargs = dict(points_kwargs)
    if ax is None:
        ax = plt.gca()

    if self.klc.is_zero_padded:
        zero_padding_ends = self.klc.zero_padding_ends
        times = [tt[zero_padding_ends[q][0]: len(tt) - zero_padding_ends[q][-1]] for q, tt in enumerate(self.klc.times)]
        fluxes = [vv[zero_padding_ends[q][0]: len(vv) - zero_padding_ends[q][-1]] for q, vv in enumerate(self.klc.fluxes)]
        exclusion_masks = [vv[zero_padding_ends[q][0]: len(vv) - zero_padding_ends[q][-1]] for q, vv in enumerate(self.klc.big_hole_masks)]
    else:
        times = self.klc.times
        fluxes = self.klc.fluxes
        exclusion_masks = self.klc.big_hole_masks

    plot_many(times=times, values=fluxes, zero_padding_ends=None, exclusion_masks=exclusion_masks, ax=ax, label_prefix=None, colors=colors,
              subtract_means=False, normalize=False, **kwargs)

    if plot_trigger:
        if self.klc.summed_times is not None:
            plot_vlines(times=self.klc.summed_times)
            plot_points_at_times(times=times, values=fluxes, point_times=self.klc.summed_times, ax=ax, **points_kwargs)

    ax.set_xlabel('time, days')
    ax.set_ylabel('Conditioned fluxes, mean subtracted')

    title = get_target_title(self, score=None, per=None, temp=None)
    ax.set_title(title)
    ax.legend(loc='upper right')


def plot_stats(self, temp_number=None, units='snr', ax=None, edge_gap_time=10, plot_trigger=True,
               points_kwargs=(('color', 'r'), ('markersize', 3)), masked_points_kwargs=(('color', 'k'), ('markersize', 2)),
               mark_zero=True, **kwargs):
    points_kwargs = dict(points_kwargs)
    masked_points_kwargs = dict(masked_points_kwargs)

    if temp_number is None:
        temp_number = self.best_temp

    time, statistics, variances, mask, _, __ = get_interleaving_stats(target=self, temp_number=temp_number)
    if units == 'snr':
        statistics = ut.safe_div(statistics, variances ** 0.5, fill_value=0.)

    if ax is None:
        ax = plt.gca()

    ax.plot(time, statistics, **kwargs)
    plt.plot(time[~mask], statistics[~mask], 'o', **masked_points_kwargs)

    if plot_trigger:
        if self.klc.summed_times is not None:
            plot_vlines(times=self.klc.summed_times)
            plot_points_at_times(times=[time], values=[statistics], point_times=self.klc.summed_times, ax=ax, **points_kwargs,
                                 label='transits')

    if mark_zero:
        ax.axhline(0, linewidth=0.5, color='k')
    ax.set_xlim(np.nanmin(time) - edge_gap_time, np.nanmax(time) + edge_gap_time)
    ax.set_xlabel('time, days')
    ylabel = 'SES statistics'
    if units == 'snr':
        ylabel += ', [SNR]'
    ax.set_ylabel(ylabel)
    title = get_target_title(self, score=None, per=None, temp=temp_number)
    ax.set_title(title)
    ax.legend(loc='upper right')


def plot_whitened_fluxes(self, ax=None, colors=(Cols[0], Cols[1]), plot_trigger=True,
                         points_kwargs=(('color', 'r'), ('markersize', 3), ('label', 'transits')),
                         mark_zero=True, **kwargs):
    points_kwargs = dict(points_kwargs)
    if ax is None:
        ax = plt.gca()

    zero_padding_ends = self.klc.zero_padding_ends
    if self.klc.is_zero_padded:
        times = [tt[zero_padding_ends[q][0]: len(tt) - zero_padding_ends[q][-1]] for q, tt in enumerate(self.klc.times)]
        fluxes = [vv[zero_padding_ends[q][0]: len(vv) - zero_padding_ends[q][-1]] for q, vv in enumerate(self.klc.whitened_fluxes_td)]
        exclusion_masks = [vv[zero_padding_ends[q][0]: len(vv) - zero_padding_ends[q][-1]] for q, vv in enumerate(self.klc.big_hole_masks)]
    else:
        times = self.klc.times
        # fluxes = self.klc.fluxes
        if len(self.klc.whitened_fluxes_td[0]) != len(self.klc.times[0]):
            fluxes = [vv[zero_padding_ends[q][0]: len(vv) - zero_padding_ends[q][-1]] for q, vv in enumerate(self.klc.whitened_fluxes_td)]
        else:
            fluxes = self.klc.whitened_fluxes_td
        exclusion_masks = self.klc.big_hole_masks

    plot_many(times=times, values=fluxes, zero_padding_ends=None, exclusion_masks=exclusion_masks, ax=ax, label_prefix=None, colors=colors,
              subtract_means=False, normalize=False, **kwargs)

    if plot_trigger:
        if self.klc.summed_times is not None:
            plot_vlines(times=self.klc.summed_times)
            plot_points_at_times(times=times, values=fluxes, point_times=self.klc.summed_times, ax=ax, **points_kwargs)

    if mark_zero:
        ax.axhline(0, linewidth=0.5, color='k')
    ax.set_xlabel('time, days')
    ax.set_ylabel('Whitened fluxes, mean subtracted')

    title = get_target_title(self, score=None, per=None, temp=None)
    ax.set_title(title)
    ax.legend(loc='upper right')


def plot_folded_stat_snr(self, temp_number=None, ax=None, plot_twice=False):
    if temp_number is None:
        temp_number = self.best_temp

    time, statistics, variances, mask, corrections, statistics_gaussianized = get_interleaving_stats(target=self, temp_number=temp_number)
    statistics = ut.safe_div(statistics, variances ** 0.5, fill_value=0.)

    # FOLDING
    binwid = min(self.templates_durations_hours[temp_number] / 24 / 5, 0.1)
    bins_time, bins_flux, bins_counts, folds_times, folds_fluxes, bins_std = fold_fluxes(
        times=time, fluxes=statistics, period=self.best_per, first_time=self.best_ftt,
        masks=mask, binwid=binwid, time_fold=self.best_per,  # self.templates_durations_hours[temp_number] / 24 * 10,
        dt=self.klc.dt_days, means=None,
        template=None, return_std=True)  # np.array(self.klc.templates[self.klc.indices_of_templates[temp_number][0]]))

    # MAKING TEMPLATE
    n_time_templ = int(np.round(self.best_per / self.klc.dt_days))
    temp_plot, time_template = get_average_stat_template(whitened_templates_td=self.klc.whitened_templates_td,
                                                         temp_number=temp_number, n_time_templ=n_time_templ,
                                                         dt_days=self.klc.dt_days, indices_of_templates=self.klc.indices_of_templates)
    temp_plot = - temp_plot * self.best_stat ** 0.5 / (len(self.klc.summed_stats_raw)) ** 0.5
    # iot = self.klc.indices_of_templates[temp_number]
    # if iot[1] - iot[0] > 1:
    #     temps = [np.nanmean([self.klc.whitened_templates_td[i][q] for i in range(iot[0], iot[1])], axis=0)
    #              for q in range(len(self.klc.whitened_templates_td[0]))]
    # else:
    #     temps = self.klc.whitened_templates_td[self.klc.indices_of_templates[temp_number][0]]
    #     temps = [np.array(t) for t in temps]
    #
    # # temps = self.klc.whitened_templates_td[self.klc.indices_of_templates[temp_number][0]]
    # # for q in range(len(temps)):
    # #     if len(temps[q]) < n_time_templ:
    # #         temps[q] = ut.pad_by_zeros(temps[q], n_time_templ, in_the_center=True)
    # temp_best_1 = np.array(np.mean([temps[q][:n_time_templ // 2] for q in self.klc.quarters], axis=0))
    # temp_best_2 = np.array(np.mean([temps[q][n_time_templ // 2:] for q in self.klc.quarters], axis=0))
    # temp_best = np.concatenate((temp_best_1, temp_best_2))  # the maximum is supposed to be at zero
    # temp_best = temp_best / np.linalg.norm(temp_best)
    # temp_best = np.roll(temp_best, -np.argmax(temp_best))  # although the maximum is supposed to be at zero from before
    # temp_best = np.fft.irfft(np.abs(np.fft.rfft(temp_best)) ** 2)  # for statistic, template is convolution
    # tem_center = n_time_templ // 2
    # temp_best = np.roll(temp_best, tem_center)  # the maximum is supposed to be at zero before
    # time_template = np.concatenate((np.arange(tem_center) - tem_center, np.arange(len(temp_best) - tem_center))) * self.klc.dt_days
    # temp_plot = - temp_best * self.best_stat ** 0.5 / (len(self.klc.summed_stats_raw)) ** 0.5

    # Plotting
    if ax is None:
        ax = plt.gca()

    # If someone wants to repeat the period two times as it is often done in the literature:
    if plot_twice:
        time_shifts_plotting = [0, self.best_per]
        labels_binning = ['binning, bin ' + ut.fstr(binwid) + ' days', None]
        labels_template = ['average per-transit snr', None]
    else:
        time_shifts_plotting = [0]
        labels_binning = ['binning, bin ' + ut.fstr(binwid) + ' days']
        labels_template = ['average per-transit snr']

    # Actual plotting:
    for j in range(len(time_shifts_plotting)):
        # Plot the folded data:
        for i in range(len(folds_times)):
            ax.plot(folds_times[i] + time_shifts_plotting[j], folds_fluxes[i], 'o', markersize=0.6, color=(0.6, 0.6, 0.6))
        # Plot the binned folded data:
        ax.errorbar(bins_time[bins_counts > 0] + time_shifts_plotting[j], bins_flux[bins_counts > 0], yerr=bins_std[bins_counts > 0],
                    xerr=None, markersize=2.5, fmt='o', markerfacecolor='k', markeredgecolor='k',
                    ecolor='k', elinewidth=1.2, capsize=None, barsabove=False, lolims=False, uplims=False,
                    xlolims=False, xuplims=False, errorevery=1, capthick=None,
                    label=labels_binning[j])
        # Plot the expected template:
        ax.plot(time_template + time_shifts_plotting[j], temp_plot, color='r', linewidth=1,
                label=labels_template[j])

    # Setting the limits:
    maxdev = np.nanmax(
        [np.nanmax(np.abs(bins_flux)), np.nanmax([np.nanmax(np.abs(folds_fluxes[i])) for i in range(len(folds_fluxes))])])
    ax.set_ylim(-maxdev * 1.01, maxdev * 1.01)  # make the limits symmetric
    ax.set_xlim(-self.best_per / 2, self.best_per / 2)
    if plot_twice:
        ax.set_xlim(-self.best_per / 2, self.best_per / 2 + self.best_per)
        ax.axvline(self.best_per / 2, color=(0.5, 0.5, 0.5), linewidth=1)
    ax.set_xlabel('Time from the mid transit, days')
    ax.set_ylabel('Statistic [snr] fold')
    title = get_target_title(self, score=None, per=None, temp=temp_number)
    ax.set_title(title)
    ax.legend(loc='upper right')


def plot_folded_whitened_flux(self, ax=None, plot_twice=False, temp_number=None):
    if temp_number is None:
        temp_number = self.best_temp
    # FOLDING
    binwid = min(self.templates_durations_hours[temp_number] / 24 / 5, 0.1)
    bins_time, bins_flux, bins_counts, folds_times, folds_fluxes, bins_std = fold_fluxes(
        times=self.klc.times, fluxes=self.klc.whitened_fluxes_td, period=self.best_per, first_time=self.best_ftt,
        masks=self.klc.big_hole_masks,
        binwid=binwid, time_fold=self.best_per,  # self.templates_durations_hours[temp_number] / 24 * 10,
        dt=self.klc.dt_days, means=None,
        template=None, return_std=True)  # np.array(self.klc.templates[self.klc.indices_of_templates[temp_number][0]]))
    n_time_templ = int(np.round(self.best_per / self.klc.dt_days))
    temp_best, time_template = get_average_whitened_template(whitened_templates_td=self.klc.whitened_templates_td,
                                                             temp_number=temp_number, n_time_templ=n_time_templ,
                                                             dt_days=self.klc.dt_days, indices_of_templates=self.klc.indices_of_templates)
    non_zero_mask = self.klc.summed_vars > 0
    ampl_est = np.nanmean(self.klc.summed_stats_raw[non_zero_mask] / self.klc.summed_vars[non_zero_mask])

    # Plotting
    if ax is None:
        ax = plt.gca()

    # If someone wants to repeat the period two times as it is often done in the literature:
    if plot_twice:
        time_shifts_plotting = [0, self.best_per]
        labels_binning = ['binning, bin ' + ut.fstr(binwid) + ' days', None]
        labels_template = ['average whitened template', None]
    else:
        time_shifts_plotting = [0]
        labels_binning = ['binning, bin ' + ut.fstr(binwid) + ' days']
        labels_template = ['average whitened template']

    # Actual plotting:
    for j in range(len(time_shifts_plotting)):
        for i in range(len(folds_times)):
            ax.plot(folds_times[i] + time_shifts_plotting[j], folds_fluxes[i], 'o', markersize=1, color=(0.5, 0.5, 0.5))
        ax.errorbar(bins_time[bins_counts > 0] + time_shifts_plotting[j], bins_flux[bins_counts > 0], yerr=bins_std[bins_counts > 0],
                    xerr=None,
                    markersize=2.5, fmt='o', markerfacecolor='k', markeredgecolor='k',
                    ecolor='k', elinewidth=1.2, capsize=None, barsabove=False, lolims=False, uplims=False,
                    xlolims=False, xuplims=False, errorevery=1, capthick=None,
                    label=labels_binning[j])
        # ax.plot(bins_time, bins_flux, '-o', markersize=3, color='b', linewidth=1, label='binning, bin ' + ut.fstr(binwid) + ' days')
        ax.plot(time_template + time_shifts_plotting[j], temp_best * ampl_est, color='r', linewidth=1,
                label=labels_template[j])
    # Setting limits:
    maxdev = np.nanmax(
        [np.nanmax(np.abs(bins_flux)), np.nanmax([np.nanmax(np.abs(folds_fluxes[i])) for i in range(len(folds_fluxes))])])
    ax.set_ylim(-maxdev * 1.01, maxdev * 1.01)  # make the limits symmetric
    ax.set_xlim(-self.best_per / 2, self.best_per / 2)
    if plot_twice:
        ax.set_xlim(-self.best_per / 2, self.best_per / 2 + self.best_per)
        ax.axvline(self.best_per / 2, color=(0.5, 0.5, 0.5), linewidth=1)
    ax.set_xlabel('Time from the mid transit, days')
    ax.set_ylabel('Whitened flux')
    title = get_target_title(self, score=None, per=None, temp=temp_number)
    ax.set_title(title)
    ax.legend(loc='upper right')


def plot_histograms(self, ax=None, figsize=(10, 5), fixed_limits=False, limits=None, plot_extrapolation=True,
                    fontsize_legend=12, fontsize_title=13, fontsize_label=12,
                    linewidth=2, use_integral=True, use_clean=True):
    # PLOT BACKGROUND AND INJECTION-RECOVERY
    if self.prob_density_planet_not_noise is None:
        self.analyze_injections_and_background()

    # --------------------------------------------
    # SETTING PARAMS
    if use_integral:
        distr_score_h0, distr_score_inj_all, distr_score_inj_success, bins, best_score = \
            self.distr_int_score_h0, self.distr_int_score_inj_all, self.distr_int_score_inj_success, self.bins_int_score_histogram,\
            self.best_int_score
    else:
        distr_score_h0, distr_score_inj_all, distr_score_inj_success, bins, best_score = \
            self.distr_stat_h0, self.distr_stat_inj_all, self.distr_stat_inj_success, self.bins_stat_histogram, \
            self.best_stat

    if fixed_limits:
        if limits is not None:
            min_plot_mes2 = limits[0]
            max_plot_mes2 = limits[1]
        else:
            min_plot_mes2 = pp.Default_min_plot_stat
            max_plot_mes2 = pp.Default_max_plot_stat
    else:
        min_plot_mes2 = 0.9 * bins[0]
        max_plot_mes2 = 1.1 * bins[-1]
    # UNIFORMLY FORMATTED PLOT
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.gca()

    # --------------------------------------------
    # PLOTTING TRIGGER
    # ax.axvline(best_score, color='k', linewidth=linewidth,
    #            label='trigger: \nsnr2=' + str(np.round(np.abs(best_score), 1)) +
    #            '\nperiod=' + str(int(np.round(self.best_per, 0))) + ' d' + '\nduration=' + str(
    #             int(np.round(self.templates_durations_hours[self.best_temp], 0))) + ' h')
    ax.axvline(best_score, color='k', linewidth=linewidth,
               label=' ' * 13 + r'snr$^2$=' + str(np.round(np.abs(best_score), 1))
                     + ',\n' + 'trigger: period=' + str(int(np.round(self.best_per, 0))) + ' d'
                     + ',\n' + ' ' * 13 + 'duration='
                     + str(int(np.round(self.templates_durations_hours[self.best_temp], 0))) + ' h')

    # --------------------------------------------
    # PLOTTING HISTOGRAMS
    if distr_score_inj_all is not None:
        ax.stairs(distr_score_inj_all, bins, label='all injections in relative period range', linewidth=linewidth, color=def_cols[0],
                  linestyle='--')  # after density correction factor, get per one star
        ax.stairs(distr_score_inj_success, bins, label='success injections',
                  linewidth=linewidth, color=def_cols[0], linestyle='-')  # after density correction factor, get per one star
    ax.stairs(distr_score_h0, bins, label='background, scrambling', linewidth=linewidth, color=def_cols[1])

    # --------------------------------------------
    # PLOTTING CLEAN
    if use_clean:
        if use_integral:
            distr_score_h0_clean = self.distr_int_score_h0_clean
        else:
            distr_score_h0_clean = self.distr_stat_h0_clean
        ax.stairs(distr_score_h0_clean, bins, label='background, scrambling no trigger',
                  linewidth=linewidth, color=def_cols[1], linestyle='--')

    # --------------------------------------------
    # PLOTTING EXTRAPOLATION
    if plot_extrapolation and use_integral:
        # MAKE EXTRAPOLATION
        if not hasattr(self, 'extrapolation_coefs_int_score_h0_hist') or self.extrapolation_coefs_int_score_h0_hist is None:
            from auxiliary_functions import get_histogram_extrapolation_coefs_in_log
            self.extrapolation_coefs_int_score_h0_hist = get_histogram_extrapolation_coefs_in_log(
                    self.distr_int_score_h0, self.bins_int_score_histogram)
        coefs = self.extrapolation_coefs_int_score_h0_hist
        # determine intersection with horizontal axis in log scale:
        bin_extra_max = (np.log(np.nanmin(self.distr_int_score_h0[self.distr_int_score_h0 != 0])) - coefs[0]) / coefs[1]
        bin_extra_max = np.nanmax([self.best_int_score + 1, bin_extra_max + 1])
        bin_extra_min = bins[np.nanargmax(self.distr_int_score_h0) + 1]
        bin_min_distr = bins[:-1][((~np.isnan(self.distr_int_score_h0)) & (self.distr_int_score_h0 > 0))][0]
        # bin_extra_min = int((bin_extra_min + bin_extra_max) / 2)
        bin_extra_min = bin_extra_min + int(np.abs(bin_extra_min - bin_min_distr) / 2)
        bins_extra = np.linspace(bin_extra_min, bin_extra_max, 10)
        label = 'extrapolation'
        if self.false_alarm_prob_trigger is not None:
            label += ', FAP=' + f"{self.false_alarm_prob_trigger:.2e}"
        ax.plot(bins_extra, np.exp(bins_extra * coefs[1] + coefs[0]), '--', label=label, color=def_cols[1])

    # --------------------------------------------
    # FORMATTING
    if fontsize_legend is not None:
        ax.legend(loc='upper right', fontsize=fontsize_legend)
    else:
        ax.legend(loc='upper right')

    title = 'KID' + self.target_id
    if self.injections_period_range is not None:
        title = (title + ', periods ['
                 + str(np.round(self.injections_period_range[0], 1)) + '-'
                 + str(np.round(self.injections_period_range[1], 1)) + ']'
                 + '] days')
    if fontsize_title is not None:
        ax.set_title(title, fontsize=fontsize_title)
    else:
        ax.set_title(title)

    xlabel = 'Recovered MES [$SNR^2$]'
    ylabel = 'Counts per effective one star \n per period bin'
    if fontsize_label is not None:
        ax.set_xlabel(xlabel, fontsize=fontsize_label)
        ax.set_ylabel(ylabel, fontsize=fontsize_label)
    else:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    ax.set_yscale('log')
    ax.set_xlim(min_plot_mes2, max_plot_mes2)
    ax.grid(which='both', zorder=0, color=0.7 * np.ones(3), linewidth=0.7)

    if fontsize_legend is not None:
        ax.legend(loc='upper right', fontsize=fontsize_legend)
    else:
        ax.legend(loc='upper right')


# ##########################################################################################################
# FORMATTING FUNCTIONS
# ##########################################################################################################

def make_cmap(my_colors=np.array([[0.9, 1, 1, 1], [0.6, 0.8, 1, 1], [0.2, 0.2, 0.9, 1]]), n_samples=None, c_under=None, c_over=None):
    """
    my_colors=np.array([[0,0.7,0.5,1], [0.2,0.2,1,1], [0.5,0,0.8,1]])  # green-blue-purple
    my_colors=np.array([[0,0.7,0.5,1], [0.2,0.2,1,1]])  # green-blue
    my_colors=np.array([[0.4,0.9,0.9,1], [0.2,0.2,0.9,1]])  # cyan-blue
    :param my_colors:
    :type my_colors:
    :param n_samples:
    :type n_samples:
    :param c_under:
    :type c_under:
    :param c_over:
    :type c_over:
    :return:
    :rtype:
    """
    if n_samples is None:
        cmap = matplotlib.colormaps['viridis']
    else:
        cmap = matplotlib.colormaps['viridis'].resampled(n_samples)
    target_len = len(cmap.colors)
    my_colors_interp = np.array(
        [np.interp(np.linspace(0, 1, target_len), np.linspace(0, 1, len(my_colors)), my_colors[:, i]) for i in range(my_colors.shape[1])]).T
    cmap.colors = my_colors_interp
    if c_under is not None:
        cmap.set_under(c_under)
    if c_over is not None:
        cmap.set_over(c_over)
    return cmap


def make_coloring(values, min_val=None, max_val=None, cmap=None, cmap_name='winter', cmap_under=None, cmap_over=None):
    """
    cmap_name='winter', cmap_under=(0.8,0,1), cmap_over=(0.8,0.8,0)
    """
    values = np.array(values, dtype=float)
    if min_val is None:
        min_val = np.nanmin(values)
    if max_val is None:
        max_val = np.nanmax(values)

    scals = (values - min_val) / (max_val - min_val)
    if cmap is None:
        cmap = matplotlib.colormaps[cmap_name]
    # d_col = np.array(cmap(1)) - np.array(cmap(0))
    if cmap_under is not None:
        cmap.set_under(cmap_under)
    if cmap_over is not None:
        cmap.set_over(cmap_over)
    colors = cmap(scals)

    mpbl = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=min_val, vmax=max_val, clip=True), cmap=cmap)
    return colors, mpbl
