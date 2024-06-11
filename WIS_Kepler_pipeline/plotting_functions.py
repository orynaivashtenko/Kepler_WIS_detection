import os
core_num = "1"
os.environ["OMP_NUM_THREADS"] = core_num
os.environ["OPENBLAS_NUM_THREADS"] = core_num
os.environ["MKL_NUM_THREADS"] = core_num
os.environ["VECLIB_MAXIMUM_THREADS"] = core_num
os.environ["NUMEXPR_NUM_THREADS"] = core_num

import numpy as np
from auxiliary_functions import fold_fluxes
import utils as ut
import pickle
# from astropy.io import fits
# import sys
import copy
# import search_pipeline as spip
# import target_class

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 15, 'font.serif': 'Times New Roman', 'lines.linewidth': 1, 'lines.markersize': 3,
                     'legend.fontsize': 13, 'legend.loc': 'upper right',
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


def get_target_title(self):
    title = 'KOI' + self.target_id + ', '
    if self.best_stat is not None:
        title += 'snr ' + str(np.round(self.best_stat, 1))
    if self.best_per is not None:
        title += ', period ' + str(np.round(self.best_per, 1)) + ' d'
    if self.best_temp is not None:
        title += ', dur ' + str(np.round(self.templates_durations_hours[self.best_temp], 1)) + ' h'
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
    klc = target.klc
    if temp_number < len(klc.statistics_interleaved):
        statistics = np.array(klc.statistics_interleaved[target.best_temp])
        variances = np.array(klc.stat_vars_interleaved[target.best_temp])
        mask = np.array(klc.stat_masks_interleaved[target.best_temp])
        corrections = np.array(klc.corrections_interleaved[target.best_temp])
        statistics_gaussianized = np.array(klc.statistics_gaussianized_interleaved[target.best_temp])
        time = np.array(klc.time_interleaved)
    else:
        statistics = np.array(klc.statistics_not_interleaved[target.best_temp - len(klc.statistics_interleaved)])
        variances = np.array(klc.stat_vars_not_interleaved[target.best_temp - len(klc.statistics_interleaved)])
        mask = np.array(klc.stat_masks_not_interleaved[target.best_temp - len(klc.statistics_interleaved)])
        corrections = np.array(klc.corrections_not_interleaved[target.best_temp - len(klc.statistics_interleaved)])
        statistics_gaussianized = np.array(
            klc.statistics_gaussianized_not_interleaved[target.best_temp - len(klc.statistics_interleaved)])
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


def vlines(times, ax=None, kwargs=(('color', 'k'), ('linewidth', 0.5))):
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
        cumsum_length = np.concatenate([[0], np.cumsum(lengths)])
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


def plot_orig_fluxes(self, ax=None, colors=(Cols[0], Cols[1]),
                points_kwargs=(('color', 'r'), ('markersize', 3), ('label', 'transits')), **kwargs):
    points_kwargs = dict(points_kwargs)
    if ax is None:
        ax = plt.gca()

    times = [tt for q, tt in enumerate(self.klc.orig_times)]
    fluxes = [vv - np.nanmean(vv) for q, vv in enumerate(self.klc.orig_fluxes)]

    plot_many(times=times, values=fluxes, zero_padding_ends=None, exclusion_masks=None, ax=ax, label_prefix=None, colors=colors,
              subtract_means=False, normalize=False, **kwargs)

    if self.klc.summed_times is not None:
        vlines(times=self.klc.summed_times)
        plot_points_at_times(times=times, values=fluxes, point_times=self.klc.summed_times, ax=ax, **points_kwargs)

    ax.set_xlabel('time, days')
    ax.set_ylabel('Original fluxes, mean subtracted')

    title = get_target_title(self)
    ax.set_title(title)
    ax.legend(loc='upper right')


def plot_fluxes(self, ax=None, colors=(Cols[0], Cols[1]),
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

    if self.klc.summed_times is not None:
        vlines(times=self.klc.summed_times)
        plot_points_at_times(times=times, values=fluxes, point_times=self.klc.summed_times, ax=ax, **points_kwargs)

    ax.set_xlabel('time, days')
    ax.set_ylabel('Conditioned fluxes, mean subtracted')

    title = get_target_title(self)
    ax.set_title(title)
    ax.legend(loc='upper right')


def plot_stats(self, temp_number=None, units='snr', ax=None, edge_gap_time=10,
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

    if self.klc.summed_times is not None:
        vlines(times=self.klc.summed_times)
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
    title = get_target_title(self)
    ax.set_title(title)
    ax.legend(loc='upper right')


def plot_whitened_fluxes(self, ax=None, colors=(Cols[0], Cols[1]),
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
        fluxes = [vv[zero_padding_ends[q][0]: len(vv) - zero_padding_ends[q][-1]] for q, vv in enumerate(self.klc.whitened_fluxes_td)]
        exclusion_masks = self.klc.big_hole_masks

    plot_many(times=times, values=fluxes, zero_padding_ends=None, exclusion_masks=exclusion_masks, ax=ax, label_prefix=None, colors=colors,
              subtract_means=False, normalize=False, **kwargs)

    if self.klc.summed_times is not None:
        vlines(times=self.klc.summed_times)
        plot_points_at_times(times=times, values=fluxes, point_times=self.klc.summed_times, ax=ax, **points_kwargs)

    if mark_zero:
        ax.axhline(0, linewidth=0.5, color='k')
    ax.set_xlabel('time, days')
    ax.set_ylabel('Whitened fluxes, mean subtracted')

    title = get_target_title(self)
    ax.set_title(title)
    ax.legend(loc='upper right')


def plot_folded_stat_snr(self, temp_number=None, ax=None, plot_twice=False):
    if temp_number is None:
        temp_number = self.best_temp

    time, statistics, variances, mask, corrections, statistics_gaussianized = get_interleaving_stats(target=self, temp_number=temp_number)
    statistics = ut.safe_div(statistics, variances ** 0.5, fill_value=0.)

    # FOLDING
    binwid = min(self.templates_durations_hours[self.best_temp] / 24 / 5, 0.1)
    bins_time, bins_flux, bins_counts, folds_times, folds_fluxes, bins_std = fold_fluxes(
        times=time, fluxes=statistics, period=self.best_per, first_time=self.best_ftt,
        masks=mask, binwid=binwid, time_fold=self.best_per,  # self.templates_durations_hours[self.best_temp] / 24 * 10,
        dt=self.klc.dt_days, means=None,
        template=None, return_std=True)  # np.array(self.klc.templates[self.klc.indices_of_templates[self.best_temp][0]]))

    # MAKING TEMPLATE
    n_time_templ = int(np.round(self.best_per / self.klc.dt_days))
    temps = self.klc.whitened_templates_td[self.klc.indices_of_templates[self.best_temp][0]]
    for q in range(len(temps)):
        if len(temps[q]) < n_time_templ:
            temps[q] = ut.pad_by_zeros(temps[q], n_time_templ, in_the_center=True)
    temp_best_1 = np.array(np.mean([temps[q][:n_time_templ // 2] for q in self.klc.quarters], axis=0))
    temp_best_2 = np.array(np.mean([temps[q][-n_time_templ // 2:] for q in self.klc.quarters], axis=0))
    temp_best = np.concatenate((temp_best_1, temp_best_2))  # the maximum is supposed to be at zero
    temp_best = temp_best / np.linalg.norm(temp_best)
    temp_best = np.roll(temp_best, -np.argmax(temp_best))  # although the maximum is supposed to be at zero from before
    temp_best = np.fft.irfft(np.abs(np.fft.rfft(temp_best)) ** 2)  # for statistic, template is convolution
    tem_center = n_time_templ // 2
    temp_best = np.roll(temp_best, tem_center)  # the maximum is supposed to be at zero before
    time_template = np.concatenate((np.arange(tem_center) - tem_center, np.arange(len(temp_best) - tem_center))) * self.klc.dt_days
    temp_plot = - temp_best * self.best_stat ** 0.5 / (len(self.klc.summed_stats_raw)) ** 0.5

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
    title = get_target_title(self)
    ax.set_title(title)
    ax.legend(loc='upper right')


def plot_folded_whitened_flux(self, ax=None, plot_twice=False):
    # FOLDING
    binwid = min(self.templates_durations_hours[self.best_temp] / 24 / 5, 0.1)
    bins_time, bins_flux, bins_counts, folds_times, folds_fluxes, bins_std = fold_fluxes(
        times=self.klc.times, fluxes=self.klc.whitened_fluxes_td, period=self.best_per, first_time=self.best_ftt,
        masks=self.klc.big_hole_masks,
        binwid=binwid, time_fold=self.best_per,  # self.templates_durations_hours[self.best_temp] / 24 * 10,
        dt=self.klc.dt_days, means=None,
        template=None, return_std=True)  # np.array(self.klc.templates[self.klc.indices_of_templates[self.best_temp][0]]))
    n_time_templ = int(np.round(self.best_per / self.klc.dt_days))
    temps = self.klc.whitened_templates_td[self.klc.indices_of_templates[self.best_temp][0]]
    for q in range(len(temps)):
        if len(temps[q]) < n_time_templ:
            temps[q] = ut.pad_by_zeros(temps[q], n_time_templ, in_the_center=True)
    temp_best_1 = np.array(np.mean([temps[q][:n_time_templ // 2] for q in self.klc.quarters], axis=0))
    temp_best_2 = np.array(np.mean([temps[q][-n_time_templ // 2:] for q in self.klc.quarters], axis=0))
    time_template_1 = np.arange(len(temp_best_1)) * self.klc.dt_days
    time_template_2 = (np.arange(len(temp_best_2)) * self.klc.dt_days * (-1))[::-1]
    temp_best = np.concatenate((temp_best_2, temp_best_1))
    time_template = np.concatenate((time_template_2, time_template_1))
    # tem_center = np.nanargmax(np.abs(temp_best))
    # time_template = np.concatenate((np.arange(tem_center) - tem_center, np.arange(len(temp_best) - tem_center))) * self.klc.dt_days

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
    title = get_target_title(self)
    ax.set_title(title)
    ax.legend(loc='upper right')
