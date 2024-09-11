#!/usr/bin/env python3

import os
core_num = "1"
os.environ["OMP_NUM_THREADS"] = core_num
os.environ["OPENBLAS_NUM_THREADS"] = core_num
os.environ["MKL_NUM_THREADS"] = core_num
os.environ["VECLIB_MAXIMUM_THREADS"] = core_num
os.environ["NUMEXPR_NUM_THREADS"] = core_num

import numpy as np
from auxiliary_functions import (fold_fluxes, get_average_stat_template, get_average_whitened_template,
                                 get_shift_reduced_full_stat_snr, get_near_transit_pieces_from_full_time,
                                 get_near_transit_pieces_of_whitened_flux)
import utils as ut
import pickle
import pipeline_parameters as pp
import locations as locs
from auxiliary_functions import get_saving_file_path, get_correct_tid_folder
# from astropy.io import fits
# import sys
import copy
# import search_pipeline as spip
# import target_class

from plotting_parameters import *


# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
class PlotWrapper(object):
    def __init__(self, save_fig=Save_fig, no_show=No_show, ax=None, figsize=None, target_id=None,
                 filename_postfix='plot.png', plot_output_location=locs.Triggers_plots_location, dpi=Plots_dpi,
                 grid=True, color_grid=Color_grid, linewidth_grid=Linewidth_grid, which_grid='both',
                 allow_plot_folder_creation=False):
        self.save_fig = save_fig
        self.no_show = no_show
        self.plot_output_location = plot_output_location
        self.dpi = dpi
        self.grid = grid
        self.grid_color = color_grid
        self.grid_linewidth = linewidth_grid
        self.which_grid = which_grid
        if self.save_fig:
            tid_folder = get_correct_tid_folder(target_id=target_id, location=plot_output_location,
                                                allow_creation=allow_plot_folder_creation, with_prefix_folder=False)
            self.file_saving_path = get_saving_file_path(output_location=tid_folder,
                                                         target_id=target_id,
                                                         filename_postfix=filename_postfix,
                                                         file_format='png')
        # fig.subplots_adjust(left=None, bottom=None, right=None, top=0.94, wspace=None, hspace=0)
        if ax is None:
            self.fig = plt.figure(figsize=figsize)
            self.ax = plt.gca()
        else:
            self.fig = plt.gcf()

    def __enter__(self):
        return self.fig, self.ax

    def __exit__(self, *args):
        if self.grid:
            self.ax.grid(which=self.which_grid, color=self.grid_color, linewidth=self.grid_linewidth, zorder=0)
        if self.save_fig:
            self.fig.savefig(self.file_saving_path, dpi=self.dpi, bbox_inches='tight')
        if self.no_show:
            plt.clf()
            plt.close(self.fig)
        else:
            plt.show()


class SubplotsWrapper(object):
    def __init__(self, nrows=2, ncols=2, sharex=True, sharey=True, gridspec_kw=Subplots_gridspec_kw,
                 save_fig=Save_fig, no_show=No_show, figsize=Figsize, target_id=None,
                 filename_postfix='plot.png', plot_output_location=locs.Triggers_plots_location, dpi=Plots_dpi,
                 grid=True, color_grid=Color_grid, linewidth_grid=Linewidth_grid,
                 zero_lines=True, zero_lines_linewidth=Linewidth_zero_level, allow_plot_folder_creation=False):
        self.nrows = nrows
        self.ncols = ncols
        self.save_fig = save_fig
        self.no_show = no_show
        self.plot_output_location = plot_output_location
        self.dpi = dpi
        self.grid = grid
        self.grid_color = color_grid
        self.grid_linewidth = linewidth_grid
        self.zero_lines = zero_lines
        self.zero_lines_linewidth = zero_lines_linewidth
        if self.save_fig:
            tid_folder = get_correct_tid_folder(target_id=target_id, location=plot_output_location,
                                                allow_creation=allow_plot_folder_creation, with_prefix_folder=False)
            self.file_saving_path = get_saving_file_path(output_location=tid_folder,
                                                         target_id=target_id,
                                                         filename_postfix=filename_postfix,
                                                         file_format='png')
        self.fig, self.axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex=sharex, sharey=sharey,
                                          gridspec_kw=gridspec_kw)

    def __enter__(self):
        return self.fig, self.axs

    def __exit__(self, *args):
        if self.grid:
            if not (self.nrows == 1 and self.ncols == 1):
                for ax in self.axs.flatten():
                    ax.grid(which='both', color=self.grid_color, linewidth=self.grid_linewidth, zorder=0)
            else:
                self.axs.grid(which='both', color=self.grid_color, linewidth=self.grid_linewidth, zorder=0)
        if self.zero_lines:
            if not (self.nrows == 1 and self.ncols == 1):
                for ax in self.axs.flatten():
                    ax.axvline(0, color='k', linestyle='-', linewidth=self.zero_lines_linewidth, zorder=2)
                    ax.axhline(0, color='k', linestyle='-', linewidth=self.zero_lines_linewidth, zorder=2)
            else:
                self.axs.axvline(0, color='k', linestyle='-', linewidth=self.zero_lines_linewidth, zorder=2)
                self.axs.axhline(0, color='k', linestyle='-', linewidth=self.zero_lines_linewidth, zorder=2)
        if self.save_fig:
            self.fig.savefig(self.file_saving_path, dpi=self.dpi, bbox_inches='tight')
        if self.no_show:
            plt.clf()
            plt.close(self.fig)
        else:
            plt.show()


# ---------------------------------------------------------------------------------------------------------------------
# Proper plotting functions
# ---------------------------------------------------------------------------------------------------------------------

def plot_orig_fluxes(target, colors=Flux_quarters_colors, plot_trigger=True,
                     ax=None, save_fig=Save_fig, no_show=No_show, plot_output_location=locs.Triggers_plots_location,
                     figsize=Figsize_flux, filename_postfix=locs.Orig_fluxes_plot_filename_postfix,
                     points_kwargs=(('color', 'r'), ('markersize', 3), ('label', 'transits')), **kwargs):
    points_kwargs = dict(points_kwargs)
    prepare_target_for_plotting(target, initialize_templates=False,
                                condition_templates=False, detect_single_transits=False)
    with PlotWrapper(target_id=target.target_id, save_fig=save_fig, no_show=no_show, ax=ax, figsize=figsize,
                     filename_postfix=filename_postfix, plot_output_location=plot_output_location) as (fig, axx):
        times = [tt for q, tt in enumerate(target.klc.orig_times)]
        fluxes = [vv - np.nanmean(vv) for q, vv in enumerate(target.klc.orig_fluxes)]
        plot_many(times=times, values=fluxes, zero_padding_ends=None, exclusion_masks=None, ax=axx, label_prefix=None, colors=colors,
                  subtract_means=False, normalize=False, **kwargs)

        if plot_trigger:
            if target.klc.summed_times is not None:
                plot_vlines(times=target.klc.summed_times)
                plot_points_at_times(times=times, values=fluxes, point_times=target.klc.summed_times, ax=axx, **points_kwargs)

        axx.set_xlabel('time, days')
        axx.set_ylabel('Original fluxes, mean subtracted')

        title = get_target_title(target, score=None, per=None, temp=None)
        axx.set_title(title)
        axx.legend(loc='upper right')


def plot_fluxes(target, colors=(Flux_quarters_colors[0], Flux_quarters_colors[1]), plot_trigger=True,
                ax=None, save_fig=Save_fig, no_show=No_show, plot_output_location=locs.Triggers_plots_location,
                figsize=Figsize_flux, filename_postfix=locs.Red_fluxes_plot_filename_postfix,
                points_kwargs=(('color', 'r'), ('markersize', 3), ('label', 'transits')), **kwargs):
    points_kwargs = dict(points_kwargs)
    prepare_target_for_plotting(target, initialize_templates=False,
                                condition_templates=False, detect_single_transits=False)
    with PlotWrapper(ax=ax, save_fig=save_fig, no_show=no_show, figsize=figsize, target_id=target.target_id,
                     filename_postfix=filename_postfix, plot_output_location=plot_output_location) as (fig, axx):

        if target.klc.is_zero_padded:
            zero_padding_ends = target.klc.zero_padding_ends
            times = [tt[zero_padding_ends[q][0]: len(tt) - zero_padding_ends[q][-1]] for q, tt in enumerate(target.klc.times)]
            fluxes = [vv[zero_padding_ends[q][0]: len(vv) - zero_padding_ends[q][-1]] for q, vv in enumerate(target.klc.fluxes)]
            exclusion_masks = [vv[zero_padding_ends[q][0]: len(vv) - zero_padding_ends[q][-1]]
                               for q, vv in enumerate(target.klc.big_hole_masks)]
        else:
            times = target.klc.times
            fluxes = target.klc.fluxes
            exclusion_masks = target.klc.big_hole_masks

        plot_many(times=times, values=fluxes, zero_padding_ends=None,
                  exclusion_masks=exclusion_masks, ax=axx, label_prefix=None, colors=colors,
                  subtract_means=False, normalize=False, **kwargs)

        if plot_trigger:
            if target.klc.summed_times is not None:
                plot_vlines(times=target.klc.summed_times)
                plot_points_at_times(times=times, values=fluxes, point_times=target.klc.summed_times, ax=axx, **points_kwargs)

        axx.set_xlabel('time, days')
        axx.set_ylabel('Conditioned fluxes, mean subtracted')

        title = get_target_title(target, score=None, per=None, temp=None)
        axx.set_title(title)
        axx.legend(loc='upper right')


def plot_stats(target, temp_number=None, units='snr', edge_gap_time=10, plot_trigger=True,
               points_kwargs=(('color', 'r'), ('markersize', 3)), masked_points_kwargs=(('color', 'k'), ('markersize', 1)),
               mark_zero=True, color=def_cols[0],
               ax=None, save_fig=Save_fig, no_show=No_show, plot_output_location=locs.Triggers_plots_location,
               figsize=Figsize_flux, filename_postfix=locs.Single_event_statistics_plot_filename_postfix,
               **kwargs):
    points_kwargs = dict(points_kwargs)
    masked_points_kwargs = dict(masked_points_kwargs)
    prepare_target_for_plotting(target, initialize_templates=False,
                                condition_templates=False, detect_single_transits=False)
    with PlotWrapper(target_id=target.target_id, save_fig=save_fig, no_show=no_show, ax=ax, figsize=figsize,
                     filename_postfix=filename_postfix, plot_output_location=plot_output_location) as (fig, axx):
        if temp_number is None:
            temp_number = target.best_temp

        time, statistics, variances, mask, _, __ = get_interleaving_stats(target=target, temp_number=temp_number)
        if units == 'snr':
            statistics = ut.safe_div(statistics, variances ** 0.5, fill_value=0.)

        axx.plot(time, statistics, color=color, **kwargs)
        plt.plot(time[~mask], statistics[~mask], 'o', **masked_points_kwargs, label='masked')

        if plot_trigger:
            if target.klc.summed_times is not None:
                plot_vlines(times=target.klc.summed_times)
                plot_points_at_times(times=[time], values=[statistics], point_times=target.klc.summed_times, ax=axx, **points_kwargs,
                                     label='transits')

        if mark_zero:
            axx.axhline(0, linewidth=Linewidth_zero_level, color='k', zorder=0)
        axx.set_xlim(np.nanmin(time) - edge_gap_time, np.nanmax(time) + edge_gap_time)
        axx.set_xlabel('time, days')
        ylabel = 'SES statistics'
        if units == 'snr':
            ylabel += ', [SNR]'
        axx.set_ylabel(ylabel)
        title = get_target_title(target, score=None, per=None, temp=temp_number)
        axx.set_title(title)
        axx.legend(loc='upper right')


def plot_whitened_fluxes(target, colors=Flux_quarters_colors, plot_trigger=True,
                         points_kwargs=(('color', 'r'), ('markersize', 3), ('label', 'transits')),
                         ax=None, save_fig=Save_fig, no_show=No_show, plot_output_location=locs.Triggers_plots_location,
                         figsize=Figsize_flux, filename_postfix=locs.Whitened_fluxes_plot_filename_postfix,
                         mark_zero=True, **kwargs):
    points_kwargs = dict(points_kwargs)
    prepare_target_for_plotting(target, initialize_templates=False,
                                condition_templates=False, detect_single_transits=False)
    with PlotWrapper(target_id=target.target_id, save_fig=save_fig, no_show=no_show, ax=ax, figsize=figsize,
                     filename_postfix=filename_postfix, plot_output_location=plot_output_location) as (fig, axx):
        zero_padding_ends = target.klc.zero_padding_ends
        if target.klc.is_zero_padded:
            times = [tt[zero_padding_ends[q][0]: len(tt) - zero_padding_ends[q][-1]] for q, tt in enumerate(target.klc.times)]
            fluxes = [vv[zero_padding_ends[q][0]: len(vv) - zero_padding_ends[q][-1]] for q, vv in enumerate(target.klc.whitened_fluxes_td)]
            exclusion_masks = [vv[zero_padding_ends[q][0]: len(vv) - zero_padding_ends[q][-1]]
                               for q, vv in enumerate(target.klc.big_hole_masks)]
        else:
            times = target.klc.times
            # fluxes = self.klc.fluxes
            if len(target.klc.whitened_fluxes_td[0]) != len(target.klc.times[0]):
                fluxes = [vv[zero_padding_ends[q][0]: len(vv) - zero_padding_ends[q][-1]]
                          for q, vv in enumerate(target.klc.whitened_fluxes_td)]
            else:
                fluxes = target.klc.whitened_fluxes_td
            exclusion_masks = target.klc.big_hole_masks

        plot_many(times=times, values=fluxes, zero_padding_ends=None, exclusion_masks=exclusion_masks,
                  ax=axx, label_prefix=None, colors=colors,
                  subtract_means=False, normalize=False, **kwargs)

        if plot_trigger:
            if target.klc.summed_times is not None:
                plot_vlines(times=target.klc.summed_times)
                plot_points_at_times(times=times, values=fluxes, point_times=target.klc.summed_times, ax=axx, **points_kwargs)

        if mark_zero:
            axx.axhline(0, linewidth=Linewidth_zero_level, color='k', zorder=0)
        axx.set_xlabel('time, days')
        axx.set_ylabel('Whitened fluxes')

        title = get_target_title(target, score=None, per=None, temp=None)
        axx.set_title(title)
        axx.legend(loc='upper right')


def plot_folded_stat_snr(target, temp_number=None, plot_twice=False, zoomed=False, zoom_wing=Zoom_wing_folded_plots,
                         ax=None, save_fig=Save_fig, no_show=No_show, plot_output_location=locs.Triggers_plots_location,
                         figsize=Figsize_flux,
                         filename_postfix=locs.Folded_statistics_plot_filename_postfix,
                         filename_postfix_zoomed=locs.Folded_statistics_zoomed_plot_filename_postfix,
                         template_linewidth=Linewidth):
    if zoomed:
        filename_postfix = filename_postfix_zoomed
    prepare_target_for_plotting(target, initialize_templates=False,
                                condition_templates=False, detect_single_transits=False)
    if temp_number is None:
        temp_number = target.best_temp

    time, statistics, variances, mask, corrections, statistics_gaussianized = get_interleaving_stats(target=target, temp_number=temp_number)
    statistics = ut.safe_div(statistics, variances ** 0.5, fill_value=0.)

    # FOLDING
    binwid = min(target.templates_durations_hours[temp_number] / 24 / 5, 0.1)
    bins_time, bins_flux, bins_counts, folds_times, folds_fluxes, bins_std = fold_fluxes(
        times=time, fluxes=statistics, period=target.best_per, first_time=target.best_ftt,
        masks=mask, binwid=binwid,  # self.templates_durations_hours[temp_number] / 24 * 10,
        dt=target.klc.dt_days, means=None, template=None, return_std=True,
        normalize_by_counts=True)  # np.array(self.klc.templates[self.klc.indices_of_templates[temp_number][0]]))

    # MAKING TEMPLATE
    n_time_templ = int(np.round(target.best_per / target.klc.dt_days))
    temp_plot, time_template = get_average_stat_template(whitened_templates_td=target.klc.whitened_templates_td,
                                                         temp_number=temp_number, n_time_templ=n_time_templ,
                                                         dt_days=target.klc.dt_days, indices_of_templates=target.klc.indices_of_templates)
    temp_plot = - temp_plot * target.best_stat ** 0.5 / (len(target.klc.summed_stats_raw)) ** 0.5

    # Plotting
    with PlotWrapper(target_id=target.target_id, save_fig=save_fig, no_show=no_show, ax=ax, figsize=figsize,
                     filename_postfix=filename_postfix, plot_output_location=plot_output_location) as (fig, axx):
        # If someone wants to repeat the period two times as it is often done in the literature:
        if plot_twice:
            time_shifts_plotting = [0, target.best_per]
            labels_binning = ['binning, bin ' + ut.fstr(binwid) + ' days', None]
            labels_template = ['average per-transit SNR', None]
        else:
            time_shifts_plotting = [0]
            labels_binning = ['binning, bin ' + ut.fstr(binwid) + ' days']
            labels_template = ['average per-transit SNR']

        # Actual plotting:
        for j in range(len(time_shifts_plotting)):
            # Plot the folded data:
            for i in range(len(folds_times)):
                axx.plot(folds_times[i] + time_shifts_plotting[j], folds_fluxes[i], 'o', markersize=0.6, color=(0.6, 0.6, 0.6))
            # Plot the binned folded data:
            axx.errorbar(bins_time[bins_counts > 0] + time_shifts_plotting[j], bins_flux[bins_counts > 0], yerr=bins_std[bins_counts > 0],
                         xerr=None, markersize=2.5, fmt='o', markerfacecolor='k', markeredgecolor='k',
                         ecolor='k', elinewidth=1.2, capsize=None, barsabove=False, lolims=False, uplims=False,
                         xlolims=False, xuplims=False, errorevery=1, capthick=None,
                         label=labels_binning[j])
            # Plot the expected template:
            axx.plot(time_template + time_shifts_plotting[j], temp_plot, color='r', linewidth=template_linewidth,
                     label=labels_template[j])

        # Setting the limits:
        set_axes_limits_fold(ax=axx, bins_flux=bins_flux, bins_time=bins_time, folds_fluxes=folds_fluxes, folds_times=folds_times,
                             best_per=target.best_per, plot_twice=plot_twice, zoomed=zoomed, zoom_wing=zoom_wing)
        axx.set_xlabel('Time from the mid transit, days')
        axx.set_ylabel('Statistic [SNR] fold')
        title = get_target_title(target, score=None, per=None, temp=temp_number)
        axx.set_title(title)
        axx.legend(loc='upper right')


def plot_folded_whitened_flux(target, plot_twice=False, temp_number=None, zoomed=False, zoom_wing=Zoom_wing_folded_plots,
                              ax=None, save_fig=Save_fig, no_show=No_show, plot_output_location=locs.Triggers_plots_location,
                              figsize=Figsize_flux,
                              filename_postfix=locs.Folded_whitened_fluxes_plot_filename_postfix,
                              filename_postfix_zoomed=locs.Folded_whitened_fluxes_zoomed_plot_filename_postfix,
                              template_linewidth=Linewidth):
    if zoomed:
        filename_postfix = filename_postfix_zoomed
    prepare_target_for_plotting(target, initialize_templates=False,
                                condition_templates=False, detect_single_transits=False)
    if temp_number is None:
        temp_number = target.best_temp
    # FOLDING
    binwid = min(target.templates_durations_hours[temp_number] / 24 / 5, 0.1)
    bins_time, bins_flux, bins_counts, folds_times, folds_fluxes, bins_std = fold_fluxes(
        times=target.klc.times, fluxes=target.klc.whitened_fluxes_td, period=target.best_per, first_time=target.best_ftt,
        masks=target.klc.big_hole_masks, binwid=binwid, # self.templates_durations_hours[temp_number] / 24 * 10,
        dt=target.klc.dt_days, means=None, template=None, return_std=True,
        normalize_by_counts=True)  # np.array(self.klc.templates[self.klc.indices_of_templates[temp_number][0]]))
    n_time_templ = int(np.round(target.best_per / target.klc.dt_days))
    temp_best, time_template = get_average_whitened_template(whitened_templates_td=target.klc.whitened_templates_td,
                                                             temp_number=temp_number, n_time_templ=n_time_templ,
                                                             dt_days=target.klc.dt_days,
                                                             indices_of_templates=target.klc.indices_of_templates)
    non_zero_mask = target.klc.summed_vars > 0
    ampl_est = np.nanmean(target.klc.summed_stats_raw[non_zero_mask] / target.klc.summed_vars[non_zero_mask])

    # Plotting
    with PlotWrapper(target_id=target.target_id, save_fig=save_fig, no_show=no_show, ax=ax, figsize=figsize,
                     filename_postfix=filename_postfix, plot_output_location=plot_output_location) as (fig, axx):

        # If someone wants to repeat the period two times as it is often done in the literature:
        if plot_twice:
            time_shifts_plotting = [0, target.best_per]
            labels_binning = ['binning, bin ' + ut.fstr(binwid) + ' days', None]
            labels_template = ['average whitened template', None]
        else:
            time_shifts_plotting = [0]
            labels_binning = ['binning, bin ' + ut.fstr(binwid) + ' days']
            labels_template = ['average whitened template']

        # Actual plotting:
        for j in range(len(time_shifts_plotting)):
            for i in range(len(folds_times)):
                axx.plot(folds_times[i] + time_shifts_plotting[j], folds_fluxes[i], 'o', markersize=1, color=(0.5, 0.5, 0.5))
            axx.errorbar(bins_time[bins_counts > 0] + time_shifts_plotting[j], bins_flux[bins_counts > 0], yerr=bins_std[bins_counts > 0],
                         xerr=None,
                         markersize=2.5, fmt='o', markerfacecolor='k', markeredgecolor='k',
                         ecolor='k', elinewidth=1.2, capsize=None, barsabove=False, lolims=False, uplims=False,
                         xlolims=False, xuplims=False, errorevery=1, capthick=None,
                         label=labels_binning[j])
            # ax.plot(bins_time, bins_flux, '-o', markersize=3, color='b', linewidth=1, label='binning, bin ' + ut.fstr(binwid) + ' days')
            axx.plot(time_template + time_shifts_plotting[j], temp_best * ampl_est, color='r', linewidth=template_linewidth,
                     label=labels_template[j])
        # Setting limits:
        set_axes_limits_fold(ax=axx, bins_flux=bins_flux, bins_time=bins_time, folds_fluxes=folds_fluxes, folds_times=folds_times,
                             best_per=target.best_per, plot_twice=plot_twice, zoomed=zoomed, zoom_wing=zoom_wing)
        axx.set_xlabel('Time from the mid transit, days')
        axx.set_ylabel('Whitened flux')
        title = get_target_title(target, score=None, per=None, temp=temp_number)
        axx.set_title(title)
        axx.legend(loc='upper right')


def plot_snrs_and_wh_fluxes_per_season(target, time_wing_days=Seasonal_plotting_wing_days,
                                       save_fig=Save_fig, no_show=No_show, figsize=Figsize_season_pieces,
                                       filename_postfix_snr=locs.Season_pieces_snr_plot_filename_postfix,
                                       filename_postfix_wh_flux=locs.Season_pieces_wh_flux_plot_filename_postfix,
                                       plot_output_location=locs.Triggers_plots_location, dpi=Plots_dpi,
                                       markersize_folds=0.7, markersize_bins=2, linewidth=Linewidth_very_thin):
    prepare_target_for_plotting(target, initialize_templates=False, condition_templates=False, detect_single_transits=False)
    transit_times = target.klc.summed_times
    seasons = get_seasons(tce=None, transit_times=transit_times)

    if np.array([len(target.klc.whitened_fluxes_td[q]) == target.klc.support
                 for q in range(len(target.klc.whitened_fluxes_td))]).all():
        unpad = True
        zero_padding_ends = target.klc.zero_padding_ends
    else:
        unpad = False
        zero_padding_ends = None

    times_near_transit_pieces_wh_flux, wh_fluxes_near_transit_pieces, masks_near_transit_pieces_flux, _ = (
        (get_near_transit_pieces_of_whitened_flux(
            transit_times=transit_times, times=target.klc.times, whitened_fluxes_td=target.klc.whitened_fluxes_td,
            masks=target.klc.big_hole_masks, time_wing_days=time_wing_days,
            unpad=unpad, zero_padding_ends=zero_padding_ends, dt_days=target.klc.dt_days)))

    statistic_full, stat_mask_full = get_shift_reduced_full_stat_snr(
        statistics_full=target.klc.statistics_full,
        stat_vars_full=target.klc.stat_vars_full,
        stat_masks_full=target.klc.stat_masks_full, temp_index=target.best_temp,
        indices_of_templates=target.klc.indices_of_templates, shift_reduction='average',
        renormalize=False, return_mask=True)
    times_near_transit_pieces_snr, near_transit_pieces_snr, mask_near_pieces_snr, _ = get_near_transit_pieces_from_full_time(
        transit_times=transit_times, time_full=target.klc.time_full, data_full=statistic_full,
        mask_full=stat_mask_full, time_wing_days=time_wing_days, peak_is_min=True, return_mask=True)

    # -----------------------------------------------------------------------------------------------
    # The actual plotting of the two plots
    for times_near_transit_pieces, data_near_transit_pieces, mask_near_transit_pieces, filename_postfix, ylabel, title in [
        [times_near_transit_pieces_wh_flux, wh_fluxes_near_transit_pieces, masks_near_transit_pieces_flux, filename_postfix_wh_flux,
         'Whitened flux', 'Whitened flux slices near transits by season'],
        [times_near_transit_pieces_snr, near_transit_pieces_snr, mask_near_pieces_snr, filename_postfix_snr,
         'Single-transit statistic in units of $SNR$', 'Single-transit statistic slices near transits by season']]:
        with SubplotsWrapper(nrows=2, ncols=2, sharex=True, sharey=True, gridspec_kw=Subplots_gridspec_kw,
                             save_fig=save_fig, no_show=no_show, figsize=figsize, target_id=target.target_id,
                             filename_postfix=filename_postfix, plot_output_location=plot_output_location, dpi=dpi,
                             grid=True, zero_lines=True, zero_lines_linewidth=Linewidth_zero_level) as (fig, axs):
            seasons_labeled = np.zeros(len(seasons)).astype(bool)
            for i in range(len(transit_times)):
                season = seasons[i]
                ax = axs[season // 2, season % 2]
                label = 'S' + str(season) if not seasons_labeled[season] else None
                col = (0.5, 0.5, 0.5)
                ax.plot((times_near_transit_pieces[i] - transit_times[i])[mask_near_transit_pieces[i]],
                        data_near_transit_pieces[i][mask_near_transit_pieces[i]], linestyle='',
                        marker='o', markersize=markersize_folds, color=col, zorder=10,
                        label=label)

                bins_time, bins_flux, bins_counts, folds_times, folds_fluxes, bins_std = fold_fluxes(
                    times=copy.deepcopy(times_near_transit_pieces_wh_flux),
                    fluxes=copy.deepcopy(wh_fluxes_near_transit_pieces),
                    period=target.best_per,
                    first_time=target.best_ftt,
                    masks=copy.deepcopy(masks_near_transit_pieces_flux),
                    binwid=target.klc.dt_days, dt=target.klc.dt_days, means=None, template=None, return_std=True,
                    normalize_by_counts=True)
                col = 'k'
                ax.errorbar(bins_time[bins_counts > 0], bins_flux[bins_counts > 0], yerr=bins_std[bins_counts > 0],
                            xerr=None, markersize=markersize_bins, fmt='o', markerfacecolor=col, markeredgecolor=col,
                            ecolor=col, elinewidth=linewidth, capsize=None, barsabove=False, lolims=False, uplims=False,
                            xlolims=False, xuplims=False, errorevery=1, capthick=None,
                            label=None)
                seasons_labeled[season] = True

            for season in seasons:
                ax = axs[season // 2, season % 2]
                ax.legend(loc='upper left')

            axs[-1, 0].set_xlabel('Time from transit, days')
            axs[-1, 0].set_ylabel(ylabel)
            fig.suptitle(get_target_title(target) + '\n' + title, y=1)


def plot_significance_histograms(target, fixed_limits=False, limits=None, plot_extrapolation=True,
                                 fontsize_legend=Fontsize_legend,
                                 fontsize_title=Fontsize_title,
                                 fontsize_label=Fontsize_label,
                                 linewidth=Linewidth_p_planet_hist,
                                 use_integral=True, use_clean=True,
                                 ax=None, save_fig=Save_fig, no_show=No_show, plot_output_location=locs.Triggers_plots_location,
                                 figsize=Figsize_p_planet_hist,
                                 filename_postfix=locs.Background_injection_histograms_plot_filename_postfix,
                                 minor_grid_bin=10, max_n_xlabels=20):
    # PLOT BACKGROUND AND INJECTION-RECOVERY
    if target.prob_density_planet_not_noise is None:
        target.analyze_injections_and_background()

    # --------------------------------------------
    # SETTING PARAMS
    if use_integral:
        distr_score_h0, distr_score_inj_all, distr_score_inj_success, bins, best_score = \
            target.distr_int_score_h0, target.distr_int_score_inj_all, target.distr_int_score_inj_success, \
            target.bins_int_score_histogram, target.best_int_score
    else:
        distr_score_h0, distr_score_inj_all, distr_score_inj_success, bins, best_score = \
            target.distr_stat_h0, target.distr_stat_inj_all, target.distr_stat_inj_success, target.bins_stat_histogram, \
            target.best_stat

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
    with PlotWrapper(target_id=target.target_id, save_fig=save_fig, no_show=no_show, ax=ax, figsize=figsize,
                     filename_postfix=filename_postfix, plot_output_location=plot_output_location) as (fig, axx):

        # --------------------------------------------
        # PLOTTING TRIGGER
        axx.axvline(best_score, color='k', linewidth=linewidth,
                    label=' ' * 13 + r'$SNR^2$=' + str(np.round(np.abs(best_score), 1))
                          + ',\n' + 'trigger: period=' + str(int(np.round(target.best_per, 0))) + ' d'
                          + ',\n' + ' ' * 13 + 'duration='
                          + str(int(np.round(target.templates_durations_hours[target.best_temp], 0))) + ' h')

        # --------------------------------------------
        # PLOTTING HISTOGRAMS
        if distr_score_inj_all is not None:
            axx.stairs(distr_score_inj_all, bins, label='all injections in relative period range', linewidth=linewidth, color=def_cols[0],
                       linestyle='--')  # after density correction factor, get per one star
            axx.stairs(distr_score_inj_success, bins, label='success injections',
                       linewidth=linewidth, color=def_cols[0], linestyle='-')  # after density correction factor, get per one star
        axx.stairs(distr_score_h0, bins, label='background, scrambling', linewidth=linewidth, color=def_cols[1])

        # --------------------------------------------
        # PLOTTING CLEAN
        if use_clean:
            if use_integral:
                distr_score_h0_clean = target.distr_int_score_h0_clean
            else:
                distr_score_h0_clean = target.distr_stat_h0_clean
            axx.stairs(distr_score_h0_clean, bins, label='background, scrambling no trigger',
                       linewidth=linewidth, color=def_cols[1], linestyle='--')

        # --------------------------------------------
        # PLOTTING EXTRAPOLATION
        if plot_extrapolation and use_integral:
            # MAKE EXTRAPOLATION
            if not hasattr(target, 'extrapolation_coefs_int_score_h0_hist') or target.extrapolation_coefs_int_score_h0_hist is None:
                from auxiliary_functions import get_histogram_extrapolation_coefs_in_log
                target.extrapolation_coefs_int_score_h0_hist = get_histogram_extrapolation_coefs_in_log(
                        target.distr_int_score_h0, target.bins_int_score_histogram)
            coefs = target.extrapolation_coefs_int_score_h0_hist
            # determine intersection with horizontal axis in log scale:
            bin_extra_max = (np.log(np.nanmin(target.distr_int_score_h0[target.distr_int_score_h0 != 0])) - coefs[0]) / coefs[1]
            bin_extra_max = np.nanmax([target.best_int_score + 1, bin_extra_max + 1])
            bin_extra_min = bins[np.nanargmax(target.distr_int_score_h0) + 1]
            bin_min_distr = bins[:-1][((~np.isnan(target.distr_int_score_h0)) & (target.distr_int_score_h0 > 0))][0]
            # bin_extra_min = int((bin_extra_min + bin_extra_max) / 2)
            bin_extra_min = bin_extra_min + int(np.abs(bin_extra_min - bin_min_distr) / 2)
            bins_extra = np.linspace(bin_extra_min, bin_extra_max, 10)
            label = 'extrapolation'
            if target.false_alarm_prob_trigger is not None:
                label += ', FAP=' + f"{target.false_alarm_prob_trigger:.2e}"
            axx.plot(bins_extra, np.exp(bins_extra * coefs[1] + coefs[0]), '--', label=label, color=def_cols[1])

        # --------------------------------------------
        # FORMATTING
        if fontsize_legend is not None:
            axx.legend(loc='upper right', fontsize=fontsize_legend)
        else:
            axx.legend(loc='upper right')

        title = 'KID' + target.target_id
        if target.injections_period_range is not None:
            title = (title + ', periods ['
                     + str(np.round(target.injections_period_range[0], 1)) + '-'
                     + str(np.round(target.injections_period_range[1], 1)) + ']'
                     + '] days')
        if fontsize_title is not None:
            axx.set_title(title, fontsize=fontsize_title)
        else:
            axx.set_title(title)

        xlabel = r'Recovered MES [$SNR^2$]'
        ylabel = 'Counts per effective one star \n per period bin'
        if fontsize_label is not None:
            axx.set_xlabel(xlabel, fontsize=fontsize_label)
            axx.set_ylabel(ylabel, fontsize=fontsize_label)
        else:
            axx.set_xlabel(xlabel)
            axx.set_ylabel(ylabel)
        axx.set_yscale('log')
        axx.set_xlim(min_plot_mes2, max_plot_mes2)
        axx.xaxis.set_minor_locator(ticker.MultipleLocator(minor_grid_bin))
        major_bin_factor = int(np.ceil(((max_plot_mes2 - min_plot_mes2) / (max_n_xlabels - 1)) / minor_grid_bin))
        axx.xaxis.set_major_locator(ticker.MultipleLocator(minor_grid_bin * major_bin_factor))
        # axx.grid(which='both', zorder=0, color=0.7 * np.ones(3), linewidth=0.7)

        if fontsize_legend is not None:
            axx.legend(loc='upper right', fontsize=fontsize_legend)
        else:
            axx.legend(loc='upper right')


def plot_periodogram_chunks(tce, best_short_periodic_scores_filename_postfix=locs.Best_short_periodic_scores_filename_postfix,
                            best_periodic_scores_filename_postfix=locs.Best_periodic_scores_filename_postfix,
                            save_fig=Save_fig, no_show=No_show, plot_output_location=locs.Triggers_plots_location,
                            figsize=Figsize_periodogram, filename_postfix=locs.Periodogram_chunks_plot_filename_postfix,
                            ):
    from search_pipeline_parameters import (Best_per_results_index_best_integral, Best_per_results_index_per,
                                            Best_per_results_index_stats, Best_per_results_index_stats_gauss,
                                            Best_per_results_index_ftt, Best_per_results_index_template)
    res_array = tce.get_periodicity_results(short_periodicity=False,
                                            best_periodic_scores_filename_postfix=best_periodic_scores_filename_postfix,
                                            return_lists=True, set_attributes=False)
    int_scores = res_array[Best_per_results_index_best_integral]
    stats = res_array[Best_per_results_index_stats]
    pers = res_array[Best_per_results_index_per]
    gauss_stats = res_array[Best_per_results_index_stats_gauss]
    ftts = res_array[Best_per_results_index_ftt]
    temps = res_array[Best_per_results_index_template]

    res_array = tce.get_periodicity_results(short_periodicity=True,
                                            best_short_periodic_scores_filename_postfix=best_short_periodic_scores_filename_postfix,
                                            return_lists=True, set_attributes=False)
    short_int_scores = res_array[Best_per_results_index_best_integral]
    short_stats = res_array[Best_per_results_index_stats]
    short_pers = res_array[Best_per_results_index_per]
    short_gauss_stats = res_array[Best_per_results_index_stats_gauss]
    short_ftts = res_array[Best_per_results_index_ftt]
    short_temps = res_array[Best_per_results_index_template]

    with SubplotsWrapper(nrows=3, ncols=1, sharex=True, sharey=False, gridspec_kw=Subplots_gridspec_kw,
                         save_fig=save_fig, no_show=no_show, figsize=figsize, target_id=None,
                         filename_postfix=filename_postfix, plot_output_location=plot_output_location, dpi=Plots_dpi,
                         grid=True, zero_lines=False) as (fig, axs):
        lw = 1.5
        lw_thin = 1
        lw_vline = 0.8

        ax = axs[0]
        ax.plot(pers, int_scores, color=def_cols[0], linewidth=lw, zorder=4, label='best scores')
        ax.plot(short_pers, short_int_scores, color=def_cols[0], linewidth=lw, zorder=4)
        ax.plot(pers, stats, color=def_cols[1], linewidth=lw_thin, label='best statistics')
        ax.plot(short_pers, short_stats, color=def_cols[1], linewidth=lw_thin)
        ax.plot(pers, gauss_stats, color=def_cols[2], linewidth=lw_thin, label='best gaussianized statistics')
        ax.plot(short_pers, short_gauss_stats, color=def_cols[2], linewidth=1)
        ax.axvline(tce.best_per, zorder=0, color='r', linewidth=lw_vline)
        ax.set_xlabel('Period, days')
        ax.set_ylabel('Statistic, $SNR^2$')
        ax.legend(loc='upper left')

        ax = axs[1]
        ax.plot(pers, Durations[temps.astype(int)], color=def_cols[0], linewidth=lw, zorder=2, label='best templates')
        ax.plot(short_pers, Durations[short_temps.astype(int)], color=def_cols[0], linewidth=2, zorder=100)
        ax.axvline(tce.best_per, zorder=0, color='r', linewidth=lw_vline, label='trigger period')
        ax.set_xlabel('Period, days')
        ax.set_ylabel('Template Duration, days')
        ax.legend(loc='upper left')

        ax = axs[2]
        ax.plot(pers, (ftts - np.nanmin(short_ftts)) / pers, color=def_cols[0], linewidth=lw, zorder=2, label='best first transit phases')
        ax.plot(short_pers, (short_ftts - np.nanmin(short_ftts)) / short_pers, color=def_cols[0], linewidth=2, zorder=100)
        ax.axvline(tce.best_per, zorder=0, color='r', linewidth=lw_vline)
        ax.set_xlabel('Period, days')
        ax.set_ylabel('Template Duration, days')
        ax.set_ylim(-0.1, 1.2)
        ax.legend(loc='upper left')

        fig.suptitle('Periodogram of maximized scores\n' + get_target_title(tce), y=1)


def plot_folding_residuals_histograms(tce=None, target_id=None, output_location=None,
                                      residuals_flux_binned=None, mf_scores_resid=None, temp_number=None,
                                      p_val_resid=None, p_val_mf=None, fn_fold_veto=None,
                                      ax=None, save_fig=Save_fig, no_show=No_show, plot_output_location=locs.Triggers_plots_location,
                                      figsize=Figsize_residuals_hist,
                                      filename_postfix=locs.Residuals_fold_veto_histogram_plot_filename_postfix,
                                      fontsize_legend=Fontsize_legend, fontsize_title=Fontsize_title, fontsize_label=Fontsize_label,
                                      linewidth=Linewidth):
    if residuals_flux_binned is None or mf_scores_resid is None:
        if tce is not None:
            target_id = tce.target_id
            temp_number = tce.best_temp_folded
            p_val_resid = tce.p_value_fold_residuals
            p_val_mf = tce.p_value_mf_fold_residuals
            fn_fold_veto = tce.fn_fold_veto
            if fn_fold_veto is None:
                from tce_parameters import Fn_fold_veto
                fn_fold_veto = Fn_fold_veto
            if output_location is None:
                output_location = tce.output_location
        elif target_id is None:
            raise ValueError('Either scores, or tce, or target_id must be provided')
        elif output_location is None:
            raise ValueError('Either scores, or tce, or output_location must be provided')
        file_path = get_saving_file_path(target_id=target_id, output_location=output_location,
                                         filename_postfix=locs.Fold_residuals_flux_binned_filename_postfix,
                                         insert_after_postfix=None, file_format='npy')
        residuals_flux_binned = np.load(file_path, allow_pickle=True)
        file_path = get_saving_file_path(target_id=target_id, output_location=output_location,
                                         filename_postfix=locs.Mf_scores_residuals_filename_postfix,
                                         insert_after_postfix=None, file_format='npy')
        mf_scores_resid = np.load(file_path, allow_pickle=True)

    from scipy.stats import chi2
    with (PlotWrapper(target_id=target_id, save_fig=save_fig, no_show=no_show, ax=ax, figsize=figsize,
                      filename_postfix=filename_postfix, plot_output_location=plot_output_location,
                      which_grid='major') as (fig, axx)):
        residuals_flux_squared = residuals_flux_binned ** 2
        lw = linewidth
        mf_scores_resid_flatten_2 = (mf_scores_resid.flatten()) ** 2
        limit_line_mf = chi2(df=1, scale=1).isf(1. - (1 - fn_fold_veto) ** (1 / len(mf_scores_resid_flatten_2)))
        limit_line_resid = chi2(df=1, scale=1).isf(1. - (1 - fn_fold_veto) ** (1 / len(residuals_flux_binned)))
        bins = np.linspace(0, max(np.max(mf_scores_resid_flatten_2), np.max(residuals_flux_squared)), 100)
        axx.hist(mf_scores_resid_flatten_2, bins=bins, histtype='step', density=True,
                 label='Matched filtering \nof residual flux', linewidth=lw, color=def_cols[0])
        axx.hist(residuals_flux_squared, bins=bins, histtype='step', density=True, label='Residual flux',
                 linewidth=lw, color=def_cols[1])
        xx = np.linspace(0, min(np.max(mf_scores_resid_flatten_2), np.max(residuals_flux_squared),
                                limit_line_mf, limit_line_resid), 100)
        axx.plot(xx, chi2(df=1, scale=1).pdf(xx), 'r--', linewidth=lw, label=r'$\chi^2$')
        axx.axvline(limit_line_mf, color=def_cols[0],
                    linestyle='-', linewidth=lw, label='Threshold')
        axx.axvline(limit_line_resid, color=def_cols[1],
                    linestyle='-', linewidth=lw, label='Threshold')
        axx.set_yscale('log')
        axx.set_xlabel('Residual squared', fontsize=fontsize_label)
        axx.set_ylabel('Probability density', fontsize=fontsize_label)
        title = 'Residuals of the folded flux distribution \n' + target_id
        if temp_number is not None:
            title += ', temp ' + str(temp_number)
        if p_val_mf is not None:
            title += ', p_vals: mf:' + str(np.round(p_val_mf, 3))
        if p_val_resid is not None:
            title += ', resid:' + str(np.round(p_val_resid, 3))

        fig.suptitle(title, fontsize=fontsize_title)
        plt.legend(fontsize=fontsize_legend, loc=(1.01, 0.5))


def plot_injections_corners(target, save_fig=Save_fig, no_show=No_show, plot_output_location=locs.Triggers_plots_location,
                            dpi=Plots_dpi,
                            figsize=Figsize_corner, filename_postfix=locs.Orig_fluxes_plot_filename_postfix):
    import corner
    if not hasattr(target, 'retrieved_int_score_success') or not hasattr(target, 'dur_inj_success'):
        target.retrieve_injections()

    pars_dict = dict([
        ['per_inj',   [target.per_inj_success, target.per_inj_bad,  r'$p_{i}$', 'Injected period, days']],
        ['a_inj',     [target.a_inj_success, target.a_inj_bad,      r'$a_{i}$', 'Injected semi-major axis, stellar radii']],
        ['inc_inj',   [target.inc_inj_success, target.inc_inj_bad,  r'$\iota_{i}$', 'Injected inclination, deg']],
        ['ecc_inj',   [target.ecc_inj_success, target.ecc_inj_bad,  r'$e_{i}$', 'Injected eccentricity']],
        ['w_inj',     [target.w_inj_success, target.w_inj_bad,      r'$w_{i}$', 'Injected periastron argument, deg']],
        ['rp_inj',    [target.rp_inj_success, target.rp_inj_bad,    r'$(R_p/R_s)_{i}$', 'Injected radius of planet, stellar radii']],
        ['dur_inj',   [target.dur_inj_success, target.dur_inj_bad,  r'dur$_{i}$', 'Injected transit duration, days']],
        ['ftt_inj',   [target.first_transit_time_inj_success, target.first_transit_time_inj_bad,
                       r'$t0_{i}$', 'Injected first transit time, days']],
        ['per_retr',  [target.retrieved_period_success, target.retrieved_period_bad,
                       r'$p_{r}$', 'Retrieved period, days']],
        ['ftt_retr',  [target.retrieved_first_transit_time_success, target.retrieved_first_transit_time_bad,
                       r'$t0_{r}$', 'Retrieved first transit time, days']],
        ['int_score_retr',  [target.retrieved_int_score_success, target.retrieved_int_score_bad,
                             r'$\eta_{r}$', 'Detection score, $SNR^2$']],
        ['stat_retr', [target.retrieved_stat_success, target.retrieved_stat_bad,
                       r'$\rho^2_{r}$', 'Statistic, $SNR^2$']],
        ['dur_retr',  [[Durations[tt] for tt in target.retrieved_template_success],
                       [Durations[tt] for tt in target.retrieved_template_bad], r'dur$_{r}$', 'Retrieved duration, days']],
        ['gauss_stat_retr', [target.retrieved_gauss_stat_success, target.retrieved_gauss_stat_bad,
                             r'$(\rho^2_g)_{r}$', 'Gauss. statistic, $SNR^2$']]])

    levels = (0.3, 0.6, 0.8)

    for keys_corner, fig_name in [
            [['inc_inj', 'ecc_inj', 'w_inj', 'rp_inj', 'dur_inj', 'int_score_retr'],
             locs.Corner_injections_orbital_plot_filename_postfix],
            [['rp_inj', 'dur_inj', 'dur_retr', 'ftt_inj', 'ftt_retr', 'int_score_retr'],
             locs.Corner_injections_times_plot_filename_postfix],
            [['rp_inj', 'dur_inj', 'dur_retr', 'int_score_retr', 'stat_retr', 'gauss_stat_retr'],
             locs.Corner_injections_scores_plot_filename_postfix]]:
        res_success = []
        res_bad = []
        labels = []
        for key in keys_corner:
            res_success.append(pars_dict[key][0])
            res_bad.append(pars_dict[key][1])
            labels.append(pars_dict[key][2])
        res_success = np.array(res_success).T
        res_bad = np.array(res_bad).T
        labels = np.array(labels)

        fig, ax = plt.subplots(len(keys_corner), len(keys_corner), figsize=figsize, gridspec_kw={'wspace': 0, 'hspace': 0})
        fig.subplots_adjust(hspace=0., wspace=0.)

        data_kwargs_b = dict(alpha=0.9, ms=2, color=def_cols[1])
        data_kwargs_s = dict(alpha=0.9, ms=2, color=def_cols[0])
        corner.corner(res_bad, color=def_cols[1], fig=fig, fill_contours=True, labels=labels,
                      data_kwargs=data_kwargs_b, levels=levels, quiet=True)
        corner.corner(res_success, color=def_cols[0], fig=fig, fill_contours=True, labels=labels,
                      data_kwargs=data_kwargs_s, levels=levels, quiet=True)
        fig.suptitle('Injection-recovery parameters \n' + get_target_title(target, score=None, per=None, temp=None))
        # axes = np.array(fig.axes).reshape((len(keys_corner), len(keys_corner)))

        for ax in fig.axes:
            ax.grid(zorder=0, color='k', alpha=0.3, linewidth=0.5)
        ax = fig.axes[0]
        ax.plot([], **data_kwargs_s, label='Success')
        ax.plot([], **data_kwargs_b, label='Bad')
        ax.legend(loc='best')

        if save_fig:
            tid_folder = get_correct_tid_folder(target_id=target.target_id, location=plot_output_location, allow_creation=False,
                                                with_prefix_folder=False)
            file_saving_path = get_saving_file_path(output_location=tid_folder,
                                                    target_id=target.target_id,
                                                    filename_postfix=filename_postfix,
                                                    file_format='png')
            fig.savefig(file_saving_path, dpi=dpi, bbox_inches='tight')

        if no_show:
            plt.clf()
            plt.close(fig)
        else:
            plt.show()


def plot_injections_radii_durations_score(target, save_fig=Save_fig, no_show=No_show, plot_output_location=locs.Triggers_plots_location,
                                          dpi=Plots_dpi, figsize=Figsize_inj_radii_dur_scatter,
                                          filename_postfix=locs.Injections_radii_durations_score_plot_filename_postfix):
    if not hasattr(target, 'retrieved_int_score_success'):
        target.retrieve_injections()

    color_all = (0.6, 0.6, 0.6)
    color_low = def_cols[0]
    color_high = def_cols[1]
    color_bad = (0.8, 0.3, 0.3)
    part_points_highlight = 0.1
    ms_small = 1
    ms_big = 3

    fig, axs = plt.subplots(1, 2, figsize=figsize)
    # ---------------------------------------------------------------
    # Plot int scores of durations
    ax = axs[0]
    durs = np.array(target.dur_inj_bad)
    scors = np.array(target.retrieved_int_score_bad)
    ax.scatter(durs, scors, s=ms_small, color=color_bad, label='bad')
    durs = np.array(target.dur_inj_success)
    scors = np.array(target.retrieved_int_score_success)
    ax.scatter(durs, scors, s=ms_small, color=color_all, label='all success')

    rps = np.array(target.rp_inj_success) * target.stellar_radius * pp.R_sun_R_e
    rps_sorted = np.sort(rps)
    part = np.arange(len(rps_sorted)) / len(rps_sorted)
    rp_low = np.round(rps_sorted[np.searchsorted(part, part_points_highlight)], 1)
    rp_high = np.round(rps_sorted[np.searchsorted(part, 1 - part_points_highlight)], 1)

    inds_selection = (rps <= rp_low)
    ax.scatter(durs[inds_selection], scors[inds_selection], s=ms_big, color=color_low, label='$r_{p}/r_{e}$<' + str(rp_low))
    inds_selection = (rps >= rp_high)
    ax.scatter(durs[inds_selection], scors[inds_selection], s=ms_big, color=color_high, label='$r_{p}/r_{e}$>' + str(rp_high))

    ax.axhline(50, color='k')
    ax.axhline(70, color='k')
    ax.set_xlabel('dur$_{inj}$, days')
    ax.set_ylabel(r'$\eta$, SNR$^2$')
    ax.set_xticks(np.arange(0, 6, 0.5))
    ax.grid()
    ax.legend()
    ax.set_title(get_target_title(target, score=None, per=None, temp=None))

    # ---------------------------------------------------------------
    # Plot int scores of radii
    ax = axs[1]

    rps = np.array(target.rp_inj_bad) * target.stellar_radius * pp.R_sun_R_e
    scors = np.array(target.retrieved_int_score_bad)
    ax.scatter(rps, scors, s=ms_small, color=color_bad, label='bad')
    rps = np.array(target.rp_inj_success) * target.stellar_radius * pp.R_sun_R_e
    scors = np.array(target.retrieved_int_score_success)
    ax.scatter(rps, scors, s=ms_small, color=color_all, label='all success')

    durs = np.array(target.dur_inj_success)
    durs_sorted = np.sort(durs)
    part = np.arange(len(durs_sorted)) / len(durs_sorted)
    dur_low = np.ceil(durs_sorted[np.searchsorted(part, part_points_highlight)]*1000)/1000  # ceil with decimals
    dur_high = np.round(durs_sorted[np.searchsorted(part, 1 - part_points_highlight)], 3)
    inds_selection = (durs <= dur_low)
    ax.scatter(rps[inds_selection], scors[inds_selection], s=ms_big, color=color_low, label='dur < ' + str(dur_low) + ' d')
    inds_selection = (durs >= dur_high)
    ax.scatter(rps[inds_selection], scors[inds_selection], s=ms_big, color=color_high, label='dur > ' + str(dur_high) + ' d')

    ax.axhline(50, color='k')
    ax.axhline(70, color='k')
    ax.set_xlabel('$r_{p}/r_{e}$')
    ax.set_ylabel(r'$\eta$, SNR$^2$')
    ax.set_xticks(np.arange(1, 21, 1))
    ax.grid(zorder=0, linewidth=0.5)
    ax.legend()
    ax.set_title(get_target_title(target, score=None, per=None, temp=None))

    if save_fig:
        tid_folder = get_correct_tid_folder(target_id=target.target_id, location=plot_output_location, allow_creation=False,
                                            with_prefix_folder=False)
        file_saving_path = get_saving_file_path(output_location=tid_folder,
                                                target_id=target.target_id,
                                                filename_postfix=filename_postfix,
                                                file_format='png')
        fig.savefig(file_saving_path, dpi=dpi, bbox_inches='tight')

    if no_show:
        plt.clf()

    plt.show()


def plot_bg_star_pixels_ppm(tce, save_fig=Save_fig, no_show=No_show, plot_output_location=locs.Triggers_plots_location,
                            dpi=Plots_dpi, figsize=Figsize_star_bg_pixels_ppm,
                            filename_postfix=locs.Star_bg_pixels_ppm_plot_filename_postfix,
                            linewidth=Linewidth_coarse, linewidth_thin=Linewidth_thin,
                            linewidth_zero_level=Linewidth_zero_level,
                            markersize=Markersize):
    seasons = get_seasons(tce=tce)
    with SubplotsWrapper(nrows=2, ncols=1, sharex=False, sharey=False, gridspec_kw={'top': 0.88, 'bottom': 0.05, 'hspace': 0.25},
                         save_fig=save_fig, no_show=no_show, figsize=figsize, target_id=tce.target_id,
                         filename_postfix=filename_postfix, plot_output_location=plot_output_location, dpi=dpi,
                         grid=True, zero_lines=False) as (fig, axs):

        ax = axs[0]
        lw = linewidth

        ax.errorbar(tce.summed_transit_times, tce.star_pixels_ppms, yerr=tce.star_pixels_stds_of_ppms,
                    marker='o', color=def_cols[0], label='star', markersize=markersize,
                    capsize=3, linewidth=lw, elinewidth=lw, capthick=lw)
        ax.errorbar(tce.summed_transit_times, tce.bg_pixels_ppms, yerr=tce.bg_pixels_stds_of_ppms,
                    marker='o', color=def_cols[1], label='bg', markersize=markersize,
                    capsize=3, linewidth=lw, elinewidth=lw, capthick=lw)
        ax.set_xlabel('Transit time, days')
        ax.set_ylabel('Depth of transits, ppm')
        ax.axhline(0, linewidth=linewidth_zero_level, color='k')
        ax.legend()

        ax = axs[1]
        seasons = seasons.astype(float)
        for i in range(4):
            seasons[seasons == i] += np.linspace(-0.1, 0.1, len(seasons[seasons == i]))
        ax.errorbar(tce.star_pixels_ppms, seasons, xerr=tce.star_pixels_stds_of_ppms,
                    marker='o', linestyle='', markersize=markersize, label='star', linewidth=linewidth_thin,
                    capsize=3, elinewidth=linewidth_thin, capthick=linewidth_thin)
        for i in range(4):
            seasons[seasons == i] -= np.linspace(-0.05, 0.05, len(seasons[seasons == i]))
        ax.errorbar(tce.bg_pixels_ppms, seasons, xerr=tce.bg_pixels_stds_of_ppms,
                    marker='o', linestyle='', markersize=markersize, label='bg', linewidth=linewidth_thin,
                    capsize=3, elinewidth=linewidth_thin, capthick=linewidth_thin)
        ax.set_yticks([0, 1, 2, 3])
        ax.set_xlabel('Depth of transits, ppm')
        ax.set_ylabel('Season')
        ax.set_ylim(-0.5, 3.5)
        ax.legend()

        fig.suptitle('Transit depths of star and background pixels comparison \n' + get_target_title(tce)
                     + '\np-values of the veto: $p_{total}=$' + str(np.round(tce.p_value_mean_bg_pixels, 3))
                     + ', $p_{worst}=$' + str(np.round(tce.p_value_worst_bg_pixels, 3)), y=1)


def plot_centroids_stat_periodic(tce=None, target_id=None,
                                 ftts_total=None, stats_2_max_periodic_total=None, ftts_coords=None,
                                 stats_2_max_periodic_coords=None, ftt_center=None, max_centroid_snr_2=None,
                                 ftt_window_plotting=Centroid_ftt_wing_days_plotting,
                                 save_fig=Save_fig, no_show=No_show, plot_output_location=locs.Triggers_plots_location,
                                 dpi=Plots_dpi,  figsize=Figsize_centroid_periodic,
                                 filename_postfix=locs.Centroid_periodic_plot_filename_postfix):
    if ftts_total is None or stats_2_max_periodic_total is None or stats_2_max_periodic_coords is None or ftt_center is None:
        if tce is None:
            raise ValueError('Either parameters or tce must be provided')
        else:
            (ftts_total, stats_2_max_periodic_total, ftts_coords, stats_2_max_periodic_coords,
             ftt_center) = tce.load_centroid_periodic_results()
            target_id = tce.target_id
            ftt_center = tce.best_ftt
            max_centroid_snr_2 = tce.centroid_snr_2

    # --------------------------------------------------------------------------------------------------------
    # Plotting
    with PlotWrapper(target_id=target_id, save_fig=save_fig, no_show=no_show, ax=None, figsize=figsize, dpi=dpi,
                     filename_postfix=filename_postfix, plot_output_location=plot_output_location,
                     grid=True) as (fig, ax):
        ax.plot(ftts_coords[0], stats_2_max_periodic_coords[0], label='Centroid ra')
        ax.plot(ftts_coords[1], stats_2_max_periodic_coords[1], label='Centroid dec')
        ax.plot(ftts_total, stats_2_max_periodic_total, label='Centroid total deviation')
        ax.axvline(ftt_center, color='r', label='$t0$ of the trigger')
        ax.axhline(max_centroid_snr_2, linestyle='--', color='k', label='Detected centroid deviation')

        ax.set_xlim(ftt_center - ftt_window_plotting, ftt_center + ftt_window_plotting)
        ax.legend()
        ax.set_xlabel('First transit time, days')
        ax.set_ylabel('Maximal statistic, $SNR^2$')
        ax.set_title('Centroid statistic folded with period close to trigger', y=1)


def plot_centroid_coordinates_snr_pieces(target_id, transit_times, centr_coords_snr_near_transit_pieces,
                                         centr_coords_times_near_transit_pieces, seasons,
                                         save_fig=Save_fig, no_show=No_show, plot_output_location=locs.Triggers_plots_location,
                                         dpi=Plots_dpi, figsize=Figsize_centroid_coordinate_pieces,
                                         filename_postfix=locs.Centroid_coordinate_pieces_plot_filename_postfix
                                         ):
    with SubplotsWrapper(nrows=2, ncols=1, sharex=True, sharey=True, gridspec_kw=Subplots_gridspec_kw,
                         save_fig=save_fig, no_show=no_show, figsize=figsize, target_id=target_id,
                         filename_postfix=filename_postfix, plot_output_location=plot_output_location, dpi=dpi,
                         grid=True, color_grid=Color_grid, linewidth_grid=Linewidth_grid,
                         zero_lines=True, zero_lines_linewidth=Linewidth_zero_level) as (fig, axs):
        for i_coor in (0, 1):
            ax = axs[i_coor]
            seasons_labeled = np.zeros(4).astype(bool)
            for i in range(len(transit_times)):
                season = seasons[i]
                ax.plot(centr_coords_times_near_transit_pieces[i_coor][i] - transit_times[i],
                        centr_coords_snr_near_transit_pieces[i_coor][i], '-',
                        markersize=2, color=def_cols[season], zorder=10,
                        label='Season ' + str(season) if not seasons_labeled[season] else None)
                seasons_labeled[season] = True
            ax.set_title('ra' if i_coor == 1 else 'dec', y=0.85, x=0.05, bbox={
                         'facecolor': 'None', 'edgecolor': 'black', 'boxstyle': 'square,pad=0.1'})
        axs[-1].legend(loc='best')
        axs[-1].set_xlabel('Time from the transit, days')
        axs[-1].set_ylabel('SNR of centroid deviation')
        fig.suptitle('SNR of centroid deviations around the transits', y=1)


def plot_centroid_snr2_seasons(target_id, transit_times, time_total, centr_snr2_near_transit_pieces,
                               seasons, save_fig=Save_fig, no_show=No_show, plot_output_location=locs.Triggers_plots_location,
                               dpi=Plots_dpi, figsize=Figsize_centroid_coordinate_pieces,
                               filename_postfix=locs.Centroid_snr2_seasons_plot_filename_postfix
                               ):
    with SubplotsWrapper(nrows=2, ncols=2, sharex=True, sharey=True, gridspec_kw=Subplots_gridspec_kw,
                         save_fig=save_fig, no_show=no_show, figsize=figsize, target_id=target_id,
                         filename_postfix=filename_postfix, plot_output_location=plot_output_location, dpi=dpi,
                         grid=True, color_grid=Color_grid, linewidth_grid=Linewidth_grid,
                         zero_lines=True, zero_lines_linewidth=Linewidth_zero_level) as (fig, axs):
        seasons_labeled = np.zeros(len(seasons)).astype(bool)
        for i in range(len(transit_times)):
            season = seasons[i]
            ax = axs[season // 2, season % 2]
            ax.plot(time_total, centr_snr2_near_transit_pieces[i], '-',
                    markersize=2, color=def_cols[season], zorder=10, label='Season ' + str(season) if not seasons_labeled[season] else None)
            seasons_labeled[season] = True

        for season in seasons:
            ax = axs[season // 2, season % 2]
            ax.legend(loc='best')

        axs[-1, 0].set_xlabel('Time from transit, days')
        axs[-1, 0].set_ylabel('Centroid deviation $SNR^2$')
        fig.suptitle('Centroid deviation $SNR^2$ near transits by season', y=1)


def plot_centroid_coordinates_cloud(target_id, transit_times, centr_coords_snr_near_transit_pieces,
                                    seasons, save_fig=Save_fig, no_show=No_show,
                                    plot_output_location=locs.Triggers_plots_location,
                                    dpi=Plots_dpi, figsize=Figsize_centroid_coordinate_pieces,
                                    filename_postfix=locs.Centroid_cloud_plot_filename_postfix):
    with SubplotsWrapper(nrows=1, ncols=1, sharex=True, sharey=True, gridspec_kw=Subplots_gridspec_kw,
                         save_fig=save_fig, no_show=no_show, figsize=figsize, target_id=target_id,
                         filename_postfix=filename_postfix, plot_output_location=plot_output_location, dpi=dpi,
                         grid=True, color_grid=Color_grid, linewidth_grid=Linewidth_grid,
                         zero_lines=True, zero_lines_linewidth=Linewidth_zero_level) as (fig, ax):
        seasons_labeled = np.zeros(4).astype(bool)
        for i in range(len(transit_times)):
            season = seasons[i]
            ax.plot(centr_coords_snr_near_transit_pieces[0][i], centr_coords_snr_near_transit_pieces[1][i], 'o',
                    markersize=1, color=def_cols[season], zorder=10,
                    label='Season ' + str(season) if not seasons_labeled[season] else None)
            seasons_labeled[season] = True
        ax.legend(loc='best')
        ax.set_xlabel('SNR of RA of centroid deviation')
        ax.set_ylabel('SNR of DEC of centroid deviation')
        ax.set_aspect('equal')
        max_lim = np.max(np.abs(np.array(list(ax.get_xlim()) + list(ax.get_ylim()))))
        ax.set_xlim(-max_lim, max_lim)
        ax.set_ylim(-max_lim, max_lim)
        fig.suptitle('Centroid angular position $SNR$ near transits', y=1)


def produce_centroid_plots(tce, save_fig=Save_fig, no_show=No_show, plot_output_location=locs.Triggers_plots_location,
                           dpi=Plots_dpi, time_wing_days=Centroid_time_wing_days_plotting,
                           ftt_window_plotting=Centroid_ftt_wing_days_plotting,):
    (centr_coords_snr_peaks, centr_coords_snr_near_transit_pieces, centr_coords_times_near_transit_pieces,
     centr_snr2_near_transit_pieces, time_total, seasons) = tce.load_centroid_near_transit_pieces(time_wing_days=time_wing_days)
    (ftts_total, stats_2_max_periodic_total, ftts_coords, stats_2_max_periodic_coords, ftt_center) = tce.load_centroid_periodic_results()
    seasons = get_seasons(tce=tce)
    plot_centroids_stat_periodic(tce=None, target_id=tce.target_id,
                                 ftts_total=ftts_total, stats_2_max_periodic_total=stats_2_max_periodic_total,
                                 ftts_coords=ftts_coords,
                                 stats_2_max_periodic_coords=stats_2_max_periodic_coords, ftt_center=ftt_center,
                                 ftt_window_plotting=ftt_window_plotting, max_centroid_snr_2=tce.centroid_snr_2,
                                 save_fig=save_fig, no_show=no_show, plot_output_location=plot_output_location,
                                 dpi=dpi, figsize=Figsize_centroid_periodic,
                                 filename_postfix=locs.Centroid_periodic_plot_filename_postfix
                                 )
    plot_centroid_coordinates_snr_pieces(target_id=tce.target_id, transit_times=tce.summed_transit_times,
                                         centr_coords_snr_near_transit_pieces=centr_coords_snr_near_transit_pieces,
                                         centr_coords_times_near_transit_pieces=centr_coords_times_near_transit_pieces,
                                         seasons=seasons,
                                         save_fig=save_fig, no_show=no_show, plot_output_location=plot_output_location,
                                         dpi=dpi, figsize=Figsize_centroid_coordinate_pieces,
                                         filename_postfix=locs.Centroid_coordinate_pieces_plot_filename_postfix
                                         )
    plot_centroid_snr2_seasons(target_id=tce.target_id, transit_times=tce.summed_transit_times,
                               time_total=time_total, centr_snr2_near_transit_pieces=centr_snr2_near_transit_pieces,
                               seasons=seasons, save_fig=save_fig, no_show=no_show,
                               plot_output_location=plot_output_location,
                               dpi=dpi, figsize=Figsize_centroid_coordinate_pieces,
                               filename_postfix=locs.Centroid_snr2_seasons_plot_filename_postfix
                               )
    plot_centroid_coordinates_cloud(target_id=tce.target_id, transit_times=tce.summed_transit_times,
                                    centr_coords_snr_near_transit_pieces=centr_coords_snr_near_transit_pieces,
                                    seasons=seasons, save_fig=save_fig, no_show=no_show,
                                    plot_output_location=plot_output_location,
                                    dpi=dpi, figsize=Figsize_centroid_coordinate_pieces,
                                    filename_postfix=locs.Centroid_cloud_plot_filename_postfix)


def plot_all_postprocessing(tce, save_fig=Save_fig, no_show=No_show, fluxes_and_stats=True,
                            folding_residuals=True, bg_star_pixels=True, centroid_plots=True,
                            plots_output_location=locs.Triggers_plots_location):
    import target_class
    target = target_class.KeplerTarget(tce.target_id, data_location=tce.data_location, output_location=tce.output_location,
                                       problems_folder=None)
    target.fetch_saved_lightcurve()
    if fluxes_and_stats:
        plot_orig_fluxes(target, save_fig=save_fig, no_show=no_show, plot_output_location=plots_output_location)
        plot_fluxes(target, save_fig=save_fig, no_show=no_show, plot_output_location=plots_output_location)
        plot_stats(target, save_fig=save_fig, no_show=no_show, plot_output_location=plots_output_location)
        plot_whitened_fluxes(target, save_fig=save_fig, no_show=no_show, plot_output_location=plots_output_location)
        plot_folded_stat_snr(target, save_fig=save_fig, no_show=no_show, plot_output_location=plots_output_location, zoomed=False)
        plot_folded_stat_snr(target, save_fig=save_fig, no_show=no_show, plot_output_location=plots_output_location, zoomed=True)
        plot_folded_whitened_flux(target, save_fig=save_fig, no_show=no_show, plot_output_location=plots_output_location, zoomed=False)
        plot_folded_whitened_flux(target, save_fig=save_fig, no_show=no_show, plot_output_location=plots_output_location, zoomed=True)
        plot_snrs_and_wh_fluxes_per_season(target, save_fig=save_fig, no_show=no_show, plot_output_location=plots_output_location)
    if folding_residuals:
        plot_folding_residuals_histograms(tce, save_fig=save_fig, no_show=no_show, plot_output_location=plots_output_location)
    if bg_star_pixels:
        plot_bg_star_pixels_ppm(tce, save_fig=save_fig, no_show=no_show, plot_output_location=plots_output_location)
    if centroid_plots:
        produce_centroid_plots(tce, save_fig=save_fig, no_show=no_show, plot_output_location=plots_output_location)


def plots_all_after_slides_injections(tce, save_fig=Save_fig, no_show=No_show):
    import target_class
    target = target_class.KeplerTarget(tce.target_id, data_location=tce.data_location, output_location=tce.output_location,
                                       problems_folder=None)
    target.fetch_saved_lightcurve()
    plot_significance_histograms(target, save_fig=save_fig, no_show=no_show)
    plot_periodogram_chunks(target, save_fig=save_fig, no_show=no_show)
    plot_injections_corners(target, save_fig=save_fig, no_show=no_show)
    plot_injections_radii_durations_score(target, save_fig=save_fig, no_show=no_show)


# ##########################################################################################################
# AUXILIARY FUNCTIONS
# ##########################################################################################################

def prepare_target_for_plotting(target, initialize_templates=False,
                                condition_templates=False, detect_single_transits=False):
    if not hasattr(target, 'best_int_score'):
        target.unpickle_small_attributes()

    go_fetch = (
        not hasattr(target, 'klc') or
        not hasattr(target.klc, 'summed_stats_raw') or
        (not hasattr(target.klc, 'tb') and initialize_templates) or
         (not hasattr(target.klc, 'whitened_templates_td') and condition_templates) or
          (not hasattr(target.klc, 'statistics_snr') and detect_single_transits))
    if go_fetch:
        target.fetch_saved_lightcurve(initialize_templates=initialize_templates, condition_templates=condition_templates,
                                      detect_single_transits=detect_single_transits)


def get_seasons(tce=None, transit_times=None):
    from auxiliary_functions import get_orig_quarters
    if tce is not None:
        if tce.transit_seasons is None:
            seasons = np.array(get_orig_quarters(tce.summed_transit_times)) % 4
        else:
            seasons = tce.transit_seasons
    else:
        seasons = np.array(get_orig_quarters(transit_times)) % 4
    return seasons


def set_axes_limits_fold(ax, bins_flux, bins_time, folds_fluxes, folds_times, best_per, plot_twice=False,
                         zoomed=False, zoom_wing=Zoom_wing_folded_plots):
    maxdev = np.nanmax(
        [np.nanmax(np.abs(bins_flux)), np.nanmax([np.nanmax(np.abs(folds_fluxes[i])) for i in range(len(folds_fluxes))])])
    ax.set_xlim(-best_per / 2, best_per / 2)
    if plot_twice:
        ax.set_xlim(-best_per / 2, best_per / 2 + best_per)
        ax.axvline(best_per / 2, color=(0.5, 0.5, 0.5), linewidth=Linewidth_zero_level, zorder=0)
    if zoomed:
        inds_bins = np.abs(bins_time) <= zoom_wing
        m1 = np.nanmax(np.abs(bins_flux[inds_bins])) if len(bins_flux[inds_bins]) > 0 and not np.all(np.isnan(bins_flux[inds_bins])) else 0
        m2 = np.nanmax([np.nanmax(np.abs(folds_fluxes[i][np.abs(folds_times[i]) <= zoom_wing]))
                        if len(folds_fluxes[i][np.abs(folds_times[i]) <= zoom_wing]) > 0
                        and not np.all(np.isnan(folds_fluxes[i][np.abs(folds_times[i]) <= zoom_wing])) else 0
                        for i in range(len(folds_fluxes))])
        maxdev = np.nanmax([m1, m2])
        ax.set_xlim(-zoom_wing, zoom_wing)
    ax.set_ylim(-maxdev * 1.01, maxdev * 1.01)  # make the limits symmetric


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
    title = 'KIC' + self.target_id
    if score is not None:
        title += r', $SNR^2$ ' + str(np.round(score, 1))
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


def plot_vlines(times, ax=None, zorder=0, kwargs=(('color', 'k'), ('linewidth', Linewidth_vlines))):
    kwargs = dict(kwargs)
    if ax is None:
        ax = plt.gca()
    for tim in times:
        ax.axvline(tim, zorder=zorder, **kwargs)


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
