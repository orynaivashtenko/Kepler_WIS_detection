#!/usr/bin/env python3

import os

from template_bank.template_module import Indices_of_templates, dt_lc

core_num = "1"
os.environ["OMP_NUM_THREADS"] = core_num
os.environ["OPENBLAS_NUM_THREADS"] = core_num
os.environ["MKL_NUM_THREADS"] = core_num
os.environ["VECLIB_MAXIMUM_THREADS"] = core_num
os.environ["NUMEXPR_NUM_THREADS"] = core_num

import numpy as np
from auxiliary_functions import (fold_fluxes, get_average_stat_template, get_average_whitened_template,
                                 get_shift_reduced_full_stat_snr, get_near_transit_pieces_from_full_time,
                                 get_near_transit_pieces_of_whitened_flux, get_seasons,
                                 stack_fluxes)
import utils as ut
import pickle
import pipeline_parameters as pp
import locations as locs
from auxiliary_functions import get_saving_file_path, get_correct_tid_folder
from tce_parameters import Min_len_part_bins_veto_parts_fold
from pipeline_parameters import Per_min_candidate
# from astropy.io import fits
# import sys
import copy
# import search_pipeline as spip
# import target_class

from plotting_parameters import *


# ---------------------------------------------------------------------------------------------------------------------

def update_parameters():
    plt.rcParams.update({'font.size': Fontsize, 'font.serif': 'Times New Roman',
                         'lines.linewidth': Linewidth, 'lines.markersize': Markersize,
                         'legend.fontsize': Fontsize_legend, 'legend.loc': 'upper right',
                         'axes.titlesize': Fontsize_title, 'axes.labelsize': Fontsize_label,
                         'xtick.labelsize': Fontsize_label, 'ytick.labelsize': Fontsize_label,
                         'figure.titlesize': Fontsize_title, 'figure.dpi': Plots_dpi})


# ---------------------------------------------------------------------------------------------------------------------
class PlotWrapper(object):
    def __init__(self, save_fig=Save_fig, no_show=No_show, ax=None, figsize=None, target_id=None,
                 filename_postfix='plot.png', plot_output_location=locs.Triggers_plots_location,
                 gridspec_kw=Fig_gridspec_kw, dpi=Plots_dpi, tight_layout=True,
                 grid=True, color_grid=Color_grid, linewidth_grid=Linewidth_grid, which_grid='both',
                 allow_plot_folder_creation=False, file_format='png'):
        self.save_fig = save_fig
        self.no_show = no_show
        self.plot_output_location = plot_output_location
        self.dpi = dpi
        self.grid = grid
        self.grid_color = color_grid
        self.grid_linewidth = linewidth_grid
        self.which_grid = which_grid
        self.gridspec_kw = gridspec_kw
        self.tight_layout = tight_layout
        if self.save_fig:
            tid_folder = get_correct_tid_folder(target_id=target_id, location=plot_output_location,
                                                allow_creation=allow_plot_folder_creation, with_prefix_folder=False)
            self.file_saving_path = get_saving_file_path(output_location=tid_folder,
                                                         target_id=target_id,
                                                         filename_postfix=filename_postfix,
                                                         file_format=file_format)
        # fig.subplots_adjust(left=None, bottom=None, right=None, top=0.94, wspace=None, hspace=0)
        if ax is None:
            self.fig = plt.figure(figsize=figsize, dpi=self.dpi)
            self.ax = plt.gca()
        else:
            self.fig = plt.gcf()
            self.ax = ax
        if self.gridspec_kw is not None:
            self.fig.subplots_adjust(**self.gridspec_kw)

    def __enter__(self):
        return self.fig, self.ax

    def __exit__(self, *args):
        if self.grid:
            self.ax.grid(which=self.which_grid, color=self.grid_color, linewidth=self.grid_linewidth, zorder=0)
        if self.tight_layout:
            self.fig.tight_layout()
        if self.save_fig:
            if self.gridspec_kw is not None:
                self.fig.savefig(self.file_saving_path, dpi=self.dpi)
            else:
                self.fig.savefig(self.file_saving_path, dpi=self.dpi, bbox_inches='tight')
        if self.no_show is None:
            pass
        elif self.no_show:
            plt.clf()
            plt.close(self.fig)
        else:
            plt.show()


class SubplotsWrapper(object):
    def __init__(self, nrows=2, ncols=2, sharex=True, sharey=True, gridspec_kw=Subplots_gridspec_kw,
                 save_fig=Save_fig, no_show=No_show, figsize=Figsize, target_id=None,
                 filename_postfix='plot.png', plot_output_location=locs.Triggers_plots_location, dpi=Plots_dpi,
                 grid=True, color_grid=Color_grid, linewidth_grid=Linewidth_grid, tight_layout=True, fig_bbox=False,
                 zero_lines=True, zero_vline_color='k', zero_hline_color='k', zero_lines_linewidth=Linewidth_zero_level,
                 allow_plot_folder_creation=False, file_format='png'):
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
        self.zero_vline_color = zero_vline_color
        self.zero_hline_color = zero_hline_color
        self.gridspec_kw = gridspec_kw
        self.tight_layout = tight_layout
        self.fig_bbox = fig_bbox
        if self.save_fig:
            tid_folder = get_correct_tid_folder(target_id=target_id, location=plot_output_location,
                                                allow_creation=allow_plot_folder_creation, with_prefix_folder=False)
            self.file_saving_path = get_saving_file_path(output_location=tid_folder,
                                                         target_id=target_id,
                                                         filename_postfix=filename_postfix,
                                                         file_format=file_format)
        if self.gridspec_kw is not None:
            self.fig, self.axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex=sharex, sharey=sharey,
                                              gridspec_kw=self.gridspec_kw, dpi=self.dpi)
        else:
            self.fig, self.axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex=sharex, sharey=sharey,
                                              dpi=self.dpi)

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
                    ax.axvline(0, color=self.zero_vline_color, linestyle='-', linewidth=self.zero_lines_linewidth, zorder=2)
                    ax.axhline(0, color=self.zero_hline_color, linestyle='-', linewidth=self.zero_lines_linewidth, zorder=2)
            else:
                self.axs.axvline(0, color=self.zero_vline_color, linestyle='-', linewidth=self.zero_lines_linewidth, zorder=2)
                self.axs.axhline(0, color=self.zero_hline_color, linestyle='-', linewidth=self.zero_lines_linewidth, zorder=2)
        if self.tight_layout:
            self.fig.tight_layout()
        if self.save_fig:
            if self.gridspec_kw is None:
                self.fig.savefig(self.file_saving_path, dpi=self.dpi, bbox_inches='tight')
            else:
                if not self.fig_bbox:
                    self.fig.savefig(self.file_saving_path, dpi=self.dpi)
                else:
                    self.fig.savefig(self.file_saving_path, dpi=self.dpi, bbox_inches=self.fig.bbox)
        if self.no_show is None:
            pass
        elif self.no_show:
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
                     add_title=True, add_legend=True,
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
                plot_vlines(times=target.klc.summed_times, ax=axx)
                plot_points_at_times(times=times, values=fluxes, point_times=target.klc.summed_times, ax=axx, **points_kwargs)

        axx.set_xlabel('time, days')
        axx.set_ylabel('Original fluxes, \nmean subtracted')

        if add_title:
            title = get_target_title(target, score=None, per=None, temp=None)
            axx.set_title(title)
        if add_legend:
            axx.legend(loc='upper right')


def plot_fluxes(target, colors=(Flux_quarters_colors[0], Flux_quarters_colors[1]), plot_trigger=True,
                ax=None, save_fig=Save_fig, no_show=No_show, plot_output_location=locs.Triggers_plots_location,
                figsize=Figsize_flux, filename_postfix=locs.Red_fluxes_plot_filename_postfix, add_title=True, add_legend=True,
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
                plot_vlines(times=target.klc.summed_times, ax=axx)
                plot_points_at_times(times=times, values=fluxes, point_times=target.klc.summed_times, ax=axx, **points_kwargs)

        axx.set_xlabel('time, days')
        axx.set_ylabel('Conditioned fluxes')

        if add_title:
            title = get_target_title(target, score=None, per=None, temp=None)
            axx.set_title(title)
        if add_legend:
            axx.legend(loc='upper right')


def plot_stats(target, temp_number=None, units='snr', edge_gap_time=10, plot_trigger=True,
               points_kwargs=(('color', 'r'), ('markersize', 3)), masked_points_kwargs=(('color', 'k'), ('markersize', 1)),
               mark_zero=True, color=def_cols[0], add_title=True, add_legend=True,
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
        plt.plot(time[~mask], statistics[~mask], 'o', **masked_points_kwargs, label='Masked')

        if plot_trigger:
            if target.klc.summed_times is not None:
                plot_vlines(times=target.klc.summed_times, ax=axx)
                plot_points_at_times(times=[time], values=[statistics], point_times=target.klc.summed_times, ax=axx, **points_kwargs,
                                     label='Transits')

        if mark_zero:
            axx.axhline(0, linewidth=Linewidth_zero_level, color='k', zorder=0)
        axx.set_xlim(np.nanmin(time) - edge_gap_time, np.nanmax(time) + edge_gap_time)
        axx.set_xlabel('time, days')
        ylabel = 'SES statistics'
        if units == 'snr':
            ylabel += ', [SNR]'
        axx.set_ylabel(ylabel)
        if add_title:
            title = get_target_title(target, score=None, per=None, temp=temp_number)
            axx.set_title(title)
        if add_legend:
            axx.legend(loc='upper right')


def plot_whitened_fluxes(target, colors=Flux_quarters_colors, plot_trigger=True,
                         points_kwargs=(('color', 'r'), ('markersize', 3), ('label', 'transits')),
                         ax=None, save_fig=Save_fig, no_show=No_show, plot_output_location=locs.Triggers_plots_location,
                         figsize=Figsize_flux, filename_postfix=locs.Whitened_fluxes_plot_filename_postfix,
                         add_title=True, add_legend=True, mark_zero=True, **kwargs):
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
                plot_vlines(times=target.klc.summed_times, ax=axx)
                plot_points_at_times(times=times, values=fluxes, point_times=target.klc.summed_times, ax=axx, **points_kwargs)

        if mark_zero:
            axx.axhline(0, linewidth=Linewidth_zero_level, color='k', zorder=0)
        axx.set_xlabel('time, days')
        axx.set_ylabel('Whitened fluxes')
        if add_title:
            title = get_target_title(target, score=None, per=None, temp=None)
            axx.set_title(title)
        if add_legend:
            axx.legend(loc='upper right')


def plot_all_series(target, save_fig=Save_fig, no_show=No_show, figsize=Figsize_all_fluxes,
                    filename_postfix=locs.All_series_plot_filename_postfix,
                    plot_output_location=locs.Triggers_plots_location, **plot_kwargs):
    with SubplotsWrapper(nrows=4, ncols=1, sharex=True, sharey=False, gridspec_kw=Subplots_gridspec_kw, save_fig=save_fig, no_show=no_show,
                         figsize=figsize, target_id=target.target_id, filename_postfix=filename_postfix,
                         plot_output_location=plot_output_location, grid=True, zero_lines=False) as (fig, axs):
        plot_orig_fluxes(target, plot_trigger=True, ax=axs[0], save_fig=False, no_show=None,
                         add_title=True, add_legend=False, **plot_kwargs)
        plot_fluxes(target, plot_trigger=True, ax=axs[1], save_fig=False, no_show=None,
                    add_title=False, add_legend=False, **plot_kwargs)
        plot_whitened_fluxes(target, colors=Flux_quarters_colors, plot_trigger=True, ax=axs[2], save_fig=False, no_show=None,
                             add_title=False, add_legend=False, **plot_kwargs)
        plot_stats(target, temp_number=target.best_temp, plot_trigger=True, ax=axs[3], save_fig=False, no_show=None,
                   add_title=False, add_legend=True, **plot_kwargs)


def plot_all_folds(target, save_fig=Save_fig, no_show=No_show, figsize=Figsize_all_folds,
                   filename_postfix=locs.All_folds_plot_filename_postfix,
                   filename_postfix_zoomed=locs.All_folds_zoomed_filename_postfix,
                   plot_output_location=locs.Triggers_plots_location):
    with SubplotsWrapper(nrows=2, ncols=1, sharex=True, sharey=False, gridspec_kw=Subplots_gridspec_kw, save_fig=save_fig, no_show=no_show,
                         figsize=figsize, target_id=target.target_id, filename_postfix=filename_postfix,
                         plot_output_location=plot_output_location, grid=True, zero_lines=False) as (fig, axs):
        plot_folded_whitened_flux(target, ax=axs[0], save_fig=False, no_show=None, plot_output_location=plot_output_location,
                                  zoomed=False, add_title=True, add_legend=True)
        plot_folded_stat_snr(target, ax=axs[1], save_fig=False, no_show=None, plot_output_location=plot_output_location,
                             zoomed=False, add_title=False, add_legend=False)

    with SubplotsWrapper(nrows=2, ncols=1, sharex=True, sharey=False, gridspec_kw=Subplots_gridspec_kw, save_fig=save_fig, no_show=no_show,
                         figsize=figsize, target_id=target.target_id, filename_postfix=filename_postfix_zoomed,
                         plot_output_location=plot_output_location, grid=True, zero_lines=False) as (fig, axs):
        plot_folded_whitened_flux(target, ax=axs[0], save_fig=False, no_show=None, plot_output_location=plot_output_location,
                                  zoomed=True, add_title=False, add_legend=True)
        plot_folded_stat_snr(target, ax=axs[1], save_fig=False, no_show=None, plot_output_location=plot_output_location,
                             zoomed=True, add_title=False, add_legend=False)


def plot_folded_stat_snr(target, temp_number=None, plot_twice=False, zoomed=False, zoom_wing=Zoom_wing_folded_plots,
                         ax=None, save_fig=Save_fig, no_show=No_show, plot_output_location=locs.Triggers_plots_location,
                         figsize=Figsize_flux, add_title=True, add_legend=True, binwid=None, fold_marker_color=(0.55, 0.55, 0.55),
                         filename_postfix=locs.Folded_statistics_plot_filename_postfix,
                         filename_postfix_zoomed=locs.Folded_statistics_zoomed_plot_filename_postfix,
                         template_linewidth=None, markersize_bin=None, elinewidth=None, markersize_fold=None):
    if zoomed:
        filename_postfix = filename_postfix_zoomed
        if markersize_bin is None:
            markersize_bin = 2.5
        if markersize_fold is None:
            markersize_fold = 0.6
        if template_linewidth is None:
            template_linewidth = Linewidth
        if elinewidth is None:
            elinewidth = 1.2
    else:
        if markersize_bin is None:
            markersize_bin = 1.5
        if markersize_fold is None:
            markersize_fold = 0.4
        if template_linewidth is None:
            template_linewidth = Linewidth * 0.7
        if elinewidth is None:
            elinewidth = 1

    prepare_target_for_plotting(target, initialize_templates=False,
                                condition_templates=False, detect_single_transits=False)
    if temp_number is None:
        temp_number = target.best_temp

    time, statistics, variances, mask, corrections, statistics_gaussianized = get_interleaving_stats(target=target, temp_number=temp_number)
    statistics = ut.safe_div(statistics, variances ** 0.5, fill_value=0.)

    # FOLDING
    if binwid is None:
        binwid = get_binwid(temp_dur_days=target.templates_durations_hours[temp_number] / 24, dt=target.klc.dt_days,
                            min_binwid_to_dt=0.2, min_part_duration=None,
                            max_binwid_to_dt=5., max_part_duration=0.2)
    # min(target.templates_durations_hours[temp_number] / 24 / 5, 0.1))
    bins_time, bins_flux, bins_counts, folds_times, folds_fluxes, bins_std = fold_fluxes(
        times=time, fluxes=statistics, period=target.best_per, first_time=target.best_ftt,
        masks=mask, binwid=binwid,  # self.templates_durations_hours[temp_number] / 24 * 10,
        dt=target.klc.dt_days, means=None, template=None, return_std=True,
        normalize_by_counts=True)  # np.array(self.klc.templates[self.klc.indices_of_templates[temp_number][0]]))

    # MAKING TEMPLATE
    n_time_templ = int(np.round(target.best_per / target.klc.dt_days))
    temp_plot, time_template = get_average_stat_template(whitened_templates_td=target.klc.whitened_templates_td,
                                                         temp_number=temp_number, n_time_templ=n_time_templ,
                                                         dt_days=target.klc.dt_days,
                                                         indices_of_templates=target.klc.indices_of_templates)
    temp_plot = - temp_plot * target.best_stat ** 0.5 / (len(target.klc.summed_stats_raw)) ** 0.5

    # Plotting
    with PlotWrapper(target_id=target.target_id, save_fig=save_fig, no_show=no_show, ax=ax, figsize=figsize,
                     filename_postfix=filename_postfix, plot_output_location=plot_output_location) as (fig, axx):
        # If someone wants to repeat the period two times as it is often done in the literature:
        if plot_twice:
            time_shifts_plotting = [0, target.best_per]
            labels_binning = ['Binning, bin ' + ut.fstr(binwid) + ' days', None]
            labels_template = ['Average per-transit SNR', None]
        else:
            time_shifts_plotting = [0]
            labels_binning = ['Binning, bin ' + ut.fstr(binwid) + ' days']
            labels_template = ['Average per-transit SNR']

        # Actual plotting:
        for j in range(len(time_shifts_plotting)):
            # Plot the folded data:
            for i in range(len(folds_times)):
                axx.plot(folds_times[i] + time_shifts_plotting[j], folds_fluxes[i], 'o',
                         markersize=markersize_fold, color=fold_marker_color)
            # Plot the binned folded data:
            axx.errorbar(bins_time[bins_counts > 0] + time_shifts_plotting[j], bins_flux[bins_counts > 0], yerr=bins_std[bins_counts > 0],
                         xerr=None, markersize=markersize_bin, fmt='o', markerfacecolor='k', markeredgecolor='k',
                         ecolor='k', elinewidth=elinewidth, capsize=None, barsabove=False, lolims=False, uplims=False,
                         xlolims=False, xuplims=False, errorevery=1, capthick=None,
                         label=labels_binning[j])
            # Plot the expected template:
            axx.plot(time_template + time_shifts_plotting[j], temp_plot, color='r', linewidth=template_linewidth,
                     label=labels_template[j])

        # Setting the limits:
        set_axes_limits_fold(ax=axx, bins_flux=bins_flux, bins_time=bins_time, folds_fluxes=folds_fluxes, folds_times=folds_times,
                             best_per=target.best_per, plot_twice=plot_twice, zoomed=zoomed, zoom_wing=zoom_wing)
        axx.set_xlabel('Time from mid transit, days')
        axx.set_ylabel('Statistic [SNR] fold')
        if not zoomed:
            axx.xaxis.set_major_locator(ticker.MaxNLocator(nbins=11))
        else:
            axx.set_xticks(np.arange(-np.round(zoom_wing/0.5) * 0.5, np.round(zoom_wing/0.5) * 0.5 + 0.01, 0.5))
        if add_title:
            title = get_target_title(target, score=None, per=None, temp=temp_number)
            axx.set_title(title)
        if add_legend:
            axx.legend(loc='upper right')


def plot_folded_whitened_flux(target, plot_twice=False, temp_number=None, zoomed=False, zoom_wing=Zoom_wing_folded_plots,
                              ax=None, save_fig=Save_fig, no_show=No_show, plot_output_location=locs.Triggers_plots_location,
                              figsize=Figsize_flux, add_title=True, add_legend=True, binwid=None, fold_marker_color=(0.55, 0.55, 0.55),
                              filename_postfix=locs.Folded_whitened_fluxes_plot_filename_postfix,
                              filename_postfix_zoomed=locs.Folded_whitened_fluxes_zoomed_plot_filename_postfix,
                              template_linewidth=None, markersize_bin=None, elinewidth=None, markersize_fold=None):
    if zoomed:
        filename_postfix = filename_postfix_zoomed
        if markersize_bin is None:
            markersize_bin = 2.5
        if markersize_fold is None:
            markersize_fold = 0.6
        if template_linewidth is None:
            template_linewidth = Linewidth_thin * 0.8
        if elinewidth is None:
            elinewidth = 1.2
    else:
        if markersize_bin is None:
            markersize_bin = 1.5
        if markersize_fold is None:
            markersize_fold = 0.4
        if template_linewidth is None:
            template_linewidth = Linewidth_thin * 0.8 * 0.7
        if elinewidth is None:
            elinewidth = 1
    prepare_target_for_plotting(target, initialize_templates=False,
                                condition_templates=False, detect_single_transits=False)
    if temp_number is None:
        temp_number = target.best_temp
    # FOLDING
    if binwid is None:
        binwid = get_binwid(temp_dur_days=target.templates_durations_hours[temp_number] / 24, dt=target.klc.dt_days,
                            min_binwid_to_dt=0.2, min_part_duration=0.2,
                            max_binwid_to_dt=5., max_part_duration=0.2)
    # min(target.templates_durations_hours[temp_number] / 24 / 5, 0.1)
    bins_time, bins_flux, bins_counts, folds_times, folds_fluxes, bins_std = fold_fluxes(
        times=target.klc.times, fluxes=target.klc.whitened_fluxes_td, period=target.best_per, first_time=target.best_ftt,
        masks=target.klc.big_hole_masks, binwid=binwid,  # self.templates_durations_hours[temp_number] / 24 * 10,
        dt=target.klc.dt_days, means=None, template=None, return_std=True,
        normalize_by_counts=True)  # np.array(self.klc.templates[self.klc.indices_of_templates[temp_number][0]]))
    n_time_templ = int(np.round(target.best_per / target.klc.dt_days))
    temp_best, time_template = get_average_whitened_template(whitened_templates_td=target.klc.whitened_templates_td,
                                                             temp_number=temp_number, n_time_templ=n_time_templ,
                                                             dt_days=target.klc.dt_days,
                                                             indices_of_templates=target.klc.indices_of_templates)
    non_zero_mask = target.klc.summed_vars > 0
    if np.sum(non_zero_mask.astype(float)) > 0:
        ampl_est = np.nanmean(target.klc.summed_stats_raw[non_zero_mask] / target.klc.summed_vars[non_zero_mask])
    else:
        ampl_est = np.nan

    # Plotting
    with PlotWrapper(target_id=target.target_id, save_fig=save_fig, no_show=no_show, ax=ax, figsize=figsize,
                     filename_postfix=filename_postfix, plot_output_location=plot_output_location) as (fig, axx):

        # If someone wants to repeat the period two times as it is often done in the literature:
        if plot_twice:
            time_shifts_plotting = [0, target.best_per]
            labels_binning = ['Binning, bin ' + ut.fstr(binwid) + ' days', None]
            labels_template = ['Average whitened template', None]
        else:
            time_shifts_plotting = [0]
            labels_binning = ['Binning, bin ' + ut.fstr(binwid) + ' days']
            labels_template = ['Average whitened template']

        # Actual plotting:
        for j in range(len(time_shifts_plotting)):
            for i in range(len(folds_times)):
                axx.plot(folds_times[i] + time_shifts_plotting[j], folds_fluxes[i], 'o', markersize=markersize_fold,
                         color=fold_marker_color)
            axx.errorbar(bins_time[bins_counts > 0] + time_shifts_plotting[j], bins_flux[bins_counts > 0], yerr=bins_std[bins_counts > 0],
                         xerr=None, markersize=markersize_bin, fmt='o', markerfacecolor='k', markeredgecolor='k',
                         ecolor='k', elinewidth=elinewidth, capsize=None, barsabove=False, lolims=False, uplims=False,
                         xlolims=False, xuplims=False, errorevery=1, capthick=None,
                         label=labels_binning[j])
            # ax.plot(bins_time, bins_flux, '-o', markersize=3, color='b', linewidth=1, label='binning, bin ' + ut.fstr(binwid) + ' days')
            axx.plot(time_template + time_shifts_plotting[j], temp_best * ampl_est, color='r', linewidth=template_linewidth,
                     label=labels_template[j])
        # Setting limits:
        set_axes_limits_fold(ax=axx, bins_flux=bins_flux, bins_time=bins_time, folds_fluxes=folds_fluxes, folds_times=folds_times,
                             best_per=target.best_per, plot_twice=plot_twice, zoomed=zoomed, zoom_wing=zoom_wing)
        if not zoomed:
            axx.xaxis.set_major_locator(ticker.MaxNLocator(nbins=11))
        else:
            axx.set_xticks(np.arange(-np.round(zoom_wing / 0.5) * 0.5, np.round(zoom_wing / 0.5) * 0.5 + 0.01, 0.5))
        axx.set_xlabel('Time from mid transit, days')
        axx.set_ylabel('Whitened flux')
        if add_title:
            title = get_target_title(target, score=None, per=None, temp=temp_number)
            axx.set_title(title)
        if add_legend:
            axx.legend(loc='upper right')


def plot_snrs_and_wh_fluxes_per_season(target, time_wing_days=Seasonal_plotting_wing_days, binwid=None,
                                       save_fig=Save_fig, no_show=No_show, figsize=Figsize_season_pieces,
                                       filename_postfix_snr=locs.Season_pieces_snr_plot_filename_postfix,
                                       filename_postfix_wh_flux=locs.Season_pieces_wh_flux_plot_filename_postfix,
                                       plot_output_location=locs.Triggers_plots_location,
                                       markersize_folds=Markersize_folds_small,
                                       markersize_bins=Markersize_bins_normal, linewidth=Linewidth_thin):
    prepare_target_for_plotting(target, initialize_templates=False, condition_templates=False, detect_single_transits=False)
    transit_times = target.klc.summed_times
    seasons = get_seasons(tce=None, transit_times=transit_times)
    if binwid is None:
        binwid = get_binwid(temp_dur_days=Durations[target.best_temp], dt=target.klc.dt_days,
                            min_binwid_to_dt=1., min_part_duration=0.2,
                            max_binwid_to_dt=None, max_part_duration=None)

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
         'Whitened flux', 'Whitened flux near transits by season'],
        [times_near_transit_pieces_snr, near_transit_pieces_snr, mask_near_pieces_snr, filename_postfix_snr,
         'Single-transit statistic in units of $SNR$', 'Single-transit statistic near transits by season']]:
        with SubplotsWrapper(nrows=2, ncols=2, sharex=True, sharey=True, gridspec_kw=Subplots_gridspec_kw, save_fig=save_fig,
                             no_show=no_show, figsize=figsize, target_id=target.target_id, filename_postfix=filename_postfix,
                             plot_output_location=plot_output_location, grid=True, zero_lines=False) as (fig, axs):
            seasons = np.array(seasons)
            seasons[~np.isnan(seasons)] = seasons[~np.isnan(seasons)].astype(int)
            times_for_seasons = [[times_near_transit_pieces[i]
                                  for i in range(len(times_near_transit_pieces)) if seasons[i] == s] for s in range(4)]
            datas_for_seasons = [[data_near_transit_pieces[i]
                                  for i in range(len(data_near_transit_pieces)) if seasons[i] == s] for s in range(4)]
            masks_for_seasons = [[mask_near_transit_pieces[i]
                                  for i in range(len(mask_near_transit_pieces)) if seasons[i] == s] for s in range(4)]
            transit_times_for_seasons = [[transit_times[i]
                                          for i in range(len(transit_times)) if seasons[i] == s] for s in range(4)]
            min_time = 0.
            max_time = 0.

            for s in range(4):
                ax = axs[s // 2, s % 2]
                label = 'S' + str(s)  # if not seasons_labeled[season] else None
                col = (0.55, 0.55, 0.55)
                if len(times_for_seasons[s]) > 0:
                    for i in range(len(times_for_seasons[s])):
                        times_masked_plot = times_for_seasons[s][i][masks_for_seasons[s][i]] - transit_times_for_seasons[s][i]
                        ax.plot(times_masked_plot,
                                datas_for_seasons[s][i][masks_for_seasons[s][i]], linestyle='',
                                marker='o', markersize=markersize_folds, color=col, zorder=10)
                        if len(times_masked_plot) > 0:
                            min_time = min(min_time, np.nanmin(times_masked_plot))
                            max_time = max(min_time, np.nanmax(times_masked_plot))
                    if len(times_for_seasons[s]) > 1:
                        bins_time, bins_flux, bins_counts, folds_times, folds_fluxes, bins_std = fold_fluxes(
                            times=copy.deepcopy(times_for_seasons[s]),
                            fluxes=copy.deepcopy(datas_for_seasons[s]),
                            period=target.best_per,
                            first_time=target.best_ftt,
                            masks=copy.deepcopy(masks_for_seasons[s]),
                            binwid=binwid, dt=target.klc.dt_days, means=None, template=None, return_std=True,
                            normalize_by_counts=True)
                    else:
                        bins_time = times_for_seasons[s][0] - transit_times_for_seasons[s][0]
                        bins_flux = datas_for_seasons[s][0]
                        bins_counts = (~np.isclose(bins_flux, 0)).astype(int)
                        bins_std = np.zeros_like(bins_flux)
                    col = 'k'
                    ax.errorbar(bins_time[bins_counts > 0], bins_flux[bins_counts > 0], yerr=bins_std[bins_counts > 0],
                                xerr=None, markersize=markersize_bins, fmt='o', markerfacecolor=col, markeredgecolor=col,
                                ecolor=col, elinewidth=linewidth, capsize=None, barsabove=False, lolims=False, uplims=False,
                                xlolims=False, xuplims=False, errorevery=1, capthick=None,
                                label=label, zorder=15)
                    ax.axhline(0, linewidth=Linewidth_zero_level, color='k', zorder=0)
                    ax.legend(loc='upper left', borderpad=0.05, handletextpad=0.05)
            if min_time != 0 and max_time != 0:
                for s in range(4):
                    ax = axs[s // 2, s % 2]
                    ax.set_xlim(min_time, max_time)

            axs[-1, 0].set_xlabel(' ' * 140 + 'Time from mid transit, days')
            axs[-1, 0].set_ylabel(' ' * 40 + ylabel)
            fig.suptitle(get_target_title(target) + ', ' + title, y=1)


def plot_wh_fluxes_even_odd_from_veto(tce, time_wing_days=Zoom_wing_folded_plots,
                                      save_fig=Save_fig, no_show=No_show, figsize=Figsize_even_odd_from_veto,
                                      filename_postfix_data=locs.Folded_binned_wh_flux_model_even_odd_filename_postfix,
                                      filename_postfix_plot=locs.Even_odd_wh_flux_from_veto_plot_filename_postfix,
                                      plot_output_location=locs.Triggers_plots_location,
                                      markersize_bins=Markersize_bins_small, linewidth=Linewidth_thin):
    file_path = get_saving_file_path(target_id=tce.target_id, output_location=tce.output_location,
                                     filename_postfix=filename_postfix_data,
                                     insert_after_postfix=None, file_format='npy')
    if not os.path.exists(file_path):
        raise FileNotFoundError('No file to load from found at ' + file_path)
    else:
        res = np.load(file_path, allow_pickle=True)
        std_to_ampl_even_odd = [tce.scores_ampls_even_odd[0][2] / tce.scores_ampls_even_odd[0][1],
                                tce.scores_ampls_even_odd[1][2] / tce.scores_ampls_even_odd[1][1]]
        with SubplotsWrapper(nrows=1, ncols=2, sharex=True, sharey=True, gridspec_kw=Subplots_gridspec_kw_3_line_title, save_fig=save_fig,
                             no_show=no_show, figsize=figsize, target_id=tce.target_id, filename_postfix=filename_postfix_plot,
                             plot_output_location=plot_output_location, grid=True, zero_lines=False) as (fig, axs):

            for i_ax, ev_od_label in [[0, 'Even'], [1, 'Odd']]:
                flux_normalized, transit_model, bins_time, _ = res[i_ax]
                ax = axs[i_ax]
                col = 'k'
                ax.plot(bins_time, flux_normalized, 'o', markersize=markersize_bins, color=col, zorder=13)
                ax.plot(bins_time, transit_model, '-', linewidth=linewidth, color=def_cols[3], zorder=10)
                ax.fill_between(x=bins_time, y1=transit_model * (1 - std_to_ampl_even_odd[i_ax]),
                                y2=transit_model * (1 + std_to_ampl_even_odd[i_ax]), color=def_cols[3], alpha=0.5,
                                linewidth=0)
                ax.axhline(0, linewidth=Linewidth_zero_level, color='k', zorder=0)
                ax.set_xlabel('Time from transit, days')
                ax.set_title(ev_od_label)
                ax.set_xlim(-time_wing_days, time_wing_days)
            axs[0].set_ylabel('Binned normalized \nhigh-passed whitened flux')
            fig.suptitle(get_target_title(tce) + ', Even-odd binned folds, p-val: '
                         + (str(np.round(tce.p_value_even_odd_ampl, 3))
                            if tce.p_value_even_odd_ampl is not None else 'None'), y=1)


def plot_snrs_and_wh_fluxes_even_odd(target,
                                     best_temp_folded=None, best_ampls=None,  # only_snr=True,
                                     binwid=None, time_wing_days=Zoom_wing_folded_plots,
                                     save_fig=Save_fig, no_show=No_show, figsize=Figsize_even_odd,
                                     # filename_postfix_snr=locs.Even_odd_snr_plot_filename_postfix,
                                     # filename_postfix_wh_flux=locs.Even_odd_wh_flux_plot_filename_postfix,
                                     filename_postfix=locs.Even_odd_wh_flux_and_snr_plot_filename_postfix,
                                     plot_output_location=locs.Triggers_plots_location,
                                     markersize_folds=Markersize_folds_small,
                                     markersize_bins=Markersize_bins_normal, linewidth=Linewidth_thin):
    prepare_target_for_plotting(target, initialize_templates=False, condition_templates=False, detect_single_transits=False)
    transit_times = target.klc.summed_times
    even_transit_times = transit_times[0::2]
    odd_transit_times = transit_times[1::2]
    if best_temp_folded is None:
        best_temp_folded = target.best_temp
    indices_of_templates = target.klc.indices_of_templates
    dt_days = target.klc.dt_days
    if binwid is None:
        binwid = get_binwid(temp_dur_days=Durations[best_temp_folded], dt=dt_days,
                            min_binwid_to_dt=1., min_part_duration=0.2,
                            max_binwid_to_dt=None, max_part_duration=None)

    if np.array([len(target.klc.whitened_fluxes_td[q]) == target.klc.support
                 for q in range(len(target.klc.whitened_fluxes_td))]).all():
        unpad = True
        zero_padding_ends = target.klc.zero_padding_ends
    else:
        unpad = False
        zero_padding_ends = None

    statistic_full, stat_mask_full = get_shift_reduced_full_stat_snr(
        statistics_full=target.klc.statistics_full,
        stat_vars_full=target.klc.stat_vars_full,
        stat_masks_full=target.klc.stat_masks_full, temp_index=best_temp_folded,
        indices_of_templates=indices_of_templates, shift_reduction='average',
        renormalize=False, return_mask=True)

    ylabel_wh_flux = 'Whitened binned flux'
    ylabel_snr = 'Single-transit statistic\nin units of $SNR$'
    with SubplotsWrapper(nrows=2, ncols=2, sharex=True, sharey=True, gridspec_kw=Subplots_gridspec_kw_2_line_title, save_fig=save_fig,
                         no_show=no_show, figsize=figsize, target_id=target.target_id, filename_postfix=filename_postfix,
                         plot_output_location=plot_output_location, grid=True, zero_lines=False) as (fig, axs):
        for ax_row_ind, ylabel in [[0, ylabel_wh_flux], [1, ylabel_snr]]:
            ax_row = axs[ax_row_ind]
            for i_ax, transit_times, ev_od_label in [[0, even_transit_times, 'Even'], [1, odd_transit_times, 'Odd']]:
                if ylabel == ylabel_wh_flux:
                    times_near_transit_pieces, data_near_transit_pieces, mask_near_transit_pieces, _ = (
                        get_near_transit_pieces_of_whitened_flux(
                            transit_times=transit_times, times=target.klc.times,
                            whitened_fluxes_td=target.klc.whitened_fluxes_td,
                            masks=target.klc.big_hole_masks, time_wing_days=time_wing_days,
                            unpad=unpad, zero_padding_ends=zero_padding_ends, dt_days=dt_days))
                else:
                    times_near_transit_pieces, data_near_transit_pieces, mask_near_transit_pieces, _ = (
                        get_near_transit_pieces_from_full_time(
                            transit_times=transit_times, time_full=target.klc.time_full, data_full=statistic_full,
                            mask_full=stat_mask_full, time_wing_days=time_wing_days, peak_is_min=True, return_mask=True))
                # ax = axs[i_ax]
                ax = ax_row[i_ax]
                col = (0.55, 0.55, 0.55)
                if len(times_near_transit_pieces) > 0:
                    for i in range(len(times_near_transit_pieces)):
                        ax.plot(times_near_transit_pieces[i][mask_near_transit_pieces[i]] - transit_times[i],
                                data_near_transit_pieces[i][mask_near_transit_pieces[i]], linestyle='',
                                marker='o', markersize=markersize_folds, color=col, zorder=4)
                    if len(times_near_transit_pieces) > 1:
                        bins_time, bins_flux, bins_counts, folds_times, folds_fluxes, bins_std = fold_fluxes(
                            times=copy.deepcopy(times_near_transit_pieces),
                            fluxes=copy.deepcopy(data_near_transit_pieces),
                            period=2 * target.best_per,
                            first_time=transit_times[0],
                            masks=copy.deepcopy(mask_near_transit_pieces),
                            binwid=binwid, dt=dt_days, means=None, template=None, return_std=True,
                            normalize_by_counts=True)
                    else:
                        bins_time = times_near_transit_pieces[0] - transit_times[0]
                        bins_flux = times_near_transit_pieces[0]
                        bins_counts = (~np.isclose(bins_flux, 0)).astype(int)
                        bins_std = np.zeros_like(bins_flux)
                    col = 'k'
                    ax.errorbar(bins_time[bins_counts > 0], bins_flux[bins_counts > 0], yerr=bins_std[bins_counts > 0],
                                xerr=None, markersize=markersize_bins, fmt='o', markerfacecolor=col, markeredgecolor=col,
                                ecolor=col, elinewidth=linewidth, capsize=None, barsabove=False, lolims=False, uplims=False,
                                xlolims=False, xuplims=False, errorevery=1, capthick=None,
                                label=None, zorder=10)
                    if best_ampls is not None:
                        if ylabel == ylabel_wh_flux:
                            n_time_templ = int(np.round(2 * target.best_per / dt_days))
                            averaged_template, time_template = get_average_whitened_template(
                                whitened_templates_td=target.klc.whitened_templates_td,
                                temp_number=best_temp_folded, n_time_templ=n_time_templ,
                                dt_days=dt_days, indices_of_templates=indices_of_templates)
                            # Interpolating the best averaged template
                            averaged_template = - averaged_template / np.linalg.norm(averaged_template)
                            averaged_template = np.interp(bins_time, time_template, averaged_template)
                            # averaged_template[bins_counts > 0] = averaged_template[bins_counts > 0] / (bins_counts[bins_counts > 0])**0.5
                            # averaged_template[bins_counts == 0] = 0
                            ax.plot(bins_time, averaged_template * best_ampls[i_ax],
                                    color='r', linewidth=linewidth, label='Template with\nhigh-pass fit amplitude')
                ax.set_xlim(-time_wing_days, time_wing_days)
                ax.axhline(0, linewidth=Linewidth_zero_level, color='k', zorder=0)
                # ax.legend(loc='upper left')
                axs[-1, i_ax].set_xlabel('Time from transit, days')
                axs[0, i_ax].set_title(ev_od_label + ' transits')
            ax_row[0].set_ylabel(ylabel)
        axs[0, 1].legend(loc='upper right', fontsize=Fontsize_legend)
        fig.suptitle(get_target_title(target) + ', Even-odd folds around transits', y=1)


def plot_significance_histograms(target, fixed_limits=False, limits=None, plot_extrapolation=True,
                                 fontsize_legend=Fontsize_legend,
                                 fontsize_title=Fontsize_title,
                                 fontsize_label=Fontsize_label,
                                 fontsize_ticks=Fontsize_label-1,
                                 dpi=Plots_dpi,
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
        max_plot_mes2 = bins[-1] + 0.1 * bins[0]
    # UNIFORMLY FORMATTED PLOT
    with PlotWrapper(target_id=target.target_id, save_fig=save_fig, no_show=no_show, ax=ax, figsize=figsize, dpi=dpi,
                     filename_postfix=filename_postfix, plot_output_location=plot_output_location, gridspec_kw=None) as (fig, axx):

        # --------------------------------------------
        # PLOTTING TRIGGER
        axx.axvline(best_score, color='k', linewidth=linewidth,
                    label=(' ' * 13 + r'$SNR^2$=' + (str(np.round(np.abs(best_score), 1))
                                                     if best_score is not None else 'None')
                           + ',\n' + 'Trigger: period=' + ((str(int(np.round(target.best_per, 0))) + ' d')
                                                           if best_score is not None else 'None')
                           + ',\n' + ' ' * 13 + 'duration='
                           + ((str(int(np.round(target.templates_durations_hours[target.best_temp], 0))) + ' h')
                              if best_score is not None else 'None')))

        # --------------------------------------------
        # PLOTTING HISTOGRAMS
        if distr_score_inj_all is not None:
            axx.stairs(distr_score_inj_all, bins, label='All injections in \nrelative period range', linewidth=linewidth, color=def_cols[0],
                       linestyle='--', zorder=10)  # after density correction factor, get per one star
            axx.stairs(distr_score_inj_success, bins, label='Success injections',
                       linewidth=linewidth, color=def_cols[0], linestyle='-', zorder=10)
            # -> after density correction factor, get per one star
        axx.stairs(distr_score_h0, bins, label='Background (scrambling)', linewidth=linewidth, color=def_cols[1], zorder=15)

        # --------------------------------------------
        # PLOTTING CLEAN
        if use_clean:
            if use_integral:
                distr_score_h0_clean = target.distr_int_score_h0_clean
            else:
                distr_score_h0_clean = target.distr_stat_h0_clean
            axx.stairs(distr_score_h0_clean, bins, label='Background, no trigger',
                       linewidth=linewidth*0.9, color=def_cols[3], linestyle=(0, (1, 1)), zorder=16)

        # --------------------------------------------
        # PLOTTING EXTRAPOLATION
        if plot_extrapolation and use_integral:
            # MAKE EXTRAPOLATION
            from auxiliary_functions import chi2_1
            x_extra = np.linspace(target.extrapolation_limit_int_score_h0_hist, target.best_int_score + 1, 10)
            extrap_density = target.background_normalization_coef * target.extrapolation_amplitude_int_score_h0_hist * chi2_1(
                x2=x_extra, coef=target.extrapolation_coef_int_score_h0_hist)  # , stat2_limit=target.extrapolation_limit_int_score_h0_hist)
            label = 'Extrapolation'
            if target.false_alarm_prob_trigger is not None:
                label += ',\nFAP=' + f"{target.false_alarm_prob_trigger:.2e}"
            axx.plot(x_extra, extrap_density, linestyle=(0, (5, 1, 1, 1)),
                     label=label, linewidth=linewidth * 0.7, color=def_cols[1], zorder=10)

            # if not hasattr(target, 'extrapolation_coefs_int_score_h0_hist') or target.extrapolation_coefs_int_score_h0_hist is None:
            #     from auxiliary_functions import get_histogram_extrapolation_coefs_in_log
            #     target.extrapolation_coefs_int_score_h0_hist = get_histogram_extrapolation_coefs_in_log(
            #             target.distr_int_score_h0, target.bins_int_score_histogram)
            # coefs = target.extrapolation_coefs_int_score_h0_hist
            # # determine intersection with horizontal axis in log scale:
            # bin_extra_max = (np.log(np.nanmin(target.distr_int_score_h0[target.distr_int_score_h0 != 0])) - coefs[0]) / coefs[1]
            # bin_extra_max = np.nanmax([target.best_int_score + 1, bin_extra_max + 1])
            # bin_extra_min = bins[np.nanargmax(target.distr_int_score_h0) + 1]
            # bin_min_distr = bins[:-1][((~np.isnan(target.distr_int_score_h0)) & (target.distr_int_score_h0 > 0))][0]
            # # bin_extra_min = int((bin_extra_min + bin_extra_max) / 2)
            # bin_extra_min = bin_extra_min + int(np.abs(bin_extra_min - bin_min_distr) / 2)
            # bins_extra = np.linspace(bin_extra_min, bin_extra_max, 10)
            # label = 'Extrapolation'
            # if target.false_alarm_prob_trigger is not None:
            #     label += ',\nFAP=' + f"{target.false_alarm_prob_trigger:.2e}"
            # axx.plot(bins_extra, np.exp(bins_extra * coefs[1] + coefs[0]), linestyle=(0, (5, 1, 1, 1)),
            #          label=label, linewidth=linewidth*0.7, color=def_cols[1], zorder=10)

        # --------------------------------------------
        # FORMATTING
        if fontsize_legend is not None:
            axx.legend(loc='upper right', fontsize=fontsize_legend)
        else:
            axx.legend(loc='upper right')

        title = 'KIC' + target.target_id
        if target.injections_period_range is not None:
            title = (title + ', periods ['
                     + str(np.round(target.injections_period_range[0], 1)) + '-'
                     + str(np.round(target.injections_period_range[1], 1))
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
        axx.minorticks_off()
        axx.tick_params(axis='x', labelsize=fontsize_ticks)
        axx.tick_params(axis='y', labelsize=fontsize_ticks)
        for spine in axx.spines.values():
            spine.set_zorder(50)

        if fontsize_legend is not None:
            axx.legend(loc=(1.01, 0), fontsize=fontsize_legend)
        else:
            axx.legend(loc=(1.01, 0))


def plot_periodogram_chunks(tce, best_short_periodic_scores_filename_postfix=locs.Best_short_periodic_scores_filename_postfix,
                            best_periodic_scores_filename_postfix=locs.Best_periodic_scores_filename_postfix,
                            save_fig=Save_fig, no_show=No_show, plot_output_location=locs.Triggers_plots_location,
                            figsize=Figsize_periodogram, filename_postfix=locs.Periodogram_chunks_plot_filename_postfix,
                            include_short_periodicity=True):
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

    if include_short_periodicity:
        res_array = tce.get_periodicity_results(short_periodicity=True,
                                                best_short_periodic_scores_filename_postfix=best_short_periodic_scores_filename_postfix,
                                                return_lists=True, set_attributes=False)
        short_int_scores = res_array[Best_per_results_index_best_integral]
        short_stats = res_array[Best_per_results_index_stats]
        short_pers = res_array[Best_per_results_index_per]
        short_gauss_stats = res_array[Best_per_results_index_stats_gauss]
        short_ftts = res_array[Best_per_results_index_ftt]
        short_temps = res_array[Best_per_results_index_template]
    else:
        short_int_scores = np.array([])
        short_stats = np.array([])
        short_pers = np.array([])
        short_gauss_stats = np.array([])
        short_ftts = np.array([])
        short_temps = np.array([])


    with SubplotsWrapper(nrows=3, ncols=1, sharex=True, sharey=False, gridspec_kw=Subplots_gridspec_kw,
                         save_fig=save_fig, no_show=no_show,
                         figsize=figsize, target_id=tce.target_id, filename_postfix=filename_postfix,
                         plot_output_location=plot_output_location, grid=True, zero_lines=False) as (fig, axs):
        lw = 1.5
        lw_thin = 1
        lw_vline = 0.8
        xticks = np.arange(50, 520, 25)
        xmin = np.nanmin(pers) - np.nanmin(pers) * 0.1
        xmax = np.nanmax(pers) + np.nanmin(pers) * 0.1

        ax = axs[0]
        ax.plot(pers, int_scores, color=def_cols[0], linewidth=lw, zorder=4, label='Best scores')
        if include_short_periodicity:
            ax.plot(short_pers, short_int_scores, color=def_cols[0], linewidth=lw, zorder=4)
        ax.plot(pers, stats, color=def_cols[1], linewidth=lw_thin, label='Best statistics')
        if include_short_periodicity:
            ax.plot(short_pers, short_stats, color=def_cols[1], linewidth=lw_thin)
        ax.plot(pers, gauss_stats, color=def_cols[2], linewidth=lw_thin, label='Best gaussianized statistics')
        if include_short_periodicity:
            ax.plot(short_pers, short_gauss_stats, color=def_cols[2], linewidth=1)
        ax.axvline(tce.best_per, zorder=0, color='r', linewidth=lw_vline)
        ax.set_xticks(xticks)
        ax.set_xlabel('Period, days')
        ax.set_ylabel('Statistic, $SNR^2$')
        ax.set_xlim(xmin, xmax)
        ax.axvline(Per_min_candidate, zorder=0, color='k', linewidth=lw_vline, linestyle='--', label='Min period candidate')
        ax.legend(loc='best')

        ax = axs[1]
        durs_plot = [Durations[int(temp)] if not np.isnan(temp) else np.nan for temp in temps]
        ax.plot(pers, durs_plot, color=def_cols[0], linewidth=lw, zorder=2, label='Best templates')
        durs_plot = [Durations[int(temp)] if not np.isnan(temp) else np.nan for temp in short_temps]
        if include_short_periodicity:
            ax.plot(short_pers, durs_plot, color=def_cols[0], linewidth=2, zorder=100)
        ax.axvline(tce.best_per, zorder=0, color='r', linewidth=lw_vline, label='Trigger period')
        ax.set_xticks(xticks)
        ax.set_xlabel('Period, days')
        ax.set_ylabel('Template Duration, days')
        ax.set_xlim(xmin, xmax)
        ax.axvline(Per_min_candidate, zorder=0, color='k', linewidth=lw_vline, linestyle='--', label='Min period candidate')
        ax.legend(loc='best')

        ax = axs[2]
        if include_short_periodicity:
            ax.plot(pers, (ftts - np.nanmin(short_ftts)) / pers, color=def_cols[0], linewidth=lw, zorder=2,
                    label='Best first transit phases')
            ax.plot(short_pers, (short_ftts - np.nanmin(short_ftts)) / short_pers, color=def_cols[0], linewidth=2, zorder=100)
        else:
            ax.plot(pers, (ftts - np.nanmin(ftts)) / pers, color=def_cols[0], linewidth=lw, zorder=2,
                    label='Best first transit phases')
        ax.axvline(tce.best_per, zorder=0, color='r', linewidth=lw_vline)
        ax.set_xlabel('Period, days')
        ax.set_xticks(xticks)
        ax.set_ylabel('First transit phase')
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(-0.1, 1.2)
        ax.axvline(Per_min_candidate, zorder=0, color='k', linewidth=lw_vline, linestyle='--', label='Min period candidate')
        ax.legend(loc='best')
        fig.suptitle('Periodogram of maximized scores\n' + get_target_title(tce), y=1)


def plot_folding_residuals_histograms(tce=None, target_id=None, output_location=None,
                                      residuals_flux_binned_fd_chi2=None, mf_scores_resid=None, chi_values_real=None,
                                      temp_number=None, fn_fold_veto=None, p_val_resid_time=None, p_val_resid_fft=None, p_val_mf=None,
                                      ax=None, save_fig=Save_fig, no_show=No_show, plot_output_location=locs.Triggers_plots_location,
                                      figsize=Figsize_residuals_hist,
                                      filename_postfix=locs.Residuals_fold_veto_histogram_plot_filename_postfix,
                                      fontsize_legend=Fontsize_legend, fontsize_title=Fontsize_title, fontsize_label=Fontsize_label,
                                      linewidth=Linewidth):
    if residuals_flux_binned_fd_chi2 is None or mf_scores_resid is None or chi_values_real is None:
        if tce is not None:
            target_id = tce.target_id
            temp_number = tce.best_temp_folded
            p_val_resid_time = tce.p_value_fold_residuals_fft
            p_val_resid_fft = tce.p_value_fold_residuals_fft
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
                                         filename_postfix=locs.Fold_residuals_flux_fd_chi2_filename_postfix,
                                         insert_after_postfix=None, file_format='npy')
        residuals_flux_binned_fd_chi2 = np.load(file_path, allow_pickle=True)
        file_path = get_saving_file_path(target_id=target_id, output_location=output_location,
                                         filename_postfix=locs.Mf_scores_residuals_filename_postfix,
                                         insert_after_postfix=None, file_format='npy')
        mf_scores_resid = np.load(file_path, allow_pickle=True)
        file_path = get_saving_file_path(target_id=target_id, output_location=output_location,
                                         filename_postfix=locs.Fold_residuals_flux_td_chi_filename_postfix,
                                         insert_after_postfix=None,
                                         file_format='npy')
        chi_values_real = np.load(file_path, allow_pickle=True)

    from scipy.stats import chi2
    with (PlotWrapper(target_id=target_id, save_fig=save_fig, no_show=no_show, ax=ax, figsize=figsize, gridspec_kw=Fig_gridspec_kw,
                      filename_postfix=filename_postfix, plot_output_location=plot_output_location,
                      which_grid='major') as (fig, axx)):
        lw = linewidth
        mf_scores_resid_flatten_2 = (mf_scores_resid.flatten()) ** 2
        limit_line_mf = chi2(df=1, scale=1).isf(1. - (1 - fn_fold_veto) ** (1 / len(mf_scores_resid_flatten_2)))
        # limit_line_resid = chi2(df=1, scale=1).isf(1. - (1 - fn_fold_veto) ** (1 / len(residuals_flux_binned_fd_chi2)))
        limit_line_resid = chi2(df=2, scale=1).isf(1. - (1 - fn_fold_veto) ** (1 / len(residuals_flux_binned_fd_chi2)))

        bins = np.linspace(0, max(np.max(mf_scores_resid_flatten_2), np.max(residuals_flux_binned_fd_chi2)), 100)
        axx.hist(mf_scores_resid_flatten_2, bins=bins, histtype='stepfilled', density=True,
                 label=('Matched filtering \nof residual flux'
                        + (',\ntemp ' + str(temp_number) if temp_number is not None else '')
                        + (',\n' + str(temp_number) if temp_number is None else ', ')
                        + ('p=' + str(np.round(p_val_mf, 3)) if p_val_mf is not None else '')),
                 linewidth=lw/4, color=def_cols[0], alpha=0.4, zorder=0)
        axx.hist(residuals_flux_binned_fd_chi2, bins=bins, histtype='step', density=True,
                 label='Residual flux\n(Fourier)' + (', p=' + str(np.round(p_val_resid_fft, 3)) if p_val_resid_fft is not None else ''),
                 linewidth=lw, color=def_cols[1], zorder=7)
        axx.hist(chi_values_real**2, bins=bins, histtype='step', density=True,
                 label='Residual flux\n(Time)' + (', p=' + str(np.round(p_val_resid_time, 3)) if p_val_resid_time is not None else ''),
                 linewidth=lw/2, linestyle='-', color=def_cols[5], zorder=10)

        xx = np.linspace(0, min(np.max(mf_scores_resid_flatten_2), limit_line_mf), 100)
        axx.plot(xx, chi2(df=1, scale=1).pdf(xx), 'g--', linewidth=lw, zorder=2,
                 label=r'$\chi^2$'+' (matched filtering)')

        xx = np.linspace(0, min(np.max(residuals_flux_binned_fd_chi2), limit_line_resid), 100)
        axx.plot(xx, chi2(df=2, scale=1).pdf(xx), 'r--', linewidth=lw, zorder=2,
                 label=r'$\chi^2$'+' (residuals Fourier)')

        axx.axvline(limit_line_mf, color=def_cols[0], zorder=1,
                    linestyle='-', linewidth=lw, label='Threshold')
        axx.axvline(limit_line_resid, color=def_cols[1], zorder=1,
                    linestyle='-', linewidth=lw, label='Threshold')
        axx.set_yscale('log')
        axx.set_xlabel('Residual score squared', fontsize=fontsize_label)
        axx.set_ylabel('Probability density', fontsize=fontsize_label)
        title = 'Residuals of the folded flux distribution KIC' + target_id
        fig.suptitle(title, fontsize=fontsize_title, y=1)
        plt.legend(fontsize=fontsize_legend, loc=(1.01, 0.))


def plot_fold_veto_parts(tce, save_fig=Save_fig, no_show=No_show, plot_output_location=locs.Triggers_plots_location,
                         figsize=Figsize_veto_parts_fold,
                         filename_postfix=locs.Veto_parts_fold_plot_filename_postfix,
                         linewidth=Linewidth_coarse,
                         linewidth_zero_level=Linewidth_zero_level,
                         markersize=Markersize_bins_normal,
                         time_wing=Zoom_wing_folded_plots):
    file_path = get_saving_file_path(target_id=tce.target_id, output_location=tce.output_location,
                                     filename_postfix=locs.Folded_binned_flux_template_filename_postfix,
                                     insert_after_postfix=None,
                                     file_format='npy')
    flux_normalized, template_wh_td = np.load(file_path, allow_pickle=True)
    wing_time_bins = len(flux_normalized) // 2
    bins_time = np.linspace(-wing_time_bins * tce.dt_days, wing_time_bins * tce.dt_days, len(flux_normalized))

    gridspec_kw = Subplots_gridspec_kw
    if len(tce.veto_fold_parts_results) > 1:
        figsize_1 = figsize[1] * len(tce.veto_fold_parts_results)
    else:
        figsize_1 = figsize[1] + 1.
        # gridspec_kw['top'] = 0.87
    with SubplotsWrapper(nrows=len(tce.veto_fold_parts_results), ncols=1, sharex=True, sharey=False, gridspec_kw=gridspec_kw,
                         save_fig=save_fig, no_show=no_show, figsize=(figsize[0], figsize_1), target_id=tce.target_id,
                         filename_postfix=filename_postfix, plot_output_location=plot_output_location, grid=True, zero_lines=False
                         ) as (fig, axs):
        if len(tce.veto_fold_parts_results) == 1:
            axs = [axs]

        for i in range(len(tce.veto_fold_parts_results)):
            ax = axs[i]
            n_parts = tce.veto_fold_parts_results[i][0]
            ampl_full = tce.veto_fold_parts_amplitudes[i][0]
            # ampl_std_full = tce.veto_fold_parts_amplitudes[i][1]
            ampls_parts = tce.veto_fold_parts_amplitudes[i][2]
            ampls_stds_parts = tce.veto_fold_parts_amplitudes[i][3]
            templates_parts = ut.get_template_by_parts_snr(template_wh_td=template_wh_td, n_parts=n_parts,
                                                           min_bins_part=Min_len_part_bins_veto_parts_fold)

            # --------
            ax.plot(bins_time, flux_normalized, 'o', color='k', markersize=markersize, label='Folded binned flux')
            ax.plot(bins_time, ampl_full * template_wh_td, linewidth=linewidth, color='k', label='Best-fit template')
            for i_temp, temp in enumerate(templates_parts):
                temp_plot = np.array(temp)
                temp_plot[temp_plot == 0] = np.nan
                ax.plot(bins_time, ampls_parts[i_temp] * temp_plot, label='Template parts' if i_temp == 0 else None,
                        color=def_cols[i_temp])
                ax.fill_between(bins_time,
                                ((ampls_parts[i_temp] - ampls_stds_parts[i_temp]) * temp_plot),
                                ((ampls_parts[i_temp] + ampls_stds_parts[i_temp]) * temp_plot),
                                alpha=0.5, color=def_cols[i_temp % 10])
            ax.set_xlabel('Time from the transit center, days')
            ax.set_title(r'$p$' + '-values: $p_{total}=$' + str(np.round(tce.veto_fold_parts_results[i][1], 3))
                         + ', $p_{worst}=$' + str(np.round(tce.veto_fold_parts_results[i][2], 3)),
                         y=0.9, bbox={'facecolor': 'white', 'edgecolor': 'None', 'boxstyle': 'square,pad=0.11'},
                         fontsize=Fontsize)
            ax.legend()
            ax.axhline(0, color='k', linewidth=linewidth_zero_level)
            ax.set_xlim(-time_wing, time_wing)
            ax.grid()
        axs[len(axs)//2].set_ylabel('Normalized blued flux')
        fig.suptitle(get_target_title(tce) + ', Consistency of amplitudes of parts of the folded ephemeris', y=1)


def plot_injections_corners(target, save_fig=Save_fig, no_show=No_show, plot_output_location=locs.Triggers_plots_location,
                            dpi=Plots_dpi, figsize=Figsize_corner, tight_layout=True, lw_1d=1.5, lw_contour=1.,
                            alpha_points=0.8, smooth=0.1, level_sigmas=(1, 2, 3), axes_scale='linear',
                            n_bins=16, ms_points=2, plot_points_where_scarce=True):
    import corner
    # We cannot pass linewidth for the contours in the corner, therefore we update the default, and then will restore
    old_lw = plt.rcParams.get('lines.linewidth')  # saving to return in the end
    plt.rcParams.update({'lines.linewidth': lw_contour})

    if (not hasattr(target, 'retrieved_int_score_success') or target.retrieved_int_score_success is None
            or not hasattr(target, 'dur_inj_success') or target.dur_inj_success is None
            or not hasattr(target, 'retrieved_template_success') or target.retrieved_template_success is None):
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
        ['dur_retr',  [[Durations[int(tt)] if not np.isnan(tt) else np.nan for tt in target.retrieved_template_success],
                       [Durations[int(tt)] if not np.isnan(tt) else np.nan for tt in target.retrieved_template_bad],
                       r'dur$_{r}$', 'Retrieved duration, days']],
        ['gauss_stat_retr', [target.retrieved_gauss_stat_success, target.retrieved_gauss_stat_bad,
                             r'$(\rho^2_g)_{r}$', 'Gauss. statistic, $SNR^2$']]])

    mask_not_nan_success = np.all([~np.isnan(pars_dict[key][0]).T for key in pars_dict.keys()], axis=0)
    mask_not_nan_bad = np.all([~np.isnan(pars_dict[key][1]).T for key in pars_dict.keys()], axis=0)
    for key in pars_dict.keys():
        pars_dict[key][0] = np.array(pars_dict[key][0])[mask_not_nan_success]
        pars_dict[key][1] = np.array(pars_dict[key][1])[mask_not_nan_bad]

    # Add a little noise because sometimes all the bad scores are from the same place in the sata, then all these
    # values are equal, and then corner plot is not satisfied:
    pars_dict['dur_retr'][1] += np.random.normal(0, 0.0001, len(pars_dict['dur_retr'][1]))
    pars_dict['ftt_retr'][1] += np.random.normal(0, 0.0001, len(pars_dict['ftt_retr'][1]))

    for keys_corner, fig_name_postfix in [
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

        levels = 1.0 - np.exp(-0.5 * np.array(level_sigmas) ** 2)

        fig, ax = plt.subplots(len(keys_corner), len(keys_corner), figsize=figsize, dpi=dpi,
                               gridspec_kw={'top': 0.93, 'bottom': 0.07, 'right': 0.98, 'left': 0.07, 'wspace': 0, 'hspace': 0})
        for a in ax:
            for b in a:
                b.set_facecolor((1, 1, 1, 0))
        fig.subplots_adjust(hspace=0., wspace=0.)

        if plot_points_where_scarce:
            # Plot contours filled with white, and points outside.
            # Contours filled with white are needed to hind points inside.
            for data, color in [[res_success, def_cols[0]], [res_bad, def_cols[1]]]:
                color_grid = np.ones((len(levels) + 1, 4))
                color_grid[0, -1] = 0
                data_kw = dict(alpha=alpha_points, ms=ms_points, color=color)
                contour_colors = [np.concatenate((matplotlib.colors.to_rgb(color), [0])) for _ in range(len(levels))]
                contour_kw = dict(colors=contour_colors)
                contourf_kw = dict(colors=color_grid)
                hist_kwargs_s = dict(density=True, color=color, linewidth=lw_1d)
                kwargs = dict(fig=fig,
                              plot_datapoints=True,
                              plot_contours=True,
                              fill_contours=True,
                              no_fill_contours=True,
                              plot_density=False,
                              bins=n_bins, axes_scale=axes_scale,
                              labels=labels, levels=levels, quiet=True, alpha=alpha_points, ms=ms_points, smooth=smooth)
                corner.corner(data, color=color, data_kwargs=data_kw, hist_kwargs=hist_kwargs_s,
                              contour_kwargs=contour_kw,
                              contourf_kwargs=contourf_kw, **kwargs)

        # Now plot for the transparent blobs
        for data, color in [[res_success, def_cols[0]], [res_bad, def_cols[1]]]:
            color_grid = get_levels_grid(rgb_color=matplotlib.colors.to_rgb(color), n_levels=len(levels), alpha=0.5, last_white=False,
                                         end_val=0.8, gradiend_alpha=True, alpha_min=0.1)
            data_kw = dict(alpha=alpha_points, ms=ms_points, color=color)
            contour_colors = [np.concatenate((matplotlib.colors.to_rgb(color), [1])) for _ in range(len(levels))]
            contour_kw = dict(colors=contour_colors)
            contourf_kw = dict(colors=color_grid)
            hist_kwargs_s = dict(density=True, color=color, linewidth=lw_1d)
            kwargs = dict(fig=fig,
                          plot_datapoints=False,
                          plot_contours=True,
                          fill_contours=True,
                          no_fill_contours=False,
                          plot_density=False,
                          bins=n_bins, axes_scale=axes_scale,
                          labels=labels, levels=levels, quiet=True, alpha=alpha_points, ms=ms_points, smooth=smooth)
            corner.corner(data, color=color, data_kwargs=data_kw, hist_kwargs=hist_kwargs_s,
                          contour_kwargs=contour_kw,
                          contourf_kwargs=contourf_kw, **kwargs)

        fig.suptitle('Injection-recovery parameters \n' + get_target_title(target, score=None, per=None, temp=None))
        # axes = np.array(fig.axes).reshape((len(keys_corner), len(keys_corner)))

        for ax in fig.axes:
            ax.grid(zorder=0, color='k', alpha=0.3, linewidth=0.5)
        ax = fig.add_axes((0.5, 0.5, 0.4, 0.4), frame_on=False)
        ax.axis('off')
        ax.plot([], linewidth=lw_1d, color=def_cols[0], label='Success, ' + str(len(res_success)))
        ax.plot([], linewidth=lw_1d, color=def_cols[1], label='Bad, ' + str(len(res_bad)))
        ax.legend(loc='upper right', fontsize=Fontsize_legend)

        if save_fig:
            tid_folder = get_correct_tid_folder(target_id=target.target_id, location=plot_output_location,
                                                allow_creation=False, with_prefix_folder=False)
            file_saving_path = get_saving_file_path(output_location=tid_folder,
                                                    target_id=target.target_id,
                                                    filename_postfix=fig_name_postfix,
                                                    file_format='png')
            if tight_layout:
                fig.tight_layout()
            fig.savefig(file_saving_path, dpi=dpi)  # , bbox_inches='tight')

        if no_show is None:
            pass
        elif no_show:
            plt.clf()
            plt.close(fig)
        else:
            plt.show()
    plt.rcParams.update({'lines.linewidth': old_lw})  # restoring as it was before


def plot_injections_radii_durations_score(target, save_fig=Save_fig, no_show=No_show, plot_output_location=locs.Triggers_plots_location,
                                          figsize=Figsize_inj_radii_dur_scatter,
                                          filename_postfix=locs.Injections_radii_durations_score_plot_filename_postfix):
    if not hasattr(target, 'retrieved_int_score_success') or target.retrieved_int_score_success is None:
        target.retrieve_injections()

    color_all = (0.6, 0.6, 0.6)
    color_low = def_cols[0]
    color_high = def_cols[1]
    color_bad = (0.8, 0.3, 0.3)
    part_points_highlight = 0.1
    ms_small = 1
    ms_big = 3

    with SubplotsWrapper(nrows=1, ncols=2, sharex=False, sharey=True, gridspec_kw=Subplots_gridspec_kw_2_line_title, save_fig=save_fig,
                         no_show=no_show, figsize=figsize, target_id=target.target_id, filename_postfix=filename_postfix,
                         plot_output_location=plot_output_location, grid=True, zero_lines=False) as (fig, axs):
        # ---------------------------------------------------------------
        # Plot int scores of durations
        ax = axs[0]
        durs = np.array(target.dur_inj_bad)
        scors = np.array(target.retrieved_int_score_bad)
        ax.scatter(durs, scors, s=ms_small, color=color_bad, label='Bad')
        durs = np.array(target.dur_inj_success)
        scors = np.array(target.retrieved_int_score_success)
        ax.scatter(durs, scors, s=ms_small, color=color_all, label='All success')

        rps = np.array(target.rp_inj_success) * target.stellar_radius * pp.R_sun_R_e
        rps_sorted = np.sort(rps)
        part = np.arange(len(rps_sorted)) / len(rps_sorted)
        rp_low = np.round(rps_sorted[np.searchsorted(part, part_points_highlight)], 1)
        rp_high = np.round(rps_sorted[np.searchsorted(part, 1 - part_points_highlight)], 1)

        inds_selection = (rps <= rp_low)
        ax.scatter(durs[inds_selection], scors[inds_selection], s=ms_big, color=color_low, label='$r_{p}/r_{e}$<' + str(rp_low))
        inds_selection = (rps >= rp_high)
        ax.scatter(durs[inds_selection], scors[inds_selection], s=ms_big, color=color_high, label='$r_{p}/r_{e}$>' + str(rp_high))

        ax.axhline(50, color='k', linewidth=Linewidth_grid)
        ax.axhline(70, color='k', linewidth=Linewidth_grid)
        ax.set_xlabel('dur$_{inj}$, days')
        ax.set_ylabel(r'$\eta$, SNR$^2$')
        ax.set_xticks(np.arange(0, 6, 0.5))
        ax.grid()
        ax.legend()

        # ---------------------------------------------------------------
        # Plot int scores of radii
        ax = axs[1]

        rps = np.array(target.rp_inj_bad) * target.stellar_radius * pp.R_sun_R_e
        scors = np.array(target.retrieved_int_score_bad)
        ax.scatter(rps, scors, s=ms_small, color=color_bad, label='Bad')
        rps = np.array(target.rp_inj_success) * target.stellar_radius * pp.R_sun_R_e
        scors = np.array(target.retrieved_int_score_success)
        ax.scatter(rps, scors, s=ms_small, color=color_all, label='All success')

        durs = np.array(target.dur_inj_success)
        durs_sorted = np.sort(durs)
        part = np.arange(len(durs_sorted)) / len(durs_sorted)
        dur_low = np.ceil(durs_sorted[np.searchsorted(part, part_points_highlight)]*1000)/1000  # ceil with decimals
        dur_high = np.round(durs_sorted[np.searchsorted(part, 1 - part_points_highlight)], 3)
        inds_selection = (durs <= dur_low)
        ax.scatter(rps[inds_selection], scors[inds_selection], s=ms_big, color=color_low, label=r'$dur$' + ' < ' + str(dur_low) + ' d')
        inds_selection = (durs >= dur_high)
        ax.scatter(rps[inds_selection], scors[inds_selection], s=ms_big, color=color_high, label=r'$dur$' + ' > ' + str(dur_high) + ' d')

        ax.axhline(50, color='k', linewidth=Linewidth_grid)
        ax.axhline(70, color='k', linewidth=Linewidth_grid)
        ax.set_xlabel('$r_{p}/r_{e}$')
        # ax.set_ylabel(r'$\eta$, SNR$^2$')
        ax.set_xticks(np.arange(1, 21, 1))
        ax.grid(zorder=0, linewidth=0.5)
        ax.legend()
        fig.suptitle(get_target_title(target, score=None, per=None, temp=None) + ' Injection-recovery detectability')


def plot_transit_info(tce, save_fig=Save_fig, no_show=No_show, plot_output_location=locs.Triggers_plots_location,
                      figsize=Figsize_transit_info,
                      filename_postfix=locs.Transit_info_plot_filename_postfix,
                      linewidth=Linewidth_thin,
                      linewidth_zero_level=Linewidth_zero_level,
                      markersize=Markersize_transits_large):
    seasons = get_seasons(tce=tce)
    bar_width = tce.best_per / 4
    with PlotWrapper(save_fig=save_fig, no_show=no_show, figsize=figsize, target_id=tce.target_id,
                     filename_postfix=filename_postfix, plot_output_location=plot_output_location,
                     grid=True, gridspec_kw=Fig_gridspec_kw_2) as (fig, ax):

        lw = linewidth
        ax.errorbar(tce.summed_transit_times, tce.transit_ppms, yerr=tce.transit_ppm_stds,
                    marker='o', linestyle='-', color='k', markersize=1,  # label='depth',
                    capsize=3, linewidth=lw, elinewidth=lw, capthick=lw)
        ax2 = ax.twinx()
        ax.set_zorder(2)
        ax2.set_zorder(1)
        ax.set_facecolor('none')
        snrs_plot = np.array(tce.summed_raw_stats)
        snrs_plot[tce.summed_vars > 0] = snrs_plot[tce.summed_vars > 0] / tce.summed_vars[tce.summed_vars > 0] ** 0.5
        snrs_plot[tce.summed_vars == 0] = np.nan
        # ax2.plot(tce.summed_transit_times, snrs_plot,
        #          marker='*', linestyle=':', color='k', linewidth=lw, markersize=1,  # label='snr',
        #          )
        ax.axhline(0, color='k', linewidth=linewidth_zero_level, zorder=2)
        if (~np.isnan(tce.transit_ppms)).any() and (~np.isnan(tce.transit_ppm_stds)).any():
            min_level = np.nanmin(tce.transit_ppms - tce.transit_ppm_stds)
            max_level = np.nanmax(tce.transit_ppms + tce.transit_ppm_stds)
            scale = max_level - min_level
            max_level = max(0, max_level)
        else:
            min_level = -100.
            max_level = 0.
            scale = 100.

        texts = []
        len_texts = []
        for i in range(len(tce.summed_transit_times)):
            len_text = 0
            text = tce.badness_of_transits[i].split()
            len_text += len(text)
            text = '\n'.join(text)
            if tce.overlapping_kois[i] != '':
                items = tce.overlapping_kois[i].split(' ')
                for item in items:
                    if item != '':
                        text += '\n' + 'KOI_' + item[10:]
                        len_text += 1
            if tce.overlapping_villanovas[i] != '':
                items = tce.overlapping_villanovas[i].split(' ')
                for item in items:
                    if item != '':
                        text += '\n' + 'Vil_' + item[10:]
                        len_text += 1
            texts.append(text)
            len_texts.append(len_text)

        max_lines_text = np.max(len_texts)
        wid_level = 0.2
        n_levels = 3 if max_lines_text >= 3 else 2
        max_level_ylim = max_level + scale * 0.2 + scale * max(wid_level * n_levels, max_lines_text * 0.05)

        for i in range(len(tce.summed_transit_times)):
            col = def_cols[int(seasons[i])] if not np.isnan(seasons[i]) else 'k'
            ax.plot([tce.summed_transit_times[i]], [tce.transit_ppms[i]],
                    marker='o', linestyle='', color=col,
                    markersize=markersize, label='Depth' if i == 0 else None, zorder=20)
            text = texts[i]
            if len(text) > 0:
                text = '\n'.join(text.split())
                y_coor = max_level + scale * 0.2 + scale * wid_level * (i % n_levels)
                ax.text(tce.summed_transit_times[i], y_coor,
                        text, fontsize=9, color='k', zorder=30, ha='center', va='bottom')
                ax.plot([tce.summed_transit_times[i]], [tce.transit_ppms[i]],
                        marker='x', linestyle='', color='k', markersize=markersize, zorder=30)
                y_coor_0 = tce.transit_ppms[i] if not np.isnan(tce.transit_ppms[i]) else 0
                ax.plot([tce.summed_transit_times[i], tce.summed_transit_times[i]],
                        [y_coor_0, y_coor],
                        linestyle=':', color='k', linewidth=linewidth, zorder=3)
                ax2.bar([tce.summed_transit_times[i]], [-snrs_plot[i]], color='k', zorder=3, alpha=0.2, width=bar_width)
            else:
                ax2.bar([tce.summed_transit_times[i]], [-snrs_plot[i]], color=col, zorder=3, alpha=0.3, width=bar_width)

        ax.set_ylim(min_level - scale * 0.2, max_level_ylim)
        ax.set_xlim(tce.summed_transit_times[0] - tce.best_per, tce.summed_transit_times[-1] + tce.best_per)
        ax.set_xlabel('Transit time, days')
        ax.set_ylabel('Depth of transits, ppm')
        ax2.set_ylabel('SES of transits, SNR')
        ax.axhline(0, linewidth=linewidth_zero_level, color='k')
        ax.bar([], [],  color=def_cols[0], alpha=0.3, width=bar_width, label='SNR')
        ax.legend(loc='best')

        fig.suptitle('Transit depths, SNRs and flags \n' + get_target_title(tce), y=1)


def plot_bg_star_pixels_ppm(tce, save_fig=Save_fig, no_show=No_show, plot_output_location=locs.Triggers_plots_location,
                            figsize=Figsize_star_bg_pixels_ppm,
                            filename_postfix=locs.Star_bg_pixels_ppm_plot_filename_postfix,
                            linewidth=Linewidth_medium, linewidth_thin=Linewidth_thin,
                            linewidth_zero_level=Linewidth_zero_level,
                            markersize=Markersize):
    seasons = get_seasons(tce=tce)
    with (SubplotsWrapper(nrows=1, ncols=2, sharex=False, sharey=False, gridspec_kw=None,
                          save_fig=save_fig, no_show=no_show, figsize=figsize, target_id=tce.target_id, filename_postfix=filename_postfix,
                          plot_output_location=plot_output_location, grid=True, zero_lines=False) as (fig, axs)):

        ax = axs[0]
        lw = linewidth
        lwt = linewidth_thin
        if tce.transit_ppms is not None:
            ax.errorbar(tce.summed_transit_times + 0.025 * tce.best_per, tce.transit_ppms, yerr=tce.transit_ppm_stds,
                        marker='o', linestyle=':', color=def_cols[2], label='Kepler aperture', markersize=markersize,
                        capsize=3, linewidth=lwt, elinewidth=lw, capthick=lw)

        ax.errorbar(tce.summed_transit_times, tce.star_pixels_ppms, yerr=tce.star_pixels_stds_of_ppms,
                    marker='o', linestyle=':', color=def_cols[0], label='Central pixels', markersize=markersize,
                    capsize=3, linewidth=lwt, elinewidth=lw, capthick=lw)
        ax.errorbar(tce.summed_transit_times+0.05 * tce.best_per, tce.bg_pixels_ppms, yerr=tce.bg_pixels_stds_of_ppms,
                    marker='o', linestyle=':', color=def_cols[1], label='BG pixels', markersize=markersize,
                    capsize=3, linewidth=lwt, elinewidth=lw, capthick=lw)

        ax.set_xlabel('Transit time, days')
        ax.set_ylabel('Depth of transits, ppm')
        ax.axhline(0, linewidth=linewidth_zero_level, color='k')
        ax.legend(loc='best')

        ax = axs[1]
        seasons = seasons.astype(float)
        n_points_max = np.max([np.sum(seasons == i) for i in range(4)])
        sh_step = 0.2 / n_points_max
        seasons_plot = np.array(seasons)
        for i in range(4):
            n_points = np.sum(seasons == i)
            seasons_plot[seasons == i] += sh_step * np.arange(-n_points / 2., n_points / 2 + 1., 1)[:n_points]
            # seasons_plot[seasons == i] += (i % 2) * sh_step / 3
        ax.errorbar(tce.star_pixels_ppms, seasons_plot, xerr=tce.star_pixels_stds_of_ppms,
                    marker='o', linestyle='', markersize=markersize, label='Central pixels', linewidth=lwt,
                    capsize=3, elinewidth=lw, capthick=lw)
        seasons_plot += 0.25
        ax.errorbar(tce.bg_pixels_ppms, seasons_plot, xerr=tce.bg_pixels_stds_of_ppms,
                    marker='o', linestyle='', markersize=markersize, label='BG pixels', linewidth=lwt,
                    capsize=3, elinewidth=lw, capthick=lw)
        ax.set_yticks([0, 1, 2, 3])
        ax.set_xlabel('Depth of transits, ppm')
        ax.set_ylabel('Season')
        ax.set_ylim(-0.5, 3.5)
        ax.legend(loc='best')

        fig.suptitle(get_target_title(tce) + '\nTransit depths of star and background pixels comparison '
                     + r'$p$' + '-values of the veto: $p_{total}=$' + (str(np.round(tce.p_value_mean_bg_pixels, 3))
                                                                       if tce.p_value_mean_bg_pixels is not None else 'None')
                     + ', $p_{worst}=$' + (str(np.round(tce.p_value_worst_bg_pixels, 3))
                                           if tce.p_value_worst_bg_pixels is not None else 'None'), y=1)


def plot_bg_star_pixels_stats(tce, edge_gap_time=10, save_fig=Save_fig, no_show=No_show,
                              plot_output_location=locs.Triggers_plots_location, figsize=Figsize_flux_wide,
                              filename_postfix=locs.Star_bg_full_stats_plot_filename_postfix):
    from auxiliary_functions import get_connected_stats_from_file, get_shift_reduced_full_stat_snr
    from tce_parameters import Connected_stats_star_pixels_filename_postfix, Connected_stats_bg_pixels_filename_postfix

    transit_times = tce.summed_transit_times
    temp_number = tce.best_temp
    transit_colors = [def_cols[(int(q)) % 4 + 2] if (not np.isnan(q)) else 'k' for q in tce.transit_orig_quarters]

    filename_postfixes = [Connected_stats_star_pixels_filename_postfix, Connected_stats_bg_pixels_filename_postfix]

    with PlotWrapper(target_id=tce.target_id, save_fig=save_fig, no_show=no_show, figsize=figsize,
                     filename_postfix=filename_postfix, plot_output_location=plot_output_location) as (fig, ax):
        for i_flux, filename_postfix in enumerate(filename_postfixes):
            (time_full, statistics_full, stat_masks_full, stat_vars_full, big_hole_masks_full,
             indices_of_templates) = get_connected_stats_from_file(
                file_path=None, output_location=tce.output_location, target_id=tce.target_id,
                filename_postfix=filename_postfix, return_only_stats=True)
            # ! this is returning only the one template that was processed!
            # Then indices_of_templates are as if it was the template number zero.
            if len(statistics_full) < 4:
                temp_num_local = 0
            else:
                temp_num_local = temp_number

            # Converting to units of SNR and minimizing over the template sub-bin shifts
            statistic_full, stat_mask_full = get_shift_reduced_full_stat_snr(
                statistics_full=statistics_full, stat_vars_full=stat_vars_full, stat_masks_full=stat_masks_full, return_mask=True,
                temp_index=temp_num_local, indices_of_templates=indices_of_templates, shift_reduction='minimize', renormalize=True)

            ax.plot(time_full, statistic_full, color=def_cols[i_flux], alpha=0.7, linewidth=Linewidth_coarse,
                    label='Star pixels' if i_flux == 0 else 'BG pixels')
            plt.plot(time_full[~stat_mask_full], statistic_full[~stat_mask_full], 'o',
                     color='k', markersize=1, label='Masked' if i_flux == 1 else None,)

        if transit_times is not None:
            for i_tt, tt in enumerate(transit_times):
                plt.axvline(tt, color=transit_colors[i_tt], zorder=0, linewidth=4*Linewidth_zero_level, linestyle=':')

        ax.axhline(0, linewidth=Linewidth_zero_level, color='k', zorder=0)
        ax.set_xlim(np.nanmin(time_full) - edge_gap_time, np.nanmax(time_full) + edge_gap_time)
        ax.set_xlabel('time, days')
        ylabel = 'SES statistics [SNR]'
        ax.set_ylabel(ylabel)
        title = get_target_title(tce, score=None, per=None, temp=temp_number)
        ax.set_title(title + '\n Star and background pixels apertures single-transit statistics')
        ax.legend(loc='upper right')


def plot_centroid_full_stats(tce, edge_gap_time=10, save_fig=Save_fig, no_show=No_show,
                             plot_output_location=locs.Triggers_plots_location, figsize=Figsize_flux_wide,
                             filename_postfix=locs.Centroid_full_stats_plot_filename_postfix):
    from auxiliary_functions import get_connected_stats_from_file, get_shift_reduced_full_stat_snr

    transit_times = tce.summed_transit_times
    temp_number = tce.best_temp
    transit_colors = [def_cols[(int(q)) % 4 + 2] if (not np.isnan(q)) else 'k' for q in tce.transit_orig_quarters]

    with PlotWrapper(target_id=tce.target_id, save_fig=save_fig, no_show=no_show, figsize=figsize,
                     filename_postfix=filename_postfix, plot_output_location=plot_output_location) as (fig, ax):

        for i_flux, temp, filename_postfix in [[0, tce.centroid_templates_ra_dec[0], locs.Connected_stats_centroid_x_filename_postfix],
                                               [1, tce.centroid_templates_ra_dec[1], locs.Connected_stats_centroid_y_filename_postfix]]:
            (time_full, statistics_full, stat_masks_full, stat_vars_full, big_hole_masks_full,
             indices_of_templates) = get_connected_stats_from_file(
                file_path=None, output_location=tce.output_location, target_id=tce.target_id,
                filename_postfix=filename_postfix, return_only_stats=True)

            # Converting to units of SNR and minimizing over the template sub-bin shifts
            statistic_full, stat_mask_full = get_shift_reduced_full_stat_snr(
                statistics_full=statistics_full, stat_vars_full=stat_vars_full, stat_masks_full=stat_masks_full, return_mask=True,
                temp_index=temp, indices_of_templates=indices_of_templates, shift_reduction='minimize', renormalize=True)

            ax.plot(time_full, statistic_full, color=def_cols[i_flux], alpha=0.7, linewidth=Linewidth_coarse,
                    label='RA' if i_flux == 0 else 'Dec')
            plt.plot(time_full[~stat_mask_full], statistic_full[~stat_mask_full], 'o',
                     color='k', markersize=1, label='Masked' if i_flux == 1 else None,)

        if transit_times is not None:
            for i_tt, tt in enumerate(transit_times):
                plt.axvline(tt, color=transit_colors[i_tt], zorder=0, linewidth=4*Linewidth_zero_level, linestyle=':')

        ax.axhline(0, linewidth=Linewidth_zero_level, color='k', zorder=0)
        ax.set_xlim(np.nanmin(time_full) - edge_gap_time, np.nanmax(time_full) + edge_gap_time)
        ax.set_xlabel('time, days')
        ylabel = 'SES statistics [SNR]'
        ax.set_ylabel(ylabel)
        title = get_target_title(tce, score=None, per=None, temp=temp_number)
        ax.set_title(title + '\n RA and Dec of centroid single-transit statistics')
        ax.legend(loc='upper right')


def plot_centroids_stat_periodic(tce=None, target_id=None,
                                 ftts_total=None, stats_2_max_periodic_total=None, ftts_coords=None,
                                 stats_2_max_periodic_coords=None, ftt_center=None, max_centroid_snr_2=None,
                                 ftt_window_plotting=Centroid_ftt_wing_days_plotting,
                                 save_fig=Save_fig, no_show=No_show, plot_output_location=locs.Triggers_plots_location,
                                 figsize=Figsize_centroid_periodic,
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
    with PlotWrapper(target_id=target_id, save_fig=save_fig, no_show=no_show, ax=None, figsize=figsize,
                     filename_postfix=filename_postfix, plot_output_location=plot_output_location,
                     grid=True) as (fig, ax):
        ax.plot(ftts_coords[0], stats_2_max_periodic_coords[0], label='Centroid RA', zorder=6)
        ax.plot(ftts_coords[1], stats_2_max_periodic_coords[1], label='Centroid Dec', zorder=8)
        ax.plot(ftts_total, stats_2_max_periodic_total, label='Centroid total \ndeviation', zorder=10)
        ax.axvline(ftt_center, color='r', label='$t0$ of the trigger', zorder=0, linewidth=Linewidth_zero_level)
        if max_centroid_snr_2 is not None:
            ax.axhline(max_centroid_snr_2, linestyle='--', color='k', zorder=1,
                       label='Detected \ncentroid \ndeviation ' + str(np.round(max_centroid_snr_2, 1)))

        ax.set_xlim(ftt_center - ftt_window_plotting, ftt_center + ftt_window_plotting)
        ax.legend(loc=(1.01, 0))
        ax.set_xlabel('First transit time, days')
        ax.set_ylabel('Maximal statistic, $SNR^2$')
        ax.set_title('Centroid statistic folded with period close to trigger', y=1)


def plot_centroid_coordinates_snr_pieces(target_id, transit_times, centr_coords_snr_near_transit_pieces,
                                         centr_coords_times_near_transit_pieces, seasons,
                                         save_fig=Save_fig, no_show=No_show, plot_output_location=locs.Triggers_plots_location,
                                         figsize=Figsize_centroid_coordinate_pieces,
                                         filename_postfix=locs.Centroid_coordinate_pieces_plot_filename_postfix,
                                         markersize_folds=Markersize_folds_small, markersize_bins=Markersize_bins_small,
                                         linewidth=Linewidth_very_thin, plot_time_wing=Centroid_time_wing_days_plotting):
    with SubplotsWrapper(nrows=4, ncols=2, sharex=True, sharey=True, gridspec_kw=Subplots_gridspec_kw_2_line_title, save_fig=save_fig,
                         no_show=no_show, figsize=figsize, target_id=target_id, filename_postfix=filename_postfix,
                         plot_output_location=plot_output_location, grid=True, color_grid=Color_grid, linewidth_grid=Linewidth_grid,
                         zero_lines=True, zero_vline_color=def_cols[3], zero_lines_linewidth=Linewidth_zero_level) as (fig, axs):
        seasons = np.array(seasons)
        seasons[~np.isnan(seasons)] = seasons[~np.isnan(seasons)].astype(int)
        for i_coor in (0, 1):
            times_for_seasons = [[centr_coords_times_near_transit_pieces[i_coor][i] - transit_times[i]
                                  for i in range(len(centr_coords_times_near_transit_pieces[i_coor]))
                                  if seasons[i] == s] for s in range(4)]
            snr_for_seasons = [[centr_coords_snr_near_transit_pieces[i_coor][i]
                                for i in range(len(centr_coords_snr_near_transit_pieces[i_coor]))
                                if seasons[i] == s] for s in range(4)]
            if plot_time_wing is None:
                min_times = [[np.nanmin(tim) for tim in times_for_seasons[s]] for s in range(4)]
                min_time = np.nanmin([np.nanmin(mtim) for mtim in min_times if len(mtim) > 0])
                max_times = [[np.nanmax(tim) for tim in times_for_seasons[s]] for s in range(4)]
                max_time = np.nanmax([np.nanmax(mtim) for mtim in max_times if len(mtim) > 0])
            else:
                min_time = -plot_time_wing
                max_time = plot_time_wing
            for s in range(4):
                ax = axs[s, i_coor]
                if len(times_for_seasons[s]) > 0:
                    if len(times_for_seasons[s]) > 1:
                        time_total, flux_total, flux2_total, counts_total, flux_total_std = stack_fluxes(
                            times_centered=times_for_seasons[s], fluxes=snr_for_seasons[s],
                            masks=[(~np.isclose(x, 0)) & (~np.isnan(x)) for x in snr_for_seasons[s]],
                            time_wing_days=None, dt=None, tmin=None, tmax=None,
                            return_std=True, normalize_by_counts=True)
                    else:
                        time_total = times_for_seasons[s][0]
                        flux_total = snr_for_seasons[s][0]
                        flux_total_std = np.zeros_like(flux_total)
                        counts_total = (~np.isclose(flux_total, 0)).astype(int)
                    for i in range(len(times_for_seasons[s])):
                        col = (0.6, 0.6, 0.6)
                        ax.plot(times_for_seasons[s][i][~np.isclose(snr_for_seasons[s][i], 0)],
                                snr_for_seasons[s][i][~np.isclose(snr_for_seasons[s][i], 0)],
                                linestyle='', marker='o', markersize=markersize_folds, color=col, zorder=10)
                    col = 'k'
                    ax.errorbar(time_total[counts_total > 0], flux_total[counts_total > 0], yerr=flux_total_std[counts_total > 0],
                                xerr=None, markersize=markersize_bins, fmt='o', markerfacecolor=col, markeredgecolor=col,
                                ecolor=col, elinewidth=linewidth, capsize=None, barsabove=False, lolims=False, uplims=False,
                                xlolims=False, xuplims=False, errorevery=1, capthick=None, zorder=10,
                                label=None)
                    ax.set_xlim(min_time, max_time)
                    if i_coor == 0:
                        ax.plot([], [], ' ', label='S'+str(s))
                        ax.legend(loc='upper left', handlelength=0, handletextpad=0.0)
            # ax.set_title('ra' if i_coor == 1 else 'dec', y=0.85, x=0.05, bbox={
            #              'facecolor': 'None', 'edgecolor': 'black', 'boxstyle': 'square,pad=0.1'})
        axs[0, 0].set_title('RA')
        axs[0, 1].set_title('Dec')

        axs[-1, 0].set_xlabel(' ' * 140 + 'Time from the transit, days')
        axs[-2, 0].set_ylabel(' ' * 20 + 'SNR of centroid deviation')
        fig.suptitle('SNR of centroid deviations around the transits', y=1)


def plot_centroid_snr2_seasons(target_id, transit_times, time_total, centr_snr2_near_transit_pieces,
                               seasons, save_fig=Save_fig, no_show=No_show, plot_output_location=locs.Triggers_plots_location,
                               figsize=Figsize_centroid_snr2_seasons, markersize=Markersize_bins_normal,
                               filename_postfix=locs.Centroid_snr2_seasons_plot_filename_postfix
                               ):
    with SubplotsWrapper(nrows=2, ncols=2, sharex=True, sharey=True, gridspec_kw=Subplots_gridspec_kw, save_fig=save_fig, no_show=no_show,
                         figsize=figsize, target_id=target_id, filename_postfix=filename_postfix, plot_output_location=plot_output_location,
                         grid=True, color_grid=Color_grid, linewidth_grid=Linewidth_grid, zero_lines=True,
                         zero_lines_linewidth=Linewidth_zero_level) as (fig, axs):
        seasons = np.array(seasons)
        seasons[~np.isnan(seasons)] = seasons[~np.isnan(seasons)].astype(int)
        seasons_labeled = np.zeros(4).astype(bool)
        for i in range(len(transit_times)):
            season = seasons[i]
            if not np.isnan(season):
                season = int(season)
                ax = axs[season // 2, season % 2]
                ax.plot(time_total, centr_snr2_near_transit_pieces[i], 'o-', linewidth=0.25,
                        markersize=markersize, color=def_cols[season], zorder=10,
                        label='S' + str(season) if not seasons_labeled[season] else None)
                seasons_labeled[season] = True
                ax.set_xlim(np.nanmin(time_total), np.nanmax(time_total))

        for season in seasons:
            if not np.isnan(season):
                season = int(season)
                ax = axs[season // 2, season % 2]
                ax.legend(loc='best', borderpad=0.05, handletextpad=0.05)

        axs[-1, 0].set_xlabel(' ' * 80 + 'Time from transit, days')
        axs[-1, 0].set_ylabel(' ' * 40 + 'Centroid deviation $SNR^2$')
        fig.suptitle('Centroid deviation $SNR^2$ near transits by season', y=1)


def plot_centroid_coordinates_cloud(target_id, transit_times, centr_coords_snr_near_transit_pieces,
                                    seasons, save_fig=Save_fig, no_show=No_show,
                                    plot_output_location=locs.Triggers_plots_location,
                                    figsize=Figsize_centroid_cloud,
                                    filename_postfix=locs.Centroid_cloud_plot_filename_postfix):
    with SubplotsWrapper(nrows=1, ncols=1, sharex=True, sharey=True, gridspec_kw=Subplots_gridspec_kw, save_fig=save_fig, no_show=no_show,
                         figsize=figsize, target_id=target_id, filename_postfix=filename_postfix, plot_output_location=plot_output_location,
                         grid=True, color_grid=Color_grid, linewidth_grid=Linewidth_grid, zero_lines=True,
                         zero_lines_linewidth=Linewidth_zero_level) as (fig, ax):
        seasons[~np.isnan(seasons)] = seasons[~np.isnan(seasons)].astype(int)
        seasons_labeled = np.zeros(4).astype(bool)
        for i in range(len(transit_times)):
            season = seasons[i]
            if not np.isnan(season):
                season = int(season)
                ax.plot(centr_coords_snr_near_transit_pieces[0][i], centr_coords_snr_near_transit_pieces[1][i], 'o',
                        markersize=1, color=def_cols[season], zorder=10,
                        label='S' + str(season) if not seasons_labeled[season] else None)
                seasons_labeled[season] = True
        ax.legend(loc='best', borderpad=0.05, handletextpad=0.05)
        ax.set_xlabel('SNR of RA of centroid deviation')
        ax.set_ylabel('SNR of Dec of centroid deviation')
        ax.set_aspect('equal')
        max_lim = np.max(np.abs(np.array(list(ax.get_xlim()) + list(ax.get_ylim()))))
        ax.set_xlim(-max_lim, max_lim)
        ax.set_ylim(-max_lim, max_lim)
        fig.suptitle('Centroid angular position $SNR$ near transits', y=1)


def produce_centroid_plots(tce, save_fig=Save_fig, no_show=No_show, plot_output_location=locs.Triggers_plots_location,
                           time_wing_days=Centroid_time_wing_days_plotting,
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
                                 save_fig=save_fig, no_show=no_show, plot_output_location=plot_output_location)
    plot_centroid_coordinates_snr_pieces(target_id=tce.target_id, transit_times=tce.summed_transit_times,
                                         centr_coords_snr_near_transit_pieces=centr_coords_snr_near_transit_pieces,
                                         centr_coords_times_near_transit_pieces=centr_coords_times_near_transit_pieces,
                                         seasons=seasons,
                                         save_fig=save_fig, no_show=no_show, plot_output_location=plot_output_location)
    plot_centroid_snr2_seasons(target_id=tce.target_id, transit_times=tce.summed_transit_times,
                               time_total=time_total, centr_snr2_near_transit_pieces=centr_snr2_near_transit_pieces,
                               seasons=seasons, save_fig=save_fig, no_show=no_show,
                               plot_output_location=plot_output_location)
    plot_centroid_coordinates_cloud(target_id=tce.target_id, transit_times=tce.summed_transit_times,
                                    centr_coords_snr_near_transit_pieces=centr_coords_snr_near_transit_pieces,
                                    seasons=seasons, save_fig=save_fig, no_show=no_show,
                                    plot_output_location=plot_output_location)
    plot_centroid_full_stats(tce, save_fig=save_fig, no_show=no_show, plot_output_location=plot_output_location)


def get_target_for_tce(tce):
    import target_class
    target = target_class.KeplerTarget(tce.target_id, data_location=tce.data_location, output_location=tce.output_location,
                                       problems_folder=None)
    return target


def plot_all_postprocessing(tce, save_fig=Save_fig, no_show=No_show, fluxes_and_stats=True,
                            folding_residuals=True, bg_star_pixels=True, centroid_plots=True,
                            veto_parts_plot=True, even_odd=True, transit_info=True,
                            plots_output_location=locs.Triggers_plots_location):
    target = get_target_for_tce(tce)
    target.fetch_saved_lightcurve()
    # If the folder does not exist, allow to create it:
    _ = get_correct_tid_folder(target_id=tce.target_id, location=plots_output_location, allow_creation=True, with_prefix_folder=False)
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
        plot_all_series(target, save_fig=save_fig, no_show=no_show, plot_output_location=plots_output_location)
        plot_all_folds(target, save_fig=save_fig, no_show=no_show, plot_output_location=plots_output_location)
    if folding_residuals:
        plot_folding_residuals_histograms(tce, save_fig=save_fig, no_show=no_show, plot_output_location=plots_output_location)
    if bg_star_pixels:
        plot_bg_star_pixels_ppm(tce, save_fig=save_fig, no_show=no_show, plot_output_location=plots_output_location)
        plot_bg_star_pixels_stats(tce, save_fig=save_fig, no_show=no_show, plot_output_location=plots_output_location)
    if centroid_plots:
        produce_centroid_plots(tce, save_fig=save_fig, no_show=no_show, plot_output_location=plots_output_location)
    if veto_parts_plot:
        plot_fold_veto_parts(tce=tce, no_show=no_show, save_fig=save_fig, plot_output_location=plots_output_location)
    if even_odd:
        plot_snrs_and_wh_fluxes_even_odd(target=target, best_temp_folded=tce.best_temp_folded,
                                         best_ampls=[tce.scores_ampls_even_odd[0][1], tce.scores_ampls_even_odd[1][1]],
                                         no_show=no_show, save_fig=save_fig, plot_output_location=plots_output_location)
        plot_wh_fluxes_even_odd_from_veto(tce=tce, no_show=no_show, save_fig=save_fig, plot_output_location=plots_output_location)
    if transit_info:
        plot_transit_info(tce=tce, save_fig=save_fig, no_show=no_show, plot_output_location=plots_output_location)


def plot_all_after_slides_injections(tce=None, target=None, save_fig=Save_fig, no_show=No_show,
                                     plots_output_location=locs.Triggers_plots_location,
                                     include_periodogram=True):
    if target is None:
        target = get_target_for_tce(tce)
    target.fetch_saved_lightcurve()
    if not hasattr(target, 'retrieved_int_score_success') or target.retrieved_int_score_success is None:
        target.retrieve_injections()
    if not hasattr(target, 'int_scores_slides' or target.int_scores_slides is None):
        target.retrieve_background()
    if not hasattr(target, 'int_scores_slides_clean') or target.int_scores_slides_clean is None:
        target.retrieve_background_clean()
    plot_significance_histograms(target, save_fig=save_fig, no_show=no_show, plot_output_location=plots_output_location)
    if include_periodogram:
        plot_periodogram_chunks(target, save_fig=save_fig, no_show=no_show, plot_output_location=plots_output_location)
    plot_injections_corners(target, save_fig=save_fig, no_show=no_show, plot_output_location=plots_output_location)
    plot_injections_radii_durations_score(target, save_fig=save_fig, no_show=no_show, plot_output_location=plots_output_location)


def plot_neighbors_maps(tce, save_fig=Save_fig, no_show=No_show, plot_output_location=locs.Triggers_plots_location,
                        figsize=Figsize_neighbors_maps_quarters,
                        filename_postfix=locs.Neighbors_maps_plot_filename_postfix,
                        ms_target=18, ms_neighbors=6, ms_kois=3,
                        neighbors_file_path=None, neighbors_file_name=None,
                        neighbors_folder=locs.Neighbors_folder, trigger_table_gaia=None,
                        gaia_trigger_table_file=locs.Gaia_trigger_table_file
                        ):
    import warnings
    from catalog_functions import test_gaia_table
    from auxiliary_functions import get_orig_quarters, get_target_pixel_sky_coords_fun, get_splitted_saps_from_pixels
    from tce_parameters import Part_of_flux_peak_threshold, Part_of_cumsum_2_targ_pix, Number_of_pixels_cut, Part_of_light
    # n_quarters = tce.n_quarters

    # Mark in which quarters it had transits!

    if trigger_table_gaia is None:
        if tce.gaia_target_table is None:
            tce.get_target_gaia_table(file_path=gaia_trigger_table_file)
        trigger_table_gaia = tce.gaia_target_table
    else:
        tce.trigger_table_gaia = trigger_table_gaia
    if tce.neighbors_table is None:
        tce.get_neighbors_info(neighbors_file_path=neighbors_file_path, neighbors_file_name=neighbors_file_name,
                               neighbors_folder=neighbors_folder)
    if tce.neighboring_kois is None:
        tce.check_neighboring_koi()

    ra_tid, dec_tid = trigger_table_gaia['ra'][0], trigger_table_gaia['dec'][0]
    neighbors_table = tce.neighbors_table

    times, tp_fluxes, sky_coords, masks, aperture_masks = get_target_pixel_sky_coords_fun(
        target_id=tce.target_id,  data_location=tce.target_pixels_location, remove_zeroth_quarter=True,
        remove_quarter_17=True, cut_edge_nans=True)
    masks_bad, reasons_bad = test_gaia_table(neighbors_table)
    orig_quarters = get_orig_quarters(np.array([np.nanmean(time) for time in times]))

    _, _, inclusion_masks = get_splitted_saps_from_pixels(
        tp_fluxes, part_of_peak=Part_of_flux_peak_threshold, number_of_pixels_cut=Number_of_pixels_cut,
        part_of_cumsum_2=Part_of_cumsum_2_targ_pix, part_of_light=Part_of_light, return_inclusion_masks=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # , category=UserWarning
        log10_tp_flux_min = np.nanmin([np.nanmin(np.log10(np.nanmean(tp_fluxes_q, axis=0))) for tp_fluxes_q in tp_fluxes])
        log10_tp_flux_max = np.nanmax([np.nanmax(np.log10(np.nanmean(tp_fluxes_q, axis=0))) for tp_fluxes_q in tp_fluxes])
        ra_min = np.nanmin([np.nanmin(sky_coords[q][0]) for q in range(len(tp_fluxes))])
        ra_max = np.nanmax([np.nanmax(sky_coords[q][0]) for q in range(len(tp_fluxes))])
        dec_min = np.nanmin([np.nanmin(sky_coords[q][1]) for q in range(len(tp_fluxes))])
        dec_max = np.nanmax([np.nanmax(sky_coords[q][1]) for q in range(len(tp_fluxes))])
    dec_min = min(dec_min - Pix_scale_degrees / 2**0.5,
                  np.nanmin(tce.neighbors_table['dec']) - Pix_scale_degrees * 0.4)
    dec_max = max(dec_max + Pix_scale_degrees / 2**0.5,
                  np.nanmax(tce.neighbors_table['dec']) + Pix_scale_degrees * 0.4)
    ra_min = min(ra_min - Pix_scale_degrees / 2**0.5,
                 np.nanmin(tce.neighbors_table['ra']) - Pix_scale_degrees * 0.4)
    ra_max = max(ra_max + Pix_scale_degrees / 2**0.5,
                 np.nanmax(tce.neighbors_table['ra']) + Pix_scale_degrees * 0.4)

    # RA and DEC have different weight in the metric:
    aspect = 1 / np.cos(dec_tid / 180 * np.pi)
    ra_wing = max(ra_max - ra_tid, ra_tid - ra_min)
    dec_wing = max(dec_max - dec_tid, dec_tid - dec_min)
    # max_wing = max(ra_wing / aspect, dec_wing)
    # ra_wing = max_wing * aspect
    # dec_wing = max_wing

    gridspec_kw = {'top': 0.9, 'bottom': 0.05, 'hspace': 0, 'wspace': 0., 'left': 0.05, 'right': 0.98}
    with SubplotsWrapper(nrows=4, ncols=4, sharex=True, sharey=True, gridspec_kw=gridspec_kw, save_fig=save_fig, no_show=no_show,
                         figsize=figsize, target_id=tce.target_id, filename_postfix=filename_postfix,
                         plot_output_location=plot_output_location, grid=False, color_grid=Color_grid, linewidth_grid=Linewidth_grid,
                         zero_lines=True, zero_lines_linewidth=Linewidth_zero_level, tight_layout=False) as (fig, axs):
        for q in range(len(tp_fluxes)):
            q_orig = orig_quarters[q]
            n_trans_in_q = np.sum(np.isclose(np.abs(tce.transit_orig_quarters - q_orig), 0))

            ax_row = (q_orig - 1) // 4
            ax_col = (q_orig - 1) % 4
            ax = axs[ax_row, ax_col]
            mpbl = plot_neighbors_pixel_map_from_tables(
                trigger_table_gaia=trigger_table_gaia, neighbors_table=neighbors_table,
                tp_fluxes_q=tp_fluxes[q], sky_coords_q=sky_coords[q],
                aperture_mask_q=aperture_masks[q], inclusion_mask_q=inclusion_masks[q],
                masks_bad=masks_bad, reasons_bad=reasons_bad,
                ax=ax, log10_tp_flux_min=log10_tp_flux_min, log10_tp_flux_max=log10_tp_flux_max,
                t_index='mean', pix_scale_degrees=Pix_scale_degrees,
                reasons_dict_plot=Reasons_dict_plot, plot_radius_pix=Plot_radius_pix,
                draw_circular_grid=True, ms_target=ms_target, ms_neighbors=ms_neighbors,
                add_colorbar=False,  # True if ax_col == 3 else False,
                set_axes_labels=False, return_mappable=True, set_aspect=False)
            if tce.neighboring_kois is not None:
                if len(tce.neighboring_kois['period']) > 0:
                    for i in range(len(tce.neighboring_kois['period'])):
                        if tce.neighboring_kois['target_id'][i] != tce.target_id:
                            ax.plot(tce.neighboring_kois['ra'][i], tce.neighboring_kois['dec'][i], 'o',
                                    color='k', markersize=ms_kois, zorder=50,
                                    label='KOI'+tce.neighboring_kois['target_id'][i]+', '+str(
                                        np.round(tce.neighboring_kois['period'][i], 1))+'d')
                    ax.legend(fontsize=8, loc='upper right')
            ax.set_xlim(ra_tid - ra_wing, ra_tid + ra_wing)
            ax.set_ylim(dec_tid - dec_wing, dec_tid + dec_wing)
            ax.set_aspect(aspect)
            ax.set_title('Q' + str(q_orig) + ', '+str(int(n_trans_in_q))+' transits', y=0.92, x=0.2, fontsize=Fontsize_legend)
            if ax_row == 3:
                ax.set_xlabel('RA, deg')
            if ax_col == 0:
                ax.set_ylabel('Dec, deg')
        for q in range(1, 17):
            if np.sum(np.isclose(np.abs(orig_quarters - q), 0)) == 0:
                ax_row = (q - 1) // 4
                ax_col = (q - 1) % 4
                axs[ax_row, ax_col].axis('off')

        # ax.set_aspect(1 / np.cos(dec_tid / 180 * np.pi), adjustable='datalim')
        ax = fig.add_axes([0.05, 0.95, 0.9, 0.04], frame_on=False)
        ax.axis('off')
        plt.colorbar(mpbl, ax=ax, aspect=90, label=r'$log_{10}$' + ' flux', orientation='horizontal')
        fig.suptitle(get_target_title(tce) + '\nTarget pixel maps for quarters with neighboring stars', y=1., fontsize=Fontsize_title)


def plot_neighbors_pixel_map_from_tables(
        trigger_table_gaia, neighbors_table, masks_bad, reasons_bad,
        tp_fluxes_q, sky_coords_q, aperture_mask_q=None, inclusion_mask_q=None, t_index='mean',
        log10_tp_flux_min=None, log10_tp_flux_max=None,
        ax=None, pix_scale_degrees=Pix_scale_degrees, reasons_dict_plot=Reasons_dict_plot,
        plot_radius_pix=Plot_radius_pix, draw_circular_grid=True, ms_target=18, ms_neighbors=6,
        add_colorbar=True, set_axes_labels=True, return_mappable=False, set_aspect=True,
        figsize=Figsize_neighbors_pixel_map_from_tables, dpi=Plots_dpi):
    """
    :param trigger_table_gaia:
    :type trigger_table_gaia:
    :param neighbors_table:
    :type neighbors_table:
    :param masks_bad:
    :type masks_bad:
    :param reasons_bad:
    :type reasons_bad:
    :param tp_fluxes_q: FOR A SPECIFIC QUARTER!
    :type tp_fluxes_q:
    :param sky_coords_q: FOR A SPECIFIC QUARTER!
    :type sky_coords_q:
    :param aperture_mask_q: FOR A SPECIFIC QUARTER!
    :type aperture_mask_q:
    :param inclusion_mask_q: FOR A SPECIFIC QUARTER!
    :type inclusion_mask_q:
    :param t_index:
    :type t_index:
    :param log10_tp_flux_min:
    :type log10_tp_flux_min:
    :param log10_tp_flux_max:
    :type log10_tp_flux_max:
    :param ax:
    :type ax:
    :param pix_scale_degrees:
    :type pix_scale_degrees:
    :param reasons_dict_plot:
    :type reasons_dict_plot:
    :param plot_radius_pix:
    :type plot_radius_pix:
    :param draw_circular_grid:
    :type draw_circular_grid:
    :param ms_target:
    :type ms_target:
    :param ms_neighbors:
    :type ms_neighbors:
    :param add_colorbar:
    :type add_colorbar:
    :param set_axes_labels:
    :type set_axes_labels:
    :param return_mappable:
    :type return_mappable:
    :param set_aspect:
    :type set_aspect:
    :param figsize:
    :type figsize:
    :param dpi:
    :type dpi:
    :return:
    :rtype:
    """

    import warnings
    ms_t = ms_target  # markersize for target
    ms_n = ms_neighbors  # markersize for neighbors

    tid = trigger_table_gaia['kepid'][0]
    ra_tid, dec_tid = trigger_table_gaia['ra'][0], trigger_table_gaia['dec'][0]
    # ----------------------------------------------------------------------------
    # Making colors of pixels:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # , category=UserWarning
        if t_index == 'mean':
            mean_stamp = np.nanmean(tp_fluxes_q, axis=0)
            coloring_arr = np.log10(mean_stamp)
            if log10_tp_flux_min is None:
                log10_tp_flux_min = np.nanmin(coloring_arr)
            if log10_tp_flux_max is None:
                log10_tp_flux_max = np.nanmax(coloring_arr)
            ras = np.nanmean(sky_coords_q[0], axis=0)
            decs = np.nanmean(sky_coords_q[1], axis=0)
        else:
            coloring_arr = np.log10(tp_fluxes_q[t_index])
            # # making normalized coloring for the fluxes
            if log10_tp_flux_min is None:
                log10_tp_flux_min = np.nanmin(coloring_arr)
            if log10_tp_flux_max is None:
                log10_tp_flux_max = np.nanmax(coloring_arr)
            ras = sky_coords_q[0][t_index]  # [i][j]
            decs = sky_coords_q[1][t_index]  # [i][j]

    colors, mpbl = make_coloring(coloring_arr, min_val=log10_tp_flux_min, max_val=log10_tp_flux_max,
                                 cmap=None, cmap_name='viridis')
    d_ra_i = np.nanmean((ras - np.roll(ras, 1, axis=0))[1:, :])
    d_ra_j = np.nanmean((ras - np.roll(ras, 1, axis=1))[:, 1:])
    d_dec_i = np.nanmean((decs - np.roll(decs, 1, axis=0))[1:, :])
    d_dec_j = np.nanmean((decs - np.roll(decs, 1, axis=1))[:, 1:])
    d_pos_i = np.array([d_ra_i, d_dec_i]) / 2
    d_pos_j = np.array([d_ra_j, d_dec_j]) / 2

    min_dec = min(np.nanmin(decs), dec_tid - plot_radius_pix * pix_scale_degrees)
    max_dec = max(np.nanmax(decs), dec_tid + plot_radius_pix * pix_scale_degrees)
    min_ra = min(np.nanmin(ras), ra_tid - plot_radius_pix * pix_scale_degrees)
    max_ra = max(np.nanmax(ras), ra_tid + plot_radius_pix * pix_scale_degrees)

    # ----------------------------------------------------------------------------
    # Making figure:
    if ax is None:
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.gca()

    # ----------------------------------------------------------------------------
    # Plotting the pixels:
    for i in range(len(coloring_arr)):
        for j in range(len(coloring_arr[i])):
            color = colors[i][j]
            ra, dec = ras[i][j], decs[i][j]
            # positions = (np.array([ra, dec]) + np.array([[0,0], d_pos_i, d_pos_i+d_pos_j, d_pos_j])).T
            positions = (np.array([ra, dec]) + np.array(
                [-d_pos_i - d_pos_j, -d_pos_i + d_pos_j, d_pos_i + d_pos_j, d_pos_i - d_pos_j])).T
            ax.fill(positions[0], positions[1], color=color)
            # ax.plot([ra], [dec], 'o', color='w', markersize=0.5)
            if aperture_mask_q is not None:
                if aperture_mask_q[i][j]:
                    ax.plot(np.concatenate((positions[0], [positions[0][0]])),
                            np.concatenate((positions[1], [positions[1][0]])), color='w',
                            linewidth=1, zorder=5)
            if inclusion_mask_q is not None:
                if inclusion_mask_q[i][j]:
                    ax.plot(np.concatenate((positions[0], [positions[0][0]])),
                            np.concatenate((positions[1], [positions[1][0]])), color=def_cols[3],
                            linestyle='--', linewidth=0.7, zorder=10)
    if add_colorbar:
        plt.colorbar(mpbl, ax=ax, aspect=40, label=r'$log_{10}$' + ' flux')

    # ----------------------------------------------------------------------------
    # Plotting the target:
    ax.plot([ra_tid], [dec_tid], marker='o', linestyle='', color=def_cols[9], markersize=ms_t, label='Target', zorder=20)
    flux_target = trigger_table_gaia['phot_g_mean_flux'][0]

    # ----------------------------------------------------------------------------
    # Plotting the neighbors:
    # impact_vals = neighbors_table['phot_g_mean_mag']
    fluxes = neighbors_table['phot_g_mean_flux']
    impact_vals = np.log10(neighbors_table['phot_g_mean_flux'])
    cols = make_coloring(values=impact_vals, cmap_name='autumn')[0]
    sizes = ms_n + (2 * ms_n - ms_n) * (impact_vals - np.nanmin(impact_vals)) / (np.nanmax(impact_vals) - np.nanmin(impact_vals))

    suspicious_labeled = False
    for i_n in range(len(neighbors_table)):
        ra_1, dec_1 = neighbors_table['ra'][i_n], neighbors_table['dec'][i_n]
        ax.plot([ra_1], [dec_1], 'o', color=cols[i_n], markersize=sizes[i_n], label='Neighbors' if i_n == 0 else None,
                zorder=23)
        # Plotting the direction to each neighbor:
        # ax.plot([ra_tid, ra_1], [dec_tid, dec_1], '--', color='k', linewidth=0.5)
        if fluxes[i_n] / flux_target >= 1e-4:
            col_text = (0.7, 0, 0)
        else:
            col_text = 'k'
        ax.text(ra_1 - 1e-4, dec_1 + 2e-4, str((np.round(impact_vals[i_n], 1))), fontsize=11, color=col_text,
                zorder=30, ha='center', va='bottom')

        flag = ''
        for i_m in range(len(masks_bad)):
            if masks_bad[i_m][i_n]:
                flag += reasons_dict_plot[reasons_bad[i_m]] + '\n '
        if flag != '':
            ax.plot([ra_1], [dec_1], 'x', color='k', markersize=sizes[i_n],
                    label='Suspicious' if (not suspicious_labeled) else None, zorder=28)
            suspicious_labeled = True
            ax.text(ra_1, dec_1 - 2e-4, str(flag), fontsize=9, color='k', zorder=30, ha='center', va='top')

    if draw_circular_grid:
        plot_circular_grid(ax=ax, center=(ra_tid, dec_tid), dr=pix_scale_degrees, r_max=plot_radius_pix*Pix_scale_degrees)

    if set_axes_labels:
        ax.set_xlabel('RA, deg')
        ax.set_ylabel('Dec, deg')
    ax.legend(fontsize=8, loc='upper right')
    ax.set_title(tid, fontsize=14)
    ax.set_ylim(min_dec, max_dec)
    ax.set_xlim(min_ra, max_ra)
    # ax.set_aspect(1 / np.cos(dec_tid / 180 * np.pi), adjustable='datalim')
    if set_aspect:
        # ax.set_aspect(1 / np.cos(dec_tid / 180 * np.pi), adjustable='box')
        ax.set_aspect(1 / np.cos(dec_tid / 180 * np.pi), adjustable='datalim')
    ax.tick_params(axis='x', labelrotation=0.1)
    if return_mappable:
        return mpbl


# ##########################################################################################################
# AUXILIARY FUNCTIONS
# ##########################################################################################################

def prepare_target_for_plotting(target, initialize_templates=False,
                                condition_templates=False, detect_single_transits=False):
    if not hasattr(target, 'best_int_score') or target.best_int_score is None:
        target.unpickle_small_attributes()

    go_fetch = (
        not hasattr(target, 'klc') or target.klc is None or
        not hasattr(target.klc, 'summed_stats_raw') or target.klc.summed_stats_raw is None or
        (not hasattr(target.klc, 'tb') and initialize_templates) or
        (target.klc.tb is None and initialize_templates) or
        (not hasattr(target.klc, 'whitened_templates_td') and condition_templates) or
        (target.klc.whitened_templates_td is None and condition_templates) or
        (not hasattr(target.klc, 'statistics_snr') and detect_single_transits) or
        (target.klc.statistics_snr is None and detect_single_transits))
    if go_fetch:
        target.fetch_saved_lightcurve(initialize_templates=initialize_templates, condition_templates=condition_templates,
                                      detect_single_transits=detect_single_transits)


def get_binwid(tce=None, dt=dt_lc, temp_dur_days=None, temp_num=None, min_binwid_to_dt=1., min_part_duration=0.2,
               max_binwid_to_dt=5., max_part_duration=0.2):
    """

    :param temp_dur_days:
    :type temp_dur_days:
    :param temp_num:
    :type temp_num:
    :param tce:
    :type tce:
    :param dt:
    :type dt:
    :param min_binwid_to_dt:
    :type min_binwid_to_dt: float or None
    :param min_part_duration:
    :type min_part_duration: float or None
    :param max_binwid_to_dt:
    :type max_binwid_to_dt: float or None
    :param max_part_duration:
    :type max_part_duration: float or None
    :return:
    :rtype:
    """
    if temp_dur_days is None:
        if temp_num is None:
            if tce is not None:
                temp_num = tce.best_temp_folded if tce.best_temp_folded is not None else tce.best_temp
            else:
                raise AssertionError('At least something should be not None')
        temp_dur_days = Durations[temp_num]
    binwid = dt
    if min_binwid_to_dt is not None:
        binwid = max(binwid, min_binwid_to_dt * dt)
    if min_part_duration is not None:
        binwid = max(binwid, temp_dur_days * min_part_duration)
    if max_binwid_to_dt is not None:
        binwid = min(binwid, max_binwid_to_dt * dt)
    if max_part_duration is not None:
        binwid = min(binwid, temp_dur_days * max_part_duration)
    return binwid


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


def save_figure(fig, as_image=True, as_pickle=False, file=None, folder=None, filename='fig', dpi=Plots_dpi,
                tight_layout=True):
    if file is None:
        if folder is None:
            folder = os.getcwd()
        file = os.path.join(folder, filename)
    file_0 = str(file)
    if as_image:
        file = str(file_0)
        if file[-len('.png'):] != '.png':
            file = file + '.png'
        if tight_layout:
            fig.savefig(file, bbox_inches='tight', dpi=dpi)
        else:
            fig.savefig(file, dpi=dpi)
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


def plot_circular_grid(ax, center, dr=Pix_scale_degrees, r_max=Plot_radius_pix*Pix_scale_degrees,
                       color=Color_grid, linewidth=Linewidth_grid, draw_zero_lines=True, zorder=0):
    radii = np.arange(dr, r_max + 0.5 * dr, dr)
    for radius in radii:
        circle = plt.Circle(center, radius, color=color, fill=False, lw=linewidth, zorder=zorder)
        ax.add_artist(circle)
    if draw_zero_lines:
        ax.axhline(center[1], color=color, lw=linewidth, zorder=zorder)
        ax.axvline(center[0], color=color, lw=linewidth, zorder=zorder)


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


# -----For_the_corner_plot-----------------------------------------------------------------------------------------------
def whiteness_levels(rgb_color, n_levels=5, last_white=False, end_val=1., start_from_saturated=True):
    if start_from_saturated:
        hsv_color = matplotlib.colors.rgb_to_hsv([rgb_color])[0]
        hsv_color[1] = 1.
        rgb_color = matplotlib.colors.hsv_to_rgb([hsv_color])[0]
    hsv_color = matplotlib.colors.rgb_to_hsv([rgb_color])[0]
    white_hsv = np.array([hsv_color[0], 1-end_val, end_val])
    s_s = np.linspace(hsv_color[1], white_hsv[1], n_levels + int(not last_white), endpoint=last_white)
    v_s = np.linspace(hsv_color[2], white_hsv[2], n_levels + int(not last_white), endpoint=last_white)
    color_grid = np.array([[hsv_color[0], s_s[i], v_s[i]] for i in range(len(s_s))])
    color_grid = matplotlib.colors.hsv_to_rgb(color_grid)
    return color_grid


def get_levels_grid(rgb_color, n_levels, alpha=1., last_white=True, end_val=1., gradiend_alpha=True, alpha_min=0.1):
    color_grid = whiteness_levels(rgb_color=rgb_color, n_levels=n_levels, last_white=last_white, end_val=end_val)[::-1]
    if gradiend_alpha:
        alphas = np.linspace(alpha_min, alpha, len(color_grid))
    else:
        alphas = np.full(len(color_grid), alpha)
    color_grid = np.array([np.append(color_grid[i], alphas[i]) for i in range(len(color_grid))])
    color_grid[0, -1] = 0
    return color_grid
