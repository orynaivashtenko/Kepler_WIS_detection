#!/usr/bin/env python3

from template_bank.template_module import Durations
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import ticker

Fontsize = 12
Fontsize_legend = 11
Fontsize_title = 13
Fontsize_label = 12
Figsize = (14, 5)
Linewidth = 1
Markersize = 3

plt.rcParams.update({'font.size': Fontsize, 'font.serif': 'Times New Roman',
                     'lines.linewidth': Linewidth, 'lines.markersize': Markersize,
                     'legend.fontsize': Fontsize_legend, 'legend.loc': 'upper right',
                     'axes.titlesize': Fontsize_title, 'axes.labelsize': Fontsize_label,
                     'figure.figsize': Figsize, 'figure.autolayout': True, 'figure.titlesize': Fontsize_title})
def_cols = plt.rcParams['axes.prop_cycle'].by_key()['color']
def_cols_20 = matplotlib.colormaps['tab20']
cmap1 = matplotlib.colormaps['cool']
Flux_quarters_colors = (def_cols[0], def_cols[2])  # ((0.5, 0.5, 1), (0.2, 0.7, 0.5))
plt.ioff()

Plots_dpi = 320
No_show = True
Save_fig = False
Subplots_gridspec_kw = {'wspace': 0, 'hspace': 0, 'top': 0.95, 'bottom': 0.05}
Subplots_gridspec_kw_2_line_title = {'wspace': 0, 'hspace': 0, 'top': 0.92, 'bottom': 0.05}

Figsize_flux = (16, 4)
Figsize_fold = (14, 5)
Figsize_periodogram = (14, 9)
Figsize_fold_residuals = (14, 4)
Figsize_p_planet_hist = (14, 4)
Figsize_corner = (14, 14)
Figsize_inj_radii_dur_scatter = (14, 4.5)
Figsize_centroid_periodic = (11, 4)
Figsize_centroid_coordinate_pieces = (14, 7)
Figsize_centroid_snr2_seasons = (9, 5)
Figsize_centroid_cloud = (5, 4.5)
Figsize_residuals_hist = (10, 4)
Figsize_star_bg_pixels_ppm = (10, 6)
Figsize_season_pieces = (16, 7)

Linewidth_coarse = 1.5
Linewidth_thin = 1
Linewidth_very_thin = 0.5
Linewidth_zero_level = 0.5
Linewidth_vlines = 0.5
Linewidth_grid = 0.5
Linewidth_p_planet_hist = 2
Color_grid = (0.7, 0.7, 0.7)

Markersize_folds_small = 0.7
Markersize_bins_small = 2

Whitening_time = 128 * 0.02
Zoom_wing_folded_plots = 1.5 * Whitening_time
Centroid_ftt_wing_days_plotting = 4 * Durations[-1]
Centroid_time_wing_days_plotting = 5 * Durations[-1]
Seasonal_plotting_wing_days = 5 * Durations[-1]


