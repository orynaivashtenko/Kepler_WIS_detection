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
Plots_dpi = 320

# , 'figure.autolayout': True # 'figure.figsize': Figsize,

def_cols = plt.rcParams['axes.prop_cycle'].by_key()['color']
def_cols_20 = matplotlib.colormaps['tab20']
cmap1 = matplotlib.colormaps['cool']
Flux_quarters_colors = (def_cols[0], def_cols[2])  # ((0.5, 0.5, 1), (0.2, 0.7, 0.5))
plt.ioff()


No_show = False
Save_fig = False
Fig_gridspec_kw = None  # {'top': 0.95, 'bottom': 0.05, 'right': 0.95, 'left': 0.05}
Fig_gridspec_kw_2 = None  # {'top': 0.9, 'bottom': 0.05, 'right': 0.95, 'left': 0.05}
Subplots_gridspec_kw = {'wspace': 0, 'hspace': 0}
Subplots_gridspec_kw_2_line_title = {'wspace': 0, 'hspace': 0}
Subplots_gridspec_kw_3_line_title = {'wspace': 0, 'hspace': 0}
# {'wspace': 0, 'hspace': 0, 'top': 0.95, 'bottom': 0.05, 'right': 0.95, 'left': 0.05}
# Subplots_gridspec_kw_2_line_title = {'wspace': 0, 'hspace': 0, 'top': 0.92, 'bottom': 0.05, 'right': 0.95, 'left': 0.05}
# Subplots_gridspec_kw_3_line_title = {'wspace': 0, 'hspace': 0, 'top': 0.85, 'bottom': 0.05, 'right': 0.95, 'left': 0.05}


# def ren_figsize(figsize, ref_height=5):
#     """renormalize figsize to have the same fontsize"""
#     # factor = ref_height / figsize[1]
#     # return (figsize[0] * factor, figsize[1] * factor)
#     factor = 1
#     return (figsize[0] * factor, figsize[1] * factor)


Figsize_flux = (18, 4)
Figsize_all_fluxes = (18, 12)
Figsize_all_folds = (18, 6)
Figsize_flux_wide = (18, 5)
Figsize_fold = (18, 5)
Figsize_periodogram = (18, 9)
Figsize_fold_residuals = (18, 4)
Figsize_p_planet_hist = (14, 4)
Figsize_corner = (12, 12)
Figsize_inj_radii_dur_scatter = (18, 5)
Figsize_centroid_periodic = (11, 4)
Figsize_centroid_coordinate_pieces = (18, 7)
Figsize_centroid_snr2_seasons = (9, 5)
Figsize_centroid_cloud = (5, 4.5)
Figsize_residuals_hist = (10, 4)
Figsize_star_bg_pixels_ppm = (18, 4)
Figsize_season_pieces = (18, 6)
Figsize_veto_parts_fold = (18, 3)
Figsize_even_odd = (18, 6)
Figsize_even_odd_from_veto = (18, 4)
Figsize_neighbors_maps_quarters = (16, 19)
Figsize_transit_info = (16, 5)
Figsize_neighbors_pixel_map_from_tables = (9, 6)

Linewidth_coarse = 1.5
Linewidth_medium = 1.2
Linewidth_thin = 1
Linewidth_very_thin = 0.5
Linewidth_zero_level = 0.5
Linewidth_vlines = 0.5
Linewidth_grid = 0.5
Linewidth_p_planet_hist = 1.5
Color_grid = (0.7, 0.7, 0.7)

Markersize_folds_small = 0.6
Markersize_bins_small = 1
Markersize_bins_normal = 2
Markersize_transits_large = 8

Whitening_time = 128 * 0.02
Zoom_wing_folded_plots = 1.5 * Whitening_time
Zoom_wing_folded_even_odd = max(Whitening_time, 1.5 * Durations[-1])
Centroid_ftt_wing_days_plotting = 4 * Durations[-1]
Centroid_time_wing_days_plotting = max(3 * Durations[-1], 1.5 * Whitening_time)
Seasonal_plotting_wing_days = max(3 * Durations[-1], 1.5 * Whitening_time)


# ------------- Pixel maps, Neighbors ----------------
Reasons_dict_plot = {
    'phot_bp_rp_excess_factor': 'bp_rp',
    'ruwe': 'ruwe',
    'astrometric_excess_noise': 'astr_noise',
    'radial_velocity': 'rv',
    'non_single_star': 'non_sing',
    'phot_variable': 'var'}

Pix_scale_degrees = 3.98 / (60 * 60)
Plot_radius_pix = 8
