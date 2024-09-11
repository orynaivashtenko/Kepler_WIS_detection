#!/usr/bin/env python3

import os
from template_bank.template_module import dt_lc

Template_bank_folder = os.path.dirname(os.path.abspath(__file__))
M_1_3_r_selected_list_filename = 'm_1_3_r_kepler_selected.npy'
M_1_3_r_selected_list_file = os.path.join(Template_bank_folder, M_1_3_r_selected_list_filename)

# Boundaries for about FGK stars, maybe some A or B or M
Max_stellar_radius_in_solar_radii = 3.  # Solar radii
Min_stellar_radius_in_solar_radii = 0.5  # Solar radii
Max_stellar_mass_in_solar_masses = 2.5  # Solar masses
Min_stellar_mass_in_solar_masses = 0.3  # Solar masses
M13r_bin_edges_template_prior = [0.2, 0.75, 1.2, 1.7]
"""mass^(1/3) / radius (in solar units) edges to the bins used to bin the template prior tables"""

Limb_darkening_coefs_template_bank_generation =[
    (0.47, 0.23),
    (0.4880, 0.2241),
    (0.6332, 0.0939),
    (0.6528, 0.0714),
    (0.6184, 0.0980),
    (0.5767, 0.1317),
    (0.5301, 0.1677),
    (0.4830, 0.2023),
    (0.4412, 0.2312),
    (0.4008, 0.2584),
    (0.3695, 0.2774),
    (0.3423, 0.2939),
    (0.3210, 0.3065),
    (0.3033, 0.3180),
    (0.2866, 0.3299),
    (0.2708, 0.3429)]

Limb_darkening_coefs_sun = (0.47, 0.23)
Template_support = 256
Min_template_duration_days = 2.1 * dt_lc
Max_template_duration_days = 1.5
