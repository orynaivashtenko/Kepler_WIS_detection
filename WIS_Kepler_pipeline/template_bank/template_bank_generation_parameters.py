#!/usr/bin/env python3

import os
from template_bank.template_module import dt_lc
from locations import Run_outputs_folder

Template_bank_folder = os.path.dirname(os.path.abspath(__file__))
Templates_preparation_folder = os.path.join(Run_outputs_folder, 'template_bank_preparation_files')
M_1_3_r_selected_list_filename = 'm_1_3_r_kepler_selected.npy'
M_1_3_r_selected_list_file = os.path.join(Templates_preparation_folder, M_1_3_r_selected_list_filename)


# For prior:
Templates_parameter_sets_prior_filename = 'templates_parameter_sets_prior.npy'
Cos_angles_max_prior_filename = 'cos_angles_max_prior.npy'
Params_all_prior_filename = 'params_all_prior.npy'
Templates_parameter_sets_prior_file = os.path.join(Templates_preparation_folder, Templates_parameter_sets_prior_filename)
Cos_angles_max_prior_file = os.path.join(Templates_preparation_folder, Cos_angles_max_prior_filename)
Params_all_prior_file = os.path.join(Templates_preparation_folder, Params_all_prior_filename)

# For the bank generation stage:
# (Those templates will have the random order they had in the Monte-Carlo generation)
Templates_bank_parameters_filename = 'template_bank_parameters.npy'
Templs_prepared_for_cos_fd_filename = 'templs_prepared_for_cos_fd.npy'
Templates_filename = 'templates.npy'
Indices_of_templates_filename = 'indices_of_templates.npy'
Templates_parameter_sets_filename = 'templates_parameter_sets.npy'
Cos_angles_max_filename = 'cos_angles_max.npy'
Params_all_filename = 'params_all.npy'
Templates_bank_parameters_file = os.path.join(Templates_preparation_folder, Templates_bank_parameters_filename)
Templs_prepared_for_cos_fd_file = os.path.join(Templates_preparation_folder, Templs_prepared_for_cos_fd_filename)
Templates_file = os.path.join(Templates_preparation_folder, Templates_filename)
Indices_of_templates_file = os.path.join(Templates_preparation_folder, Indices_of_templates_filename)
Templates_parameter_sets_file = os.path.join(Templates_preparation_folder, Templates_parameter_sets_filename)
Cos_angles_max_file = os.path.join(Templates_preparation_folder, Cos_angles_max_filename)
Params_all_file = os.path.join(Templates_preparation_folder, Params_all_filename)

# --------------------------------------------------------------------------------------------------------------------
# Boundaries for about FGK stars, maybe some A or B or M
Max_stellar_radius_in_solar_radii = 3.  # Solar radii
Min_stellar_radius_in_solar_radii = 0.5  # Solar radii
Max_stellar_mass_in_solar_masses = 2.5  # Solar masses
Min_stellar_mass_in_solar_masses = 0.3  # Solar masses
M13r_bin_edges_template_prior = [0.2, 0.75, 1.2, 1.7]
"""mass^(1/3) / radius (in solar units) edges to the bins used to bin the template prior tables"""

Per_min = 30  # days
Per_max = 500  # days
N_ecc = 20  # 60  # # Probability to transit at a fixed eccentricity depends strongly on eccentricity
N_w = 10  # 20
N_cos_incl = 300  # 1000 this splits the full range of inclinations, and then only the relevant ones are selected
Rp = 0.02  # assume the shape does not depend on the planet radius, although it is not exactly correct
N_shifts = 3  # we checked that with our parameters this can be 3 for everyone who needs shifts
Shift_step = dt_lc / N_shifts
Min_cos_shift = 0.93
Min_cos = 0.97

Limb_darkening_coefs_template_bank_generation = [
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
# m_1_3_r_s, period, a, inc, ecc, w, u1, u2
Earth_default_parameters = [1., 365., 215., 90., 0., 90., Limb_darkening_coefs_sun[0], Limb_darkening_coefs_sun[1]]
Template_support = 256
Min_template_duration_days = 2.1 * dt_lc
Max_template_duration_days = 1.5
