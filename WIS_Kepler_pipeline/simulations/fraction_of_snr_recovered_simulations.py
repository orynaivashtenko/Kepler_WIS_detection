#!/usr/bin/env python3
import os
core_num = "1"
os.environ["OMP_NUM_THREADS"] = core_num # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = core_num # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = core_num # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = core_num # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = core_num # export NUMEXPR_NUM_THREADS=6

import numpy as np
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import pipeline_parameters as pp
import search_pipeline as spip
import simulations.lightcurve_simulation as sim
import multiprocessing as mp


def find_min_stat(stats_periodic_interleaved,
                  stats_periodic_not_interleaved,
                  phases_interleaved,
                  phases_not_interleaved,
                  tmin):
    if np.nanmin(stats_periodic_interleaved) < np.nanmin(stats_periodic_not_interleaved):
        stats_periodic = stats_periodic_interleaved
        phases = phases_interleaved
        temp_addition = 0
    else:
        stats_periodic = stats_periodic_not_interleaved
        phases = phases_not_interleaved
        temp_addition = len(stats_periodic_interleaved[0])
    best_per_ind = np.nanargmin([np.nanmin(stats_periodic[ip]) for ip in range(len(stats_periodic))])
    best_temp_internal_ind = np.nanargmin(np.nanmin(stats_periodic[best_per_ind], axis=-1))
    best_phase_ind = np.nanargmin(stats_periodic[best_per_ind, best_temp_internal_ind])
    best_ftt = phases[best_phase_ind] + tmin
    best_temp = best_temp_internal_ind + temp_addition
    best_stat = np.nanmin(stats_periodic)
    return best_stat, best_ftt, best_per_ind, best_temp, best_phase_ind


def get_a_fixed_duration(duration, period):
    a = 1 / np.sin(np.pi * duration / period)  # Somehow this formula ensures fixed duration
    return a


def simulated_star_run(snr_true=10., per_true=100., duration=None, tmin=131., m_1_3_r_s=1., inc=90., ecc=0., w=90.,
                       u1=pp.Default_limb_darkening_coefs[0], u2=pp.Default_limb_darkening_coefs[1]):
    if duration is None:
        a = pp.get_a_from_p(period=per_true, m_1_3_r_s=m_1_3_r_s)
    else:
        a = get_a_fixed_duration(duration=duration, period=per_true)
    np.random.seed(seed=None)
    ftt_true = np.random.uniform(tmin, tmin+per_true)
    t, d, snr = sim.get_simulated_star(n_quarters=16, mean_flux=1e5, sigma_high_rel=1e-4, tmin=tmin, t_between=1.,
                                       inject_planet=True, period=per_true, first_transit_center=ftt_true, rp=0.02, a=a, inc=inc,
                                       ecc=ecc, w=w, u1=u1, u2=u2, snr_required=snr_true)

    klc = spip.KeplerLightCurve(target_id=None, get_data=False, times=t, fluxes=d)
    klc.condition_data()
    klc.condition_templates()
    klc.detect_single_transits()
    klc.connect_quarters()
    klc.interleave_shifts()
    klc.sum_periods(p_min=per_true-0.1, p_max=per_true+0.1)
    klc.calculate_periodic_scores_with_correction(save_uncorrected=True)

    best_stat, best_ftt, best_per_ind, best_temp, best_phase_ind = find_min_stat(
        stats_periodic_interleaved=klc.statistics_periodic_interleaved,
        stats_periodic_not_interleaved=klc.statistics_periodic_not_interleaved,
        phases_interleaved=klc._phases_interleaved,
        phases_not_interleaved=klc._phases_not_interleaved,
        tmin=klc.tmin)

    best_stat_u, best_ftt_u, best_per_ind_u, best_temp_u, best_phase_ind_u = find_min_stat(
        stats_periodic_interleaved=klc.uncorrected_stats_periodic[0],
        stats_periodic_not_interleaved=klc.uncorrected_stats_periodic[1],
        phases_interleaved=klc._phases_interleaved,
        phases_not_interleaved=klc._phases_not_interleaved,
        tmin=klc.tmin)

    best_stat_g, best_ftt_g, best_per_ind_g, best_temp_g, best_phase_ind_g = find_min_stat(
        stats_periodic_interleaved=klc.stats_gaussianized_periodic_interleaved,
        stats_periodic_not_interleaved=klc.stats_gaussianized_periodic_not_interleaved,
        phases_interleaved=klc._phases_interleaved,
        phases_not_interleaved=klc._phases_not_interleaved,
        tmin=klc.tmin)

    res = [[snr, ftt_true, per_true, a],
           [best_stat, best_ftt, klc.periods[best_per_ind], best_temp],
           [best_stat_u, best_ftt_u, klc.periods[best_per_ind_u], best_temp_u],
           [best_stat_g, best_ftt_g, klc.periods[best_per_ind_g], best_temp_g]]
    return res


Snrs_inj = np.arange(5, 15, 0.5)
Pers_inj = np.linspace(30, 500, 6)
Pars = dict(inc=90., ecc=0, w=0, u1=pp.Default_limb_darkening_coefs[0], u2=pp.Default_limb_darkening_coefs[1])

# Saving_dir = os.path.join(os.getcwd(), 'snr_recovery_results')
Saving_dir = '/home/orynai/Kepler_Notebooks/Run_12/Snr_recovery'
if not os.path.isdir(Saving_dir):
    os.mkdir(Saving_dir)
Counter_file = os.path.join(Saving_dir, 'counter.txt')


#%%
# ############################################################################################################
# FIXED DURATION SIMULATION.
# ############################################################################################################

# Simulation in which we feed the simulated injection-recovery.
# In the first simulation, we fix the period and the transit duraetion, and find stellar density which will provide it.
# In the second simulation, we assume that there is a star of Solar mass (set period and find semi-major axis
# with the Kepler's 3-rd law).
# We take a test planet around this star with some fixed parameters, as specified below.
# We try a grid of SNRs and periods, and the template duration is defined by the period,
# As if we physically put the same planet at different distances from the star. The
# duration of the template will be different for different templates

# for every grid node we
# run 100 simulations.

Filename = 'simulated_injection_recovery_fixed_dur.txt'

Duration = 0.4
snrs_true = Snrs_inj
pers_true = Pers_inj
snr_per_grid = [[snrs_true[i], pers_true[j]] for i in range(len(snrs_true)) for j in range(len(pers_true))]


def simulated_star_fixed_duration_run_wrapper(
        snr_per, duration=Duration, m_1_3_r_s=1., saving_dir=Saving_dir, filename=Filename,
        inc=Pars['inc'], ecc=Pars['ecc'], w=Pars['w'], u1=Pars['u1'], u2=Pars['u2']):
    snr_true = snr_per[0]
    per_true = snr_per[1]
    res = simulated_star_run(snr_true=snr_true, per_true=per_true, duration=duration, m_1_3_r_s=m_1_3_r_s, inc=inc, ecc=ecc, w=w,
                             u1=u1, u2=u2)
    snr_true, ftt_true, per_true, a = res[0]
    best_stat, best_ftt, best_per, best_temp = res[1]
    best_stat_u, best_ftt_u, best_per_u, best_temp_u = res[2]
    best_stat_g, best_ftt_g, best_per_g, best_temp_g = res[3]

    res_ret = (snr_true, duration, ftt_true, per_true, a, best_stat, best_ftt, best_per, best_temp,
               best_stat_u, best_ftt_u, best_per_u, best_temp_u, best_stat_g, best_ftt_g, best_per_g, best_temp_g)

    with open(os.path.join(saving_dir, filename), 'a') as fil1:
        print(*res_ret, file=fil1)
    return res_ret


n_repetitions = 100
for i in range(len(snr_per_grid)):
    if __name__ == '__main__':
        #     with mp.Pool(mp.cpu_count()) as pool:
        with mp.Pool(n_repetitions + 1) as pool:
            _ = []
            _ = pool.map_async(simulated_star_fixed_duration_run_wrapper, [snr_per_grid[i] for _ in range(n_repetitions)]).get()
    with open(Counter_file, 'w') as fil1:
        print('Fixed duration', i, '/', len(snr_per_grid), end=' ')

# %%
# ############################################################################################################
# FIXED STAR SIMULATION
# ############################################################################################################

# Simulation in which we feed the simulated injection-recovery.
# In the first simulation, we fix the period and the transit duraetion, and find stellar density which will provide it.
# In the second simulation, we assume that there is a star of Solar mass (set period and find semi-major axis
# with the Kepler's 3-rd law).
# We take a test planet around this star with some fixed parameters, as specified below.
# We try a grid of SNRs and periods, and the template duration is defined by the period,
# As if we physically put the same planet at different distances from the star. The
# duration of the template will be different for different templates

# for every grid node we
# run 100 simulations.
Filename = 'simulated_injection_recovery_fixed_star.txt'

snrs_true = Snrs_inj
pers_true = Pers_inj
snr_per_grid = [[snrs_true[i], pers_true[j]] for i in range(len(snrs_true)) for j in range(len(pers_true))]


def simulated_star_fixed_star_run_wrapper(
        snr_per, duration=None, m_1_3_r_s=1., saving_dir=Saving_dir, filename=Filename,
        inc=Pars['inc'], ecc=Pars['ecc'], w=Pars['w'], u1=Pars['u1'], u2=Pars['u2']):
    snr_true = snr_per[0]
    per_true = snr_per[1]
    res = simulated_star_run(snr_true=snr_true, per_true=per_true, duration=duration, m_1_3_r_s=m_1_3_r_s, inc=inc, ecc=ecc, w=w,
                             u1=u1, u2=u2)
    snr_true, ftt_true, per_true, a = res[0]
    best_stat, best_ftt, best_per, best_temp = res[1]
    best_stat_u, best_ftt_u, best_per_u, best_temp_u = res[2]
    best_stat_g, best_ftt_g, best_per_g, best_temp_g = res[3]

    res_ret = (snr_true, duration, ftt_true, per_true, a, best_stat, best_ftt, best_per, best_temp,
               best_stat_u, best_ftt_u, best_per_u, best_temp_u, best_stat_g, best_ftt_g, best_per_g, best_temp_g)

    with open(os.path.join(saving_dir, filename), 'a') as fil1:
        print(*res_ret, file=fil1)
    return res_ret


n_repetitions = 100
for i in range(0, len(snr_per_grid)):
    if __name__ == '__main__':
        #     with mp.Pool(mp.cpu_count()) as pool:
        with mp.Pool(n_repetitions + 1) as pool:
            _ = []
            _ = pool.map_async(simulated_star_fixed_star_run_wrapper, [snr_per_grid[i] for _ in range(n_repetitions)]).get()
    with open(Counter_file, 'w') as fil1:
        print('Fixed star', i, '/', len(snr_per_grid), end=' ')
