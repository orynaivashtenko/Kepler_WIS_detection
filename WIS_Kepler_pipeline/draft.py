#!/usr/bin/env python3

import numpy as np
import os
import sys

sys.path.append('/home/orynai/Kepler_Earths/WIS_Kepler_pipeline')
import search_pipeline as spip
import templates.template_module as tm
import auxiliary_functions as af
import simulations.lightcurve_simulation as sim
import file_operations as fop
import multiprocessing as mp

core_num = "1"
os.environ["OMP_NUM_THREADS"] = core_num  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = core_num  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = core_num  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = core_num  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = core_num  # export NUMEXPR_NUM_THREADS=6


# Simulation of injections for the planets which were also injections


def simulated_star_run(snr_true=None, P_true=100, Ph_0_true=50, rp=0.01,
                       # rp is not important because the depth will be defined by the snr_true
                       a=215, inc=90, ecc=0, w=90, u1=0.47, u2=0.23, Pmin=None, Pmax=None):
    if Pmin is None:
        Pmin = P_true - 1.
    if Pmax is None:
        Pmax = P_true + 1.
    np.random.seed(seed=None)
    t, d = sim.generate_lc_quarters(T=1600, dt=tm.dt_lc, flux0=1e5, source='stored_smoothed_asd_from_fft',
                                    asd_max=None, asd_min=None, pow_asd=None, freq_shift=None,
                                    variations_relative=0.01, variations_absolute=None, std_mean_quarters=0.1, sensitivity_std=0.05,
                                    std_read_relative=0.,
                                    planet=True, period=P_true, template_num=None, transit_duration=None, first_transit_center=Ph_0_true,
                                    use_physical_parameters=True, rp=rp, a=a, inc=inc, ecc=ecc, w=w, u1=u1, u2=u2, supersampling=True,
                                    snr=None, n_quarters=17, return_template=False, return_psd=False)
    klc = spip.KeplerLightCurve(target_id=None, get_data=False, times=t, fluxes=d)
    klc.condition_data()
    klc.condition_templates()
    klc.detect_single_transits()
    klc.connect_quarters()
    klc.interleave_shifts()
    klc.sum_periods(Pmin=Pmin, Pmax=Pmax)
    klc.veto_mes()
    klc.save_best_candidates(save_to_file=False)

    best_stat = 0
    best_candidates = klc.best_periodic_scores_data
    argm = np.argmin(best_candidates[4])
    ti, i = argm // len(best_candidates[4][0]), argm % len(best_candidates[4][0])
    # the structure is (25,10), where 10 is just 10 best per each template
    smin = best_candidates[4][ti][i]  # minimal statistic
    if smin < best_stat:
        best_stat = smin
        best_per = best_candidates[1][ti][i]
        best_temp = ti
        best_phase = best_candidates[3][ti][i]
        best_first_transit_time = best_candidates[5][ti][i][0]
        summed_transit_times = best_candidates[5][ti][i]
        summed_raw_stats = np.array(best_candidates[6][ti][i])
        summed_trends = np.array(best_candidates[7][ti][i])
        summed_vars = np.array(best_candidates[8][ti][i])
        summed_vars[summed_vars == 0] = np.nan
        ampls = summed_raw_stats / summed_vars
        ppms = summed_raw_stats / summed_vars / summed_trends * 1e6
        best_ppm = np.abs(np.nanmean(ppms))
        best_ampl = np.abs(np.nanmean(ampls))

    return best_stat, best_per, best_first_transit_time, best_temp


Saving_dir = '/SMALLDATA/orynai/paper_pipeline_performance_simulation'
trial_periods = [30., 147., 265., 382., 500.]
for period in trial_periods:
    Detection_filename = 'simulated_injection_injection_recovery_per_' + str(int(period)) + '.txt'

    def simulated_injection_injection_wrapper(injection_parameters_indices, saving_dir=Saving_dir, period=period,
                                              detection_filename=Detection_filename):
        for index in injection_parameters_indices:
            with open(os.path.join(Saving_dir, 'injection_parameters_per_' + str(int(period)) + '.txt'), 'r') as fil:
                lines = fil.readlines()
                lines = [line for line in lines if (line.strip()[0] != '#' and len(line.strip()) > 3)]

                injection_params = lines[index].split()
                per_inj = float(injection_params[0])
                a_inj = float(injection_params[1])
                inc_inj = float(injection_params[2])
                ecc_inj = float(injection_params[3])
                w_inj = float(injection_params[4])
                u1_inj = float(injection_params[5])
                u2_inj = float(injection_params[6])
                rp_inj = float(injection_params[7])
            phase_inj = np.random.uniform(0, per_inj)
            best_stat, best_per, best_first_transit_time, best_temp = simulated_star_run(P_true=per_inj, Ph_0_true=phase_inj,
                                                                                         rp=rp_inj, a=a_inj, inc=inc_inj, ecc=ecc_inj,
                                                                                         w=w_inj, u1=u1_inj, u2=u2_inj, Pmin=None,
                                                                                         Pmax=None)
            with open(os.path.join(saving_dir, detection_filename), 'a') as fil1:
                print(per_inj, a_inj, inc_inj, ecc_inj, w_inj, u1_inj, u2_inj, rp_inj, phase_inj, best_stat, best_per,
                      best_temp, best_first_transit_time, sep=' ', end='\n', file=fil1)


    n_working_cores = 100
    n_injections = 10000
    indices_of_injections = np.arange(n_injections).reshape((n_working_cores, int(n_injections / n_working_cores)))

    if __name__ == '__main__':
        with mp.Pool(n_working_cores) as pool:
            results = []
            results = pool.map_async(simulated_injection_injection_wrapper,
                                     [indices_of_injections[j] for j in range(n_working_cores)]).get()
