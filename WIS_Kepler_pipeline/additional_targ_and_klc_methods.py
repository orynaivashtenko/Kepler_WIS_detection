import matplotlib.pyplot as plt
import numpy as np
import os
import warnings
from auxiliary_functions import histogram
import search_pipeline as spip
import utils as ut
import copy

time_chunk_minimization_size = 20
threshold_to_std_veto = 35 ** 0.5


# def mask_villanova_eclipses(self):
#     """
#     Keys are 'KIC', 'period', 'period_err', 'bjd0', 'bjd0_err', 'pwidth', 'swidth', 'sep', 'morph'.
#
#     :return:
#     :rtype:
#     """
#     # TODO THIS IS NOT WRITTEN, THESE ARE JUST RANDOM LINES!!!
#     pass
#     assert self.is_villanova_binary, 'The target is not villanova binary!'
#     period = self.villanova_dict['period']
#         self.villanova_dict['pwidth'] * period
#         self.villanova_dict['swidth'] * period
#         self.villanova_dict['swidth'] * period
#         # Take all the transit times with TTV, take transit duration, take fractional TDV and mask.
#         transit_times = np.array(self.ttv_dict['T'], dtype=float)  # this is including TTVs
#         transit_durations = np.ones(len(transit_times)) * float(
#             self.koi_dict['koi_duration']) / 24  # we need it in days, and it was in hours
#         transit_durations = transit_durations * (1. + np.array(self.ttv_dict['TDV'], dtype=float))  # add TDVs
#         self.klc.mask_ephemeris(durations=transit_durations, transit_times=transit_times)
#         #
#         #
#         # # Take all the transit times with TTV, take transit duration, take fractional TDV and mask.
#         # transit_times = np.array(self.ttv_dict['T'], dtype=float)  # this is including TTVs
#         # transit_durations = np.ones(len(transit_times)) * float(self.koi_dict['koi_duration'])/24  # we need it in days, and it was in hours
#         # transit_durations = transit_durations * (1. + np.array(self.ttv_dict['TDV'], dtype=float))  # add TDVs
#         # self.klc.mask_planet(durations=transit_durations,  # we need it in days, and it was in hours
#         #                      transit_times=transit_times)
#
#         #     with open(os.path.join(Default_files_folder, Default_villanova_dict_filename), 'rb') as fl:
#         #         vilcat = pickle.load(fl)
#         #     with open('/home/orynai/Kepler_Notebooks/villanova_catalog', 'r') as vil_file:
#         #         lines = vil_file.readlines()
#         #     # print(*lines[:100])
#         #
#         #     lines = [(line.strip()).split(',') for line in lines if ((len(line.strip()) > 0) and (line.strip())[0] != '#')]
#         #
#         #     # print(*lines[:10])
#         #
#         #     vil_periods = np.array([float(line[1]) for line in lines])  # error is about 0.0000002 days
#         #     vil_phases = np.array([float(line[3]) for line in lines])  # error is about one lc exposure time
#         #     vil_kmags = np.array([float(line[8]) for line in lines])
#         #     vil_relevant_period_indices = vil_periods > 1  # we keep harmonics, but we do not need some 0.1 day periods
#         #
#         #     vil_periods = vil_periods[vil_relevant_period_indices]
#         #     vil_phases = vil_phases[vil_relevant_period_indices]
#         #     vil_kmags = vil_kmags[vil_relevant_period_indices]
#         #
#         #     # Kepler time is measured in BJD - 2454833
#         #     # Villanova phases are in BJD - 2400000
#         #     vil_phases = vil_phases + 2400000. - 2454833
#         #
#         #
#         #


# def generate_injection_parameters_repetitions(targ, n_injections=n_injections_completeness, add_to_existing_file=False,
#                                               per_wing_injection_relative=Per_wing_injection_and_slides,
#                                               injection_parameters_filename_postfix=Injection_parameters_filename_postfix,
#                                               all_templates_injection_parameters_filename_postfix=
#                                               All_templates_injection_parameters_filename_postfix
#                                               ):
#     """
#     This is not so randomized because dor one ecc we generate many other parameters, but ecc is always teh same.
#     I wrote a better one where every time everything is randomized
#     :param n_injections:
#     :type n_injections:
#     :param add_to_existing_file:
#     :type add_to_existing_file:
#     :param per_wing_injection_relative: days to the left and the same to the right used in the time slides in units of period
#     :type per_wing_injection_relative: float
#     :param injection_parameters_filename_postfix:
#     :type injection_parameters_filename_postfix:
#     :param all_templates_injection_parameters_filename_postfix:
#     :type all_templates_injection_parameters_filename_postfix:
#     :return:
#     :rtype:
#
#     TODO: DO NOT FORGET THE BUG AND LOOK AT THE TRANSIT TIME AND NOT THE BADLY DEFINED PHASE!
#     :return:
#     """
#     if targ.stellar_mass is None or targ.stellar_radius is None:
#         targ.failures += 'Unable to generate correct injections because of invalid stellar radius or mass'
#         m_1_3_r_s = 1.
#     else:
#         m_1_3_r_s = targ.stellar_mass ** (1 / 3) / targ.stellar_radius
#     targ.m_1_3_r_s = m_1_3_r_s  # M**(1/3)*R[in M_sun and R_sun]
#     if (targ.target_id + '_' + injection_parameters_filename_postfix in os.listdir(targ.output_location)) and not add_to_existing_file:
#         with open(os.path.join(targ.output_location, targ.target_id + '_' + injection_parameters_filename_postfix), 'w'):
#             pass  # will delete the contents of the file
#
#     effective_wf_fd = np.load('templates/effective_whitening_filter_fourier_domain.npy', allow_pickle=True)
#     temp_bank_whit_norm = np.load('templates/temp_bank_whit_norm.npy', allow_pickle=True)
#     template_shifts = spip.read_quarters_from_file(spip.shifts_template_bank_file, dimension=2)
#     n_templates = len(template_shifts)  # it is 25, not 45
#     actual_template_indices = [j for j in range(len(template_shifts)) for _ in range(len(template_shifts[j]))]
#     # actual_template_indices = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 11, 12,
#     #                            13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
#     assert (len(actual_template_indices) == len(temp_bank_whit_norm)), 'Lengths of the bank and its indices should match!'
#     t_length = (len(effective_wf_fd) - 1) * 2 * tm.dt_lc  # len like that because it was with rfft #0.02043368 is long cadence in days
#     t = tm.time(T=t_length + tm.dt_lc)
#     period = targ.best_per
#
#     targ.per_wing_injection_and_slides = per_wing_injection_relative * targ.best_per
#
#     d_ecc = 0.01
#     n_w = 100
#     d_inc = 0.02
#     u = (0.47, 0.23)  # limb darkening coefficients. nothing is too sensitive to them.
#     snr = 7  # snr is just for leakage
#     n_per = 10
#
#     templates_parameter_sets = [[] for _ in range(n_templates)]
#     for period in np.random.uniform(targ.best_per - targ.per_wing_injection_and_slides,
#                                     targ.best_per + targ.per_wing_injection_and_slides, n_per):
#         a = (G / (4 * np.pi ** 2) * (period * 24 * 60 * 60) ** 2) ** (1 / 3) * m_1_3_r_s * (
#                 M_sun ** (1 / 3) / R_sun)  # 3rd Kepler law #in units of R
#         # a = (G * M /(4*np.pi**2) * (per*24*60*60)**2)**(1/3) / R #3rd Kepler law #in units of R
#         # r_min = a*(1-e) -> 2*r_star=a*(1-e_max)
#         e_max = 1 - 2 / a
#         if e_max < 0:
#             e_range = [0.]
#         else:
#             e_range = np.random.uniform(0, e_max, int(np.round(e_max / d_ecc)))
#             if len(e_range) == 0:
#                 e_range = [0.]
#         for ecc in e_range:
#             ws = np.random.uniform(0, 360, n_w)
#             for w in ws:
#                 # i_max = np.arccos(1/a)*180/np.pi
#                 i_max = np.arccos(1 / a * (1 + ecc * np.cos(w / 180 * np.pi)) / (1 - ecc ** 2)) * 180 / np.pi
#                 for inc in np.random.uniform(i_max, 90, int(np.round((90 - i_max) / d_inc))):
#                     rp = 0.02  # assume the shape does not depend on the planet radius, although it is not exactly correct
#                     flux_obs = 1. - tm.flux_par(t=t, params=None, per=period, rp=rp, a=a, inc=inc, ecc=ecc, w=w, u1=u[0], u2=u[1], t0=0,
#                                                 supersample_factor=20, exp_time=tm.dt_lc)
#                     norm = np.linalg.norm(flux_obs)
#                     if norm > 0:
#                         wh_flux_obs = np.fft.rfft(np.fft.ifftshift(flux_obs)) * effective_wf_fd
#                         wh_flux_obs = wh_flux_obs / np.linalg.norm(wh_flux_obs)
#                         if snr is not None:
#                             wh_flux_obs = snr * wh_flux_obs + np.fft.rfft(np.random.normal(0, 1, (len(effective_wf_fd) - 1) * 2)) / (
#                                     (len(effective_wf_fd) - 1) * 2) ** 0.5
#                             wh_flux_obs = wh_flux_obs / np.linalg.norm(wh_flux_obs)
#                         cos_angles = [wh_flux_obs @ temp_bank_whit_norm[j] for j in range(len(temp_bank_whit_norm))]
#                         args = np.argsort(cos_angles)
#                         ind_1 = actual_template_indices[args[-1]]
#                         ind_2 = actual_template_indices[args[-2]]
#                         # templates_parameter_sets[ind_1].append([a, inc, ecc, w, u[0], u[1], rp])
#                         templates_parameter_sets[ind_1].append([period, a, inc, ecc, w, u[0], u[1]])
#                         if ind_2 != ind_1:
#                             templates_parameter_sets[ind_2].append([period, a, inc, ecc, w, u[0], u[1]])
#
#     np.save(os.path.join(targ.output_location, targ.target_id + '_' + all_templates_injection_parameters_filename_postfix),
#             np.array(templates_parameter_sets, dtype=object), allow_pickle=True)
#
#     log_rp_edges, period_edges, occurrences_table = get_occurrence_table(targ.stellar_radius)
#     rp_occurrences = occurrences_table[-1]
#     # We do not know occurrence priors for periods > 400 days, so just use the same ones as for 400 days
#     # For other periods, it will be checked right now:
#     for i in range(len(period_edges) - 1):
#         if (targ.best_per > period_edges[i]) and (targ.best_per <= period_edges[i + 1]):
#             rp_occurrences = occurrences_table[i]
#             break
#     # TODO: account for the case when the value is near the edge of the bin, then the density should be estimated from the two Bins
#
#     n_mult = int(np.ceil(n_injections / np.sum(rp_occurrences)))  # factor to multiply the occurrences to get the generation numbers
#     rp_numbers = (np.ceil(np.array(rp_occurrences) * n_mult)).astype(int)  # numbers of rp to generate
#     rps = []
#     for i in range(len(log_rp_edges) - 1):
#         rps = rps + list(np.exp(np.random.uniform(log_rp_edges[i], log_rp_edges[i + 1], rp_numbers[i])))
#     np.random.shuffle(rps)
#
#     best_temp_par_list_0 = templates_parameter_sets[targ.best_temp]
#     # select indices of the parameters to match the number (maybe with repetitions, but it is not important because there
#     # will be random radii and phases)
#     selection_indices = np.random.choice(np.arange(len(best_temp_par_list_0)), n_injections, replace=True)
#     targ.best_temp_injection_par_list = list(np.array(best_temp_par_list_0, dtype=object)[selection_indices])
#     targ.best_temp_injection_par_list = [list(targ.best_temp_injection_par_list[i]) + [rps[i]] for i in range(
#         len(targ.best_temp_injection_par_list))]
#
#     targ.save_injection_parameters()


def minimize_over_chunks(klc, time_chunk_size=time_chunk_minimization_size):
    """
    Finds the minimal (best, because snr is negative by convention) running statistics
    over subbin shifter templates and over time chunks of length `time_chunk_size`.
    This is done for all the different templates and all the quarters.

    Parameters
    ----------
    klc : KeplerLightCurve
    time_chunk_size : TYPE, optional
        DESCRIPTION. The default is time_chunk_minimization_size.

    Returns
    -------
    None.

    """
    klc.indices_of_min = [[[] for q in klc.quarters] for tempnum in range(len(klc.indices_of_templates))]
    klc.quasi_indep_snr = [[[] for q in klc.quarters] for tempnum in range(len(klc.indices_of_templates))]
    for tempnum in range(len(klc.indices_of_templates)):
        for q in klc.quarters:
            for timind in np.arange(0, len(
                    klc.statistics_snr[klc.indices_of_templates[tempnum][0]][q]) - time_chunk_size,
                                    time_chunk_size):
                # TODO:INCLUDE MASK
                klc.indices_of_min[tempnum][q].append(
                    np.argmin([np.min(klc.statistics_snr[ti][q][timind:timind + time_chunk_size])
                               for ti in np.arange(klc.indices_of_templates[tempnum][0],
                                                   klc.indices_of_templates[tempnum][1])]))
                klc.quasi_indep_snr[tempnum][q].append(
                    np.min([np.min(klc.statistics_snr[ti][q][timind:timind + time_chunk_size])
                            for ti in np.arange(klc.indices_of_templates[tempnum][0],
                                                klc.indices_of_templates[tempnum][1])]))
                

def veto_stat_outliers(klc, threshold_to_std=threshold_to_std_veto):
    # template_widths = [len(np.where(klc.initial_templates[tt]>0)[0]) for tt in range(len(klc.initial_templates))]
    whitened_template_widths = [[len(np.where(
        np.abs(klc.whitened_templates_td[tt][q]) > 0.5 * np.abs(np.min(klc.whitened_templates_td[tt][q])))[0])
                                 for q in klc.quarters] for tt in range(len(klc.initial_templates))]
    for tt in range(len(klc.statistics_raw)):
        for q in range(len(klc.statistics_raw[tt])):
            half_wid = (whitened_template_widths[tt][q] + 1) // 2
            var_stat = np.var(klc.statistics_raw[tt][q][klc.stat_masks[tt][q]])
            outlier_mask = klc.statistics_raw[tt][q] ** 2 < 3 ** 2 * var_stat  # for variance measurement only
            var_stat = np.var(klc.statistics_raw[tt][q][outlier_mask & klc.stat_masks[tt][q]])
            outlier_indices = np.where(klc.statistics_raw[tt][q] ** 2 > threshold_to_std ** 2 * var_stat)[
                0]  # the actual measurement

            length = len(klc.stat_masks[tt][q])
            for ind in outlier_indices:
                klc.stat_masks[tt][q][max(0, ind - half_wid): min(ind + half_wid, length)] = False


def veto_around_min(klc):
    # take moving chunks of some like template width.
    # if the min is masked, mask everything.
    whitened_template_widths = [[len(np.where(
        np.abs(klc.whitened_templates_td[tt][q]) > 0.5 * np.abs(np.min(klc.whitened_templates_td[tt][q])))[0])
                                 for q in klc.quarters] for tt in range(len(klc.initial_templates))]
    for tt in range(len(klc.statistics_raw)):
        for q in range(len(klc.statistics_raw[tt])):
            wid = whitened_template_widths[tt][q]
            half_wid = (whitened_template_widths[tt][q] + 1) // 2
            for ind in np.arange(0, len(klc.statistics_raw[tt][q]) - half_wid, half_wid):
                indmin = ind + np.nanargmin(klc.statistics_raw[tt][q][ind: ind + wid])
                if not klc.stat_masks[tt][q][indmin]:
                    if klc.statistics_raw[tt][q][indmin] == np.nanmin(klc.statistics_raw[tt][q][
                                                                      max(0, indmin - half_wid): min(
                                                                               indmin + half_wid, len(
                                                                                   klc.statistics_raw[tt][
                                                                                       q]))]):
                        klc.stat_masks[tt][q][max(0, indmin - half_wid): min(indmin + half_wid, len(
                            klc.statistics_raw[tt][q]))] = False



# plotting methods
def plot_results(klc,
                 plot_all=None,
                 original_data=None,
                 psds=None,
                 whitened_fluxes_td=None,
                 whitened_fluxes_fd=None,
                 single_event_statistic_snr_without_mask=None,
                 single_event_statistic_before_summation=None,
                 single_event_statistic_snr=None,
                 fluxes_with_veto=None,
                 histogram_single_event=None,
                 summed_statistic_minimal_of_phase=None,
                 summed_statistic_of_phase=None,
                 histogram_summed_statistic=None,
                 minimal_statistic_periods=None,
                 before_summation=False,
                 temp_num=None, period=None, per_ind=None, nbins_hist=25,
                 figsize=None, edgelist_hist=None, edgemin_hist=None, edgemax_hist=None
                 ):
    """
    klc : KeplerLightCurve
    """
    if figsize is None:
        width = 12
        height = 5
    else:
        width = figsize[0]
        height = figsize[1]

    if plot_all == True:
        if original_data is None:
            original_data = True
        if psds is None:
            psds = True
        if whitened_fluxes_td is None:
            whitened_fluxes_td = True
        if whitened_fluxes_fd is None:
            whitened_fluxes_fd = True
        if single_event_statistic_snr_without_mask is None:
            single_event_statistic_snr_without_mask = True
        if single_event_statistic_before_summation is None:
            single_event_statistic_before_summation = True
        if single_event_statistic_snr is None:
            single_event_statistic_snr = True
        if fluxes_with_veto is None:
            fluxes_with_veto = True
        if histogram_single_event is None:
            histogram_single_event = True
        if summed_statistic_minimal_of_phase is None:
            summed_statistic_minimal_of_phase = True
        if summed_statistic_of_phase is None:
            summed_statistic_of_phase = True
        if histogram_summed_statistic is None:
            histogram_summed_statistic = True
        if minimal_statistic_periods is None:
            minimal_statistic_periods = True
    elif plot_all == False:
        if original_data is None:
            original_data = False
        if psds is None:
            psds = False
        if whitened_fluxes_td is None:
            whitened_fluxes_td = False
        if whitened_fluxes_fd is None:
            whitened_fluxes_fd = False
        if single_event_statistic_snr_without_mask is None:
            single_event_statistic_snr_without_mask = False
        if single_event_statistic_before_summation is None:
            single_event_statistic_before_summation = False
        if single_event_statistic_snr is None:
            single_event_statistic_snr = False
        if fluxes_with_veto is None:
            fluxes_with_veto = False
        if histogram_single_event is None:
            histogram_single_event = False
        if summed_statistic_minimal_of_phase is None:
            summed_statistic_minimal_of_phase = False
        if summed_statistic_of_phase is None:
            summed_statistic_of_phase = False
        if histogram_summed_statistic is None:
            histogram_summed_statistic = False
        if minimal_statistic_periods is None:
            minimal_statistic_periods = False
    elif plot_all is None:
        if original_data is None:
            original_data = True
        if psds is None:
            psds = True
        if whitened_fluxes_td is None:
            whitened_fluxes_td = True
        if whitened_fluxes_fd is None:
            whitened_fluxes_fd = True
        if single_event_statistic_snr_without_mask is None:
            single_event_statistic_snr_without_mask = False
        if single_event_statistic_before_summation is None:
            single_event_statistic_before_summation = True
        if single_event_statistic_snr is None:
            single_event_statistic_snr = True
        if fluxes_with_veto is None:
            fluxes_with_veto = True
        if histogram_single_event is None:
            histogram_single_event = True
        if summed_statistic_minimal_of_phase is None:
            summed_statistic_minimal_of_phase = True
        if summed_statistic_of_phase is None:
            summed_statistic_of_phase = False
        if histogram_summed_statistic is None:
            histogram_summed_statistic = True
        if minimal_statistic_periods is None:
            minimal_statistic_periods = True

    if before_summation:
        if plot_all == True:
            original_data = True
            psds = True
            whitened_fluxes_td = True
            whitened_fluxes_fd = True
            single_event_statistic_snr_without_mask = False
            single_event_statistic_before_summation = False
            single_event_statistic_snr = True
            fluxes_with_veto = True
            histogram_single_event = True
            summed_statistic_minimal_of_phase = False
            summed_statistic_of_phase = False
            histogram_summed_statistic = False
            minimal_statistic_periods = False

        if temp_num is None:
            temp_num = 11
    else:
        if period is None:
            if per_ind is not None:
                period = klc.periods[per_ind]
            else:
                per_ind = klc.pind_measured
                period = klc.periods[klc.pind_measured]
        else:
            per_ind = np.argmin(np.abs(klc.periods - period))
            period = klc.periods[per_ind]

        if temp_num is None:
            temp_num = klc.tempind_measured

    if original_data:
        plt.figure(figsize=(width, height))
        for q in klc.quarters:
            plt.plot(klc.orig_times[q], klc.orig_fluxes[q])
        plt.title('Original unprocessed PDS lightcurves')
        plt.xlabel('time, days')
        plt.ylabel('flux')
        plt.tight_layout()
        plt.show()

    if psds:
        plt.figure(figsize=(width, height))
        for q in klc.quarters:
            plt.plot(klc.psdfreqs[q], np.log(np.abs(klc.psds[q])))
        plt.title('Measured PSDS of all the quarters')
        plt.xlabel('frequency, Hz')
        plt.ylabel('log[PSD]')
        plt.tight_layout()
        plt.show()

    if whitened_fluxes_td:
        plt.figure(figsize=(width, height))
        labeled = False
        for q in klc.quarters:
            plt.plot(klc.times[q], klc.whitened_fluxes_td[q])
            if not labeled:
                plt.plot(klc.times[q][~klc.big_hole_masks[q]],
                         klc.whitened_fluxes_td[q][~klc.big_hole_masks[q]],
                         'o', color='k', label='big masked holes')
                labeled = True
            else:
                plt.plot(klc.times[q][~klc.big_hole_masks[q]],
                         klc.whitened_fluxes_td[q][~klc.big_hole_masks[q]],
                         'o', color='k')
        plt.title('Whitened fluxes in time domain')
        plt.xlabel('time, days')
        plt.ylabel('whitened flux')
        plt.legend()
        plt.tight_layout()
        plt.show()

    if whitened_fluxes_fd:
        plt.figure(figsize=(width, height))
        for q in klc.quarters:
            plt.plot(klc.freqs[q], np.real(klc.whitened_fluxes_fd[q]))
        plt.title('Whitened fluxes in Fourier domain')
        plt.xlabel('frequency, Hz')
        plt.ylabel('whitened flux')
        plt.tight_layout()
        plt.show()

    if single_event_statistic_snr_without_mask:
        plt.figure(figsize=(width, height))
        for q in klc.quarters:
            plt.plot(klc.times[q], klc.statistics_snr[klc.indices_of_templates[temp_num][0]][q])
        plt.title('Single-event statistic in units of SNR for template #' + str(temp_num))
        plt.xlabel('time, days')
        plt.ylabel('statistic [SNR]')
        plt.tight_layout()
        plt.show()

    if single_event_statistic_snr:
        plt.figure(figsize=(width, height))
        for q in klc.quarters:
            if q % 2 == 0:
                clr = (0.2, 0.5, 0.8)
            else:
                clr = (0.2, 0.7, 0.4)
            plt.plot(klc.times[q], klc.statistics_snr[klc.indices_of_templates[temp_num][0]][q], color=clr)
        # plot what was masked in statistic masks
        label_made = False
        for q in klc.quarters:
            if not label_made:
                if (~klc.stat_masks[klc.indices_of_templates[temp_num][0]][q]).any():
                    label_made = True
                    plt.plot(klc.times[q][~klc.stat_masks[klc.indices_of_templates[temp_num][0]][q]],
                             klc.statistics_snr[klc.indices_of_templates[temp_num][0]][q][
                                 ~klc.stat_masks[klc.indices_of_templates[temp_num][0]][q]],
                             'o', color=(0.8, 0.5, 0), label='masked statistic', markersize=5)
            else:
                plt.plot(klc.times[q][~klc.stat_masks[klc.indices_of_templates[temp_num][0]][q]],
                         klc.statistics_snr[klc.indices_of_templates[temp_num][0]][q][
                             ~klc.stat_masks[klc.indices_of_templates[temp_num][0]][q]],
                         'o', color=(0.8, 0.5, 0), markersize=5)
        # plot what from this was masked in conditioning
        label_made = False
        for q in klc.quarters:
            if not label_made:
                if (~klc.stat_masks[klc.indices_of_templates[temp_num][0]][q]).any():
                    label_made = True
                    plt.plot(klc.times[q][~klc.big_hole_masks[q]],
                             klc.statistics_snr[klc.indices_of_templates[temp_num][0]][q][
                                 ~klc.big_hole_masks[q]],
                             'o', color='k', label='masked in conditioning', markersize=3)
            else:
                plt.plot(klc.times[q][~klc.big_hole_masks[q]],
                         klc.statistics_snr[klc.indices_of_templates[temp_num][0]][q][~klc.big_hole_masks[q]],
                         'o', color='k', markersize=3)
        if not before_summation:
            times_with_transits = np.array([])
            stat_with_transits = np.array([])
            for q in klc.quarters:
                times_with_transits = np.concatenate(
                    (times_with_transits, klc.times[q][klc.origrnal_transit_indices[q]]))
                stat_with_transits = np.concatenate((stat_with_transits,
                                                     klc.statistics_snr[klc.indices_of_templates[temp_num][0]][q][
                                                         klc.origrnal_transit_indices[q]]))
            plt.plot(times_with_transits, stat_with_transits, 'o', color=(0.8, 0, 0), label='transits',
                     markersize=4)

        plt.title('Single-event statistic in units of SNR for template #' + str(temp_num))
        plt.xlabel('time, days')
        plt.ylabel('statistic [SNR]')
        plt.legend()
        plt.tight_layout()
        plt.show()

    if single_event_statistic_before_summation:
        plt.figure(figsize=(width, height))
        if klc.is_min_interleaved:
            plt.plot(klc.time_interleaved, klc.statistics_interleaved[klc.tempind_min_interleaved], linewidth=2)
            plt.plot(klc.time_interleaved[klc.summed_transit_indices],
                     klc.statistics_interleaved[klc.tempind_min_interleaved][klc.summed_transit_indices], 'o',
                     label='summed values')
        else:
            plt.plot(klc.time_not_interleaved, klc.statistics_not_interleaved[klc.tempind_min_not_interleaved],
                     linewidth=2)
            plt.plot(klc.time_not_interleaved[klc.summed_transit_indices],
                     klc.statistics_not_interleaved[klc.tempind_min_not_interleaved][klc.summed_transit_indices],
                     'o',
                     label='summed values')
        plt.title('Single-event statistic in initial units used for summation. template #' + str(temp_num))
        plt.xlabel('time, days')
        plt.ylabel('statistic raw corrected for moving variance')
        plt.legend()
        plt.tight_layout()
        plt.show()

    if fluxes_with_veto:
        plt.figure(figsize=(width, height))
        labeled_chi_squared_veto = False
        labeled_maximal_deviation_std_veto = False
        labeled_maximal_deviation_amplitude_veto = False
        labeled_extreme_position_veto = False
        labeled_parts_len_veto = False
        labeled_parts_snr_veto = False
        labeled_inner_product_veto = False
        labeled_template_width_veto = False
        labeled = False
        colors = [(0, 0, 0), (0.3, 0.3, 0.3)]
        for q in klc.quarters:
            plt.plot(klc.times[q], klc.fluxes[q], color=colors[q % 2], linewidth=2)
            if not labeled:
                plt.plot(klc.times[q][~klc.big_hole_masks[q]], klc.fluxes[q][~klc.big_hole_masks[q]], 'o',
                         color=(0, 0.5, 1), markersize=5, label='big inpainted holes')
                labeled = True
            else:
                plt.plot(klc.times[q][~klc.big_hole_masks[q]], klc.fluxes[q][~klc.big_hole_masks[q]], 'o',
                         color=(0, 0.5, 1), markersize=5)
        for q in klc.quarters:
            chi_squared_veto = []
            maximal_deviation_std_veto = []
            maximal_deviation_amplitude_veto = []
            extreme_position_veto = []
            parts_len_veto = []
            parts_snr_veto = []
            inner_product_veto = []
            template_width_veto = []
            tin = klc.indices_of_templates[temp_num][0]
            for ind in range(len(klc.veto_bad_trigger_inds[tin][q][0])):
                if klc.veto_bad_trigger_inds[tin][q] != [[], [], []]:
                    if 'chi_squared' in klc.veto_bad_trigger_inds[tin][q][1][ind]:
                        chi_squared_veto.append(klc.veto_bad_trigger_inds[tin][q][0][ind])
                    if 'maximal_deviation_std' in klc.veto_bad_trigger_inds[tin][q][1][ind]:
                        maximal_deviation_std_veto.append(klc.veto_bad_trigger_inds[tin][q][0][ind])
                    if 'maximal_deviation_amplitude' in klc.veto_bad_trigger_inds[tin][q][1][ind]:
                        maximal_deviation_amplitude_veto.append(klc.veto_bad_trigger_inds[tin][q][0][ind])
                    if 'extreme_position' in klc.veto_bad_trigger_inds[tin][q][1][ind]:
                        extreme_position_veto.append(klc.veto_bad_trigger_inds[tin][q][0][ind])
                    if 'inconsistent_parts_len' in klc.veto_bad_trigger_inds[tin][q][1][ind]:
                        parts_len_veto.append(klc.veto_bad_trigger_inds[tin][q][0][ind])
                    if 'inconsistent_parts_snr' in klc.veto_bad_trigger_inds[tin][q][1][ind]:
                        parts_snr_veto.append(klc.veto_bad_trigger_inds[tin][q][0][ind])
                    if 'inner_product' in klc.veto_bad_trigger_inds[tin][q][1][ind]:
                        inner_product_veto.append(klc.veto_bad_trigger_inds[tin][q][0][ind])
                    if 'template_width' in klc.veto_bad_trigger_inds[tin][q][1][ind]:
                        template_width_veto.append(klc.veto_bad_trigger_inds[tin][q][0][ind])
            if len(chi_squared_veto) > 0:
                if not labeled_chi_squared_veto:
                    plt.plot(klc.times[q][chi_squared_veto], klc.fluxes[q][chi_squared_veto], 'o', color='r',
                             markersize=5, label='chi squared veto')
                    labeled_chi_squared_veto = True
                else:
                    plt.plot(klc.times[q][chi_squared_veto], klc.fluxes[q][chi_squared_veto], 'o', color='r',
                             markersize=5)
            if len(maximal_deviation_std_veto) > 0:
                if not labeled_maximal_deviation_std_veto:
                    plt.plot(klc.times[q][maximal_deviation_std_veto], klc.fluxes[q][maximal_deviation_std_veto],
                             'o', color=(1, 0.5, 0), markersize=5, label='maximal deviation std veto')
                    labeled_maximal_deviation_std_veto = True
                else:
                    plt.plot(klc.times[q][maximal_deviation_std_veto], klc.fluxes[q][maximal_deviation_std_veto],
                             'o', color=(1, 0.5, 0), markersize=5)
            if len(maximal_deviation_amplitude_veto) > 0:
                if not labeled_maximal_deviation_amplitude_veto:
                    plt.plot(klc.times[q][maximal_deviation_amplitude_veto],
                             klc.fluxes[q][maximal_deviation_amplitude_veto], 'o', color=(0.6, 0.4, 0),
                             markersize=5, label='maximal deviation amplitude veto')
                    labeled_maximal_deviation_amplitude_veto = True
                else:
                    plt.plot(klc.times[q][maximal_deviation_amplitude_veto],
                             klc.fluxes[q][maximal_deviation_amplitude_veto], 'o', color=(0.6, 0.4, 0),
                             markersize=5)
            if len(extreme_position_veto) > 0:
                if not labeled_extreme_position_veto:
                    plt.plot(klc.times[q][extreme_position_veto], klc.fluxes[q][extreme_position_veto], 'o',
                             color=(0.4, 0.3, 1), markersize=5, label='extreme position veto')
                    labeled_extreme_position_veto = True
                else:
                    plt.plot(klc.times[q][extreme_position_veto], klc.fluxes[q][extreme_position_veto], 'o',
                             color=(0.4, 0.3, 1), markersize=5)
            if len(parts_len_veto) > 0:
                if not labeled_parts_len_veto:
                    plt.plot(klc.times[q][parts_len_veto], klc.fluxes[q][parts_len_veto], 'o', color=(0.7, 0, 1),
                             markersize=5, label='inconsistent parts len')
                    labeled_parts_len_veto = True
                else:
                    plt.plot(klc.times[q][parts_len_veto], klc.fluxes[q][parts_len_veto], 'o', color=(0.7, 0, 1),
                             markersize=5)
            if len(parts_snr_veto) > 0:
                if not labeled_parts_snr_veto:
                    plt.plot(klc.times[q][parts_snr_veto], klc.fluxes[q][parts_snr_veto], 'o', color=(0.9, 0, 1),
                             markersize=5, label='inconsistent parts snr')
                    labeled_parts_snr_veto = True
                else:
                    plt.plot(klc.times[q][parts_snr_veto], klc.fluxes[q][parts_snr_veto], 'o', color=(0.9, 0, 1),
                             markersize=5)

            if len(inner_product_veto) > 0:
                if not labeled_inner_product_veto:
                    plt.plot(klc.times[q][inner_product_veto], klc.fluxes[q][inner_product_veto], 'o',
                             color=(0.6, 0.6, 0.6), markersize=3, label='inner product veto')
                    labeled_inner_product_veto = True
                else:
                    plt.plot(klc.times[q][inner_product_veto], klc.fluxes[q][inner_product_veto], 'o',
                             color=(0.6, 0.6, 0.6), markersize=3)
            if len(template_width_veto) > 0:
                if not labeled_template_width_veto:
                    plt.plot(klc.times[q][template_width_veto], klc.fluxes[q][template_width_veto], 'o',
                             color=(0, 0.9, 0), markersize=3, label='template width veto')
                    labeled_template_width_veto = True
                else:
                    plt.plot(klc.times[q][template_width_veto], klc.fluxes[q][template_width_veto], 'o',
                             color=(0, 0.9, 0), markersize=3)
        if not before_summation:
            times_with_transits = np.array([])
            fluxes_with_transits = np.array([])
            for q in klc.quarters:
                times_with_transits = np.concatenate(
                    (times_with_transits, klc.times[q][klc.origrnal_transit_indices[q]]))
                fluxes_with_transits = np.concatenate(
                    (fluxes_with_transits, klc.fluxes[q][klc.origrnal_transit_indices[q]]))
            plt.plot(times_with_transits, fluxes_with_transits, 'o', color=(0.3, 1, 0.3), markersize=3,
                     label='transits')
        plt.title('Conditioned fluxes in time domain with vetos for template #' + str(temp_num))
        plt.xlabel('time, days')
        plt.ylabel('flux')
        plt.legend(fontsize=(plt.rcParams['font.size'] - 3))
        plt.tight_layout()
        plt.show()

    if histogram_single_event:
        tin = klc.indices_of_templates[temp_num][0]
        datstat = np.array([])
        for q in klc.quarters:
            datstat = np.concatenate((datstat, klc.statistics_snr[tin][q][klc.stat_masks[tin][q]]))
        histogram(datstat, n=nbins_hist, edgelist=edgelist_hist, edgemin=edgemin_hist, edgemax=edgemax_hist,
                  xscale='square', yscale='log',
                  density=False, normalize=True, ylogshift=True, logzeros='zero',
                  return_results=False, plot=True, new_figure=True, ax=None, figsize=(width / 3 * 2, height),
                  show=True,
                  xlabel='statistic squared', ylabel='log counts',
                  title='Histogram of statistic in units of snr (uncorrected) for all the quarters for template #' + str(
                      temp_num),
                  xlim=None, ylim=None,
                  color='b', linestyle='solid', marker='', linewidth=None, alpha=0.9, label=None,
                  loc_legend=None, plot_type='bar')

    if summed_statistic_minimal_of_phase:
        plt.figure(figsize=(width, height))
        if klc.is_min_interleaved:
            plt.plot(klc.phases_interleaved,
                     klc.statistics_periodic_interleaved[klc.pind_measured][klc.tempind_min_interleaved])
            plt.plot([klc.phases_interleaved[klc.phind_measured]],
                     [klc.statistics_periodic_interleaved[klc.pind_measured][klc.tempind_min_interleaved][
                          klc.phind_measured]], 'o',
                     label='minimal value determined')
        else:
            plt.plot(klc.phases_not_interleaved,
                     klc.statistics_periodic_not_interleaved[klc.pind_measured][klc.tempind_min_not_interleaved])
            plt.plot([klc.phases_not_interleaved[klc.phind_measured]],
                     [klc.statistics_periodic_not_interleaved[klc.pind_measured][
                          klc.tempind_min_not_interleaved][klc.phind_measured]], 'o',
                     label='minimal value determined')
        plt.title('Summed statistic as function of phase for minimal parameters: period='
                  + str(np.round(klc.periods[klc.pind_measured], 2)) + ' days, template #' + str(klc.tempind_measured))
        plt.xlabel('initial phase, days')
        plt.ylabel('statistic, [SNR]')
        plt.legend()
        plt.tight_layout()
        plt.show()

    if summed_statistic_of_phase:
        plt.figure(figsize=(width, height))
        if temp_num in klc.interleaved_numbers:
            plt.plot(klc.phases_interleaved, klc.statistics_periodic_interleaved[per_ind][temp_num])
        else:
            temp_num_ind = np.where(klc.not_interleaved_numbers == temp_num)[0][0]
            plt.plot(klc.phases_not_interleaved, klc.statistics_periodic_not_interleaved[per_ind][temp_num_ind])
        plt.title(
            'Summed statistic as function of phase for period=' + str(period) + ' days, template #' + str(temp_num))
        plt.xlabel('initial phase, days')
        plt.ylabel('statistic, [SNR]')
        plt.tight_layout()
        plt.show()

    if histogram_summed_statistic:
        if temp_num in klc.interleaved_numbers:
            temp_num_ind = np.where(klc.interleaved_numbers == temp_num)[0][0]
            datstat = klc.statistics_periodic_interleaved[per_ind][temp_num_ind]
        else:
            temp_num_ind = np.where(klc.not_interleaved_numbers == temp_num)[0][0]
            datstat = klc.statistics_periodic_not_interleaved[per_ind][temp_num_ind]
        histogram(datstat[~np.isnan(datstat)], n=nbins_hist, edgelist=edgelist_hist, edgemin=edgemin_hist,
                  edgemax=edgemax_hist, xscale='square', yscale='log',
                  density=False, normalize=True, ylogshift=True, logzeros='zero',
                  return_results=False, plot=True, new_figure=True, ax=None, figsize=(width / 3 * 2, height),
                  show=True,
                  xlabel='statistic squared', ylabel='log counts',
                  title='Histogram of summed statistic for all phases for period=' + str(
                      np.round(period, 2)) + ' days, template #' + str(temp_num),
                  xlim=None, ylim=None,
                  color='b', linestyle='solid', marker='', linewidth=None, alpha=0.9, label=None,
                  loc_legend=None, plot_type='bar')

    if minimal_statistic_periods:
        mins = []
        for p in range(len(klc.periods)):
            mins.append(min(np.nanmin(klc.statistics_periodic_interleaved[p]),
                            np.nanmin(klc.statistics_periodic_not_interleaved[p])))
        plt.figure(figsize=(width, height))
        plt.plot(klc.periods, mins)
        plt.title('Minimum of summed statistic as function of period')
        plt.xlabel('periods, days')
        plt.ylabel('statistic, [SNR]')
        plt.tight_layout()
        plt.show()


def print_veto_profile(self, quarters, indices=None, times=None, temp_inds=12, path=None, filename=None):
    """
    klc method
    quarters : int or list of int or str 'all' or None.
        None is when time is specified and quarter has to be found
    indices : int or list of int or str 'all'

    time : float or list of float or str 'all'

    temp_inds : int or list of int or str 'all'

    file : str or None
    """
    if path is None:
        path = self.output_location
    if quarters == 'all':
        quarters = self.quarters
    elif isinstance(quarters, int):
        quarters = [quarters]
    else:
        assert hasattr(quarters, "__len__"), 'Invalid quarters value specified!'
        if hasattr(indices, "__len__") and len(indices) > 1:
            assert (len(quarters) == len(indices)), 'Undefined way to match indices to quarters!'
        elif hasattr(times, "__len__") and len(times) > 1:
            assert (len(quarters) == len(times)), 'Undefined way to match times to quarters!'

    if isinstance(temp_inds, int):
        temp_inds = [temp_inds]

    if isinstance(indices, int):
        indices = [indices]

    if isinstance(times, float):
        times = [times]

    if indices is None:
        indices = []
        if quarters is not None and len(quarters) == 1:
            for tim in times:
                ind1 = np.argmin(np.abs(self.times[quarters[0]] - tim))
                if np.abs(self.times[quarters[0]][ind1] - tim) > 2 * self.dt_days:
                    warnings.warn('indices not in this quarters!')
                else:
                    indices.append(ind1)
        else:
            quarters = []
            indices = []
            for tim in times:
                timedifmins = []
                indmins = []
                for q in self.quarters:
                    ind1 = np.argmin(np.abs(self.times[q] - tim))
                    indmins.append(ind1)
                    timedifmins.append(np.abs(self.times[q][ind1] - tim))
                qindmin = np.argmin(timedifmins)
                quarters.append(self.quarters[qindmin])
                indices.append(indmins[qindmin])

    if len(quarters) == 1:
        quarters = np.array([quarters[0]] * len(indices))

    if filename is None:
        filename0 = 'veto_klc' + str(self.target_id)
        filename = filename0
        num = 0
        while (filename in os.listdir(path)) or (filename + '.txt' in os.listdir(path)):
            num += 1
            filename = filename0 + '_' + str(num)
    if filename[-4:] != '.txt':
        filename = filename + '.txt'

    with open(os.path.join(path, filename), 'w') as file:
        for i in range(len(indices)):
            q = quarters[i]
            index = indices[i]
            if temp_inds == 'all':
                temp_inds = list(self.triggers[q][index].keys())
            for j in range(len(temp_inds)):
                temp_ind = temp_inds[j]
                print('quarter:', q, file=file)
                print('index:', index, file=file)
                print('template index:', temp_ind, file=file)
                if index not in self.triggers[q].keys():
                    print('not a trigger')
                    return None
                if temp_ind not in self.triggers[q][index].keys():
                    print('not a trigger')
                    return None
                print('whether passed vetting:', self.triggers[q][index][temp_ind][0], file=file)
                if not self.triggers[q][index][temp_ind][0]:
                    print('reasons why:', self.triggers[q][index][temp_ind][1], file=file)
                if len(self.triggers[q][index][temp_ind][2]) == 8:
                    print('chi squared:', self.triggers[q][index][temp_ind][2][0], file=file)
                    print('threshold chi squared:', self.triggers[q][index][temp_ind][2][1], file=file)
                    print('max dev std:', self.triggers[q][index][temp_ind][2][2], file=file)
                    print('threshold max dev std:', self.triggers[q][index][temp_ind][2][3], file=file)
                    print('max dev ampl:', self.triggers[q][index][temp_ind][2][4], file=file)
                    print('threshold max dev ampl:', self.triggers[q][index][temp_ind][2][5], file=file)
                    print('inner product abs:', np.abs(self.triggers[q][index][temp_ind][2][6]), file=file)
                    print('threshold inner product abs:', self.triggers[q][index][temp_ind][2][7], file=file)
                if len(self.triggers[q][index][temp_ind][3]) != 0:
                    print('chi2 parts snr local:', *self.triggers[q][index][temp_ind][3][0][:-1], file=file)
                    print('threshold chi2 parts snr local:', self.triggers[q][index][temp_ind][3][1], file=file)
                    print('chi2 parts snr total:', self.triggers[q][index][temp_ind][3][0][-1], file=file)
                    print('threshold chi2 parts snr total:', self.triggers[q][index][temp_ind][3][2], file=file)
                if len(self.triggers[q][index][temp_ind][4]) != 0:
                    print('chi2 parts len local:', *self.triggers[q][index][temp_ind][4][0][:-1], file=file)
                    print('threshold chi2 parts len local:', self.triggers[q][index][temp_ind][3][1], file=file)
                    print('chi2 parts len total:', self.triggers[q][index][temp_ind][4][0][-1], file=file)
                    print('threshold chi2 parts len total:', self.triggers[q][index][temp_ind][3][2], file=file)
                if len(self.triggers[q][index][temp_ind][5]) != 0:
                    print('snr2 of this trigger:', self.triggers[q][index][temp_ind][5][0], file=file)
                    print('snr2 of the best template:', self.triggers[q][index][temp_ind][5][1], file=file)
                print('', file=file)


def save_ses_quarters(self, path=None):
    """
    klc method
    Saves the essential SES results that will be used in the periodicity search
    as .npy file that can be read with `load_ses_quarters` funtion:
        self.times
        self.trends_orig_fluxes
        self.statistics_raw
        self.stat_masks
        self.stat_tot_vars

    Parameters
    ----------
    path : str
        path to the folder in which the data should be saved.

    Returns
    -------
    None.

    """
    if path is None:
        path = self.output_location

    len_data = sum(
        [self.support for q in self.quarters])  # is assumed to be support length times number of quarters
    ses_quarter_data = np.zeros((4 + 3 * self.templ_num_shifts, len_data), dtype=np.float32)
    ses_quarter_data[0][0] = self.support
    ses_quarter_data[0][1] = self.templ_num
    ses_quarter_data[0][2] = self.templ_num_shifts
    ses_quarter_data[0][3] = self.dt_days
    ses_quarter_data[0][4] = len(self.quarters)
    for tn in range(self.templ_num):
        ses_quarter_data[0][5 + tn] = self.indices_of_templates[tn][0]
    ind_last = 0
    for q in self.quarters:
        ses_quarter_data[1][ind_last + 0] = self.zero_padding_ends[q][0]
        ses_quarter_data[1][ind_last + 1] = self.zero_padding_ends[q][1]
        ses_quarter_data[1][ind_last + 2] = len(
            self.statistics_raw[0][q])  # this depends on whether data is cut from zeros or not
        ses_quarter_data[1][ind_last + 3] = len(self.times[q])
        for tn in range(self.templ_num_shifts):
            ses_quarter_data[1][ind_last + 4 + tn] = self.stat_tot_vars[tn][q]
        ses_quarter_data[2][ind_last:ind_last + len(self.times[q])] = self.times[q]
        ses_quarter_data[3][ind_last:ind_last + len(self.times[q])] = self.trends_orig_fluxes[q]
        row_pos = 4
        for k in range(self.templ_num_shifts):
            ses_quarter_data[k + row_pos][ind_last:ind_last + len(self.statistics_raw[k][q])] = \
                self.statistics_raw[k][q]
        row_pos += self.templ_num_shifts
        for k in range(self.templ_num_shifts):
            ses_quarter_data[k + row_pos][ind_last:ind_last + len(self.statistics_raw[k][q])] = \
                self.stat_masks[k][q]
        row_pos += self.templ_num_shifts
        for k in range(self.templ_num_shifts):
            ses_quarter_data[k + row_pos][ind_last:ind_last + len(self.statistics_raw[k][q])] = \
                self.statistics_snr_no_rank[k][q]
        ind_last += self.support
    if self.target_id is not None:
        np.save(os.path.join(path, str(self.target_id) + '_ses_quarter_data'),
                np.array(ses_quarter_data, dtype=object), allow_pickle=True)
    else:
        np.save(os.path.join(path, '_ses_quarter_data'), np.array(ses_quarter_data, dtype=object),
                allow_pickle=True)


def load_ses_quarters(self, filename=None):
    """
    klc method
    Only works for data created by `save_ses_quarters`. Maybe contains assertion bugs.

    Sets:
    self.times
    self.trends_orig_fluxes
    self.statistics_raw
    self.stat_masks
    self.stat_tot_vars
    self.dt_days

    :param filename: path to the .npy file where the data is stored.
    :type filename: str
    :return:
    :rtype:
    """
    if filename is None:
        filename = os.path.join(self.output_location, str(self.target_id) + '_ses_quarter_data.npy')
    else:
        assert filename[-4:] == '.npy', 'file should have .npy type!'
    ses_quarter_data = np.load(filename, allow_pickle=True)
    self.support = ses_quarter_data[0][0]
    self.templ_num = ses_quarter_data[0][1]
    self.templ_num_shifts = ses_quarter_data[0][2]
    self.dt_days = float(ses_quarter_data[0][3])
    self.quarters = range(int(ses_quarter_data[0][4]))
    self.indices_of_templates = []
    for tn in range(self.templ_num - 1):
        self.indices_of_templates.append([ses_quarter_data[0][5 + tn], ses_quarter_data[0][6 + tn]])
    self.indices_of_templates.append([ses_quarter_data[0][6 + self.templ_num - 1], self.templ_num_shifts])
    ind_last = 0
    self.zero_padding_ends = [[0, 0] for _ in self.quarters]
    self.whitened_fluxes_td = []
    self.times = []
    self.stat_tot_vars = [[] for _ in range(self.templ_num_shifts)]
    self.statistics_raw = [[] for _ in range(self.templ_num_shifts)]
    self.stat_masks = [[] for _ in range(self.templ_num_shifts)]
    self.statistics_snr_no_rank = [[] for _ in range(self.templ_num_shifts)]
    for q in self.quarters:
        self.zero_padding_ends[q][0] = ses_quarter_data[1][ind_last + 0]
        self.zero_padding_ends[q][1] = ses_quarter_data[1][ind_last + 1]
        len_time = ses_quarter_data[1][ind_last + 2]
        len_stat = ses_quarter_data[1][ind_last + 3]
        for tn in range(self.templ_num_shifts):
            self.stat_tot_vars[tn].append(ses_quarter_data[1][ind_last + 4 + tn])
        self.times.append(ses_quarter_data[2][ind_last:ind_last + len_time])
        self.trends_orig_fluxes.append(ses_quarter_data[3][ind_last:ind_last + len_time])
        row_pos = 4
        for k in range(self.templ_num_shifts):
            self.statistics_raw[k].append(ses_quarter_data[k + row_pos][ind_last:ind_last + len_stat])
        row_pos += self.templ_num_shifts
        for k in range(self.templ_num_shifts):
            self.stat_masks[k].append(ses_quarter_data[k + row_pos][ind_last:ind_last + len_stat])
        row_pos += self.templ_num_shifts
        for k in range(self.templ_num_shifts):
            self.statistics_snr_no_rank[k].append(ses_quarter_data[k + row_pos][ind_last:ind_last + len_stat])
        ind_last += self.support


def save_trigger_info(self, triggers, path=None, filename_postfix='trigger_info'):
    """
    klc method
    :param self:
    :type self:
    :param path:
    :type path:
    :param filename_postfix:
    :type filename_postfix:
    :return:
    :rtype:
    """
    if path is None:
        path = self.output_location
    np.save(os.path.join(path, str(self.target_id) + '_' + filename_postfix), np.array(triggers, dtype=object), allow_pickle=True)


def save_all_results(self, path=None, n_per_temp_saving=spip.N_best_saving):
    """
    klc method
    :param self:
    :type self:
    :param path:
    :type path:
    :param n_per_temp_saving:
    :type n_per_temp_saving:
    :return:
    :rtype:
    """
    if path is None:
        path = self.output_location
    self.save_conditioned_data(path)
    if ('triggers' in self.__dict__) and (self.triggers is not None):
        save_trigger_info(self, self.triggers, path)
    self.save_best_candidates(n_per_temp=n_per_temp_saving, path=path)

def determine_mins(self, ngap_moving_var=spip.Ngap_moving_var):
    """
    klc method
    Finds the best value of statistics and saves the corresponding parameters.

    Parameters
    ----------
    ngap_moving_var : TYPE, optional
        DESCRIPTION. The default is ngap_moving_var.

    Returns
    -------


    """
    if len(self.interleaved_numbers) > 0:
        pind_min_interleaved = np.nanargmin(  # argmin over periods
            [np.nanmin(  # min over templates
                [np.nanmin(  # min over phases
                    self.statistics_periodic_interleaved[pi][ti])
                    for ti in range(len(self.statistics_periodic_interleaved[pi]))])
                for pi in range(len(self.statistics_periodic_interleaved))])
        tempind_min_interleaved = np.nanargmin(  # argmin over templates
            [np.nanmin(  # min over phases
                self.statistics_periodic_interleaved[pind_min_interleaved][ti])  # in the best period
                for ti in range(len(self.statistics_periodic_interleaved[pind_min_interleaved]))])
        phasind_min_interleaved = np.nanargmin(  # argmin over phases
            self.statistics_periodic_interleaved[pind_min_interleaved][tempind_min_interleaved])
    else:
        pind_min_interleaved = np.nan
        tempind_min_interleaved = np.nan
        phasind_min_interleaved = np.nan

    if len(self.not_interleaved_numbers) > 0:
        pind_min_not_interleaved = np.nanargmin(  # argmin over periods
            [np.nanmin(  # min over templates
                [np.nanmin(  # min over phases
                    self.statistics_periodic_not_interleaved[pi][ti])
                    for ti in range(len(self.statistics_periodic_not_interleaved[pi]))])
                for pi in range(len(self.statistics_periodic_not_interleaved))])
        tempind_min_not_interleaved = np.nanargmin(  # argmin over templates
            [np.nanmin(  # min over phases
                self.statistics_periodic_not_interleaved[pind_min_not_interleaved][ti])  # in the best period
                for ti in range(len(self.statistics_periodic_not_interleaved[pind_min_not_interleaved]))])
        phasind_min_not_interleaved = np.nanargmin(  # argmin over phases
            self.statistics_periodic_not_interleaved[pind_min_not_interleaved][tempind_min_not_interleaved])
    else:
        pind_min_not_interleaved = np.nan
        tempind_min_not_interleaved = np.nan
        phasind_min_not_interleaved = np.nan

    if len(self.not_interleaved_numbers) > 0:
        pind_incorrect = np.nanargmin(  # argmin over periods
            [np.nanmin(  # min over templates
                [np.nanmin(  # min over phases
                    np.abs(self.statistics_periodic_not_interleaved[pi][ti]))
                    for ti in range(len(self.statistics_periodic_not_interleaved[pi]))])
                for pi in range(len(self.statistics_periodic_not_interleaved))])
        tempind_incorrect = np.nanargmin(  # argmin over templates
            [np.nanmin(  # min over phases
                np.abs(self.statistics_periodic_not_interleaved[pind_min_not_interleaved][ti]))
                # in the best period
                for ti in range(len(self.statistics_periodic_not_interleaved[pind_min_not_interleaved]))])
        self.std_free_place = ut.outlier_free_std(
            self.statistics_periodic_not_interleaved[pind_incorrect][tempind_incorrect])
    else:
        pind_incorrect = np.nanargmin(  # argmin over periods
            [np.nanmin(  # min over templates
                [np.nanmin(  # min over phases
                    np.abs(self.statistics_periodic_interleaved[pi][ti]))
                    for ti in range(len(self.statistics_periodic_interleaved[pi]))])
                for pi in range(len(self.statistics_periodic_interleaved))])
        tempind_incorrect = np.nanargmin(  # argmin over templates
            [np.nanmin(  # min over phases
                np.abs(self.statistics_periodic_interleaved[pind_min_interleaved][ti]))  # in the best period
                for ti in range(len(self.statistics_periodic_interleaved[pind_min_interleaved]))])
        self.std_free_place = ut.outlier_free_std(
            self.statistics_periodic_interleaved[pind_incorrect][tempind_incorrect])

    if (np.nanmin(self.statistics_periodic_interleaved) < np.nanmin(self.statistics_periodic_not_interleaved)
            or len(self.statistics_periodic_not_interleaved) == 0):
        self.is_min_interleaved = True
        self.snr_measured = np.nanmin(self.statistics_periodic_interleaved)
        self.pind_measured = pind_min_interleaved
        self.period_measured = self.periods[self.pind_measured]
        self.phind_measured = phasind_min_interleaved
        # self.ph_measured = self._phases_interleaved[
        #   phasind_min_interleaved] #relative phase, wrt the beginning of the data
        self.ph_measured = (self.time_interleaved[
            phasind_min_interleaved]) % self.period_measured  # phase wrt the beginning of Kepler time
        self.tempind_min_interleaved = tempind_min_interleaved
        self.tempind_measured = self.interleaved_numbers[tempind_min_interleaved]
        self.std_around_min = ut.outlier_free_std(np.concatenate((
            self.statistics_periodic_interleaved[self.pind_measured][tempind_min_interleaved][
            :self.phind_measured - ngap_moving_var // 2 * self.n_shifts],
            self.statistics_periodic_interleaved[self.pind_measured][tempind_min_interleaved][
            self.phind_measured + ngap_moving_var // 2 * self.n_shifts:])))
    else:
        self.is_min_interleaved = False
        self.snr_measured = np.nanmin(self.statistics_periodic_not_interleaved)
        self.pind_measured = pind_min_not_interleaved
        self.period_measured = self.periods[self.pind_measured]
        self.phind_measured = phasind_min_not_interleaved
        # self.ph_measured = self._phases_not_interleaved[
        #   phasind_min_not_interleaved]  # relative phase, wrt the beginning of the data
        self.ph_measured = (self.time_not_interleaved[
            phasind_min_not_interleaved]) % self.period_measured  # phase wrt the beginning of Kepler time
        self.tempind_min_not_interleaved = tempind_min_not_interleaved
        self.tempind_measured = self.not_interleaved_numbers[tempind_min_not_interleaved]
        self.std_around_min = ut.outlier_free_std(np.concatenate((
            self.statistics_periodic_not_interleaved[self.pind_measured][tempind_min_not_interleaved][
            :self.phind_measured - ngap_moving_var // 2],
            self.statistics_periodic_not_interleaved[self.pind_measured][tempind_min_not_interleaved][
            self.phind_measured + ngap_moving_var // 2:])))

    expected_transit_times = [self.tmin + self.ph_measured + i * self.period_measured
                              for i in range(int(np.ceil((self.tmax - self.tmin - self.ph_measured) / self.period_measured)))]
    self.expected_transit_times = expected_transit_times
    if self.is_min_interleaved:
        summed_transit_points = [np.argmin(np.abs(self.time_interleaved - expected_transit_times[i])) for i in
                                 range(len(expected_transit_times))]
        self.summed_transit_times = np.array(self.time_interleaved[summed_transit_points])
        summed_stats = np.array(self.statistics_interleaved[tempind_min_interleaved][summed_transit_points])
        summed_trends = np.array(self.trend_orig_flux_interleaved[summed_transit_points])
        summed_vars = np.array(self.stat_vars_interleaved[tempind_min_interleaved][summed_transit_points])
    else:
        summed_transit_points = [np.argmin(np.abs(self.time_not_interleaved - expected_transit_times[i])) for i in
                                 range(len(expected_transit_times))]
        self.summed_transit_times = np.array(self.time_not_interleaved[summed_transit_points])
        summed_stats = np.array(self.statistics_not_interleaved[tempind_min_not_interleaved][summed_transit_points])
        summed_trends = np.array(self.trend_orig_flux_not_interleaved[summed_transit_points])
        summed_vars = np.array(self.stat_vars_not_interleaved[tempind_min_not_interleaved][summed_transit_points])
    denom = np.sum(summed_vars * summed_trends ** 2) ** 0.5
    self.summed_transit_indices = copy.deepcopy(summed_transit_points)
    self.summed_transit_statistics_raw = summed_stats
    self.summed_transit_statistics_variances = copy.deepcopy(summed_vars)
    self.summed_transit_statistics_trends = copy.deepcopy(summed_trends)
    self.summed_transit_statistics = (summed_stats * summed_trends) / denom
    self.origrnal_transit_indices = [[] for q in self.quarters]
    for t in self.expected_transit_times:
        qmin = 0
        imin = 0
        minval = np.abs(t - self.times[0][0])
        for q in self.quarters:
            if (((t - self.times[q][0]) >= 0) and (np.abs(t - self.times[q][0]) < minval)):
                qmin = q
                minval = np.abs(t - self.times[q][0])
        minval = t - self.times[qmin][0]
        for i in range(len(self.times[qmin])):
            if np.abs(t - self.times[qmin][i]) < minval:
                imin = i
                minval = np.abs(t - self.times[qmin][i])
        self.origrnal_transit_indices[qmin].append(imin)


def find_moving_averages(self, interpolate=True, detr_length=Detr_length, detr_method=Detr_method,
                         detr_endmethod=Detr_endmethod, detr_ends=Detr_ends,
                         detr_endlen=Detr_endlen, keep_overall_level=False,
                         remove_outliers=False, nsig_outliers=Nsig_outliers_0):
    """
    Calculates the large trends of the original fluxes by taking the moving average of
    type detr_method ('mean' or 'median') of slices of size detr_length.
    Saves the trends (quarter-specific) and saves the detrended original fluxes separately.
    :param interpolate: Whether to interpolate the data in the holes containing numpy.nan.
        The interpolation will be linear, the holes in the ends of the data will
        be filled by a constant.
    :type interpolate: bool
    :param detr_length: Detrending length. Size of a slice to take the moving average
    :type detr_length: int
    :param detr_method: The type of moving average to take. Median is less sensitive to
        outliers, and mean is less sensitive to small number of samples.
        Median is preferred in the middle of the data.
    :type detr_method: str : 'mean' or 'median'
    :param detr_endmethod: The type of moving average to take in the ends of the data.
        Mean is less sensitive to small number of samples, so it is preferred
        in the ends of the data.
    :type detr_endmethod: str : 'mean' or 'median'
    :param detr_ends: How to make slices on the ends of the data. Read details in detrend_mov_ave().
        'symmetric' and 'asymmetric' preserve the size of the data,
        'cut' decreases the size of the data by detr_length.
    :type detr_ends: str : 'symmetric' or 'asymmetric' or 'cut'
    :param detr_endlen: Minimal slice taken in the end of the data. If None, will be taken
        as detr_len//4
    :type detr_endlen: int or None
    :param keep_overall_level: Whether to preserve the overall average (mean or median, defined by
        detr_method) of the data
    :type keep_overall_level: bool
    :param remove_outliers:
    :type remove_outliers:
    :param nsig_outliers:
    :type nsig_outliers:
    :return: creates self.detrended_orig_fluxes and self.trends_orig_fluxes
    :rtype:
    """
    self.detrended_orig_fluxes = []
    self.trends_orig_fluxes = []
    for q in self.quarters:
        msk = ~np.isnan(self.orig_fluxes[q])
        if interpolate:
            o_f = ut.interpolate_over_mask(np.array(self.orig_fluxes[q]), msk, interp_method='linear', ends='continue',
                                           mitigate_hole_ends=False, endhole_method='median')
            detr, trend = ut.detrend_mov_ave(np.array(o_f), length=detr_length, mask=None, method=detr_method,
                                             endmethod=detr_endmethod, ends=detr_ends, endlen=detr_endlen,
                                             return_trend=True,
                                             keep_overall_level=keep_overall_level, remove_outliers=remove_outliers,
                                             nsig_outliers=nsig_outliers)
        else:
            detr, trend = ut.detrend_mov_ave(np.array(self.orig_fluxes[q]), length=detr_length, mask=msk,
                                             method=detr_method,
                                             endmethod=detr_endmethod, ends=detr_ends, endlen=detr_endlen,
                                             return_trend=True,
                                             keep_overall_level=keep_overall_level, remove_outliers=remove_outliers,
                                             nsig_outliers=nsig_outliers)
        detr[~msk] = np.nan
        self.detrended_orig_fluxes.append(detr)
        self.trends_orig_fluxes.append(trend)


def veto_mes(self, mes_threshold=Threshold_mes_veto, veto_threshold=Veto_threshold_for_mes):
    # WARNING: this veto, if done after the periodicity search on the score-corrected data, will destroy the
    # distribution. Therefore, we are removing this veto from the search and will do it after the search and after
    # the time-slides and p_periodic estimation as a post-processing veto. Also, we make sure to do it on the
    # non-score-corrected data.
    if len(self.interleaved_numbers) > 0:
        self.mes_veto_masks_interleaved = np.ones(np.shape(self.statistics_periodic_interleaved)).astype(bool)
        indices = np.array(np.where(self.statistics_periodic_interleaved < -np.abs(mes_threshold))).T
        for pi, ti, phi in indices:
            expected_transit_times = [self.tmin + self._phases_interleaved[phi] + i * self.periods[pi]
                                      for i in range(
                    int(np.ceil((self.tmax - self.tmin - self._phases_interleaved[phi]) / self.periods[pi])))]

            summed_transit_points = [np.argmin(np.abs(self.time_interleaved - expected_transit_times[i])) for i in
                                     range(len(expected_transit_times))]
            summed_stats = np.array(self.statistics_interleaved[ti][summed_transit_points])  # raw
            summed_trends = np.array(self.trend_orig_flux_interleaved[summed_transit_points])
            summed_vars = np.array(self.stat_vars_interleaved[ti][summed_transit_points])
            amplitude_vars = []
            amplitudes_normalized = []
            for si in range(len(summed_vars)):
                if not np.isclose(summed_stats[si], 0):
                    amplitude_vars.append(1 / summed_vars[si])
                    amplitudes_normalized.append(
                        summed_stats[si] / summed_vars[si] / summed_trends[si] * self.mean_total)
                else:
                    amplitude_vars.append(np.inf)
                    amplitudes_normalized.append(0.)
            veto_score = ut.unequal_amplitude_veto(values=-np.array(amplitudes_normalized), variances=amplitude_vars,
                                                   A_all=None, indmax=None, A_1=None)
            if veto_score > veto_threshold:
                self.statistics_periodic_interleaved[pi][ti][phi] = 0.
                self.mes_veto_masks_interleaved[pi][ti][phi] = False

    if len(self.not_interleaved_numbers) > 0:
        self.mes_veto_masks_not_interleaved = np.ones(np.shape(self.statistics_periodic_not_interleaved)).astype(
            bool)
        indices = np.array(np.where(self.statistics_periodic_not_interleaved < -np.abs(mes_threshold))).T
        for pi, ti, phi in indices:
            expected_transit_times = [self.tmin + self._phases_not_interleaved[phi] + i * self.periods[pi]
                                      for i in range(
                    int(np.ceil((self.tmax - self.tmin - self._phases_not_interleaved[phi]) / self.periods[pi])))]

            summed_transit_points = [np.argmin(np.abs(self.time_not_interleaved - expected_transit_times[i])) for i
                                     in range(len(expected_transit_times))]
            summed_stats = np.array(self.statistics_not_interleaved[ti][summed_transit_points])  # raw
            summed_trends = np.array(self.trend_orig_flux_not_interleaved[summed_transit_points])
            summed_vars = np.array(self.stat_vars_not_interleaved[ti][summed_transit_points])
            amplitude_vars = []
            amplitudes_normalized = []
            for si in range(len(summed_vars)):
                if not np.isclose(summed_stats[si], 0) and not np.isinf(summed_vars[si]):
                    amplitude_vars.append(1 / summed_vars[si])
                    amplitudes_normalized.append(
                        summed_stats[si] / summed_vars[si] / summed_trends[si] * self.mean_total)
                else:
                    amplitude_vars.append(np.inf)
                    amplitudes_normalized.append(0.)
            if len(np.array(amplitudes_normalized)) < 3:
                veto_score = np.inf
            else:
                veto_score = ut.unequal_amplitude_veto(values=-np.array(amplitudes_normalized),
                                                       variances=amplitude_vars, A_all=None, indmax=None, A_1=None)
            if veto_score > veto_threshold:
                self.statistics_periodic_not_interleaved[pi][ti][phi] = 0.
                self.mes_veto_masks_not_interleaved[pi][ti][phi] = False



"""DRAFTS"""

"""
# # # EXPLICIT COLLECTION
            # for ti in range(len(stats_periodic[0])):  # over templates
            #     phase_ind_min_per = []
            #     scores_max_per = []
            #     for pi in range(len(stats_periodic)):  # over periods
            #         valid_idx = np.where(stats_periodic[pi][ti] < 0)[0]
            #         ph_i_min = valid_idx[np.nanargmax(coh_scores[pi][ti][valid_idx])]
            #         phase_ind_min_per.append(ph_i_min)
            #         scores_max_per.append(coh_scores[pi][ti][ph_i_min])
            #     score_best_per_inds = (np.argsort(scores_max_per)[::-1])[:n_per_temp]
            #     scores_sel = np.array(scores_max_per)[score_best_per_inds]
            #     per_sel = self.periods[score_best_per_inds]
            #     phase_ind_sel = np.array(phase_ind_min_per)[score_best_per_inds]
            #     phases_sel = np.array(phases)[phase_ind_sel]  # relative phases wrt the beginning of this data
            #     phases_abs = (np.array(times)[phase_ind_sel]) % per_sel  # phases wrt the beginning of Kepler time
            #     stats_sel = np.array(stats_periodic)[phase_ind_sel]
            #     vol_priors_sel = np.array(vol_priors)[phase_ind_sel]
            #
            #
            #     summed_transit_times_per = []
            #     summed_stats_raw_per = []
            #     # summed_trends_per = []
            #     summed_vars_per = []
            #     summed_corrections_per = []
            #     for j in range(len(score_best_per_inds)):
            #         period = per_sel[j]
            #         phase = phases_sel[j]
            #         expected_transit_times = [self.tmin + phase + i * period  # i am not sure tmin does not miss some nan in the beginning
            #                                   for i in range(int(np.ceil((self.tmax - self.tmin - phase) / period)))]
            #         summed_transit_points = [np.nanargmin(np.abs(times - expected_transit_times[i])) for
            #                                  i in range(len(expected_transit_times))]
            #         summed_transit_times_per.append(times[summed_transit_points])
            #         summed_stats_raw_per.append(stats[ti][summed_transit_points])
            #         # summed_trends_per.append(self.trend_orig_flux_interleaved[summed_transit_points])
            #         summed_vars_per.append(variances[ti][summed_transit_points])
            #         summed_corrections_per.append(cors[ti][summed_transit_points])
            #
            #     periods_selected.append(per_sel)
            #     phase_indices_selected.append(phase_ind_sel)
            #     coherent_scores_selected.append(scores_sel)
            #     statistic_values_selected.append(stats_sel)
            #     # statistic_values_selected.append(np.array(scores_max_per)[stat_best_per_inds])
            #     phases_selected.append(phases_abs)
            #     summed_transit_times_selected.append(summed_transit_times_per)
            #     summed_stats_raw_selected.append(summed_stats_raw_per)
            #     # summed_trends_selected.append(summed_trends_per)
            #     summed_vars_selected.append(summed_vars_per)
            #     summed_corrections_selected.append(summed_corrections_per)

        # # # # # INTERLEAVED AND NON-INTERLEAVED SEPARATELT
        # # period_indices_selected = []
        # periods_selected = []
        # phase_indices_selected = []  # relative phase indices wrt the local beginning of data (quarters may be cut)
        # phases_selected = []  # phase wrt the beginning of Kepler time
        # statistic_values_selected = []
        # summed_transit_times_selected = []
        # summed_stats_raw_selected = []
        # # summed_trends_selected = []
        # summed_vars_selected = []
        # summed_corrections_selected = []
        # if len(self.statistics_periodic_interleaved) > 0:
        #     for ti in range(len(self.statistics_periodic_interleaved[0])):  # over templates
        #         phase_ind_min_per = []
        #         stats_min_per = []
        #         for pi in range(len(self.statistics_periodic_interleaved)):  # over periods
        #             ph_i_min = np.nanargmin(self.statistics_periodic_interleaved[pi][ti])
        #             phase_ind_min_per.append(ph_i_min)
        #             stats_min_per.append(self.statistics_periodic_interleaved[pi][ti][ph_i_min])
        #         stat_best_per_inds = np.argsort(stats_min_per)[:n_per_temp]
        #         per_sel = self.periods[stat_best_per_inds]
        #         phase_ind_sel = np.array(phase_ind_min_per)[stat_best_per_inds]
        #         phases_sel = np.array(self._phases_interleaved)[phase_ind_sel]  # relative phases wrt the beginning of this data
        #         phases_abs = (np.array(self.time_interleaved)[phase_ind_sel]) % per_sel  # phases wrt the beginning of Kepler time
        #
        #         summed_transit_times_per = []
        #         summed_stats_raw_per = []
        #         # summed_trends_per = []
        #         summed_vars_per = []
        #         summed_corrections_per = []
        #         for j in range(len(stat_best_per_inds)):
        #             period = per_sel[j]
        #             phase = phases_sel[j]
        #             expected_transit_times = [self.tmin + phase + i * period  # i am not sure tmin does not miss some nan in the beginning
        #                                       for i in range(int(np.ceil((self.tmax - self.tmin - phase) / period)))]
        #             summed_transit_points = [np.nanargmin(np.abs(self.time_interleaved - expected_transit_times[i])) for
        #                                      i in range(len(expected_transit_times))]
        #             summed_transit_times_per.append(self.time_interleaved[summed_transit_points])
        #             summed_stats_raw_per.append(self.statistics_interleaved[ti][summed_transit_points])
        #             # summed_trends_per.append(self.trend_orig_flux_interleaved[summed_transit_points])
        #             summed_vars_per.append(self.stat_vars_interleaved[ti][summed_transit_points])
        #             summed_corrections_per.append(self.corrections_interleaved[ti][summed_transit_points])
        #         # period_indices_selected.append(stat_best_per_inds)
        #         periods_selected.append(per_sel)
        #         phase_indices_selected.append(phase_ind_sel)
        #         statistic_values_selected.append(np.array(stats_min_per)[stat_best_per_inds])
        #         phases_selected.append(phases_abs)
        #         summed_transit_times_selected.append(summed_transit_times_per)
        #         summed_stats_raw_selected.append(summed_stats_raw_per)
        #         # summed_trends_selected.append(summed_trends_per)
        #         summed_vars_selected.append(summed_vars_per)
        #         summed_corrections_selected.append(summed_corrections_per)
        # if len(self.statistics_periodic_not_interleaved) > 0:
        #     for ti in range(len(self.statistics_periodic_not_interleaved[0])):  # over templates
        #         phase_ind_min_per = []
        #         stats_min_per = []
        #         for pi in range(len(self.statistics_periodic_not_interleaved)):  # over periods
        #             ph_i_min = np.nanargmin(self.statistics_periodic_not_interleaved[pi][ti])
        #             phase_ind_min_per.append(ph_i_min)
        #             stats_min_per.append(self.statistics_periodic_not_interleaved[pi][ti][ph_i_min])
        #         stat_best_per_inds = np.argsort(stats_min_per)[:n_per_temp]
        #         per_sel = self.periods[stat_best_per_inds]
        #         phase_ind_sel = np.array(phase_ind_min_per)[stat_best_per_inds]
        #         phases_sel = np.array(self._phases_not_interleaved)[phase_ind_sel]  # relative phases wrt the beginning of this data
        #         phases_abs = (np.array(self.time_not_interleaved)[phase_ind_sel]) % per_sel  # phases wrt the beginning of Kepler time
        #
        #         summed_transit_times_per = []
        #         summed_stats_raw_per = []
        #         # summed_trends_per = []
        #         summed_vars_per = []
        #         summed_corrections_per = []
        #         for j in range(len(stat_best_per_inds)):
        #             period = per_sel[j]
        #             phase = phases_sel[j]
        #             expected_transit_times = [self.tmin + phase + i * period
        #                                       for i in range(int(np.ceil((self.tmax - self.tmin - phase) / period)))]
        #             summed_transit_points = [np.nanargmin(np.abs(self.time_not_interleaved - expected_transit_times[i]))
        #                                      for i in range(len(expected_transit_times))]
        #             summed_transit_times_per.append(self.time_not_interleaved[summed_transit_points])
        #             summed_stats_raw_per.append(self.statistics_not_interleaved[ti][summed_transit_points])
        #             # summed_trends_per.append(self.trend_orig_flux_not_interleaved[summed_transit_points])
        #             summed_vars_per.append(self.stat_vars_not_interleaved[ti][summed_transit_points])
        #             summed_corrections_per.append(self.corrections_not_interleaved[ti][summed_transit_points])
        #         # period_indices_selected.append(stat_best_per_inds)
        #         periods_selected.append(per_sel)
        #         phase_indices_selected.append(phase_ind_sel)
        #         statistic_values_selected.append(np.array(stats_min_per)[stat_best_per_inds])
        #         phases_selected.append(phases_abs)
        #         summed_transit_times_selected.append(summed_transit_times_per)
        #         summed_stats_raw_selected.append(summed_stats_raw_per)
        #         # summed_trends_selected.append(summed_trends_per)
        #         summed_vars_selected.append(summed_vars_per)
        #         summed_corrections_selected.append(summed_corrections_per)
        
        # self.best_periodic_scores_data[Search_results_index_per] = periods_selected
        # self.best_periodic_scores_data[Search_results_index_stat] = statistic_values_selected
        # self.best_periodic_scores_data[Search_results_index_sum_tr_times] = summed_transit_times_selected
        # self.best_periodic_scores_data[Search_results_index_sum_stats] = summed_stats_raw_selected
        # # self.best_periodic_scores_data[Search_results_index_sum_trends] = summed_trends_selected
        # self.best_periodic_scores_data[Search_results_index_sum_vars] = summed_vars_selected
        # self.best_periodic_scores_data[Search_results_index_sum_cors] = summed_corrections_selected
        # if self.inds_pieces_scramble is not None:
        #     self.best_periodic_scores_data[Search_results_index_slides] = [self.inds_pieces_scramble, self.inds_pieces_in_scrambled]
        # elif self.time_slide_shifts is not None:
        #     self.best_periodic_scores_data[Search_results_index_slides] = [self.time_slide_shifts, times_quarters]
        # self.best_periodic_scores_data[Search_results_index_per_list] = self.periods


 # if save_inverted:
        #     # period_indices_selected = []
        #     periods_selected = []
        #     phase_indices_selected = []
        #     phases_selected = []
        #     statistic_values_selected = []
        #     summed_transit_times_selected = []
        #     summed_stats_raw_selected = []
        #     # summed_trends_selected = []
        #     summed_vars_selected = []
        #     summed_corrections_selected = []
        #     if len(self.statistics_periodic_interleaved) > 0:
        #         for ti in range(len(self.statistics_periodic_interleaved[0])):  # over templates
        #             phase_ind_max_per = []
        #             stats_max_per = []
        #             for pi in range(len(self.statistics_periodic_interleaved)):  # over periods
        #                 ph_i_max = np.nanargmax(self.statistics_periodic_interleaved[pi][ti])
        #                 phase_ind_max_per.append(ph_i_max)
        #                 stats_max_per.append(self.statistics_periodic_interleaved[pi][ti][ph_i_max])
        #             stat_best_per_inds = (np.argsort(stats_max_per)[::-1])[:n_per_temp]
        #             per_sel = self.periods[stat_best_per_inds]
        #             phase_ind_sel = np.array(phase_ind_max_per)[stat_best_per_inds]
        #             phases_sel = np.array(self._phases_interleaved)[phase_ind_sel]
        #             phases_abs = (np.array(self.time_interleaved)[phase_ind_sel]) % per_sel  # phases wrt the beginning of Kepler time
        #
        #             summed_transit_times_per = []
        #             summed_stats_raw_per = []
        #             # summed_trends_per = []
        #             summed_vars_per = []
        #             summed_corrections_per = []
        #             for j in range(len(stat_best_per_inds)):
        #                 period = per_sel[j]
        #                 phase = phases_sel[j]
        #                 expected_transit_times = [self.tmin + phase + i * period
        #                                           for i in
        #                                           range(int(np.ceil((self.tmax - self.tmin - phase) / period)))]
        #                 summed_transit_points = [np.nanargmin(np.abs(self.time_interleaved - expected_transit_times[i]))
        #                                          for i in range(len(expected_transit_times))]
        #                 summed_transit_times_per.append(self.time_interleaved[summed_transit_points])
        #                 summed_stats_raw_per.append(self.statistics_interleaved[ti][summed_transit_points])
        #                 # summed_trends_per.append(self.trend_orig_flux_interleaved[summed_transit_points])
        #                 summed_vars_per.append(self.stat_vars_interleaved[ti][summed_transit_points])
        #                 summed_corrections_per.append(self.corrections_interleaved[ti][summed_transit_points])
        #             # period_indices_selected.append(stat_best_per_inds)
        #             periods_selected.append(per_sel)
        #             phase_indices_selected.append(phase_ind_sel)
        #             statistic_values_selected.append(np.array(stats_max_per)[stat_best_per_inds])
        #             phases_selected.append(phases_abs)
        #             summed_transit_times_selected.append(summed_transit_times_per)
        #             summed_stats_raw_selected.append(summed_stats_raw_per)
        #             # summed_trends_selected.append(summed_trends_per)
        #             summed_vars_selected.append(summed_vars_per)
        #             summed_corrections_selected.append(summed_corrections_per)
        #     if len(self.statistics_periodic_not_interleaved) > 0:
        #         for ti in range(len(self.statistics_periodic_not_interleaved[0])):  # over templates
        #             phase_ind_max_per = []
        #             stats_max_per = []
        #             for pi in range(len(self.statistics_periodic_not_interleaved)):  # over periods
        #                 ph_i_max = np.nanargmax(self.statistics_periodic_not_interleaved[pi][ti])
        #                 phase_ind_max_per.append(ph_i_max)
        #                 stats_max_per.append(self.statistics_periodic_not_interleaved[pi][ti][ph_i_max])
        #             stat_best_per_inds = (np.argsort(stats_max_per)[::-1])[:n_per_temp]
        #             per_sel = self.periods[stat_best_per_inds]
        #             phase_ind_sel = np.array(phase_ind_max_per)[stat_best_per_inds]
        #             phases_sel = np.array(self._phases_not_interleaved)[phase_ind_sel]
        #             phases_abs = (np.array(self.time_not_interleaved)[phase_ind_sel]) % per_sel  # phases wrt the beginning of Kepler time
        #
        #             summed_transit_times_per = []
        #             summed_stats_raw_per = []
        #             # summed_trends_per = []
        #             summed_vars_per = []
        #             summed_corrections_per = []
        #             for j in range(len(stat_best_per_inds)):
        #                 period = per_sel[j]
        #                 phase = phases_sel[j]
        #                 expected_transit_times = [self.tmin + phase + i * period
        #                                           for i in
        #                                           range(int(np.ceil((self.tmax - self.tmin - phase) / period)))]
        #                 summed_transit_points = [
        #                     np.nanargmin(np.abs(self.time_not_interleaved - expected_transit_times[i])) for i in
        #                     range(len(expected_transit_times))]
        #                 summed_transit_times_per.append(self.time_not_interleaved[summed_transit_points])
        #                 summed_stats_raw_per.append(self.statistics_not_interleaved[ti][summed_transit_points])
        #                 # summed_trends_per.append(self.trend_orig_flux_not_interleaved[summed_transit_points])
        #                 summed_vars_per.append(self.stat_vars_not_interleaved[ti][summed_transit_points])
        #                 summed_corrections_per.append(self.corrections_not_interleaved[ti][summed_transit_points])
        #             # period_indices_selected.append(stat_best_per_inds)
        #             periods_selected.append(per_sel)
        #             phase_indices_selected.append(phase_ind_sel)
        #             statistic_values_selected.append(np.array(stats_max_per)[stat_best_per_inds])
        #             phases_selected.append(phases_abs)
        #             summed_transit_times_selected.append(summed_transit_times_per)
        #             summed_stats_raw_selected.append(summed_stats_raw_per)
        #             # summed_trends_selected.append(summed_trends_per)
        #             summed_vars_selected.append(summed_vars_per)
        #             summed_corrections_selected.append(summed_corrections_per)
        #
        #     times_quarters = [(np.nanmin(self.times[q]), np.nanmax(self.times[q])) for q in
        #                       self.quarters]  # attention: there should be no zero-padding
        # # warning: you updated time slides, but the indices of the shuffled pieces are not saved here because you did not need them yet.
        #
        #     self.best_inverted_periodic_scores_data = [[] for _ in range(Search_results_length)]
        #     self.best_inverted_periodic_scores_data[Search_results_index_per] = periods_selected
        #     self.best_inverted_periodic_scores_data[Search_results_index_stat] = statistic_values_selected
        #     self.best_inverted_periodic_scores_data[Search_results_index_sum_tr_times] = summed_transit_times_selected
        #     self.best_inverted_periodic_scores_data[Search_results_index_sum_stats] = summed_stats_raw_selected
        #     # self.best_inverted_periodic_scores_data[Search_results_index_sum_trends] = summed_trends_selected
        #     self.best_inverted_periodic_scores_data[Search_results_index_sum_vars] = summed_vars_selected
        #     self.best_inverted_periodic_scores_data[Search_results_index_sum_cors] = summed_corrections_selected
        #     if self.inds_pieces_scramble is not None:
        #         self.best_inverted_periodic_scores_data[Search_results_index_slides] = [
        #                                                                         self.inds_pieces_scramble, self.inds_pieces_in_scrambled]
        #     elif self.time_slide_shifts is not None:
        #         self.best_inverted_periodic_scores_data[Search_results_index_slides] = [self.time_slide_shifts, times_quarters]
        #     self.best_inverted_periodic_scores_data[Search_results_index_per_list] = self.periods
        #     if save_to_file:
        #         filename_postfix = filename_postfix_inverted_period + str(self.pmin) + '_' + str(self.pmax)
        #         file_path = af.get_saving_file_path(output_location=path, target_id=self.target_id,
        #                                             filename_postfix=filename_postfix, file_format='npy')
        #         np.save(file_path, np.array(self.best_periodic_scores_data, dtype=object), allow_pickle=True)
        
        
        
        # self.best_stat_inv = 0
        # best_candidates = self.best_inverted_periodic_scores_data
        # argm = np.argmax(best_candidates[Search_results_index_stat])  # we took inverted best scores collection, they are positive
        # ti, i = argm // len(best_candidates[Search_results_index_stat][0]), argm % len(best_candidates[Search_results_index_stat][0])
        # # the structure used to be (25,10), where 10 was just 10 best per each template
        # smax = best_candidates[Search_results_index_stat][ti][i]  # maximal statistic
        # if smax > self.best_stat_inv:
        #     self.best_stat_inv = smax
        #     self.best_per_inv = best_candidates[Search_results_index_per][ti][i]
        #     self.best_temp_inv = ti
        #     self.best_first_transit_time_inv = best_candidates[Search_results_index_sum_tr_times][ti][i][0]
        #     self.summed_transit_times_inv = best_candidates[Search_results_index_sum_tr_times][ti][i]
        #     self.summed_raw_stats_inv = np.array(best_candidates[Search_results_index_sum_stats][ti][i])
        #     # self.summed_trends_inv = np.array(best_candidates[Search_results_index_sum_trends][ti][i])
        #     self.summed_vars_inv = np.array(best_candidates[Search_results_index_sum_vars][ti][i])
        #     self.summed_vars_inv[self.summed_vars_inv == 0] = np.nan
        
        
######### FROM TARGET CLASS  ##########
# self.best_stat_inv = 0
# for best_candidates in best_inv_candidates_concat:  # if there is a big MES score, further processing
#     argm = np.argmax(best_candidates[spip.Search_results_index_stat])  # we took inverted best scores collection, they are positive
#     ti, i = argm // len(best_candidates[spip.Search_results_index_stat][0]
#                         ), argm % len(best_candidates[spip.Search_results_index_stat][0])
#     # the structure is (25,10), where 10 is just 10 best per each template
#     smax = best_candidates[spip.Search_results_index_stat][ti][i]  # maximal statistic
#     if smax > self.best_stat_inv:
#         self.best_stat_inv = smax
#         self.best_per_inv = best_candidates[spip.Search_results_index_per][ti][i]
#         self.best_temp_inv = ti
#         self.best_first_transit_time_inv = best_candidates[spip.Search_results_index_sum_tr_times][ti][i][0]
#         self.summed_transit_times_inv = best_candidates[spip.Search_results_index_sum_tr_times][ti][i]
#         self.summed_raw_stats_inv = np.array(best_candidates[spip.Search_results_index_sum_stats][ti][i])
#         self.summed_trends_inv = np.array(best_candidates[spip.Search_results_index_sum_trends][ti][i])
#         self.summed_vars_inv = np.array(best_candidates[spip.Search_results_index_sum_vars][ti][i])
#         self.summed_vars_inv[self.summed_vars_inv == 0] = np.nan
        
        """