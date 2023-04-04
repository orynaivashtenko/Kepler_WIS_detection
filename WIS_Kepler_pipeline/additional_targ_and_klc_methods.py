import matplotlib.pyplot as plt
import numpy as np
from auxiliary_functions import histogram

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
    for tt in range(len(klc.statistics_raw_cor)):
        for q in range(len(klc.statistics_raw_cor[tt])):
            half_wid = (whitened_template_widths[tt][q] + 1) // 2
            var_stat = np.var(klc.statistics_raw_cor[tt][q][klc.stat_masks[tt][q]])
            outlier_mask = klc.statistics_raw_cor[tt][q] ** 2 < 3 ** 2 * var_stat  # for variance measurement only
            var_stat = np.var(klc.statistics_raw_cor[tt][q][outlier_mask & klc.stat_masks[tt][q]])
            outlier_indices = np.where(klc.statistics_raw_cor[tt][q] ** 2 > threshold_to_std ** 2 * var_stat)[
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
    for tt in range(len(klc.statistics_raw_cor)):
        for q in range(len(klc.statistics_raw_cor[tt])):
            wid = whitened_template_widths[tt][q]
            half_wid = (whitened_template_widths[tt][q] + 1) // 2
            for ind in np.arange(0, len(klc.statistics_raw_cor[tt][q]) - half_wid, half_wid):
                indmin = ind + np.nanargmin(klc.statistics_raw_cor[tt][q][ind: ind + wid])
                if not klc.stat_masks[tt][q][indmin]:
                    if klc.statistics_raw_cor[tt][q][indmin] == np.nanmin(klc.statistics_raw_cor[tt][q][
                                                                           max(0, indmin - half_wid): min(
                                                                               indmin + half_wid, len(
                                                                                   klc.statistics_raw_cor[tt][
                                                                                       q]))]):
                        klc.stat_masks[tt][q][max(0, indmin - half_wid): min(indmin + half_wid, len(
                            klc.statistics_raw_cor[tt][q]))] = False



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
