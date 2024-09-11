# import target_class
# from target_class import KeplerTarget
# from pipeline_parameters import *
# from auxiliary_functions import fold_fluxes
# from utils import pad_by_zeros
#
# import matplotlib.pyplot as plt
# plt.rcParams.update({'font.size': 11, 'font.serif': 'Times New Roman', 'lines.linewidth': 2})
# Def_col = plt.rcParams['axes.prop_cycle'].by_key()['color']
#
# np.set_printoptions(threshold=1000, linewidth=10000)
#
# # TODO see additional_targ_and_klc_methods for some more plotting methods
# # TODO: write this!
#
#
# class KeplerTargetPlot(KeplerTarget):
#     def __init__(self, target_id='003097222', data_location=Data_location, output_location=Output_location,
#                  load_pickled_attributes=True,
#                  get_transit_and_binary_info=True,
#                  gaia_kepler_file_path=Gaia_kepler_file,
#                  koi_dict_file=Koi_dict_file,
#                  ttv_dict_file=Ttv_dict_file,
#                  villanova_dict_file=Villanova_dict_file,
#                  overwrite_parameters_by_unpickle=False,
#                  allow_output_folder_creation=True):
#         KeplerTarget.__init__(self, target_id=target_id, data_location=data_location, output_location=output_location,
#                               load_pickled_attributes=load_pickled_attributes, get_transit_and_binary_info=get_transit_and_binary_info,
#                               gaia_kepler_file_path=gaia_kepler_file_path, koi_dict_file=koi_dict_file, ttv_dict_file=ttv_dict_file,
#                               villanova_dict_file=villanova_dict_file,
#                               overwrite_location_parameters_by_unpickle=overwrite_parameters_by_unpickle,
#                               allow_output_folder_creation=allow_output_folder_creation)
#
#     def plot_fluxes(self, ax=None, show=True, figsize=(15, 5), clrs=((0.5, 0.5, 1), (0.2, 0.7, 0.5))):
#         # TODO
#         if self.klc is None or self.klc.veto_masks is None:
#             self.get_ses_for_postprocessing(no_filters=False, nsig_outliers_0_pos=5, nsig_outliers_0_neg=15,
#                                             template_index=self.best_temp, apply_score_correction=True, apply_veto=True)
#         temp_n_sh = 0  # self.klc.indices_of_templates[self.best_temp][0]
#         tms = []
#         flxs = []
#         for tim in self.summed_transit_times:
#             for q in self.klc.quarters:
#                 if (tim >= self.klc.times[q][0]) and (tim <= self.klc.times[q][-1]):
#                     ind = np.argmin(np.abs(self.klc.times[q] - tim))
#                     tms.append(self.klc.times[q][ind])
#                     flxs.append(self.klc.fluxes[q][ind])
#         if ax is None:
#             fig = plt.figure(figsize=figsize)
#             ax = fig.gca()
#         for q in self.klc.quarters:
#             ax.plot(self.klc.times[q], self.klc.fluxes[q], color=clrs[q % 2], linewidth=1)
#         ax.plot(tms, flxs, 'o', color='k', label='summed raw', markersize=4)
#         for q in self.klc.quarters:
#             vm = self.klc.veto_masks[temp_n_sh][q].astype(bool)
#             ax.plot(self.klc.times[q][~vm], self.klc.fluxes[q][~vm], 'o', color='r', markersize=2)
#         q = self.klc.quarters[-1]
#         vm = self.klc.veto_masks[temp_n_sh][q].astype(bool)
#         ax.plot(self.klc.times[q][~vm], self.klc.fluxes[q][~vm], 'o', color='r', markersize=2, label='vetted statistics')  # just for label
#         ax.set_xlabel('time, days')
#         ax.set_ylabel('fluxes, mean subtracted')
#         ax.set_title('KOI' + self.target_id + ' conditioned fluxes\n' + 'period: ' + str(np.round(self.best_per, 1))
#                      + ', duration: ' + str(self.templates_durations_hours[self.best_temp]) + ' hours')
#         ax.legend(loc='upper right')
#         if show:
#             plt.show()
#
#     def plot_whitened_fluxes(self, ax=None, show=True, figsize=(15, 5), clrs=((0.5, 0.5, 1), (0.2, 0.7, 0.5))):
#         # TODO
#         if self.klc is None or self.klc.veto_masks is None:
#             self.get_ses_for_postprocessing(no_filters=False, nsig_outliers_0_pos=5, nsig_outliers_0_neg=15,
#                                             template_index=self.best_temp, apply_score_correction=True, apply_veto=True)
#         temp_n_sh = 0  # self.klc.indices_of_templates[self.best_temp][0]
#         tms = []
#         flxs = []
#         for tim in self.summed_transit_times:
#             for q in self.klc.quarters:
#                 if (tim >= self.klc.times[q][0]) and (tim <= self.klc.times[q][-1]):
#                     ind = np.argmin(np.abs(self.klc.times[q] - tim))
#                     tms.append(self.klc.times[q][ind])
#                     flxs.append(self.klc.whitened_fluxes_td[q][ind])
#         if ax is None:
#             fig = plt.figure(figsize=figsize)
#             ax = fig.gca()
#         for q in self.klc.quarters:
#             ax.plot(self.klc.times[q], self.klc.whitened_fluxes_td[q], color=clrs[q % 2], linewidth=1)
#         ax.plot(tms, flxs, 'o', color='k', label='summed raw', markersize=4)
#         for q in self.klc.quarters:
#             vm = self.klc.veto_masks[temp_n_sh][q].astype(bool)
#             ax.plot(self.klc.times[q][~vm], self.klc.whitened_fluxes_td[q][~vm], 'o', color='r', markersize=2)
#         q = self.klc.quarters[-1]
#         vm = self.klc.veto_masks[temp_n_sh][q].astype(bool)
#         ax.plot(self.klc.times[q][~vm], self.klc.whitened_fluxes_td[q][~vm], 'o', color='r', markersize=2,
#                 label='vetted statistics')  # just for label
#         ax.set_xlabel('time, days')
#         ax.set_ylabel('whitened fluxes')
#         ax.set_title('KOI' + self.target_id + ' whitened fluxes\n' + 'period: ' + str(np.round(self.best_per, 1))
#                      + ', duration: ' + str(self.templates_durations_hours[self.best_temp]) + ' hours')
#         ax.legend(loc='upper right')
#         if show:
#             plt.show()
#
#     def plot_ses_raw(self, ax=None, figsize=(15, 5), clrs=((0.5, 0.5, 1), (0.2, 0.7, 0.5))):
#         # TODO
#         if self.klc is None or self.klc.veto_masks is None:
#             self.get_ses_for_postprocessing(no_filters=False, nsig_outliers_0_pos=5, nsig_outliers_0_neg=15,
#                                             template_index=self.best_temp, apply_score_correction=True, apply_veto=True)
#         temp_n_sh = 0  # self.klc.indices_of_templates[self.best_temp][0]
#         if ax is None:
#             fig = plt.figure(figsize=figsize)
#             ax = fig.gca()
#         for q in self.klc.quarters:
#             ax.plot(self.klc.times[q], self.klc.statistics_raw[temp_n_sh][q], color=clrs[q % 2], linewidth=1)
#         ax.plot(self.summed_transit_times, self.summed_raw_stats, 'o', color='k', label='summed raw', markersize=4)
#         for q in self.klc.quarters:
#             vm = self.klc.veto_masks[temp_n_sh][q].astype(bool)
#             ax.plot(self.klc.times[q][~vm], self.klc.statistics_raw[temp_n_sh][q][~vm], 'o', color='r', markersize=2)
#         ax.plot([], [], 'o', color='r', markersize=2, label='vetted statistics')
#         ax.set_xlabel('Time, days')
#         ax.set_ylabel('Statistics raw')
#         ax.set_title('KOI' + self.target_id + ' SES raw\n' + 'period: ' + str(np.round(self.best_per, 1))
#                      + ', duration: ' + str(self.templates_durations_hours[self.best_temp]) + ' hours')
#         ax.legend(loc='upper right')
#
#     def plot_ses_snr(self, ax=None, figsize=(15, 5), clrs=((0.5, 0.5, 1), (0.2, 0.7, 0.5))):
#         # TODO
#         if self.klc is None or self.klc.veto_masks is None:
#             self.get_ses_for_postprocessing(no_filters=False, nsig_outliers_0_pos=5, nsig_outliers_0_neg=15,
#                                             template_index=self.best_temp, apply_score_correction=True, apply_veto=True)
#         temp_n_sh = 0  # self.klc.indices_of_templates[self.best_temp][0]
#         if ax is None:
#             fig = plt.figure(figsize=figsize)
#             ax = fig.gca()
#         for q in self.klc.quarters:
#             ax.plot(self.klc.times[q], self.klc.statistics_snr[temp_n_sh][q], color=clrs[q % 2], linewidth=1)
#         ax.plot(self.summed_transit_times, self.summed_raw_stats / self.summed_vars ** 0.5, 'o', color='k', label='summed raw / std',
#                 markersize=4)
#         for q in self.klc.quarters:
#             vm = self.klc.veto_masks[temp_n_sh][q].astype(bool)
#             ax.plot(self.klc.times[q][~vm], self.klc.statistics_snr[temp_n_sh][q][~vm], 'o', color='r', markersize=2)
#         ax.plot([], [], 'o', color='r', markersize=2, label='vetted statistics')
#         ax.set_xlabel('Time, days')
#         ax.set_ylabel('Statistics snr')
#         ax.set_title('KOI' + self.target_id + ' SES SNR\n' + 'period: ' + str(np.round(self.best_per, 1))
#                      + ', duration: ' + str(self.templates_durations_hours[self.best_temp]) + ' hours')
#         ax.legend(loc='upper right')
#
#     def plot_folded_red_flux(self, ax=None, figsize=(8, 6), plot_twice=False):
#         # TODO
#         # PLOT FOLD RED FLUX
#         if self.klc is None or self.klc.veto_masks is None:
#             self.get_ses_for_postprocessing(no_filters=False, nsig_outliers_0_pos=5, nsig_outliers_0_neg=15,
#                                             template_index=self.best_temp, apply_score_correction=True, apply_veto=True)
#         binwid = min(self.templates_durations_hours[self.best_temp] / 24 / 5, 0.1)
#         means = [np.nanmean(self.klc.orig_fluxes[q]) for q in self.klc.quarters]
#
#         bins_time, bins_flux, bins_counts, folds_times, folds_fluxes, time_template, template_cropped, bins_std = fold_fluxes(
#             times=self.klc.times, fluxes=self.klc.fluxes, period=self.best_per, first_time=self.summed_transit_times[0],
#             masks=self.klc.big_hole_masks,
#             binwid=binwid, time_fold=self.best_per,  # self.templates_durations_hours[self.best_temp] / 24 * 20,
#             dt=self.klc.dt_days,
#             means=means,
#             template=np.array(self.klc.templates[self.klc.indices_of_templates[self.best_temp][0]]),
#             return_std=True)
#         if ax is None:
#             fig = plt.figure(figsize=figsize)
#             ax = fig.gca()
#         for i in range(len(folds_times)):
#             ax.plot(folds_times[i], folds_fluxes[i], 'o', markersize=1, color=(0.5, 0.5, 0.5))
#         ax.errorbar(bins_time[bins_counts > 0], bins_flux[bins_counts > 0], yerr=bins_std[bins_counts > 0], xerr=None,
#                     markersize=2.5, fmt='o', markerfacecolor='k', markeredgecolor='k',
#                     ecolor='k', elinewidth=1.2, capsize=None, barsabove=False, lolims=False, uplims=False,
#                     xlolims=False, xuplims=False, errorevery=1, capthick=None,
#                     label='binning, bin ' + ut.fstr(np.round(binwid, 3)) + ' days')
#         # ax.plot(bins_time, bins_flux, '-o', markersize=3, color='b', linewidth=1,
#         #         label='binning, bin ' + ut.fstr(np.round(binwid, 3)) + ' days')
#         ax.plot(time_template, 1 - template_cropped * np.abs(np.nanmean(
#             self.best_ampls_no_filt[self.summed_trends > 0] / self.summed_trends[self.summed_trends > 0])),
#                 color='r', linewidth=1,
#                 label=r'template $\cdot$ amplitude,' + '\nduration: ' + self.templates_durations_hours[self.best_temp] + ' hours')
#
#         if plot_twice:
#             for i in range(len(folds_times)):
#                 ax.plot(folds_times[i] + self.best_per, folds_fluxes[i], 'o', markersize=1, color=(0.5, 0.5, 0.5))
#             ax.errorbar(bins_time[bins_counts > 0] + self.best_per, bins_flux[bins_counts > 0], yerr=bins_std[bins_counts > 0], xerr=None,
#                         markersize=2.5, fmt='o', markerfacecolor='k', markeredgecolor='k',
#                         ecolor='k', elinewidth=1.2, capsize=None, barsabove=False, lolims=False, uplims=False,
#                         xlolims=False, xuplims=False, errorevery=1, capthick=None)
#             ax.plot(time_template + self.best_per, 1 - template_cropped * np.abs(np.nanmean(
#                 self.best_ampls_no_filt[self.summed_trends > 0] / self.summed_trends[self.summed_trends > 0])),
#                     color='r', linewidth=1)
#
#         maxdev = np.nanmax(
#             [np.nanmax(np.abs(bins_flux - 1)), np.nanmax([np.nanmax(np.abs(folds_fluxes[i] - 1)) for i in range(len(folds_fluxes))])])
#         ax.set_ylim(1 - maxdev * 1.01, 1 + maxdev * 1.01)  # make the limits symmetric
#         ax.set_xlim(-self.best_per / 2, self.best_per / 2)
#         if plot_twice:
#             ax.set_xlim(-self.best_per / 2, self.best_per / 2 + self.best_per)
#             ax.axvline(self.best_per / 2, color=(0.5, 0.5, 0.5), linewidth=1)
#         ax.legend(loc='upper right')
#         ax.set_title('KOI' + self.target_id + ' fold, ' + str(len(self.summed_raw_stats)) + ' transits')
#         ax.set_xlabel('Time from the mid transit, days')
#         ax.set_ylabel('Flux normalized by mean')
#
#     def plot_folded_whitened_flux(self, ax=None, figsize=(8, 6), plot_twice=False):
#         if self.klc is None or self.klc.veto_masks is None:
#             self.get_ses_for_postprocessing(no_filters=False, nsig_outliers_0_pos=5, nsig_outliers_0_neg=15,
#                                             template_index=self.best_temp, apply_score_correction=True, apply_veto=True)
#         binwid = min(self.templates_durations_hours[self.best_temp] / 24 / 5, 0.1)
#         if ax is None:
#             fig = plt.figure(figsize=figsize)
#             ax = fig.gca()
#         bins_time, bins_flux, bins_counts, folds_times, folds_fluxes, bins_std = fold_fluxes(
#             times=self.klc.times, fluxes=self.klc.whitened_fluxes_td, period=self.best_per, first_time=self.summed_transit_times[0],
#             masks=self.klc.big_hole_masks,
#             binwid=binwid, time_fold=self.best_per,  # self.templates_durations_hours[self.best_temp] / 24 * 10,
#             dt=self.klc.dt_days, means=None,
#             template=None, return_std=True)  # np.array(self.klc.templates[self.klc.indices_of_templates[self.best_temp][0]]))
#         n_time_templ = int(np.round(self.best_per / self.klc.dt_days))
#         temps = self.klc.whitened_templates_td[self.klc.indices_of_templates[self.best_temp][0]]
#         for q in range(len(temps)):
#             if len(temps[q]) < n_time_templ:
#                 temps[q] = pad_by_zeros(temps[q], n_time_templ, in_the_center=True)
#         temp_best_1 = np.array(np.mean([temps[q][:n_time_templ // 2] for q in self.klc.quarters], axis=0))
#         temp_best_2 = np.array(np.mean([temps[q][-n_time_templ // 2:] for q in self.klc.quarters], axis=0))
#         temp_best = np.concatenate((temp_best_2, temp_best_1))
#         tem_center = np.nanargmax(np.abs(temp_best))
#         time_template = np.concatenate((np.arange(tem_center) - tem_center, np.arange(len(temp_best) - tem_center))) * self.klc.dt_days
#
#         for i in range(len(folds_times)):
#             ax.plot(folds_times[i], folds_fluxes[i], 'o', markersize=1, color=(0.5, 0.5, 0.5))
#         ax.errorbar(bins_time[bins_counts > 0], bins_flux[bins_counts > 0], yerr=bins_std[bins_counts > 0], xerr=None,
#                     markersize=2.5, fmt='o', markerfacecolor='k', markeredgecolor='k',
#                     ecolor='k', elinewidth=1.2, capsize=None, barsabove=False, lolims=False, uplims=False,
#                     xlolims=False, xuplims=False, errorevery=1, capthick=None,
#                     label='binning, bin ' + ut.fstr(binwid) + ' days')
#         # ax.plot(bins_time, bins_flux, '-o', markersize=3, color='b', linewidth=1, label='binning, bin ' + ut.fstr(binwid) + ' days')
#         ax.plot(time_template, -temp_best * self.best_ampl_no_filt, color='r', linewidth=1,
#                 label='average whitened template,' + '\ntransit duration: ' + self.templates_durations_hours[self.best_temp] + ' hours')
#
#         if plot_twice:
#             for i in range(len(folds_times)):
#                 ax.plot(folds_times[i] + self.best_per, folds_fluxes[i], 'o', markersize=1, color=(0.5, 0.5, 0.5))
#             ax.errorbar(bins_time[bins_counts > 0] + self.best_per, bins_flux[bins_counts > 0], yerr=bins_std[bins_counts > 0], xerr=None,
#                         markersize=2.5, fmt='o', markerfacecolor='k', markeredgecolor='k',
#                         ecolor='k', elinewidth=1.2, capsize=None, barsabove=False, lolims=False, uplims=False,
#                         xlolims=False, xuplims=False, errorevery=1, capthick=None)
#             ax.plot(time_template + self.best_per, -temp_best * self.best_ampl_no_filt, color='r', linewidth=1)
#
#         maxdev = np.nanmax(
#             [np.nanmax(np.abs(bins_flux)), np.nanmax([np.nanmax(np.abs(folds_fluxes[i])) for i in range(len(folds_fluxes))])])
#         ax.set_ylim(-maxdev * 1.01, maxdev * 1.01)  # make the limits symmetric
#         ax.set_xlim(-self.best_per / 2, self.best_per / 2)
#         if plot_twice:
#             ax.set_xlim(-self.best_per / 2, self.best_per / 2 + self.best_per)
#             ax.axvline(self.best_per / 2, color=(0.5, 0.5, 0.5), linewidth=1)
#         ax.set_xlabel('Time from the mid transit, days')
#         ax.set_ylabel('Whitened flux')
#         ax.legend(loc='upper right')
#         ax.set_title('KOI' + self.target_id + ' whitened fold, ' + str(len(self.summed_raw_stats)) + ' transits')
#
#     def plot_folded_stat_snr(self, ax=None, figsize=(8, 6), plot_twice=False):
#         if self.klc is None or self.klc.veto_masks is None:
#             self.get_ses_for_postprocessing(no_filters=False, nsig_outliers_0_pos=5, nsig_outliers_0_neg=15,
#                                             template_index=self.best_temp, apply_score_correction=True, apply_veto=True)
#         time_full, trend_orig_flux_full = af.connect_quarters_fun(self.klc.times, self.klc.trends_orig_fluxes,
#                                                                   zero_padding_ends=None,
#                                                                   is_data_cut=~self.klc.is_zero_padded,
#                                                                   is_time_cut=~self.klc.is_zero_padded,
#                                                                   dt=self.klc.dt_days, fill_gaps_by_nan=False)
#         stats_snr = self.klc.statistics_snr[
#                     self.klc.indices_of_templates[self.best_temp][0]:self.klc.indices_of_templates[self.best_temp][1]]
#         masks_temp = self.klc.stat_masks[self.klc.indices_of_templates[self.best_temp][0]:self.klc.indices_of_templates[self.best_temp][1]]
#         stats_snr_full = np.array([af.connect_quarters_fun(self.klc.times, stats_snr[k],
#                                                            zero_padding_ends=None, is_data_cut=~self.klc.is_zero_padded,
#                                                            is_time_cut=~self.klc.is_zero_padded, dt=self.klc.dt_days,
#                                                            fill_gaps_by_nan=False)[1]
#                                    for k in range(len(stats_snr))])
#         stat_masks_full = np.array([(af.connect_quarters_fun(self.klc.times, masks_temp[k],
#                                                              zero_padding_ends=None, is_data_cut=~self.klc.is_zero_padded,
#                                                              is_time_cut=~self.klc.is_zero_padded, dt=self.klc.dt_days,
#                                                              fill_gaps_by_nan=False)[1]).astype(bool)
#                                     for k in range(len(stats_snr))])
#         if len(stats_snr_full) > 1:
#             time_interleaved = [np.arange(len(trend_orig_flux_full)) * self.klc.dt_days / self.klc.n_shifts + time_full[0]]
#             statistics_interleaved = [np.array(stats_snr_full.T.flatten())]
#             masks_interleaved = [np.array(stat_masks_full.T.flatten())]
#         else:
#             time_interleaved = [time_full]
#             statistics_interleaved = stats_snr_full
#             masks_interleaved = stat_masks_full
#
#         binwid = min(self.templates_durations_hours[self.best_temp] / 24 / 5, 0.1)
#         if ax is None:
#             fig = plt.figure(figsize=figsize)
#             ax = fig.gca()
#         bins_time, bins_flux, bins_counts, folds_times, folds_fluxes, bins_std = fold_fluxes(
#             times=time_interleaved, fluxes=statistics_interleaved, period=self.best_per, first_time=self.summed_transit_times[0],
#             masks=masks_interleaved,
#             binwid=binwid, time_fold=self.best_per,  # self.templates_durations_hours[self.best_temp] / 24 * 10,
#             dt=self.klc.dt_days, means=None,
#             template=None, return_std=True)  # np.array(self.klc.templates[self.klc.indices_of_templates[self.best_temp][0]]))
#
#         n_time_templ = int(np.round(self.best_per / self.klc.dt_days))
#         temps = self.klc.whitened_templates_td[self.klc.indices_of_templates[self.best_temp][0]]
#         for q in range(len(temps)):
#             if len(temps[q]) < n_time_templ:
#                 temps[q] = ut.pad_by_zeros(temps[q], n_time_templ, in_the_center=True)
#         temp_best_1 = np.array(np.mean([temps[q][:n_time_templ // 2] for q in self.klc.quarters], axis=0))
#         temp_best_2 = np.array(np.mean([temps[q][-n_time_templ // 2:] for q in self.klc.quarters], axis=0))
#         temp_best = np.concatenate((temp_best_2, temp_best_1))
#         temp_best = temp_best / np.linalg.norm(temp_best)
#         tem_center = np.nanargmax(np.abs(temp_best))
#         temp_best = np.fft.irfft(np.abs(np.fft.rfft(temp_best)))  # for statistic, template is convolution
#         temp_best = np.roll(temp_best, tem_center)
#         tem_center = np.nanargmax(np.abs(temp_best))
#         time_template = np.concatenate((np.arange(tem_center) - tem_center, np.arange(len(temp_best) - tem_center))) * self.klc.dt_days
#
#         for i in range(len(folds_times)):
#             ax.plot(folds_times[i], folds_fluxes[i], 'o', markersize=0.6, color=(0.6, 0.6, 0.6))
#         ax.errorbar(bins_time[bins_counts > 0], bins_flux[bins_counts > 0], yerr=bins_std[bins_counts > 0], xerr=None,
#                     markersize=2.5, fmt='o', markerfacecolor='k', markeredgecolor='k',
#                     ecolor='k', elinewidth=1.2, capsize=None, barsabove=False, lolims=False, uplims=False,
#                     xlolims=False, xuplims=False, errorevery=1, capthick=None,
#                     label='binning, bin ' + ut.fstr(binwid) + ' days')
#         ax.plot(time_template, temp_best * self.best_stat / (len(self.summed_raw_stats)) ** 0.5, color='r', linewidth=1,
#                 label='average per-transit snr')
#
#         if plot_twice:
#             for i in range(len(folds_times)):
#                 ax.plot(folds_times[i] + self.best_per, folds_fluxes[i], 'o', markersize=0.6, color=(0.6, 0.6, 0.6))
#             ax.errorbar(bins_time[bins_counts > 0] + self.best_per, bins_flux[bins_counts > 0], yerr=bins_std[bins_counts > 0], xerr=None,
#                         markersize=2.5, fmt='o', markerfacecolor='k', markeredgecolor='k',
#                         ecolor='k', elinewidth=1.2, capsize=None, barsabove=False, lolims=False, uplims=False,
#                         xlolims=False, xuplims=False, errorevery=1, capthick=None)
#             ax.plot(time_template + self.best_per, temp_best * self.best_stat / (len(self.summed_raw_stats)) ** 0.5, color='r', linewidth=1)
#         maxdev = np.nanmax(
#             [np.nanmax(np.abs(bins_flux)), np.nanmax([np.nanmax(np.abs(folds_fluxes[i])) for i in range(len(folds_fluxes))])])
#         ax.set_ylim(-maxdev * 1.01, maxdev * 1.01)  # make the limits symmetric
#         ax.set_xlim(-self.best_per / 2, self.best_per / 2)
#         if plot_twice:
#             ax.set_xlim(-self.best_per / 2, self.best_per / 2 + self.best_per)
#             ax.axvline(self.best_per / 2, color=(0.5, 0.5, 0.5), linewidth=1)
#         ax.set_xlabel('Time from the mid transit, days')
#         ax.set_title('Statistic [snr]', fontsize=14)
#         ax.legend(loc='upper right')
#         ax.set_title('KOI' + self.target_id + ' statistic snr fold, ' + str(len(self.summed_raw_stats)) + ' transits')
#
#     def plot_background(self, ax=None, figsize=(10, 5), fixed_limits=True, limits=None, plot_extrapolation=True,
#                         fontsize_legend=None):
#         # PLOT BACKGROUND AND INJECTION-RECOVERY
#         if self.prob_density_planet_not_noise is None:
#             self.analyze_injections_and_background()
#         bins = self.bins_stat_histogram
#         if fixed_limits:
#             if limits is not None:
#                 min_plot_mes2 = limits[0]
#                 max_plot_mes2 = limits[1]
#             else:
#                 min_plot_mes2 = Default_min_plot_stat
#                 max_plot_mes2 = Default_max_plot_stat
#         else:
#             min_plot_mes2 = 0.5 * bins[0]
#             max_plot_mes2 = 1.1 * bins[-1]
#         # UNIFORMLY FORMATTED PLOT
#         if ax is None:
#             fig = plt.figure(figsize=figsize)
#             ax = fig.gca()
#
#         ax.stairs(self.distr_stat_inj_all, bins, label='all injections', linewidth=2)  # after density correction factor, get per one star
#         ax.stairs(self.distr_stat_inj_success, bins, label='success injections',
#                   linewidth=2)  # after density correction factor, get per one star
#         # Divide by the number of the timeslides, get effectively per one star:
#         ax.stairs(self.distr_stat_h0, bins, label='background H0', linewidth=2)
#
#         ax.axvline(self.best_stat, color='k', label='trigger, snr=' + str(np.round(np.abs(self.best_stat), 1)) +
#                    ',\n  period=' + str(np.round(self.best_per, 1)) + ' d')
#
#         # TODO: this is a bootstrap, you need to make a function for this:
#         # log10_rp_edges, period_edges, occurrences_table = get_occurrence_table(self.stellar_radius)  # Per one star
#         log10_best_per_low = np.log10(self.best_per - self.per_wing_injection_and_slides)
#         log10_best_per_up = np.log10(self.best_per + self.per_wing_injection_and_slides)
#         # log10_period_edges = np.log10(period_edges)
#         self.log10_best_period_edges = (log10_best_per_low, log10_best_per_up)
#
#         ax.set_title('KOI' + self.target_id + ', periods ['
#                      + str(np.round(10 ** (self.log10_best_period_edges[0]), 1)) + '-'
#                      + str(np.round(10 ** (self.log10_best_period_edges[1]), 1))
#                      + '] days, duration=' + str(self.templates_durations_hours[self.best_temp]) + ' hours', fontsize=13)
#         ax.set_xlabel('Recovered MES [$SNR^2$]')
#         ax.set_ylabel('Counts per effective one star \n per period bin per template')
#         ax.set_yscale('log')
#         ax.set_xlim(min_plot_mes2, max_plot_mes2)
#
#         if plot_extrapolation:
#             # MAKE EXTRAPOLATION
#             coefs = self.extrapolation_coefs_stat_h0_hist
#             # determine intersection with horizontal axis in log scale:
#             bin_extra_max = (np.log(np.nanmin(self.distr_stat_h0[self.distr_stat_h0 != 0])) - coefs[0]) / coefs[1]
#             bin_extra_max = np.nanmax([self.best_stat + 1, bin_extra_max + 1])
#             bin_extra_min = bins[np.nanargmax(self.distr_stat_h0) + 1]
#             bin_min_distr = bins[:-1][((~np.isnan(self.distr_stat_h0)) & (self.distr_stat_h0 > 0))][0]
#             # bin_extra_min = int((bin_extra_min + bin_extra_max) / 2)
#             bin_extra_min = bin_extra_min + int(np.abs(bin_extra_min - bin_min_distr) / 2)
#             bins_extra = np.linspace(bin_extra_min, bin_extra_max, 10)
#             ax.plot(bins_extra, np.exp(bins_extra * coefs[1] + coefs[0]), '--',
#                     label='extrapolation, FAP=' + f"{self.false_alarm_prob_trigger:.2e}")
#         if fontsize_legend is not None:
#             ax.legend(loc='upper right', fontsize=fontsize_legend)
#         else:
#             ax.legend(loc='upper right')
#
