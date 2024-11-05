# #!/usr/bin/env python3
# import os
#
# os.environ["OMP_NUM_THREADS"] = "1"  #
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"  #
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"
#
# import copy
# import warnings
# from scipy.signal import sosfiltfilt
# import numba
# from scipy.stats import chi2
#
# import template_bank.template_module as tm
# import auxiliary_functions as af
# import utils as ut
# from search_pipeline_parameters import *
#
#
# def veto_parts_scores(data_wh_fd, template_wh_full, indices_padded, mask_padded=None,
#                       tlen=None, n_parts=4, min_bins_part=4, nperseg_psd=Nperseg_psd,
#                       splitting='snr', lobe_hlen=4):
#     """
#     Splits the tamplate into parts and tests between two cases: all parts are of the same amplitude
#     or every part has a different amplitude.
#     :param data_wh_fd: Whitened data in the Fourier domain for the entire quarter.
#     :type data_wh_fd: numpy.ndarray 1d of float
#     :param template_wh_full: Whitened template in time domain rolled in such a way that the
#         center of the transit is at zero index.
#     :type template_wh_full: numpy.ndarray 1d of float
#     :param indices_padded: The position where the transit center is located. Should be in the padded units,
#         meaning it is counted from the beginning of the padded arrays.
#     :type indices_padded: numpy.ndarray 1d of int
#     :param mask_padded: Goodness mask corresponding to `data_wh_fd` but in time domain.
#     :type mask_padded: numpy.ndarray 1d of bool
#     :param tlen: Length of the non-zero part of the template (usually set by the whitening filter, can be
#         also affected by the notch filter). If None, will be measured.
#     :type tlen: int or None
#     :param fn: False negative rate that is needed to ve achieved in the veto. Calculated from chi2 distribution
#     :type fn: float
#     :param n_parts: Number of parts to which the template should be split (expecting to have the same SNR in them)
#     :type n_parts: int
#     :param min_bins_part: Minimal number of bins that should be in every part.
#         If less, number of parts will be decreased
#     :type min_bins_part: int
#     :param nperseg_psd: size of the slice that was used for PSD measurement in the pipeline
#     :type nperseg_psd: int
#     :param splitting: According to what to split the template: equal parts in snr or in just take the main
#         lobe of the template, for example if it is too short and can be tested in a usual chi-square veto
#         and not in parts veto.
#     :type splitting: str: 'snr' or 'lobe'
#     :param lobe_hlen: half-length of the lobe if `splitting`=='lobe' or if the number of parts is only one.
#     :type lobe_hlen: int
#     :return: is_goods, `True` where the trigger passed the veto, and `False` otherwise.
#         Same size as the input `indices_padded`
#     :rtype: numpy.ndarray of bool. Same size as the input `indices_padded`
#     """
#     # Indices are assumed to be given in the padded coordinates!
#     template_wh_full = np.array(template_wh_full)  # this is template in Time domain
#     if mask_padded is None:
#         mask_padded = ~np.ones(len(template_wh_full)).astype(bool)  # data is given in Fourier, do there will be no nans in time
#     if splitting == 'snr':
#         # PREPARING TEMPLATE
#         if tlen is None:
#             temp_smthnd = np.abs(template_wh_full - template_wh_full[len(template_wh_full) // 2])
#             # -> the value in the middle is supposed to be close to zero, but sometimes no
#             temp_smthnd = (temp_smthnd + np.roll(temp_smthnd, 1) + np.roll(temp_smthnd, -1)) / 3
#             # -> averaging the rolled template making it smooth
#             inds_small = np.where((temp_smthnd / np.max(temp_smthnd) < 0.001))[0]  # choosing where the template is close to zero
#             inds_small = inds_small[np.where((inds_small > nperseg_psd) & (inds_small <= len(template_wh_full) - nperseg_psd))[0]]
#             # -> if suddenly there is zero in the beginning, ignore
#             tlen = inds_small[0] + (len(temp_smthnd) - inds_small[-1])  # taking parts where the template is not close to zero
#             # this weird procedure is needed because we are dealing with the whitened template, and its shape can be complex,
#             # so at some point it can reach zero, but it does not mean that it is the end of the transit.
#             # We are smoothening it, so that if it is zero only in one point,
#             # the smoothened one will be nonzero.
#             # then, we are setting a threshold and selecting everything non-zero
#             # note the rolling! the transit center should be in the zeroth index!
#         template_wh_full[tlen // 2: len(template_wh_full) - tlen // 2] = 0.
#         template_wh_full = template_wh_full / np.linalg.norm(template_wh_full)
#         # DETERMINING NUMBER OF SPLITTING PARTS
#         chi2_temp_indices = ut._split_template_by_snr(template_wh_full=template_wh_full, n_parts=n_parts, min_bins_part=min_bins_part)
#         n_parts = len(chi2_temp_indices) - 1
#         # FOR TOO SHORT TEMPLATES:
#         # if the template is so short that there is only one part, we will just make one vetting on it and not parts.
#         if n_parts <= 1:
#             is_good = _veto_lobe_time(data_wh_fd=data_wh_fd, template_wh_full=template_wh_full,
#                                       mask_padded=mask_padded, indices_padded=indices_padded, fn=fn, lobe_hlen=lobe_hlen)
#         # FOR LONG ENOUGH TEMPLATES:
#         else:
#             is_good = ut._split_and_match_veto_parts_time(data_wh_fd=data_wh_fd, indices_padded=indices_padded,
#                                                           template_wh_full=template_wh_full, chi2_temp_indices=chi2_temp_indices, fn=fn)
#             if n_parts >= 3:  # at least 3 parts so that there can be less
#                 if n_parts % 2 == 1:
#                     n_parts_additional = 2
#                 else:
#                     n_parts_additional = 3
#                 chi2_temp_indices_additional = ut._split_template_by_snr(template_wh_full=template_wh_full, n_parts=n_parts_additional,
#                                                                          min_bins_part=min_bins_part)
#                 is_good_additional = ut._split_and_match_veto_parts_time(data_wh_fd=data_wh_fd, indices_padded=indices_padded,
#                                                                          template_wh_full=template_wh_full,
#                                                                          chi2_temp_indices=chi2_temp_indices_additional, fn=fn)
#                 is_good = is_good & is_good_additional
#     elif splitting == 'lobe':
#         is_good = _veto_lobe_time(data_wh_fd=data_wh_fd, template_wh_full=template_wh_full,
#                                   mask_padded=mask_padded, indices_padded=indices_padded, fn=fn, lobe_hlen=lobe_hlen)
#     else:
#         raise ValueError("`splitting` must be 'snr' or 'lobe'!")
#     return is_good
#
#
# def _veto_lobe_time(data_wh_fd, template_wh_full, mask_padded, indices_padded, fn, lobe_hlen):
#     hlen = lobe_hlen
#     data_wh_full = np.fft.irfft(data_wh_fd)
#     template_for_fit = np.array(template_wh_full)
#     template_for_fit[hlen: len(template_for_fit) - hlen] = 0
#     template_for_fit = template_for_fit / np.linalg.norm(template_for_fit)
#     template_cut_td = np.concatenate((template_for_fit[-hlen:], template_for_fit[: hlen]))
#     template_for_fit = np.fft.rfft(template_for_fit)
#     stat = np.real(np.fft.irfft(np.conjugate(template_for_fit) * data_wh_fd))
#     stat = stat / ut.outlier_free_std(stat[mask_padded], n_iterations=2, nsig=3)
#     # -> this is supposed to ensure the template normalization, although formally this biases amplitude
#     p_values = []
#     for ind in indices_padded:
#         residuals = (data_wh_full[ind - hlen: ind + hlen] - template_cut_td * stat[ind])
#         residuals = residuals[mask_padded[ind - hlen: ind + hlen]]
#         score = np.sum(residuals ** 2)
#     return score
