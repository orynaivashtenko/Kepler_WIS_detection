import os
core_num = "1"
os.environ["OMP_NUM_THREADS"] = core_num
os.environ["OPENBLAS_NUM_THREADS"] = core_num
os.environ["MKL_NUM_THREADS"] = core_num
os.environ["VECLIB_MAXIMUM_THREADS"] = core_num
os.environ["NUMEXPR_NUM_THREADS"] = core_num
import numpy as np

Index_tid = 0
Index_integral = 1
Index_stat = 2
Index_per = 3
Index_ftt = 4
Index_stat_gauss = 5
Index_template = 6
Trigger_len = 7
Indices_dict = {'target_id': Index_tid, 'int_score': Index_integral,
                'stat': Index_stat, 'per': Index_per, 'ftt': Index_ftt,
                'gauss_stat': Index_stat_gauss, 'template': Index_template}


def load_triggers(file, sep=' '):
    target_ids = []
    integral_scores = []
    stats = []
    pers = []
    ftts = []
    gauss_stats = []
    templates = []
    with open(file, "r") as fil:
        lines = fil.readlines()
    for line in lines:
        target_id, integral_score, stat, per, ftt, gauss_stat, template = trigger_fields_from_line(line=line, sep=sep)
        target_ids.append(target_id)
        integral_scores.append(integral_score)
        stats.append(stat)
        pers.append(per)
        ftts.append(ftt)
        gauss_stats.append(gauss_stat)
        templates.append(template)
    target_ids = np.array(target_ids, dtype=str)
    integral_scores = np.array(integral_scores, dtype=float)
    stats = np.array(stats, dtype=float)
    pers = np.array(pers, dtype=float)
    ftts = np.array(ftts, dtype=float)
    gauss_stats = np.array(gauss_stats, dtype=float)
    templates = np.array(templates, dtype=int)
    return target_ids, integral_scores, stats, pers, ftts, gauss_stats, templates


def print_triggers(target_ids=None, integral_scores=None, stats=None, pers=None, ftts=None, gauss_stats=None,
                   templates=None, file=None, sep=' ', end='\n'):
    with open(file, "w") as fil:
        for i in range(len(target_ids)):
            print_trigger_line(target_id=target_ids[i], integral_score=integral_scores[i], stat=stats[i], per=pers[i],
                               ftt=ftts[i], gauss_stat=gauss_stats[i], template=templates[i],
                               file=fil, sep=sep, end=end)


def list_from_trigger_fields(target_id=None, integral_score=None, stat=None, per=None, ftt=None,
                             gauss_stat=None, template=None, to_tuple=True):
    """ Index_tid = 0
        Index_integral = 1
        Index_stat = 2
        Index_per = 3
        Index_ftt = 4
        Index_stat_gauss = 5
        Index_template = 6
        Trigger_len = 7"""
    list1 = ['' for _ in range(Trigger_len)]
    list1[Index_tid] = target_id
    list1[Index_integral] = integral_score
    list1[Index_stat] = stat
    list1[Index_per] = per
    list1[Index_ftt] = ftt
    list1[Index_stat_gauss] = gauss_stat
    list1[Index_template] = template
    if to_tuple:
        list1 = tuple(list1)
    return list1


def line_from_trigger_fields(target_id=None, integral_score=None, stat=None, per=None, ftt=None,
                             gauss_stat=None, template=None, sep=' '):
    list1 = list_from_trigger_fields(target_id=target_id, integral_score=integral_score, stat=stat, per=per, ftt=ftt,
                                     gauss_stat=gauss_stat, template=template)
    list1 = [str(e) for e in list1]
    line = sep.join(list1)
    return line


def print_trigger_line(target_id=None, integral_score=None, stat=None, per=None, ftt=None, gauss_stat=None,  template=None,
                       trigger_as_list=None, file=None, sep=' ', end='\n'):
    if trigger_as_list is None:
        list1 = list_from_trigger_fields(target_id=target_id, integral_score=integral_score, stat=stat, per=per, ftt=ftt,
                                         gauss_stat=gauss_stat, template=template)
    else:
        list1 = trigger_as_list
    print(*list1, sep=sep, end=end, file=file)


def trigger_fields_from_line(line, sep=' '):
    list1 = line.strip().split(sep)
    target_id, integral_score, stat, per, ftt, gauss_stat, template = trigger_fields_from_list(list1)
    return target_id, integral_score, stat, per, ftt, gauss_stat, template


def trigger_fields_from_list(list1):
    target_id = list1[Index_tid]
    integral_score = float(list1[Index_integral]) if list1[Index_integral] != 'None' else np.nan
    stat = float(list1[Index_stat]) if list1[Index_stat] != 'None' else np.nan
    per = float(list1[Index_per]) if list1[Index_per] != 'None' else np.nan
    ftt = float(list1[Index_ftt]) if list1[Index_ftt] != 'None' else np.nan
    gauss_stat = float(list1[Index_stat_gauss]) if list1[Index_stat_gauss] != 'None' else np.nan
    template = int(list1[Index_template]) if (list1[Index_template] != 'None' and list1[Index_template] != 'nan') else np.nan
    return target_id, integral_score, stat, per, ftt, gauss_stat, template


def trigger_from_line(line, sep=' '):
    target_id, integral_score, stat, per, ftt, gauss_stat, template = trigger_fields_from_line(line=line, sep=sep)
    return Trigger(target_id=target_id, integral_score=integral_score, stat=stat, per=per, ftt=ftt,
                   gauss_stat=gauss_stat, template=template)


def get_trigger_from_target_id(target_id, trigger_list_file):
    target_ids, integral_scores, stats, pers, ftts, gauss_stats, templates = load_triggers(file=trigger_list_file)
    index = np.where(target_ids == target_id)[0][0]
    integral_score = integral_scores[index]
    stat = stats[index]
    per = pers[index]
    ftt = ftts[index]
    gauss_stat = gauss_stats[index]
    template = templates[index]
    return list_from_trigger_fields(target_id=target_id, integral_score=integral_score, stat=stat, per=per, ftt=ftt,
                                    gauss_stat=gauss_stat, template=template)


def copy_trigger_between_files(target_id, source_file, destination_file):
    trigger_as_list = get_trigger_from_target_id(target_id, source_file)
    with open(destination_file, 'a') as fil:
        print_trigger_line(trigger_as_list=trigger_as_list, file=fil)


# def select_one_trigger_per_star(triggers_file='slidable_triggers_run_11.txt'):
#     """
#     trigger is [tid, triggering_integral_score_with_rate_to_baseline, triggering_stat, triggering_period, triggering_ftt, triggering_max_templ, triggering_stat_gauss]
#     """
#     with open(triggers_file, 'r') as fil:
#         all_triggers = fil.readlines()
#
#     all_triggers = [line.strip().split() for line in all_triggers]
#     tids_tce_set = list(set([line[0] for line in all_triggers]))
#     all_triggers = np.array(
#         [[str(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4]), int(float(x[5])), float(x[6])] for x in all_triggers],
#         dtype='object')
#
#     triggers_per_tids = [[all_triggers[i] for i in range(len(all_triggers)) if all_triggers[i][0] == tid] for tid in tids_tce_set]
#     # max_triggers_tids = [triggers_per_tid[np.argmax([triggers_per_tid[i][1] for i in range(len(triggers_per_tid))])] for triggers_per_tid in triggers_per_tids]
#     max_triggers_tids = []
#     # Resolving conflits on the edges of the chunk when sometimes we do not catch the maximum but do catch the integral.
#     # And we want to select the trigger which does contain the maximum
#     delta_period = 0.5
#     delta_integral = 0.2
#     delta_stat = 1.
#     for ii, tid in enumerate(tids_tce_set):
#         triggers_per_tid = [all_triggers[i] for i in range(len(all_triggers)) if all_triggers[i][0] == tid]
#         max_trigger = triggers_per_tid[np.argmax([triggers_per_tid[i][1] for i in range(len(triggers_per_tid))])]
#         for trigger in triggers_per_tid:
#             if (np.abs(trigger[3] - max_trigger[3]) < delta_period) and (max_trigger[1] - trigger[1] < delta_integral) and (
#                     trigger[2] - max_trigger[2] > delta_stat):
#                 max_trigger = trigger
#         max_triggers_tids.append(max_trigger)
#
#     return max_triggers_tids

# ##########################################################################################################

class Trigger(object):
    def __init__(self, target_id=None, integral_score=None, stat=None, per=None, ftt=None, gauss_stat=None, template=None):
        self.target_id = target_id
        self.stat = stat
        self.per = per
        self.ftt = ftt
        self.gauss_stat = gauss_stat
        self.int_score = integral_score
        self.template = template

    def from_line(self, line, sep=' '):
        target_id, integral_score, stat, per, ftt, gauss_stat, template = trigger_fields_from_line(line=line, sep=sep)
        self.target_id = target_id
        self.stat = stat
        self.per = per
        self.ftt = ftt
        self.gauss_stat = gauss_stat
        self.int_score = integral_score
        self.template = template

    def print_trigger(self, file=None):
        print_trigger_line(target_id=self.target_id, stat=self.stat, per=self.per, ftt=self.ftt, gauss_stat=self.gauss_stat,
                           integral_score=self.int_score, template=self.template, file=file)


# #####################################################################################################################
# If we want to use normal and gaussianized together:
# #####################################################################################################################

def print_full_trigger(target_id=None, stat=None, per=None, ftt=None, gauss_stat=None, integral_score=None, template=None,
                       g_stat=None, g_per=None, g_ftt=None, g_gauss_stat=None, g_int_score=None, g_template=None,
                       file=None):
    line = ['' for _ in range(Trigger_len * 2 - 1)]
    line[Index_tid] = target_id
    line[Index_integral] = integral_score
    line[Index_stat] = stat
    line[Index_per] = per
    line[Index_ftt] = ftt
    line[Index_stat_gauss] = gauss_stat
    line[Index_template] = template
    line[Index_integral - 1 + Trigger_len] = g_int_score
    line[Index_stat - 1 + Trigger_len] = g_stat
    line[Index_per - 1 + Trigger_len] = g_per
    line[Index_ftt - 1 + Trigger_len] = g_ftt
    line[Index_stat_gauss - 1 + Trigger_len] = g_gauss_stat
    line[Index_template - 1 + Trigger_len] = g_template
    print(*line, sep=' ', end='\n', file=file)


class FullTrigger(object):
    def __init__(self, target_id=None, stat=None, per=None, ftt=None, gauss_stat=None, integral_score=None, template=None,
                 g_stat=None, g_per=None, g_ftt=None, g_gauss_stat=None, g_int_score=None, g_template=None):
        self.target_id = target_id
        self.stat = stat
        self.per = per
        self.ftt = ftt
        self.gauss_stat = gauss_stat
        self.int_score = integral_score
        self.template = template
        self.g_stat = g_stat
        self.g_per = g_per
        self.g_ftt = g_ftt
        self.g_gauss_stat = g_gauss_stat
        self.g_int_score = g_int_score
        self.g_template = g_template

    def print_trigger(self, file=None):
        print_full_trigger(target_id=self.target_id, stat=self.stat, per=self.per, ftt=self.ftt, gauss_stat=self.gauss_stat,
                           integral_score=self.int_score, template=self.template,
                           g_stat=self.g_stat, g_per=self.g_per, g_ftt=self.g_ftt, g_gauss_stat=self.g_gauss_stat,
                           g_int_score=self.g_int_score, g_template=self.g_template,
                           file=file)


