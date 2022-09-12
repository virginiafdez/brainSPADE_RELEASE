import numpy as np
import itertools
import torch
DISEASE = {'wmh': 0, 'tumour': 1, 'edema': 2}

def get_slice_nos(n_items, n_intervals = 3, random = False, mode = 'interleaved'):
    '''
    Get n_items sliced numbers between 0 and 1.0.
    :param n_items: Number of items (list) you require
    :param n_intervals: Number of intervals between which to divide 1.0 (aka: 4 will result in 4 blocks: 0.0-0.25, 0.25-0.5 etc)
    :param random: Whether within each block, sample average between upper and lower interval thresholds (False), or sample random number (True)
    :param mode: interleaved (we sample from one interval, then the other, then the other and restart)
    or stacked (we sample N elements from an interval, then N from another interval etc.)
    :return:
    '''
    space = 1.0 / n_intervals
    intervals = []
    for i in range(n_intervals):
        intervals.append([space * i, space * (i + 1)])
    slice_no = []
    if mode == 'interleaved':
        interval_counter = 0
        for item_ in range(n_items):
            if random:
                slice_no.append(np.random.uniform(intervals[interval_counter][0], intervals[interval_counter][1]))
            else:
                slice_no.append((intervals[interval_counter][0] + intervals[interval_counter][1])/2)
            interval_counter += 1
            if interval_counter == len(intervals):
                interval_counter = 0
    elif mode == 'stacked':
        n_elements_per_interval = np.floor(n_items/len(intervals))
        for interval in intervals:
            for element in range(n_elements_per_interval):
                if random:
                    slice_no.append(np.random.uniform(interval[0], interval[1]))
                else:
                    slice_no.append((interval[0] + interval[1]) / 2.0)

    return slice_no

def even_conditioning_list_noslice():
    possibilities = [list(l) for l in list(itertools.product([0, 1], repeat=len(DISEASE.values())))]
    return possibilities

def wmh_conditioning_list_noslice():
    return [[1,0,0]]

def tumour_conditioning_list_noslice():
    return [[0, 1, 0], [0, 0, 1], [0, 1, 1]]

def nolesion_conditioning_list_noslice():
    return [[0,0,0]]

def even_conditioning_list(n_slices_per_disease_comb = 3, n_slices_per_interval = 1):

    possibilities = [list(l) for l in list(itertools.product([0,1], repeat=len(DISEASE.values())))]
    slice_numbers = get_slice_nos(len(possibilities)*n_slices_per_disease_comb*n_slices_per_interval,
                                  n_intervals=n_slices_per_disease_comb,
                                  random = True, mode = 'interleaved')

    out_cond = []
    for p_ind, p in enumerate(possibilities):
        for int_r in range(n_slices_per_disease_comb):
            out_cond.append(p + [slice_numbers[p_ind*n_slices_per_disease_comb + int_r]])

    return  out_cond

def wmh_conditioning_list(n_slices_per_disease_comb=3, n_slices_per_interval = 1):

    possibilities = [[1,0,0]]
    slice_numbers = get_slice_nos(len(possibilities) * n_slices_per_disease_comb * n_slices_per_interval,
                                  n_intervals=n_slices_per_disease_comb,
                                  random=True, mode='interleaved')
    out_cond = []
    for p_ind, p in enumerate(possibilities):
        for int_r in range(n_slices_per_disease_comb):
            out_cond.append(p + [slice_numbers[p_ind*n_slices_per_disease_comb + int_r]])

    return  out_cond

def tumour_conditioning_list(n_slices_per_disease_comb=3, n_slices_per_interval=1):

    possibilities = [[0, 1, 0], [0, 0, 1], [0, 1, 1]]
    slice_numbers = get_slice_nos(len(possibilities) * n_slices_per_disease_comb * n_slices_per_interval,
                                  n_intervals=n_slices_per_disease_comb,
                                  random=True, mode='interleaved')
    out_cond = []
    for p_ind, p in enumerate(possibilities):
        for int_r in range(n_slices_per_disease_comb):
            out_cond.append(p + [slice_numbers[p_ind*n_slices_per_disease_comb + int_r]])

    return  out_cond

def nolesion_conditioning_list(n_slices_per_disease_comb=3, n_slices_per_interval=1):

    possibilities = [[0, 0, 0]]
    slice_numbers = get_slice_nos(len(possibilities) * n_slices_per_disease_comb * n_slices_per_interval,
                                  n_intervals=n_slices_per_disease_comb,
                                  random=True, mode='interleaved')
    out_cond = []
    for p_ind, p in enumerate(possibilities):
        for int_r in range(n_slices_per_disease_comb):
            out_cond.append(p + [slice_numbers[p_ind*n_slices_per_disease_comb + int_r]])

    return  out_cond

def check_conditioning(cond_tokens, x_hat, slice = False, exclusive = False):

    channels = {'generic': 6, 'wmh': 6, 'tumour': 7, 'edema': 8}
    d_token = {'wmh': 0, 'tumour': 1, 'edema': 2}

    if slice:
        disease_tokens = cond_tokens[:-1]
    else:
        disease_tokens = cond_tokens

    if 1 not in disease_tokens:
        # No lesion
        if torch.sum(x_hat[min(channels.values()):, ...] > 0.5) == 0:
            return True
        else:
            return False

    count = 0
    for dt_ind, dt in enumerate(disease_tokens):
        if dt == 1:
            if torch.sum(x_hat[channels[d_token[dt]]] > 0.5) == 0:
                if exclusive:
                    count += 1
                else:
                    return True
    if exclusive:
        if count >= disease_tokens.sum():
            return True
        else:
            return False




