import os
import shutil
import collections

checkpoints = "/home/vf19/Documents/brainSPADE_2D/brainSPADE_1DEC/BRAINSPADEV3_23"
sub_folder = "D_outputs"
del_freq = 2
leave = 500

def sorter(items, folder):
    # Code plots: B0_ep4_it1080_code_2
    # Validation: validation_epoch_inference_5998
    # Images: epoch_5999_iter_3137604
    sorted_ = []
    ordered = {}
    for item_ in items:
        it = item_.strip(".png")
        if folder == 'images':
            it_sp = it.split("_")
            if int(it_sp[1]) not in ordered.keys():
                ordered[int(it_sp[1])] = {int(it_sp[3]):item_}
            else:
                ordered[int(it_sp[1])][int(it_sp[3])] = item_
        elif folder == 'validation_results':
            it_sp = it.split("_")
            ordered[int(it_sp[-1])] = item_
        elif folder == 'D_outputs':
            it_sp = it.split("_")
            if int(it_sp[3]) not in ordered.keys():
                ordered[int(it_sp[3])] = {int(it_sp[-1]):item_}
            else:
                ordered[int(it_sp[3])][int(it_sp[-1])] = item_
        elif folder == 'code_plots':
            it_sp = it.split("_")
            if int(it_sp[1].strip("ep")) not in ordered.keys():
                ordered[int(it_sp[1].strip("ep"))] = {int(it_sp[2].strip("it")): {
                    int(it_sp[-1].strip("code")): item_
                }}
            else:
                if int(it_sp[2].strip("it")) not in ordered[int(it_sp[1].strip("ep"))].keys():
                    ordered[int(it_sp[1].strip("ep"))][int(it_sp[2].strip("it"))] = {
                        int(it_sp[-1].strip("code")): item_
                    }
                else:
                    ordered[int(it_sp[1].strip("ep"))][int(it_sp[2].strip("it"))][int(it_sp[-1].strip("code"))] \
                        = item_

    # Ordered by epochs
    ordered = collections.OrderedDict(sorted(ordered.items()))
    if folder == 'validation_results':
        sorted_ = list(ordered.values())
    elif folder == 'images' or folder == 'D_outputs':
        for epoch, all_iters in ordered.items():
            order_epoch = collections.OrderedDict(sorted(all_iters.items()))
            sorted_ += list(order_epoch.values())
    elif folder == 'code_plots':
        for epoch, all_iters in ordered.items():
            order_epoch = collections.OrderedDict(sorted(all_iters.items()))
            for it, all_its in order_epoch.items():
                order_it = collections.OrderedDict(sorted(all_its.items()))
                sorted_ += list(order_it.values())


    return sorted_

look_at_items = [i for i in os.listdir(os.path.join(checkpoints, 'web', sub_folder)) if ".png" in i]
sorted_ = sorter(look_at_items, folder = sub_folder)
sorted_ = sorted_[:-leave]
to_remove = []
for s_ind, s in enumerate(sorted_):
    if s_ind % del_freq == 0:
        to_remove.append(os.path.join(checkpoints, 'web', sub_folder, s))

for r in to_remove:
    os.remove(r)