'''
Inefficient, but RAM-friendly code that forwards a semantic map through the generator (with random style)
and retrieves the training sample that looks more similar, then plots them together.
'''

from data.spadenai_v2_sliced import SpadeNaiSlice
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from monai.data.dataloader import DataLoader
from torch.nn import L1Loss as L1
from options.memorising_options import MemorisingOptions
from models.pix2pix_model import Pix2PixModel
from skimage.metrics import structural_similarity as ssim
import util.util as util
import moreutils as uvir
from copy import deepcopy

######################### FUNCTIONS #######################################################

def findNearest(all_subjects, this_subject, loss = 'l1', first = 1):
    '''
    For a set of subjects, and a subject that doesn't belong to that set,
    find its nearest neighbour based on
    :param all_subjects:
    :param this_subject:
    :param loss:
    :param first:
    :return:
    Returns the best K elements.
    '''

    possible_losses = ['l1', 'l2', 'ssim']
    if loss not in possible_losses:
        ValueError("Supported comparative losses are l1, l2 and ssim. Enter any of those"
                   "as argument loss or leave to default l1.")
    extended_this = np.repeat(np.expand_dims(this_subject, 0), all_subjects.shape[0], axis = 0)
    across_dimensions = []
    for ind, i in enumerate(extended_this.shape):
        if ind != 0:
            across_dimensions.append(ind)
    if loss == 'l1':
        diff = np.mean(np.abs((all_subjects-extended_this)), axis = tuple(across_dimensions))
    elif loss == 'l2':
        diff = np.mean(np.sqrt((all_subjects-extended_this)**2), axis = tuple(across_dimensions))
    elif loss == 'ssim':
        diff = []
        for sub in range(all_subjects.shape[0]):
            # We log 100 - SSIM (to have a minimum = better behavior)
            diff.append(100-ssim(all_subjects[sub,...], extended_this[sub,...]))
        diff = np.asarray(diff)
    best_indices = np.argsort(diff)
    diff = diff[best_indices]
    return diff[:first], best_indices[:first]


######################### MAIN ############################################################
options = MemorisingOptions().parse()
if not os.path.isdir(options.results_dir):
    os.makedirs(options.results_dir)
model = Pix2PixModel(options).eval()
sequences = options.sequences
print("Loading all the dataset...")

# Modify options for the base dataset
options_base = deepcopy(options)
options_base.image_dir = options.image_dir_base
options_base.label_dir = options.label_dir_base
dataset_base = SpadeNaiSlice(options_base, 'test')
dataloader = DataLoader(dataset_base.sliceDataset, batch_size=options.batchSize, shuffle=False,
                        num_workers=options.nThreads,
                        drop_last=False)
whole_set = []
with torch.no_grad():
    for ind, data_i in enumerate(dataloader):
        whole_set.append(data_i['image'][:, 1:2, ...].detach().cpu().numpy())
whole_set = np.concatenate(whole_set)

# Options and fake dataset
dataset = SpadeNaiSlice(options, 'test')
dataloader = DataLoader(dataset.sliceDataset, batch_size=options.batchSize, shuffle=False,
                        num_workers=options.nThreads,
                        drop_last=False)

for mod in sequences:
    # Resets dataset
    options.fix_seq = mod
    options.diff_slice = False # We want style and image to be the same (corresponding with label)
    dataset = SpadeNaiSlice(options, 'test')
    dataloader = DataLoader(dataset.sliceDataset, batch_size=options.batchSize, shuffle=False,
                            num_workers=options.nThreads,
                            drop_last=False)
    print("Iterating and forwarding...")
    for ind, data_i in enumerate(dataloader):
        if ind > np.ceil(options.max_n_ims/options.batchSize):
            break
        if options.no_style_test:
            if not os.path.isdir(os.path.join(options.results_dir, 'no_style')):
                os.makedirs(os.path.join(options.results_dir, 'no_style'))
            generated_ns = model(data_i, mode='random_style_inference')
            generated_ns = generated_ns.detach().cpu().numpy()
            for b in range(generated_ns.shape[0]):
                diffs, diff_inds = findNearest(whole_set, generated_ns[b, ...],
                                               loss='l1', first=5)
                closest_ims = whole_set[diff_inds, ...]

                # Plot
                to_plot = [util.tensor2im(torch.from_numpy(generated_ns[b,...]))]
                for c in range(closest_ims.shape[0]):
                    to_plot.append(util.tensor2im(torch.from_numpy(closest_ims[c, ...])))
                tile = uvir.arrange_tile(to_plot)
                f = plt.figure(figsize=(25,25))
                plt.imshow(tile)
                plt.axis('off')
                plt.title("max. diff %.3f, min. diff %.3f" % (max(diffs), min(diffs)),
                          fontsize = 28)
                plt.savefig(os.path.join(options.results_dir, 'no_style',
                                         "training_sample_%s_%d.png" % (mod, ind*options.batchSize+b)))
                plt.close(f)

        if options.style_test:
            if not os.path.isdir(os.path.join(options.results_dir, 'style')):
                os.makedirs(os.path.join(options.results_dir, 'style'))
            generated_s = model(data_i, mode = 'inference')
            generated_s = generated_s.detach().cpu().numpy()
            for b in range(generated_s.shape[0]):
                diffs, diff_inds = findNearest(whole_set, generated_s[b, ...],
                                               loss = 'l1', first=10)
                closest_ims = whole_set[diff_inds, ...]

                # Plot
                to_plot = [util.tensor2im(torch.from_numpy(generated_s[b, ...]))]
                for c in range(closest_ims.shape[0]):
                    to_plot.append(util.tensor2im(torch.from_numpy(closest_ims[c, ...])))
                tile = uvir.arrange_tile(to_plot)
                f = plt.figure(figsize=(25,25))
                plt.imshow(tile)
                plt.axis('off')
                plt.title("max. diff %.3f, min. diff %.3f" %(max(diffs), min(diffs)),
                          fontsize = 28)
                plt.savefig(os.path.join(options.results_dir, 'style',
                                         "training_sample_%s_%d.png" %(mod, ind*options.batchSize+b)))
                plt.close(f)

dataset.clearCache()








