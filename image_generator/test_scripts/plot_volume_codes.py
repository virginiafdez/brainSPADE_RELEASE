'''
To see the impact of the slice number in the code of SPADE's latent space, this script iterates through several
volumes, and plots the codes of the slices belonging to the same volume passed through the encoder.
Different shapes are used per subject, different colours per modality.

'''

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from models.pix2pix_model import Pix2PixModel
from data.spadenai_v2 import SpadeNai
from options.lookatcodes_options import LACodeOptions
from sklearn.manifold import TSNE

# Options, dataloader and model
opt = LACodeOptions().parse()
model = Pix2PixModel(opt)

# Creation of folder
if not os.path.exists(opt.savecode_dir):
    os.makedirs(opt.savecode_dir)
name_file = "plot_volume_codes.png"

# Other variables
mods = opt.sequences
subjects_per_mod = 3 # Must be < len(shapes) below.

# Colors
colors = ['green', 'blue', 'pink', 'orange']
shapes = ["^", ".", "*", "D", "s", "+"]
colors_sub = {'green': ['darkgreen', 'lime', 'olive', 'darkolivegreen', 'lawngreen'],
              'blue': ['royalblue', 'dodgerblue', 'navy', 'turquoise', 'powderblue'],
              'pink': ['hotpink', 'magenta', 'mediumvioletred', 'lightpink', 'violet'],
              'orange': ['darksalmon', 'coral', 'orange', 'peru', 'orange']}

# Start
print("Do reduction starting... ")
# Windows for TSNE
PER = range(15, 55, 10) # Perplexity. Number of close neighbours considered for each point.
LR_ = range(300, 700, 100)
codes = {}
codes_dataset = {}
codes_nr = []
datasets = []
paths = []
images = []
modalities = []
subject_id = []

# Forward pass and append codes
opt.diff_style_volume = False # Ensure correspondance between image and style.
if subjects_per_mod > len(shapes):
    subjects_per_mod = len(shapes)

# Storage of codes.
for mod_i, mod in enumerate(mods):
    opt.fix_seq = mod
    dataset_container = SpadeNai(opt, mode='test')
    if mod_i == 0:
        subjects = np.random.choice(dataset_container.input_vol_dict, subjects_per_mod)
    else:
        for sub in subjects:
            img_file = sub['image_file']
            subname = img_file.split("/")[-1].replace(mods[mod_i-1], mod)
            sub['image_file'] = os.path.join("/".join(img_file.split("/")[:-1]), subname)
    for sub_ind, subject in enumerate(subjects): # Invididual subject slice-by-slice iteration.
        dataset_container.setWorkingVolume(
            image_paths={mod: os.path.join(dataset_container.image_dir, subject['image_file'])},
            label_path=  os.path.join(dataset_container.label_dir, subject['label_file']))
        for i in range(int(np.ceil(dataset_container.lenIteratorWV() / opt.batchSize))):
                data_i = dataset_container.iterateThroughWV(mod, opt.batchSize, use_same_style_slice=False)
                z, _ = generated = model(data_i, mode='encode_only').detach().cpu()
                codes_nr.append(generated)
                modalities += data_i['this_seq']
                paths += data_i['label_path']
                subject_id += sub_ind
dataset_container.clearCache()

# TSNE
codes_nr = torch.cat(codes_nr, dim=0).numpy()
subject_id = np.asarray(subject_id)
modalities = np.asarray(modalities)

for lr in LR_:
    for per in PER:
        Y = TSNE(n_components=2, init='random', random_state=0, perplexity=per,
                 learning_rate=lr).fit(codes_nr).fit_transform(codes_nr)

        plt.figure(figsize=(10, 7))
        minim = [10000, 10000, 10000]
        maxim = [-10000, -10000, -10000]
        legend_ = []
        for mod_ind, mod in enumerate(mods):
            for sub_id in range(len(subjects)):
                if Y[(modalities == mod) * (subject_id == sub_id) ,0].shape[0] == 0:
                    continue
                plt.scatter(Y[(modalities == mod) * (subject_id == sub_id) ,0],
                            Y[(modalities == mod) * (subject_id == sub_id) ,1],
                            color = colors_sub[colors[mod_ind]][0],
                            s = 8, marker=shapes[sub_id])
                legend_.append("%s-%s" %(mod, sub_id))
                plt.legend(legend_)
            name_fig = "TSNE_slice-impact_per%d_lr%d.png" %('TSNE', per, lr)
            plt.title("Average codes TSNE - perplexity %d - lr %d" %(per, lr))
            plt.savefig(os.path.join(opt.savecode_dir, name_file))
