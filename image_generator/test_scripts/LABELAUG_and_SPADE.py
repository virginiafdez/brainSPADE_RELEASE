'''
Instead of using LDMs to generate the label, we use a set of base real labels, augment them, and then forward them through SPADE.
'''

import matplotlib.pyplot as plt
import torch
import os
import numpy as np
from options.test_options import FullPipelineOptions
from data.spadenai_v2_sliced import SpadeNaiSlice
from models.pix2pix_model import Pix2PixModel
import moreutils as uvir
from monai.data.dataloader import DataLoader
from datetime import datetime
from util import util
from models.modality_discrimination.modisc_v2 import Modisc
from data.sampling_ds_utils import mod_disc_pass_Accuracy
from copy import deepcopy
import monai
import nibabel as nib

# CONTROL VARIABLES
channels = {'generic': 6, 'wmh': 6, 'tumour': 7}
lesion_tokens = {1: 'wmh', 2: 'tumour'}
n_healthy_labels = 5
save_as_npy = True

############################################### INPUT SETTINGS
stage_1 = True
stage_2 = True

# For stage 1
path_to_raw_labels = ""
result_save_dir = "/home/vf19/Documents/brainSPADE_2D/DATA/TUMOUR_SHIFT_EXPERIMENT/ADNIVAL_SYNTHETIC/validation"
n_aug = 10 # Number of augmentations per original label.
batch_size = 8
plot_raw_labels = True
transforms =  monai.transforms.RandAffine(scale_range=(0.2, 0.2, 0.2), rotate_range=(0.5, 0.5, 0.5),
                            translate_range=(3, 3, 3), mode='bilinear',
                            as_tensor_output=False, prob=1.0, padding_mode='border')
datasets_present = ["OAS", "ADNI", "SABRE", "BRATS-TCIA", "WMH", ]

# For stage 2
brainspade_checkpoint = "/home/vf19/Documents/brainSPADE_2D/brainSPADE_1DEC"
brainspade_name = "BRAINSPADEV3_20"
target_datasets = ["ADNI60"]
path_to_styles = "/home/vf19/Documents/brainSPADE_2D/DATA/MICCAI22/ADNI_REORG_TR/styles_for_SPADE"
appendix_styles = "style_ADNI60" # If the folders are named other than "style"
path_to_styles_labels = "/home/vf19/Documents/brainSPADE_2D/DATA/MICCAI22/ADNI_REORG_TR/styles_for_SPADE/style_ADNI60_mask"
spade_channels = {'healthy': 6, 'total': 8, 'lesions': 2}
lesions = [] # Which lesions are part of this.
n_passes = 1 # Number of passes through the same dataset of labels.
format = 'npy' # Either "spade", "png" or "npy". Spade is a npz with keys: img, img_affine, img_header and modality.


# INITIAL SET UP -------------------------------------------------------------------------------------------------
# Create directories
if not os.path.isdir(result_save_dir):
    os.makedirs(result_save_dir)
if not os.path.isdir(os.path.join(result_save_dir,"labels")):
    os.makedirs(os.path.join(result_save_dir,"labels"))
    with open(os.path.join(result_save_dir, 'labels_rendered.txt'), 'w') as f:
        f.write("NOLABS")
        f.close()
    rendered_labels = False
else:
    r = ""
    with open(os.path.join(result_save_dir, 'labels_rendered.txt'), 'r') as f:
        r = f.readlines()[0]
        f.close()
    if r == "NOLABS":
        rendered_labels = False
    else:
        rendered_labels = True
if not os.path.isdir(os.path.join(result_save_dir,"labels")):
    os.makedirs(os.path.join(result_save_dir,"labels"))
if not os.path.isdir(os.path.join(result_save_dir,"examples")):
    os.makedirs(os.path.join(result_save_dir,"examples"))

################# Initialise and load models_ ####################################
opt = FullPipelineOptions().load_options(os.path.join(brainspade_checkpoint, brainspade_name, 'opt'))
opt.label_dir = os.path.join(result_save_dir, 'labels')
datasets_original = deepcopy(opt.datasets)
opt.datasets += target_datasets
device = torch.device("cuda")
brainspade = Pix2PixModel(opt).eval()
# To make sure that generated images flal within the modalities
modality_discriminator = Modisc(len(opt.sequences), len(datasets_original), 0.2, 2, 1)
modality_discriminator = util.load_network(modality_discriminator, 'MD',
                                           -1, opt, strict = False).eval()
max_attempts_per_img = 20 # Up to 200 instances per img.

# We need to create the models_ and load their state dict
plot_every = 15
if plot_raw_labels:
    if not os.path.isdir(os.path.join(result_save_dir, 'examples_labels_raw')):
        os.makedirs(os.path.join(result_save_dir, 'examples_labels_raw'))

# Label sampling process ----------------------------------------------------------------------------------
if stage_1 and not rendered_labels:
    if not os.path.isdir(os.path.join(result_save_dir, 'labels_pv_NPY')):
        os.makedirs(os.path.join(result_save_dir, 'labels_pv_NPY'))
    all_original_labels = os.listdir(path_to_raw_labels)
    processed = 0
    for lab in all_original_labels:
        label = np.load(os.path.join(path_to_raw_labels, lab))
        label = np.transpose(label, [-1, 0, 1])
        for aug_i in range(n_aug):
            label_i = transforms(label)
            new_name = "%s_%d" %(lab.split(".")[0], aug_i)
            # We plot if needed
            if plot_raw_labels and processed % plot_every == 0:
                f, (a0, a1) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, label_i.shape[0]]},
                                           figsize=(label_i.shape[0] * 3, 3))
                a0p = a0.imshow(np.argmax(label_i, 0))
                out = [label_i[i, ...] for i in range(label_i.shape[0])]
                out = np.concatenate(out, -1)
                a1p = a1.imshow(out, vmin=0.0, vmax=1.0)
                f.colorbar(a1p, ax=a1, fraction=0.006)
                plt.savefig(os.path.join(result_save_dir, 'examples_labels_raw', "sample_%d.png" %processed))
                plt.close(f)
            # Save labels
            label_i_npy = deepcopy(label_i)
            label_i = np.expand_dims(label_i, -1) # From CxHxW > CxHxWx1
            label_i = np.transpose(label_i, [1, 2, 3, 0]) # From CxHxWx1 > HxWx1xC
            np.save(os.path.join(result_save_dir, 'labels_pv_NPY', "%s%s" %(new_name, ".npy")),
                    label_i_npy)
            np.savez(os.path.join(result_save_dir, 'labels', "%s%s" %(new_name, ".npz")),
                     label = label_i,
                     slice = int(lab.split(".")[0].split("_")[-1]),
                     dataset_name = uvir.findDataset(datasets_present, lab))

            processed += 1

    with open(os.path.join(result_save_dir, 'labels_rendered.txt'), 'w') as f:
        f.write("LABS")
        f.close()


# Brainspade call ----------------------------------------------------------------------------------
if stage_2:
    disp_every = 20
    brainspade.to(device)
    modality_discriminator.to(device)
    log_file = os.path.join(result_save_dir, "run_logs.txt")
    with open(log_file, 'w') as f:
        f.write(str(datetime.today))
    modalities = ['T1', 'FLAIR']

    for mod in modalities:
        print("Processing modality %s" %mod)
        # We modify the settings for the dataset created for each modality
        opt.image_dir =  os.path.join(path_to_styles, "%s_%s" %(appendix_styles,mod))
        opt.fix_seq = mod
        opt.nThreads = 0
        opt.label_dir = os.path.join(result_save_dir, 'labels')
        opt.fix_seq = mod
        opt.non_corresponding_dirs = True
        opt.style_label_dir = os.path.join(path_to_styles_labels)
        colors = uvir.get_rgb_colours()

        # Set up place where we'll save images.
        result_im_dir = os.path.join(result_save_dir, 'images_%s' %mod)
        if not os.path.isdir(result_im_dir):
            os.makedirs(result_im_dir)

        # Create dataset
        dataset_container = SpadeNaiSlice(opt, mode="test")
        dataloader = DataLoader(dataset_container.sliceDataset, batch_size=opt.batchSize, shuffle=False,
                                num_workers=int(opt.nThreads), drop_last=opt.isTrain,
                                )

        # Loop: number of instances per label
        for p in range(n_passes):
            dataset_container.resetDatasets(fixed_modality=mod)
            dataloader = DataLoader(dataset_container.sliceDataset, batch_size=opt.batchSize, shuffle=False,
                                    num_workers=int(opt.nThreads), drop_last=opt.isTrain,
                                    )

            # Enumerate all labels
            for ind, i in enumerate(dataloader):
                print("Batch %d / %d" % (ind, len(dataset_container)))

                gen = brainspade(i, 'inference') # Forward
                mod_accuracy = mod_disc_pass_Accuracy(modality_discriminator, gen, i,
                                                      opt.sequences) # Modality accuracy

                # We try until our modality discriminator doesn't fail.
                n_attempts = 0
                accuracy_th = mod_accuracy < 0.75 # Threshold
                while True in accuracy_th and n_attempts < max_attempts_per_img:
                    # Switch style
                    i_tuned = dataset_container.switchStyles(i, flags=accuracy_th)
                    gen = brainspade(i_tuned, 'inference')
                    mod_accuracy = mod_disc_pass_Accuracy(modality_discriminator, gen, i_tuned,
                                                      opt.sequences)
                    accuracy_th = mod_accuracy < 0.75
                    n_attempts += 1
                if n_attempts == max_attempts_per_img:
                    print("Number of maximum attempts reached for %d out of %d images in the batch"
                          %(np.sum(accuracy_th), opt.batchSize))

                # Save the resulting images
                for b in range(gen.shape[0]):
                    file_name = i['label_path'][b].split("/")[-1].replace("Parcellation", i['this_seq'][b])
                    to_save_img = gen[b, ...].detach().cpu()
                    to_save_img = util.tensor2im(to_save_img) # Scale 0-255
                    if format == 'png':
                        uvir.saveSingleImage(to_save_img, None, os.path.join(result_save_dir,
                                                                                     result_im_dir, file_name.replace(".npz", "_%d.png" %p)),
                                             skullstrip=False)
                    elif format == 'spade':
                        np.savez(os.path.join(result_save_dir, result_im_dir, file_name.replace(".npz", "_%d.npz" %p)),
                                 img = to_save_img.unsqueeze(0).numpy(), img_affine = i['affine'][b,...],
                                 modality = i['this_seq'][b])
                    elif format == 'npy':
                        np.save(os.path.join(result_save_dir, result_im_dir, file_name.replace(".npz", "_%d.npy" %p)),
                                to_save_img[..., 1])
                    else:
                        ValueError("Unknown format %s. It can either be spade, npy or png." %format)

                    # Plot results
                    if p == 0 and (ind*opt.batchSize + b) % disp_every == 0:
                        f = plt.figure()
                        repeated_img = util.tensor2im(i['style_image'][b, ...].detach().cpu())
                        out = np.concatenate([repeated_img, to_save_img,
                                              colors[torch.argmax(i['label'][b, ...].detach().cpu(), 0).numpy()]],
                                              1)
                        plt.imshow(out)
                        plt.title("Generated label")
                        plt.savefig(os.path.join(result_save_dir, 'examples', file_name.replace(".npz", ".png")))
                        plt.close(f)

                    with open(log_file, 'a') as f:
                        f.write("%s: LABEL %s IMAGE %s\n" %(file_name.split(".")[0],
                                                            i['label_path'][b],
                                                            i['image_path'][b]))

            dataset_container.clearCache()

            print("Fin de passe %d" %p)

        with open(os.path.join(result_save_dir, 'log.txt'), 'w') as f:
            f.write("%s\n" %str(datetime.today()))
            f.write("Original label paths: %s\n" %path_to_raw_labels)
            f.write("Number augmentations: %d\n" %n_aug)
            f.write("Brainspade model: %s\n" %os.path.join(brainspade_checkpoint, brainspade_name))
            f.write("Number of passes per modality: %d\n" %n_passes)
            f.close()