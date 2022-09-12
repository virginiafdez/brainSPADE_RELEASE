'''
LABEL AND IMAGE GENERATOR WITH VOLUMETRIC PARTIAL VOLUME MAPS LABELS
Partial Volume Maps:
TYPICALLY (according to GIF nomenclature)
Channel 0: Background
Channel 1: CSF
Channel 2: Grey Matter
Channel 3: White Matter
Channel 4: Deep Grey Matter
Channel 5: Brainstem
Channel 6: WMH
Channel 7: Tumour
Channel 8: Edema
Subsequent: empty
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
from models.VQVAE_DDPM.import_utils import define_VAE3D, define_DDPM_conditioned, define_DDPM_unconditioned
from omegaconf import OmegaConf
from vqvae_ldm.training_and_testing.models_.ddim import DDIMSampler
from util import util
from models.modality_discrimination.modisc_v2 import Modisc
from data.sampling_ds_utils import mod_disc_pass_Accuracy
from copy import deepcopy
import monai
import nibabel as nib
from test_scripts.conditioning_lists import even_conditioning_list, nolesion_conditioning_list, \
    wmh_conditioning_list, tumour_conditioning_list, tumour_conditioning_list_noslice, wmh_conditioning_list_noslice, \
    nolesion_conditioning_list_noslice, even_conditioning_list_noslice

# CONTROL VARIABLES
channels = {'generic': 6, 'wmh': 6, 'tumour': 7, 'edema': 8}
lesion_tokens = {1: 'wmh', 2: 'tumour', 3: 'edema'}
n_healthy_labels = 5 # Number of healthy label channels (excl .background)
save_as_npy = True # Whether to, in addition to saving as NPZ (for SPADE), save as NPY

################INPUT SETTINGS#########################
stage_1 = True # Whether to perform label generation
stage_2 = True # Whether to perform style image generation

# For stage 1
##### INPUT BY USER
vq_model_path = "/media/vf19/BigCrumb/LESION_GENERATOR/LESION_GENERATOR/VAE_GDL_MIX_L11/best_model.pth"
vq_config_file = "/media/vf19/BigCrumb/LESION_GENERATOR/LESION_GENERATOR/VAE_GDL_MIX_L11/config_file.yaml"
ldm_model_path = "/media/vf19/BigCrumb/LESION_GENERATOR/LESION_GENERATOR/LDM_CA/best_model.pth"
ldm_config_file = "/media/vf19/BigCrumb/LESION_GENERATOR/LESION_GENERATOR/LDM_CA/config_file.yaml"
result_save_dir = "/media/vf19/BigCrumb/BRAINSPADE_DS/JOURNALPAPER_2D_SYNTHETIC_DS/WMH-TUM-EDE-NL/train"
n_samples = 10
batch_size = 3
use_conditioning = True
if use_conditioning:
    # Define scheduling. Define all conditioning combinations you want (if you want more weight, place the
    # same token multiple times). You can retrieve the conditioning list in conditioning_lists.py.
    conditioning_list = even_conditioning_list_noslice()
plot_raw_labels = True
monai_aug = None
save_volumes = False

#### AUTOMATIC INPUT
# monai.transforms.RandAffine(scale_range=(0.2, 0.2, 0.2), rotate_range=(0.75, 0.75, 0.75),
#                             translate_range=(20, 20, 20), mode='bilinear',
#                             as_tensor_output=False, prob=1.0, padding_mode='border')
for_loop_samples = np.ceil(n_samples / batch_size)

# For stage 2
brainspade_checkpoint = "/home/vf19/Documents/brainSPADE_2D/brainSPADE_1DEC"
brainspade_name = "BRAINSPADEV3_23"
target_datasets = ["ADNI60"]
volumetric_sweep = False # All the slices are saved and used to generate images.
n_labels_per_volume = 15  # Only this number of slices (randomly selected) is saved as 2D and used to generate images.
path_to_styles = "/home/vf19/Documents/brainSPADE_2D/DATA/MICCAI22/ADNI_REORG_TR/styles_for_SPADE"
appendix_styles = "style_ADNI60" # If the folders are named other than "style"
path_to_styles_labels = "/home/vf19/Documents/brainSPADE_2D/DATA/MICCAI22/ADNI_REORG_TR/styles_for_SPADE/style_ADNI60_mask"
spade_channels = {'healthy': 6, 'total': 12, 'lesions': 3, 'empty': 3}
lesions = ['wmh', 'tumour', 'edema'] # Which lesions are part of this.
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
if save_volumes:
    if not os.path.isdir(os.path.join(result_save_dir, 'volume_labels')):
        os.makedirs(os.path.join(result_save_dir, 'volume_labels'))

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

#####################################################################################################
#######################################STAGE 1#######################################################
#####################################################################################################
# We need to create the models_ and load their state dict
config_stage_1 = OmegaConf.load(vq_config_file)
vqvae = define_VAE3D(vq_config_file)
weights = torch.load(vq_model_path)
vqvae.load_state_dict(weights)
vqvae.to(device)


if use_conditioning: ##### UC needs to be 3d
    ldm = define_DDPM_conditioned(ldm_config_file) ##### UC needs to be 3d
else:
    ldm = define_DDPM_unconditioned(ldm_config_file) ##### UC needs to be 3D

weights = torch.load(ldm_model_path) # load weights
ldm.load_state_dict(weights) # load weights INTO model
ldm.eval().to(device)
ddim_sampler = DDIMSampler(ldm) # necessary for a fast sampling that skips time points
ldm.log_every_t = 15 # leave default
num_timesteps = 50  # number of time-steps used for the ddim sampling (< 1000)
plot_every = 15 # plot examples

if plot_raw_labels:
    if not os.path.isdir(os.path.join(result_save_dir, 'examples_labels_raw')):
        os.makedirs(os.path.join(result_save_dir, 'examples_labels_raw'))

# Label sampling process ----------------------------------------------------------------------------------
if stage_1 and not rendered_labels:
    processed = 0
    if use_conditioning:
        counter_condlist = 0
    while processed < n_samples and not rendered_labels:
        # Sample label and decode
        if n_samples - processed - batch_size < 0:
            b_size = n_samples - processed # Smaller batch size because you've reached the maximum number of samples.
        else:
            b_size = batch_size
        sample_shape = (b_size, config_stage_1['stage1']['params']['hparams']['z_channels'], 48, 64) # Sample shape
        if use_conditioning:
            # Format the conditioning list.
            cond_list_sub = []
            for b in range(b_size):
                cond_list_sub.append(conditioning_list[counter_condlist])
                counter_condlist += 1
                if counter_condlist == len(conditioning_list):
                    counter_condlist = 0

            cond = torch.FloatTensor(cond_list_sub).cuda()  # Convert the list to tensors.
            with torch.no_grad():
                if ldm.conditioning_key == "concat":
                    cond = cond.unsqueeze(-1).unsqueeze(-1)
                    cond = cond.expand(list(cond.shape[0:2]) + list(sample_shape[2:])).float()
                elif ldm.conditioning_key == "crossattn":
                    cond = cond.unsqueeze(1).float()
                elif ldm.conditioning_key == "hybrid":
                    cond_crossatten = cond.unsqueeze(1)
                    cond_concat = cond.unsqueeze(-1).unsqueeze(-1)
                    cond_concat = cond_concat.expand(list(cond.shape[0:2]) + list(sample_shape[2:]))
                    cond = {'c_concat': [cond_concat.float()], 'c_crossattn': [cond_crossatten.float()]}

            # Sample latent space
            latent_vectors, _ = ddim_sampler.sample(num_timesteps,conditioning=cond, batch_size=b_size,
                                                    shape=sample_shape[1:], eta=1.0)

        else:
            latent_vectors, intermediates = ddim_sampler.sample(num_timesteps,
                                                                batch_size=sample_shape[0],
                                                                shape=sample_shape[1:],
                                                                eta=1.0)
        with torch.no_grad():
            x_hat = vqvae.reconstruct_ldm_outputs(latent_vectors)
            x_hat = torch.softmax(x_hat, 1)

        # Save label
        number_subjects = n_samples + 1000
        for b in range(x_hat.shape[0]):
            # Isolate single subject.
            volume_label = x_hat[b, ...].detach().cpu()
            if monai_aug is not None:
                volume_label = monai_aug(volume_label)
            volume_label = torch.clamp(volume_label, 0, 1.0) # We clip between 0 and 1

            # Select slices or sweep
            if volumetric_sweep:
                slices = np.arange(0, volume_label.shape[-1], 1)
            else:
                slices = np.random.choice(range(volume_label.shape[-1]), n_labels_per_volume, replace = False)

            for s_ind in slices:

            # Cases: no lesions should be present or single lesion present because LDM has a single fused "lesion" channel
            # with no preference for any disease, that you still need to place correctly for SPADE>
            if len(lesions) == 0:
                # No lesions necessary (AKA: you don't care about the lesion channels)
                additional_channels = torch.zeros([spade_channels['total'] - out_label_tmp.shape[0], ]
                                                  + list(out_label_tmp.shape[1:]))  # If number of channels lesion > 2 changes in SPADE, modify this
                out_label = torch.cat([out_label_tmp, additional_channels], 0)

            elif len(lesions) == 1 and out_label_tmp.shape[0] == (spade_channels['healthy']+1):
                # Your LDM outputs a single lesion (you still need to complete other channels).
                additional_channels = torch.zeros([spade_channels['lesions'],]
                                                  + list(out_label_tmp.shape[1:])) # If number of channels lesion > 2 changes in SPADE, modify this
                out_label = torch.cat([out_label_tmp[:spade_channels['healthy'], ...], additional_channels], 0)
                if use_conditioning:
                    # If len(lesions) = 1, you cannot have multiple lesions at the same time...
                    lesion_type = lesion_tokens[np.argmax(np.asarray(cond_list_sub[b]))]
                else:
                    lesion_type = lesions[0]
                out_label[channels[lesion_type],...] = out_label_tmp[6, ...] # Assign single channel to lesions type
            else:
                additional_channels = torch.zeros([spade_channels['empty'], ] + list(out_label_tmp.shape[1:]))
                out_label = torch.cat([out_label_tmp, additional_channels], 0)
            # Convert and post-process
            out_label = out_label.numpy()
            out_label_argmax = np.argmax(out_label, 0)

            # Plot labels.
            if plot_raw_labels and processed%plot_every == 0:
                f, (a0, a1) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, out_label.shape[0]]},
                                           figsize=(out_label.shape[0] * 3, 3))
                a0p = a0.imshow(out_label_argmax)
                out = [out_label[i, ...] for i in range(out_label.shape[0])]
                out = np.concatenate(out, -1)
                a1p = a1.imshow(out, vmin=0.0, vmax=1.0)
                f.colorbar(a1p, ax=a1, fraction=0.006)
                if use_conditioning:
                    title = ""
                    for c_ind, c_item in enumerate(cond_list_sub[b]):
                        if c_item==1:
                            title += " %s, " %lesion_tokens[c_ind+1]
                    if title == "":
                        title = "no lesion"

                plt.title(title)
                plt.savefig(os.path.join(result_save_dir, 'examples_labels_raw', "sample_%d.png" %processed))
                plt.close(f)

            # One-hot encoded (not PV map label)
            out_label_ohe = np.zeros_like(out_label)
            for ch in range(out_label.shape[0]):
                out_label_ohe[ch, ...] = out_label_argmax == ch

            out_label_npy_pv = deepcopy(out_label)
            out_label = np.expand_dims(out_label, -1) # From CxHxW > CxHxWx1
            out_label = np.transpose(out_label, [1, 2, 3, 0]) # From CxHxWx1 > HxWx1xC

            # Save img.
            subject_id = str(processed+1000)
            append = "0"*(len(str(number_subjects))-len(str(processed)))
            out_name = "Parcellation_SYNTHETIC_sub-%s_ses-v0_0.npz" %(append+subject_id)
            if not os.path.isdir(os.path.join(result_save_dir, 'labels_NPY')):
                os.makedirs(os.path.join(result_save_dir, 'labels_NPY'))
            if not os.path.isdir(os.path.join(result_save_dir, 'labels_pv_NPY')):
                os.makedirs(os.path.join(result_save_dir, 'labels_pv_NPY'))
            np.save(os.path.join(result_save_dir, 'labels_NPY', out_name.replace("npz", "npy")),
                    out_label_ohe)
            np.save(os.path.join(result_save_dir, 'labels_pv_NPY', out_name.replace("npz", "npy")),
                    out_label_npy_pv)
            np.savez(os.path.join(result_save_dir, 'labels', out_name),
                     label = out_label,
                     slice = 0,
                     dataset_name = 'SYNTHETIC')

            processed += 1

    with open(os.path.join(result_save_dir, 'labels_rendered.txt'), 'w') as f:
        f.write("LABS")
        f.close()

vqvae.cpu()
ldm.cpu()

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
            f.write("VQ-VAE model: %s\n" %vq_model_path)
            f.write("LDM model: %s\n" %ldm_model_path)
            f.write("Conditioning:%s\n" %use_conditioning)
            if use_conditioning:
                f.write("Conditioning ratios: %s" %str(conditioning_list))
            f.write("Brainspade model: %s\n" %os.path.join(brainspade_checkpoint, brainspade_name))
            f.write("Number of passes per modality: %d\n" %n_passes)
            f.close()