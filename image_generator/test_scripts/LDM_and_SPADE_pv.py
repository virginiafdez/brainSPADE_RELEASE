'''
LABEL AND IMAGE GENERATOR WITH PARTIAL VOLUME MAPS LABELS
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
from models.VQVAE_DDPM.import_utils import define_VQVAE, define_DDPM_conditioned, define_DDPM_unconditioned, define_VAE
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
    nolesion_conditioning_list_noslice, even_conditioning_list_noslice, check_conditioning

# CONTROL VARIABLES
channels = {'generic': 6, 'wmh': 6, 'tumour': 7, 'edema': 8}
lesion_tokens = {1: 'wmh', 2: 'tumour', 3: 'edema'}
n_healthy_labels = 6

# PARAMETERS
############################################### INPUT SETTINGS
stage_1 = False # Whether to perform label generation
stage_2 = True # Whether to perform style image generation

# For stage 1
vq_model_path = "/media/vf19/BigCrumb/LESION_GENERATOR/LESION_GENERATOR/VAE_TUMOURS_BIG_L1/final_model.pth"
vq_config_file = "/media/vf19/BigCrumb/LESION_GENERATOR/LESION_GENERATOR/VAE_TUMOURS_BIG_L1/ae_kl_newdataset_pv.yaml"
ldm_model_path = "/media/vf19/BigCrumb/LESION_GENERATOR/LESION_GENERATOR/LDM_TUMOURS_CH320_1_L1/best_model.pth"
ldm_config_file = "/media/vf19/BigCrumb/LESION_GENERATOR/LESION_GENERATOR/LDM_TUMOURS_CH320_1_L1/ldm_CH320_1_L1.yaml"
result_save_dir = "/media/vf19/BigCrumb/BRAINSPADE_DS/MICCAI22/WORKSHOPS/STRAIGHT_CREATED_DATASETS/ABIDE_STYLED_SYNTHETIC/validation"
n_samples = 20000
batch_size = 64
type_stage_1 = "VAE" # VQ-VAE or VAE
save_as_npy = False # Whether to, in addition to saving as NPZ (for SPADE), save as NPY
for_loop_samples = np.ceil(n_samples / batch_size)
use_slice_cond = True # USE SLICE CONDITIONING
use_lesion_cond = True # USE LESION CONDITIONING
random_slice_conditioning = True
plot_raw_labels = True
monai_aug = None
SOFTMAX = False # Whether the results need to be softmaxed or not.
num_timesteps = 80
correct = True
lesions = ['tumour', 'edema']#['wmh', 'tumour', 'edema'] # Which lesions are part of this.
if random_slice_conditioning and use_lesion_cond:
    # Define scheduling. Define all conditioning combinations you want (if you want more weight, place the
    # same token multiple times). You can retrieve the conditioning list in conditioning_lists.py.
    if use_slice_cond :
        # conditioning_list = wmh_conditioning_list(n_slices_per_disease_comb=6, n_slices_per_interval=1)
        # conditioning_list = tumour_conditioning_list(n_slices_per_disease_comb=6, n_slices_per_interval=1)
        conditioning_list = nolesion_conditioning_list(n_slices_per_disease_comb=6, n_slices_per_interval=1)
    else:
        conditioning_list = even_conditioning_list_noslice()

# monai.transforms.RandAffine(scale_range=(0.2, 0.2, 0.2), rotate_range=(0.75, 0.75, 0.75),
#                             translate_range=(20, 20, 20), mode='bilinear',
#                             as_tensor_output=False, prob=1.0, padding_mode='border')

if not os.path.isdir(result_save_dir):
    os.makedirs(result_save_dir)
# For stage 2

brainspade_checkpoint = "/home/vf19/Documents/brainSPADE_2D/brainSPADE_1DEC"
override_label_dir = "/media/vf19/BigCrumb/BRAINSPADE_DS/MICCAI22/WORKSHOPS/STRAIGHT_CREATED_DATASETS/SABRE_STYLED_SYNTHETIC/validation"
brainspade_name = "BRAINSPADEV3_23"
target_datasets = ["ABIDE-CALTECH"]
modalities = ['T1',]
path_to_styles = "/media/vf19/BigCrumb/BRAINSPADE_DS/MICCAI22/WORKSHOPS/STRAIGHT_CREATED_DATASETS/STYLES/ABIDE-CALTECH"
appendix_styles = "styles" # If the folders are named other than "style"
path_to_styles_labels = "/media/vf19/BigCrumb/BRAINSPADE_DS/MICCAI22/WORKSHOPS/STRAIGHT_CREATED_DATASETS/STYLES/ABIDE-CALTECH/styles_mask"
spade_channels = {'healthy': 6, 'total': 12, 'lesions': 3, 'empty': 3}
n_passes = 1 # Number of passes through the same dataset of labels.
format = 'npy' # Either "spade", "png" or "npy". Spade is a npz with keys: img, img_affine, img_header and modality.
save_as_npy = True # Whether to, in addition to saving as NPZ (for SPADE), save as NPY

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
    if r.strip("\n") == "NOLABS":
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

#####################################################################################################
#######################################STAGE 1#######################################################
#####################################################################################################
# We need to create the models_ and load their state dict
config_stage_1 = OmegaConf.load(vq_config_file)
if type_stage_1 == "VAE":
    vqvae = define_VAE(vq_config_file)
else:
    vqvae = define_VQVAE(vq_config_file)
weights = torch.load(vq_model_path)
vqvae.load_state_dict(weights)
vqvae.to(device)

if use_slice_cond or use_lesion_cond:
    ldm = define_DDPM_conditioned(ldm_config_file)
else:
    ldm = define_DDPM_unconditioned(ldm_config_file)
weights = torch.load(ldm_model_path)
ldm.load_state_dict(weights)
ldm.eval().to(device)
ddim_sampler = DDIMSampler(ldm)

ldm.log_every_t = 15
plot_every = 15
if plot_raw_labels:
    if not os.path.isdir(os.path.join(result_save_dir, 'examples_labels_raw')):
        os.makedirs(os.path.join(result_save_dir, 'examples_labels_raw'))

# Label sampling process ----------------------------------------------------------------------------------
if stage_1 and not rendered_labels:
    processed = 0
    if os.path.isdir(os.path.join(result_save_dir, 'labels')):
        processed += len(os.listdir(os.path.join(result_save_dir, 'labels')))
    if use_lesion_cond or (use_slice_cond and not random_slice_conditioning):
        counter_condlist = 0
    while processed < n_samples and not rendered_labels:
        # Sample label and decode
        if n_samples - processed - batch_size < 0:
            b_size = n_samples - processed # Smaller batch size because you've reached the maximum number of samples.
        else:
            b_size = batch_size
        sample_shape = (b_size, config_stage_1['stage1']['params']['hparams']['z_channels'], 48, 64) # Sample shape

        if use_lesion_cond or (use_slice_cond and not random_slice_conditioning):
            cond_list = []
            for lesion_token, lesion_name in lesion_tokens.items():
                if lesion_name in lesions:
                    cond_list.append(np.array([[i] for i in list(np.random.choice([0, 1], size=b_size,))]))
            if use_slice_cond:
                slice_list = list(np.random.uniform(0.2, 0.50, size=b_size)) # Slice
                slice_list = [np.round(i, 8) for i in slice_list]
                slice_list = np.array([[i] for i in slice_list])
                cond_list.append(slice_list)
            cond_list = np.stack(cond_list, -1).squeeze()
            cond = torch.FloatTensor(cond_list)  # Convert the list to tensors.
            cond = cond.to(device)

            # # Format the conditioning list.
            # cond_list_sub = []
            # for b in range(b_size):
            #     cond_list_sub.append(conditioning_list[counter_condlist])
            #     counter_condlist += 1
            #     if counter_condlist == len(conditioning_list):
            #         counter_condlist = 0
            # if use_slice_cond and random_slice_conditioning:
            #     for cond_list_sub_sub in cond_list_sub:
            #         cond_list_sub_sub[-1] = np.random.uniform(0.15, 0.58)
            # cond = torch.FloatTensor(cond_list_sub).cuda()  # Convert the list to tensors.

        elif use_slice_cond and random_slice_conditioning:
                cond_list_sub = list(np.random.uniform(0.15, 0.58, size=b_size))
                cond_list_sub = [[i] for i in cond_list_sub]
                cond = torch.FloatTensor(cond_list_sub).cuda()  # Convert the list to tensors.

        if use_slice_cond or use_lesion_cond:
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
            with torch.no_grad():
                x_hat = vqvae.reconstruct_ldm_outputs(latent_vectors)
                if SOFTMAX:
                    x_hat = torch.softmax(x_hat, 1)

        else:
            latent_vectors, intermediates = ddim_sampler.sample(num_timesteps,
                                                                batch_size=sample_shape[0],
                                                                shape=sample_shape[1:],
                                                                eta=1.0)
            with torch.no_grad():
                x_hat = vqvae.reconstruct_ldm_outputs(latent_vectors)
                if SOFTMAX:
                    x_hat = torch.softmax(x_hat, 1)

        # Save label
        number_subjects = 100000
        for b in range(x_hat.shape[0]):
            out_label_tmp = x_hat[b, ...].detach().cpu()
            if monai_aug is not None:
                out_label_tmp = monai_aug(out_label)
            if correct:
                out_label_tmp[0, torch.amax(out_label_tmp[1:, ...], 0) < 0.075] = 1.0
            out_label_tmp = torch.clamp(out_label_tmp, 0, 1.0) # We clip between 0 and 1

            # Cases: no lesions should be present or single lesion present because LDM has a single fused "lesion" channel
            # with no preference for any disease, that you still need to place correctly for SPADE>
            if len(lesions) == 0:
                # No lesions necessary (AKA: you don't care about the lesion channels)
                additional_channels = torch.zeros([spade_channels['total'] - n_healthy_labels ]
                                                  + list(out_label_tmp.shape[1:]))  # If number of channels lesion > 2 changes in SPADE, modify this
                out_label = torch.cat([out_label_tmp[:n_healthy_labels,...], additional_channels], 0)

            elif len(lesions) == 1 and out_label_tmp.shape[0] == (spade_channels['healthy']+1):
                # Your LDM outputs a single lesion (you still need to complete other channels).
                additional_channels = torch.zeros([spade_channels['lesions'],]
                                                  + list(out_label_tmp.shape[1:])) # If number of channels lesion > 2 changes in SPADE, modify this
                out_label = torch.cat([out_label_tmp[:spade_channels['healthy'], ...], additional_channels], 0)
                if use_lesion_cond:
                    # If len(lesions) = 1, you cannot have multiple lesions at the same time...
                    if not use_slice_cond:
                        lesion_type = lesion_tokens[np.argmax(np.asarray(cond_list_sub[b]))]
                    else:
                        lesion_type = lesion_tokens[np.argmax(np.asarray(cond_list_sub[b][:-1]))]
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
            cond = cond.squeeze()
            if plot_raw_labels and processed%plot_every == 0:
                f, (a0, a1) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, out_label.shape[0]]},
                                           figsize=((out_label.shape[0]-spade_channels['empty']) * 6, 8))
                a0p = a0.imshow(out_label_argmax)
                out = [out_label[i, ...] for i in range(out_label.shape[0] - spade_channels['empty'])]
                out = np.concatenate(out, -1)
                a1p = a1.imshow(out, vmin=0.0, vmax=1.0)
                f.colorbar(a1p, ax=a1, fraction=0.006)
                plt.axis('off')
                title_ = ""
                if use_lesion_cond:
                    # We include the lesion type in the title
                    title_ = ""
                    for c_ind, c_item in enumerate(cond[b][:-1]):
                        if c_item==1:
                            title_ += " %s, " %lesion_tokens[c_ind+1]
                    if title_ == "":
                        title_ = "no lesion"
                if use_slice_cond:
                    title_ += ", %d" %(int(cond[b][-1]*256))
                plt.title(title_)
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
            if use_slice_cond:
                slice_nbr = str(int(cond[b][-1]*256))
            else:
                slice_nbr = '0'
            out_name = "Parcellation_SYNTHETIC_sub-%s_ses-v0_%s.npz" %(append+subject_id, slice_nbr)
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
    for mod in modalities:
        print("Processing modality %s" %mod)
        # We modify the settings for the dataset created for each modality
        opt.image_dir =  os.path.join(path_to_styles, "%s_%s" %(appendix_styles,mod))
        opt.fix_seq = mod
        opt.nThreads = 0
        if not stage_1 and override_label_dir is not None:
            opt.label_dir = os.path.join(override_label_dir, 'labels')
        else:
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
        opt.cache_dir = os.path.join(result_save_dir, 'spadenai_cache_dir')
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
                        f = plt.figure(figsize=(15, 6))
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
        dataset_container.clearCache()
        with open(os.path.join(result_save_dir, 'log.txt'), 'w') as f:
            f.write("%s\n" %str(datetime.today()))
            f.write("VQ-VAE model: %s\n" %vq_model_path)
            f.write("LDM model: %s\n" %ldm_model_path)
            f.write("Lesion conditioning:%s\n" %use_lesion_cond)
            f.write("Slice conditioning:%s\n" %use_slice_cond)
            if use_slice_cond:
                f.write("Random slice conditioning:%s\n" %random_slice_conditioning)
            f.write("Brainspade model: %s\n" %os.path.join(brainspade_checkpoint, brainspade_name))
            f.write("Number of passes per modality: %d\n" %n_passes)
            f.close()