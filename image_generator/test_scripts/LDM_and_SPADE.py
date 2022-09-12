'''
LABEL AND IMAGE GENERATOR WITH SEMANTIC LABELS
For earlier LDM condldm_models that do not produce PV maps, but Semantic Labels.
The semantic labels are "converted" to PV maps via a process of one-hot-encoding and blurring.
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

# CONTROL VARIABLES
channels = {'generic': 6, 'wmh': 6, 'tumour': 7}
lesion_tokens = {1: 'wmh', 2: 'tumour'}
n_healthy_labels = 5
save_as_npy = True

############################################### INPUT SETTINGS
stage_1 = True
stage_2 = True

# For stage 1
vq_model_path = "/home/vf19/Documents/brainSPADE_2D/LESION_GENERATOR/LDM/VAE_LDM_HEALTHY_UNCONDITIONED/vae_best_model.pth"
vq_config_file = "/home/vf19/Documents/brainSPADE_2D/LESION_GENERATOR/MLRUNS/configs/ae_kl_v0.yaml"
ldm_model_path = "/home/vf19/Documents/brainSPADE_2D/LESION_GENERATOR/LDM/VAE_LDM_HEALTHY_UNCONDITIONED/ldm_best_model.pth"
ldm_config_file = "/home/vf19/Documents/brainSPADE_2D/LESION_GENERATOR/MLRUNS/configs/ldm_v0.yaml"
result_save_dir = "/home/vf19/Documents/brainSPADE_2D/DATA/TUMOUR_SHIFT_EXPERIMENT/ADNI_SYNTHETIC"
n_samples = 1000
batch_size = 8
type_stage_1 = "VAE"
for_loop_samples = np.ceil(n_samples / batch_size)
use_conditioning = False
if use_conditioning:
    conditioning_values = {'none': 0, 'wmh': 1, 'tumour': 2}
    ratios = {'none': 0, 'wmh': 0.5, 'tumour': 0.5}
    conditioning_tokens = []
    for k in conditioning_values.keys():
        conditioning_tokens += int(n_samples*ratios[k])*[conditioning_values[k]]
multiple_labels = False # If lesions have different labels or a single one regardless of the type of lesion.
plot_raw_labels = True
# For stage 2
brainspade_checkpoint = "/home/vf19/Documents/brainSPADE_2D/brainSPADE_1DEC"
brainspade_name = "BRAINSPADEV3_20"
target_datasets = ["ADNI60"]
path_to_styles = "/home/vf19/Documents/brainSPADE_2D/DATA/TUMOUR_SHIFT_EXPERIMENT/ADNI_60/styles_for_SPADE"
appendix_styles = "style_ADNI60" # If the folders are named other than "style"
path_to_styles_labels = "/home/vf19/Documents/brainSPADE_2D/DATA/TUMOUR_SHIFT_EXPERIMENT/ADNI_60/styles_for_SPADE/style_ADNI60_mask"
lesions = [] # Which lesions are part of this.
n_passes = 3 # Number of passes through the same dataset of labels.
format = 'npy' # Either "spade", "png" or "npy". Spade is a npz with keys: img, img_affine, img_header and modality.

if len(lesions) > 1 or use_conditioning:
    multiple_lesions = True
else:
    multiple_lesions = False

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
config_stage_1 = OmegaConf.load(vq_config_file)
if type_stage_1 == "VAE":
    vqvae = define_VAE(vq_config_file)
else:
    vqvae = define_VQVAE(vq_config_file)
weights = torch.load(vq_model_path)
vqvae.load_state_dict(weights)
vqvae.to(device)

if use_conditioning:
    ldm = define_DDPM_conditioned(ldm_config_file)
    weights = torch.load(ldm_model_path)
    ldm.load_state_dict(weights)
    ldm.eval().to(device)
    ddim_sampler = DDIMSampler(ldm)
else:
    ldm = define_DDPM_unconditioned(ldm_config_file)
    weights = torch.load(ldm_model_path)
    ldm.load_state_dict(weights)
    ldm.eval().to(device)
    ddim_sampler = DDIMSampler(ldm)

ldm.log_every_t = 15
num_timesteps = 50
plot_every = 20
if plot_raw_labels:
    if not os.path.isdir(os.path.join(result_save_dir, 'examples_labels_raw')):
        os.makedirs(os.path.join(result_save_dir, 'examples_labels_raw'))

# Label sampling process ----------------------------------------------------------------------------------
if stage_1 and not rendered_labels:
    processed = 0
    while processed < n_samples and not rendered_labels:
        # Sample label and decode
        if n_samples - processed - batch_size < 0:
            b_size = n_samples - processed
        else:
            b_size = batch_size
        sample_shape = (b_size, config_stage_1['stage1']['params']['hparams']['z_channels'], 48, 64) # Sample shape
        if use_conditioning:
            if processed == for_loop_samples - 1:
                conditioning_tokens_batch = conditioning_tokens[batch_size * for_loop_samples:]
            else:
                conditioning_tokens_batch = conditioning_tokens[batch_size*for_loop_samples:
                                                                batch_size*(for_loop_samples + 1)]
            conditioning_tokens_batch = torch.tensor(conditioning_tokens_batch).unsqueeze(1).unsqueeze(1).unsqueeze(1)
            conditioning_tokens_batch = conditioning_tokens_batch.expand([conditioning_tokens_batch.shape[0], 1] +
                                                                         list(sample_shape.shape[2:])).to(device)


            latent_vectors, intermediates = ddim_sampler.sample(num_timesteps,
                                                                batch_size=sample_shape[0],
                                                                shape=sample_shape[1:],
                                                                eta=1.0)
            # latent_vectors = ldm.p_sample_loop([conditioning_tokens_batch], sample_shape,
            #                                    return_intermediates=False)
        else:
            latent_vectors, intermediates = ddim_sampler.sample(num_timesteps,
                                                                batch_size=sample_shape[0],
                                                                shape=sample_shape[1:],
                                                                eta=1.0)
            #latent_vectors = ldm.p_sample_loop(sample_shape, return_intermediates=False)
        with torch.no_grad():
            x_hat = vqvae.reconstruct_ldm_outputs(latent_vectors)

        # Save label
        number_subjects = n_samples
        for b in range(x_hat.shape[0]):
            if len(lesions) > 1:
                # Multiple lesion scenario
                if not multiple_labels:
                    if use_conditioning:
                        lesion_type = lesion_tokens[int(conditioning_tokens_batch[b].squeeze())]
                        out_label = uvir.round_and_cap(x_hat[b, 0, ...].detach().cpu(), n_healthy_labels + 1)
                        out_label[out_label == channels['generic']] = channels[lesion_type]
                    else:
                        ValueError("The lesion type cannot be identified and the lesion label"
                                   "for SPADE would nto be reliable.")
                else:
                    out_label = uvir.round_and_cap(x_hat[b,0,...].detach().cpu(), n_healthy_labels +
                                                   len(lesions))
            elif len(lesions) == 1:
                lesion_type = lesions[0]
                out_label = uvir.round_and_cap(x_hat[b, 0, ...].detach().cpu(), n_healthy_labels + 1)
                out_label[out_label == channels['generic']] = channels[lesion_type]
            else:
                out_label = uvir.round_and_cap(x_hat[b, 0, ...].detach().cpu(), n_healthy_labels)

            # Convert to One Hot encoding.
            out_label = uvir.to_one_hot(out_label.unsqueeze(-1).numpy(), 8, lesion_id = lesions,
                                        lesion_values=channels)

            # Make sure regions with very low standard deviation from the HEALTHY channels are
            # zeroed out.
            out_label = uvir.blurSTD(out_label, ignore_last=2)

            # Convert to SPADE order of channels.
            out_label = uvir.translateFromSPADE(out_label)

            if plot_raw_labels and processed%plot_every == 0:
                f = plt.figure()
                plt.subplot(1,2,1)
                plt.imshow(x_hat[b, 0, ...].detach().cpu().numpy())
                plt.title("Unprocessed image")
                plt.subplot(1,2,2)
                plt.imshow(np.argmax(out_label, -1)[..., 0])
                plt.title("Processed image")
                plt.savefig(os.path.join(result_save_dir, 'examples_labels_raw', "sample_%d.png" %processed))
                plt.close(f)

            # Save img.
            subject_id = str(processed)
            append = "0"*(len(str(number_subjects))-len(str(processed)))
            out_name = "Parcellation_SYNTHETIC_sub-%s_ses-v0_0.npz" %(append+subject_id)
            if not os.path.isdir(os.path.join(result_save_dir, 'labels_NPY')):
                os.makedirs(os.path.join(result_save_dir, 'labels_NPY'))
            np.save(os.path.join(result_save_dir, 'labels_NPY', out_name.replace("npz", "npy")),
                    out_label)
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
                out_cond = ""
                for key, val in ratios.items():
                    out_cond += "%s: %f" %(key, val)
                out_cond += "\n"
                f.write("Conditioning ratios: %s" %out_cond)
            f.write("Brainspade model: %s\n" %os.path.join(brainspade_checkpoint, brainspade_name))
            f.write("Number of passes per modality: %d\n" %n_passes)
            f.close()