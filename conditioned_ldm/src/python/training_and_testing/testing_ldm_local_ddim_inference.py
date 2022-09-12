'''
Runs multiple inferences on a multi-channel LDM and saves the results in output_dir.
Saves:
- Figures containing the argmaxed channels and individual channels
- Samples

If a true samples directory is provided (true_samples_dir), performs a nearest-neighbour pairing
between true and generated samples, and performs a one-sample t-test on the mean square errors between
nearest neighbour pairs for each of the tissue, attempting to see if there is a statistical significance
between the two distributions.

'''
import matplotlib.pyplot as plt
import mlflow.pytorch
import numpy as np
import torch
from condldm_models.ddim import DDIMSampler
import os
from scipy import stats
import monai

# Load model -----------------------------------------------------------------------------------------------------------
# Add here the path to your local trained final condldm_models
final_stage1_model_path = "[PATH TO VAE MLRUNS folder]/artifacts/final_model"
final_ldm_model_path = "[PATH TO LDM MLRUNS folder]/MLRUNS/artifacts/final_model"
output_dir = "" # Where to save
true_samples_dir = "" # Path to ground truths folder.

device = torch.device("cuda")
vqvae = mlflow.pytorch.load_model(final_stage1_model_path)
vqvae = vqvae.to(device)
vqvae.eval()

diffusion = mlflow.pytorch.load_model(final_ldm_model_path)
diffusion = diffusion.to(device)
diffusion.eval()

# Create output directory if not existing
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)
if not os.path.isdir(os.path.join(output_dir, 'figures')):
    os.makedirs(os.path.join(output_dir, 'figures'))
if not os.path.isdir(os.path.join(output_dir, 'samples')):
    os.makedirs(os.path.join(output_dir, 'samples'))

# Sampling process -----------------------------------------------------------------------------------------------------
# Add here the sample shape
n_samples = 300
batch_size = 20
sample_shape = (batch_size, 3, 48, 64)


if len(os.listdir(os.path.join(output_dir, 'samples'))) < n_samples:
    # Change here the frequency of intermediates stored
    counter = 0
    if len(os.listdir(os.path.join(output_dir, 'samples'))) > 0:
        all_f = [int(i.split("_")[-1].strip(".npy")) for i in os.listdir(os.path.join(output_dir, 'samples'))]
        counter = max(all_f)
    for sample_id in range(int(n_samples / batch_size)+1):
        if len(os.listdir(os.path.join(output_dir, 'samples'))) >= n_samples:
            break
        ddim = DDIMSampler(diffusion)
        bs = sample_shape[0]
        shape = sample_shape[1:]
        diffusion.log_every_t = 10
        num_timesteps = 50
        latent_vectors, intermediates = ddim.sample(num_timesteps, batch_size=bs, shape=shape, eta=1.0,
                                                    )
        with torch.no_grad():
            x_hat = vqvae.reconstruct_ldm_outputs(latent_vectors)
        for b in range(x_hat.shape[0]):
            # Figure
            f, (a0, a1) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, x_hat.shape[1]]}, figsize = (x_hat.shape[1]*3, 3))
            a0p = a0.imshow(torch.argmax(x_hat[b, ...], 0).detach().cpu())
            out = [x_hat[b, i, ...].detach().cpu() for i in range(x_hat.shape[1])]
            out = torch.cat(out, -1)
            a1p = a1.imshow(out, vmin = 0.0, vmax = 1.0)
            f.colorbar(a1p, ax = a1, fraction = 0.006)
            plt.savefig(os.path.join(output_dir, 'figures', 'sample_%d.png' %(int(n_samples / batch_size)*sample_id+b)))
            plt.close(f)

            # Sample
            out_sample = np.clip(x_hat[b, ...].detach().cpu().numpy(), 0, 1)
            np.save(os.path.join(output_dir, 'samples', 'sample_%d.npy' %(counter)),
                    out_sample)
            counter += 1
if true_samples_dir != "":
    #############################################################################
    # Nearest neighbour study
    #############################################################################
    #Auxiliary functions
    def to_OHE(label, n_labels):
        label_out = np.zeros([label.shape[0]]+[n_labels]+list(label.shape)[1:])
        for ch in range(n_labels):
            label_out[:, ch, ...] = label == ch
        return label_out

    def ReduceScoreChannel(scores, targets, mean = "channel"):
        '''
        Reduces dice score output (BxC) avoiding counting in NANS if NANS are due to
        absent labels in ground truth.
        scores: BxC tensor containing dice scores

        '''
        if mean == 'channel':
            scores_tot = torch.zeros(scores.shape[1])
            class_count = torch.zeros(scores.shape[1])
            for i in range(scores.shape[0]):
                for class_ in range(scores.shape[1]):
                    if (targets[i, class_, :, :] > 0.4).sum() > 0:
                        scores_tot[class_] += scores[i, class_]
                        class_count[class_] += 1

            scores_tot /=  class_count
            #scores_tot[np.isnan(scores_tot)] = 0.0
        elif mean == 'batch':
            scores_tot = torch.zeros(scores.shape[0])
            for subj_ in range(scores.shape[0]):
                pos_classes = 0
                for i in range(scores.shape[1]):
                    if (targets[subj_, i, :, :] > 0.4).sum() > 20:
                        if np.isnan(scores[subj_, i]):
                            scores[subj_, i] =0.0
                        scores_tot[subj_] += scores[subj_, i]
                        pos_classes +=1
                scores_tot[subj_] /= pos_classes
        else:
            ValueError("Unrecognized mean: must be 'channel' or 'batch'")
        return scores_tot


    if not os.path.isdir(os.path.join(output_dir, 'nn_stats')):
        os.makedirs(os.path.join(output_dir, 'nn_stats'))

    all_gt_images = []
    all_syn_images = []
    pairs = {}
    distributions = {'healthy': {}, 'synth': {}}
    tissue_names = {0: 'background', 1: 'csf', 2: 'grey matter',
                    3: 'white matter', 4: 'basal ganglia', 5: 'brainstem',
                    6:"tumour"}

    for i in os.listdir(true_samples_dir):
        load_img = np.load(os.path.join(true_samples_dir, i))
        pad_h = int((256 - load_img.shape[1]) / 2)
        pad_w = int((256 - load_img.shape[2]) / 2)
        out_img = []
        for ch in range(load_img.shape[0]):
            if ch == 0:
                out_img.append(np.pad(load_img[ch, ...], ((pad_h,), (pad_w,)), constant_values=1.0))
            else:
                out_img.append(np.pad(load_img[ch, ...], ((pad_h,), (pad_w,)), constant_values=0.0))

        out_img = np.stack(out_img, 0)
        all_gt_images.append(out_img)

    for i in os.listdir(os.path.join(output_dir, 'samples')):
        load_img = np.load(os.path.join(output_dir, 'samples', i))
        pad_h = int((256 - load_img.shape[1]) / 2)
        pad_w = int((256 - load_img.shape[2]) / 2)
        out_img = []
        for ch in range(load_img.shape[0]):
            if ch == 0:
                out_img.append(np.pad(load_img[ch, ...], ((pad_h,), (pad_w,)), constant_values=1.0))
            else:
                out_img.append(np.pad(load_img[ch, ...], ((pad_h,), (pad_w,)), constant_values=0.0))
        out_img = np.stack(out_img, 0)
        all_syn_images.append(out_img)

    all_gt_images = np.stack(all_gt_images, 0)
    all_syn_images = np.stack(all_syn_images, 0)
    all_gt_images_argmax = np.argmax(all_gt_images, 1)
    all_syn_images_argmax = np.argmax(all_syn_images, 1)
    all_gt_images_argmax_ohe = torch.from_numpy(to_OHE(all_gt_images_argmax, all_gt_images.shape[1]))

    dice = monai.metrics.DiceMetric(include_background=False)


    for i in range(all_syn_images_argmax.shape[0]):

        #' Hard dice'
        im_i = all_syn_images_argmax[i, ...]
        im_i_r = np.expand_dims(im_i, 0)
        im_i_r = np.repeat(im_i_r, all_gt_images_argmax.shape[0], axis = 0)
        # To calculate the Dice, we need to convert gt_i and gt_
        im_i_r_ohe = to_OHE(im_i_r, all_syn_images.shape[1])

        # Find nearest neighbour based on argmax
        dice_i = dice(torch.from_numpy(im_i_r_ohe), all_gt_images_argmax_ohe)

        dice_i = ReduceScoreChannel(dice_i, torch.from_numpy(all_gt_images), mean = "batch")
        nearest_ind = torch.argmax(dice_i).numpy()

        # Plot NN and synthetic sample
        f, ((a0, a1), (a2, a3)) = plt.subplots(2, 2, gridspec_kw={'width_ratios': [1, 6],
                                                              'height_ratios': [1, 1]},
                                   figsize=(7*3, 6))
        # Plot grund truth
        a0p = a0.imshow(all_gt_images_argmax[nearest_ind,...])
        out = [all_gt_images[nearest_ind,ts,...] for ts in range(all_gt_images.shape[1])]
        out = np.concatenate(out, -1)
        a1p = a1.imshow(out, vmin=0.0, vmax=1.0)
        f.colorbar(a1p, ax=a1, fraction=0.006)
            # Plot synthetic images
        a2p = a2.imshow(all_syn_images_argmax[i, ...])
        out = [all_syn_images[i,ts,...] for ts in range(all_syn_images.shape[1])]
        out = np.concatenate(out, -1)
        a3p = a3.imshow(out, vmin=0.0, vmax=1.0)
        f.colorbar(a3p, ax=a3, fraction=0.006)
        plt.suptitle("Ground truth (top) and synthetic sample (bottom) paired up.")
        plt.savefig(os.path.join(output_dir, 'nn_stats', 'nn_sample_%d.png' %i))

        plt.close(f)

        all_pixels_brain_gt = (all_gt_images_argmax[nearest_ind,...] > 0.0).sum()
        all_pixels_brain_syn = (all_syn_images_argmax[i,...] > 0.0).sum()
        for tissue in range(all_syn_images.shape[1]):
            name_tissue = tissue_names[tissue]
            if name_tissue not in distributions['healthy'].keys():
                distributions['healthy'][name_tissue] = []
            if name_tissue not in distributions['synth'].keys():
                distributions['synth'][name_tissue] = []
            distributions['healthy'][tissue_names[tissue]].append(np.mean(all_gt_images[nearest_ind, tissue, all_gt_images_argmax[nearest_ind,...] > 0.0]) /all_pixels_brain_gt)
            distributions['synth'][tissue_names[tissue]].append(np.mean(all_syn_images[i, tissue, all_syn_images_argmax[i,...] > 0.0])/ all_pixels_brain_syn)



    # Now we run a statistical test comparing mses to 0
    with open(os.path.join(output_dir, 'nn_stats', 'statistical_normal_tests.txt'), 'w') as f:
        f.write("Tissue\tP-VALUE\tt\tEquality between distributions\n")
        f.close()

    for tissue, distrib_syn in distributions['synth'].items():
        f = plt.figure()
        plt.boxplot([distrib_syn, distributions['healthy'][tissue]])
        plt.title("Boxplot for %s ratio / total brain " %tissue)
        plt.savefig(os.path.join(output_dir, 'nn_stats', "boxplot_%s.png" %tissue))
        t, pval = stats.ttest_ind(np.asarray(distrib_syn), np.asarray(distributions['healthy'][tissue]))

        if pval/2 < 0.025:
            eq = "No"
        else:
            eq = "Yes"

        with open(os.path.join(output_dir, 'nn_stats', 'statistical_normal_tests.txt'), 'a') as f:
            f.write("%s\t%.3f\t%.3f\t%s\n" %(tissue, pval, t, eq))

