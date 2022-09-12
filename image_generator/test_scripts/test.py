from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
import os
from util import util as ut
import torch
import moreutils as uvir
import numpy as np
import matplotlib.pyplot as plt
import util.util as util
from copy import deepcopy
import nibabel as nib
from models.modality_discrimination.modisc_v2 import Modisc
from data.spadenai_v2 import SpadeNai
from data.spadenai_v2_sliced import SpadeNaiSlice
from monai.data import DataLoader

uvir.set_deterministic(True, 1)

# Initialisation
opt = TestOptions().parse()
model = Pix2PixModel(opt)
modalities = opt.sequences
if not os.path.exists(opt.results_dir):
    os.makedirs(opt.results_dir)

# Global. Plot losses and validation results.
## All tests.
# 0: Global plots: losses and validation results
# 1: Generation of N samples of images (both styles) and uncertainty
# 2: Lesion: Generation of slices containing lesions and "wrong" lesions
# 3: Cross-style generation: extraction of modality from one slice to generate another slice
# 4: 3D Volume render: uses the test3D folder to save all the samples of the same 3D volume into a series
# of 2D images, then forward pass all these images and reconstruct the synthetic 3D model.
# 5: Histograms: Plot of histograms for real and fake images in both modalities.
# 6: Cross-style generation: generates a modality by taking as input style the image from one dataset,
# using the semantic map of another dataset. Valid only if you have more than 1 dataset.
# 7: Generation of images: Sample some images many times and plots the different instances to see if there is
# any variation.
# 8: Calculation of Maximum-Mean-Dtance between real and fake images distributions.
# 9: Sequence analysis. For a specific sequence, loads several labels and uses different styles from
# different slices and datasets to see the impact on the output.
# 10: Additional sequence analysis. Generates images using another style (sequence) embedded in opt.guest_mods.


test_flags = [False, False, False, False, True, False, False, False, False, False, False]  # Which test sections run

# Test 0: based on loss_log.txt and results_training.txt, plot training curves and validation results.
if test_flags[0]:
    uvir.plotTrainingLosses_1dec(os.path.join(opt.checkpoints_dir, opt.name, "loss_log.txt"), True, opt.results_dir)
    if os.path.isfile(os.path.join(opt.checkpoints_dir, opt.name, 'web', 'validation_results', 'results_training.txt')):
        f = open(os.path.join(opt.checkpoints_dir, opt.name, 'web', 'validation_results', 'results_training.txt'), 'r')
        lines = f.readlines()
        name_fig = lines[0].strip("\n").split("\t")
        metrics = {}
        metrics_ind = {}
        for ind, metric in enumerate(name_fig):
            metrics[metric] = []
            metrics_ind[ind] = metric

        for line in lines[1:]:
            line_sp = [l for l in line.strip("\n").split("\t") if l and l != "\t"]
            found_inds = []
            for ind, met_value in enumerate(line_sp):
                if ind == 4:
                    print("So")
                metrics[metrics_ind[ind]].append(float(met_value))
                found_inds.append(ind)
            for key, value in metrics_ind.items():
                if key not in found_inds:
                    metrics[value].append(0.0)

        if "Epoch" in metrics.keys():
            offset = 1

        # Plot figure
        plt.figure(figsize=(12, 12))
        n_sq = int(np.ceil(np.sqrt(
            len(list(metrics.keys())) - 1)))
        counter = 1

        for key, value in metrics.items():
            if key == "Epoch":
                continue
            plt.subplot(n_sq, n_sq, counter)
            plt.plot(metrics["Epoch"], value)
            plt.xlabel("Epoch")
            plt.ylabel(key)
            plt.title("Validation %s" % (key))
            counter += 1
        plt.savefig(os.path.join(opt.results_dir, 'validation_results.png'))


# Test 1: We generate some samples, and we see the variation on the generated samples
# Number of samples necessary to calculate the standard deviation

if test_flags[1]:

    print("Test 1: processing sequence images...")
    # Parameter initialisation
    number_samples = 15  # Number of samples generated from the same image
    ssim_values = []  # Values yielded for structural similarity
    metrics = {'sequence': [],
               'ssim': [],
               'l2': [],
               'ssim_ss': []
               }  # We want the SSIM between other and equivalent sequence input, and SSIM between other and Other seq input

    counter_resid = 1

    if not os.path.exists(os.path.join(opt.results_dir, 'generated_im')):
        os.makedirs(os.path.join(opt.results_dir, 'generated_im'))

    if opt.mod_classifier is not None:
        # We load a modality classifier
        try:
            mod_disc = Modisc(len(opt.sequences), len(opt.datasets), 0.2, 2, 1).cuda()
            mod_disc = util.load_network(mod_disc, 'MD', str(opt.mod_classifier), opt, strict=False)
            mod_disc.eval()
            instance_accuracy = {'seq': [], 'slice':[], 'mod': [], 'dat': []}
            targets_mod = {}  # Targets for modality classification
            targets_dat = {}  # Targets for dataset classification
            for mod_ind, mod in enumerate(opt.sequences):
                targets_mod[mod] = torch.zeros(len(opt.sequences))
                targets_mod[mod][mod_ind] = 1.0
            for dat_ind, dat in enumerate(opt.datasets):
                targets_dat[dat] = torch.zeros(len(opt.datasets))
                targets_dat[dat][dat_ind] = 1.0
        except:
            mod_disc = None
            Warning("We could not load the modality classification network. ")
    else:
        mod_disc = None

    # Create dataset container
    if opt.dataset_type == 'sliced':
        dataset_container = SpadeNaiSlice(opt, mode='test')
    else:
        dataset_container = SpadeNai(opt, mode='test')
        dataset_container.flushStoredSlices(allow_stored=True)
    dataset_container.setDiffSlices(False)

    for mod in modalities:
        dataset_container.resetDatasets(fixed_modality=mod)
        dataloader = DataLoader(dataset_container.sliceDataset,
                                batch_size=opt.batchSize, shuffle=False,
                                num_workers=int(opt.nThreads), drop_last=opt.isTrain)

        for i, data_i in enumerate(dataloader):
            print("Iteration %d" % i)
            this_seq = data_i['this_seq']  # Main sequence of the data item

            # Modality discriminator: set ground truths
            if mod_disc is not None:
                instance_accuracy_sample = {'mod': np.zeros((len(this_seq), number_samples)),
                                            'dat': np.zeros((len(this_seq), number_samples))}  # List that will contain the modality classification accuracy of each sample.
                # Create targets for modality and dataset discrimination
                target_mod = torch.zeros(len(data_i['this_seq']), len(opt.sequences))  # No one hot target
                for b in range(len(data_i['this_seq'])):
                    target_mod[b, :] = targets_mod[data_i['this_seq'][b]]
                target_dat = torch.zeros(len(data_i['this_seq']), len(opt.datasets))  # No one hot target
                for b in range(len(data_i['this_dataset'])):
                    target_dat[b, :] = targets_dat[data_i['this_dataset'][b]]

            for samp in range(number_samples):

                # We generate a sample from our model
                if opt.mode == 'eval':
                    model.eval()  # Turn on test mode

                generated = model(data_i, 'generator_test')

                # Modality
                if mod_disc is not None:
                    logits_mod, logits_dat = mod_disc(generated)
                    pred_mod = torch.softmax(logits_mod, 1).detach().cpu()
                    pred_dat = torch.softmax(logits_dat, 1).detach().cpu()
                    # Shape: pred_mod BxNUM_MODS, pred_dat BxNUM_DATS. Same for target.
                    # We do 1-Relative Error
                    for b in range(pred_mod.shape[0]):
                        instance_accuracy_sample['mod'][b, samp] = 100*(torch.argmax(pred_mod[b,...], 0) == torch.argmax(
                            target_mod[b,...], 0)).type('torch.DoubleTensor').numpy()
                        instance_accuracy_sample['dat'][b, samp] = 100*(torch.argmax(pred_dat[b, ...], 0) == torch.argmax(
                            target_dat[b,...], 0)).type('torch.DoubleTensor').numpy()


                if samp == 0:
                    # We save the first generated images
                    generated_0 = generated.clone()

                    # Original image processing. It is key that we calculate structural similarity with the central channel ONLY.
                    original_im = data_i['image'].clone().detach().cpu()  # Original Image

                    # Uncertainties
                    uncert = torch.zeros_like(generated)  # Uncertainty vectors for the instances of the generated image
                    uncert = uncert.unsqueeze(0)  # We add another dimension
                    uncert[0, ...] = generated

                    # Metrics
                    generated = generated.detach().cpu()

                    # Mask for Skull Strip SS
                    mask = data_i['label'][:, 0:1, ...].detach().cpu() == 0  # 0th element is background, equal to 1 for

                    # background, not zero for no background.
                    metrics['sequence']+=this_seq
                    metrics['ssim'] += uvir.structural_Similarity(generated, original_im)
                    metrics['l2'] += uvir.l2(generated, original_im)
                    metrics['ssim_ss'] += uvir.structural_Similarity(
                        generated * mask,
                        original_im * mask)
                else:
                    # Add uncertainties and images
                    generated = generated.unsqueeze(0)
                    uncert = torch.cat((generated, uncert), dim=0)

            if mod_disc is not None:
                for b in range(len(this_seq)):
                    instance_accuracy['mod'].append(np.mean(instance_accuracy_sample['mod'][b, ...]))
                    instance_accuracy['dat'].append(np.mean(instance_accuracy_sample['dat'][b, ...]))
                instance_accuracy['seq'] += data_i['this_seq']
                instance_accuracy['slice']+=list(data_i['slice_no'].numpy())

            # Plot
            for b in range(generated_0.shape[0]):  # For each image of the batch (should be one)

                all_path = os.path.join(opt.results_dir, 'generated_im')

                # Processing of new_name
                final_name = data_i['label_path'][b].split("/")[-1].replace(".nii.gz", "").replace("Parcellation_", "")
                final_name = final_name + "_" + str(data_i['slice_no'][b].numpy())

                # Sequence
                this_seq = data_i['this_seq'][b]
                label = data_i['label'][b, ...]  # Input label
                label = uvir.PVM2Label(label)
                input_ = data_i['image'][b]
                synth = generated_0[b]  # Input image
                all_uncerts = uncert[:, b, :, :, :]  # We select the batch
                std_uncert = all_uncerts.std(dim=0)  # Standard deviation

                tile = opt.batchSize > 8
                std_uncert = std_uncert.detach().cpu().numpy()  # Take the channels
                std_uncert = 255 * np.transpose(std_uncert, (1, 2, 0))

                # We perform the same task for the alternative sequences
                # All images
                all_images_2save = []
                all_names_2save = []
                all_uncerts_2_save = []
                all_uncerts_names_2_save = []

                # PROCESSING OF IMAGE GRID
                all_images_2save.append(label)  # Append the label
                all_images_2save.append(input_)
                all_images_2save.append(synth)  # Append synthesized image
                all_names_2save.append("Label")
                all_names_2save.append("Ground truth " + this_seq)
                all_names_2save.append("Gen. " + this_seq)

                # PROCESSING OF UNCERTAINTY RIDGE
                all_uncerts_2_save.append(synth)
                all_uncerts_2_save.append(std_uncert)
                all_uncerts_names_2_save.append(this_seq)
                all_uncerts_names_2_save.append("Uncertainty")

                # We save

                uvir.saveFigs_stacks(all_images_2save,
                              os.path.join(all_path, final_name + '_generated_GAN_' + this_seq + '.png'), True,
                              opt.label_nc, -1, True,
                              titles=all_names_2save, batch_accollades={}, index_label=0,
                              bound_normalization=opt.bound_normalization)
                uvir.saveFigs(all_uncerts_2_save,
                              os.path.join(all_path, final_name + '_uncertainties_GAN' + this_seq + '.png'), True,
                              opt.label_nc, -1,
                              False, titles=all_uncerts_names_2_save, batch_accollades={}, index_label=-1,
                              bound_normalization=opt.bound_normalization)

        uvir.write_plot_metrics(metrics, names={'sequence': 'Sequence', 'ssim': 'SSIM', 'l2': 'L2',
                                                'ssim_ss': "SSIM SkullStrip"},
                                filename=os.path.join(all_path, 'metrics_results.txt'))

    if mod_disc is not None:
        with open(os.path.join(all_path, 'accuracy.txt'), 'w') as f:
            f.write("Accuracy on sample\tAccuracy Mod (%)\tAccuracy Dat (%)\n")
            for ind, i in enumerate(instance_accuracy['mod']):
                towrite = np.round(i * 100, 2)
                f.write("%s\t%d\t%f\t%f\n" % (instance_accuracy['seq'][ind], instance_accuracy['slice'][ind],
                                              i, instance_accuracy['dat'][ind]))
            f.write("Mean:\t%f\t%f\n" % (np.mean(instance_accuracy['mod']),
                                         np.mean(instance_accuracy['dat'])))
            f.write("STD:\t%f\t%f\n" % (np.std(instance_accuracy['mod']),
                                        np.std(instance_accuracy['dat'])))

    dataset_container.clearCache()
if test_flags[2]:
    print("Test 2: Lesion inclusion")
    if not os.path.isdir(os.path.join(opt.results_dir, 'lesion_generation')):
        os.makedirs(os.path.join(opt.results_dir, 'lesion_generation'))

    if opt.mode == 'eval':
        model = model.eval()

    # Just in case, we pre-save the original options.
    opt_backup = deepcopy(opt)

    # Modificatin of parameters
    # Here we need to use working Volumes
    opt.fix_seq = None
    opt.intensify_lesions = 1  # Period of sampling lesions: 1. We always sample lesions in case of volume dataset
    # Otherwise it's irrelevant.

    # Create dataset container
    if opt.dataset_type == 'sliced':
        # In sliced datasets, you cannot know a priori which lesions are going to be present.
        # It is thus recommended to use a different set of data, containing lesions, for this,
        # that you can input in the prompt.
        input_image_dir = input("Input the path to a directory containing images with lesions.")
        input_label_dir = input("Input the path to a directory containing label with lesions.")
        opt_copy = deepcopy(opt)
        opt_copy.image_dir = input_image_dir
        opt_copy.label_dir = input_label_dir
        dataset_container = SpadeNaiSlice(opt_copy, mode='test')
    else:
        dataset_container = SpadeNai(opt, mode='test')
        dataset_container.flushStoredSlices(allow_stored=True)
        dataset_container.setDiffSlices(False)

    # Store images here
    images_stored = {}

    # Loop around each dataset
    for mod in opt.sequences:
        dataset_container.resetDatasets(fixed_modality=mod)
        dataloader = DataLoader(dataset_container.sliceDataset,
                                batch_size=opt.batchSize, shuffle=False,
                                num_workers=int(opt.nThreads), drop_last=opt.isTrain)

        # Loop along the dataloader with fixed sequence == mod.
        for ind_i, data_i in enumerate(dataloader):
            generated = model(data_i, mode='generator_test')
            # We store the generated images with key: label_path.
            for b in range(generated.shape[0]):
                key_path = data_i['label_path'][b]
                if key_path not in images_stored.keys():
                    images_stored[key_path] = {}
                if 'Label' not in images_stored[key_path].keys():
                    label = uvir.PVM2Label(data_i['label'][b, ...].detach().cpu())
                    images_stored[key_path]['Label'] = ut.tensor2label(label, opt.label_nc, tile=None)

                images_stored[key_path]['Generated ' + mod] = ut.tensor2im(generated[b, ...].detach().cpu(), tile=None)
                images_stored[key_path]['Ground truth ' + mod] = ut.tensor2im(data_i['image'][b, ...].detach().cpu(),
                                                                              tile=None)

    for key_path, images in images_stored.items():
        images_2_save = list(images.values())
        titles_2_save = list(images.keys())
        all_path = os.path.join(opt.results_dir, 'lesion_generation')
        name_file = key_path.split("/")[-1].replace(".npz", ".png").replace("Parcellation", "Lesion_GEN")
        uvir.saveFigs(images_2_save, os.path.join(all_path, name_file), True, opt.label_nc, False,
                      titles=titles_2_save, batch_accollades={}, index_label=0, bound_normalization=opt.bound_normalization)

    opt = opt_backup
    dataset_container.clearCache()
if test_flags[3]:

    print("Test 3: cross-slice generation")

    if not os.path.isdir(os.path.join(opt.results_dir, 'cross_slice')):
        os.makedirs(os.path.join(opt.results_dir, 'cross_slice'))

    # Backup options to modify
    opt_backup = opt

    if opt.mode == 'eval':
        model.eval()

    # Create dataset container
    if opt.dataset_type == 'sliced':
        dataset_container = SpadeNaiSlice(opt, mode='test')
        dataset_container.setDiffSlices(True, reset_datasets=True)
    else:
        dataset_container = SpadeNai(opt, mode='test')
        dataset_container.flushStoredSlices(allow_stored=True)
        dataset_container.setIntensifyLesions(False)
        dataset_container.setDiffSlices(True, reset_datasets=True)

    dataloader = DataLoader(dataset_container.sliceDataset,
                            batch_size=opt.batchSize, shuffle=False,
                            num_workers=int(opt.nThreads), drop_last=opt.isTrain)

    # We only generate max_images.
    counter_image = 0
    max_images = 20

    for ind, data_i in enumerate(dataloader):

        generated = model(data_i, 'generator_test')  # Forward image from one modality using another slice as reference.
        ground_truth = data_i['image']
        style = data_i['style_image']
        data_i['style_image'] = data_i['image']
        generated_ori = model(data_i, 'generator_test')

        # Saving images
        for b in range(generated.shape[0]):
            tile = opt.batchSize > 8
            label = uvir.PVM2Label(data_i['label'][b, ...].detach().cpu())

            all_images_2save = []
            all_images_2save.append(ut.tensor2label(data_i['label'][b], opt.label_nc, tile=tile))  # Label
            all_images_2save.append(ut.tensor2im(style[b, ...].detach().cpu(), tile=tile))  # Other slice
            all_images_2save.append(ut.tensor2im(generated[b, ...].detach().cpu(), tile=tile))  # Generated (other)
            all_images_2save.append(
                ut.tensor2im(generated_ori[b, ...].detach().cpu(), tile=tile))  # Generated (original)
            all_images_2save.append(
                ut.tensor2im(ground_truth[b, ...].detach().cpu(), tile=tile))  # Original slice

            # Names
            titles = ["Label", "Style slice", "Generated (SS)", "Generated (GT)", "Ground truth"]
            name = "cross_sequence_%s_%d.png" % (data_i['this_seq'][b], counter_image)

            uvir.saveFigs(all_images_2save,
                          os.path.join(opt.results_dir, 'cross_slice', name), True, opt.label_nc, False,
                          titles=titles, batch_accollades={}, index_label=0, bound_normalization=opt.bound_normalization)

            counter_image += 1

        if counter_image >= max_images:
            break

    opt = opt_backup
    dataset_container.clearCache()

if test_flags[4] and opt.dataset_type == 'volume':

    print("Test 4: whole 3D volume render")
    dir_3D = os.path.join(opt.results_dir, 'test_3D')
    if not os.path.isdir(dir_3D):
        os.makedirs(dir_3D)

    modalities = ['T1', 'FLAIR', 'B0']
    number_of_volumes = 5  # Number of volumes per modality

    # Dataset container creation
    opt.fix_seq = None
    dataset_container = SpadeNai(opt, mode='test')

    # We select number_of_volumes labels from the directory.
    labels = np.random.choice(os.listdir(dataset_container.label_dir), number_of_volumes, replace=False)

    for l in labels:
        image_paths = {}
        images = dataset_container.findEquivalentPath(l, dataset_container.image_dir,
                                                      keywords=["sub", "ses", ".npz"] + opt.datasets)

        for mod in modalities:
            for i in images:
                if mod in i.split("_")[0]:
                    image_paths[mod] = os.path.join(dataset_container.image_dir, i)

        dataset_container.setWorkingVolume(image_paths, os.path.join(dataset_container.label_dir,l))

        # Now, we loop along each modality, and we reconstruct every volume.
        for mod in modalities:
            if mod not in image_paths.keys():
                nf = True
                continue
            else:
                nf = False
            for i in range(int(np.ceil(dataset_container.lenIteratorWV()/opt.batchSize))):
                data_i = dataset_container.iterateThroughWV(mod, opt.batchSize, use_same_style_slice=True)
                generated = model(data_i, mode='generator_test')
                dataset_container.working_volume.storeSlice(generated.detach().cpu(),
                                                                modality=mod)

            name_file_RECON = l.replace(
                "Parcellation", "GEN_" + mod).replace(".npz", ".nii.gz")
            dataset_container.working_volume.storeReconstruction(os.path.join(dir_3D, name_file_RECON),
                                                                 modality = mod)

            nii_file_GT = nib.Nifti1Image(dataset_container.working_volume.volumes['gt' + mod].numpy(),
                                          affine=dataset_container.working_volume.affines['gt' + mod])
            name_file_GT = data_i['image_path'][0].split("/")[-1].replace(".npz", ".nii.gz")

            nib.save(nii_file_GT, os.path.join(dir_3D, name_file_GT))

        # We also save the label as Nii file just in case.
        nii_file_LAB = nib.Nifti1Image(dataset_container.working_volume.volumes['Parcellation'].numpy(),
                                       affine=dataset_container.working_volume.affines['Parcellation'])
        name_file_LAB = l.replace(".npz", ".nii.gz")

        if not nf:
            del nii_file_GT
        del nii_file_LAB
    dataset_container.clearCache()
    del dataset_container  # We delete to avoid clotting.

elif opt.dataset_type == 'sliced':
    print("Skipped test 4: whole volume generation. Dataset type passed is sliced. Pass a volume dataset to through this dataset.")
if test_flags[5] and opt.dataset_type == 'volume':

    print("Test 5: Histograms")

    if not os.path.isdir(os.path.join(opt.results_dir, 'histograms')):
        os.makedirs(os.path.join(opt.results_dir, 'histograms'))

    opt_backup = deepcopy(opt)

    # We select number_of_volumes labels from the directory.
    if opt.dataset_type == 'sliced':
        dataset_container = SpadeNaiSlice(opt, mode='test')
    else:
        dataset_container = SpadeNai(opt, mode='test')
        dataset_container.flushStoredSlices(allow_stored=True)

    labels = os.listdir(dataset_container.label_dir)

    for l in labels:
        print("Histogram processing of label %s" % l)
        image_paths = {}
        images = dataset_container.findEquivalentPath(l, dataset_container.image_dir,
                                                      keywords=["sub", "ses", ".npz"] + opt.datasets)

        for mod in modalities:
            for i in images:
                if mod in i.split("_")[0]:
                    image_paths[mod] = os.path.join(dataset_container.image_dir, i)

        dataset_container.setWorkingVolume(image_paths, os.path.join(dataset_container.label_dir, l))

        hist_vols = {}
        for mod in modalities:
            print("Beginning %s..." % mod)
            if mod not in dataset_container.working_volume.paths.keys():
                print("%s not found here, skip" %mod)
                break
            for i in range(int(np.ceil(dataset_container.lenIteratorWV()/opt.batchSize))):
                data_i = dataset_container.iterateThroughWV(mod, opt.batchSize, use_same_style_slice=True)
                generated = model(data_i, mode='generator_test')
                dataset_container.working_volume.storeSlice(generated.detach().cpu(),modality=mod)

            hist_vols[mod] = [dataset_container.working_volume.normalizeVolume(mod),
                              dataset_container.working_volume.volumes['recon' + mod]]

            print("%s ended" % mod)

        # Now that we have all modalities within hist_vols, plot 3D histogram.
        save_name = l.replace(".npz", ".png").replace("Parcellation", "histogram")
        uvir.plot3DHist(os.path.join(opt.results_dir, 'histograms', save_name),
                        hist_vols, threshold = -1000)

    dataset_container.clearCache()
    del hist_vols, dataset_container
    opt = opt_backup
elif opt.dataset_type == 'sliced':
    print("Skipped test 5: histogram. Dataset type passed is sliced. Pass a volume dataset to through this dataset.")
if test_flags[6]:
    print("Test 6. Cross-style generation. ")

    opt_backup = opt
    opt.diff_style_volume = True # Necessary!

    if opt.mode == 'eval':
        model.eval()

    if not os.path.isdir(os.path.join(opt.results_dir, 'style_translation')):
        os.makedirs(os.path.join(opt.results_dir, 'style_translation'))

    # We select number_of_volumes labels from the directory.
    if opt.dataset_type == 'sliced':
        opt.diff_slices = True # This really doesn't matter because the style volume is different.
        dataset_container = SpadeNaiSlice(opt, mode='test')

    else:
        dataset_container = SpadeNai(opt, mode='test')
        dataset_container.flushStoredSlices(allow_stored=True)

    dataloader = DataLoader(dataset_container.sliceDataset,
                            batch_size=opt.batchSize, shuffle=False,
                            num_workers=int(opt.nThreads), drop_last=opt.isTrain)

    for ind_i, data_i in enumerate(dataloader):
        generated = model(data_i, mode='generator_test')
        for b in range(generated.shape[0]):
            to_save = []
            titles_to_save = ["Label",
                              "Ground truth %s" % data_i['this_dataset'][b],
                              "Style %s" % data_i['style_dataset'][b],
                              "Generated"
                              ]
            to_save.append(util.tensor2label(uvir.PVM2Label(data_i['label'][b, ...].detach().cpu()),
                                             opt.label_nc, tile=None))
            to_save.append(util.tensor2im(data_i['image'][b, ...].detach().cpu(), tile=None))
            to_save.append(util.tensor2im(data_i['style_image'][b, ...].detach().cpu(), tile=None))
            to_save.append(util.tensor2im(generated[b, ...].detach().cpu(), tile=None))

            new_dataset = data_i['style_dataset'][b]
            save_name = data_i['label_path'][b].split("/")[-1].replace(".npz", "to_" + new_dataset + ".png").replace(
                "Parcellation", "Cross_sequence")
            uvir.saveFigs(to_save,
                          os.path.join(opt.results_dir, 'style_translation', save_name), True, opt.label_nc,
                          False, titles=titles_to_save, batch_accollades={},
                          index_label=0, bound_normalization=opt.bound_normalization)

    # Reset and clean stuff.
    dataset_container.clearCache()
    opt.diff_style_volume = False
    del dataloader, dataset_container
    opt = opt_backup
if test_flags[7]:

    print("Test 7: Generation of instances.")
    if opt.mode == 'eval':
        model = model.eval()

    opt_backup = opt
    opt.batchSize = 1

    # Parameters
    n_samples = 15  # For how many images we will do this.
    max_images = 50

    save_dir = os.path.join(opt.results_dir, 'generated_instances')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    opt_backup = deepcopy(opt)

    # Create container and dataloader
    if opt.dataset_type == 'sliced':
        dataset_container = SpadeNaiSlice(opt, mode='test')
    else:
        dataset_container = SpadeNai(opt, mode='test')
    dataloader = DataLoader(dataset_container.sliceDataset,
                            batch_size=1, shuffle=False,
                            num_workers=int(opt.nThreads), drop_last=opt.isTrain)

    for ind_i, data_i in enumerate(dataloader):
        if ind_i > max_images:
            break
        this_seq = data_i['this_seq']  # Main sequence of the data item
        for samp in range(n_samples):
            generated_ = model(data_i, 'generator_test')

            # Storage of the tile
            if samp == 0:
                generated = torch.zeros_like(
                    generated_.detach().cpu())  # Uncertainty vectors for the instances of the generated image
                generated[0, -1] = generated_.detach().cpu()
            else:
                generated = torch.cat((generated, generated_.detach().cpu()), dim=0)

        # We take the input label and images
        label = uvir.PVM2Label(data_i['label'].detach().cpu())
        label = label[0]
        input = data_i['image'][0, 0:1, ...]  # Input image
        label = ut.tensor2label(label, opt.label_nc + 2, tile=None)
        input = ut.tensor2im(input, tile=None)
        generated = ut.tensor2im(generated, tile=None)

        # Final name
        final_name = data_i['label_path'][0].split("/")[-1].strip("Parcellation_").replace(".npz", ".png")
        uvir.saveFigs_1row(input, generated, create_dir=True, divide=4,
                           sequence=this_seq, fig_path=os.path.join(save_dir, final_name), bound_normalization=opt.bound_normalization)

    opt = opt_backup
    dataset_container.clearCache()
    del dataset_container, dataloader
if test_flags[8] and False:
    # Deprecated!
    print("MMD test.")
    number_passes = 20  # Number of passes through dataloader

    if opt.mode == 'eval':
        model.eval()

    # Initialise MMD results file
    with open(os.path.join(opt.results_dir, 'mmd_results.txt'), 'w') as f:
        f.write("Modality\tMMD\tNumber of samples\tSlice\t")

    dataset_container = SpadeNai(opt, mode='test')
    dataset_container.flushStoredSlices(True)

    for mod in modalities:
        # Initialise dataset
        dataset_container.resetDatasets(fixed_modality=mod)
        dataloader = DataLoader(dataset_container.sliceDataset,
                                batch_size=opt.batchSize, shuffle=False,
                                num_workers=int(opt.nThreads), drop_last=opt.isTrain)
        # Initialise values
        mmd_values = []
        n_samples = 0

        print("Going through %s data" % mod)

        for ind_i, data_i in enumerate(dataloader):
            gen = []
            for s in range(number_passes):
                generated = model(data_i, mode='generator_test')
                gen.append(generated.detach().cpu())
            gen = torch.cat(gen, dim =1)
            gen = torch.mean(gen, dim=1)
            gt = data_i['image'][:, 0, ...].detach().cpu()

            for b in range(gen.shape[0]):
                mmd_value, _ = uvir.MMDScore(gt[b, ...], gen[b, ...])
                mmd_values.append(mmd_value)
                with open(os.path.join(opt.results_dir, 'mmd_results.txt'), 'a') as f:
                    f.write("%s\t%.3f\t%d\t%d\n" % (mod, np.mean(mmd_values), number_passes, data_i['slice_no'][b]))

    dataset_container.clearCache()
if test_flags[9]:

    print("Dataset/slice analysis...")

    # Create saving directoryy
    save_dir = os.path.join(opt.results_dir, 'style_slice_analysis')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # We fetch a volume belonging to each of the datasets.
    if opt.dataset_type == 'volume':
        dataset_container = SpadeNai(opt, mode='test')
        dataset_container.setDiffSlices(True)

        # Select images
        possibles = os.listdir(dataset_container.label_dir)
        chosen_labels = []
        labels_dataset = {}
        for ds in dataset_container.datasets:
            possibles_ds = [i for i in possibles if ds in i]
            chosen_label = np.random.choice(possibles_ds)
            chosen_labels.append(chosen_label)
            labels_dataset[chosen_label] = ds

        # Now we have the labels for each of the datasets.
        # Create a dictionary with key: label, value: dictionary with equivalent modality images.
        eq_image = {}
        for label in chosen_labels:
            ims = dataset_container.findEquivalentPath(label, dataset_container.image_dir,
                                                       dataset_container.datasets + ['sub', 'ses', '.npz'],
                                                       first_come_first_served=False)
            eq_image[label] = {}
            for i in ims:
                mod = i.split("/")[-1].split("_")[0]
                eq_image[label][mod] = os.path.join(dataset_container.image_dir, i)

        # We generate a sample from our model
        if opt.mode == 'eval':
            model.eval()  # Turn on test mode
        else:
            model.train()

        # Categories labels
        number_slices = 5
        label_slices = {}

        # The first thing we are going to do is to set up a batch to load.
        dict_list = {}
        ground_truth_tile = {}

        for l_ind_main, l_main in enumerate(chosen_labels):
            # For each main style, we generate instances for each of the slice semantic maps.
            # Check slices for label
            if l_main in label_slices.keys():
                slices_l = label_slices[l_main]
            else:
                load_vols = np.load(os.path.join(dataset_container.label_dir, l_main))
                sh_lab = load_vols['label'].shape
                label_slices[l_main] = list(np.linspace(0, sh_lab[
                    dataset_container.cut_ids[dataset_container.cut]] - 1,
                                                        number_slices))
                label_slices[l_main] = [int(i) for i in label_slices[l_main]]
                slices_l = label_slices[l_main]

            for slice_main in label_slices[l_main]:
                tile = {}  # For each modality/slice/style, we'll generate an output tile

                # SHUFFLE STYLES.
                for l_ind_style, l_style in enumerate(chosen_labels):
                    if l_style == l_main:
                        gt_flag = True
                        label_style_dict = None
                    else:
                        gt_flag = False
                        # Set volume
                        label_style_dict = {}
                        for key in eq_image[l_style].keys():
                            label_style_dict[key] = os.path.join(dataset_container.label_dir, l_style)

                    dataset_container.setWorkingVolume(eq_image[l_style],
                                                       os.path.join(dataset_container.label_dir, l_main),
                                                       style_label_paths=label_style_dict)
                    # Loop along modalities
                    for mod in dataset_container.modalities:
                        # Initialise tile[mod] and ground_truth_tile[mod] if doesn't exist.
                        if mod not in tile.keys():
                            tile[mod] = []
                        if mod not in ground_truth_tile:
                            ground_truth_tile[mod] = {}

                        # Check slices for style
                        if l_style in label_slices.keys():
                            slices_s = label_slices[l_style]
                        else:
                            label_slices[l_style] = list(np.linspace(0,
                                                                     dataset_container.working_volume.volumes[
                                                                         'gt' + mod].shape[
                                                                         dataset_container.cut_ids[
                                                                             dataset_container.cut]] - 1,
                                                                     number_slices))
                            label_slices[l_style] = [int(i) for i in label_slices[l_style]]
                            slices_s = label_slices[l_style]

                        if mod not in eq_image[l_style].keys():
                            # For this modality, the selected style has no image: zeros.
                            tile[mod].append(torch.zeros([number_slices, ] + dataset_container.new_size[:-1]).type(
                                'torch.FloatTensor'))
                            if labels_dataset[l_style] not in ground_truth_tile[mod].keys():
                                ground_truth_tile[mod][labels_dataset[l_style]] = torch.zeros([number_slices] +
                                                                                              dataset_container.new_size[
                                                                                              :-1]
                                                                                              ).type(
                                    'torch.FloatTensor')
                        else:
                            # Modality is present. We load the slices and we generate.
                            data_i = dataset_container.getSlicefromWV(all_slices=[list(i) for i in
                                                                                  list(zip([slice_main] * len(slices_s),
                                                                                           slices_s))],
                                                                      modality=[mod] * len(slices_s),
                                                                      diff_style_slice=False)
                            generated = model(data_i, mode='generator_test')

                            # Append ground truth (if needed) and generated images.

                            if labels_dataset[l_style] not in ground_truth_tile[mod].keys():
                                ground_truth_tile[mod][labels_dataset[l_style]] = data_i['style_image'][:, 0,
                                                                                  ...].detach().cpu()
                            tile[mod].append(generated.detach().cpu().squeeze(1))

                # End of loop along the styles and modalities: now we save the tiles.
                for mod_ind, mod in enumerate(dataset_container.modalities):
                    # Tile is a N_STYLES list with size BxCxHxW where C = 1, and B = number of slices.
                    # We want to put the styles in the rows and the slices in the columns
                    output_tile = []
                    for gen_ind, gen in enumerate(tile[mod]):
                        tile_row = []
                        for b in range(gen.shape[0]):
                            tile_row.append(gen[b, ...].numpy())
                        tile_row = np.concatenate(tile_row, axis=-1)
                        output_tile.append(tile_row)

                    output_tile = np.concatenate(output_tile, axis=0)

                    # Save it
                    ticks_slices = list(range(number_slices)).copy()
                    ticks_slices.reverse()
                    ticks_datasets = dataset_container.datasets.copy()
                    ticks_datasets.reverse()

                    uvir.plotTile(output_tile, os.path.join(save_dir, "%s_%s_%d.png"
                                                            % (mod, labels_dataset[l_main], slice_main)),
                                  "Generated. Base %s_%s_%d" % (mod, labels_dataset[l_main], slice_main),
                                  ticks_slices, ticks_datasets)

            # Ground truth tile:
            # DICT: Key: modalities, value: dictionary with Key: dataset, value: N_SLICESxHxW
            for mod, values in ground_truth_tile.items():
                output_tile = []
                for style_name, styles_row in values.items():
                    tile_row = []
                    for b in range(styles_row.shape[0]):
                        tile_row.append(styles_row[b, ...].numpy())
                    tile_row = np.concatenate(tile_row, axis=-1)
                    output_tile.append(tile_row)

                output_tile = np.concatenate(output_tile, axis=0)
                # Save it
                ticks_slices = list(range(number_slices)).copy()
                ticks_slices.reverse()
                ticks_datasets = dataset_container.datasets.copy()
                ticks_datasets.reverse()

                uvir.plotTile(output_tile, os.path.join(save_dir, "GT_%s.png" % mod),
                              "Ground truth %s" % mod,
                              ticks_slices, ticks_datasets)

        dataset_container.clearCache()
        del dataset_container
    else:
        print("Test can't be run with dataset_type = sliced. Choose volume and pass volumes instead.")

if test_flags[10]:

    print("Unseen modalities test")

    # Backup
    opt_backup = opt

    # We backup sequences
    backup_seqs = opt.sequences
    backup_imdir = opt.image_dir
    backup_labdir = opt.label_dir

    # We replace those by guests if specified
    opt.sequences = opt.guest_mods
    if opt.sequences:
        opt.image_dir = opt.guest_images_dir
        opt.label_dir = opt.guest_labels_dir
        datasets = []
        for i in os.listdir(opt.guest_images_dir):
            for ds in opt.datasets:
                if ds in i:
                    datasets.append(ds)
                    break
        opt.datasets = datasets

        go_on_with_test = True
    else:
        go_on_with_test = False

    if opt.mode == 'eval':
        model. eval()

    # We run test
    if go_on_with_test:

        # Create storage directory.
        save_dir = os.path.join(opt.results_dir, "new_sequences")
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        # Create container
        if opt.dataset_type == 'volume':
            dataset_container = SpadeNai(opt, mode = 'test')
        else:
            dataset_container = SpadeNaiSlice(opt, mode = 'test')

        for mod_ind, mod in enumerate(opt.sequences):
            dataset_container.resetDatasets(fixed_modality=mod)
            dataloader = DataLoader(dataset_container.sliceDataset,
                                    batch_size=opt.batchSize, shuffle=False,
                                    num_workers=int(opt.nThreads), drop_last=opt.isTrain)
            for i, data_i in enumerate(dataloader):
                # We get the image
                generated = model(data_i, 'inference')  # We use the "this_seq" decoder.
                generated = generated.detach().cpu()

                for b in range(generated.shape[0]):
                    # We save it.
                    save_images = []
                    save_names = []
                    save_images.append(data_i['label'][b, ...])
                    save_images.append(data_i['style_image'][b, ...])
                    save_images.append(data_i['image'][b, ...])
                    save_images.append(generated[b, ...])
                    save_names = ["Label", "Style %s" % (data_i['this_seq'][b]),
                                  "GT %s" % (data_i['this_seq'][b]), "Generated"]

                    # Name
                    name = data_i['label_path'][b].split("/")[-1].replace("Parcellation", "Generated").replace(
                        ".npz", "_" + data_i['this_seq'][b] + ".png"
                    )

                    uvir.saveFigs(save_images, os.path.join(save_dir, name), True, opt.label_nc, -1, True,
                                  titles=save_names, batch_accollades={}, index_label=0, bound_normalization=opt.bound_normalization)

    dataset_container.clearCache()
    del dataset_container
    opt = opt_backup