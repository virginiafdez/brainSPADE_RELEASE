'''
Look at codes
Author: Virginia Fernandez (King's College London)
Date: October 2020
The script looks and plots mean codes for different modalities.
Different styles can be included.
'''

import data
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from models.pix2pix_model import Pix2PixModel
from data.spadenai_v2 import SpadeNai
from data.spadenai_v2_sliced import SpadeNaiSlice
from options.lookatcodes_options import LACodeOptions
import moreutils as uvir
import data.dataset_utils as dutils
from sklearn.manifold import TSNE, LocallyLinearEmbedding
from support_tests.hoyermetric import HoyerMetric
from sklearn.decomposition import PCA as PCA
import moviepy.video.io.ImageSequenceClip
from monai.data import DataLoader

# Options, dataloader and model
opt = LACodeOptions().parse()
model = Pix2PixModel(opt)

## Switches
do_reduction = opt.do_reduction
do_heatmaps = opt.do_heatmaps
do_sparsity = opt.do_sparsity
do_new_seq = opt.do_new_seq
do_volume = opt.do_volume
#
red_algo = opt.reduction_algorithm.lower()
if red_algo == 'tsne' or red_algo=='pca':
    ValueError("The reduction algorithm must be either PCA or TSNE")

# Creation of folder
if not os.path.exists(opt.savecode_dir):
    os.makedirs(opt.savecode_dir)


# Other variables
mods = opt.sequences

# Colors
colors = ['green', 'blue', 'pink', 'orange']
shapes = ["^", ".", "*", "D", "s", "+"]
colors_sub = {'green': ['darkgreen', 'lime', 'olive', 'darkolivegreen', 'lawngreen'],
              'blue': ['royalblue', 'dodgerblue', 'navy', 'turquoise', 'powderblue'],
              'pink': ['hotpink', 'magenta', 'mediumvioletred', 'lightpink', 'violet'],
              'orange': ['darksalmon', 'coral', 'orange', 'peru', 'orange']}

if do_reduction:
    print("Do reduction starting... ")
    if not os.path.exists(os.path.join(opt.savecode_dir, 'plot_analysis')):
        os.makedirs(os.path.join(opt.savecode_dir, 'plot_analysis'))
    if opt.reduction_itemizing:
        itemizing = os.path.join(opt.savecode_dir, 'plot_analysis', 'itemizing')
        if not os.path.isdir(itemizing):
            os.makedirs(itemizing)
        chosen_perp = 35
        chosen_lr = 300


    # Windows for TSNE
    PER = range(15, 55, 10) # Perplexity. Number of close neighbours considered for each point.
    LR_ = range(300, 700, 100)
    if red_algo == 'pca':
        LR = [1]

    codes = {}
    codes_dataset = {}
    codes_nr = []
    datasets = []
    modalities = []
    paths = []
    images = []

    # Forward pass and append codes
    if opt.dataset_type == 'volume':
        dataset_container = SpadeNai(opt, mode = 'test')
    else:
        dataset_container = SpadeNaiSlice(opt, mode='test')

    for mod in mods:
        opt.fix_seq = mod
        dataset_container.resetDatasets(fixed_modality=mod)
        dataloader = DataLoader(dataset_container.sliceDataset,
                                batch_size=opt.batchSize, shuffle=False,
                                num_workers=int(opt.nThreads), drop_last=opt.isTrain)

        for i, data_i in enumerate(dataloader):

            this_seq = data_i['this_seq'][0]
            if opt.mode == 'eval':
                model.eval()  # Turn on test mode
            # We get the MU
            _, generated = model(data_i, 'encode_only')  # We use the "this_seq" decoder.
            generated = generated.detach().cpu()
            codes_nr.append(generated)
            images.append(data_i['style_image'][:, 0, ...].detach().cpu())
            datasets += data_i['this_dataset']
            modalities += data_i['this_seq']
            paths += data_i['label_path']

    # Ammending of plots
    codes_nr = torch.cat(codes_nr, dim = 0).numpy()
    datasets = np.asarray(datasets)
    modalities = np.asarray(modalities)
    images = torch.cat(images, dim = 0)

    if opt.z_dim > 3:
        for per in PER:
            for lr in LR_:
                # Reduced algorithm
                if red_algo == 'tsne':
                    Y = TSNE(n_components=2, init='pca', random_state=0, perplexity=per,
                                  learning_rate=lr).fit(codes_nr).fit_transform(codes_nr)
                elif red_algo == 'pca':
                    Y = PCA(n_components=2).fit(codes_nr).fit_transform(codes_nr)
                elif red_algo == 'lle':
                    Y = LocallyLinearEmbedding(n_components=2, n_neighbors=per,
                                                    ).fit(codes_nr).fit_transform(codes_nr)
                # Plot
                plt.figure(figsize=(10, 7))
                n_subplots = len(list(codes.keys()))
                minim = [10000, 10000, 10000]
                maxim = [-10000, -10000, -10000]
                legend_ = []
                for seq_ind, seq in enumerate(opt.sequences):
                    for ds_ind, ds in enumerate(opt.datasets):
                        if Y[(modalities == seq) * (datasets == ds) ,0].shape[0] == 0:
                            continue
                        plt.scatter(Y[(modalities == seq) * (datasets == ds) ,0],
                                    Y[(modalities == seq) * (datasets == ds) ,1],
                                    cmap = plt.cm.Spectral,
                                    color = colors_sub[colors[seq_ind]][ds_ind],
                                    s = 8,
                                    marker=shapes[ds_ind])
                        legend_.append("%s-%s" %(seq, ds))
                plt.legend(legend_)
                name_fig = "average_code_%s_per%d_lr%d.png" %(red_algo, per, lr)
                plt.title("Average codes %s - perplexity %d - lr %d" %(red_algo, per, lr))
                plt.savefig(os.path.join(opt.savecode_dir, 'plot_analysis', name_fig))

                # Itemizing
                if  opt.reduction_itemizing:
                    if per == chosen_perp and lr == chosen_lr:
                        for ind_i, img in enumerate(images):
                            plt.figure(figsize=(16, 7))
                            plt.subplot(1, 2, 1)
                            plt.imshow(img, cmap = 'gray')
                            plt.title("Image")

                            # Code plot
                            plt.subplot(1, 2, 2)
                            minim = [10000, 10000, 10000]
                            maxim = [-10000, -10000, -10000]
                            legend_ = []
                            for seq_ind, seq in enumerate(opt.sequences):
                                for ds_ind, ds in enumerate(opt.datasets):
                                    plt.scatter(Y[(modalities == seq) * (datasets == ds), 0],
                                                Y[(modalities == seq) * (datasets == ds), 1],
                                                cmap=plt.cm.Spectral,
                                                color=colors_sub[colors[seq_ind]][ds_ind],
                                                s=8,
                                                marker=shapes[ds_ind])
                                    legend_.append("%s-%s" % (seq, ds))
                            plt.scatter(Y[ind_i, 0], Y[ind_i, 1], color = 'red', s = 12, marker = "D")
                            legend_.append("This image")
                            plt.legend(legend_)
                            plt.title("Code plots")
                            name_fig = "itemizing_%d" %ind_i
                            plt.savefig(os.path.join(opt.savecode_dir, 'plot_analysis', 'itemizing', name_fig))

                    # Save in video format
                    fps = 0.5
                    image_files = [os.path.join(opt.savecode_dir, 'plot_analysis', 'itemizing')
                                   + '/' + img for img in os.listdir(os.path.join(opt.savecode_dir,
                                                                                  'plot_analysis', 'itemizing'))
                                   if img.endswith(".png")]
                    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
                    clip.write_videofile(os.path.join(opt.savecode_dir, 'plot_analysis', 'itemizing',
                                                      'whole_movie.mp4'))
    else:
        if opt.z_dim == 3:
            fig = plt.figure(figsize=(10, 7))
            axes = fig.add_subplot(111, projection='3d')
            legend_ = []
            for seq_ind, seq in enumerate(opt.sequences):
                for ds_ind, ds in enumerate(opt.datasets):
                    axes.scatter(xs=codes_nr[(modalities == seq) and (datasets == ds) ,0],
                                 ys=codes_nr[(modalities == seq) and (datasets == ds), 1],
                                 zs=codes_nr[(modalities == seq) and (datasets == ds), 2],
                                 cmap=plt.cm.Spectral,
                                 color=colors_sub[colors[seq_ind]][ds_ind],
                                 s=8,
                                 marker = shapes[ds_ind])
                    legend_.append("%s-%s" %(seq, ds))
            plt.title("Average codes")
            name_fig = "average_codes_raw.png"
            plt.savefig(os.path.join(opt.savecode_dir, 'plot_analysis', name_fig))

    dataset_container.clearCache()

if do_heatmaps:
    '''
    We again loop through the dataset. This time, we call the encoder AND the decoder,
    but in between, we zero each item of the code. Of course, this will imply looping 
    around the same image Z_DIM times, and this for each image. It will take forever!
    This is why we only do this for 2 images, one from each modality, and we only do
    it for N_CODES codes, the ones with the highest absolute value.
    '''

    print("Do heatmaps starting...")
    N_ITERS = 20

    if not os.path.exists(os.path.join(opt.savecode_dir, 'HEATMAPS')):
        os.makedirs(os.path.join(opt.savecode_dir, 'HEATMAPS'))

    # Forward pass and append codes
    dataset_container.resetDatasets(modalities = opt.sequences, fixed_modality=None)
    dataloader = DataLoader(dataset_container.sliceDataset,
                                batch_size=opt.batchSize, shuffle=False,
                                num_workers=int(opt.nThreads), drop_last=opt.isTrain)

    # Loop along dataloader
    sample_id = 0
    for i, data_i in enumerate(dataloader):
        this_seq = data_i['this_seq']  # Main sequence of the data item
        if opt.mode == 'eval':
            model.eval()  # Turn on test mode
        original_im = model(data_i, 'generator_test').detach().cpu()
        gen_ims = model(data_i, 'generate_mu')

        for batch_ind in range(original_im.shape[0]):
            uvir.plotMus(original_im[batch_ind,...],
                              gen_ims[batch_ind, ...],
                         os.path.join(opt.savecode_dir, 'HEATMAPS', 'sample_%d_mod%s' %(sample_id,
                                                                                             this_seq[batch_ind])),
                         bound_normalization=opt.bound_normalization)
            sample_id += 1
        if (i+1) > N_ITERS:
            break

    dataset_container.clearCache()

if do_sparsity:
    '''
    Calculates the mean sparsity of the codes for each of the modality and their standard deviation
    using Hoyer Metric
    '''

    print("Do sparsity starting... ")
    save_text_file = os.path.join(opt.savecode_dir, 'sparsity_data')
    hmetrics = []
    all_modalities = []

    # Forward pass and append codes
    dataset_container.resetDatasets(modalities=opt.sequences, fixed_modality=None)
    dataloader = DataLoader(dataset_container.sliceDataset,
                            batch_size=opt.batchSize, shuffle=False,
                            num_workers=int(opt.nThreads), drop_last=opt.isTrain)

    hmet = HoyerMetric(dimension = opt.z_dim)
    for i, data_i in enumerate(dataloader):
        all_modalities += data_i['this_seq']  # Main sequence of the data item
        if opt.mode == 'eval':
            model.eval()  # Turn on test mode
        generated, _ = model(data_i, 'encode_only')  # We use the "this_seq" decoder.
        generated = generated.detach().cpu()
        # Mean per batch
        hmetrics.append(hmet.__call__(generated, reduce=False, normalize=True))

    # We mask each modality
    hmetrics = torch.cat(hmetrics).numpy() # From the list we create a whole tensor
    all_modalities = np.array(all_modalities)
    list_mods = np.unique(all_modalities)
    write_mode = 'w'
    for mod in list_mods:
        mask = all_modalities == mod
        hmetrics_sub = hmetrics[mask]
        with open(save_text_file, write_mode) as f:
            mean_ = np.round(np.mean(hmetrics_sub), 2)
            std_ = np.round(np.std(hmetrics_sub), 2)
            f.write("Modality %s: mean %f std %f \n" %(mod, mean_, std_))
        write_mode = 'a'

if do_new_seq:

    print("Do new seq starting... ")
    if not os.path.exists(os.path.join(opt.savecode_dir, 'plot_analysis')):
        os.makedirs(os.path.join(opt.savecode_dir, 'plot_analysis'))

    # Windows for TSNE
    PER = range(15, 55, 10)  # Perplexity. Number of close neighbours considered for each point.
    LR_ = range(100, 500, 100)
    if red_algo == 'pca':
        PER = [1]
        LR = [1]
    if red_algo == 'lle':
        PER = [1]

    # Initialise
    codes_nr = []
    modalities = []
    datasets = []
    dataset_container = SpadeNai(opt, mode='test')

    # We start with the non-guest mode.
    for mod in opt.sequences:
        dataset_container.resetDatasets(fixed_modality=mod)
        dataloader = DataLoader(dataset_container.sliceDataset,
                                batch_size=opt.batchSize, shuffle=False,
                                num_workers=int(opt.nThreads), drop_last=opt.isTrain)
        for i, data_i in enumerate(dataloader):
            if opt.mode == 'eval':
                model.eval()  # Turn on test mode
            # We get the MU
            _, generated = model(data_i, 'encode_only')  # We use the "this_seq" decoder.
            # Mean per batch
            generated = generated.detach().cpu()
            # Save values
            codes_nr.append(generated)
            modalities += data_i['this_seq']
            datasets += data_i['this_dataset']

    dataset_container.clearCache()

    # Backup old data
    backup_seqs = opt.sequences
    backup_ds = opt.datasets
    backup_imdir = opt.image_dir
    backup_labdir = opt.label_dir

    # Guest modalities

    opt.sequences = opt.guest_mods
    if opt.sequences:
        opt.image_dir = opt.guest_images_dir
        opt.label_dir = opt.guest_labels_dir
        ds_ = []
        for i in os.listdir(opt.guest_images_dir):
            for ds in opt.datasets:
                if ds in i and ds not in ds_:
                    ds_.append(ds)
                    break
        opt.datasets = ds_
        guest_label = opt.guest_label
        skip_guests = False
    else:
        skip_guests = True

    # In case there's a match between the original datasets and the ones from the new modalities,
    # we re-label them to make sure they are different.
    guest_mods_tags = []
    for ind, gm in enumerate(opt.sequences):
        repeated = False
        for mod in mods:
            if gm == mod:
                guest_mods_tags.append("%s_%s" % (gm, guest_label))
                repeated = True
                break
        if not repeated:
            guest_mods_tags.append(gm)

    # Backup data
    backup_imdir = opt.image_dir
    backup_labdir = opt.label_dir
    opt.sequences = opt.guest_mods
    opt.image_dir = opt.guest_images_dir
    opt.label_dir = opt.guest_labels_dir

    dataset_container = SpadeNai(opt, mode = 'test')

    for mod_ind, mod in enumerate(opt.sequences):
        dataset_container.resetDatasets(fixed_modality=mod)
        dataloader = DataLoader(dataset_container.sliceDataset,
                                batch_size=opt.batchSize, shuffle=False,
                                num_workers=int(opt.nThreads), drop_last=opt.isTrain)
        for i, data_i in enumerate(dataloader):

            this_seq = data_i['this_seq']  # Main sequence of the data item
            if opt.mode == 'eval':
                model.eval()  # Turn on test mode
            # We get the MU
            _, generated = model(data_i, 'encode_only')  # We use the "this_seq" decoder.
            generated = generated.detach().cpu()
            # Save values
            codes_nr.append(generated)
            modalities += data_i['this_seq']
            datasets += data_i['this_dataset']

    # Restore data
    opt.sequences = backup_seqs
    opt.datasets = backup_ds
    opt.image_dir = backup_imdir
    opt.label_dir = backup_labdir

    # Plot
    # Ammending of plots
    codes_nr = torch.cat(codes_nr, dim = 0).numpy()
    datasets = np.asarray(datasets)
    modalities = np.asarray(modalities)

    if opt.z_dim > 3:
        for per in PER:
            for lr in LR_:
                # Reduced algorithm
                if red_algo == 'tsne':
                    Y = TSNE(n_components=2, init='random', random_state=0, perplexity=per,
                                  learning_rate=lr).fit(codes_nr).fit_transform(codes_nr)
                elif red_algo == 'pca':
                    Y = PCA(n_components=2).fit(codes_nr).fit_transform(codes_nr)
                elif red_algo == 'lle':
                    Y = LocallyLinearEmbedding(n_components=2, n_neighbors=per,
                                                    ).fit(codes_nr).fit_transform(codes_nr)
                # Plot
                plt.figure(figsize=(10, 7))
                minim = [10000, 10000, 10000]
                maxim = [-10000, -10000, -10000]
                legend_ = []
                for seq_ind, seq in enumerate(opt.sequences+opt.guest_mods):
                    for ds_ind, ds in enumerate(opt.datasets):
                        plt.scatter(Y[(modalities == seq) * (datasets == ds) ,0],
                                    Y[(modalities == seq) * (datasets == ds) ,1],
                                    cmap = plt.cm.Spectral,
                                    color = colors_sub[colors[seq_ind]][ds_ind],
                                    s = 8,
                                    marker = shapes[ds_ind])
                        legend_.append("%s-%s" %(seq, ds))
                plt.legend(legend_)
                name_fig = "new_seq_average_code_%s_per%d_lr%d.png" %(red_algo, per, lr)
                plt.title("Average codes %s - perplexity %d - lr %d" %(red_algo, per, lr))
                plt.savefig(os.path.join(opt.savecode_dir, 'plot_analysis', name_fig))

    else:
        if opt.z_dim == 3:
            fig = plt.figure(figsize=(10, 7))
            axes = fig.add_subplot(111, projection='3d')
            legend_ = []
            for seq_ind, seq in enumerate(opt.sequences+opt.guest_mods):
                for ds_ind, ds in enumerate(opt.datasets):
                    axes.scatter(xs=codes_nr[(modalities == seq) and (datasets == ds) ,0],
                                 ys=codes_nr[(modalities == seq) and (datasets == ds), 1],
                                 zs=codes_nr[(modalities == seq) and (datasets == ds), 2],
                                 cmap=plt.cm.Spectral,
                                 color=colors_sub[colors[seq_ind]][ds_ind],
                                 s=8,
                                 marker = shapes[ds_ind])
                    legend_.append("%s-%s" %(seq, ds))
            plt.title("Average codes")
            name_fig = "new_seq_average_codes_raw.png"
            plt.savefig(os.path.join(opt.savecode_dir, 'plot_analysis', name_fig))

if do_volume and opt.dataset_type == 'volume':

    # Load a volume and plot the codes resulting of different slices
    if not os.path.isdir(os.path.join(opt.savecode_dir, 'whole_volume')):
        os.makedirs(os.path.join(opt.savecode_dir, 'whole_volume'))

    number_of_volumes = 5  # Number of volumes per modality

    # Dataset container creation
    opt.fix_seq = None
    dataset_container = SpadeNai(opt, mode='test')

    # We select number_of_volumes labels from the directory.
    #labels = np.random.choice(os.listdir(dataset_container.label_dir), number_of_volumes, replace=False)
    labels = [i for i in os.listdir(dataset_container.label_dir) if 'BRATS-TCIA' in i or 'SABRE' in i]
    labels = labels[:number_of_volumes]

    # Definition of algorithm
    Y = TSNE(n_components=2, init='pca', random_state=0, perplexity=25,
             learning_rate=400)

    for l in labels:
        image_paths = {}
        images = dataset_container.findEquivalentPath(l, dataset_container.image_dir,
                                                      keywords=["sub", "ses", ".npz"] + opt.datasets)

        for mod in opt.sequences:
            for i in images:
                if mod in i.split("_")[0]:
                    image_paths[mod] = os.path.join(dataset_container.image_dir, i)

        dataset_container.setWorkingVolume(image_paths, os.path.join(dataset_container.label_dir,l))

        # Now, we loop along each modality, and we reconstruct every volume.
        codes = []
        mods = []
        for mod in opt.sequences:
            if mod not in image_paths.keys():
                nf = True
                continue
            else:
                nf = False
            for i in range(int(np.ceil(dataset_container.lenIteratorWV()/opt.batchSize))):
                data_i = dataset_container.iterateThroughWV(mod, opt.batchSize, use_same_style_slice=False,
                                                            use_same_slice_no=True)
                gen_code, _ = model(data_i, mode='encode_only')
                codes.append(gen_code.detach().cpu())
                mods += data_i['this_seq']

        codes = torch.cat(codes, 0)
        codes_red = Y.fit(codes).fit_transform(codes)
        mods = np.asarray(mods)

        plt.figure(figsize=(16, 7))

        for mod_ind, mod in enumerate(opt.sequences):
            minim = [10000, 10000, 10000]
            maxim = [-10000, -10000, -10000]
            legend_ = []
            plt.scatter(codes_red[(mods == mod), 0],
                        codes_red[(mods == mod), 1],
                        cmap=plt.cm.Spectral,
                        color=colors_sub[colors[mod_ind]][0],
                        s=8)
            legend_.append("%s" %mod)
        plt.legend(legend_)
        plt.title("TSNE-reduced code plots")
        name_fig = "%s_code-plots.png" %l.split(".")[0]
        plt.savefig(os.path.join(opt.savecode_dir, 'whole_volume', name_fig))
    dataset_container.clearCache()
