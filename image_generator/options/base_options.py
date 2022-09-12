"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import sys
import argparse
import os
from util import util
import torch
import models
import data
import pickle
import shutil

class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        # Dataset
        parser.add_argument('--label_dir', type=str, required=True,
                            help='path to the directory that contains label images')
        parser.add_argument('--image_dir', type=str, default=None,
                            help='path to the directory that contains photo images')
        # experiment specifics
        parser.add_argument('--name', type=str, default='label2coco', help='name of the experiment. It decides where to store samples and models_')

        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='checkpoints', help='models_ are saved here')
        parser.add_argument('--model', type=str, default='pix2pix', help='which model to use')
        parser.add_argument('--norm_G', type=str, default='spectralinstance', help='instance normalization or batch normalization')
        parser.add_argument('--norm_D', type=str, default='spectralinstance', help='instance normalization or batch normalization')
        parser.add_argument('--norm_E', type=str, default='spectralinstance', help='instance normalization or batch normalization')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')

        # input/output sizes
        parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        parser.add_argument('--new_size', type=int, nargs='*', default=[256, 256], help='New height and width to resize the images to.')
        parser.add_argument('--label_nc', type=int, default=182, help='# of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.')
        parser.add_argument('--contain_dontcare_label', action='store_true', help='if the label map contains dontcare label (dontcare=255)')
        parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
        parser.add_argument('--pv', type=bool, default=False, help='Whether we use partial volume maps or binary segs.')

        # for setting inputs
        parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data argumentation')
        parser.add_argument('--nThreads', default=0, type=int, help='# threads for loading data')
        parser.add_argument('--max_dataset_size', type=int, default=None, help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        parser.add_argument('--load_from_opt_file', action='store_true', help='load the options from checkpoints and use that as default')
        parser.add_argument('--cache_filelist_write', action='store_true', help='saves the current filelist into a text file, so that it loads faster')
        parser.add_argument('--cache_filelist_read', action='store_true', help='reads from the file list cache')
        parser.add_argument('--sequences', type = str, nargs ='*',  default =["T1", "FLAIR"],
                            help = "List of sequences that make up the dataset. Must coincide with the label. ")
        parser.add_argument('--fix_seq', type = str, default= None, help = "In case you want a fix modality to be picked up, specify the modality. ")

        parser.add_argument('--datasets', type = str, nargs ='*',
                            help = "List of datasets included in the images. Must coincide with the label. ")
        # for displays

        parser.add_argument('--display_winsize', type=int, default=400, help='display window size')

        # for generator
        parser.add_argument('--netG', type=str, default='spade', help='selects model to use for netG (pix2pixhd | spade)')
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        parser.add_argument('--init_type', type=str, default='xavier', help='network initialization [normal|xavier|kaiming|orthogonal]')
        parser.add_argument('--init_variance', type=float, default=0.02, help='variance of the initialization distribution')
        parser.add_argument('--z_dim', type=int, default=256,
                            help="dimension of the latent z vector")
        parser.add_argument('--n_decoders', type = str, default = "dontcare", help = 'Names of the decoder tails, with dashes in between: i.e. FLAIR-T1')
        parser.add_argument('--upsampling_type', type = str, default="upsample", help = "Type of convolution type: transposed upsample subpixel")

        # DATASET features

        parser.add_argument('--nef', type=int, default=16, help='# of encoder filters in the first conv layer')
        parser.add_argument('--use_vae', action='store_true', help='enable training with an image encoder.')
        parser.add_argument('--type_prior', default='N', help = 'Type of prior, S - spherical, uniform, or N - normal.')

        parser.add_argument('--skullstrip', action = 'store_true', help= "Whether to skull-strip or not the images.")
        parser.add_argument('--intensity_transform', type = bool, default = False, help = 'Activate '
                                                                                          'intensity augmentation transform')

        parser.add_argument('--dataset_type', type = str, default="volume", help = "Type of dataset, loading from volumes or slices."
                                                                                   "Can only be volume or sliced.")
        parser.add_argument('--cut', type=str, default = "a", help="Slice cut used (sagittal, coronal or axial")

        parser.add_argument('--cache_dir', type =str, default = None, help = "Directory where you can cache images"
                                                                     "and label volumes with PersistentDataset")
        parser.add_argument('--cache_type', type = str, default = "none", help = "If ram, CacheDataset is used (caching to RAM),"
                                                                               "if cache, PeristentDataset is used (caching to disk),"
                                                                                 "if none, Dataset is used (no caching)")
        parser.add_argument('--dataset_mode', type=str, default="custom", help = "Type of dataset. Can only be custom or"
                                                                                 "monai_dataset.")
        parser.add_argument('--diff_slice', action = 'store_true', help = "Use different slices as style input."
                                                                          "Both slices come from the same volume.")
        parser.add_argument('--diff_style_volume', action = 'store_true', help="Add another volume to the "
                                                                                  "persistent dataset volumes of"
                                                                                  "an image from a different style (dataset)."
                                                                               "The slices come from different images,"
                                                                               "and have, if can be, a different style.")
        parser.add_argument('--non_corresponding_dirs', action = 'store_true', help = "Whether the label directory"
                                                                                      "is different from the image directory."
                                                                                      "Since the files are not equivalent, "
                                                                                      "non_corresponding_style is True."
                                                                                      "You need to provide style_label_dir"
                                                                                      "for skullstrip mode.")
        parser.add_argument('--style_label_dir', type =str, default = None, help = "Directory leading to the labels of the "
                                                                                   "images (different from label_dir because"
                                                                                   "non_corresponding_dirs is True")
        parser.add_argument('--bound_normalization', action='store_true', help = "With this flag, the default normalisation"
                                                                                 "is replaced by a -1 to 1 normalisation.")

        self.initialized = True

        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, unknown = parser.parse_known_args()

        if opt.dataset_type not in ['sliced', 'volume']:
            ValueError("Dataset type can only be sliced or volume. ")

        # Modify dataset requirements according to type of dataset passed

        elif opt.dataset_type == 'volume':
            # Volumetric dataset - specific parameters.
            parser.add_argument('--intensify_lesions', type=int, default=None,
                                help="Every [THIS] iterations along the dataset,"
                                     "select slices with lesions.")
            parser.add_argument('--store_and_use_slices', type=bool, default=None, help="Whether you want to populate "
                                                                                        "selected slices and re-use the "
                                                                                        "exact numbers. ")
            parser.add_argument('--lesion_sampling_mode', type=str, default='threshold',
                                help="If you want the lesions to be sampled out of a single threshold or "
                                     "from a series of intervals controlling the amount of lesion each time."
                                     "Can only be: 'interval' or 'threshold'.")
            parser.add_argument('--threshold_low', type=float, default=100,
                                help='Number of minimum lesions required for threshold sampling of lesions')
            parser.add_argument('--ruleout_offset', type=float, default=0.00,
                                help="Percentage of image left out (from both"
                                     "sides) before sampling. Must be between"
                                     "0 and 0.4. ")
            parser.add_argument('--sample_lesions', type=bool, default=False,
                                help="Whether to sample only slices with lesions.")
            parser.add_argument('--continuous_slices', type=bool, default=False,
                                help="Whether to sample slices in an ordered way.")
        elif opt.dataset_type == 'sliced':
            parser.add_argument('--intensify_lesions', action = 'store_true',
                                help="Only select slices with lesions on them."
                                     "The lesion ID (or nolesion ID) must be in the filename."
                                     "FILE_<lesion-id>_<slice-no>.npz. If <lesion-id> is not"
                                     "nolesion, the slice will be considered."
                                     )
            if self.isTrain:
                parser.add_argument('--intensify_lesions_val', action='store_true',
                                    help="Only select slices with lesions on them. For validation."
                                         "The lesion ID (or nolesion ID) must be in the filename."
                                         "FILE_<lesion-id>_<slice-no>.npz. If <lesion-id> is not"
                                         "nolesion, the slice will be considered."
                                    )
        else:
            ValueError("Dataset can only be sliced or volumetric.")

        # modify model-related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)

        # # modify dataset-related parser options.
        # # Custom calls the modify_commandline_options in Custom Dataset via a function in init.py
        # # If monai_dataset, calls modify_commandline_options in monai_dataset_4_SPADE
        # if opt.dataset_mode == 'custom':
        #     dataset_option_setter = data.get_option_setter()
        #     parser = dataset_option_setter(parser, self.isTrain)
        # elif opt.dataset_mode == 'monai_dataset':
        #     parser = data.monai_dataset.monai_dataset_4_SPADE.MonaiDatasetForSPADE.modify_commandline_options(parser, self.isTrain)

        opt, unknown = parser.parse_known_args()

        # if there is opt_file, load it.
        # The previous default options will be overwritten
        if opt.load_from_opt_file:
            parser = self.update_options_from_file(parser, opt)
        opt = parser.parse_args()
        self.parser = parser

        # Overwrite some options
        opt.use_vae = True
        #opt.sequences = ["T1", "FLAIR"]

        if opt.n_decoders != "dontcare":
            opt.n_decoders = opt.n_decoders.split("-") # We split

        return opt

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def option_file_path(self, opt, makedir=False):
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        if makedir:
            util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt')
        return file_name

    def save_options(self, opt):
        file_name = self.option_file_path(opt, makedir=True)
        # Save previous options before
        if opt.continue_train and os.path.isfile(file_name+'.txt'):
            # Backup
            old_filename = file_name.replace('opt', 'opt_old')
            shutil.copy(file_name+'.txt', old_filename+'.txt')

        with open(file_name + '.txt', 'wt') as opt_file:
            for k, v in sorted(vars(opt).items()):
                comment = ''
                default = self.parser.get_default(k)
                if v != default:
                    comment = '\t[default: %s]' % str(default)
                opt_file.write('{:>25}: {:<30}{}\n'.format(str(k), str(v), comment))

        with open(file_name + '.pkl', 'wb') as opt_file:
            pickle.dump(opt, opt_file)

    def update_options_from_file(self, parser, opt):
        new_opt = self.load_options(opt)
        for k, v in sorted(vars(opt).items()):
            if hasattr(new_opt, k) and v != getattr(new_opt, k):
                new_val = getattr(new_opt, k)
                parser.set_defaults(**{k: new_val})
        return parser

    def load_options(self, opt):
        file_name = self.option_file_path(opt, makedir=False)
        new_opt = pickle.load(open(file_name + '.pkl', 'rb'))
        return new_opt

    def parse(self, save=False):

        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test

        self.print_options(opt)
        if opt.isTrain:
            self.save_options(opt)

        # Set semantic_nc based on the option.
        # This will be convenient in many places
        opt.semantic_nc = opt.label_nc + \
            (1 if opt.contain_dontcare_label else 0) + \
            (0)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        assert len(opt.gpu_ids) == 0 or opt.batchSize % len(opt.gpu_ids) == 0, \
            "Batch size %d is wrong. It must be a multiple of # GPUs %d." \
            % (opt.batchSize, len(opt.gpu_ids))

        if opt.phase == 'train' and opt.batchSize < 4:
            opt.latent_triplet_loss = False
            print("Triplet loss deactivated because batch size is insufficient.")

        self.opt = opt
        return self.opt
