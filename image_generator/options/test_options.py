"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from .base_options import BaseOptions
import pickle

class TestOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--lesion', type=str, default = '', help = "Path to test the lesion included images")
        parser.add_argument('--mod_classifier', type = int, default = None, help = "Epoch of the modality classifier,"
                                                                             "stored in the checkpoints folder and"
                                                                             "named X_net_MD.pth")
        parser.set_defaults(preprocess_mode='scale_width_and_crop', crop_size=256, load_size=256, display_winsize=256)
        parser.set_defaults(serial_batches=True)
        parser.set_defaults(no_flip=True)
        parser.set_defaults(phase='test')
        parser.add_argument('--mod_disc_dir', type = str, default=None, help = "Optional path specified for the modality discriminator."
                                                                               "Otherwise, checkpoints is picked. ")
        parser.add_argument('--mode', type =str, default = 'eval', help = "Mode to put the network in. ")
        parser.add_argument('--guest_mods', type = str, nargs ='*', default = [],
                            help = "List of new modalities for the new_sequences test")
        parser.add_argument('--guest_images_dir', type=str, default = '',
                            help = "Guest-images directory. ")
        parser.add_argument('--guest_labels_dir', type=str, default = '',
                            help = "Guest-labels directory. ")

        self.isTrain = False
        return parser

class SamplingOptions(TestOptions):
    def initialize(self, parser):
        super(SamplingOptions, self).initialize(parser)
        parser.add_argument('--store_dir', type=str, help='Directory in which the images are to be stored.')
        parser.add_argument('--max_glevel', type=int,required = True, help = "Maximum granularity level")
        parser.add_argument('--parc_dir', type=str, help = "Root of all directories containing parcellation files."
                                                           "The parcellation files will be root_i, where"
                                                           "i is the granularity level.")
        parser.add_argument('--style_dir', type = str, nargs ='*', default = None,
                            help = "List of style directories: either it's one folder containing the"
                                   "style images, or a folder per modality.")
        parser.add_argument('--style_dir_mask', type=str, help = "Path to the mask volumes "
                                                                         "corresponding to the style images.")
        parser.add_argument('--skip_gt', default = False, type = bool, help = "If true, ground truth images"
                                                                              "aren't saved.")
        parser.add_argument('--store_root_gt', default = 'gt', type = str,
                            help = "Root of the ground truth images, to which modality names will be appended.")
        parser.add_argument('--store_root_lab', default = 'labels', type = str,
                            help = "Root of the ground truth images, to which granularity levels will be appended.")
        parser.add_argument('--store_same', default = False, type = bool,
                            help = "Whether the ground truth images are separated into different folders or not.")
        parser.add_argument('--use_mod_disc', default = False, type = bool,
                            help = "Whether to use the modality ")
        parser.add_argument('--mod_disc_th', type = float, default = 0.85,
                            help = "Accuracy threshold at which we consider the resulting image to be accurate")
        parser.add_argument('--mod_disc_attempts', type = int, default = 50,
                            help = "Number of attempts until modality discrimination constraint is applied.")
        parser.add_argument('--allowed_styles', type = str, nargs ='*', default = None,
                            help = "List of styles that you want ot consider for the style images."
                                   "Must be embedded in the image files name.")
        parser.add_argument('--sample_from', type = str, default = None, help = "Directory from which you want to base new slices."
                                                                                "Must be a directory with png files (NAME_slice.png")
        parser.add_argument('--lesion_ids_ind', type = int, nargs = '*', default = [], help = "Indices of the channels of the PV maps of the lesions"
                                                                                               "that need to be saved")
        parser.add_argument('--lesion_ids_names', type = str, nargs = '*', default = [], help = "Names of the lesions corresponding to those indices")

        return parser

class FullPipelineOptions(BaseOptions):

    def load_options(self, file):

        new_opt = pickle.load(open(file + '.pkl', 'rb'))

        #Remove training parameters
        new_opt.isTrain = False
        new_opt.semantic_nc = new_opt.label_nc + \
            (1 if new_opt.contain_dontcare_label else 0) + \
            (0)
        # set gpu ids
        str_ids = new_opt.gpu_ids.split(',')
        new_opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                new_opt.gpu_ids.append(id)

        assert len(new_opt.gpu_ids) == 0 or new_opt.batchSize % len(new_opt.gpu_ids) == 0, \
            "Batch size %d is wrong. It must be a multiple of # GPUs %d." \
            % (new_opt.batchSize, len(new_opt.gpu_ids))

        if new_opt.phase == 'train' and new_opt.batchSize < 4:
            new_opt.latent_triplet_loss = False
            print("Triplet loss deactivated because batch size is insufficient.")

        return new_opt

