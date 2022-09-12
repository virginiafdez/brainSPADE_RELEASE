import sys
import argparse
import os
import torch
import pickle

class ModiscOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('--name', type=str, default = "segnet", help='Name of the network')
        parser.add_argument('--checkpoints_dir', type=str, default='checkpoints', help='condldm_models are saved here')
        parser.add_argument('--batchSize', type=int, default=8, help='input batch size')
        parser.add_argument('--epoch_save', type=int, default=20,
                            help='Every how many epochs we save the network.'),
        parser.add_argument('--LR_init', type=float, default=0.01,
                            help='Height and width of the used image.')
        parser.add_argument('--LR_end', type=float, default=0.0001,
                            help="Learning rate at the end (low)")
        parser.add_argument('--num_epochs', type=int, default=500, help='Number of epochs.')
        parser.add_argument('--continue_train', action='store_true',
                            help='Load latest model.')
        parser.add_argument('--nThreads', type=int, default=4,
                            help='Number of workers.')
        parser.add_argument('--skullstrip', action='store_true',
                            help='Whether we skull strip the images or not.')
        parser.add_argument('--isTrain', action='store_true', help="Training or testing")
        parser.add_argument('--use_tboard', action='store_true',
                            help = 'Enables Tensorboard.')

        parser.add_argument('--sequences', type = str, nargs = '*', default = ['T1', 'FLAIR'],
                            help = 'List of modalities to classify!')
        parser.add_argument('--datasets', type = str, nargs = '*', help = 'List of modalities to classify!')
        parser.add_argument('--image_dir', type=str, required = True, help='Directory where the dataset is.')
        parser.add_argument('--label_dir', type=str, required=True, help='Directory where the dataset is.')
        parser.add_argument('--cache_dir', type=str, default=None, help="Directory where cache will be stored")
        parser.add_argument('--max_dataset_size', type=int, default=None,
                            help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        parser.add_argument('--new_size', type=int, nargs='*',
                            help='New height and width to resize the images to.')
        parser.add_argument('--image_dir_val', type=str, help='Directory where the dataset is.')
        parser.add_argument('--label_dir_val', type=str, help='Directory where the dataset is.')
        parser.add_argument('--max_dataloader', type=int,  help="Maximum number of images considered.")
        parser.add_argument('--cache_type', type=str, default="none",
                            help="If ram, CacheDataset is used (caching to RAM),"
                                 "if cache, PeristentDataset is used (caching to disk),"
                                 "if none, Dataset is used (no caching)")
        parser.add_argument('--bound_normalization', action='store_true', help = "With this flag, the default normalisation"
                                                                                 "is replaced by a -1 to 1 normalisation.")


        self.initialized = True

        return parser

    def gather_options(self):
        # Initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # Get the basic options
        opt, unknown = parser.parse_known_args()
        # Modify dataset requirements according to type of dataset passed
        parser.add_argument('--intensify_lesions', action = 'store_true',
                                help="Only select slices with lesions on them."
                                     "The lesion ID (or nolesion ID) must be in the filename."
                                     "FILE_<lesion-id>_<slice-no>.npz. If <lesion-id> is not"
                                     "nolesion, the slice will be considered."
                                     )
        if opt.isTrain:
            parser.add_argument('--intensify_lesions_val', action='store_true',
                                    help="Only select slices with lesions on them. For validation."
                                         "The lesion ID (or nolesion ID) must be in the filename."
                                         "FILE_<lesion-id>_<slice-no>.npz. If <lesion-id> is not"
                                         "nolesion, the slice will be considered."
                                    )
        opt = parser.parse_args()

        self.parser = parser

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
            os.makedirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt')
        return file_name

    def save_options(self, opt):
        if os.path.exists(os.path.join(opt.checkpoints_dir, opt.name)):
            file_name = self.option_file_path(opt, makedir=False)
        else:
            file_name = self.option_file_path(opt, makedir=True)
        with open(file_name + '.txt', 'wt') as opt_file:
            for k, v in sorted(vars(opt).items()):
                comment = ''
                default = self.parser.get_default(k)
                if v != default:
                    comment = '\t[default: %s]' % str(default)
                opt_file.write('{:>25}: {:<30}{}\n'.format(str(k), str(v), comment))

        with open(file_name + '.pkl', 'wb') as opt_file:
            pickle.dump(opt, opt_file)

    def load_options(self, opt):
        file_name = self.option_file_path(opt, makedir=False)
        new_opt = pickle.load(open(file_name + '.pkl', 'rb'))
        return new_opt

    def parse(self):

        opt = self.gather_options()

        # Modify values / add options
        opt.diff_slice = False
        opt.diff_style_volume = False
        opt.non_corresponding_dirs = False
        opt.intensity_transform = True
        opt.cut = 'a'
        opt.dataset_type = 'sliced'
        opt.fix_seq = None
        if opt.cache_dir is None:
            opt.cache_dir = os.path.join(opt.checkpoints_dir, opt.name)

        self.print_options(opt)
        self.save_options(opt)

        self.opt = opt
        return self.opt
