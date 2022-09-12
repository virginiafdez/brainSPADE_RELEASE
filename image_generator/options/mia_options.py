from .base_options import BaseOptions
import pickle
import sys
import argparse
import os
from util import util
import torch
import models
import data
import pickle
import shutil
from options.unparsed_options import UnparsedOptions

class MiaOptions():

    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        # "victim" options
        parser.add_argument('--path_to_victim_opt', type = str, required = True,
                            help = "Path to the file where the options for the brainSPADE network are.")
        # Main options
        parser.add_argument('--num_epochs', type = int, default = 100, help = "Number of training epochs.")
        parser.add_argument('--isTrain', action = 'store_true', help ="Whether we are in training mode. ")
        parser.add_argument('--name', type=str, default='label2coco', help='name of the experiment. It decides where to store samples and models_')
        parser.add_argument('--epochs_save', type = int, default = 20, help = "Every when to save the networks separately from latest.")
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='checkpoints', help='models_ are saved here')
        # Mia network options
        parser.add_argument('--mia_out_channels', type = int, default=1, help = "Number of channels that come out from the network")
        parser.add_argument('--mia_channels', type = int, nargs='*', default=[256, 128, 16, 8,1], help = "Number of channels that go in-out each level of the network."
                                                                                                      "")
        parser.add_argument('--mia_strides', type = int, nargs='*', default=[2, 2, 2, 2, 2], help = "Strides for each level of the network. Length must be"
                                                                                                 "equal to that of mia_channels.")

        parser.add_argument('--mia_num_res_units', type = int, default=2, help = "Number of residual units per level."
                                                                                 "")
        # Learning rate options.
        parser.add_argument('--lr_decay', action = 'store_true', help = 'Whether to use decay or not for '
                                                                         'the learning rate.')
        parser.add_argument('--g_lr', type = float, default = 0.0001, help = "Starting learning rate for the generator.")
        parser.add_argument('--d_lr', type = float, default = 0.0001, help = "Starting learning rate for the discriminator.")
        parser.add_argument('--g_lr_end', type = float, default = 0.0001, help = "End learning rate for the generator.")
        parser.add_argument('--d_lr_end', type = float, default = 0.0001, help = "End learning rate for the discriminator.")

        # Weights options
        parser.add_argument('--use_gan_loss', action = 'store_true', help = "Whether to use GAN loss.")
        parser.add_argument('--use_vgg_loss', action = 'store_true', help="Whether to use VGG loss.")
        parser.add_argument('--gan_loss_w', type=float, default=0.01, help = "Weight to give to the GAN Loss")
        parser.add_argument('--vgg_loss_w', type=float, default=0.01, help="Weight to give to the VGG Loss")
        parser.add_argument('--rec_loss_w', type=float, default=1.0, help="Weight to give to the reconstruction Loss")

        # Dataset options.
        parser.add_argument('--image_dir', type = str, required = True, help = "Path to image directory")
        parser.add_argument('--label_dir', type=str, required=True, help="Path to image directory")
        parser.add_argument('--image_dir_val', type = str, required = True, help = "Path to image directory")
        parser.add_argument('--label_dir_val', type=str, required=True, help="Path to image directory")
        parser.add_argument('--cache_dir', type = str, default=None, help = "Path to cache directory."
                                                                            )
        parser.add_argument('--cache_disk', action = 'store_true', help = "Whether to cache the files to disk, not to RAM.")
        parser.add_argument('--nThreads', type = int, default=0, help = "Number of workers for dataloader")
        parser.add_argument('--max_dataset_size', type = int, default=None, help = "Maximum dataset size.")
        parser.add_argument('--intensity_transform', action = 'store_true', help = "Whether to augment or not the data")
        parser.add_argument('--continue_train', action = 'store_true', help = "Continue training flag")
        parser.add_argument('--batchSize', type = int, default = 6, help = "Batch size")
        parser.add_argument('--dataset_type', type=str, default='sliced', help="Type of dataset")

        # Discriminator
        parser.add_argument('--netD_subarch', type=str, default='n_layer',
                            help='architecture of each discriminator')
        parser.add_argument('--num_D', type=int, default=2,
                            help='number of discriminators to be used in multiscale')
        parser.add_argument('--n_layers_D', type=int, default=4,
                            help='# layers in each discriminator')
        parser.add_argument('--ndf', type=int, default=64,
                            help='# layers in each discriminator')
        parser.add_argument('--D_modality_class', action = 'store_true',
                            help='# Whether to use the modality IDs to help the discriminator.')
        parser.add_argument('--norm_D', type=str, default='spectralinstance',
                            help='instance normalization or batch normalization')
        parser.add_argument('--no_ganFeat_loss', action = 'store_true', help = "Whether to return intermediate "
                                                                               "features of discriminator.")
        parser.add_argument('--G_per_D', type = int, default = 1, help = "Number of generator steps per discriminator steps")
        self.initialize = True
        self.isTrain = False
        return parser

    def parse(self, save=True):

        opt = self.gather_options()
        opt.isTrain = self.isTrain

        # Some values, hard set
        opt.diff_slice = False
        opt.diff_style_volume = False
        opt.non_corresponding_dirs = False
        opt.style_label_dir = None

        self.print_options(opt)
        if opt.isTrain:
            self.save_options(opt)
        self.opt = opt

        if save:
            self.save_options(self.opt)

        return self.opt

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, unknown = parser.parse_known_args()

        # Load unparsed dataset
        parser.add_argument("--victim_options", default=UnparsedOptions(opt.path_to_victim_opt).initialize())
        opt, unknown = parser.parse_known_args()
        opt.no_ganFeat_loss = True

        # Add a certain number of variables from victim options to main options
        args_of_inters = ['model', 'norm_G', 'norm_E', 'new_size', 'label_nc',
                          'contain_dontcare_label', 'output_nc', 'pv', 'no_flip',
                          'sequences', 'datasets',  'netG', 'ngf', 'init_type',
                          'init_variance', 'z_dim', 'upsampling_type', 'nef',
                          'use_vae', 'type_prior', 'skullstrip', 'cut', 'dataset_mode',
                          'preprocess_mode', 'fix_seq',
                          ]

        for item, value in dict(opt.victim_options.__dict__.items()).items():
            if item in args_of_inters:
                parser.add_argument("--%s" %item, default = value)

        opt, unknown = parser.parse_known_args()
        parser.add_argument('--mia_in_channels', type = int, default = 16 * opt.ngf)
        opt, unknown = parser.parse_known_args()

        if len(opt.mia_channels) != len(opt.mia_strides):
            ValueError("Length of channels must equal that of strides.")

        opt, unknown = parser.parse_known_args()

        if opt.cache_dir is None:
            opt.cache_dir = os.path.join(opt.checkpoints_dir, opt.name, 'cache_dir')
            if not os.path.isdir(opt.cache_dir):
                os.makedirs(opt.cache_dir)

        # If there is opt_file, load it.
        # The previous default options will be overwritten
        if opt.continue_train:
            parser = self.update_options_from_file(parser, opt)

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
            util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt')
        return file_name

    def save_options(self, opt):
        file_name = os.path.join(opt.checkpoints_dir, opt.name, 'opt')
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
        file_name = os.path.join(opt.checkpoints_dir, opt.name, 'opt')
        new_opt = pickle.load(open(file_name + '.pkl', 'rb'))
        return new_opt

