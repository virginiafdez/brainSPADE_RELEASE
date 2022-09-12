"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from .base_options import BaseOptions
import pickle

class MemorisingOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        parser.add_argument('--image_dir_base', type=str,  help='directory with the original images.')
        parser.add_argument('--label_dir_base', type=str, help='directory with the original labels')
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.set_defaults(preprocess_mode='scale_width_and_crop', crop_size=256, load_size=256, display_winsize=256)
        parser.set_defaults(serial_batches=True)
        parser.set_defaults(no_flip=True)
        parser.set_defaults(phase='test')
        parser.add_argument('--no_style_test', action='store_true', help = "Run a test using random style vector")
        parser.add_argument('--style_test', action = 'store_true', help = "Run a test using style vector from input image.")
        parser.add_argument('--max_n_ims', type = int, default = 100, help = "Max number of training images for which this is computed.")
        self.isTrain = False
        return parser
