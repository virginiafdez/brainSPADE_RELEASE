"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        # Dataset
        parser.add_argument('--label_dir_val', type=str, required=False, default = '',
                            help='path to the directory that contains label images for validation.')
        parser.add_argument('--image_dir_val', type=str, required=False, default = '',
                            help='path to the directory that contains photo images for validation.')
        parser.add_argument('--non_continuous_slices', type = bool, default = False, help = "If true, processes"
                                                                                            "a whole volume - all the "
                                                                                            "slices. ")
        parser.add_argument('--n_passes', type = int, default = 1, help = "Number of passes through a volume dataset.")


        # for displays
        parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
        parser.add_argument('--test_freq', type=int, default=1, help = 'frequency of testing resuts (in epochs)')
        parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        parser.add_argument('--debug', action='store_true', help='only do one epoch and displays at each iteration')
        parser.add_argument('--tf_log', action='store_true', help='if specified, use tensorboard logging. Requires tensorflow installed')
        parser.add_argument('--print_grad_freq', type = int, default = 5000, help = 'Gradient norm of the modality discrimination'
                                                                                    'layers.')
        parser.add_argument('--disp_D_freq', type = int, default = 600, help = "Frequency of display of discriminator outputs.")
        parser.add_argument('--use_tboard', action = 'store_true', help = "Use tensorboard Summary Writer. Warning: can te a lot of memory!")
        parser.add_argument('--tboard_gradients', action='store_true',
                            help="Plot gradients on tensorboard")
        parser.add_argument('--tboard_activations', action='store_true',
                            help="Plot activations on tensorboard")


        # Training schedule
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--niter', type=int, default=50, help='# of iter at starting learning rate. This is NOT the total #epochs. Totla #epochs is niter + niter_decay')
        parser.add_argument('--niter_decay', type=int, default=0, help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--optimizer', type=str, default='adam')
        parser.add_argument('--beta1', type=float, default=0.0, help='momentum term of adam')
        parser.add_argument('--beta2', type=float, default=0.9, help='momentum term of adam')
        parser.add_argument('--no_TTUR', action='store_true', help='Use TTUR training scheme')
        parser.add_argument('--TTUR_factor', type = int, default = 2,
                            help = "Factor of downgrading LR of generator wr to discriminator.")
        parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=10, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--save_epoch_copy', type=int, default=10,
                            help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--batch_acc_fr', type=int, default = 0, help="Frequency of batch accumulation.")
        parser.add_argument('--display_enc_freq', type=int, default = 100, help="Frequency (in iterations) at which "
                                                                           "we save codes.")
        parser.add_argument('--gradients_freq', type = int, default = None,
                            help = "Frequency of calculation of gradients of the losses wr to last layers of"
                                   "the network. If None, they are not calculated at all.")
        parser.add_argument('--activations_freq', type = int, default = None,
                            help = "Frequency of calculation of gradients of the losses wr to last layers of"
                                   "the network. If None, they are not calculated at all.")
        parser.add_argument('--steps_accuracy', type = int, default = 20, help="Number of iterations on which to base the"
                                                                               "accuracy calculation. ")

        # the default values for beta1 and beta2 differ by TTUR option
        opt, _ = parser.parse_known_args()
        if opt.no_TTUR:
            parser.set_defaults(beta1=0.5, beta2=0.999)

        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')


        # Discriminator
        parser.add_argument('--D_steps_per_G', type=int, default=1, help='number of discriminator iterations per generator iterations.')
        parser.add_argument('--D_modality_class', action='store_true', help = "Flag that incorporates the modality to the "
                                                                              "discriminator input. ")
        parser.add_argument('--topK_discrim', action='store_true', help = 'Use a Top-K filtering to train the discriminator'
                                                                          'and the generator.')
        parser.add_argument('--drop_first', action = 'store_true', help = "Drop first discriminator (smaller receptive field)"
                                                                          "when calculating the loss. ")
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        parser.add_argument('--netD', type=str, default='multiscale', help='(n_layers|multiscale|image)')

        # Modality discrimination loss
        parser.add_argument('--mod_disc', action='store_true', help = "Whether Modality Discrimination"
                                                                               " loss is included. ")
        parser.add_argument('--mod_disc_path', type=str, default=None, help="Path to modality discriminator network.  ")
        parser.add_argument('--mod_disc_epoch', type=int, default=10)
        parser.add_argument('--mod_disc_dir', type = str, default=None, help = "Optional path specified for the modality discriminator."
                                                                               "Otherwise, checkpoints is picked. ")
        parser.add_argument('--train_modisc', type = bool, default = False, help = "Whether to make the modality discriminator"
                                                                                  "trainable or not.")

        # Generator losses (no modality discriminator) and encoder

        parser.add_argument('--lambda_feat', type=float, default=10.0, help='weight for feature matching loss')
        parser.add_argument('--lambda_vgg', type=float, default=10.0, help='weight for vgg loss')
        parser.add_argument('--lambda_slice_consistency', type=float, default=0.0, help = 'Weight given to slice'
                                                                                          'by slice consistency.')
        parser.add_argument('--activation_slice_consistency', type=int, default=100, help = "epoch in which you "
                                                                                            "want slice consistency"
                                                                                            "to get activated")
        parser.add_argument('--no_ganFeat_loss', action='store_true', help='if specified, do *not* use discriminator '
                                                                           'feature matching loss')
        parser.add_argument('--no_vgg_loss', action='store_true', help='if specified, do *not* use VGG feature matching loss')
        parser.add_argument('--gan_mode', type=str, default='hinge', help='(ls|original|hinge)')

        parser.add_argument('--lambda_kld', type=float, default=0.05)


        parser.add_argument('--train_enc_only', type =int, default = None, help="From this epoch onwards, train only"
                                                                                 "the encoder (and not the decoder)")

        parser.add_argument('--nullDecoderLosses', type = bool, default = False, help = 'Whether to train only'
                                                                                        'on the encoder losses.')

        parser.add_argument('--disc_acc_lowerth', type = float, default = 0.65, help = "Accuracy (0-1) of the discriminator"
                                                                                       "below which discriminator only is trained.")
        parser.add_argument('--disc_acc_upperth', type = float, default = 0.85, help = "Accuracy (0-1) of the discriminator"
                                                                                       "above which generator only is trained.")
        parser.add_argument('--lambda_mdmod', type=float, default=2.5, help="Weight given to modality discrimination loss.")
        parser.add_argument('--lambda_mddat', type=float, default=1.0, help="Weight given to dataset discrimination loss.")
        parser.add_argument('--pretrained_E', type=str, default=None, help="Path to pretrained encoder if needed.")
        parser.add_argument('--freezeE', type = bool, default = False, help = "Freeze weights of the encoder.")
        parser.add_argument('--self_supervised_training', type=float, default=0.0,
                            help="Weight given to self-supervised loss. ")
        parser.add_argument('--distance_metric', type=str, default='l1',
                            help="Distance metric used for the self-supervised loss ")

        self.isTrain = True
        return parser
