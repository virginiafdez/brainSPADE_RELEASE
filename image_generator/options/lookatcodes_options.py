""""""

from .base_options import BaseOptions


class LACodeOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        parser.add_argument('--savecode_dir', type=str, default='./savecode_dir/', help='saves results here.')
        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--styles', nargs = '*', default=[], type = str,help='List of the different styles')
        parser.add_argument('--lesion', type=str, default = '', help = "Path to test the lesion included images")
        parser.add_argument('--whole_volume', type=str, default='', help="Path to test whole 3D volume nii file")
        parser.add_argument('--do_reduction', action='store_true',  help="Flag. Run reduction test.")
        parser.add_argument('--do_heatmaps', action='store_true', help="Flag. Do heatmaps removing each element of"
                                                                         "code.")
        parser.add_argument('--do_sparsity', action='store_true', help="Flag. Calculate mean sparsity of vectors"
                                                                         "(T1 and FlAIR and both)")
        parser.add_argument('--reduction_algorithm', type = str, default = 'pca', help = 'tsne or pca')
        parser.add_argument('--do_volume', action='store_true', help = "Plot codes equivalent to whole volume")
        parser.add_argument('--do_new_seq', action='store_true', help = "Flag. Do code plots with additional sequences.")
        parser.add_argument('--guest_mods', type=str, nargs='*', default = [], help = 'List of additional modalities')
        parser.add_argument('--guest_images_dir', type=str, default="", help='List of image directory with guest modalities')
        parser.add_argument('--guest_labels_dir', type=str, default="", help='List of label directory with guest modalities')
        parser.add_argument('--guest_label', type=str, default="guest",
                            help='Label identifying the "guest" dataset')
        parser.set_defaults(serial_batches=True)
        parser.set_defaults(phase='test')
        parser.add_argument('--mod_disc_dir', type = str, default=None, help = "Optional path specified for the modality discriminator."
                                                                               "Otherwise, checkpoints is picked. ")
        parser.add_argument('--mode', type =str, default = 'eval', help = "Mode to put the network in. ")
        parser.add_argument('--reduction_itemizing', type=bool, default=False,
                            help="In the reduction task, save individual codes plots + images with the image code colored.")

        self.isTrain = False
        return parser
