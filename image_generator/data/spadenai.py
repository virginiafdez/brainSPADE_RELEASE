import monai as monai
import monai.transforms as mt
import torch
import numpy as np
import os
import random
import torchvision.transforms.functional as FTV
from data.monai_dataset.custom_transforms import Slicer, CropPad
import moreutils as uvir
from monai.data.dataset import PersistentDataset

class SpadeNai(torch.utils.data.Dataset):

    def __init__(self, opt, mode):
        '''
        Creates an instance of SpadeNai dataset.
        :param opt: Opt (from argparse). Must contain at least the following elements: image_dir and label_dir,
        label_dir_val and image_dir_val if mode = 'validation'; fix_seq; cut; batchSize; datasets (list with
        the different datasets involved); cache_dir or checkpoints_dir+name to define the directory where the nii
        images will be cached; n_passes (number of passes through volume dataset before considering accomplishment
        of 1 pass through 2D dataset); non_continuous_slices (boolean: False if 1 volume = 1 slice, True if 1 volume
        = all the volume slices); new_size (array containing the HxW shape desired for the input to the network).
        :param mode: str, 'train' 'test' or 'validation' (it's only different if it's validation or not).
        '''

        self.opt = opt
        if mode == 'validation':
            self.image_dir = opt.image_dir_val
            self.label_dir = opt.label_dir_val
        else:
            self.image_dir = opt.image_dir
            self.label_dir = opt.label_dir

        self.cut_ids = {'a': 2, 's': 0, 'c': 1}  # Cut (a: axial, s: sagittal, c: coronal)
        self.cut = opt.cut
        self.selected_index = self.cut_ids[self.cut] # Index corresponding to the selected cut.

        self.epochs = opt.niter

        if opt.fix_seq is not None:
            self.modalities = [opt.fix_seq]
            self.fix_seq = True
        else:
            self.modalities = opt.sequences
        self.datasets = opt.datasets
        self.batchSize = opt.batchSize
        if opt.cache_dir is None:
            self.cache_dir = os.path.join(opt.checkpoints_dir, opt.name, "spadenai_cache_dir")
        else:
            self.cache_dir = os.path.join(opt.cache_dir, "spadenai_cache_dir")
        if not os.path.isdir(self.cache_dir):
            os.makedirs(self.cache_dir)

        # Volume dataset
        self.input_vol_dict = self.monaifyInput()
        self.volumeDataset = PersistentDataset(self.input_vol_dict, self.multipleVolumesLoader(),
                                                                  cache_dir=self.cache_dir)
        self.loader = torch.utils.data.DataLoader(self.volumeDataset, shuffle = False,
                                                  batch_size = 1, num_workers = self.opt.nThreads)
        self.iterator = iter(self.loader)

        self.ctr = {'current': 0, 'max': len(self.volumeDataset)}

        self.n_passes = opt.n_passes
        self.non_corresponding_slices = opt.non_corresponding_slices

        # Transforms
        if opt.intensity_transform and (mode == 'train' or mode == 'default'):
            self.transform = self.IntensityTransform(prob_bias_field=0.5,
                                                     prob_rand_adjust= 0.5,
                                                     prob_gaussian_noise=0.05)
        else:
            self.transform = None
        pad_size = opt.new_size
        pad_size.append(-1)
        self.padder = CropPad(pad_size)
        self.slicer = Slicer(cut = self.cut, output_channels=1,
                             non_corresponding=opt.non_corresponding_slices)

        self.skullstrip = opt.skullstrip
        self.diff_slices = opt.diff_slice

    def monaifyInput(self):

        '''
        Prepares an alternating dictionary containing the image and label paths in a dictionary to be passed
        to a Monai-like loader.

        :return:
        '''

        all_images = {}
        counters = {}
        length_all = 0
        for mod in self.modalities:
            all_images[mod] = [i for i in os.listdir(self.image_dir) if i.split("_")[0] == mod]
            random.shuffle(all_images[mod])
            counters[mod] = 0
            length_all += len(all_images[mod])

        out = []
        counter_mod = 0

        while length_all > 0:
            chosen_mod = self.modalities[counter_mod]
            image_name = all_images[chosen_mod][counters[chosen_mod]]
            label_name = image_name.replace(chosen_mod, "Parcellation", 1)

            # Extract additional metadata
            metadata = np.load(os.path.join(self.image_dir, image_name))

            out.append({'image_file': os.path.join(self.image_dir, image_name),
                        'label_file': os.path.join(self.label_dir, label_name),
                        'affine': metadata['img_affine'],
                        'modality': chosen_mod,
                        'dataset': str(metadata['dataset_name'])})

            counters[chosen_mod] += 1
            counter_mod += 1
            if counters[chosen_mod] >= len(all_images[chosen_mod]):
                counters[chosen_mod] = 0
            if counter_mod == len(self.modalities):
                counter_mod = 0

            length_all -= 1

        return out

    def multipleVolumesLoader(self):

        '''
        Defines a compose loader for images and labels.
        Returns the loader.
        :return:
        '''

        image_loader = mt.LoadImageD(
            keys=('image_file'),
            reader = "numpyreader",
            npz_keys = ('img',)
        )

        label_loader = mt.LoadImageD(
            keys=('label_file'),
            reader = "numpyreader",
            npz_keys = ('label',)
        )

        both_loaders = mt.Compose([image_loader, label_loader])

        return both_loaders

    def __len__(self):

        return self.ctr['max']*self.n_passes

    def __getitem__(self, item):

        # We retrieve the next volume
        volumes = next(self.iterator)

        # Pad label and image to the unique input size
        label = volumes['label_file'].squeeze(0) # Squeeze batch dimension
        image = volumes['image_file'].squeeze(0) # Squeeze batch dimension
        label_vol = self.padder(label, islabel = True) # Move channels first as if they were a batch
        image_vol = self.padder(image, islabel = False)

        # Slice
        label, image, slices = self.slicer(label_vol, image_vol)

        if self.diff_slices:
            label_style, style, st_slices = self.slicer(label_vol, image_vol)
            label_style = label_style.permute(2, 0, 1)  # Channels on top
            style_slice = st_slices['image']
        else:
            style = image
            style_slice = slices['image']

        # Put channels first and transform
        image = image.permute(2, 0, 1) # Channels on top
        label = label.permute(2, 0, 1) # Channels on top
        style = style.permute(2, 0, 1) # Channels on top

        if self.transform is not None:
            s = np.random.random()
            temp_transform = self.transform.set_random_state(s)
            image = temp_transform(image.numpy())
            style = temp_transform(style.numpy())
        else:
            image = image.numpy()
            style = style.numpy()

        image = self.normalize(torch.from_numpy(image))
        style = self.normalize(torch.from_numpy(style))

        if self.skullstrip:
            image = uvir.SkullStrip(image, label, value=-1)
            style = uvir.SkullStrip(style, label_style, value = -1)

        # Build dataset output
        out_dict = {}
        out_dict['image'] = image.type('torch.FloatTensor')
        out_dict['style_image'] = style.type('torch.FloatTensor')
        out_dict['label'] = label.type('torch.FloatTensor')
        out_dict['this_seq'] = self.input_vol_dict[self.ctr['current']]['modality']
        out_dict['this_dataset'] = self.input_vol_dict[self.ctr['current']]['dataset']
        out_dict['slice_no'] = slices['label']
        out_dict['slice_style_no'] = style_slice
        out_dict['image_path'] = self.input_vol_dict[self.ctr['current']]['image_file']
        out_dict['label_path'] = self.input_vol_dict[self.ctr['current']]['label_file']
        out_dict['image_slice_no'] = slices['image']

        # Update the loader if needed
        self.ctr['current'] += 1
        if self.ctr['current'] == self.ctr['max']:
            self.ctr['current'] = 0
            self.loader = torch.utils.data.DataLoader(self.volumeDataset, shuffle = False,
                                                      batch_size = 1, num_workers = self.opt.nThreads)
            self.iterator = iter(self.loader)

        return out_dict

    def IntensityTransform(self, prob_bias_field = 1.0, prob_rand_adjust = 1.0, prob_gaussian_noise = 1.0):
        '''
        Returns Monai Compose of Random Bias Field, Random Adjust Contrast and Random Gaussian Noise.
        :return:
        '''

        return monai.transforms.Compose(
            [monai.transforms.RandBiasField(coeff_range=(1, 1.2), prob = prob_bias_field),
             monai.transforms.RandAdjustContrast(gamma=(0.87, 1.3), prob=prob_rand_adjust),
             monai.transforms.RandGaussianNoise(prob=prob_gaussian_noise, mean=0.0, std=np.random.uniform(0.005, 0.015))]
        )

    def SlicerTransform(self, crop_size, samples):

        '''
        Returns a composed slicer transform. First, we pad the input volumes to ensure that the
        input size is the same. Then, slice samples from the volume.
        :param crop_size: [HxW] list with the desired input size to the algorithm
        :param samples: Number of slices to be sampled.
        :return:
        '''

        pad_size = crop_size # Corresponding to slice dimension (we don't care about those)
        pad_size.append(-1)
        crop_size.append(1)
        composed_tf = monai.transforms.Compose([
            monai.transforms.SpatialPad([crop_size, -1]),
            monai.transforms.RandSpatialCropSamples(roi_size=crop_size,
                                                    num_samples=samples,
                                                    random_size=False)
        ])
        return composed_tf

    def normalize(self, tensor, standardize = True):

        means_ = (0.5, 0.5, 0.5)
        stds_ = (0.5, 0.5, 0.5)

        if tensor.min() == tensor.max():
            if standardize:
                return FTV.normalize(tensor, means_, stds_, True)
            else:
                return tensor
        else:
            normalized = (tensor - tensor.min()) / (tensor.max() - tensor.min())
            if standardize:
                return FTV.normalize(normalized, means_, stds_, True)
            else:
                return normalized
