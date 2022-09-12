'''
Similar to SpadeNai Version 1, this loads NPZ files to use with Persistent and Iterable dataset from
MONAI to train a brainSPADE project.
Contrarily to Spadenai, SpadeNaiSlice loads pre-saved slices, so no slicing process is handled here.
'''
import monai as monai
import monai.transforms as mt
import torch
import numpy as np
import os
import random
import pickle
import torchvision.transforms.functional as FTV
from data.custom_transforms import Slicer, CropPad
import moreutils as uvir
from monai.data.dataset import PersistentDataset, CacheDataset,Dataset
#from monai.data.dataset import PersistentDataset, CacheDataset
from monai.data.iterable_dataset import IterableDataset
from copy import deepcopy
import shutil
import nibabel as nib
from custom_packages.custom_loaders import LoadNPZ

class SpadeNaiSlice():

    def __init__(self, opt, mode, **kwargs):

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

        # Mode
        if mode == 'validation':
            self.image_dir = opt.image_dir_val
            self.label_dir = opt.label_dir_val
        else:
            self.image_dir = opt.image_dir
            self.label_dir = opt.label_dir

        # Necessary to specify the slicer transform
        self.cut_ids = {'a': 2, 's': 0, 'c': 1}  # Cut (a: axial, s: sagittal, c: coronal)
        self.cut = opt.cut

        # Whereas we have one or multiple modalities.
        if opt.fix_seq is not None:
            self.modalities = [opt.fix_seq]
            self.fix_seq = True
        else:
            self.modalities = opt.sequences
            self.fix_seq = False

        self.datasets = opt.datasets
        self.batchSize = opt.batchSize
        self.max_dataset_size = opt.max_dataset_size

        #Cache_dir for peristent dataset
        if opt.cache_dir is None:
            self.cache_dir = os.path.join(opt.checkpoints_dir, opt.name, "spadenai_cache_dir")
        else:
            self.cache_dir = os.path.join(opt.cache_dir, "spadenai_cache_dir")
        if not os.path.isdir(self.cache_dir):
            os.makedirs(self.cache_dir)

        self.non_corresponding_slices = opt.diff_slice or opt.diff_style_volume
        self.diff_slices = opt.diff_slice # Look for same mother volume, different slices.
        self.diff_style_volume = opt.diff_style_volume
        self.non_corresponding_dirs = opt.non_corresponding_dirs

        # Transforms
        if opt.intensity_transform and (mode == 'train' or mode == 'default'):
            self.transform = self.IntensityTransform(prob_bias_field=0.5,
                                                     prob_rand_adjust= 0.5,
                                                     prob_gaussian_noise=0.05)
        else:
            self.transform = None

        # Whether to only use slices with lesions
        if mode == 'validation':
            self.intensify_lesions = {'flag': opt.intensify_lesions_val,
                                      }
        else:
            self.intensify_lesions = {'flag': opt.intensify_lesions,
                                      }

        # New 2D size
        pad_size = opt.new_size
        pad_size.append(-1)
        self.new_size = opt.new_size

        # Padder
        self.padder = CropPad(pad_size)

        # Skull strip
        self.skullstrip = opt.skullstrip
        if self.non_corresponding_dirs and not opt.isTrain:
            # Not available for train mode
            self.style_label_dir = opt.style_label_dir
            self.non_corresponding_slices = True
            # Diff slice and diff_style_volume are not important in this case.

        # Persistent Dataset for the volumes
        # We need a list of dictionary inputs with the image path, label path etc.
        if self.fix_seq:
            if self.non_corresponding_dirs and not opt.isTrain:
                self.input_vol_dict = self.monaifyInput_non_corr(self.modalities[0])
            else:
                self.input_vol_dict = self.monaifyInput(self.modalities[0])
        else:
            if self.non_corresponding_dirs and not opt.isTrain:
                self.input_vol_dict = self.monaifyInput_non_corr()
            else:
                self.input_vol_dict = self.monaifyInput()

        # Creation of Persistent Dataset.
        if opt.cache_type=='cache':
            self.volumeDataset = PersistentDataset(self.input_vol_dict, self.multipleVolumesLoader(),
                                                   cache_dir=self.cache_dir,
                                                   pickle_protocol=pickle.HIGHEST_PROTOCOL)

        elif opt.cache_type=='none':
            self.volumeDataset = Dataset(self.input_vol_dict, self.multipleVolumesLoader())
        elif opt.cache_type=='ram':
            self.volumeDataset = CacheDataset(self.input_vol_dict, self.multipleVolumesLoader())

        self.ctr = {'current': 0, 'max': len(self.volumeDataset)}

        # Iterable dataset for the slices
        self.sliceDataset = IterableDataset(self.getIteratorFun(self.volumeDataset))


        # These parameter dont't have sense with pre-loaded slices, it's used in volume SpadeNai.
        # but we set it to False for compatibility with
        # other scripts.
        self.store_slices = False


    def getIteratorFun(self, dataset):

        for volumes in dataset:
            # Pad label and image to the unique input size
            if isinstance(volumes['label_file'], monai.data.meta_tensor.MetaTensor):
                volumes['label_file'] = np.asarray(volumes['label_file'])
            if isinstance(volumes['image_file'], monai.data.meta_tensor.MetaTensor):
                volumes['image_file'] = np.asarray(volumes['image_file'])
            if 'style_file' in volumes.keys():
                if isinstance(volumes['style_file'], monai.data.meta_tensor.MetaTensor):
                    volumes['style_file'] = np.asarray(volumes['style_file'])
            if 'style_label_file' in volumes.keys():
                if isinstance(volumes['style_label_file'], monai.data.meta_tensor.MetaTensor):
                    volumes['style_label_file'] = np.asarray(volumes['style_label_file'])
            label = torch.from_numpy(volumes['label_file'])[..., 0, :]
            image = torch.from_numpy(volumes['image_file'])
            if image.shape[2] == 3:
                image = image[..., 0:1]

            # Pad to fit the self.size
            label = self.padder(label, islabel = True)
            image = self.padder(image, islabel = False)

            if self.non_corresponding_slices:
                # That is: if non_corresponding dirs, or diff_slices or diff_style_volume.
                style = torch.from_numpy(volumes['style_file'])
                label_style = torch.from_numpy(volumes['style_label_file'])[..., 0, :]
                style = self.padder(style, islabel=False)
                label_style = self.padder(label_style, islabel=True)
                style_dataset = volumes['style_dataset']
            else:
                style_dataset = volumes['dataset']

            # Put channels first and transform
            image = image.permute(2, 0, 1) # Channels on top
            label = label.permute(2, 0, 1) # Channels on top
            if self.non_corresponding_slices:
                style = style.permute(2, 0, 1) # Channels on top
                label_style=label_style.permute(2,0,1)

            # Inner transform
            if self.transform is not None:
                s = np.random.random()
                temp_transform = self.transform.set_random_state(s)
                image = np.array(temp_transform(image.numpy()))
                if self.non_corresponding_slices:
                    style = np.asarray(temp_transform(style.numpy()))
            else:
                image = image.numpy()
                if self.non_corresponding_slices:
                    style = style.numpy()

            image = self.normalize(torch.from_numpy(image))
            if self.non_corresponding_slices:
                style = self.normalize(torch.from_numpy(style))

            if self.skullstrip:
                image = uvir.SkullStrip(image, label, value=-1)
                if self.non_corresponding_slices:
                    style = uvir.SkullStrip(style, label_style, value = -1)

            if not self.non_corresponding_slices:
                style = deepcopy(image)
                style_dataset = volumes['dataset']

            current_idx = self.ctr['current']
            self.ctr['current'] += 1
            if self.ctr['current'] == self.ctr['max']:
                self.ctr['current'] = 0

            # Build dataset output
            if isinstance(image, np.ndarray):
                image = torch.from_numpy(image)
            if isinstance(label, np.ndarray):
                label = torch.from_numpy(label)
            if isinstance(style, np.ndarray):
                style = torch.from_numpy(style)

            out_ = {'image': image.type('torch.FloatTensor'),
                   'style_image': style.type('torch.FloatTensor'),
                   'label': label.type('torch.FloatTensor'),
                   'this_seq': self.input_vol_dict[current_idx]['modality'],
                   'this_dataset': self.input_vol_dict[current_idx]['dataset'],
                   'slice_no': self.input_vol_dict[current_idx]['slice_no'],
                   'slice_style_no':  self.input_vol_dict[current_idx]['style_slice_no'],
                   'image_path': self.input_vol_dict[current_idx]['image_file'],
                   'label_path':self.input_vol_dict[current_idx]['label_file'],
                   'style_label_path': self.input_vol_dict[current_idx]['label_file'],
                   'image_slice_no':  self.input_vol_dict[current_idx]['slice_no'],
                   'affine': self.input_vol_dict[current_idx]['affine'],
                   'style_dataset': style_dataset,
                   }
            yield  out_

    def setDiffSlices(self, set_flag, reset_datasets = False):

        self.diff_slices = set_flag
        self.non_corresponding_slices = self.diff_slices or self.diff_style_volume
        if reset_datasets:
            self.volumeDataset = PersistentDataset(self.input_vol_dict, self.multipleVolumesLoader(),
                                                   cache_dir=self.cache_dir)
            self.ctr = {'current': 0, 'max': len(self.volumeDataset)}
            self.sliceDataset = IterableDataset(self.getIteratorFun(self.volumeDataset))

    def findEquivalentPath(self, ori_path, folder, keywords = [], positions = [], first_come_first_served = False):

        '''
        Looks into the items contained in folder for a name matching ori_path.
        Matches are compared keyword by keyword, where the keyword is a root present between hyphens within the
        file name.
        Example:
        sub-3_ses-3_20.png (ori_path)
        Someotherfile_sub-3_ses-3_24.png (one of the items of folder)
        If keywords are "sub" and "ses", the function will output the above because in "sub-3" and "ses-3" match
        in both names.
        :param ori_path: Original path to compare to. If full path, directories are removed from it.
        :param folder:  Folder where you want to look for files.
        :param keywords: list of keywords to look for matches
        :param positions: numeric positions (hyphen-separated spaces: A_B_C (A is 0, B is 1 etc.) that match.
        :param first_come_firs_served: bool. If True, outputs the first found item, else None. If False, outputs a list
        with all matching files.
        :return:
        '''

        if "/" in ori_path:
            ori_path = ori_path.split("/")[-1]

        if not first_come_first_served:
            outputs = []

        all_files = os.listdir(folder)
        for i in all_files:
            i_sp = i.split("_")
            o_sp = ori_path.split("_")
            coincidences = []
            for ind_i, i_sub in enumerate(i_sp):
                for ind_o, o_sub in enumerate(o_sp):
                    for kw in keywords:
                        if kw in i_sub and kw in o_sub:
                            if i_sub == o_sub:
                                coincidences.append(True)
                            else:
                                coincidences.append(False)
                            break
            if False not in coincidences:
                if first_come_first_served:
                    return i
                else:
                    outputs.append(i)

        if first_come_first_served:
            return None
        else:
            return outputs

    def compute_slices(self, paths_base, slices_base, paths_style, slices_style):
        '''
        Add the slices of a batch (base and style) to the self.stored_slices object.
        :param paths_base: List of size B (Batch Size) with the paths to the labels.
        :param slices_base: List containing the slice numbers associated to these slices.
        :param path_styles: List of size B (Batch Size) with the paths to the style labels
        :param slices_styles: List containing the style slice numbers associated to these slices.
        :return:
        '''

        for p_ind, p in enumerate(paths_base):
            name_label_base = p.split("/")[-1]
            if name_label_base not in self.stored_slices['base'].keys():
                self.stored_slices['base'][name_label_base] =int(slices_base[p_ind])
            name_label_style = paths_style[p_ind].split("/")[-1]
            if name_label_style not in self.stored_slices['style'].keys():
                self.stored_slices['style'][name_label_style] =int(slices_style[p_ind])

    def compute_lesions(self):

        self.intensify_lesions['ctr'] += 1
        if self.intensify_lesions['ctr'] > self.intensify_lesions['per']:
            self.intensify_lesions['ctr'] = 1
        if self.intensify_lesions['sampling'] == 'interval':
            self.intensify_lesions['intervaler'].increaseIntervaler()

    def ruleOutOffset(self, img):

        offset_slices = int(self.ruleout_offset*img.shape[self.selected_index])
        return self.slicer.chopSlice(img, [offset_slices, img.shape[self.selected_index] - offset_slices],
                                     self.cut)

    def monaifyInput(self, fixed_seq = None,):

        '''
        Prepares an alternating dictionary containing the image and label paths in a dictionary to be passed
        to a Monai-like loader.
        If intensify lesions is true, looks for files with lesions present.
        :param fixed_seq: if not None, takes the value of one of the modalities; the only one for which we'll
        draw out volumes.
        Makes use of the following two attributes:
            - diff_style_volume: if true, the style and style label image come from a slice from an entirely
            different volume. Overwrites diff_slice.
            - diff_slice: if diff_slice_volume isn't true, and diff_slice is True, we try to find an image
            from the same volume but different slice number for the style and label style.
        :return:
        '''

        all_images = {}
        counters = {}
        length_all = 0

        if fixed_seq is not None:
            # Single sequence dataset.
            all_images[fixed_seq] = [i for i in os.listdir(self.image_dir) if i.split("_")[0] == fixed_seq]
            if self.intensify_lesions['flag']:
                all_images[fixed_seq] = [i for i in all_images[fixed_seq] if "nolesion" not in i]
            random.shuffle(all_images[fixed_seq])
            counters[fixed_seq] = 0
            length_all += len(all_images[fixed_seq])
        else:
            non_empty_dirs = []
            for mod in self.modalities:
                # Filter images per modality
                all_images[mod] = [i for i in os.listdir(self.image_dir) if i.split("_")[0] == mod]
                if self.intensify_lesions['flag']:
                    all_images[mod] = [i for i in all_images[mod] if "nolesion" not in i]
                # Shuffle images
                if len(all_images[mod]) == 0:
                    Warning("The image directory %s doesn't contain images for modality %s."
                            "The modality will be removed from self.modalities for the dataset,"
                            "as it may lead to empty list errors in the future.")
                    if mod in self.modalities:
                        self.modalities.remove(mod)
                else:
                    non_empty_dirs.append(mod)
                    random.shuffle(all_images[mod])

                if len(non_empty_dirs) == 1:
                    self.modalities = non_empty_dirs
                    self.fix_seq = True
                    Warning("Turned dataset to fix_seq = %s as only images for one "
                            "modality are available." %non_empty_dirs[0])

                counters[mod] = 0
                length_all += len(all_images[mod])

        out = []
        counter_mod = 0
        if self.max_dataset_size is not None:
            if self.max_dataset_size < length_all:
                length_all = self.max_dataset_size

        while length_all > 0:
            if fixed_seq is None:
                chosen_mod = self.modalities[counter_mod]
            else:
                chosen_mod = fixed_seq

            # Find each image and corresponding one
            image_name = all_images[chosen_mod][counters[chosen_mod]]
            label_name = image_name.replace(chosen_mod, "Parcellation", 1)

            if self.diff_style_volume:
                # We try to retrieve a different style
                dataset = self.getDataset(image_name)# Find dataset for this image / volume.
                possible_ims = [i for i in all_images[chosen_mod] if dataset not in i]
                if len(possible_ims) == 0:
                    Warning("Cross-dataset cannot be implemented because there is only one single valid dataset"
                            "in the directory. The style volume will be different, but belong to the same dataset.")
                    style_name = np.random.choice([i for i in all_images[chosen_mod] if i != image_name])
                else:
                    style_name = np.random.choice([i for i in all_images[chosen_mod] if dataset not in i])

                style_label_name = style_name.replace(chosen_mod, "Parcellation", 1) # Style label for style image for skull strip.

            elif self.diff_slices:
                # We look for slices from the same volume, but different slices.
                root = "_".join(image_name.split("_")[:-1])
                possible_ims = [i for i in all_images[chosen_mod] if root in i and i!=image_name]
                if len(possible_ims) == 0:
                    Warning("Non-corresponding slices is activated, but no other slices found for the same image.")
                    style_name = image_name
                    style_label_name = label_name
                else:
                    style_name = np.random.choice(possible_ims)
                    style_label_name = style_name.replace(chosen_mod, "Parcellation", 1)

            # Extract additional metadata
            metadata = np.load(os.path.join(self.image_dir, image_name))

            if self.diff_style_volume or self.diff_slices:
                out.append({'image_file': os.path.join(self.image_dir, image_name),
                            'label_file': os.path.join(self.label_dir, label_name),
                            'style_file': os.path.join(self.image_dir, style_name),
                            'style_label_file': os.path.join(self.label_dir, style_label_name),
                            'slice_no': int(image_name.strip(".npz").split("_")[-1]),
                            'style_slice_no': int(style_name.strip(".npz").split("_")[-1]),
                            'affine': metadata['img_affine'],
                            'modality': chosen_mod,
                            'dataset': str(metadata['dataset_name']),
                            'style_dataset': self.getDataset(style_name)})
            else:
                out.append({'image_file': os.path.join(self.image_dir, image_name),
                            'label_file': os.path.join(self.label_dir, label_name),
                            'affine': metadata['img_affine'],
                            'modality': chosen_mod,
                            'slice_no': int(image_name.strip(".npz").split("_")[-1]),
                            'style_slice_no': int(image_name.strip(".npz").split("_")[-1]),
                            'dataset': str(metadata['dataset_name'])})

            counters[chosen_mod] += 1
            counter_mod += 1
            if counters[chosen_mod] >= len(all_images[chosen_mod]):
                counters[chosen_mod] = 0
            if counter_mod == len(self.modalities):
                counter_mod = 0

            length_all -= 1

        return out

    def monaifyInput_non_corr(self, fixed_seq = None):

        '''
        Prepares an alternating dictionary containing the image and label paths in a dictionary to be passed
        to a Monai-like loader. The files contained in image_dir and label_dir are not corresponding.
        :param fixed_seq: if not None, takes the value of one of the modalities; the only one for which we'll
        draw out volumes.
        :return:
        '''

        all_images = {}
        # We make a list of all the images present for each / the modality.
        if fixed_seq is not None:
            # Single sequence dataset.
            all_images[fixed_seq] = [i for i in os.listdir(self.image_dir) if i.split("_")[0] == fixed_seq]
            random.shuffle(all_images[fixed_seq])
        else:
            non_empty_dirs = []
            for mod in self.modalities:
                # Filter images per modality
                all_images[mod] = [i for i in os.listdir(self.image_dir) if i.split("_")[0] == mod]
                # Shuffle images
                if len(all_images[mod]) == 0:
                    Warning("The image directory %s doesn't contain images for modality %s."
                            "The modality will be removed from self.modalities for the dataset,"
                            "as it may lead to empty list errors in the future.")
                    self.modalities.remove(mod)
                else:
                    non_empty_dirs.append(mod)
                    random.shuffle(all_images[mod])

                if len(non_empty_dirs) == 1:
                    self.modalities = non_empty_dirs
                    self.fix_seq = True
                    Warning("Turned dataset to fix_seq = %s as only images for one "
                            "modality are available." % non_empty_dirs[0])

        # List each modality
        all_labels = os.listdir(self.label_dir)
        if self.intensify_lesions['flag']:
            all_labels = [i for i in all_labels if "nolesion" not in i]
        length_all = len(all_labels)
        if self.max_dataset_size is not None:
            if self.max_dataset_size < len(all_labels):
                length_all = self.max_dataset_size

        out = []
        counter_mod = 0
        counter = 0

        while length_all > 0:
            if fixed_seq is None:
                chosen_mod = self.modalities[counter_mod]
            else:
                chosen_mod = fixed_seq

            # Find each image and corresponding one
            image_name = np.random.choice(all_images[chosen_mod])
            label_name = all_labels[counter]
            dataset = self.getDataset(image_name)
            style_label_name = image_name.replace(chosen_mod, "Parcellation", 1)

            # Extract additional metadata
            metadata = np.load(os.path.join(self.image_dir, image_name))

            out.append({'image_file': os.path.join(self.image_dir, image_name),
                        'label_file': os.path.join(self.label_dir, label_name),
                        'style_file': os.path.join(self.image_dir, image_name),
                        'style_label_file': os.path.join(self.style_label_dir, style_label_name),
                        'slice_no': int(label_name.strip(".npz").split("_")[-1]),
                        'style_slice_no': int(image_name.strip(".npz").split("_")[-1]),
                        'affine': metadata['img_affine'],
                        'modality': chosen_mod,
                        'dataset': dataset,
                        'style_dataset': dataset})

            counter_mod += 1
            if counter_mod == len(self.modalities):
                counter_mod = 0
            length_all -= 1
            counter += 1

        return out

    def resetDatasets(self, modalities = None, fixed_modality = None):

        '''
        Resets the Persistent and Iterable datasets.
        Also removes files from cache directory.
        :param modality:
        :return:
        '''

        if modalities is None:
            if fixed_modality is None:
                ValueError("If modalities is None, fixed modality mustn't be NONE")

        if os.path.isdir(self.cache_dir):
            Warning("Removing cache directory for reset...")
            shutil.rmtree(self.cache_dir)

        if modalities is None:
            self.modalities = [fixed_modality]
            self.fix_seq = True
            if self.non_corresponding_dirs:
                self.input_vol_dict = self.monaifyInput_non_corr(fixed_modality)
            else:
                self.input_vol_dict = self.monaifyInput(fixed_modality)
        else:
            self.fix_seq = False
            self.modalities = modalities
            if self.non_corresponding_dirs:
                self.input_vol_dict = self.monaifyInput_non_corr()
            else:
                self.input_vol_dict = self.monaifyInput()

        if self.opt.cache_type == 'cache':
            self.volumeDataset = PersistentDataset(self.input_vol_dict, self.multipleVolumesLoader(),
                                                   cache_dir=self.cache_dir,
                                                   pickle_protocol=pickle.HIGHEST_PROTOCOL)

        elif self.opt.cache_type == 'none':
            self.volumeDataset = Dataset(self.input_vol_dict, self.multipleVolumesLoader(),
                                         pickle_protocol=pickle.HIGHEST_PROTOCOL)
        elif self.opt.cache_type == 'ram':
            self.volumeDataset = CacheDataset(self.input_vol_dict, self.multipleVolumesLoader(),
                                              pickle_protocol=pickle.HIGHEST_PROTOCOL)
        self.ctr = {'current': 0, 'max': len(self.volumeDataset)}

        self.sliceDataset = IterableDataset(self.getIteratorFun(self.volumeDataset))

    def clearCache(self):
        if os.path.isdir(self.cache_dir):
            Warning("Removing cache directory for reset...")
            shutil.rmtree(self.cache_dir)

    def multipleVolumesLoader(self):

        '''
        Defines a compose loader for images and labels.
        Returns the loader.
        :return:
        '''

        image_loader = mt.LoadImageD(
            keys=('image_file'),
            reader = "numpyreader",
            npz_keys = ('img')
        )

        label_loader = mt.LoadImageD(
            keys=('label_file'),
            reader = "numpyreader",
            npz_keys = ('label',)
        )

        # image_loader = LoadNPZ(
        #     keys=('image_file',),
        #     npz_keys = ('img',)
        # )
        #
        # label_loader = LoadNPZ(
        #     keys=('label_file',),
        #     npz_keys = ('label',)
        # )


        if self.non_corresponding_slices:
            style_loader = mt.LoadImageD(
                keys=('style_file'),
                reader = "numpyreader",
                npz_keys = ('img',)
            )
            style_label_loader = mt.LoadImageD(
                keys=('style_label_file'),
                reader = "numpyreader",
                npz_keys = ('label',)
            )

            both_loaders = mt.Compose([image_loader, label_loader, style_loader, style_label_loader])
        else:
            both_loaders = mt.Compose([image_loader, label_loader])

        return both_loaders

    def imageOnlyLoader(self):

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

        return mt.Compose([image_loader, label_loader])

    def __len__(self):

        return int(self.ctr['max']/self.batchSize)

    def IntensityTransform(self, prob_bias_field = 1.0, prob_rand_adjust = 1.0, prob_gaussian_noise = 1.0):
        '''
        Returns Monai Compose of Random Bias Field, Random Adjust Contrast and Random Gaussian Noise.
        :return:
        '''

        if self.opt.dataset_type == 'volume':
            return monai.transforms.Compose(
                [monai.transforms.RandBiasField(coeff_range=(1, 1.3), prob=prob_bias_field),
                 monai.transforms.RandAdjustContrast(gamma=(0.95, 1.25), prob=prob_rand_adjust),
                 monai.transforms.RandGaussianNoise(prob=prob_gaussian_noise, mean=0.0,
                                                    std=np.random.uniform(0.005, 0.015))]
            )
        else:
            return monai.transforms.Compose(
                [monai.transforms.RandBiasField(coeff_range=(0.2, 0.6), prob = prob_bias_field),
                 monai.transforms.RandAdjustContrast(gamma=(0.85, 1.25), prob=prob_rand_adjust),
                 monai.transforms.RandGaussianNoise(prob=prob_gaussian_noise, mean=0.0, std=np.random.uniform(0.05, 0.15))]
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

        #
        if not self.opt.bound_normalization:
            normalise_fct = monai.transforms.NormalizeIntensity()
            return torch.from_numpy(np.asarray(normalise_fct(tensor)))
        else:
            if tensor.shape[0] == 1:
                means_ = (0.5,)
                stds_ = (0.5,)
            else:
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

    def getDataset(self, path):

        '''
        Retrieves the dataset from a certain path.
        :param path:
        :return:
        '''

        for ds in self.datasets:
            if ds in path.split("/")[-1]:
                return ds
        return None

    def switchStyles(self, input_dict, flags = None):

        '''
        For each image of the batch in input dict, we modify the style_image.
        For this, we use a the working volume option from SpadeNai dataset.
        Although not very efficient, it's the best way to embed it within the existing code.
        We set a working volume with the new style volume, and the original label volumes,
        and we sample a dictionary from them. Then we replace only the style image (and number etc.)
        on the input_dict.
        :param input_dict:
        :param same_volume: If True, the same image volume as the one that serves as input is used, only
        that a different slice is drawn.
        :param flags: For which elements of the batch we need to switch styles.
        :return:
        '''

        if flags is None:
            flags = [True]*input_dict['label'].shape[0]

        loader_imgs = self.imageOnlyLoader()

        for b in range(input_dict['label'].shape[0]):
            if flags[b]:
                # If image has been labelled with bad accuracy
                # We chose a different style.
                new_style = np.random.choice([i for i in os.listdir(self.image_dir) if
                                              i.split("_")[0] == input_dict['this_seq'][b]
                                              and i != input_dict['image_path'][b]]
                                             )
                new_style_slice_no = int(new_style.split("_")[-1].strip(".npz"))
                dataset = self.getDataset(new_style)
                new_style_label = new_style.replace(input_dict['this_seq'][b], "Parcellation")
                new_style = os.path.join(self.image_dir, new_style)

                if self.non_corresponding_dirs:
                    new_style_label = os.path.join(self.style_label_dir, new_style_label)
                else:
                    new_style_label = os.path.join(self.label_dir, new_style_label)

                # Load image and corresponding label and post-process.
                    # Simply load
                loaded = loader_imgs({'image_file': new_style, 'label_file': new_style_label})
                label_style = torch.from_numpy(loaded['label_file'])[..., 0, :]
                image_style = torch.from_numpy(loaded['image_file'])
                    # Pad to fit the self.size
                label_style = self.padder(label_style, islabel=True)
                image_style = self.padder(image_style, islabel=False)
                    # Put channels first and transform
                image_style = image_style.permute(2, 0, 1)  # Channels on top
                label_style = label_style.permute(2, 0, 1)  # Channels on top
                image_style = self.normalize(image_style)
                if self.skullstrip:
                    image_style = uvir.SkullStrip(image_style, label_style, value=-1)


                # Modify input dict with new style
                input_dict['style_image'][b,...] = image_style
                input_dict['slice_style_no'][b, ...] = new_style_slice_no
                input_dict['style_label_path'][b] = new_style_label
                input_dict['this_dataset'][b] =  dataset,

        return input_dict
