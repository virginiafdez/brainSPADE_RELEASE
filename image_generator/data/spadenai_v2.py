import monai as monai
import monai.transforms as mt
import torch
import numpy as np
import os
import random
import torchvision.transforms.functional as FTV
from data.custom_transforms import Slicer, CropPad
import moreutils as uvir
from monai.data.dataset import PersistentDataset, Dataset, CacheDataset
from monai.data.iterable_dataset import IterableDataset
import shutil
import nibabel as nib
import pickle

class SpadeNai():

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

        if mode == 'validation':
            self.image_dir = opt.image_dir_val
            self.label_dir = opt.label_dir_val
        else:
            self.image_dir = opt.image_dir
            self.label_dir = opt.label_dir

        # Necessary to specify the slicer transform
        self.cut_ids = {'a': 2, 's': 0, 'c': 1}  # Cut (a: axial, s: sagittal, c: coronal)
        self.cut = opt.cut
        self.selected_index = self.cut_ids[self.cut] # Index corresponding to the selected cut.

        # Whereas we have one or multiple modalities.
        if opt.fix_seq is not None:
            self.modalities = [opt.fix_seq]
            self.fix_seq = True
        else:
            self.modalities = opt.sequences
            self.fix_seq = False

        # Store slices
        if 'store_and_use_slices' in kwargs.keys():
            store_and_use_slices = kwargs['store_and_use_slices']
        else:
            store_and_use_slices = opt.store_and_use_slices

        if store_and_use_slices:
            self.stored_slices = {'base': {}, 'style': {}}
            self.store_slices = True
        else:
            self.store_slices = False
            self.stored_slices = None

        self.datasets = opt.datasets
        self.batchSize = opt.batchSize

        #Cache_dir for peristent dataset
        if opt.cache_dir is None:
            self.cache_dir = os.path.join(opt.checkpoints_dir, opt.name, "spadenai_cache_dir")
        else:
            self.cache_dir = os.path.join(opt.cache_dir, "spadenai_cache_dir")
        if not os.path.isdir(self.cache_dir):
            os.makedirs(self.cache_dir)

        # Transforms
        if opt.intensity_transform and (mode == 'train' or mode == 'default'):
            self.transform = self.IntensityTransform(prob_bias_field=0.5,
                                                     prob_rand_adjust= 0.5,
                                                     prob_gaussian_noise=0.05)
        else:
            self.transform = None

        pad_size = opt.new_size
        pad_size.append(-1)
        self.new_size = opt.new_size
        self.padder = CropPad(pad_size)
        self.slicer = Slicer(cut = self.cut, output_channels=1,
                             non_corresponding=False)

        self.skullstrip = opt.skullstrip
        self.ruleout_offset = opt.ruleout_offset
        self.diff_slices = opt.diff_slice
        self.diff_style_volume = opt.diff_style_volume
        self.non_corresponding_slices = self.diff_style_volume or self.diff_slices


        # Persistent Dataset for the volumes

        if self.fix_seq:
            self.input_vol_dict = self.monaifyInput(self.modalities[0])
        else:
            self.input_vol_dict = self.monaifyInput()

        # Creation of Persistent Dataset.
        if opt.cache_type=='cache':
            self.volumeDataset = PersistentDataset(self.input_vol_dict,
                                                   self.multipleVolumesLoader(self.diff_style_volume),
                                                   cache_dir=self.cache_dir)
        elif opt.cache_type=='none':
            self.volumeDataset = Dataset(self.input_vol_dict, self.multipleVolumesLoader(self.diff_style_volume))
        elif opt.cache_type=='ram':
            self.volumeDataset = CacheDataset(self.input_vol_dict, self.multipleVolumesLoader(self.diff_style_volume))

        self.ctr = {'current': 0, 'max': len(self.volumeDataset)}

        # Iterable dataset for the slices
        if self.diff_style_volume:
            self.sliceDataset = IterableDataset(self.getIteratorFun_WStyle(self.volumeDataset))
        else:
            self.sliceDataset = IterableDataset(self.getIteratorFun(self.volumeDataset))

        if opt.intensify_lesions is not None and mode != 'validation':
            self.intensify_lesions = {'per': opt.intensify_lesions,
                                      'ctr': 1,
                                      'flag': True,
                                      'sampling': opt.lesion_sampling_mode,
                                      'intervaler': None}
            if opt.lesion_sampling_mode not in ['interval', 'threshold']:
                ValueError("Sampling options must be interval or threshold only. ")
            if opt.lesion_sampling_mode == 'interval':
                self.intensify_lesions['intervaler'] = Intervaler()

        else:
            self.intensify_lesions = {'per': -1, 'ctr': 1, 'flag': False, 'sampling': opt.lesion_sampling_mode,
                                      'intervaler': None}

        self.working_volume = None

    def getIteratorFun(self, dataset):

        for volumes in dataset:
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
            # Pad label and image to the unique input size
            label = torch.from_numpy(volumes['label_file'])
            image = torch.from_numpy(volumes['image_file'])

            # Before cropping, we apply ruleout offset if needed
            if self.ruleout_offset != 0:
                label = self.ruleOutOffset(label)
                image = self.ruleOutOffset(image)

            label_vol = self.padder(label, islabel = True) # Move channels first as if they were a batch
            image_vol = self.padder(image, islabel = False)

            # Slice
            label_name = self.input_vol_dict[self.ctr['current']]['label_file'].split("/")[-1]
            if self.store_slices:
                if label_name in self.stored_slices['base'].keys():
                    label, image, slices = self.slicer(label_vol, image_vol,
                                                       slice_no=self.stored_slices['base'][label_name])
                else:
                    if self.intensify_lesions['flag']:
                        if self.intensify_lesions['ctr'] == self.intensify_lesions['per']:
                            self.intensify_lesions['ctr'] = 0
                            if self.intensify_lesions['sampling'] == 'interval':
                                thresholds = self.intensify_lesions['intervaler'].getThreshold()
                                label, image, slices = self.slicer(label_vol, image_vol, lesion=True,
                                                                   lesion_th_low=thresholds[0],
                                                                   lesion_th_up=thresholds[1])
                            else:
                                label, image, slices = self.slicer(label_vol, image_vol, lesion=True)
                        else:
                            label, image, slices = self.slicer(label_vol, image_vol, lesion=False)
                            self.intensify_lesions['ctr'] += 1
                    else:
                        label, image, slices = self.slicer(label_vol, image_vol, lesion=False)
                    self.stored_slices['base'][label_name] = slices['label']
            else:
                if self.intensify_lesions['flag']:
                    if self.intensify_lesions['ctr'] == self.intensify_lesions['per']:
                        self.intensify_lesions['ctr'] = 0
                        if self.intensify_lesions['sampling'] == 'interval':
                            thresholds = self.intensify_lesions['intervaler'].getThreshold()
                            label, image, slices = self.slicer(label_vol, image_vol, lesion=True,
                                                               lesion_th_low=thresholds[0],
                                                               lesion_th_up=thresholds[1])
                        else:
                            label, image, slices = self.slicer(label_vol, image_vol, lesion=True)
                    else:
                        label, image, slices = self.slicer(label_vol, image_vol, lesion=False)
                        self.intensify_lesions['ctr'] += 1
                else:
                    label, image, slices = self.slicer(label_vol, image_vol, lesion=False)

            if self.diff_slices:
                if self.store_slices:
                    # Store slices: we check if for this specific volume, we have "style" slices saved.
                    if label_name in self.stored_slices['style'].keys():
                        label_style, style, st_slices = self.slicer(label_vol, image_vol,
                                                           slice_no=self.stored_slices['style'][label_name])
                    else:
                        label_style, style, st_slices = self.slicer(label_vol, image_vol, lesion=False)
                        self.stored_slices['style'][label_name] = st_slices['image']
                else:
                    label_style, style, st_slices = self.slicer(label_vol, image_vol, lesion=False)
                label_style = label_style.permute(2, 0, 1)  # Channels on top
                style_slice = st_slices['image']
            else:
                style = image
                style_slice = slices['image']

            # Put channels first and transform
            image = image.permute(2, 0, 1) # Channels on top
            label = label.permute(2, 0, 1) # Channels on top
            style = style.permute(2, 0, 1) # Channels on top

            # Inner transform
            if self.transform is not None:
                s = np.random.random()
                temp_transform = self.transform.set_random_state(s)
                image = np.asarray(temp_transform(image.numpy()))
                style = np.asarray(temp_transform(style.numpy()))
            else:
                image = image.numpy()
                style = style.numpy()

            image = self.normalize(torch.from_numpy(image))
            style = self.normalize(torch.from_numpy(style))

            if self.skullstrip:
                image = uvir.SkullStrip(image, label, value=-1)
                if self.diff_slices:
                    style = uvir.SkullStrip(style, label_style, value = -1)
                else:
                    style = uvir.SkullStrip(style, label, value = -1)

            current_idx = self.ctr['current']
            self.ctr['current'] += 1
            if self.ctr['current'] == self.ctr['max']:
                self.ctr['current'] = 0

            # Build dataset output
            yield {'image': image.type('torch.FloatTensor'),
                   'style_image': style.type('torch.FloatTensor'),
                   'label': label.type('torch.FloatTensor'),
                   'this_seq': self.input_vol_dict[current_idx]['modality'],
                   'this_dataset': self.input_vol_dict[current_idx]['dataset'],
                   'slice_no': slices['label'],
                   'slice_style_no': style_slice,
                   'image_path': self.input_vol_dict[current_idx]['image_file'],
                   'label_path':self.input_vol_dict[current_idx]['label_file'],
                   'style_label_path': self.input_vol_dict[current_idx]['label_file'],
                   'image_slice_no': slices['image']
                   }

    def getIteratorFun_WStyle(self, dataset):
        '''
        Same as getIteratorFun, but using a totally separate style volume.
        Here, diff_slices is active 100% of the time.
        :param dataset:
        :return:
        '''

        for volumes in dataset:
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

            # Pad label and image to the unique input size
            label = torch.from_numpy(volumes['label_file'])
            image = torch.from_numpy(volumes['image_file'])
            style = torch.from_numpy(volumes['style_file'])
            style_label = torch.from_numpy(volumes['style_label_file'])

            # Before cropping, we apply ruleout offset if needed
            if self.ruleout_offset != 0:
                label = self.ruleOutOffset(label)
                image = self.ruleout_offset(image)
                style = self.ruleout_offset(style)
                style_label = self.ruleout_offset(style_label)

            label_vol = self.padder(label, islabel=True)  # Move channels first as if they were a batch
            image_vol = self.padder(image, islabel=False)
            style_vol = self.padder(style, islabel=False)
            style_label_vol = self.padder(style_label, islabel=False)

            # Slice
            label_name = self.input_vol_dict[self.ctr['current']]['label_file'].split("/")[-1]

            # Draw slices from the main label and volume (label and ground truth)
            if self.store_slices:
                if label_name in self.stored_slices['base'].keys():
                    label, image, slices = self.slicer(label_vol, image_vol,
                                                       slice_no=self.stored_slices['base'][label_name])
                    self.stored_slices['base'][label_name] = slices['label']
                else:
                    if self.intensify_lesions['flag']:
                        if self.intensify_lesions['ctr'] == self.intensify_lesions['per']:
                            self.intensify_lesions['ctr'] = 0
                            if self.intensify_lesions['sampling'] == 'interval':
                                thresholds = self.intensify_lesions['intervaler'].getThreshold()
                                label, image, slices = self.slicer(label_vol, image_vol, lesion=True,
                                                                   lesion_th_low=thresholds[0],
                                                                   lesion_th_up=thresholds[1])
                            else:
                                label, image, slices = self.slicer(label_vol, image_vol, lesion=True)
                        else:
                            label, image, slices = self.slicer(label_vol, image_vol, lesion=False)
                            self.intensify_lesions['ctr'] += 1
                    else:
                        label, image, slices = self.slicer(label_vol, image_vol, lesion=False)
                    self.stored_slices['base'][label_name] = slices['label']
            else:
                if self.intensify_lesions['flag']:
                    if self.intensify_lesions['ctr'] == self.intensify_lesions['per']:
                        self.intensify_lesions['ctr'] = 0
                        if self.intensify_lesions['sampling'] == 'interval':
                            thresholds = self.intensify_lesions['intervaler'].getThreshold()
                            label, image, slices = self.slicer(label_vol, image_vol, lesion=True,
                                                               lesion_th_low=thresholds[0],
                                                               lesion_th_up=thresholds[1])
                        else:
                            label, image, slices = self.slicer(label_vol, image_vol, lesion=True)
                    else:
                        label, image, slices = self.slicer(label_vol, image_vol, lesion=False)
                        self.intensify_lesions['ctr'] += 1
                else:
                    label, image, slices = self.slicer(label_vol, image_vol, lesion=False)

            # Draw style slices.
            style_name = self.input_vol_dict[self.ctr['current']]['style_label_file'].split("/")[-1]

            if self.store_slices:
                # Store slices: we check if for this specific volume, we have "style" slices saved.
                if style_name in self.stored_slices['style'].keys():
                    label_style, style, st_slices = self.slicer(style_label_vol, style_vol,
                                                                slice_no = self.stored_slices['style'][style_name],
                                                                lesion=False)
                else:
                    label_style, style, st_slices = self.slicer(style_label_vol, style_vol, lesion=False)
                    self.stored_slices['style'][style_name] = st_slices['image']
                label_style = label_style.permute(2, 0, 1)  # Channels on top
                style_slice = st_slices['image']
            else:
                label_style, style, st_slices = self.slicer(style_label_vol, style_vol, lesion=False)
                label_style = label_style.permute(2, 0, 1)  # Channels on top
                style_slice = st_slices['image']

            # Put channels first and transform
            image = image.permute(2, 0, 1)  # Channels on top
            label = label.permute(2, 0, 1)  # Channels on top
            style = style.permute(2, 0, 1)  # Channels on top

            # Inner transform
            if self.transform is not None:
                s = np.random.random()
                temp_transform = self.transform.set_random_state(s)
                image = np.asarray(temp_transform(image.numpy()))
                style = np.asarray(temp_transform(style.numpy()))
            else:
                image = image.numpy()
                style = style.numpy()

            image = self.normalize(torch.from_numpy(image))
            style = self.normalize(torch.from_numpy(style))

            if self.skullstrip:
                image = uvir.SkullStrip(image, label, value=-1)
                style = uvir.SkullStrip(style, label_style, value=-1)

            current_idx = self.ctr['current']
            self.ctr['current'] += 1
            if self.ctr['current'] == self.ctr['max']:
                self.ctr['current'] = 0

            if isinstance(image, np.ndarray):
                image = torch.from_numpy(image)
            if isinstance(label, np.ndarray):
                label = torch.from_numpy(label)
            if isinstance(style, np.ndarray):
                style = style.from_numpy(style)

            # Build dataset output
            yield {'image': image.type('torch.FloatTensor'),
                   'style_image': style.type('torch.FloatTensor'),
                   'label': label.type('torch.FloatTensor'),
                   'this_seq': self.input_vol_dict[current_idx]['modality'],
                   'this_dataset': self.input_vol_dict[current_idx]['dataset'],
                   'slice_no': slices['label'],
                   'slice_style_no': style_slice,
                   'image_path': self.input_vol_dict[current_idx]['image_file'],
                   'label_path': self.input_vol_dict[current_idx]['label_file'],
                   'image_slice_no': slices['image'],
                   'style_path': self.input_vol_dict[current_idx]['style_file'],
                   'style_dataset': self.getDataset(self.input_vol_dict[current_idx]['style_file']),
                   'style_label_path': self.input_vol_dict[current_idx]['style_label_file']
                   }

    def flushStoredSlices(self, allow_stored):
        if allow_stored:
            self.store_slices = True
            self.stored_slices = {'base': {}, 'style': {}}
        else:
            self.store_slices = False
            self.stored_slices = None

    def setIntensifyLesions(self, set_flag, period = None):

        if set_flag:
            if period == None:
                ValueError("Error: if you want to set self.intensify_lesions, the period must be provided.")
            self.intensify_lesions['flag'] = set_flag
            self.intensify_lesions['ctr'] = 0
            self.intensify_lesions['per'] = period
        else:
            self.intensify_lesions['flag'] = set_flag
            self.intensify_lesions['ctr'] = 1
            self.intensify_lesions['per'] = -1

    def setDiffSlices(self, set_flag, reset_datasets = False):

        self.diff_slices = set_flag
        self.non_corresponding_slices = self.diff_style_volume or self.diff_slices
        if reset_datasets:
            self.volumeDataset = PersistentDataset(self.input_vol_dict,
                                                   self.multipleVolumesLoader(self.diff_style_volume),
                                                   cache_dir=self.cache_dir)
            self.ctr = {'current': 0, 'max': len(self.volumeDataset)}

            if self.diff_style_volume:
                self.sliceDataset = IterableDataset(self.getIteratorFun_WStyle(self.volumeDataset))
            else:
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

    def monaifyInput(self, fixed_seq = None):

        '''
        Prepares an alternating dictionary containing the image and label paths in a dictionary to be passed
        to a Monai-like loader.
        :param fixed_seq: if not None, takes the value of one of the modalities; the only one for which we'll
        draw out volumes.
        :return:
        '''

        all_images = {}
        counters = {}
        length_all = 0
        if fixed_seq is not None:
            all_images[fixed_seq] = [i for i in os.listdir(self.image_dir) if i.split("_")[0] == fixed_seq]
            random.shuffle(all_images[fixed_seq])
            counters[fixed_seq] = 0
            length_all += len(all_images[fixed_seq])
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

        while length_all > 0:
            if fixed_seq is None:
                chosen_mod = self.modalities[counter_mod]
            else:
                chosen_mod = fixed_seq

            image_name = all_images[chosen_mod][counters[chosen_mod]]
            label_name = image_name.replace(chosen_mod, "Parcellation", 1)

            if self.diff_style_volume:
                dataset = self.getDataset(image_name)
                possible_ims = [i for i in all_images[chosen_mod] if dataset not in i]
                if len(possible_ims) == 0:
                    Warning("Cross-dataset cannot be implemented because there is only one single valid dataset"
                            "in the directory. The style volume will be different, but belong to the same dataset.")
                    style_name = np.random.choice([i for i in all_images[chosen_mod] if i != image_name])
                else:
                    style_name = np.random.choice([i for i in all_images[chosen_mod] if dataset not in i])

                style_label_name = style_name.replace(chosen_mod, "Parcellation", 1)

            # Extract additional metadata
            metadata = np.load(os.path.join(self.image_dir, image_name))

            if self.diff_style_volume:
                out.append({'image_file': os.path.join(self.image_dir, image_name),
                            'label_file': os.path.join(self.label_dir, label_name),
                            'style_file': os.path.join(self.image_dir, style_name),
                            'style_label_file': os.path.join(self.label_dir, style_label_name),
                            'affine': metadata['img_affine'],
                            'modality': chosen_mod,
                            'dataset': str(metadata['dataset_name']),
                            'style_dataset': self.getDataset(style_name)})
            else:
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
            self.input_vol_dict = self.monaifyInput(fixed_modality,)
        else:
            self.fix_seq = False
            self.modalities = modalities
            self.input_vol_dict = self.monaifyInput()

        if  self.opt.cache_type == 'cache':
            self.volumeDataset = PersistentDataset(self.input_vol_dict,
                                                   self.multipleVolumesLoader(self.diff_style_volume),
                                                   cache_dir=self.cache_dir)
        elif self.opt.cache_type == 'none':
            self.volumeDataset = Dataset(self.input_vol_dict, self.multipleVolumesLoader(self.diff_style_volume))
        elif self.opt.cache_type == 'ram':
            self.volumeDataset = CacheDataset(self.input_vol_dict, self.multipleVolumesLoader(self.diff_style_volume))

        self.ctr = {'current': 0, 'max': len(self.volumeDataset)}

        if self.diff_style_volume:
            self.sliceDataset = IterableDataset(self.getIteratorFun_WStyle(self.volumeDataset))
        else:
            self.sliceDataset = IterableDataset(self.getIteratorFun(self.volumeDataset))

    def clearCache(self):
        if os.path.isdir(self.cache_dir):
            Warning("Removing cache directory for reset...")
            shutil.rmtree(self.cache_dir)

    def multipleVolumesLoader(self, cross_style_vol):

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

        if cross_style_vol:
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

    def __len__(self):

        return int(self.ctr['max']/self.batchSize)

    def IntensityTransform(self, prob_bias_field = 1.0, prob_rand_adjust = 1.0, prob_gaussian_noise = 1.0):
        '''
        Returns Monai Compose of Random Bias Field, Random Adjust Contrast and Random Gaussian Noise.
        :return:
        '''

        return monai.transforms.Compose(
            [monai.transforms.RandBiasField(coeff_range=(1, 1.2), prob = prob_bias_field),
             monai.transforms.RandAdjustContrast(gamma=(0.9, 1.15), prob=prob_rand_adjust),
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

    def setWorkingVolume(self, image_paths, label_path,style = None,
                         style_label_paths = None):
        '''
        Associates an Offline Volume object to the container.
        :param image_paths: dictionary, key: sequence/modality, value: path to each image volume
        :param label_path: string, path to the label volume
        :param style: style to which all the volume files correspond (string) or dictionary associating
        a style to each sequence/modality.
        :param style_label_paths: dictionary, key: sequence/modality, value: path to the parcellation
        file associated to the image volumes in case it's not the one indicated by label_path.
        :return:
        '''

        # Find dataset associated to label and image paths.
        label_dataset = self.getDataset(label_path.split("/")[-1])
        image_datasets = {}
        if style is None:
            for seq, path in image_paths.items():
                image_datasets[seq] = self.getDataset(path.split("/")[-1])
                if image_datasets[seq] is None:
                    image_datasets[seq] = "SMITH"
        else:
            for seq in image_paths.keys():
                if type(style) is dict:
                    image_datasets[seq] = style[seq]
                else:
                    image_datasets[seq] = style

        self.working_volume = OfflineVolume(image_paths, label_path, label_dataset, cut = self.cut,
                                            style_dataset=image_datasets, labels_style=style_label_paths)

        for vol_type, volume in self.working_volume.volumes.items():
            if volume is not None:
                if 'Parcellation' in vol_type or 'label' in vol_type:
                    self.working_volume.volumes[vol_type] = self.padder(volume, islabel = True) # Move channels first as if they were a batch
                else:
                    self.working_volume.volumes[vol_type] = self.padder(volume, islabel=False)

    def getSlicefromWV(self, all_slices, modality, diff_style_slice = False):

        '''
        Returns a SPADE-compatible input dictionary in the form of a batch.
        The image and labels are retrieved from self.working_volumes, but modalities can be alternative.
        :param all_slices: it can be a list of ints (each int can be an int (slice to retrieve from both
        the parcellation and image file), a 2-element list (with the slice for the label and the slice
        for the image) or None - don't care, choose randomly).
        :param modality: string or list of strings, which modality you want for the output. If all_slices is a
        list, modality needs to be a list as well.
        :param diff_style_slice: if True, the style slice number is different from the label slice number.
        :return: dictionary, compatible with brainSPADEv3 net input.
        '''

        # Sanity check: all_slices must be correct.
        if type(all_slices) != type(modality):
            ValueError("In getSlicefromWV, all_slices and modality must be either lists or int. But the same.")
        if type(all_slices) is list and type(modality) is list:
            if len(all_slices) != len(modality):
                ValueError("The length of all_slices and of modality must be the SAME. ")
        elif type(all_slices) is int and type(modality) is int:
            all_slices = [all_slices]
            modality = [modality]

        # Initialise output structure
        output_dict = {}

        # Slice or slices.
        for s_ind, s in enumerate(all_slices):
            if type(s) is not int and type(s) is not list and s is not None:
                ValueError("Each slice must be either an integer, or a 2-element list.")
            else:
                if type(s) is list and len(s) != 2:
                    ValueError("If the slice number is a list, the length must be 2: label and image.")

            if s is None:
                # If None, random slice selected.
                # We check whether the label corresponds to this particular modality volume.
                # If not, we handle things independently.
                if modality[s_ind] in self.working_volume.non_corr.keys():
                    s_lab = np.random.choice(self.working_volume.volumes['Parcellation'].shape[self.cut_ids[self.cut]])
                    label_slice = self.slicer.chopSlice(self.working_volume.volumes["Parcellation"],
                                                        s_lab,
                                                        self.self.cut)

                    label_style_slice, image_slice, st_slices = self.slicer(
                        self.working_volume.volumes["label"+modality[s_ind]],
                        self.working_volume.volumes['gt'+modality[s_ind]],
                        lesion=False)

                    s_img = st_slices['image']

                    if diff_style_slice:
                        # Obviously, if diff_slice is true, we take this same image as style_image,
                        # because the output['image'] will not correspond to the label anwyay.
                        style_slice = image_slice
                else:
                    # Parcellation and image modality correspond, so we can use them jointly.
                    label_slice, image_slice, slices = self.slicer(self.working_volume.volumes["Parcellation"],
                                                                   self.working_volume.volumes['gt'+modality[s_ind]],
                                                                   lesion=False)
                    s_img = s_lab = slices['label']

                    # In case that we want a different style image (different slice number even if the image
                    # is the same as the ground truth):
                    if diff_style_slice:
                        label_style_slice, style_slice, st_slices = self.slicer(
                            self.working_volume.volumes['Parcellation'],
                            self.working_volume.volumes['gt'+modality[s_ind]],
                            lesion=False)
                        # Note: s_img corresponds to the slice for output['image'], not output['style_image']!

                # Normalise
                image_slice = image_slice.permute(2, 0, 1)  # Channels on top
                label_slice = label_slice.permute(2, 0, 1)  # Channels on top
                image_slice = self.normalize(image_slice)
                if diff_style_slice:
                    style_slice = style_slice.permute(2, 0, 1)  # Channels on top
                    label_style_slice = label_style_slice.permute(2, 0, 1)  # Channels on top
                    style_slice = self.normalize(style_slice)

                # Skullstrip
                if self.skullstrip:
                    if modality[s_ind] in self.working_volume.non_corr.keys():
                        # Volumes are different: we need to use label_style_slice.
                        image_slice = uvir.SkullStrip(image_slice, label_style_slice, value=-1)
                    else:
                        # Parcellation and Image (GT) correspond
                        image_slice = uvir.SkullStrip(image_slice, label_slice, value=-1)
                    if diff_style_slice:
                        # Style image (on the side).
                        style_slice = uvir.SkullStrip(style_slice, label_style_slice, value=-1)

            else:
                # s is specified.
                if type(s) is list:
                    # Different for labels and images.
                    s_lab = s[0]
                    s_img = s[1]
                else:
                    # Same for labels and images.
                    s_lab = s_img = s

                # We use s_img and s_lab.
                label_slice = self.slicer.chopSlice(self.working_volume.volumes['Parcellation'], s_lab,
                                                    cut = self.cut)
                image_slice = self.slicer.chopSlice(self.working_volume.volumes['gt'+modality[s_ind]], s_img,
                                                    cut = self.cut)

                if modality[s_ind] in self.working_volume.non_corr.keys():
                    # The image volume doesn't correspond to the label.
                    # We need the style_label_slice anyway.
                    label_style_slice = self.slicer.chopSlice(
                        self.working_volume.volumes['label'+modality[s_ind]],
                        s_img, cut = self.cut)
                    label_style_slice = label_style_slice.permute(2, 0, 1)  # Channels on top
                elif s_lab != s_img:
                    # If the volumes are equivalent but the slices are different, we also need
                    # label_style_slice for skull stripping.
                    label_img_slice = self.slicer.chopSlice(
                        self.working_volume.volumes['Parcellation'],
                        s_img, cut = self.cut)
                    label_img_slice = label_img_slice.permute(2, 0, 1)  # Channels on top

                if diff_style_slice:
                    if modality[s_ind] in self.working_volume.non_corr.keys():
                        style_slice = image_slice
                    else:
                        label_style_slice, style_slice, st_slices = self.slicer(self.working_volume.volumes['Parcellation'],
                                                            self.working_volume.volumes['gt'+modality[s_ind]],
                                                            lesion=False)

                # Normalise
                # Normalise
                image_slice = image_slice.permute(2, 0, 1)  # Channels on top
                label_slice = label_slice.permute(2, 0, 1)  # Channels on top
                image_slice = self.normalize(image_slice)
                if diff_style_slice:
                    style_slice = style_slice.permute(2, 0, 1)  # Channels on top
                    label_style_slice = label_style_slice.permute(2, 0, 1)  # Channels on top
                    style_slice = self.normalize(style_slice)

                # Skullstrip
                if self.skullstrip:
                    if s_lab != s_img or modality[s_ind] in self.working_volume.non_corr.keys():
                        if modality[s_ind] in self.working_volume.non_corr.keys():
                            image_slice = uvir.SkullStrip(image_slice, label_style_slice, value=-1)
                        else:
                            image_slice = uvir.SkullStrip(image_slice, label_img_slice, value=-1)
                    else:
                        image_slice = uvir.SkullStrip(image_slice, label_slice, value=-1)
                    if diff_style_slice:
                        style_slice = uvir.SkullStrip(style_slice, label_style_slice, value=-1)

            # Assemble output dictionary.

            if s_ind == 0:
                output_dict['image'] = image_slice.type('torch.FloatTensor').unsqueeze(0)
                output_dict['label'] = label_slice.type('torch.FloatTensor').unsqueeze(0)
                if diff_style_slice:
                    output_dict['style_image'] = style_slice.type('torch.FloatTensor').unsqueeze(0)
                    output_dict['slice_style_no'] = [st_slices['image']]
                else:
                    output_dict['style_image'] = image_slice.type('torch.FloatTensor').unsqueeze(0)
                    output_dict['slice_style_no'] = [s_img]
                output_dict['this_seq'] = [modality[s_ind]]
                output_dict['this_dataset'] = [self.working_volume.style_dataset[modality[s_ind]]]
                output_dict['slice_no'] = [s_lab]
                output_dict['image_path'] = [self.working_volume.paths[modality[s_ind]]]
                output_dict['label_path'] = [self.working_volume.paths['Parcellation']]
                output_dict['image_slice_no'] = [s_img]
            else:
                output_dict['image'] = torch.cat([output_dict['image'],
                                                    image_slice.type('torch.FloatTensor').unsqueeze(0)],
                                                   dim = 0)
                output_dict['label'] = torch.cat([output_dict['label'],
                                                    label_slice.type('torch.FloatTensor').unsqueeze(0)],
                                                   dim = 0)
                if diff_style_slice:
                    output_dict['style_image'] = torch.cat([output_dict['style_image'],
                                                    style_slice.type('torch.FloatTensor').unsqueeze(0)],
                                                   dim = 0)
                else:
                    output_dict['style_image'] = torch.cat([output_dict['style_image'],
                                                    image_slice.type('torch.FloatTensor').unsqueeze(0)],
                                                   dim = 0)
                output_dict['this_seq'].append(modality[s_ind])
                output_dict['this_dataset'].append(self.working_volume.style_dataset[modality[s_ind]])
                output_dict['slice_no'].append(s_lab)
                if diff_style_slice:
                    output_dict['slice_style_no'].append(st_slices['image'])
                else:
                    output_dict['slice_style_no'].append(s_img)
                output_dict['image_path'].append(self.working_volume.paths[modality[s_ind]])
                output_dict['label_path'].append(self.working_volume.paths['Parcellation'])
                output_dict['image_slice_no'].append(s_img)

        return output_dict

    def iterateThroughWV(self, modality, number_slices = 1,  use_same_style_slice=True,
                         use_same_slice_no=False):
        '''
        Iterates through the volume (modality)
        :param modality:
        :return:
        '''

        if self.working_volume.slice_iterator[modality] is None:
            self.working_volume.slice_iterator[modality] = {
                'current': 0,
                'max': self.working_volume.volumes['Parcellation'].shape[self.cut_ids[self.cut]],
                'same_slice': use_same_style_slice,
                'style_slice': None,
                'style_slice_no': None
            }

        output_dict = {}

        for s in range(number_slices):
            # Chop slice
            label_slice = self.slicer.chopSlice(self.working_volume.volumes["Parcellation"],
                                                self.working_volume.slice_iterator[modality]['current'],
                                                self.cut)
            if modality not in self.working_volume.non_corr.keys():
                image_slice = self.slicer.chopSlice(self.working_volume.volumes['gt'+modality],
                                                    self.working_volume.slice_iterator[modality]['current'],
                                                    self.cut)
                # Put channels first and transform
                image_slice = image_slice.permute(2, 0, 1)  # Channels on top
                label_slice = label_slice.permute(2, 0, 1)
                # Normalise
                image_slice = self.normalize(image_slice)
                # Skullstrip
                if self.skullstrip:
                    image_slice = uvir.SkullStrip(image_slice, label_slice, value=-1)

            else:
                label_image_slice, image_slice, slices_im = self.slicer(self.working_volume.volumes['label'+modality],
                                                                        self.working_volume.volumes['gt'+modality])
                # Normalise
                image_slice = image_slice.permute(2, 0, 1)  # Channels on top
                label_image_slice = label_image_slice.permute(2, 0, 1)  # Channels on top
                image_slice = self.normalize(torch.from_numpy(image_slice))
                if self.skullstrip:
                    image_slice = uvir.SkullStrip(image_slice, label_image_slice, value = -1)

            # STYLE
            if self.working_volume.slice_iterator[modality]['style_slice'] is not None \
                and self.working_volume.slice_iterator[modality]['same_slice']:
                style_slice = self.working_volume.slice_iterator[modality]['style_slice']
                slice_style_no = self.working_volume.slice_iterator[modality]['style_slice_no']
            else:
                if use_same_slice_no:
                    style_slice_no = self.working_volume.slice_iterator[modality]['current']
                else:
                    style_slice_no = None
                label_style_slice, style_slice, st_slices = self.slicer(self.working_volume.volumes['Parcellation'],
                                                                     self.working_volume.volumes['gt'+modality],
                                                                     lesion = False,
                                                                        slice_no=style_slice_no)
                # Normalise
                style_slice = style_slice.permute(2, 0, 1)  # Channels on top
                label_style_slice = label_style_slice.permute(2, 0, 1)  # Channels on top
                style_slice = self.normalize(style_slice)
                if self.skullstrip:
                    style_slice = uvir.SkullStrip(style_slice, label_style_slice, value = -1)

                slice_style_no = st_slices['image']
                if self.working_volume.slice_iterator[modality]['style_slice'] is None \
                        and self.working_volume.slice_iterator[modality]['same_slice']:
                        self.working_volume.slice_iterator[modality]['style_slice'] = style_slice
                        self.working_volume.slice_iterator[modality]['style_slice_no'] = slice_style_no

            # Build output dictionary
            if s == 0:
                # We save images
                output_dict['image'] = image_slice.type('torch.FloatTensor').unsqueeze(0)
                output_dict['label'] = label_slice.type('torch.FloatTensor').unsqueeze(0)
                output_dict['style_image'] = style_slice.type('torch.FloatTensor').unsqueeze(0)
                output_dict['this_seq'] = [modality]
                output_dict['this_dataset'] = [self.working_volume.style_dataset[modality]]
                output_dict['slice_no'] = [self.working_volume.slice_iterator[modality]['current']]
                output_dict['slice_style_no'] = [slice_style_no]
                output_dict['image_path'] = [self.working_volume.paths[modality]]
                output_dict['label_path'] = [self.working_volume.paths['Parcellation']]
                if modality in self.working_volume.non_corr.keys():
                    output_dict['image_slice_no'] = slices_im['image']
                else:
                    output_dict['image_slice_no'] = [self.working_volume.slice_iterator[modality]['current']]
            else:
                output_dict['image'] = torch.cat([output_dict['image'],
                                                  image_slice.type('torch.FloatTensor').unsqueeze(0)],
                                                  dim = 0)
                output_dict['label'] = torch.cat([output_dict['label'],
                                                  label_slice.type('torch.FloatTensor').unsqueeze(0)],
                                                  dim = 0)
                output_dict['style_image'] = torch.cat([output_dict['style_image'],
                                                    style_slice.type('torch.FloatTensor').unsqueeze(0)],
                                                   dim = 0)
                output_dict['this_seq'].append(modality)
                output_dict['this_dataset'].append(self.working_volume.style_dataset[modality])
                output_dict['slice_no'].append(self.working_volume.slice_iterator[modality]['current'])
                output_dict['slice_style_no'].append(slice_style_no)
                output_dict['image_path'].append(self.working_volume.paths[modality])
                output_dict['label_path'].append(self.working_volume.paths['Parcellation'])
                if modality in self.working_volume.non_corr.keys():
                    output_dict['image_slice_no'].append(slices_im['image'])
                else:
                    output_dict['image_slice_no'].append([self.working_volume.slice_iterator[modality][
                                                              'current']])

            self.working_volume.slice_iterator[modality]['current'] += 1
            if self.working_volume.slice_iterator[modality]['current'] == self.working_volume.slice_iterator[modality][
                'max']:
                break

        return output_dict

    def lenIteratorWV(self):

        return self.working_volume.volumes['Parcellation'].shape[self.cut_ids[self.cut]]

    def back_up_stored_slices(self, store):

        file_name = os.path.join(store, 'back_up_stored_slices.txt')
        with open(file_name, 'w') as f:
            f.write("---base\n")
            for name, slice in self.stored_slices['base'].items():
                f.write("%s:%d\n" % (name, slice))
            f.write("---style\n")
            for name, slice in self.stored_slices['style'].items():
                f.write("%s:%d\n" % (name, slice))
            f.close()

    def read_stored_slices(self, store):

        file_name = os.path.join(store, 'back_up_stored_slices.txt')

        with open(file_name, 'r') as f:
            all_lines = f.readlines()
            f.close()

        key = ""
        for line in all_lines:
            if "---" in line:
                key = line.strip("---").strip("\n")
            else:
                l_sp = line.strip("\n").split(":")
                self.stored_slices[key][l_sp[0]] = int(l_sp[-1])



class OfflineVolume():
    '''
    This class manages something called working_volume. A container of a label and its corresponding
    modality volumes, and the reconstructed versions of those.
    Meant to be used in offline (not using Persitent + Iterable dataset and Dataloaders) during test
    time.
    '''

    def __init__(self, image_paths, label_path, orig_dataset, cut, style_dataset = None,
                 labels_style = None, bound_normalization = False):

        '''

        :param image_paths: Dictionary, key: sequence/modality, value: path to the image volume.
        :param label_path: String. Path to the label.
        :param orig_dataset: Original dataset, to which the label belongs.
        :param cut: string, 'a' (axial), 's' (sagittal) or 'c' (coronal) to be applied on the slicing process.
        :param style_dataset: dictionary. Key: sequence/modality, value: string to the dataset to which
        each image volume belongs, if different from the label.
        :param labels_style: dictionary. Key: sequence/modality, value: path to the labels equivalent to the image
        volumes, in case they do not match the label specified in label_path. Recommended for skullstrip=True cases!
        '''

        self.volumes = {'Parcellation': self.loadVolume(label_path, label = True)}
        self.paths = {'Parcellation': label_path}
        self.affines = {}
        self.headers = {}
        self.non_corr = {}
        self.slice_iterator = {}
        self.style_dataset = {}
        self.cut = cut
        self.bound_normalization = bound_normalization
        self.cut_ids = self.cut_ids = {'a': 2, 's': 0, 'c': 1}
        metadata = np.load(label_path, allow_pickle=True)
        self.affines['Parcellation'] = metadata['label_affine']
        self.headers['Parcellation'] = metadata['label_header']
        for seq, path in image_paths.items():
            self.volumes['gt'+seq] = self.loadVolume(path, label = False)
            metadata = np.load(path, allow_pickle=True)
            self.affines['gt'+seq] = metadata['img_affine']
            self.headers['gt'+seq] = metadata['img_header']
            self.volumes['recon'+seq] = None
            self.paths[seq] = path
            self.slice_iterator[seq] = None
            self.style_dataset[seq] = style_dataset[seq] if style_dataset[seq] is not None else orig_dataset
            if labels_style is not None:
                # In case the volumes do not correspond to the label, labels_style is to provide the equivalent
                # Parcellations, in case of skullstrip it comes handy.
                if seq in labels_style.keys():
                    if labels_style[seq] != label_path:
                        self.non_corr[seq] = True
                        self.volumes['label'+seq] = self.loadVolume(labels_style[seq], label = True)
                        self.paths['label'+seq] = labels_style[seq]
                    else:
                        self.non_corr[seq] = False # Image volumes and label are same.
                else:
                    self.non_corr[seq] = False # Image volumes and label are same.
                self.skullStripVolume(seq, style_label=labels_style is not None)

        self.orig_dataset = orig_dataset

    def loadVolume(self, path, label = False):

        '''
        Loads a NPZ volume using Monai Transforms.
        :param path:
        :param label:
        :return:
        '''

        if label:
            loader = mt.LoadImageD(
                keys=('label_file'),
                reader="numpyreader",
                npz_keys=('label',)
            )
            loaded = loader({'label_file': path})
            if isinstance(loaded['label_file'], monai.data.meta_tensor.MetaTensor):
                loaded['label_file'] = loaded['label_file'].get_array()
            return torch.from_numpy(loaded['label_file'])

        else:
            loader = mt.LoadImageD(
                keys=('image_file'),
                reader="numpyreader",
                npz_keys=('img',)
            )
            loaded = loader({'image_file': path})
            if isinstance(loaded['image_file'], monai.data.meta_tensor.MetaTensor):
                loaded['image_file'] = loaded['image_file'].get_array()
            return torch.from_numpy(loaded['image_file'])

    def skullStripVolume(self, modality, style_label = False):
        '''
        For a specific loaded volume, skull strips it.
        :param modality:
        :return:
        '''

        if style_label:
            key = "label"+modality
        else:
            key = 'Parcellation'

        for slice_no in range(self.volumes['gt'+modality].shape[self.cut_ids[self.cut]]):
            # Skull strip needs a 3 x H x W image input (numpy)
            in_image = self.volumes['gt'+modality][..., slice_no, :].permute(-1,0,1)
            # It needs input label as CxHxW
            in_label = np.transpose(self.volumes[key][..., slice_no, :], [-1, 0, 1])
            out_image = uvir.SkullStrip(in_image, label = in_label, value=-1)
            self.volumes['gt'+modality][..., slice_no, 0] = out_image[0,...]

    def storeSlice(self, slice, modality):

        '''
        Stores a slice in the self.volumes[recon]
        :param slice:
        :return:
        '''

        if self.volumes['recon'+modality] is None:
            self.volumes['recon' + modality] = slice.squeeze(1)
        else:
            slice = slice.squeeze(1)
            self.volumes['recon' + modality] = torch.cat([self.volumes['recon' + modality],
                                                          slice], dim = 0)

    def storeReconstruction(self, path, modality):

        if self.cut == 'a':
            out_vol = self.volumes['recon' + modality].permute(1,2,0)
        elif self.cut == 'c':
            out_vol = self.volumes['recon' + modality].permute(1,0,2)

        if self.bound_normalization:
            out_vol = (out_vol + 1) / 2.0 * 255
        else:
            out_vol =255* (out_vol - out_vol.min())/(out_vol.max()-out_vol.min())

        if modality in self.non_corr.keys():
            nii_file = nib.Nifti1Image(out_vol.numpy(),
                                       affine=self.affines['label'])
        else:
            nii_file = nib.Nifti1Image(out_vol.numpy(),
                                       affine=self.affines['gt'+ modality])
        nib.save(nii_file, path)

    def normalizeVolume(self, modality):

        out = self.volumes['gt'+modality]
        for i in range(out.shape[self.cut_ids[self.cut]]):
            if self.cut == 'a':
                s = out[:, :, i].unsqueeze(0)
                s = torch.cat([s, s, s], 0)
                s = self.normalize(s)
                s = s[1, ...]
                out[:, :, i] = s
            elif self.cut == 'c':
                s = out[:, i, :].unsqueeze(0)
                s = torch.cat([s, s, s], 0)
                s = self.normalize(s)
                s = s[1, ...]
                out[:, i, :] = s
            elif self.cut == 's':
                s = out[i, :, :].unsqueeze(0)
                s = torch.cat([s, s, s], 0)
                s = self.normalize(s)
                s = s[1, ...]
                out[i, :, :] = s
        return out



    def normalize(self, tensor, standardize = True):

        normalise_fct = monai.transforms.NormalizeIntensity()
        return normalise_fct(tensor)

        # Deprecated
        # means_ = (0.5, 0.5, 0.5)
        # stds_ = (0.5, 0.5, 0.5)
        #
        # if tensor.min() == tensor.max():
        #     if standardize:
        #         return FTV.normalize(tensor, means_, stds_, True)
        #     else:
        #         return tensor
        # else:
        #     normalized = (tensor - tensor.min()) / (tensor.max() - tensor.min())
        #     if standardize:
        #         return FTV.normalize(normalized, means_, stds_, True)
        #     else:
        #         return normalized


class Intervaler:

    def __init__(self, max = 650, intervals = 5):
       self.lesion_intervals = np.linspace(0, max, intervals)
       self.current_interval = 1

    def getThreshold(self):

        if self.current_interval == len(self.lesion_intervals)-1:
            out = [self.lesion_intervals[self.current_interval], 9999]
        else:
            out = [self.lesion_intervals[self.current_interval],
                   self.lesion_intervals[self.current_interval+1]]
        return out

    def increaseIntervaler(self):

        self.current_interval+=1
        if self.current_interval == len(self.lesion_intervals):
            self.current_interval=1


