import numpy as np
import torch
import monai.transforms as mt
class Slicer():

    def __init__(self, cut, output_channels = 1, non_corresponding = False):

        torch.manual_seed(0)
        self.non_corresponding = non_corresponding
        if cut not in ['a', 'c', 's']:
            ValueError("Cut can only be a, c or s (axial, coronal or sagittal).")
        else:
            self.cut = cut
        if output_channels%2 == 0:
            output_channels += 1
            Warning("Increased output channels by one. Must be an uneven number!")
        self.output_channels = output_channels
        self.cut_ids = {'a': 2, 's': 0, 'c': 1}  # Cut (a: axial, s: sagittal, c: coronal)


    def __call__(self, label, image, lesion = False, override_correspondance = None,
                 slice_no = None, lesion_th_low = 50, lesion_th_up = 99999):

        '''
        Slices label (HxWxDxCH) and image (HxWxD)
        :param label: Label volume HxWxCxCH
        :param image: Image volume HxWxC
        :param lesion: If true, retrieves a lesion slice (overriden by slice_no)
        :param override_correspondance: If True, the general attribute of forcing corresponding slices is overidden.
        :param slice_no: int, slice number, if deterministic.
        :return:
        '''

        if label.shape[-1] != image.shape:
            Warning("Label and Image have different shapes. If non-corresponding-slices is not on,"
                    "might do an out-of-bounds because of it.")

        if slice_no is None:
            if lesion:
                label_slice_nb = self.findLesions(label, threshold_low = lesion_th_low,
                                                  threshold_up=lesion_th_up)
                if label_slice_nb is None:
                    label_slice_nb = np.random.choice(range(1, label.shape[self.cut_ids[self.cut]] - 1))
            else:
                label_slice_nb = np.random.choice(range(1, label.shape[self.cut_ids[self.cut]]-1))

            if override_correspondance is not None:
                non_corresponding = override_correspondance
            else:
                non_corresponding = self.non_corresponding

            if non_corresponding:
                image_slice_nb = np.random.choice(range(1, image.shape[self.cut_ids[self.cut]]-1))
            else:
                image_slice_nb = label_slice_nb
        else:
            label_slice_nb = image_slice_nb = slice_no

        label_slice = self.chopSlice(label, coords=label_slice_nb, cut = self.cut)

        if self.output_channels == 1:
            try:
                image_slice = self.chopSlice(image, coords=image_slice_nb, cut = self.cut)
            except:
                print("So")
        else:
            image_slice = self.chopSlice(image,
                                         coords = [image_slice_nb - int(0.5*(self.output_channels -1)),
                                                   image_slice_nb + int(0.5*(self.output_channels +1))],
                                         cut = self.cut)



        return label_slice, image_slice, {'label': label_slice_nb, 'image': image_slice_nb}

    def chopSlice(self, vol3d, coords, cut='a'):
        '''
        Chops a sub-3D block from a 3D volume in the direction
        indicated by cut.
        :param vol3d: 3D volume in array format (or tensor)
        :param coords: Tuple or list containing the min and max indexes (max is not included)
        :param cut: 'a', 's' or 'c', indicating the direction of chopping (axial, coronal or sagittal).
        :return:
        '''
        if len(vol3d.shape) == 4:  # Label (HxWxDxC)
            if cut == 'a':
                if type(coords) == list:
                    return vol3d[:, :, coords[0]:coords[1], :]
                else:
                    return vol3d[:, :, coords, :]
            if cut == 's':
                if type(coords) == list:
                    return vol3d[coords[0]:coords[1], :, :, :]
                else:
                    return vol3d[coords, :, :, :]
            if cut == 'c':
                if type(coords) == list:
                    return vol3d[:, coords[0]:coords[1], :, :]
                else:
                    return vol3d[:, coords, :, :]
            else:
                ValueError("The cut must be 'c' (coronal), 'a' (axial) or 's' (sagittal).")
        elif len(vol3d.shape) == 3:  # Image (HxWxD)
            if cut == 'a':
                if type(coords) == list:
                    return vol3d[:, :, coords[0]:coords[1]]
                else:
                    return vol3d[:, :, coords]
            if cut == 's':
                if type(coords) == list:
                    return vol3d[coords[0]:coords[1], :, :]
                else:
                    return vol3d[coords, :, :]
            if cut == 'c':
                if type(coords) == list:
                    return vol3d[:, coords[0]:coords[1], :]
                else:
                    return vol3d[:, coords, :]
            else:
                ValueError("The cut must be 'c' (coronal), 'a' (axial) or 's' (sagittal).")

    def findLesions(self, label, threshold_low, threshold_up = 9999):

        '''
        Returns a random slice among all slices that contain a lesion (last 2 channels)
        :param label: HxWxDxCH array
        :param threshold: number of minimum pixels
        :return:
        '''

        lesions = np.where(label[..., -6:] > 0.35) # All lesion pixels
        counts_ = []
        counted = []
        for i in lesions[self.cut_ids[self.cut]]:
            if i not in counted:
                counts_.append([i, list(lesions[self.cut_ids[self.cut]]).count(i)])
                counted.append(i)
        counts_ = [i[0] for i in counts_ if i[1] > threshold_low and i[1] < threshold_up]
        if len(counts_) > 0:
            return np.random.choice(counts_)
        else:
            print("No lesions found in this label.")
            return None

class CropPad():

    def __init__(self, shape):

        self.new_shape = shape

    def __call__(self, img, islabel = False):

        '''
        Assumes that the channel dimension is at the end if img.dim() = 4.
        Pads / crops img so that its shape matches that of self.new_shape
        :param img: 3 or 4 dim tensor (HxWxDxCH or HxWxD). If dim = 4, disregards last.
        :param islabel. If True, the first channel of the fourth dimension is treated separately,
        and the pad values are filled with 1 instead of zeroes.
        :return:
        '''

        shape_ = img.shape

        pad_list = []
        crop_list = []
        if len(shape_) == 4:
            sub_shape = shape_[:-1]
        else:
            sub_shape = shape_

        # Loop around the shape to find the number of rows / columns to add or remove.
        for s_ind, s in enumerate(sub_shape):
            if self.new_shape[s_ind] == -1:
                pad_list.extend([0,0])
                crop_list.extend([0,0])
            else:
                diff = self.new_shape[s_ind] - s
                if diff > 0:
                    pad_list.extend([(int(np.ceil(diff / 2))), (int(np.floor(diff/2)))])
                    crop_list.extend([0,0])
                elif diff == 0:
                    pad_list.extend([0,0])
                    crop_list.extend([0,0])
                else:
                    crop_list.extend([(int(np.ceil((-diff)/2))), (int(np.floor((-diff)/2)))])
                    pad_list.extend([0,0])

        if len(shape_) == 4:
            pad_list.extend([0,0])
        # Padding phase
        pad_list.reverse() # The function needs the padding list to be inputted from last dim to first.
        if islabel:
            if shape_[-1] > 1:
                background = img[..., 0:1]
                rest = img[..., 1:]
                background = torch.nn.functional.pad(background, pad_list, value = background.max())
                rest = torch.nn.functional.pad(rest, pad_list, value = background.min())
                out_img = torch.cat([background, rest], dim = -1)
            else:
                out_img = torch.nn.functional.pad(img, pad_list, value=img.max())

        else:
            out_img = torch.nn.functional.pad(img, pad_list)

        # Cropping phase
        shape_ = out_img.shape
        if len(shape_) == 4:
            out_img = out_img[(crop_list[0]):(shape_[0] - crop_list[1]),
                              (crop_list[2]):(shape_[1] - crop_list[3]),
                              (crop_list[4]):(shape_[2] - crop_list[5]),
                              :]
        else:
            out_img = out_img[(crop_list[0]):(shape_[0] - crop_list[1]),
                              (crop_list[2]):(shape_[1] - crop_list[3]),
                              (crop_list[4]):(shape_[2] - crop_list[5])]

        return out_img




