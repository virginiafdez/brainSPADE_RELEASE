import torch.nn.functional as nn
import torch
import numpy as np
import torchvision.transforms.functional as FTV

class Transform_brainSPADE:

    def __init__(self, new_size = None, normalize = False):
        self.new_size = new_size
        self.normalize = normalize

    def __call__(self, input):
        # Convert to tensor
        out = torch.from_numpy(input.astype('float32'))
        out[torch.isnan(out)] = 0.0
        # If length of shape is 2, we unsqueeze one.
        if len(input.shape) == 2:
            out = out.unsqueeze(0)
        elif len(input.shape) == 3:
            out = out.permute(2,0,1) # Channels first
        else:
            ValueError("Invalid shape of tensor for input to transforms.")

        if self.new_size is not None:
            out = self.resize(out)
        if self.normalize:
            out = self.normalize_0_1(out)
            out = self.standardize_img(out)
        return out

    def resize(self, img):
        '''
        Resizes image with interpolation. The Height and Width are input in self.new_size
        :param img: Input image.
        :return:
        '''
        n_unsq = 0
        while len(img.shape) < 4:
            img = img.unsqueeze(0)
            n_unsq += 1
        img = nn.interpolate(img, self.new_size)
        for n in range(n_unsq):
            img = img.squeeze(0)
        return img

    def standardize_img(self, image, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), inplace=False):
        '''
        Normalises the input image using a distribution with mean (mean), std (std).
        :param image:
        :param mean:
        :param std:
        :param inplace:
        :return:
        '''


        return FTV.normalize(image, mean, std, inplace)

    def normalize_0_1(self, image):

        if image.min() == image.max():
            Warning("Could not normalize between 0 and 1 because min = max")
            return (image - image.min())

        else:
            out = (image - image.min()) / (image.max()-image.min())
            return out