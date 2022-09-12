""" Script to fil the labels pixels that have no channel assign to them.

Based on:
https://stackoverflow.com/questions/41550979/fill-holes-with-majority-of-surrounding-values-python
"""
from collections import Counter

import numpy as np
from scipy.ndimage import label, binary_dilation


def impute(arr, mask):
    imputed_array = np.copy(arr)

    labels, count = label(mask)
    for idx in range(1, count + 1):
        hole = labels == idx
        surrounding_values = arr[binary_dilation(hole) & ~hole]
        most_frequent = Counter(surrounding_values).most_common(1)[0][0]
        imputed_array[hole] = most_frequent

    return imputed_array


image = np.load("/media/walter/Storage/Downloads/example_noisy_label.npy")

# get pixels without channel
mask = np.sum(image, axis=-1)
mask = mask == 0

# Convert on-hot encoding to single channel image
image_single_channel = np.argmax(image, axis=-1)

# Fill missing pixels
image_corrected = impute(image_single_channel, mask)

# Convert back to one-hot encoding
one_hot_image = (np.arange(image.shape[-1]) == image_corrected[..., None]).astype(int)
