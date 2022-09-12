import numpy as np
import os
import matplotlib as plt
from PIL import Image
import torch
import nibabel as nib

def compute_NMI_Loss(im1, im2):

    """
    Computes the Normalized Mutual Information Between two images
    :param im1: Pytorch tensor image
    :param im2: Pytorch tensor image
    :return: Because it's a loss, it'll give 1-NMI: 0 means that image 1 tells us everything about image 2
    """

    im1_bis = im1.clone()
    im2_bis = im2.clone()
    # We go from tensor to numpy
    im1_bis = im1_bis.detach().cpu().numpy()
    im2_bis = im2_bis.detach().cpu().numpy()
    # We compute the 2D histogram
    joint_his, _, _ = np.histogram2d(im1_bis.ravel(), im2_bis.ravel(), bins = 125)
    joint_his_NZ = joint_his != 0

    # NMI
    # Need to fetch probabilities
    joint_prob = joint_his/np.sum(joint_his) # Normalize histogram
    prob1 = np.sum(joint_prob, axis = 1)
    prob2 = np.sum(joint_prob, axis = 0)
    prob1_prob2 = prob1[:, None] * prob2[None,:]
    joint_prob_NZ = joint_prob > 0
    # Mutual Information
    MI = np.sum(joint_prob[joint_prob_NZ] * np.log(joint_prob[joint_prob_NZ]/prob1_prob2[joint_prob_NZ]))

    #Entropy
    prob1_NZ = prob1 > 0
    entropy1 = -np.sum(prob1[prob1_NZ] * np.log(prob1[prob1_NZ]))
    prob2_NZ = prob2> 0
    entropy2 = -np.sum(prob2[prob2_NZ] * np.log(prob2[prob2_NZ]))

    return 1-(2*MI/(entropy1+entropy2))


# Test function
slice = 101
T1_image = Image.open("/home/vf19/Documents/Project_MRES/borrowed_code/SPADE/SABRE/MODEL_1_L/pseudo2d_both/train/images/T1_10667_I_OOSIZ_SABREv3_0.png")
T1_image = np.asarray(T1_image)
FLAIR_image = Image.open("/home/vf19/Documents/Project_MRES/borrowed_code/SPADE/SABRE/MODEL_1_L/pseudo2d_both/train/images/FLAIR_10667_I_OOSIZ_SABREv3_0.png")
FLAIR_image = np.asarray(FLAIR_image)
T1_image_other = Image.open("/home/vf19/Documents/Project_MRES/borrowed_code/SPADE/SABRE/MODEL_1_L/pseudo2d_both/train/images/T1_172545I_SABREv3_0.png")
T1_image_other = np.asarray(T1_image_other)


# We display them
T1_image_t = torch.from_numpy(T1_image)
FLAIR_image_t = torch.from_numpy(FLAIR_image)
T1_image_other_t = torch.from_numpy(T1_image_other)

# Normalize MI

nmi_same = compute_NMI_Loss(T1_image_t, FLAIR_image_t)
nmi_diff = compute_NMI_Loss(T1_image_t, T1_image_other_t)
nmi_diff_seq = compute_NMI_Loss(T1_image_other_t, FLAIR_image_t)

print("Same image, different sequences: ", nmi_same)
print("Different images, same sequence: ", nmi_diff)
print("Different images, different sequences: ", nmi_diff_seq)

