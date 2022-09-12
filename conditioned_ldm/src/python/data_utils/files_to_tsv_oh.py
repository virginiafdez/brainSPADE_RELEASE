'''
Creates TSV files for training datasets for VAE-LDM label generators.
Expected format of the dataset: at least train, validation and/or test folders, containing the labels in npy format.
    The names of the labels must contain 1) the slice number 2) the diseases present.
    These are specified in the file name, each in a different separation token (portion separated by "_").
    Example: Parcellation_BRATS-TCIA_sub-159_space-MNI152NLin2009aSym_tumour-edema_65.npy
    Slice number always goes in the last position (65).
    Disease always go in the one before last position. If multiple disease present, separation is with "-".
How to run: It accepts 6 system arguments:
1) directory containing the training, val and test folders
2) directory where you want the TSV paths to point (can be a relative volume path)
3) whether you want to process the training, validation and/or test folders. This is a string. For training write tr,
for testing write te and for validation write va. Combinations are accepted. Example: trvate (all), trva (training and validation).
4) Lesions present in this dataset that you want to take into account, separated by "-". Example: wmh-tumour.
5) Whether to add a slice number column or not. If so, adds the slice number normalised by 256.
6) Name of the folder. Within each folder (training, validation and test), the program will look for a folder named labels.
If you specify the name of the folder, it will look for a folder with that name. Example: pv_maps.

'''

import os
import csv
import sys

root_dir = sys.argv[1]
save_dir = sys.argv[2]
train_test_val= str(sys.argv[3])
lesions = sys.argv[4].split("-")
add_slice_no = sys.argv[5]
if add_slice_no == "True":
    add_slice_no = True
elif add_slice_no == "False":
    add_slice_no = False
name_folder = sys.argv[6]


def findSimilarName(original, directory, slice_index=-1, extension='.png'):
    all_files = os.listdir(directory)
    keywords = ["sub", "ses"]
    root = original.strip(extension)
    for f_sup in all_files:
        f = f_sup.strip(extension)
        f_sp = f.split("_")
        keys = []
        for key in keywords:
            positives_targ = [sp for sp in f_sp if key in sp]
            if len(positives_targ) == 0:
                continue
            else:
                positives_targ = positives_targ[0]
                ori_sp = root.split("_")
                positives_ori = [sp for sp in ori_sp if key in sp]
                if len(positives_ori) == 0:
                    continue
                else:
                    positives_ori = positives_ori[0]
                    if positives_targ == positives_ori:
                        keys.append(True)
                    else:
                        keys.append(False)
                        break

        # Now we compare the slice number
        slice = f_sp[slice_index]
        slice_ori = root.split("_")[-1]
        if slice != slice_ori:
            keys.append(False)
        if False not in keys:
            return f + extension

    return None

def createTSV(root_dir, save_dir, flag,lesions,
              add_slice_no = False, name_folder = ""):

    lesion_dicts = {"nolesion":0, 'wmh':1, 'tumour':2}

    with open(os.path.join(root_dir, "dataset_%s.tsv" % flag), 'wt') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        row_names = ['label']
        for lesion in lesions:
            row_names.append(lesion)
        if add_slice_no:
            row_names.append('slice_no')
        tsv_writer.writerow(row_names)
        if name_folder == "":
            name_folder = "labels"
        lind = True
        for label in os.listdir(os.path.join(root_dir, flag, name_folder)):
            to_write = [os.path.join(save_dir, flag, name_folder, label)]
            lesion_type = label.split("_")[-2].lower().split("-")
            lesions_in = {}
            for lesion in lesions:
                if lesion in lesion_type:
                    lesions_in[lesion] = 1
                else:
                    lesions_in[lesion] = 0
            for lesion in lesions:
                to_write.append(lesions_in[lesion])
            if add_slice_no:
                slice_no = float(label.split("_")[-1].split(".")[0]) / 256
                to_write.append(slice_no)
            if lind:
                print(to_write)
                lind = False
            tsv_writer.writerow(to_write)

if "tr" in train_test_val:
    createTSV(root_dir, save_dir, 'train', lesions = lesions, add_slice_no=add_slice_no,
              name_folder = name_folder)
if "va" in train_test_val:
    createTSV(root_dir, save_dir, 'validation', lesions = lesions, add_slice_no=add_slice_no,
              name_folder = name_folder)
if "te" in train_test_val:
    createTSV(root_dir, save_dir, 'test', lesions = lesions ,add_slice_no=add_slice_no,
              name_folder = name_folder)

