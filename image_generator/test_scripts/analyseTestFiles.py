'''
SCRIPT that samples images from a certain modality and stores ground truths in another folder, then
you are show each and you have to rate whether it's real or fake.
'''


import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time

# The file structure is considered to be the same you use when you run blindGenerateTest.py, which
# generates the files you have to use here.

checkpoint_dir = "/home/vf19/Documents/brainSPADE_2D/brainSPADE_1DEC"
model = "BRAINSPADEV3_20"
test_name = 'TEST'
modality = 'T1'
file_ground_truth = os.path.join(checkpoint_dir, model, test_name, 'blind_test', modality, modality + "_directory.txt")
file_rating = os.path.join(checkpoint_dir, model, test_name, 'blind_test', modality, modality + "_rating.txt")
save_to = os.path.join(checkpoint_dir, model, test_name, 'blind_test', modality, modality + "_summary.txt")

def show(file_rating, folder):
    print("Beginning test for modality %s...\n" %modality)
    list_images = [i for i in os.listdir(os.path.join(folder)) if ".png" in i]
    gather_sample_names = {}
    with open(file_rating, 'r') as f:
        lines = f.readlines()
        header = lines[0]
    for line in lines[1:]:
        gather_sample_names[line.strip("\t").strip("\t\n").strip("\n")] = 0
    for ind, im in enumerate(list_images):
        img = Image.open(os.path.join(folder, im))
        f = plt.figure()
        plt.imshow(img)
        plt.show()

        correct_input = False
        while not correct_input:
            input_tag = input("Real (1) or Fake (0)?\n")
            if input_tag in ["R", "r"]:
                plt.imshow(img)
                plt.show()
            else:
                try:
                    suggested_tag = int(input_tag)
                    if suggested_tag not in [0, 1]:
                        print("Please, input 0 or 1\n")
                    else:
                        correct_input = True
                except:
                    continue

        gather_sample_names[im] = suggested_tag
        plt.close(f)

    new_lines = [header.strip("\n")]
    for line in lines[1:]:
        line_sp = line.split("\t")
        new_lines.append("%s\t%d" %(line_sp[0], gather_sample_names[line_sp[0]]))

    with open(file_rating, 'w') as f:
        f.write("\n".join(new_lines))


def analyseTestFiles(file_gt, file_rating, save_to):

    '''
    For two symmetric files containing one sample name + 1 or 0 separated from the sample line
    by a \t sign, one file being the ground truth and the other file having the user's ratings,
    calculates the true positive rate, true negative rate etc.
    :param file_gt:
    :param file_rating:
    :param save_to:
    :return:
    '''

    with open(file_gt, 'r') as f:
        lines_gt = f.readlines()
        f.close()
    with open(file_rating, 'r') as f:
        lines_rate = f.readlines()
        f.close()

    samples = {}
    for s_ind, s in enumerate(lines_gt):
        if s_ind == 0:
            continue
        s_split = s.replace("\n", "").split("\t")
        sample_name = s_split[0]
        tag = int(s_split[-1])
        if sample_name in samples.keys():
            samples[sample_name]['gt'] = tag
        else:
            samples[sample_name] = {}
            samples[sample_name]['gt'] = tag
        s_split = lines_rate[s_ind].replace("\n", "").split("\t")
        sample_name = s_split[0]
        if tag == "":
            ValueError("Line %s not populated in the rating file!" %(sample_name))
        tag = int(s_split[-1])
        if sample_name in samples.keys():
            samples[sample_name]['rate'] = tag
        else:
            samples[sample_name] = {}
            samples[sample_name]['rate'] = tag

    # Now we calculate the TPR, FPR, FNR, TNR
    results = {'tp': 0,
               'fp': 0,
               'tn': 0,
               'fn': 0}
    for s_id, s in samples.items():
        if s['gt'] == 0 and s['rate'] == 0:
            results['tn'] += 1
        if s['gt'] == 1 and s['rate'] == 1:
            results['tp'] += 1
        if s['gt'] == 0 and s['rate'] == 1:
            results['fp'] += 1
        if s['gt'] == 1 and s['rate'] == 0:
            results['fn'] += 1

    tpr = 100*(results['tp'] / (results['tp']+results['fn']))
    fpr = 100*(results['fp'] / (results['tn']+results['fp']))
    precision = 100*(results['tp']/(results['tp']+results['fp']))
    accuracy = 100*((results['tp']+results['tn'])/(results['tn']+results['tp']+
                                              results['fn']+results['fp']))
    tnr = 100*(results['tn']/(results['tn']+results['fp']))

    with open(save_to, 'w') as f:
        f.write("True positive rate %.2f\n" %(tpr))
        f.write("False positive rate %.2f\n" % (fpr))
        f.write("Precision (TP/(TP+FP)) %.2f\n" % (precision))
        f.write("Accuracy ((TP+TN)/(FP+FN+TP+TN))) %.2f\n" % (accuracy))
        f.write("True negative rate %.2f\n" % (tnr))

show(file_rating, os.path.join(checkpoint_dir, model,test_name, 'blind_test',
                               modality))
analyseTestFiles(file_ground_truth, file_rating, save_to)

