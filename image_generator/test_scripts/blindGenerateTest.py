'''
This script generates fake images and saves them along the fakes, to perform a blind test (AKA: user saying if the image is real or fake).
After the storage, call analyseTestFiles to 1) label the images 2) get a statistical analysis.
'''

from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
import os
import moreutils as uvir
import numpy as np
from data.spadenai_v2_sliced import SpadeNaiSlice
from data.spadenai_v2 import SpadeNai
import monai

uvir.set_deterministic(True, 1)

# Initialisation
opt = TestOptions().parse()
how_many = 150
opt.batchSize = 1
model = Pix2PixModel(opt)
model = model.eval()
modalities = opt.sequences
if not os.path.exists(opt.results_dir):
    os.makedirs(opt.results_dir)

if not os.path.exists(os.path.join(opt.results_dir, 'blind_test')):
    os.makedirs(os.path.join(opt.results_dir, 'blind_test'))

ssim_values = []  # Values yielded for structural similarity

for mod in modalities:

    opt.fix_seq = mod
    if opt.dataset_type == 'volume':
        dataset_container = SpadeNai(opt, mode= 'test')
    else:
        dataset_container = SpadeNaiSlice(opt, mode = 'test')

    dataloader = monai.data.dataloader.DataLoader(
        dataset_container.sliceDataset,
        batch_size=opt.batchSize, shuffle=False,
        num_workers=int(opt.nThreads), drop_last=opt.isTrain)

    all_ids = list(range(0, 2*len(dataset_container)))
    sample_names_ordered = ["sample_" + str(i) + ".png" for i in all_ids]
    samples = {}
    for s in sample_names_ordered:
        samples[s] = 0
    np.random.shuffle(all_ids)
    if not os.path.exists(os.path.join(opt.results_dir, 'blind_test', mod)):
        os.makedirs(os.path.join(opt.results_dir, 'blind_test', mod))
        with open(os.path.join(opt.results_dir, 'blind_test', mod, mod + "_directory.txt"), 'w') as f:
            f.write("Image name\tReal/Fake\n")
        with open(os.path.join(opt.results_dir, 'blind_test', mod, mod + "_rating.txt"), 'w') as f:
            f.write("Image name\tReal/Fake\n")
        counter = 0
        for i, data_i in enumerate(dataloader):
            if i * opt.batchSize >= how_many:
                break
            generated = model(data_i, 'generator_test').detach().cpu()

            # We skull strip the generated image and the ground truth.
            for b in range(generated.shape[0]):  # For each image of the batch (should be one)
                # We process the path to get the output name
                name_gt = "sample_%s.png" % (str(all_ids[counter]))
                name = "sample_%s.png" % (str(all_ids[counter + 1]))
                samples[name_gt] = 1 # The ground truths are 1, the rest are 0
                label = data_i['label'][b].detach().cpu()  # Input label
                input = data_i['image'][b, 1:2, ...].detach().cpu()
                synth = generated[b]  # Input image
                uvir.saveSingleImage(input, label, skullstrip=True, path=os.path.join(
                    opt.results_dir, 'blind_test', mod, name_gt), denormalize = True)
                uvir.saveSingleImage(synth, label, skullstrip=True, path=os.path.join(
                    opt.results_dir, 'blind_test', mod, name), denormalize = True)
                counter += 2
        # We write the text files:
        for sample in sample_names_ordered:
            with open(os.path.join(opt.results_dir, 'blind_test', mod, mod + "_directory.txt"), 'a') as f:
                f.write("%s\t%d\n" % (sample, samples[sample]))
            with open(os.path.join(opt.results_dir, 'blind_test', mod, mod + "_rating.txt"), 'a') as f:
                f.write("%s\t\n" % (sample))

    else:
        Warning("blind_test folder exists in the checpoint test directory. Please delete it to run "
                "the function again")
        continue
