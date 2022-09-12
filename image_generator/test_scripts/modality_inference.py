
import data
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
import os
import util_virginia as uvir

# This test is focused to produce data from a different modality
opt = TestOptions().parse()
opt.batchSize = 1
model = Pix2PixModel(opt)
modalities = opt.sequences

# Where you want to store the generated images
if not os.path.exists(opt.results_dir):
    os.makedirs(opt.results_dir)

# Create dataloader  model
dataloader = data.create_dataloader(opt)
model = Pix2PixModel(opt)

for i, data_i in enumerate(dataloader):
    this_seq = data_i['this_seq']  # Main sequence of the data item
    if i * opt.batchSize >= opt.how_many:
        break
    model.eval()  # Turn on test mode
    generated = model(data_i, 'generator_test')

    # Save images
    for batch in range(generated.shape[0]):
        all_path = os.path.join(opt.results_dir)
        # Name of the result to save:
        img_path = data_i['path']
        img_path_split = img_path[batch].split('/')
        name_short = img_path_split[-1]
        name_short = name_short.split('.png')[0].split("_")
        final_name =  "_".join(name_short[1:])
        # Store images
        label = data_i['label'][batch]  # Input label
        input = data_i['image'][batch]
        sequence = data_i['this_seq'][batch]
        synth = generated[batch]  # Input image
        # Store images
        tile = opt.batchSize > 8
        all_images_2save = []
        all_names_2save = []
        all_images_2save.append(label)  # Append the label
        all_images_2save.append(input)
        all_images_2save.append(synth)  # Append synthesized image
        all_names_2save.append("Label")
        all_names_2save.append("Input " + sequence)
        all_names_2save.append("Gen. " + sequence)

        uvir.saveFigs(all_images_2save, os.path.join(all_path, final_name + '_generated_' + sequence + '.png'),
                      True, 29, -1, True,
                      titles=all_names_2save, batch_accollades={}, index_label=0)