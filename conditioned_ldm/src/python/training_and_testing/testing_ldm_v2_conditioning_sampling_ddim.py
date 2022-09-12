import mlflow.pytorch
import torch
import os
from condldm_models.ddim import DDIMSampler
import numpy as np
import matplotlib.pyplot as plt

save_to = "" # Saving directory
device = torch.device("cuda")
diffusion = mlflow.pytorch.load_model(
    "[PATH TO LDM MLRUNS FOLDER]/artifacts/best_model" # crossattn
    )

diffusion = diffusion.to(device)
diffusion.eval()
correct = True

vqvae = mlflow.pytorch.load_model(
    "[PATH TO VAE MLRUNS FOLDER]/artifacts/best_model")
vqvae = vqvae.to(device)
vqvae.eval()

# Colors
colors = np.array([[0, 0, 0],
                   [0, 102, 204],
                   [0, 0, 153],
                   [51, 102, 255],
                   [102, 102, 153],
                   [153, 153, 255],
                   [255, 255, 0],
                   [255, 102, 0],
                   [201, 14, 64]
                   ])

# Create conditioning maps.

cond_map = {0: 'tumour', 1: 'edema', 2: 'slice_no'} #### ! Modify according to the lesion conditioning you have applied.

sample_shape = (10, 3, 48, 64)
n_batches_sampled = 10
ddim_steps = 50
# Now we create the save_to path.
if not os.path.isdir(save_to):
    os.makedirs(save_to)

print(f"This models uses {diffusion.conditioning_key} conditioning")

counter = 0

for nb in range(n_batches_sampled):
    cond_list = list(np.random.uniform(0.2, 0.50, size = 10))
    cond_list = [np.round(i, 8) for i in cond_list]
    cond_list = np.array([[i] for i in cond_list])
    tumour = np.array([[i] for i in list(np.random.choice([0, 1], size = 10,))]) # 15
    edema = np.array([[i] for i in list(np.random.choice([0, 1], size=10, ))])
    cond_list = np.stack([tumour, edema, cond_list], -1).squeeze()
    cond = torch.FloatTensor(cond_list)  # Convert the list to tensors.
    cond = cond.to(device)
    with torch.no_grad():
        if diffusion.conditioning_key == "concat":
            cond = cond.unsqueeze(-1).unsqueeze(-1)
            cond = cond.expand(list(cond.shape[0:2]) + list(sample_shape[2:])).float()
        elif diffusion.conditioning_key == "crossattn":
            cond = cond.unsqueeze(1).float()
        elif diffusion.conditioning_key == "hybrid":
            cond_crossatten = cond.unsqueeze(1)
            cond_concat = cond.unsqueeze(-1).unsqueeze(-1)
            cond_concat = cond_concat.expand(list(cond.shape[0:2]) + list(sample_shape[2:]))
            cond = {
                'c_concat': [cond_concat.float()],
                'c_crossattn': [cond_crossatten.float()],
            }


    # Comment if you dont want to use DDIM sampling (faster)
    sampler = DDIMSampler(diffusion)
    latent_vectors, _ = sampler.sample(
        ddim_steps,
        conditioning=cond,
        batch_size=cond.shape[0],
        shape=sample_shape[1:],
        eta=1.0
    )

    # Comment if you don't want to use the normal sample
    #latent_vectors = diffusion.p_sample_loop(cond=cond, shape=sample_shape, return_intermediates=False)

    with torch.no_grad():
        sampled_image = vqvae.reconstruct_ldm_outputs(latent_vectors)

    for b in range(sampled_image.shape[0]):
        title_ = ""
        lesions = ""
        slice_number = ""
        for key, value in cond_map.items():
            if value == 'slice_no':
                slice_number += "%.3f" % cond_list[b][key]
            else:
                if cond_list[b][key] == 1:
                    lesions += value + ", "
        if lesions == "":
            lesions = "nolesion, "
        title_ = lesions + slice_number

        out_img = sampled_image[b, ...].detach().cpu().numpy()
        if correct:
            out_img[0, np.amax(out_img[1:, ...], 0) < 0.075] = 1.0
        out_indices = np.argmax(out_img, 0 )
        out_img = colors[out_indices]
        f = plt.figure(figsize=(7,7))
        plt.imshow(out_img)
        plt.title(title_)
        plt.axis('off')
        plt.savefig(os.path.join(save_to, "sample_%d.png" %(counter)))
        plt.close(f)
        counter += 1
