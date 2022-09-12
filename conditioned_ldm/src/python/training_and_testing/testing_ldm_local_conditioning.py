from pathlib import Path
import imageio
import matplotlib.pyplot as plt
import mlflow.pytorch
import numpy as np
import torch
from tqdm import tqdm
from condldm_models.ddim import DDIMSampler
import os

output_dir = "" # Where to save
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)
n_samples = 30
batch_size = 10
conditioning_classes = [0, 1, 2]
conditioning_names = {0: 'no lesion', 1: 'wmh', 2: 'tumour'}

# Load model -----------------------------------------------------------------------------------------------------------
# Add here the path to your local trained final condldm_models
final_stage1_model_path = "[YOUR_PATH]/artifacts/best_model" # Path to MLRUNS folder of the VAE
final_ldm_model_path = "[YOUR_PATH]/artifacts/best_model" # Path to MLRUNS folder of the LDM (this code > mlruns > [NUMBERS] > )

device = torch.device("cuda")
vqvae = mlflow.pytorch.load_model(final_stage1_model_path)
vqvae = vqvae.to(device)
vqvae.eval()

diffusion = mlflow.pytorch.load_model(final_ldm_model_path)
diffusion = diffusion.to(device)
diffusion.eval()
diffusion.log_every_t = 10
num_timesteps = 50

# Sampling process -----------------------------------------------------------------------------------------------------
# Add here the sample shape
n_passes = int(n_samples/batch_size)
for pass_i in range(n_passes):
    sample_shape = (batch_size, 3, 48, 64)
    # Conditioning tokens
    conditioning_tokens = torch.from_numpy(np.random.choice(conditioning_classes, batch_size, replace=True))
    conditioning_tokens_b = conditioning_tokens.unsqueeze(1).unsqueeze(1).unsqueeze(1)
    conditioning_tokens_b = conditioning_tokens_b.expand(
        [conditioning_tokens_b.shape[0], 1] + list(sample_shape)[2:])
    conditioning_tokens_b = conditioning_tokens_b.to(device)
    # Change here the frequency of intermediates stored
    latent_vectors = diffusion.p_sample_loop([conditioning_tokens_b], sample_shape, return_intermediates=False)
    with torch.no_grad():
        x_hat = vqvae.reconstruct_ldm_outputs(latent_vectors)
    for sample in range(x_hat.shape[0]):
        f = plt.figure()
        plt.imshow(torch.argmax(x_hat[sample, ...].detach().cpu(), 0))
        plt.title("Conditioning: %s" %(conditioning_names[int(conditioning_tokens[sample])]))
        plt.savefig(os.path.join(output_dir, "sample_%d.png" %(n_passes*pass_i+sample)))
        plt.close(f)