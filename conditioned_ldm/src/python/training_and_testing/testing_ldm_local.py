from pathlib import Path

import imageio
import matplotlib.pyplot as plt
import mlflow.pytorch
import numpy as np
import torch
from tqdm import tqdm

# Load model -----------------------------------------------------------------------------------------------------------
# Add here the path to your local trained final models_
final_stage1_model_path = "" # Path to MLRUNS folder of VAE
final_ldm_model_path = "" # Path to MLRUNS folder of LDM

device = torch.device("cuda")
vqvae = mlflow.pytorch.load_model(final_stage1_model_path)
vqvae = vqvae.to(device)
vqvae.eval()

diffusion = mlflow.pytorch.load_model(final_ldm_model_path)
diffusion = diffusion.to(device)
diffusion.eval()

# Sampling process -----------------------------------------------------------------------------------------------------
# Add here the sample shape
sample_shape = (4, 32, 48, 64)
# Change here the frequency of intermediates stored
diffusion.log_every_t = 100

latent_vectors, intermediates = diffusion.p_sample_loop(sample_shape, return_intermediates=True)
with torch.no_grad():
    x_hat = vqvae.reconstruct_ldm_outputs(latent_vectors)


plt.imshow(x_hat[0, 0].cpu())
plt.show()

# Create gif files of the markov chain
output_dir = Path(f"path/to/project/conditioned_ldm/outputs/folder/gifs/ImageName")
output_dir.mkdir(exist_ok=True, parents=True)
for i, intermediary in tqdm(enumerate(intermediates), total=len(intermediates)):
    with torch.no_grad():
        x_hat = vqvae.reconstruct_ldm_outputs(intermediary)

    img_row_0 = np.concatenate(
        (
            x_hat[0, 0].cpu().numpy(),
            x_hat[1, 0].cpu().numpy(),
        ),
        axis=1
    )

    img_row_1 = np.concatenate(
        (
            x_hat[2, 0].cpu().numpy(),
            x_hat[3, 0].cpu().numpy(),
        ),
        axis=1
    )

    img = np.concatenate(
        (
            img_row_0,
            img_row_1,
        ),
        axis=0
    )

    plt.imshow(img, cmap="gray")
    plt.axis('off')
    plt.savefig(output_dir / f"img_{i}.png", bbox_inches='tight')

with imageio.get_writer(output_dir / "movie.gif", mode='I') as writer:
    for i, _ in enumerate(intermediates):
        image = imageio.imread(output_dir / f"img_{i}.png")
        writer.append_data(image)
