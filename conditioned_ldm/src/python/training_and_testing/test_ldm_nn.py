import matplotlib.pyplot as plt
import mlflow.pytorch
import torch
from util import get_test_data_loader
import os

# Load model -----------------------------------------------------------------------------------------------------------
# Add here the path to your local trained final models_
final_stage1_model_path = "/home/vf19/Documents/brainSPADE_2D/LESION_GENERATOR/MLRUNS/mlruns/1/b1495d156a8446af9398a1a05e21ee66/artifacts/final_model"
final_ldm_model_path = "/home/vf19/Documents/brainSPADE_2D/LESION_GENERATOR/MLRUNS/mlruns/2/4e4ce18f1dbb4515a4917f6b346f6e7b/artifacts/best_model"
test_ids = "/home/vf19/Documents/brainSPADE_2D/DATA/LESION_GENERATOR_NP_ONECH/dataset_train.tsv"
save_dir = "/home/vf19/Documents/brainSPADE_2D/LESION_GENERATOR/LDM/LDM_OC_2_BIG/test_nn"
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)


device = torch.device("cuda")
vqvae = mlflow.pytorch.load_model(final_stage1_model_path)
vqvae = vqvae.to(device)
vqvae.eval()

device = torch.device("cuda")
diffusion = mlflow.pytorch.load_model(final_ldm_model_path)
diffusion = diffusion.to(device)
diffusion.eval()

# Load training dataset
test_loader =  get_test_data_loader(10, test_ids, drop_last= False, num_workers=8,
                                    one_channel=True)
test_imgs = []
for i in test_loader:
    test_imgs.append(i['label'])
test_imgs = torch.cat(test_imgs, 0).squeeze(1) # Squeeze channel dim

# Sampling process -----------------------------------------------------------------------------------------------------
n_samples = 10

# Add here the sample shape
sample_shape = (n_samples, 32, 48, 64)

# Change here the frequency of intermediates stored
diffusion.log_every_t = 100

latent_vectors, intermediates = diffusion.p_sample_loop(sample_shape, return_intermediates=True)
with torch.no_grad():
    e, embed_idx, latent_loss = vqvae.codebook(latent_vectors)
    x_hat = vqvae.decoder(e)

for s in range(x_hat.shape[0]):
    sample_rep = x_hat[s, 0, ...].unsqueeze(0).repeat(test_imgs.shape[0], 1, 1).cpu()
    nearest_ind = torch.argmin(torch.mean(torch.sqrt((test_imgs-sample_rep)**2), [-1, -2])) # Minimum square error image
    f = plt.figure()
    out = torch.cat([x_hat[s, 0, ...].cpu(), test_imgs[nearest_ind, ...]], -1)
    plt.imshow(out)
    plt.title("Sample, NN from the training set")
    plt.savefig(os.path.join(save_dir, "sample_%d.png" %s))

