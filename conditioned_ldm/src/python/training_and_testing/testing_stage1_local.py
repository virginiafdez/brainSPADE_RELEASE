import matplotlib.pyplot as plt
import mlflow.pytorch
import torch
from monai import transforms
from monai.data import Dataset
from torch.utils.data import DataLoader
import os
from condldm_util import get_test_data_loader, get_testing_data_loader_pv
import numpy as np

# Load model -----------------------------------------------------------------------------------------------------------
# Add here the path to your local trained final model (for example,
# "/media/walter/Storage/Projects/conditioned_ldm/mlruns/0/42c436676d7c4e3fa708bea9d6f48782/artifacts/final_model")
final_model_path = "[PATH TO VAE MLRUNS FOLDER]/artifacts/best_model"
type = "vae"
if type not in ["vae", "vqvae"]:
    ValueError("Type not supported. Must be vae or vqvae.")
device = torch.device("cuda")
vqvae = mlflow.pytorch.load_model(final_model_path)
vqvae = vqvae.to(device)
vqvae.eval()

# Custom dataloader for local files ------------------------------------------------------------------------------------
# Add here the path to the local npy file to test
all_paths = "[PATH TO DATA]/dataset_test.tsv"
save_to = "" # Saving folder
if not os.path.isdir(save_to):
    os.makedirs(save_to)

# test_loader = get_test_data_loader(batch_size=12, testing_ids=all_paths, num_workers=0, drop_last=False,
#                                    )
test_loader = get_testing_data_loader_pv(batch_size=12, testing_ids=all_paths, num_workers=0, drop_last=False,
                                   )

COLOURS = np.array([[0, 0, 0],[0, 102, 204],[0, 0, 153],[51, 102, 255],[102, 102, 153],
                    [153, 153, 255],[255, 255, 0],[255, 102, 0],[255, 0, 0]]
                  )
USE_SOFTMAX = False

limit = 500
with torch.no_grad():
    for ind, data_i in enumerate(test_loader):
        if ind * 12 > limit:
            break
        img = data_i['image'].to(device)
        if type == 'vae':
            x_tilde, _, _ = vqvae(img)
        else:
            z = vqvae.encoder(img)
            e, embed_idx, latent_loss = vqvae.codebook(z)
            x_tilde = vqvae.decoder(e)

        for b_ind in range(x_tilde.shape[0]):
            gt_img = np.argmax(img[b_ind, ...].detach().cpu().numpy(), 0)
            if USE_SOFTMAX:
                out_img = torch.softmax(x_tilde[b_ind, ...].detach().cpu(), 0).numpy()
            else:
                out_img = x_tilde[b_ind, ...].detach().cpu().numpy()
            out_img = np.argmax(out_img, 0)
            out_img = np.concatenate([gt_img, out_img,], 1)
            out_img = COLOURS[out_img]
            out_channels = []
            out_channels_gt = []
            for c in range(x_tilde.shape[1]):
                out_channels.append(x_tilde[b_ind, c, ...].detach().cpu().numpy())
                out_channels_gt.append(img[b_ind, c, ...].detach().cpu().numpy())
            out_channels = np.concatenate(out_channels, -1)
            out_channels_gt = np.concatenate(out_channels_gt, -1)
            out_channels = np.concatenate([out_channels, out_channels_gt], 0)
            plt.figure(figsize=(20,15))
            plt.subplot(2,1,1)
            plt.imshow(out_img, cmap = 'gray')
            plt.title("Left: ground truth, Right: reconstruction")
            plt.axis('off')
            plt.subplot(2,1,2)
            plt.imshow(out_channels)
            plt.title("Channels (top: reconstruction, bottom: ground truth)")
            plt.axis('off')
            plt.savefig(os.path.join(save_to, 'test_img_%d.png' %(ind*x_tilde.shape[0]+b_ind)))
            plt.close('all')
