import matplotlib.pyplot as plt
import mlflow.pytorch
import torch
from monai import transforms
from monai.data import Dataset
from torch.utils.data import DataLoader
import os

# Load model -----------------------------------------------------------------------------------------------------------
# Add here the path to your local trained final model (for example,
# "/media/walter/Storage/Projects/conditioned_ldm/mlruns/0/42c436676d7c4e3fa708bea9d6f48782/artifacts/final_model")
final_model_path = "/home/vf19/Documents/brainSPADE_2D/LESION_GENERATOR/MLRUNS/mlruns/1/22ad595c36594b46bda8b164efeb97a4/artifacts/final_model"

device = torch.device("cuda")
vqvae = mlflow.pytorch.load_model(final_model_path)
vqvae = vqvae.to(device)
vqvae.eval()

# Custom dataloader for local files ------------------------------------------------------------------------------------
# Add here the path to the local npy file to test
all_paths = "/home/vf19/Documents/brainSPADE_2D/DATA/LESION_GENERATOR_NP_ONECH/test/labels"
save_to = "/home/vf19/Documents/brainSPADE_2D/LESION_GENERATOR/LDM/vq_vae_2/test"
if not os.path.isdir(save_to):
    os.makedirs(save_to)

test_dicts = []
for i in os.listdir(all_paths):
    test_dicts.append({'label': os.path.join(all_paths, i)})


# it is possible to implement a for loop to go through a tsv file and

test_transforms = transforms.Compose([
    transforms.LoadImaged(keys=["label"], reader="NumpyReader"),
    transforms.AddChanneld(keys=["label"]),
    transforms.SpatialPadd(
        keys=["label"],
        spatial_size=[192, 256],
        method="symmetric",
        mode="minimum"
    ),
    transforms.CenterSpatialCropd(
        keys=["label"],
        roi_size=[192, 256]
    ),
    transforms.ConcatItemsd(keys=["label"], name="image"),
    transforms.ToTensord(keys=["image"])
])

test_ds = Dataset(
    data=test_dicts,
    transform=test_transforms,
)
test_loader = DataLoader(
    test_ds,
    batch_size=16,
    num_workers=2,
    drop_last=False,
    pin_memory=True
)

batch = next(iter(test_loader))

img = batch["image"].to(device)
with torch.no_grad():
    z = vqvae.encoder(img)
    e, embed_idx, latent_loss = vqvae.codebook(z)
    x_tilde = vqvae.decoder(e)

    # or
    # indices = vqvae.encode_code(img)
    # x_tilde = vqvae.decode_code(indices)

    # or
    # x_tilde = vqvae.reconstruct(img)

print(x_tilde.shape)
for b in range(x_tilde.shape[0]):
    plt.imshow(x_tilde[b,0,...].cpu(), cmap = "gray")
    plt.axis('off')
    plt.savefig(os.path.join(save_to, "batch_%d" %b))
    plt.close('all')

# plt.imshow(x_tilde[0, 0].cpu(), cmap="gray")
# plt.axis('off')



plt.show()
