import mlflow.pytorch
import os
import torch

# path_file = "/home/vf19/Documents/brainSPADE_2D/LESION_GENERATOR/MLRUNS/mlruns/2/" \
#             "4e4ce18f1dbb4515a4917f6b346f6e7b/artifacts/best_model/data/model.pth"
path_file = "/home/vf19/Documents/brainSPADE_2D/LESION_GENERATOR/MLRUNS/mlruns/4/98b9825ff598438fafb150c9f96a4e36/artifacts/best_model"
save_in = "/home/vf19/Documents/brainSPADE_2D/LESION_GENERATOR/LDM/VAE_LDM_CONDITIONED_TMN"
name = "best_ldm_model.pth"

model = mlflow.pytorch.load_model(path_file)
torch.save(model.state_dict(), os.path.join(save_in, name))