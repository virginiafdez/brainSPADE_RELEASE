from vqvae_ldm.training_and_testing.models_.vqvae import VQVAE
from vqvae_ldm.training_and_testing.models_.ddpm_v2 import DDPM
from vqvae_ldm.training_and_testing.models_.ae_kl_v1 import AutoencoderKL
from vqvae_ldm.training_and_testing.models_.aekl_no_attention_3d import AutoencoderKL as AutoencoderKL_3D
from vqvae_ldm.training_and_testing.models_.ddpm_v2_conditioned import DDPM as DDPM_C
from omegaconf import OmegaConf

def define_VAE(path_to_config):

    config = OmegaConf.load(path_to_config)
    model = AutoencoderKL(**config["stage1"]["params"])
    return model

def define_VQVAE(path_to_config):

    config = OmegaConf.load(path_to_config)
    model = VQVAE(**config["stage1"]["params"])
    return model

def define_DDPM_unconditioned(path_to_config):

    config = OmegaConf.load(path_to_config)
    model = DDPM(**config["ldm"]["params"])
    return model

def define_DDPM_conditioned(path_to_config):
    config = OmegaConf.load(path_to_config)
    model = DDPM_C(**config["ldm"].get("params", dict()))
    return model

def define_VAE3D(path_to_config):
    config = OmegaConf.load(path_to_config)
    model = AutoencoderKL_3D(**config["stage1"]["params"])
    return model