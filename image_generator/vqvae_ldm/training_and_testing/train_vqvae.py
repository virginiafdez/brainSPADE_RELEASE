""" VQVAE with the encoder and decoder structure from the Pinaya et al. 2021 paper """
import argparse
import warnings
from pathlib import Path

import torch
import torch.optim as optim
from monai.config import print_config
from monai.utils import set_determinism
from omegaconf import OmegaConf
from tensorboardX import SummaryWriter

from vqvae_ldm.training_and_testing.models_.vqvae import VQVAE
from vqvae_ldm.training_and_testing.training_functions import train_vqvae
from vqvae_ldm.training_and_testing.util import get_training_data_loader, log_mlflow

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=2, help="Random seed to use.")
    parser.add_argument("--run_dir", help="Location of model to resume.")
    parser.add_argument("--training_ids", help="Location of file with training ids.")
    parser.add_argument("--validation_ids", help="Location of file with validation ids.")
    parser.add_argument("--config_file", help="Location of file with validation ids.")
    # training param
    parser.add_argument("--batch_size", type=int, default=256, help="Training batch size.")
    parser.add_argument("--n_epochs", type=int, default=25, help="Number of epochs to train.")
    parser.add_argument("--eval_freq", type=int, default=10, help="Number of epochs to betweeen evaluations.")
    parser.add_argument("--augmentation", type=int, default=1, help="Use of augmentation, 1 (True) or 0 (False).")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of loader workers")
    parser.add_argument("--experiment", help="Mlflow experiment name.")
    #Dataset
    parser.add_argument("--one_channel", type=str, default="False", help="Whether the input is a single channel with all different"
                                                        "tissues or a  OHE.")

    args = parser.parse_args()
    args.one_channel = bool(args.one_channel)
    return args


def main(args):
    set_determinism(seed=args.seed)
    print_config()

    output_dir = Path("/project/outputs/runs/")
    output_dir.mkdir(exist_ok=True, parents=True)

    run_dir = output_dir / args.run_dir
    if run_dir.exists() and (run_dir / "checkpoint.pth").exists():
        resume = True
    else:
        resume = False
        run_dir.mkdir(exist_ok=True)

    print(f"Run directory: {str(run_dir)}")
    print(f"Arguments: {str(args)}")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    writer_train = SummaryWriter(log_dir=str(run_dir / "train"))
    writer_val = SummaryWriter(log_dir=str(run_dir / "val"))

    print("Getting data...")
    train_loader, val_loader = get_training_data_loader(
        batch_size=args.batch_size,
        training_ids=args.training_ids,
        validation_ids=args.validation_ids,
        augmentation=bool(args.augmentation),
        num_workers=args.num_workers,
        one_channel=args.one_channel
    )

    print("Creating model...")
    config = OmegaConf.load(args.config_file)
    model = VQVAE(**config["stage1"]["params"])

    print(f"Let's use {torch.cuda.device_count()} GPUs!")
    device = torch.device("cuda")
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    model = model.to(device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), betas=(0.5, 0.9), lr=config["stage1"]["base_lr"])

    # Get Checkpoint
    best_loss = float("inf")
    start_epoch = 0
    if resume:
        print(f"Using checkpoint!")
        checkpoint = torch.load(str(run_dir / "checkpoint.pth"))
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
        best_loss = checkpoint["best_loss"]
    else:
        print(f"No checkpoint found.")

    # Train model
    print(f"Starting Training")
    val_loss = train_vqvae(
        model=model,
        start_epoch=start_epoch,
        best_loss=best_loss,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        one_channel=args.one_channel,
        n_epochs=args.n_epochs,
        eval_freq=args.eval_freq,
        writer_train=writer_train,
        writer_val=writer_val,
        device=device,
        run_dir=run_dir,
    )

    log_mlflow(
        model=model,
        config=config,
        args=args,
        experiment=args.experiment,
        run_dir=run_dir,
        val_loss=val_loss,
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
