from pathlib import PosixPath

import matplotlib.pyplot as plt
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
from mlflow import start_run
from monai import transforms
from monai.data import CacheDataset
from omegaconf import OmegaConf
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import numpy as np
import monai

# ----------------------------------------------------------------------------------------------------------------------
# DATA LOADING
# ----------------------------------------------------------------------------------------------------------------------
def get_data_dicts(
        ids_path: str,
        shuffle: bool = False,
        conditioned: bool = False,
        lesions=None,
):
    """ Get data dicts for data loaders."""
    df = pd.read_csv(ids_path, sep="\t")
    if shuffle:
        df = df.sample(frac=1, random_state=1)

    data_dicts = []
    for index, row in df.iterrows():
        out_dict = {
            "label": row["label"],
        }

        if conditioned:
            for lesion in lesions:
                if lesion in row.keys():
                    out_dict[lesion] = float(row[lesion])

        data_dicts.append(out_dict)

    print(f"Found {len(data_dicts)} subjects.")
    return data_dicts


def get_training_data_loader_pv(
        batch_size: int,
        training_ids: str,
        validation_ids: str,
        only_val: bool = False,
        augmentation: bool = True,
        drop_last: bool = False,
        num_workers: int = 8,
):
    # Define transformations
    val_transforms = transforms.Compose(
        [
            transforms.LoadImaged(keys=["label"]),
            transforms.SpatialPadd(
                keys=["label"],
                spatial_size=[256, 256],
                method="symmetric",
                mode="minimum"
            ),
            transforms.CenterSpatialCropd(
                keys=["label"],
                roi_size=[256, 256]
            ),
            transforms.ConcatItemsd(keys=["label"], name="image"),
            transforms.ToTensord(keys=["image"])
        ]
    )

    if augmentation:
        train_transforms = transforms.Compose(
            [
                transforms.LoadImaged(keys=["label"]),
                transforms.SpatialPadd(
                    keys=["label"],
                    spatial_size=[256, 256],
                    method="symmetric",
                    mode="minimum"
                ),
                transforms.CenterSpatialCropd(
                    keys=["label"],
                    roi_size=[256, 256]
                ),
                transforms.ConcatItemsd(keys=["label"], name="image"),
                transforms.RandAffined(
                    keys=["image"],
                    prob=0.0,
                    rotate_range=[0, 0.1],
                    shear_range=[0.001, 0.15],
                    scale_range=[0, 0.3],
                    padding_mode='zeros',
                    mode='nearest',
                ),
                transforms.ToTensord(keys=["image"])
            ]
        )

    else:
        train_transforms = val_transforms

    val_dicts = get_data_dicts(
        ids_path=validation_ids,
        shuffle=False,
    )
    val_ds = CacheDataset(
        data=val_dicts,
        transform=val_transforms,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=True
    )

    if only_val:
        return val_loader

    train_dicts = get_data_dicts(
        ids_path=training_ids,
        shuffle=False,
    )
    train_ds = CacheDataset(
        data=train_dicts,
        transform=train_transforms,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=True
    )

    return train_loader, val_loader


def get_testing_data_loader_pv(
        batch_size: int,
        testing_ids: str,
        drop_last: bool = False,
        num_workers: int = 8,
):
    # Define transformations
    test_transforms = transforms.Compose([
        transforms.LoadImaged(keys=["label"]),
        transforms.SpatialPadd(
            keys=["label"],
            spatial_size=[256, 256],
            method="symmetric",
            mode="minimum"
        ),
        transforms.CenterSpatialCropd(
            keys=["label"],
            roi_size=[256, 256]
        ),
        transforms.ConcatItemsd(keys=["label"], name="image"),
        transforms.ToTensord(keys=["image"])
    ])

    test_dicts = get_data_dicts(
        ids_path=testing_ids,
        shuffle=False,
    )

    test_ds = CacheDataset(
        data=test_dicts,
        transform=test_transforms,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=True
    )

    return test_loader


def get_training_data_loader_pv_conditioned(
        batch_size: int,
        training_ids: str,
        validation_ids: str,
        lesions: list,
        only_val: bool = False,
        augmentation: bool = True,
        drop_last: bool = False,
        num_workers: int = 8,
):
    """
    Get data loaders for scenario with Partial Volume maps and conditioning.
    """
    # Define transformations
    val_transforms = transforms.Compose([
        transforms.LoadImaged(keys=["label"]),
        transforms.SpatialPadd(
            keys=["label"],
            spatial_size=[256, 256],
            method="symmetric",
            mode="minimum"
        ),
        transforms.CenterSpatialCropd(
            keys=["label"],
            roi_size=[256, 256]
        ),
        transforms.ConcatItemsd(keys=["label"], name="image"),
        transforms.ToTensord(keys=["image", ] + lesions)
    ])

    if augmentation:
        train_transforms = transforms.Compose([
            transforms.LoadImaged(keys=["label"]),
            transforms.SpatialPadd(
                keys=["label"],
                spatial_size=[256, 256],
                method="symmetric",
                mode="minimum"
            ),
            transforms.CenterSpatialCropd(
                keys=["label"],
                roi_size=[256, 256]
            ),
            transforms.ConcatItemsd(keys=["label"], name="image"),
            transforms.RandAffined(
                keys=["image"],
                prob=1.0,
                rotate_range=[0, 0.1],
                shear_range=[0.001, 0.15],
                scale_range=[0, 0.3],
                padding_mode='zeros',
                mode='nearest',
            ),
            transforms.ToTensord(keys=["image"] + lesions)
        ])

    else:
        train_transforms = val_transforms

    val_dicts = get_data_dicts(
        ids_path=validation_ids,
        shuffle=False,
        conditioned=True,
        lesions=lesions
    )
    val_ds = CacheDataset(
        data=val_dicts,
        transform=val_transforms,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=True
    )

    if only_val:
        return val_loader

    train_dicts = get_data_dicts(
        ids_path=training_ids,
        shuffle=False,
        conditioned=True,
        lesions=lesions
    )
    train_ds = CacheDataset(
        data=train_dicts,
        transform=train_transforms,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=True
    )

    return train_loader, val_loader


def get_training_data_loader_pv_conditioned_vol(
        batch_size: int,
        training_ids: str,
        validation_ids: str,
        lesions: list,
        only_val: bool = False,
        augmentation: bool = True,
        drop_last: bool = False,
        num_workers: int = 8,
):
    """
    Get data loaders for scenario with Partial Volume maps and conditioning.
    """
    # Define transformations
    val_transforms = transforms.Compose([
        transforms.LoadImaged(keys=["label"]),
        transforms.SpatialPadd(
            keys=["label"],
            spatial_size=[256, 256, 135],
            method="symmetric",
            mode="edge"
        ),
        transforms.CenterSpatialCropd(
            keys=["label"],
            roi_size=[256, 256, 128]
        ),
        transforms.ConcatItemsd(keys=["label"], name="image"),
        transforms.ToTensord(keys=["image", ] + lesions)
    ])

    if augmentation:
        train_transforms = transforms.Compose([
            transforms.LoadImaged(keys=["label"]),
            transforms.SpatialPadd(
                keys=["label"],
                spatial_size=[256, 256, 135],
                method="symmetric",
                mode="edge"
            ),
            transforms.CenterSpatialCropd(
                keys=["label"],
                roi_size=[256, 256, 128]
            ),
            transforms.ConcatItemsd(keys=["label"], name="image"),
            transforms.RandAffined(
                keys=["image"],
                prob=1.0,
                rotate_range=[0, 0.1],
                shear_range=[0.001, 0.15],
                scale_range=[0, 0.3],
                padding_mode='zeros',
                mode='nearest',
            ),
            transforms.ToTensord(keys=["image"] + lesions)
        ])

    else:
        train_transforms = val_transforms

    val_dicts = get_data_dicts(
        ids_path=validation_ids,
        shuffle=False,
        conditioned=True,
        lesions=lesions
    )
    val_ds = CacheDataset(
        data=val_dicts,
        transform=val_transforms,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=False
    )

    if only_val:
        return val_loader

    train_dicts = get_data_dicts(
        ids_path=training_ids,
        shuffle=False,
        conditioned=True,
        lesions=lesions
    )
    train_ds = CacheDataset(
        data=train_dicts,
        transform=train_transforms,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=False
    )

    return train_loader, val_loader



def get_testing_data_loader_pv_conditioned_vol(
        batch_size: int,
        testing_ids: str,
        lesions: list,
        drop_last: bool = False,
        num_workers: int = 8,
):
    """
    Get data loaders for scenario with Partial Volume maps and conditioning.
    """
    # Define transformations
    test_transforms = transforms.Compose([
        transforms.LoadImaged(keys=["label"]),
        transforms.SpatialPadd(
            keys=["label"],
            spatial_size=[256, 256, 135],
            method="symmetric",
            mode="edge"
        ),
        transforms.CenterSpatialCropd(
            keys=["label"],
            roi_size=[256, 256, 128]
        ),
        transforms.ConcatItemsd(keys=["label"], name="image"),
        transforms.ToTensord(keys=["image", ] + lesions)
    ])

    test_dicts = get_data_dicts(
        ids_path=testing_ids,
        shuffle=False,
        conditioned=True,
        lesions=lesions
    )
    test_ds = CacheDataset(
        data=test_dicts,
        transform=test_transforms,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=False
    )

    return test_loader

def get_conditioned_data_dicts(
        ids_path: str,
        shuffle: bool = False,
):
    """ Get data dicts for data loaders."""
    df = pd.read_csv(ids_path, sep="\t")
    if shuffle:
        df = df.sample(frac=1, random_state=1)

    data_dicts = []
    for index, row in df.iterrows():
        data_dicts.append(
            {
                "image": row["label"],
                "lesion_type": row["lesion_type"]
            }
        )

    print(f"Found {len(data_dicts)} subjects.")
    return data_dicts


def get_conditioned_training_data_loader(
        batch_size: int,
        training_ids: str,
        validation_ids: str,
        only_val: bool = False,
        augmentation: bool = True,
        drop_last: bool = False,
        num_workers: int = 8,
):
    # Define transformations
    val_transforms = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image"]),
            transforms.AddChanneld(keys=["image"]),
            transforms.SpatialPadd(
                keys=["image"],
                spatial_size=[256, 256],
                method="symmetric",
                mode="minimum"
            ),
            transforms.CenterSpatialCropd(
                keys=["image"],
                roi_size=[256, 256]
            ),
            transforms.ToTensord(keys=["image", "lesion_type"])
        ]
    )

    if augmentation:
        train_transforms = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image"]),
                transforms.AddChanneld(keys=["image"]),
                transforms.SpatialPadd(
                    keys=["image"],
                    spatial_size=[256, 256],
                    method="symmetric",
                    mode="minimum"
                ),
                transforms.CenterSpatialCropd(
                    keys=["image"],
                    roi_size=[256, 256]
                ),
                transforms.RandAffined(
                    keys=["image"],
                    prob=1.0,
                    rotate_range=[0, 0.1],
                    shear_range=[0.001, 0.15],
                    scale_range=[0, 0.3],
                    padding_mode='zeros',
                    mode='nearest',
                ),
                transforms.ToTensord(keys=["image", "lesion_type"])
            ]
        )
    else:
        train_transforms = val_transforms

    val_dicts = get_conditioned_data_dicts(
        ids_path=validation_ids,
        shuffle=False,
    )
    val_ds = CacheDataset(
        data=val_dicts,
        transform=val_transforms,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=True
    )

    if only_val:
        return val_loader

    train_dicts = get_conditioned_data_dicts(
        ids_path=training_ids,
        shuffle=False,
    )
    train_ds = CacheDataset(
        data=train_dicts,
        transform=train_transforms,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=True
    )

    return train_loader, val_loader


def get_test_data_loader(
        batch_size: int,
        testing_ids: str,
        drop_last: bool = False,
        num_workers: int = 8,
):
    # Define transformations
    test_transforms = transforms.Compose(
        [
            transforms.LoadImaged(keys=["label"]),
            transforms.AddChanneld(keys=["label"]),
            transforms.SpatialPadd(
                keys=["label"],
                spatial_size=[256, 256],
                method="symmetric",
                mode="minimum"
            ),
            transforms.CenterSpatialCropd(
                keys=["label"],
                roi_size=[256, 256]
            ),
            transforms.ConcatItemsd(keys=["label"], name="image"),
            transforms.ToTensord(keys=["image"])
        ]
    )

    test_dicts = get_data_dicts(
        ids_path=testing_ids,
        shuffle=False,
    )
    test_ds = CacheDataset(
        data=test_dicts,
        transform=test_transforms,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=True
    )

    return test_loader


# ----------------------------------------------------------------------------------------------------------------------
# LOGS
# ----------------------------------------------------------------------------------------------------------------------
def recursive_items(dictionary):
    for key, value in dictionary.items():
        if type(value) is dict:
            yield from recursive_items(value)
        else:
            yield (key, value)


def log_mlflow(
        model,
        config,
        args,
        experiment: str,
        run_dir: PosixPath,
        val_loss: float,
):
    """Log model and performance on Mlflow system"""
    config = {
        **OmegaConf.to_container(config),
        **vars(args)
    }
    print(f"Setting mlflow experiment: {experiment}")
    mlflow.set_experiment(experiment)

    with start_run():
        print(f"MLFLOW URI: {mlflow.tracking.get_tracking_uri()}")
        print(f"MLFLOW ARTIFACT URI: {mlflow.get_artifact_uri()}")

        for key, value in recursive_items(config):
            mlflow.log_param(key, value)

        mlflow.log_artifacts(str(run_dir), artifact_path="events")
        mlflow.log_metric(f"loss", val_loss, 0)

        raw_model = model.module if hasattr(model, "module") else model

        mlflow.pytorch.log_model(raw_model, "final_model")
        raw_model.load_state_dict(torch.load(str(run_dir / "best_model.pth")))
        mlflow.pytorch.log_model(raw_model, "best_model")


def get_figure(
        img,
        recons,
        n_batch = 5,
):
    '''
    Gets a figure to plot in tensorboard.
    img: BxCxHxW input images
    recons: BxCxHxW generated images
    '''
    colors = np.array([[0, 0, 0],
                       [0, 102, 204],
                       [0, 0, 153],
                       [51, 102, 255],
                       [102, 102, 153],
                       [153, 153, 255],
                       [255, 255, 0],
                       [255, 14, 64],
                       [255, 102, 0]
                       ]
                     )
    tile_gt = []
    tile_sy = []
    selected = np.random.choice(list(range(img.shape[0])), n_batch)
    for i in selected:
        tile_gt.append(np.argmax(img[i, ...].detach().cpu().numpy(), 0))
        tile_sy.append(np.argmax(recons[i, ...].detach().cpu().numpy(), 0))
    tile_gt = np.concatenate(tile_gt, -1)
    tile_sy = np.concatenate(tile_sy, -1)
    tile = np.concatenate([tile_sy, tile_gt], 0)
    tile = colors[tile]

    f = plt.figure(figsize=(3*n_batch, 6))
    plt.imshow(tile)
    plt.axis('off')
    return f

def get_figure_channels(
        img,
        recons,
):
    '''
    Gets a figure to plot in tensorboard.
    img: BxCxHxW input images
    recons: BxCxHxW generated images
    '''
    selected_item = np.random.choice(range(img.shape[1]))
    tile_gt = []
    tile_rec = []
    for j in range(recons.shape[1]):
        tile_gt.append(img[selected_item, j, ...].detach().cpu().numpy())
        tile_rec.append(recons[selected_item, j, ...].detach().cpu().numpy())
    tile_gt = np.concatenate(tile_gt, -1)
    tile_rec = np.concatenate(tile_rec, -1)
    whole = np.concatenate([tile_gt, tile_rec], 0)
    f = plt.figure(figsize=(3*img.shape[1], 10))
    plt.imshow(whole)
    plt.title("top: ground truth, bottom: reconstruction")
    plt.axis('off')
    return f


def get_figure_ldm(
        imgs,
):
    colors = np.array([[0, 0, 0],
                       [0, 102, 204],
                       [0, 0, 153],
                       [51, 102, 255],
                       [102, 102, 153],
                       [153, 153, 255],
                       [255, 255, 0],
                       [255, 14, 64],
                       [255, 102, 0]
                       ]
                      )

    if imgs.shape[1] > 1:
        imgs = torch.argmax(imgs, 1).detach().cpu().numpy() # Channel
    out_imgs = []
    for i in range(imgs.shape[0]):
        out_imgs.append(colors[imgs[i,...]])
    out_imgs = np.concatenate(out_imgs, 1)
    f = plt.figure(dpi=300)
    plt.imshow(out_imgs)
    plt.axis('off')
    return f

def log_reconstructions(
        img: torch.Tensor,
        recons: torch.Tensor,
        writer: SummaryWriter,
        step: int,
):
    fig = get_figure(
        img,
        recons,
    )
    writer.add_figure(f"RECONSTRUCTION", fig, step)

    fig_chan = get_figure_channels(img, recons)

    writer.add_figure(f" CHANNEL-BASED RECONSTRUCTION", fig_chan, step)

def log_inferred_ldm(
        imgs: torch.Tensor,
        writer: SummaryWriter,
        step: int,
):
    fig = get_figure_ldm(
        imgs,
    )
    writer.add_figure(f"INFERENCE LDM", fig, step)

def log_3d_img(
        rec_imgs: torch.Tensor,
        gt_imgs: torch.Tensor,
        writer: SummaryWriter,
        step: int,
        n_plots = 1,
        tag = "val_images"
):
    colors = np.array([[0, 0, 0],
                       [0, 102, 204], # lighter blue
                       [0, 0, 153], # darkish blue
                       [51, 102, 255], # even darkish blue
                       [102, 102, 153],
                       [153, 153, 255],
                       [255, 255, 0],
                       [255, 102, 0],
                       [201, 14, 64]
                       ])

    # Plot images
    if n_plots > rec_imgs.shape[0]:
        n_plots = rec_imgs.shape[0]

    # Tensor should be BxCxHxWxD
    rec_imgs = rec_imgs.detach().cpu()
    rec_img = torch.argmax(rec_imgs, 1).numpy() # B x H x W x D
    if gt_imgs is not None:
        gt_imgs = gt_imgs.detach().cpu()
        gt_img = torch.argmax(gt_imgs, 1).numpy() # B x H x W x D

    img_ids = np.random.choice(range(rec_img.shape[0]), n_plots, replace = False)
    img_ids = [int(i) for i in img_ids]

    for i in img_ids:

        # rec_img_ = colors[rec_img[i, ...]].transpose(-1, 0, 1, 2) # H x W x D x 3 (color) > 3 x H x W x D
        # gt_img_ = colors[gt_img[i, ...]].transpose(-1, 0, 1, 2) # H x W x D x 3 (color) > 3 x H x W x D
        rec_img_ = np.expand_dims(rec_img[i, ...], 0) # 1 x H x W x D
        if gt_imgs is not None:
            gt_img_ = np.expand_dims(gt_img[i, ...], 0)  # 1 x H x W x D
            out_img = np.concatenate([rec_img_, gt_img_], 1)
        else:
            out_img = rec_img_

        individual_channels = []
        individual_channels_gt = []
        for ch in range(rec_imgs.shape[1]):
            individual_channels.append(np.expand_dims((rec_imgs[i, ch, ...]*8).numpy(), 0))
            if gt_imgs is not None:
                individual_channels_gt.append(np.expand_dims((gt_imgs[i, ch, ...] * 8).numpy(),0))
            # Each time is 3 x H x W x D
        individual_channels = np.concatenate(individual_channels, 2)
        if gt_imgs is not None:
            individual_channels_gt = np.concatenate(individual_channels_gt, 2)
            individual_channels = np.concatenate([individual_channels, individual_channels_gt], 1)

        out_img = np.concatenate([out_img, individual_channels], 2)
        out_img = np.expand_dims(out_img, 0)
        #individual_channels = np.expand_dims(individual_channels, 0)
        monai.visualize.plot_2d_or_3d_image(data=out_img, step=step, writer=writer,
                                            index=0,
                                            tag="%s_%d" %(tag,i), max_channels=1,
                                            max_frames=6, frame_dim=-1)
        # monai.visualize.plot_2d_or_3d_image(data=individual_channels, step=step, writer=writer,
        #                                     index=0,
        #                                     tag="individual_channels_%d" %i, max_channels=1,
        #                                     max_frames=6, frame_dim=-1)
