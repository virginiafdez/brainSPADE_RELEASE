from collections import OrderedDict
from pathlib import PosixPath

import monai
import numpy as np
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from additional_losses.unified_fl import Unified_FL
from condldm_util import log_reconstructions, log_inferred_ldm


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


# ----------------------------------------------------------------------------------------------------------------------
# AE KL PVS
# ----------------------------------------------------------------------------------------------------------------------
def train_ae_kl_pv(
        model,
        discriminator,
        perceptual_net,
        loss_fct: str,
        start_epoch: int,
        best_loss: float,
        train_loader,
        val_loader,
        optimizer_g,
        scheduler_g,
        optimizer_d,
        scheduler_d,
        n_epochs: int,
        eval_freq: int,
        writer_train: SummaryWriter,
        writer_val: SummaryWriter,
        device: torch.device,
        run_dir: PosixPath,
        gan_weight: float,
        perceptual_weight: float,
        kl_weight: float,
        weight_l1: float,
):
    scaler_g = GradScaler()
    scaler_d = GradScaler()
    raw_model = model.module if hasattr(model, "module") else model

    val_loss = eval_ae_kl_pv(
        model=model,
        discriminator=discriminator,
        perceptual_net=perceptual_net,
        loader=val_loader,
        device=device,
        step=len(train_loader) * start_epoch,
        writer=writer_val,
        kl_weight=kl_weight,
        gan_weight=gan_weight,
        perceptual_weight=perceptual_weight,
        loss_fct = loss_fct,
        weight_l1 = weight_l1
    )
    print(f"epoch {start_epoch} val loss: {val_loss:.4f}")
    for epoch in range(start_epoch, n_epochs):
        train_epoch_ae_kl_pv(
            model=model,
            discriminator=discriminator,
            loss_fct=loss_fct,
            weight_l1 = weight_l1,
            perceptual_net=perceptual_net,
            loader=train_loader,
            optimizer_g=optimizer_g,
            scheduler_g=scheduler_g,
            optimizer_d=optimizer_d,
            scheduler_d=scheduler_d,
            device=device,
            epoch=epoch,
            writer=writer_train,
            kl_weight=kl_weight,
            gan_weight=gan_weight,
            perceptual_weight=perceptual_weight,
            scaler_g=scaler_g,
            scaler_d=scaler_d,
        )

        if (epoch + 1) % eval_freq == 0:
            val_loss = eval_ae_kl_pv(
                model=model,
                discriminator=discriminator,
                perceptual_net=perceptual_net,
                loader=val_loader,
                device=device,
                step=len(train_loader) * epoch,
                writer=writer_val,
                kl_weight=kl_weight,
                gan_weight=gan_weight,
                perceptual_weight=perceptual_weight,
                loss_fct = loss_fct,
                weight_l1 = weight_l1
            )
            print(f"epoch {epoch + 1} val loss: {val_loss:.4f}")

            # Save checkpoint
            checkpoint = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "discriminator": discriminator.state_dict(),
                "optimizer_g": optimizer_g.state_dict(),
                "scheduler_g": scheduler_g.state_dict(),
                "optimizer_d": optimizer_d.state_dict(),
                "scheduler_d": scheduler_d.state_dict(),
                "best_loss": best_loss,
            }
            torch.save(checkpoint, str(run_dir / "checkpoint.pth"))

            if val_loss <= best_loss:
                print(f"New best val loss {val_loss}")
                best_loss = val_loss
                torch.save(raw_model.state_dict(), str(run_dir / "best_model.pth"))

    print(f"Training finished!")
    print(f"Saving final model...")
    torch.save(raw_model.state_dict(), str(run_dir / "final_model.pth"))

    return val_loss


def train_epoch_ae_kl_pv(
        model,
        discriminator,
        loss_fct,
        perceptual_net,
        loader,
        optimizer_g,
        scheduler_g,
        optimizer_d,
        scheduler_d,
        device: torch.device,
        epoch: int,
        writer: SummaryWriter,
        kl_weight: float,
        gan_weight: float,
        perceptual_weight: float,
        scaler_g,
        scaler_d,
        weight_l1: float
):
    model.train()
    discriminator.train()

    pbar = tqdm(enumerate(loader), total=len(loader))

    for step, x in pbar:
        img = x["image"].to(device)

        # GENERATOR
        optimizer_g.zero_grad(set_to_none=True)
        with autocast(enabled=False):
            x_hat, z_mu, z_sigma = model(img)
            if loss_fct == 'gdice':
                l1_loss = 0.1 * F.l1_loss(img.float(), x_hat.float(), reduction='none')
                x_hat = torch.softmax(x_hat.float(), 1)  # We need this to ensure perceptual loss works. y
                recon_loss_fct_g = monai.losses.GeneralizedDiceLoss(include_background=True,
                                                                    sigmoid=False, smooth_dr=0.0001,
                                                                    smooth_nr=0.0001)
                recon_los_fct_ng = monai.losses.DiceLoss(include_background=True,
                                                         sigmoid=False, smooth_dr=0.0001, smooth_nr=0.0001)

                recon_loss = (recon_loss_fct_g(x_hat, img.float()) + recon_los_fct_ng(x_hat,
                                                                                      img.float()) + l1_loss).mean()
            elif loss_fct == 'l1':
                #weights = torch.ones_like(x_hat)
                #weights[:, 6, ...] *= 5
                recon_loss = F.l1_loss(img.float(), x_hat.float(), reduction='none')
                #recon_loss = recon_loss * weights
                recon_loss = recon_loss.mean()
            elif loss_fct == 'bce':
                recon_loss = F.binary_cross_entropy_with_logits(img.float(), x_hat.float(), reduction='none')
                recon_loss = recon_loss.mean()
            elif loss_fct == 'unified_fl':
                x_hat = torch.softmax(x_hat.float(), 1)
                unified_fl = Unified_FL(n_classes=x_hat.shape[1])
                recon_loss = unified_fl(x_hat, img.float())
            else:
                ValueError("Unrecognised loss function")

            x_hat_perceptual = x_hat.view(-1, 1, x_hat.shape[2], x_hat.shape[3])
            img_perceptual = img.view(-1, 1, img.shape[2], img.shape[3])

            p_loss = torch.mean(
                perceptual_net.forward(
                    x_hat_perceptual.float(),
                    img_perceptual.float()
                )
            )

            kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3])
            kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

            logits_fake = discriminator(x_hat.contiguous().float())
            real_label = torch.ones_like(logits_fake, device=logits_fake.device)
            g_loss = F.mse_loss(logits_fake, real_label)

            loss = recon_loss + kl_weight * kl_loss + perceptual_weight * p_loss + gan_weight * g_loss

            loss = loss.mean()
            recon_loss = recon_loss.mean()
            p_loss = p_loss.mean()
            kl_loss = kl_loss.mean()
            g_loss = g_loss.mean()

            losses = OrderedDict(
                loss=loss,
                recon_loss=recon_loss,
                p_loss=p_loss,
                kl_loss=kl_loss,
                g_loss=g_loss,
            )

        scaler_g.scale(losses["loss"]).backward()
        scaler_g.unscale_(optimizer_g)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        scaler_g.step(optimizer_g)
        scaler_g.update()
        scheduler_g.step()

        # DISCRIMINATOR
        optimizer_d.zero_grad(set_to_none=True)

        with autocast(enabled=False
                      ):
            logits_fake = discriminator(x_hat.contiguous().detach())
            fake_label = torch.zeros_like(logits_fake, device=logits_fake.device)
            loss_d_fake = F.mse_loss(logits_fake, fake_label)
            logits_real = discriminator(img.contiguous().detach())
            real_label = torch.ones_like(logits_real, device=logits_real.device)
            loss_d_real = F.mse_loss(logits_real, real_label)

            d_loss = gan_weight * (loss_d_fake + loss_d_real) * 0.5

            d_loss = d_loss.mean()

        scaler_d.scale(d_loss).backward()
        scaler_d.unscale_(optimizer_d)
        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1)
        scaler_d.step(optimizer_d)
        scaler_d.update()
        scheduler_d.step()

        losses["d_loss"] = d_loss

        writer.add_scalar("lr_g", get_lr(optimizer_g), epoch * len(loader) + step)
        writer.add_scalar("lr_d", get_lr(optimizer_d), epoch * len(loader) + step)
        for k, v in losses.items():
            writer.add_scalar(f"{k}", v.item(), epoch * len(loader) + step)

        pbar.set_postfix(
            {
                "epoch": epoch,
                "loss": f"{losses['loss'].item():.6f}",
                "recon_loss": f"{losses['recon_loss'].item():.6f}",
                "p_loss": f"{losses['p_loss'].item():.6f}",
                "g_loss": f"{losses['g_loss'].item():.6f}",
                "d_loss": f"{losses['d_loss'].item():.6f}",
                "lr_g": f"{get_lr(optimizer_g):.6f}",
                "lr_d": f"{get_lr(optimizer_d):.6f}",
            },
        )


@torch.no_grad()
def eval_ae_kl_pv(
        model,
        discriminator,
        perceptual_net,
        loader,
        device: torch.device,
        step: int,
        writer: SummaryWriter,
        kl_weight: float,
        gan_weight: float,
        perceptual_weight: float,
        loss_fct: str,
        weight_l1 = 0.005
):
    model.eval()
    discriminator.eval()
    total_losses = OrderedDict()
    plot_batch = np.random.randint(len(loader))
    to_plot_x_hat = None
    for x_ind, x in enumerate(loader):
        img = x["image"].to(device)

        with autocast(enabled=False):
            x_hat, z_mu, z_sigma = model(img)
            if loss_fct == 'gdice':
                # l1_loss = weight_l1 * F.l1_loss(img.float(), x_hat.float(), reduction='none')
                ce_loss = weight_l1 * F.binary_cross_entropy_with_logits(img.float(), x_hat.float(), reduction='none')
                x_hat = torch.softmax(x_hat.float(), 1)  # We need this to ensure perceptual loss works. y
                recon_loss_fct_g = monai.losses.GeneralizedDiceLoss(include_background=True,
                                                                    sigmoid=False, smooth_dr=0.0001,
                                                                    smooth_nr=0.0001)
                recon_los_fct_ng = monai.losses.DiceLoss(include_background=True,
                                                         sigmoid=False, smooth_dr=0.0001, smooth_nr=0.0001)

                recon_loss = (recon_loss_fct_g(x_hat, img.float()) + recon_los_fct_ng(x_hat,
                                                                                      img.float()) + ce_loss).mean()
                # recon_loss = recon_loss_fct_g(x_hat, img.float())
                # recon_loss = (recon_loss_fct_g(x_hat, img.float()) + l1_loss).mean()
                # recon_loss = (recon_loss_fct_g(x_hat, img.float()) + ce_loss).mean()
            elif loss_fct == 'l1':
                weights = torch.ones_like(x_hat)
                weights[:, 6, ...] *= 5
                recon_loss = F.l1_loss(img.float(), x_hat.float(), reduction='none')
                recon_loss = recon_loss * weights
                recon_loss = recon_loss.mean()
            elif loss_fct == 'bce':
                recon_loss = F.binary_cross_entropy_with_logits(img.float(), x_hat.float(), reduction='none')
                recon_loss = recon_loss.mean()
            elif loss_fct == 'unified_fl':
                x_hat = torch.softmax(x_hat.float(), 1)
                unified_fl = Unified_FL(n_classes=x_hat.shape[1])
                recon_loss = unified_fl(x_hat, img.float())
            if torch.isnan(recon_loss):
                print("nan loss")

            x_hat_perceptual = x_hat.view(-1, 1, x_hat.shape[2], x_hat.shape[3])
            img_perceptual = img.view(-1, 1, img.shape[2], img.shape[3])

            p_loss = torch.mean(
                perceptual_net.forward(
                    x_hat_perceptual.float(),
                    img_perceptual.float()
                )
            )

            kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3])
            kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

            logits_fake = discriminator(x_hat.contiguous().float())
            real_label = torch.ones_like(logits_fake, device=logits_fake.device)
            g_loss = F.mse_loss(logits_fake, real_label)

            logits_fake = discriminator(x_hat.contiguous().detach())
            fake_label = torch.zeros_like(logits_fake, device=logits_fake.device)
            loss_d_fake = F.mse_loss(logits_fake, fake_label)
            logits_real = discriminator(img.contiguous().detach())
            real_label = torch.ones_like(logits_real, device=logits_real.device)
            loss_d_real = F.mse_loss(logits_real, real_label)
            d_loss = (loss_d_fake + loss_d_real) * 0.5

            loss = recon_loss + kl_weight * kl_loss + perceptual_weight * p_loss + gan_weight * g_loss

            loss = loss.mean()
            recon_loss = recon_loss.mean()
            p_loss = p_loss.mean()
            kl_loss = kl_loss.mean()
            g_loss = g_loss.mean()
            d_loss = d_loss.mean()

            losses = OrderedDict(
                loss=loss,
                recon_loss=recon_loss,
                p_loss=p_loss,
                kl_loss=kl_loss,
                g_loss=g_loss,
                d_loss=d_loss,
            )

            if x_ind == plot_batch:
                to_plot_x_hat = x_hat
                to_plot_img = img

        for k, v in losses.items():
            total_losses[k] = total_losses.get(k, 0) + v.item() * img.shape[0]

    for k in total_losses.keys():
        total_losses[k] /= len(loader.dataset)

    for k, v in total_losses.items():
        writer.add_scalar(f"{k}", v, step)

    log_reconstructions(
        img=to_plot_img,
        recons=to_plot_x_hat,
        writer=writer,
        step=step,
    )

    return total_losses["recon_loss"]


# ----------------------------------------------------------------------------------------------------------------------
# Latent Diffusion
# ----------------------------------------------------------------------------------------------------------------------
def train_ldm(
        model,
        vqvae,
        start_epoch: int,
        best_loss: float,
        train_loader,
        val_loader,
        optimizer,
        n_epochs: int,
        eval_freq: int,
        writer_train: SummaryWriter,
        writer_val: SummaryWriter,
        device: torch.device,
        run_dir: PosixPath,
):
    scaler = GradScaler()
    raw_model = model.module if hasattr(model, "module") else model

    val_loss = eval_ldm(
        model=model,
        vqvae=vqvae,
        loader=val_loader,
        device=device,
        step=len(train_loader) * start_epoch,
        writer=writer_val,
        sample=False,
    )
    print(f"epoch {start_epoch} val loss: {val_loss:.4f}")

    for epoch in range(start_epoch, n_epochs):
        train_epoch_ldm(
            model=model,
            vqvae=vqvae,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            writer=writer_train,
            scaler=scaler,
        )

        if (epoch + 1) % eval_freq == 0:
            val_loss = eval_ldm(
                model=model,
                vqvae=vqvae,
                loader=val_loader,
                device=device,
                step=len(train_loader) * epoch,
                writer=writer_val,
                sample=True if (epoch + 1) % (eval_freq * 2) == 0 else False,
            )

            print(f"epoch {epoch + 1} val loss: {val_loss:.4f}")

            # Save checkpoint
            checkpoint = {
                "epoch": epoch + 1,
                "diffusion": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_loss": best_loss,
            }
            torch.save(checkpoint, str(run_dir / "checkpoint.pth"))

            if val_loss <= best_loss:
                print(f"New best val loss {val_loss}")
                best_loss = val_loss
                torch.save(raw_model.state_dict(), str(run_dir / "best_model.pth"))

    print(f"Training finished!")
    print(f"Saving final model...")
    torch.save(raw_model.state_dict(), str(run_dir / "final_model.pth"))

    return val_loss


def train_epoch_ldm(
        model,
        vqvae,
        loader,
        optimizer,
        device: torch.device,
        epoch: int,
        writer: SummaryWriter,
        scaler: GradScaler,
):
    model.train()
    raw_vqvae = vqvae.module if hasattr(vqvae, "module") else vqvae

    pbar = tqdm(enumerate(loader), total=len(loader))
    for step, x in pbar:
        img = x["image"].to(device)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=False):
            with torch.no_grad():
                e = raw_vqvae.get_ldm_inputs(img.to(device))

            loss = model(e).mean()

        losses = OrderedDict(loss=loss)

        scaler.scale(losses["loss"]).backward()
        scaler.step(optimizer)
        scaler.update()

        writer.add_scalar("lr", get_lr(optimizer), epoch * len(loader) + step)

        for k, v in losses.items():
            writer.add_scalar(f"{k}", v.item(), epoch * len(loader) + step)

        pbar.set_postfix(
            {
                "epoch": epoch,
                "loss": f"{losses['loss'].item():.5f}",
                "lr": f"{get_lr(optimizer):.6f}"
            }
        )


@torch.no_grad()
def eval_ldm(
        model,
        vqvae,
        loader,
        device,
        step: int,
        writer: SummaryWriter,
        sample: bool = False,
):
    model.eval()
    raw_vqvae = vqvae.module if hasattr(vqvae, "module") else vqvae
    total_losses = OrderedDict()

    for x in loader:
        img = x["image"].to(device)
        with autocast(enabled=False):
            e = raw_vqvae.get_ldm_inputs(img.to(device))
            loss = model(e).mean()

        losses = OrderedDict(loss=loss)

        for k, v in losses.items():
            total_losses[k] = total_losses.get(k, 0) + v.item() * img.shape[0]

    for k in total_losses.keys():
        total_losses[k] /= len(loader.dataset)

    for k, v in total_losses.items():
        writer.add_scalar(f"{k}", v, step)

    if sample:
        infer_ldm(
            model=model,
            vqvae=vqvae,
            latent_shape=[8, ] + list(e.shape[1:]),
            writer=writer,
            step=step,
        )

    return total_losses["loss"]


def infer_ldm(
        model,
        vqvae,
        latent_shape,
        writer: SummaryWriter,
        step: int,
):
    model.eval()
    raw_model = model.module if hasattr(model, "module") else model
    raw_vqvae = vqvae.module if hasattr(vqvae, "module") else vqvae
    # Sample image
    sample_shape = latent_shape
    # Sample and denoise
    latent_vectors = raw_model.p_sample_loop(sample_shape, return_intermediates=False)
    with torch.no_grad():
        x_hat = raw_vqvae.reconstruct_ldm_outputs(latent_vectors)

    log_inferred_ldm(
        imgs=x_hat,
        writer=writer,
        step=step,
    )


# ----------------------------------------------------------------------------------------------------------------------
# Latent Diffusion Conditioned
# ----------------------------------------------------------------------------------------------------------------------
def train_conditioned_ldm(
        model,
        vqvae,
        start_epoch: int,
        best_loss: float,
        train_loader,
        val_loader,
        optimizer,
        n_epochs: int,
        eval_freq: int,
        writer_train: SummaryWriter,
        writer_val: SummaryWriter,
        device: torch.device,
        run_dir: PosixPath,
        lesions=None,
):
    scaler = GradScaler()
    raw_model = model.module if hasattr(model, "module") else model

    val_loss = eval_conditioned_ldm(
        model=model,
        vqvae=vqvae,
        loader=val_loader,
        device=device,
        step=len(train_loader) * start_epoch,
        writer=writer_val,
        sample=False,
        lesions=lesions,
    )
    print(f"epoch {start_epoch} val loss: {val_loss:.4f}")

    for epoch in range(start_epoch, n_epochs):
        train_epoch_conditioned_ldm(
            model=model,
            vqvae=vqvae,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            writer=writer_train,
            scaler=scaler,
            lesions=lesions
        )

        if (epoch + 1) % eval_freq == 0:
            val_loss = eval_conditioned_ldm(
                model=model,
                vqvae=vqvae,
                loader=val_loader,
                device=device,
                step=len(train_loader) * epoch,
                writer=writer_val,
                sample=True if (epoch + 1) % (eval_freq * 2) == 0 else False,
                lesions=lesions
            )

            print(f"epoch {epoch + 1} val loss: {val_loss:.4f}")

            # Save checkpoint
            checkpoint = {
                "epoch": epoch + 1,
                "diffusion": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_loss": best_loss,
            }
            torch.save(checkpoint, str(run_dir / "checkpoint.pth"))

            if val_loss <= best_loss:
                print(f"New best val loss {val_loss}")
                best_loss = val_loss
                torch.save(raw_model.state_dict(), str(run_dir / "best_model.pth"))

    print(f"Training finished!")
    print(f"Saving final model...")
    torch.save(raw_model.state_dict(), str(run_dir / "final_model.pth"))

    return val_loss


def train_epoch_conditioned_ldm(
        model,
        vqvae,
        loader,
        optimizer,
        device: torch.device,
        epoch: int,
        writer: SummaryWriter,
        scaler: GradScaler,
        lesions=None,
):
    model.train()
    raw_vqvae = vqvae.module if hasattr(vqvae, "module") else vqvae
    raw_model = model.module if hasattr(model, "module") else model

    pbar = tqdm(enumerate(loader), total=len(loader))
    for step, x in pbar:
        img = x["image"] #.to(device)
        cond = []
        for lesion in lesions:
            cond.append(x[lesion])  # One hot encoded representation.
        cond = torch.stack(cond, -1)  # So that we have a BxC (C = number of lesions)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=False):
            with torch.no_grad():
                e = raw_vqvae.get_ldm_inputs(img.to(device))
                if raw_model.conditioning_key == "concat":
                    # Here, we have multiple lesions that can be combined into one label.
                    # Example: [0, 1, 0] and lesions = ['wmh', 'tumour', 'edema'] will stand for
                    # no wmh, tumour, no edema.
                    cond = cond.unsqueeze(-1).unsqueeze(-1)  # BxCxHxW
                    cond = cond.expand(list(cond.shape[0:2]) + list(e.shape[2:])).float()
                    cond = cond.to(device)
                elif raw_model.conditioning_key == "crossattn":
                    cond = cond.unsqueeze(1).float()
                    cond = cond.to(device)
                elif raw_model.conditioning_key == "hybrid":
                    cond_crossatten = cond.unsqueeze(1)
                    cond_concat = cond.unsqueeze(-1).unsqueeze(-1)
                    cond_concat = cond_concat.expand(list(cond.shape[0:2]) + list(e.shape[2:]))
                    cond = {
                        'c_concat': [cond_concat.float().to(device)],
                        'c_crossattn': [cond_crossatten.float().to(device)],
                    }

            outputs = model(e, c=cond)
            loss = outputs[0].mean()

        losses = OrderedDict(loss=loss)

        scaler.scale(losses["loss"]).backward()
        scaler.step(optimizer)
        scaler.update()

        writer.add_scalar("lr", get_lr(optimizer), epoch * len(loader) + step)

        for k, v in losses.items():
            writer.add_scalar(f"{k}", v.item(), epoch * len(loader) + step)

        pbar.set_postfix(
            {
                "epoch": epoch,
                "loss": f"{losses['loss'].item():.5f}",
                "lr": f"{get_lr(optimizer):.6f}"
            }
        )

@torch.no_grad()
def eval_conditioned_ldm(
        model,
        vqvae,
        loader,
        device,
        step: int,
        writer: SummaryWriter,
        sample: bool = False,
        lesions=None
):
    model.eval()
    raw_vqvae = vqvae.module if hasattr(vqvae, "module") else vqvae
    raw_model = model.module if hasattr(model, "module") else model
    total_losses = OrderedDict()

    for x in loader:
        img = x["image"].to(device)

        cond = []
        for lesion in lesions:
            cond.append(x[lesion])  # One hot encoded representation.
        cond = torch.stack(cond, -1)  # So that we have a BxC (C = number of lesions)

        with autocast(enabled=True):
            with torch.no_grad():
                e = raw_vqvae.get_ldm_inputs(img.to(device))
                if raw_model.conditioning_key == "concat":
                    # Here, we have multiple lesions that can be combined into one label.
                    # Example: [0, 1, 0] and lesions = ['wmh', 'tumour', 'edema'] will stand for
                    # no wmh, tumour, no edema.
                    cond = cond.unsqueeze(-1).unsqueeze(-1)  # BxCxHxW
                    cond = cond.expand(list(cond.shape[0:2]) + list(e.shape[2:])).float()
                    cond = cond.to(device)
                elif raw_model.conditioning_key == "crossattn":
                    cond = cond.unsqueeze(1).float()
                    cond = cond.to(device)
                elif raw_model.conditioning_key == "hybrid":
                    cond_crossatten = cond.unsqueeze(1)
                    cond_concat = cond.unsqueeze(-1).unsqueeze(-1)
                    cond_concat = cond_concat.expand(list(cond.shape[0:2]) + list(e.shape[2:]))
                    cond = {
                        'c_concat': [cond_concat.float().to(device)],
                        'c_crossattn': [cond_crossatten.float().to(device)],
                    }

            outputs = model(e, c=cond)
            loss = outputs[0].mean()
        losses = OrderedDict(loss=loss)

        for k, v in losses.items():
            total_losses[k] = total_losses.get(k, 0) + v.item() * img.shape[0]

    for k in total_losses.keys():
        total_losses[k] /= len(loader.dataset)

    for k, v in total_losses.items():
        writer.add_scalar(f"{k}", v, step)

    # Get conditionings for samples
    cond = []
    for lesion in lesions:
        cond.append(x[lesion])  # One hot encoded representation.
    cond = torch.stack(cond, -1)  # So that we have a BxC (C = number of lesions)
    cond = cond[:4]

    if raw_model.conditioning_key == "concat":
        # Here, we have multiple lesions that can be combined into one label.
        # Example: [0, 1, 0] and lesions = ['wmh', 'tumour', 'edema'] will stand for
        # no wmh, tumour, no edema.
        cond = cond.unsqueeze(-1).unsqueeze(-1)  # BxCxHxW
        cond = cond.expand(list(cond.shape[0:2]) + list(e.shape[2:])).float()
        cond = cond.to(device)
    elif raw_model.conditioning_key == "crossattn":
        cond = cond.unsqueeze(1).float()
        cond = cond.to(device)
    elif raw_model.conditioning_key == "hybrid":
        cond_crossatten = cond.unsqueeze(1)
        cond_concat = cond.unsqueeze(-1).unsqueeze(-1)
        cond_concat = cond_concat.expand(list(cond.shape[0:2]) + list(e.shape[2:]))
        cond = {
            'c_concat': [cond_concat.float().to(device)],
            'c_crossattn': [cond_crossatten.float().to(device)],
        }

    if sample:
        infer_conditioned_ldm(
            model=raw_model,
            vqvae=raw_vqvae,
            cond=cond,
            sample_shape=[4, ] + list(e.shape[1:]),
            writer=writer,
            step=step,
        )

    return total_losses["loss"]


def infer_conditioned_ldm(
        model,
        vqvae,
        cond,
        sample_shape,
        writer: SummaryWriter,
        step: int,
):
    model.eval()

    # Sample image
    # Sample and denoise
    latent_vectors = model.p_sample_loop(cond=cond, shape=sample_shape, return_intermediates=False)
    with torch.no_grad():
        x_hat = vqvae.reconstruct_ldm_outputs(latent_vectors)

    log_inferred_ldm(
        imgs=x_hat,
        writer=writer,
        step=step,
    )
