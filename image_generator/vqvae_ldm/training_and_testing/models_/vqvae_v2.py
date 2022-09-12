""" Unified Encoder and decoder but with my previous Quantizer"""
from typing import Tuple, Dict

import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import autocast


class Quantizer(nn.Module):
    def __init__(self, n_embed, embed_dim, commitment_cost=0.25, decay=0.99, eps=1e-5):
        super().__init__()
        self.embedding = nn.Embedding(n_embed, embed_dim)
        self.embedding.weight.requires_grad = False

        self.embed_dim = embed_dim
        self.n_embed = n_embed
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.eps = eps

        self.register_buffer('N', torch.zeros(n_embed))
        self.register_buffer('embed_avg', self.embedding.weight.data.clone())

    def forward(self, x):
        b, c, h, w = x.shape
        weight = self.embedding.weight

        # convert inputs from BCHW -> BHWC and flatten input
        flat_inputs = x.permute(0, 2, 3, 1).contiguous().view(-1, self.embed_dim)

        # Calculate distances
        distances = (flat_inputs ** 2).sum(dim=1, keepdim=True) \
                    - 2 * torch.mm(flat_inputs, weight.t()) \
                    + (weight.t() ** 2).sum(dim=0, keepdim=True)

        # Encoding
        embed_idx = torch.max(-distances, dim=1)[1]
        embed_onehot = F.one_hot(embed_idx, self.n_embed).type(flat_inputs.dtype)

        # Quantize and unflatten
        embed_idx = embed_idx.view(b, h, w)
        quantized = self.embedding(embed_idx).permute(0, 3, 1, 2).contiguous()

        # Use EMA to update the embedding vectors
        if self.training:
            self.N.data.mul_(self.decay).add_(1 - self.decay, embed_onehot.sum(0))

            # Laplace smoothing of the cluster size
            embed_sum = torch.mm(flat_inputs.t(), embed_onehot)
            self.embed_avg.data.mul_(self.decay).add_(1 - self.decay, embed_sum.t())

            n = self.N.sum()
            weights = (self.N + self.eps) / (n + self.n_embed * self.eps) * n
            embed_normalized = self.embed_avg / weights.unsqueeze(1)
            self.embedding.weight.data.copy_(embed_normalized)

        latent_loss = self.commitment_cost * F.mse_loss(quantized.detach(), x)

        # Stop optimization from accessing the embedding
        quantized_st = (quantized - x).detach() + x

        return quantized_st, embed_idx, latent_loss


@torch.jit.script
def swish(x):
    return x * torch.sigmoid(x)


def Normalize(in_channels):
    return nn.GroupNorm(
        num_groups=32,
        num_channels=in_channels,
        eps=1e-6,
        affine=True
    )


class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=2,
            padding=0
        )

    def forward(self, x):
        pad = (0, 1, 0, 1)
        x = nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.norm2 = Normalize(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0
            )

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = F.silu(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.k = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.v = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.proj_out = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = F.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_


class Encoder(nn.Module):
    def __init__(
            self,
            in_channels: int,
            n_channels: int,
            z_channels: int,
            ch_mult: Tuple[int],
            num_res_blocks: int,
            resolution: Tuple[int],
            attn_resolutions: Tuple[int],
            **ignorekwargs,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.n_channels = n_channels
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.attn_resolutions = attn_resolutions

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)

        blocks = []
        # initial convolution
        blocks.append(
            nn.Conv2d(
                in_channels,
                n_channels,
                kernel_size=3,
                stride=1,
                padding=1
            )
        )

        # residual and downsampling blocks, with attention on smaller res (16x16)
        for i in range(self.num_resolutions):
            block_in_ch = n_channels * in_ch_mult[i]
            block_out_ch = n_channels * ch_mult[i]
            for _ in range(self.num_res_blocks):
                blocks.append(ResBlock(block_in_ch, block_out_ch))
                block_in_ch = block_out_ch
                if max(curr_res) in attn_resolutions:
                    blocks.append(AttnBlock(block_in_ch))

            if i != self.num_resolutions - 1:
                blocks.append(Downsample(block_in_ch))
                curr_res = tuple(ti // 2 for ti in curr_res)

        # non-local attention block
        blocks.append(ResBlock(block_in_ch, block_in_ch))
        blocks.append(AttnBlock(block_in_ch))
        blocks.append(ResBlock(block_in_ch, block_in_ch))

        # normalise and convert to latent size
        blocks.append(Normalize(block_in_ch))
        blocks.append(
            nn.Conv2d(
                block_in_ch,
                z_channels,
                kernel_size=3,
                stride=1,
                padding=1
            )
        )

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class Decoder(nn.Module):
    def __init__(
            self,
            n_channels: int,
            z_channels: int,
            out_channels: int,
            ch_mult: Tuple[int],
            num_res_blocks: int,
            resolution: Tuple[int],
            attn_resolutions: Tuple[int],
            **ignorekwargs,
    ) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.z_channels = z_channels
        self.out_channels = out_channels
        self.ch_mult = ch_mult
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.attn_resolutions = attn_resolutions

        block_in_ch = n_channels * self.ch_mult[-1]
        curr_res = tuple(ti // 2 ** (self.num_resolutions - 1) for ti in resolution)

        blocks = []
        # initial conv
        blocks.append(
            nn.Conv2d(
                z_channels,
                block_in_ch,
                kernel_size=3,
                stride=1,
                padding=1
            )
        )

        # non-local attention block
        blocks.append(ResBlock(block_in_ch, block_in_ch))
        blocks.append(AttnBlock(block_in_ch))
        blocks.append(ResBlock(block_in_ch, block_in_ch))

        for i in reversed(range(self.num_resolutions)):
            block_out_ch = n_channels * self.ch_mult[i]

            for _ in range(self.num_res_blocks):
                blocks.append(ResBlock(block_in_ch, block_out_ch))
                block_in_ch = block_out_ch

                if max(curr_res) in self.attn_resolutions:
                    blocks.append(AttnBlock(block_in_ch))

            if i != 0:
                blocks.append(Upsample(block_in_ch))
                curr_res = tuple(ti * 2 for ti in curr_res)

        blocks.append(Normalize(block_in_ch))
        blocks.append(
            nn.Conv2d(
                block_in_ch,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1
            )
        )

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class VQVAE(nn.Module):
    def __init__(
            self,
            n_embed: int,
            embed_dim: int,
            commitment_cost: float,
            vq_decay: float,
            hparams: Dict
    ) -> None:
        super().__init__()
        self.n_embed = n_embed
        self.encoder = Encoder(**hparams)
        self.decoder = Decoder(**hparams)
        self.codebook = Quantizer(embed_dim=embed_dim, n_embed=n_embed, commitment_cost=commitment_cost, decay=vq_decay)
        self.quant_conv = torch.nn.Conv2d(hparams["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, hparams["z_channels"], 1)

    def encode_code(self, x):
        z = self.encoder(x)
        z = self.quant_conv(z)
        indices = self.codebook(z)[1]
        return indices

    def decode_code(self, latents):
        latents = self.codebook.embedding(latents).permute(0, 3, 1, 2).contiguous()
        latents = self.post_quant_conv(latents)
        return self.decoder(latents)

    def forward(self, x):
        with autocast(enabled=True):
            z = self.encoder(x)
            z = self.quant_conv(z)
            with autocast(enabled=False):
                e, embed_idx, latent_loss = self.codebook(z.float())
            h = self.post_quant_conv(e.half())
            x_tilde = self.decoder(h)

        avg_probs = lambda e: torch.histc(e.float(), bins=self.n_embed, max=self.n_embed).float().div(e.numel())
        perplexity = lambda avg_probs: torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        perplexity_code = perplexity(avg_probs(embed_idx))

        return x_tilde, latent_loss, perplexity_code, embed_idx

    def reconstruct(self, x):
        z = self.encode_code(x)
        x_recon = self.decode_code(z)
        return x_recon

    def get_ldm_inputs(self, img):
        z = self.encoder(img)
        z = self.quant_conv(z)
        e, _, _ = self.codebook(z.float())
        return e

    def reconstruct_ldm_outputs(self, e):
        e, _, _ = self.codebook(e)
        e = self.post_quant_conv(e)
        x_hat = self.decoder(e)
        return x_hat