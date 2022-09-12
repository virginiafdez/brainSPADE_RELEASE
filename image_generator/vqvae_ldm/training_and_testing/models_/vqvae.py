from typing import Dict

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


class ResidualLayer(nn.Sequential):
    def __init__(self, n_channels, p_dropout):
        super().__init__(nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1),
                         nn.ReLU(True),
                         nn.Dropout2d(p_dropout),
                         nn.Conv2d(n_channels, n_channels, kernel_size=1))

    def forward(self, x):
        return F.relu(x + super().forward(x), True)


class Encoder(nn.Module):
    def __init__(
            self,
            in_channels: int,
            n_channels: int,
            z_channels: int,
            num_res_blocks: int,
            num_downsamplings: int,
            p_dropout: float,
            **ignorekwargs,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.n_channels = n_channels
        self.z_channels = z_channels
        self.num_res_blocks = num_res_blocks
        self.num_downsamplings = num_downsamplings

        blocks = []
        # initial convolutions
        blocks.append(
            nn.Sequential(
                *[
                    nn.Conv2d(
                        in_channels,
                        n_channels,
                        kernel_size=4,
                        stride=2,
                        padding=1
                    ),
                    nn.ReLU(),
                ]
            )
        )

        for idx in range(self.num_downsamplings - 1):
            blocks.append(
                nn.Sequential(
                    *[
                        nn.Conv2d(
                            n_channels,
                            n_channels,
                            kernel_size=4,
                            stride=2,
                            padding=1
                        ),
                        nn.ReLU(),
                    ]
                )
            )

        blocks.append(
            nn.Sequential(
                *[
                    ResidualLayer(
                        n_channels=n_channels,
                        p_dropout=p_dropout
                    ) for _ in range(num_res_blocks)
                ]
            )
        )

        blocks.append(
            nn.Conv2d(
                n_channels,
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
            num_res_blocks: int,
            num_downsamplings: int,
            p_dropout: float,
            **ignorekwargs,
    ) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.z_channels = z_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.num_downsamplings = num_downsamplings

        blocks = []
        # initial conv
        blocks.append(
            nn.Conv2d(
                z_channels,
                n_channels,
                kernel_size=3,
                stride=1,
                padding=1
            )
        )

        blocks.append(
            nn.Sequential(
                *[
                    ResidualLayer(
                        n_channels=n_channels,
                        p_dropout=p_dropout
                    ) for _ in range(num_res_blocks)
                ]
            )
        )

        for _ in range(self.num_downsamplings - 1):
            blocks.append(
                nn.Sequential(
                    *[
                        nn.ConvTranspose2d(
                            n_channels,
                            n_channels,
                            kernel_size=4,
                            stride=2,
                            padding=1
                        ),
                        nn.ReLU(),
                    ]
                )
            )

        blocks.append(
            nn.ConvTranspose2d(
                n_channels,
                out_channels,
                kernel_size=4,
                stride=2,
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
        self.embed_dim = embed_dim
        self.encoder = Encoder(**hparams)
        self.decoder = Decoder(**hparams)
        self.codebook = Quantizer(n_embed, embed_dim, commitment_cost=commitment_cost, decay=vq_decay)

    def encode_code(self, x):
        z = self.encoder(x)
        indices = self.codebook(z)[1]
        return indices

    def decode_code(self, latents):
        latents = self.codebook.embedding(latents).permute(0, 3, 1, 2).contiguous()
        return self.decoder(latents)

    def forward(self, x):
        with autocast(enabled=True):
            z = self.encoder(x)
            with autocast(enabled=False):
                e, embed_idx, latent_loss = self.codebook(z.float())
            x_tilde = self.decoder(e.half())

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
        e, _, _ = self.codebook(z.float())
        return e

    def reconstruct_ldm_outputs(self, e):
        e, _, _ = self.codebook(e)
        x_hat = self.decoder(e)
        return x_hat