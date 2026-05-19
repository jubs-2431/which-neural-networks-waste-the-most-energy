"""
Edge AI Energy GAN - Model Definitions
Generator and Discriminator for synthesizing neural network
inference energy measurements across edge hardware platforms.

Based on: "Which Neural Networks Waste the Most Energy?" (Shah, TAMS/UNT)
Architecture: Conditional Tabular GAN (CTGAN-style)
"""

import torch
import torch.nn as nn


# ─── Device & Model Encodings ────────────────────────────────────────────────

DEVICES = {
    "apple_silicon": 0,
    "raspberry_pi4": 1,
    "jetson_nano":   2,
    "coral_tpu":     3,
    "stm32":         4,
    "snapdragon888": 5,
}

MODELS = {
    "mobilenetv3_small": 0,
    "mobilenetv2":       1,
    "resnet18":          2,
    "tiny_vit_5m":       3,
    "efficientnet_b0":   4,
}

NUM_DEVICES  = len(DEVICES)   # 6
NUM_MODELS   = len(MODELS)    # 5
LATENT_DIM   = 64             # noise vector dimension
COND_DIM     = NUM_DEVICES + NUM_MODELS  # 11  (one-hot concatenated)
DEFAULT_DATA_DIM = 4          # [energy_J, power_W, latency_ms, energy_std]


# ─── Residual Block ───────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """Residual block with layer norm — stabilises tabular GAN training."""

    def __init__(self, dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(x + self.block(x))


# ─── Generator ────────────────────────────────────────────────────────────────

class Generator(nn.Module):
    """
    Conditional Generator.

    Input:  noise z (LATENT_DIM) + one-hot condition c (COND_DIM)
    Output: synthetic measurement vector (DATA_DIM)
             [energy_J, power_W, latency_ms, energy_std]

    All outputs are in normalised [0,1] space; the DataScaler
    in data_utils.py inverts them back to physical units.
    """

    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        cond_dim:   int = COND_DIM,
        hidden_dim: int = 256,
        data_dim:   int = DEFAULT_DATA_DIM,
        n_residual: int = 3,
    ):
        super().__init__()
        in_dim = latent_dim + cond_dim

        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim) for _ in range(n_residual)]
        )

        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim // 2, data_dim),
            nn.Sigmoid(),   # keep outputs in [0, 1]
        )

    def forward(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z, cond], dim=-1)
        x = self.input_proj(x)
        x = self.residual_blocks(x)
        return self.output_head(x)


# ─── Discriminator ────────────────────────────────────────────────────────────

class Discriminator(nn.Module):
    """
    Conditional Discriminator (Critic).

    Input:  measurement vector (DATA_DIM) + one-hot condition (COND_DIM)
    Output: scalar score (real vs. fake), no sigmoid — uses WGAN-GP loss.

    Spectral normalisation on every linear layer prevents mode collapse
    on small tabular datasets.
    """

    def __init__(
        self,
        data_dim:   int = DEFAULT_DATA_DIM,
        cond_dim:   int = COND_DIM,
        hidden_dim: int = 256,
        n_layers:   int = 4,
        dropout:    float = 0.2,
    ):
        super().__init__()
        in_dim = data_dim + cond_dim

        layers = [
            nn.utils.spectral_norm(nn.Linear(in_dim, hidden_dim)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
        ]
        for _ in range(n_layers - 1):
            layers += [
                nn.utils.spectral_norm(nn.Linear(hidden_dim, hidden_dim)),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(dropout),
            ]
        layers.append(nn.utils.spectral_norm(nn.Linear(hidden_dim, 1)))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([x, cond], dim=-1)
        return self.net(inp)


# ─── Gradient Penalty (WGAN-GP) ───────────────────────────────────────────────

def gradient_penalty(
    discriminator: Discriminator,
    real: torch.Tensor,
    fake: torch.Tensor,
    cond: torch.Tensor,
    device: torch.device,
    lambda_gp: float = 10.0,
) -> torch.Tensor:
    """
    Computes the WGAN-GP gradient penalty term.
    Interpolates between real and fake samples and penalises
    gradients that deviate from unit norm.
    """
    batch_size = real.size(0)
    alpha = torch.rand(batch_size, 1, device=device)
    interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)

    d_interp = discriminator(interpolated, cond)
    gradients = torch.autograd.grad(
        outputs=d_interp,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True,
        retain_graph=True,
    )[0]

    grad_norm = gradients.view(batch_size, -1).norm(2, dim=1)
    penalty    = lambda_gp * ((grad_norm - 1) ** 2).mean()
    return penalty


# ─── Factory Helper ───────────────────────────────────────────────────────────

def build_gan(device: torch.device):
    """Instantiate G and D and move to device."""
    G = Generator().to(device)
    D = Discriminator().to(device)
    return G, D


def build_gan_with_dims(device: torch.device, data_dim: int):
    """Instantiate G and D for a specific feature dimensionality."""
    G = Generator(data_dim=data_dim).to(device)
    D = Discriminator(data_dim=data_dim).to(device)
    return G, D
