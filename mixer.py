import math

import torch
from torch import nn


class PatchEmbed2d(nn.Module):
    def __init__(self,
                 patch_size: int or tuple,
                 emb_dim: int,
                 in_channels: int = 3
                 ):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, emb_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self,
                input: torch.Tensor
                ) -> torch.Tensor:
        # input: BxCxHxW -> BxNxC'
        return self.proj(input).flatten(2).transpose(1, 2)


class MLPBlock(nn.Module):
    def __init__(self,
                 num_tokens: int,
                 num_channels: int,
                 token_mlp_dim: int,
                 channel_mlp_dim: int,
                 droppath_rate: float
                 ):
        super().__init__()
        self.droppath_rate = droppath_rate
        # flax's nn.LayerNorm applies only to the last dimension only, i.e., channel dimension
        self.token_mixer = nn.Sequential(nn.LayerNorm(num_channels),
                                         nn.Conv1d(num_tokens, token_mlp_dim, 1),
                                         nn.GELU(),
                                         nn.Conv1d(token_mlp_dim, num_tokens, 1))
        self.channel_mixer = nn.Sequential(nn.LayerNorm(num_channels),
                                           nn.Linear(num_channels, channel_mlp_dim),
                                           nn.GELU(),
                                           nn.Linear(channel_mlp_dim, num_channels))

    def forward(self,
                x: torch.Tensor
                ) -> torch.Tensor:
        # x: BxNxC
        x = x + self.drop_path(self.token_mixer(x))
        return x + self.drop_path(self.channel_mixer(x))

    def drop_path(self,
                  input: torch.Tensor,
                  ) -> torch.Tensor:
        if not self.training or self.droppath_rate == 0:
            return input

        keep_prob = 1 - self.droppath_rate
        # 1 with prob. of keep_prob
        drop = input.new_empty(input.size(0), 1, 1).bernoulli_(keep_prob)
        return input.div(keep_prob).mul(drop)


class MLPMixer(nn.Module):
    def __init__(self,
                 num_classes: int,
                 emb_dim: int,
                 token_mlp_dim: int,
                 channel_mlp_dim: int,
                 patch_size: int,
                 num_layers: int,
                 droppath_rate: float = 0,
                 image_size: int = 224,
                 ):
        super().__init__()
        image_size = (image_size, image_size) if isinstance(image_size, int) else image_size
        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        num_tokens = math.prod(image_size) // math.prod(patch_size)

        self.patch_emb = PatchEmbed2d(patch_size, emb_dim)
        self.blocks = nn.Sequential(*[MLPBlock(num_tokens, emb_dim, token_mlp_dim, channel_mlp_dim, dr)
                                      for dr in [x.item() for x in torch.linspace(0, droppath_rate, num_layers)]])
        self.ln = nn.LayerNorm(emb_dim)
        self.fc = nn.Linear(emb_dim, num_classes)
        self._init_weights()

    def _init_weights(self):
        nn.init.zeros_(self.fc.weight)

    def forward(self,
                input: torch.Tensor
                ) -> torch.Tensor:
        x = self.patch_emb(input)  # BxNxC
        x = self.blocks(x)
        return self.fc(self.ln(x).mean(dim=1))


from homura import Registry

MLPMixers = Registry("MLPMixer")


@MLPMixers.register
def mixer_s32(num_classes,
              **kwargs
              ) -> MLPMixer:
    return MLPMixer(num_classes, 512, 256, 2048, 32, 8, **kwargs)


@MLPMixers.register
def mixer_s16(num_classes,
              **kwargs
              ) -> MLPMixer:
    return MLPMixer(num_classes, 512, 256, 2048, 16, 8, **kwargs)


@MLPMixers.register
def mixer_b32(num_classes,
              **kwargs
              ) -> MLPMixer:
    return MLPMixer(num_classes, 768, 384, 3072, 32, 12, **kwargs)


@MLPMixers.register
def mixer_b16(num_classes,
              **kwargs
              ) -> MLPMixer:
    return MLPMixer(num_classes, 768, 384, 3072, 16, 12, **kwargs)
