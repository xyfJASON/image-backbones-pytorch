"""
Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner,
Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby
An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
https://arxiv.org/abs/2010.11929
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

__all__ = ['ViT', 'vit_tiny', 'vit_small', 'vit_base', 'vit_large', 'vit_huge']


class SelfAttention(nn.Module):
    """ Multi-head self-attention """
    def __init__(self, embed_dim: int, n_head: int, pdrop: float = 0.1):
        super().__init__()
        assert embed_dim % n_head == 0
        self.n_head = n_head
        head_dim = embed_dim // n_head
        self.scale = head_dim ** -0.5

        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(pdrop)

    def forward(self, x: Tensor):
        """(batch, n_tokens, embed_dim) -> (batch, n_tokens, embed_dim)"""
        bs, nt, ed = x.shape
        k = self.key(x).view(bs, nt, self.n_head, int(ed) // self.n_head).permute(0, 2, 1, 3)    # [bs, nh, nt, dim]
        q = self.query(x).view(bs, nt, self.n_head, int(ed) // self.n_head).permute(0, 2, 1, 3)  # [bs, nh, nt, dim]
        v = self.value(x).view(bs, nt, self.n_head, int(ed) // self.n_head).permute(0, 2, 1, 3)  # [bs, nh, nt, dim]
        attn_mat = (q @ k.transpose(2, 3)) * self.scale                                          # [bs, nh, nt, nt]
        attn_mat = F.softmax(attn_mat, dim=-1)
        output = attn_mat @ v                                                                    # [bs, nh, nt, dim]
        output = output.permute(0, 2, 1, 3).reshape(bs, nt, ed)                                  # [bs, nt, ed]
        output = self.dropout(self.proj(output))
        return output


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim: int, n_head: int, pdrop: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.selfattn = SelfAttention(embed_dim, n_head)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(pdrop),
        )

    def forward(self, x: Tensor):
        x = self.selfattn(self.ln1(x)) + x
        x = self.mlp(self.ln2(x)) + x
        return x


class ViT(nn.Module):
    def __init__(self, img_size: int = 224, patch_size: int = 16, n_layer: int = 12, embed_dim: int = 768,
                 n_head: int = 12, pdrop: float = 0.1, n_classes: int = 1000):
        super().__init__()
        self.img_size = img_size
        n_patches = img_size // patch_size
        n_tokens = n_patches * n_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size))
        self.pos_embed = nn.Parameter(torch.randn(1, n_tokens + 1, embed_dim))
        self.dropout = nn.Dropout(pdrop)

        self.encoder = nn.Sequential(*[TransformerEncoderBlock(embed_dim, n_head, pdrop) for _ in range(n_layer)])

        self.classifier = nn.Linear(embed_dim, n_classes)

    def forward(self, x: Tensor):
        """(batch, C, H, W) -> (batch, n_classes)"""
        assert x.shape[-2:] == (self.img_size, self.img_size)
        # x ==> embedded token
        embedx = self.patch_embed(x)  # [batch, embed_dim, n_patches, n_patches]
        bs, ed, w, h = embedx.shape
        embedx = embedx.permute(0, 2, 3, 1).reshape(bs, w*h, ed)  # [batch, n_tokens, embed_dim]
        # concatenate with cls token
        embedx = torch.cat([self.cls_token.repeat(bs, 1, 1), embedx], dim=1)  # [batch, n_tokens+1, embed_dim]
        # add position embedding
        embedx = self.dropout(embedx + self.pos_embed)  # [batch, n_tokens+1, embed_dim]
        # encoder
        feature = self.encoder(embedx)  # [batch, n_tokens+1, embed_dim]
        # classifier
        output = self.classifier(feature[:, 0, :])
        return output


def vit_tiny(n_classes: int, img_size: int = 224, patch_size: int = 32, pdrop: float = 0.1):
    return ViT(img_size=img_size, patch_size=patch_size, n_layer=12, embed_dim=192, n_head=3, pdrop=pdrop, n_classes=n_classes)


def vit_small(n_classes: int, img_size: int = 224, patch_size: int = 32, pdrop: float = 0.1):
    return ViT(img_size=img_size, patch_size=patch_size, n_layer=12, embed_dim=384, n_head=6, pdrop=pdrop, n_classes=n_classes)


def vit_base(n_classes: int, img_size: int = 224, patch_size: int = 32, pdrop: float = 0.1):
    return ViT(img_size=img_size, patch_size=patch_size, n_layer=12, embed_dim=768, n_head=12, pdrop=pdrop, n_classes=n_classes)


def vit_large(n_classes: int, img_size: int = 224, patch_size: int = 32, pdrop: float = 0.1):
    return ViT(img_size=img_size, patch_size=patch_size, n_layer=24, embed_dim=1024, n_head=16, pdrop=pdrop, n_classes=n_classes)


def vit_huge(n_classes: int, img_size: int = 224, patch_size: int = 32, pdrop: float = 0.1):
    return ViT(img_size=img_size, patch_size=patch_size, n_layer=32, embed_dim=1280, n_head=16, pdrop=pdrop, n_classes=n_classes)


def _test_overhead():
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from utils.overhead import calc_flops, count_params, calc_inference_time

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = vit_tiny(n_classes=10, img_size=32, patch_size=4).to(device)
    x = torch.randn(1, 3, 32, 32).to(device)

    count_params(model)
    print('=' * 60)
    calc_flops(model, x)
    print('=' * 60)
    calc_inference_time(model, x)


if __name__ == '__main__':
    _test_overhead()
