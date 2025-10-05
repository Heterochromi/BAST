import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn
import numpy as np


"""
Reference: https://github.com/lucidrains/vit-pytorch
"""


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        """

        :param dim:
        :param depth:
        :param heads:
        :param dim_head:
        :param mlp_dim:
        :param dropout:
        """
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            Attention(
                                dim, heads=heads, dim_head=dim_head, dropout=dropout
                            ),
                        ),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


# --- Variant-switchable BAST model ---
class BAST_Variant(nn.Module):
    def __init__(
        self,
        *,
        image_size,  # e.g., (129, 61) - (freq, time)
        patch_size,  # e.g., 16
        patch_overlap,  # e.g., 10
        num_coordinates_output,  # e.g., 2 (azimuth, elevation)
        dim,  # embedding dimension, e.g., 512
        depth,  # transformer depth, e.g., 6
        heads,  # number of attention heads, e.g., 8
        mlp_dim,  # MLP dimension, e.g., 1024
        channels=2,  # input channels (stereo: left/right)
        dim_head=64,
        dropout=0.2,
        emb_dropout=0.0,
        binaural_integration="CROSS_ATTN",  # Changed default
        max_sources=4,
        num_classes_cls=1,
    ):
        super().__init__()
        self.binaural_integration = binaural_integration
        self.max_sources = max_sources
        self.num_coordinates_output = num_coordinates_output
        self.num_classes_cls = num_classes_cls

        # --- Compute patch grid dimensions and padding ---
        image_height, image_width = image_size
        patch_height = patch_width = patch_size
        if patch_overlap != 0:
            num_patches_height = (
                int(
                    np.ceil(
                        (image_height - patch_height) / (patch_height - patch_overlap)
                    )
                )
                + 1
            )
            num_patches_width = (
                int(
                    np.ceil((image_width - patch_width) / (patch_width - patch_overlap))
                )
                + 1
            )
            padding_height = (
                (num_patches_height - 1) * (patch_height - patch_overlap)
                + patch_height
                - image_height
            )
            padding_width = (
                (num_patches_width - 1) * (patch_width - patch_overlap)
                + patch_width
                - image_width
            )
        else:
            num_patches_height = int(np.ceil(image_height / patch_height))
            num_patches_width = int(np.ceil(image_width / patch_width))
            padding_height = num_patches_height * patch_height - image_height
            padding_width = num_patches_width * patch_width - image_width

        self.num_patches_height = num_patches_height
        self.num_patches_width = num_patches_width
        self.num_patches = num_patches_height * num_patches_width

        # FIX #1: Use 2 channels per patch (stereo together)
        patch_dim = 2 * patch_height * patch_width  # Both channels in each patch

        # --- Patch Embedding for STEREO input ---
        self.to_patch_embedding = nn.Sequential(
            nn.ReflectionPad2d((0, padding_width, padding_height, 0)),
            nn.Unfold(
                kernel_size=(patch_size, patch_size),
                stride=patch_height - patch_overlap,
            ),
            Rearrange(
                "b (c k1 k2) n -> b n (k1 k2 c)",
                c=2,  # stereo
                k1=patch_height,
                k2=patch_width,
                n=self.num_patches,
            ),
            nn.Linear(patch_dim, dim),
        )

        # FIX #2: Add positional embeddings (no CLS token initially)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)

        # FIX #3: Early binaural processing with fewer layers
        self.early_transformer = Transformer(
            dim,
            depth=2,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
        )

        # Optional: Cross-channel attention for better binaural integration
        if binaural_integration == "CROSS_ATTN":
            self.cross_attn = nn.MultiheadAttention(
                dim, heads, dropout=dropout, batch_first=True
            )
            self.norm_cross = nn.LayerNorm(dim)

        # FIX #4: Deep transformer after integration
        self.deep_transformer = Transformer(
            dim,
            depth=depth - 2,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
        )

        # FIX #5: Use learnable query slots instead of CLS token + pooling
        self.object_queries = nn.Parameter(torch.randn(max_sources, dim))
        self.query_pos_embed = nn.Parameter(torch.randn(max_sources, dim))

        # Cross-attention: queries attend to patch features
        self.decoder_cross_attn = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.decoder_self_attn = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.decoder_ffn = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

        # FIX #6: Detection head per query (not from single pooled feature)
        self.det_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, self.num_coordinates_output + 1 + self.num_classes_cls),
        )

    def forward(self, img):
        # Input: [B, 2, H, W] (stereo spectrogram)
        B = img.size(0)

        # FIX #1: Patch embedding with both channels
        x = self.to_patch_embedding(img)  # [B, N, dim]
        x = x + self.pos_embedding
        x = self.dropout(x)

        # FIX #3: Early light processing before integration
        x = self.early_transformer(x)  # [B, N, dim]

        # FIX #2: Early binaural integration option
        # (You can also split into L/R here if you want, but keeping together is better)
        if self.binaural_integration == "CROSS_ATTN":
            # Cross-attention between left and right channel features
            # This is a simplified version - you could split patches by channel
            x_attended, _ = self.cross_attn(x, x, x)
            x = self.norm_cross(x + x_attended)

        # Deep processing after integration
        x = self.deep_transformer(x)  # [B, N, dim]

        # FIX #5: Use learnable object queries instead of pooling
        queries = repeat(self.object_queries, "n d -> b n d", b=B)
        queries = queries + self.query_pos_embed.unsqueeze(0)

        # Decoder layer: queries attend to encoded features
        # Self-attention between queries
        q_self, _ = self.decoder_self_attn(queries, queries, queries)
        queries = self.norm1(queries + q_self)

        # Cross-attention: queries attend to patch features
        q_cross, _ = self.decoder_cross_attn(queries, x, x)
        queries = self.norm2(queries + q_cross)

        # FFN
        q_ffn = self.decoder_ffn(queries)
        queries = self.norm3(queries + q_ffn)

        # FIX #6: Predict from each query slot
        predictions = self.det_head(queries)  # [B, max_sources, loc+obj+cls]

        # Split outputs
        loc_out = predictions[..., : self.num_coordinates_output]
        obj_logit = predictions[..., self.num_coordinates_output]
        cls_logit = predictions[..., self.num_coordinates_output + 1 :]

        return loc_out, obj_logit, cls_logit


class AngularLossWithCartesianCoordinate(nn.Module):
    def __init__(self):
        super(AngularLossWithCartesianCoordinate, self).__init__()

    def forward(self, x, y):
        x = x / torch.linalg.norm(x, dim=1)[:, None]
        y = y / torch.linalg.norm(y, dim=1)[:, None]
        dot = torch.clamp(torch.sum(x * y, dim=1), min=-0.999, max=0.999)
        loss = torch.mean(torch.acos(dot))
        return loss


class MixWithCartesianCoordinate(nn.Module):
    def __init__(self):
        super(MixWithCartesianCoordinate, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, x, y):
        loss1 = self.mse(x, y)
        x = x / torch.linalg.norm(x, dim=1)[:, None]
        y = y / torch.linalg.norm(y, dim=1)[:, None]
        dot = torch.clamp(torch.sum(x * y, dim=1), min=-0.999, max=0.999)
        loss2 = torch.mean(torch.acos(dot))
        return loss1 + loss2
