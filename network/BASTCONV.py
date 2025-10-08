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


class DecoderLayer(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, queries, encoder_output):
        # Self-attention
        q_self, _ = self.self_attn(queries, queries, queries)
        queries = self.norm1(queries + q_self)

        # Cross-attention
        q_cross, _ = self.cross_attn(queries, encoder_output, encoder_output)
        queries = self.norm2(queries + q_cross)

        # FFN
        q_ffn = self.ffn(queries)
        queries = self.norm3(queries + q_ffn)

        return queries


class Tokenizer(nn.Module):
    def __init__(
        self,
        kernel_size=3,
        stride=1,
        padding=1,
        pooling_kernel_size=3,
        pooling_stride=2,
        pooling_padding=1,
        n_conv_layers=1,
        n_input_channels=3,
        n_output_channels=64,
        in_planes=64,
        activation=nn.ReLU,
        max_pool=True,
        conv_bias=False,
    ):
        super().__init__()

        n_filter_list = (
            [n_input_channels]
            + [in_planes for _ in range(n_conv_layers - 1)]
            + [n_output_channels]
        )

        n_filter_list_pairs = zip(n_filter_list[:-1], n_filter_list[1:])

        self.conv_layers = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv2d(
                        chan_in,
                        chan_out,
                        kernel_size=(kernel_size, kernel_size),
                        stride=(stride, stride),
                        padding=(padding, padding),
                        bias=conv_bias,
                    ),
                    nn.MaxPool2d(
                        kernel_size=pooling_kernel_size,
                        stride=pooling_stride,
                        padding=pooling_padding,
                    )
                    if max_pool
                    else nn.Identity(),
                )
                for chan_in, chan_out in n_filter_list_pairs
            ]
        )

    def sequence_length(self, n_channels=3, height=224, width=224):
        return self.forward(torch.zeros((1, n_channels, height, width))).shape[1]

    def forward(self, x):
        return rearrange(self.conv_layers(x), "b c h w -> b (h w) c")


# --- Variant-switchable BAST model ---
class BAST_CONV(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        num_coordinates_output,  # e.g., 2 (azimuth, elevation) or 3 (x,y,z)
        dim,  # embedding dimension, e.g., 512
        heads,  # number of attention heads, e.g., 8
        num_encoder_layers=6,
        num_decoder_layers=3,
        mlp_ratio=4,
        dropout=0.2,
        emb_dropout=0.0,
        binaural_integration="CROSS_ATTN",
        max_sources=4,
        num_classes_cls=1,
        n_conv_input_channels=1,
        kernel_size=3,
        stride=1,
        padding=1,
        pooling_kernel_size=3,
        pooling_stride=2,
        pooling_padding=1,
        n_conv_layers=1,
        in_planes=64,
        conv_bias=False,
    ):
        super().__init__()
        self.binaural_integration = binaural_integration
        self.max_sources = max_sources
        self.num_coordinates_output = num_coordinates_output
        self.num_classes_cls = num_classes_cls
        self.num_decoder_layers = num_decoder_layers
        self.num_encoder_layers = num_encoder_layers

        # --- Compute patch grid dimensions and padding ---
        dim_head = dim // heads
        mlp_dim = int(dim * mlp_ratio)
        image_height, image_width = image_size

        self.tokenizer = Tokenizer(
            n_input_channels=n_conv_input_channels,
            n_output_channels=dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            pooling_kernel_size=pooling_kernel_size,
            pooling_stride=pooling_stride,
            pooling_padding=pooling_padding,
            in_planes=in_planes,
            max_pool=True,
            activation=nn.ReLU,
            n_conv_layers=n_conv_layers,
            conv_bias=conv_bias,
        )

        length = self.tokenizer.sequence_length(
            n_channels=n_conv_input_channels, height=image_height, width=image_width
        )

        # self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, dim))
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, length, dim), requires_grad=True
        )
        nn.init.trunc_normal_(self.pos_embedding, std=0.2)

        self.dropout = nn.Dropout(emb_dropout)

        # self.early_transformer = Transformer(
        #     dim,
        #     depth=num_encoder_layers - 2,
        #     heads=heads,
        #     dim_head=dim_head,
        #     mlp_dim=mlp_dim,
        #     dropout=dropout,
        # )

        self.cross_attn = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.norm_cross = nn.LayerNorm(dim)

        # since we calculate difference and sum then cat them together, the dim doubles
        self.deep_transformer = Transformer(
            dim * 2,
            depth=num_encoder_layers,
            heads=heads,
            dim_head=(dim * 2) // heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
        )
        self.encoder_projection = nn.Linear(dim * 2, dim)

        self.decoder_layers = nn.ModuleList(
            [
                DecoderLayer(dim, heads, mlp_dim, dropout)
                for _ in range(num_decoder_layers)
            ]
        )

        self.object_queries = nn.Parameter(torch.randn(max_sources, dim))
        self.query_pos_embed = nn.Parameter(torch.randn(max_sources, dim))

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

        img_l = img[:, 0:1, :, :]
        img_r = img[:, 1:2, :, :]

        x_l = self.tokenizer(img_l)
        x_r = self.tokenizer(img_r)

        x_l = x_l + self.pos_embedding
        x_r = x_r + self.pos_embedding

        x_l = self.dropout(x_l)
        x_r = self.dropout(x_r)

        # x_l = self.early_transformer(x_l)
        # x_r = self.early_transformer(x_r)

        attended_l, _ = self.cross_attn(query=x_l, key=x_r, value=x_r)
        attended_r, _ = self.cross_attn(query=x_r, key=x_l, value=x_l)

        x_l = self.norm_cross(x_l + attended_l)
        x_r = self.norm_cross(x_r + attended_r)

        x_diff = x_r - x_l  # [B, N, dim] - spatial cues
        x_sum = (x_l + x_r) * 0.5  # [B, N, dim] - spectral cues
        x = torch.cat([x_diff, x_sum], dim=-1)

        # Deep processing after integration
        x = self.deep_transformer(x)  # [B, N, dim]

        x = self.encoder_projection(x)

        queries = repeat(self.object_queries, "n d -> b n d", b=B)
        queries = queries + self.query_pos_embed.unsqueeze(0)

        for decoder_layer in self.decoder_layers:
            queries = decoder_layer(queries, x)

        # Predict from each query slot
        predictions = self.det_head(queries)  # [B, max_sources, loc+obj+cls]

        # Split outputs
        loc_out = predictions[..., : self.num_coordinates_output]
        obj_logit = predictions[..., self.num_coordinates_output]
        cls_logit = predictions[..., self.num_coordinates_output + 1 :]

        return loc_out, obj_logit, cls_logit
