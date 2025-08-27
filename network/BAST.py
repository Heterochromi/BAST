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
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
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
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x



# --- Variant-switchable BAST model ---
class BAST_Variant(nn.Module):
    def __init__(self, *,
                 image_size,  # e.g., (129, 61)
                 patch_size,  # e.g., 16
                 patch_overlap,  # e.g., 10
                 num_coordinates_output,  # e.g., 2
                 dim,  # embedding dimension, e.g., 1024 or lower
                 depth,  # transformer depth, e.g., 3
                 heads,  # number of attention heads, e.g., 16
                 mlp_dim,  # MLP dimension, e.g., 1024
                 pool='conv',  # pooling method: 'mean', 'conv', 'linear'
                 channels=2,  # input channels (stereo: left/right)
                 dim_head=64,
                 dropout=0.2,
                 emb_dropout=0.,
                 binaural_integration='SUB',  # 'SUB', 'ADD', or 'CONCAT'
                 share_params=False,
                 transformer_variant='vanilla',
                 max_sources = 4,
                 classify_sound=False,
                 num_classes_cls=1,
                 ):
        super().__init__()
        self.pool = pool
        self.binaural_integration = binaural_integration
        self.share_params = share_params
        self.transformer_variant = transformer_variant
        self.classify_sound = classify_sound
        self.max_sources = max_sources
        self.num_coordinates_output = num_coordinates_output
        self.num_classes_cls = num_classes_cls


        # --- Compute patch grid dimensions and padding ---
        image_height, image_width = image_size
        patch_height = patch_width = patch_size
        if patch_overlap != 0:
            num_patches_height = int(np.ceil((image_height - patch_height) / (patch_height - patch_overlap))) + 1
            num_patches_width = int(np.ceil((image_width - patch_width) / (patch_width - patch_overlap))) + 1
            padding_height = (num_patches_height - 1) * (patch_height - patch_overlap) + patch_height - image_height
            padding_width = (num_patches_width - 1) * (patch_width - patch_overlap) + patch_width - image_width
        else:
            num_patches_height = int(np.ceil(image_height / patch_height))
            num_patches_width = int(np.ceil(image_width / patch_width))
            padding_height = num_patches_height * patch_height - image_height
            padding_width = num_patches_width * patch_width - image_width

        self.num_patches_height = num_patches_height
        self.num_patches_width = num_patches_width
        self.num_patches = num_patches_height * num_patches_width
        patch_dim = 1 * patch_height * patch_width  # each branch has 1 channel

        # --- Shared Patch Embedding (for both variants) ---
        self.to_patch_embedding = nn.Sequential(
            nn.ReflectionPad2d((0, padding_width, padding_height, 0)),  # (left, right, top, bottom)
            nn.Unfold(kernel_size=(patch_size, patch_size), stride=patch_height - patch_overlap),
            Rearrange('b (c k1 k2) n -> b n (k1 k2 c)', k1=patch_height, k2=patch_width, n=self.num_patches),
            nn.Linear(patch_dim, dim)
        )

        if transformer_variant == 'vanilla':
            # --- Vanilla branch: add CLS token and positional embeddings ---
            self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
            self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))
            self.dropout = nn.Dropout(emb_dropout)
            self.transformer1 = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
            if not share_params:
                self.transformer2 = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
            integration_dim = dim if binaural_integration != 'CONCAT' else dim * 2
            self.transformer3 = Transformer(integration_dim, depth, heads, dim_head, mlp_dim, dropout)
            # Detection head (per-source slots): [loc (num_coordinates_output), objectness (1), classes (num_classes_cls)]
        
        
        # --- Optional Pooling ---
        if pool == 'conv':
            self.patch_pooling = nn.Sequential(
                nn.Conv1d(self.num_patches if transformer_variant == 'vanilla' else (
                            self.num_patches_height * self.num_patches_width), 1, 1),
                nn.GELU()
            )
        elif pool == 'linear':
            self.patch_pooling = nn.Sequential(
                nn.Linear(self.num_patches if transformer_variant == 'vanilla' else (
                            self.num_patches_height * self.num_patches_width), 1),
                nn.GELU()
            )


        self.det_head = nn.Sequential(
                nn.LayerNorm(integration_dim),
                nn.Linear(integration_dim, self.max_sources * (self.num_coordinates_output + 1 + self.num_classes_cls))
            )


    def process_branch(self, img_branch, transformer_module):
        # Shared patch embedding: output shape [B, N, dim]
        x = self.to_patch_embedding(img_branch)
        if self.transformer_variant == 'vanilla':
            b, n, d = x.shape
            cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
            x = torch.cat((cls_tokens, x), dim=1)
            x = x + self.pos_embedding[:, :x.shape[1]]
            x = self.dropout(x)
        # Apply the corresponding transformer module
        x = transformer_module(x)
        return x

    def forward(self, img):
        # Assume input img shape: [B, channels, H, W] with channels = 2 (stereo)
        img_l = img[:, 0:1, :, :]
        img_r = img[:, 1:2, :, :]
        x_l = self.process_branch(img_l, self.transformer1)
        if self.share_params:
            x_r = self.process_branch(img_r, self.transformer1)
        else:
            x_r = self.process_branch(img_r, self.transformer2)
        # Binaural integration
        if self.binaural_integration == 'ADD':
            x = x_l + x_r
        elif self.binaural_integration == 'SUB':
            x = x_l - x_r
        elif self.binaural_integration == 'CONCAT':
            x = torch.cat((x_l, x_r), dim=-1)
        else:
            raise ValueError("Unsupported binaural_integration option.")

        # Integration transformer stage
        x = self.transformer3(x)

        # Pooling to shared feature
        if self.pool == 'mean':
            if self.transformer_variant == 'vanilla':
                feat = x.mean(dim=1)
            else:
                raise ValueError("Unknown transformer variant.")
        elif self.pool == 'cls':
            if self.transformer_variant == 'vanilla':
                feat = x[:, 0]
            else:
                raise ValueError("CLS pooling is only supported for vanilla variant.")
        elif self.pool == 'conv':
            if self.transformer_variant == 'vanilla':
                feat = self.patch_pooling(x[:, 1:])[:, 0, :]
            else:
                x_seq = rearrange(x, 'b h w d -> b (h w) d')
                feat = self.patch_pooling(x_seq)[:, 0, :]
        elif self.pool == 'linear':
            if self.transformer_variant == 'vanilla':
                feat = self.patch_pooling(x[:, 1:].transpose(1, 2)).squeeze(-1)
            else:
                x_seq = rearrange(x, 'b h w d -> b (h w) d')
                feat = self.patch_pooling(x_seq.transpose(1, 2)).squeeze(-1)
        else:
            raise ValueError("Unsupported pooling type.")
        # --- Detection outputs ---
        B = feat.size(0)
        feat_n = self.det_head(feat)
        det = feat_n.view(B, self.max_sources, self.num_coordinates_output + 1 + self.num_classes_cls)
        loc_out   = det[..., :self.num_coordinates_output]
        obj_logit = det[..., self.num_coordinates_output]
        cls_logit = det[..., self.num_coordinates_output + 1:]
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


class AngularLossWithPolarCoordinate(nn.Module):
    def __init__(self):
        super(AngularLossWithPolarCoordinate, self).__init__()

    def forward(self, x, y):
        x1 = x[:, 1]
        y_r = torch.atan2(y[:, 1], y[:, 0])
        diff = torch.abs(x1 - y_r)
        loss = torch.mean(torch.pow(diff, 2))
        return loss


class MSELossWithPolarCoordinate(nn.Module):
    def __init__(self, w_x: float = 1.0, w_y: float = 1.0, reduction: str = 'mean'):
        super(MSELossWithPolarCoordinate, self).__init__()
        self.w_x = w_x
        self.w_y = w_y
        self.mse = nn.MSELoss(reduction=reduction)

    def forward(self, x, y):
        # x: [B, 2] where x[:,0] = radius (r), x[:,1] = angle (theta in radians)
        # y: [B, 2] target Cartesian coordinates [x, y]
        pred_x = (x[:, 0] * torch.cos(x[:, 1])).unsqueeze(1)
        pred_y = (x[:, 0] * torch.sin(x[:, 1])).unsqueeze(1)
        target_x = y[:, 0:1]
        target_y = y[:, 1:2]
        loss_x = self.mse(pred_x, target_x)
        loss_y = self.mse(pred_y, target_y)
        return self.w_x * loss_x + self.w_y * loss_y


class AzElLossDegrees(nn.Module):
    def __init__(self, az_weight: float = 1.0, el_weight: float = 1.0, reduction: str = 'mean'):
        super(AzElLossDegrees, self).__init__()
        self.az_weight = az_weight
        self.el_weight = el_weight
        self.reduction = reduction
    def _convert_to_radians(self, x):
        return x * (torch.pi/180)
    def _convert_to_degrees(self, x):
        return x * (180/torch.pi)

    def forward(self, pred, target):
        # pred/target shape: [..., 2] = [azimuth_deg, elevation_deg]
        pred_az_deg = pred[..., 0]
        pred_el_deg = pred[..., 1]
        tgt_az_deg = target[..., 0]
        tgt_el_deg = target[..., 1]

        # Circular azimuth error (radians), robust to wrap-around
        delta_az_rad = self._convert_to_radians(pred_az_deg - tgt_az_deg)
        az_err = torch.atan2(torch.sin(delta_az_rad), torch.cos(delta_az_rad))
        # az_err = self._convert_to_degrees(az_err)
        loss_az = az_err**2

        # Elevation MSE in degrees
       
        delta_el_deg = self._convert_to_radians(pred_el_deg - tgt_el_deg)
        loss_el = delta_el_deg**2
        return self.az_weight * loss_az.mean() + self.el_weight * loss_el.mean()

# class AzElLossDegrees(nn.Module):
#     def __init__(self, az_weight: float = 1.0, el_weight: float = 1.0, reduction: str = 'mean'):
#         super(AzElLossDegrees, self).__init__()
#         self.az_weight = az_weight
#         self.el_weight = el_weight
#         self.reduction = reduction
#     def _convert_to_radians(self, x):
#         return x * (torch.pi/180)
#     def _convert_to_degrees(self, x):
#         return x * (180/torch.pi)

#     def forward(self, pred, target):
#         # pred/target shape: [..., 2] = [azimuth_deg, elevation_deg]
#         pred_az_deg = pred[..., 0]
#         pred_el_deg = pred[..., 1]
#         tgt_az_deg = target[..., 0]
#         tgt_el_deg = target[..., 1]

#         # Circular azimuth error (radians), robust to wrap-around
#         delta_az_rad = self._convert_to_radians(pred_az_deg - tgt_az_deg)
#         az_err = torch.atan2(torch.sin(delta_az_rad), torch.cos(delta_az_rad))
#         az_err = self._convert_to_degrees(az_err)
#         loss_az = az_err**2

#         # Elevation MSE in degrees
       
#         delta_el_deg = pred_el_deg - tgt_el_deg
#         loss_el = delta_el_deg**2
#         return self.az_weight * loss_az.mean() + self.el_weight * loss_el.mean()

