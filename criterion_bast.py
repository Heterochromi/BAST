import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


class AngularLossWithCartesianCoordinate(nn.Module):
    """
    Angular loss between two sets of vectors (2D az/el in degrees or 3D Cartesian).
    For 2D (az, el), inputs are expected in degrees and will be converted to unit vectors.
    For 3D, inputs are treated as Cartesian and will be normalized to unit vectors.
    """

    def __init__(self):
        super(AngularLossWithCartesianCoordinate, self).__init__()

    def forward(self, x, y):
        # x, y: [N, D] where D in {2, 3}
        def to_unit_vec(t):
            D = t.size(-1)
            if D == 3:
                v = t / torch.linalg.norm(t, dim=1, keepdim=True).clamp_min(1e-8)
                return v
            elif D == 2:
                az = t[:, 0] * (torch.pi / 180)  # deg -> rad
                el = t[:, 1] * (torch.pi / 180)
                cx = torch.cos(el) * torch.cos(az)
                cy = torch.cos(el) * torch.sin(az)
                cz = torch.sin(el)
                v = torch.stack([cx, cy, cz], dim=-1)
                v = v / torch.linalg.norm(v, dim=1, keepdim=True).clamp_min(1e-8)
                return v
            else:
                raise ValueError(f"Unsupported loc dimension: {D}; expected 2 or 3.")

        xv = to_unit_vec(x)
        yv = to_unit_vec(y)
        dot = torch.clamp(torch.sum(xv * yv, dim=1), min=-0.999999, max=0.999999)
        loss = torch.mean(torch.acos(dot))
        return loss


class MixWithCartesianCoordinate(nn.Module):
    """
    Combined MSE + Angular loss. Inputs can be 2D (az, el in deg) or 3D Cartesian.
    """

    def __init__(self):
        super(MixWithCartesianCoordinate, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, x, y):
        loss1 = self.mse(x, y)

        def to_unit_vec(t):
            D = t.size(-1)
            if D == 3:
                v = t / torch.linalg.norm(t, dim=1, keepdim=True).clamp_min(1e-8)
                return v
            elif D == 2:
                az = t[:, 0] * (torch.pi / 180)  # deg -> rad
                el = t[:, 1] * (torch.pi / 180)
                cx = torch.cos(el) * torch.cos(az)
                cy = torch.cos(el) * torch.sin(az)
                cz = torch.sin(el)
                v = torch.stack([cx, cy, cz], dim=-1)
                v = v / torch.linalg.norm(v, dim=1, keepdim=True).clamp_min(1e-8)
                return v
            else:
                raise ValueError(f"Unsupported loc dimension: {D}; expected 2 or 3.")

        xv = to_unit_vec(x)
        yv = to_unit_vec(y)
        dot = torch.clamp(torch.sum(xv * yv, dim=1), min=-0.999999, max=0.999999)
        loss2 = torch.mean(torch.acos(dot))
        return loss1 + loss2


class UncertaintyWeighter(nn.Module):
    """
    Learn per-task weights using homoscedastic uncertainty (Kendall & Gal 2018).
    Wraps loc/cls losses: total = sum(0.5 * (exp(-s_i) * L_i + s_i))
    """

    def __init__(self, init_log_vars: dict[str, float] | None = None):
        super().__init__()
        if init_log_vars is None:
            init_log_vars = {"loc": 0.0, "cls": 0.0}
        self.log_vars = nn.ParameterDict()
        for name, val in init_log_vars.items():
            # Explicitly create parameter without specifying device - will inherit from module
            self.log_vars[name] = nn.Parameter(torch.tensor(val, dtype=torch.float32))

    def forward(self, loss_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        # Expects keys "loc" and "cls" in loss_dict
        total = torch.tensor(0.0)
        for name in ("loc", "cls"):
            L = loss_dict[name]
            s = self.log_vars[name]
            # Ensure s is on same device as L
            if s.device != L.device:
                s = s.to(L.device)
            term = 0.5 * (torch.exp(-s) * L + s)
            total = total.to(term.device) + term
        return total


def focal_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float | None = 0.25,
    gamma: float = 2.0,
    reduction: str = "mean",
    pos_weight: torch.Tensor | float | int | None = None,
) -> torch.Tensor:
    """
    Focal Binary Cross Entropy with logits.
    - logits: [..., C]
    - targets: same shape as logits
    - alpha: balance factor (can be None)
    - gamma: focusing parameter
    - pos_weight: same semantics as in BCEWithLogitsLoss
    """
    p = torch.sigmoid(logits)

    # Handle pos_weight types
    pw = None
    if pos_weight is not None:
        if isinstance(pos_weight, (float, int)):
            pw = torch.tensor(pos_weight, device=logits.device, dtype=logits.dtype)
        else:
            pw = pos_weight

    ce = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none", pos_weight=pw
    )
    p_t = p * targets + (1 - p) * (1 - targets)
    focal = ce * ((1 - p_t) ** gamma)
    if alpha is not None:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        focal = alpha_t * focal

    if reduction == "mean":
        return focal.mean()
    if reduction == "sum":
        return focal.sum()
    return focal


class SetCriterionBAST(nn.Module):
    """
    Hungarian matching-based criterion without objectness.

    Combines:
      - Hungarian matching using (localization + classification) costs
      - Localization loss (configurable)
      - Multi-label classification focal BCE on matched pairs only

    Expected model outputs (without objectness):
      outputs: (loc_out, cls_logit)
        - loc_out: [B, K, D] (D=2 for az/el in deg, or D=3 for Cartesian)
        - cls_logit: [B, K, C]
    Targets format per sample:
      {'loc': [N, D], 'cls': [N, C]}
    """

    def __init__(
        self,
        loc_criterion: nn.Module,
        num_classes: int,
        loc_weight: float = 1.0,
        cls_weight: float = 1.0,
        cls_cost_weight: float = 0.25,
        loc_cost_weight: float = 1.0,
        max_sources: int = 4,
        cls_focal_alpha: float | None = 0.25,
        cls_focal_gamma: float = 2.0,
        cls_pos_weight=None,
        task_weighter: nn.Module | None = None,
    ):
        super().__init__()
        self.loc_criterion = loc_criterion
        self.num_classes = num_classes
        self.loc_weight = loc_weight
        self.cls_weight = cls_weight

        # Matching cost weights
        self.cls_cost_weight = cls_cost_weight
        self.loc_cost_weight = loc_cost_weight
        self.max_sources = max_sources

        # Focal loss params for classification
        self.cls_focal_alpha = cls_focal_alpha
        self.cls_focal_gamma = cls_focal_gamma
        self.cls_pos_weight = cls_pos_weight
        self.task_weighter = task_weighter

    @staticmethod
    def _pairwise_loc_cost(
        pred_loc: torch.Tensor, gt_loc: torch.Tensor
    ) -> torch.Tensor:
        """
        pred_loc: [K, D]
        gt_loc:   [N, D]
        Returns a cost matrix [K, N] with angular distance (0..1) if D=2/3.
        """
        if gt_loc.numel() == 0:
            return torch.zeros(pred_loc.size(0), 0, device=pred_loc.device)

        pred = pred_loc[:, None, :]  # [K,1,D]
        tgt = gt_loc[None, :, :]  # [1,N,D]
        D = pred.size(-1)

        if D == 3:
            vp = pred / pred.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            vt = tgt / tgt.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        elif D == 2:
            # (az, el) in degrees -> unit vectors
            az_p = pred[..., 0] * (torch.pi / 180)
            el_p = pred[..., 1] * (torch.pi / 180)
            az_t = tgt[..., 0] * (torch.pi / 180)
            el_t = tgt[..., 1] * (torch.pi / 180)

            def to_vec(az, el):
                x = torch.cos(el) * torch.cos(az)
                y = torch.cos(el) * torch.sin(az)
                z = torch.sin(el)
                return torch.stack([x, y, z], dim=-1)

            vp = to_vec(az_p, el_p)
            vt = to_vec(az_t, el_t)
            vp = vp / vp.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            vt = vt / vt.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        else:
            raise ValueError(f"Unsupported loc dimension: {D}; expected 2 or 3.")

        cos_sim = (vp * vt).sum(dim=-1).clamp(-1 + 1e-6, 1 - 1e-6)  # [K,N]
        ang = torch.acos(cos_sim)  # radians
        return ang / torch.pi  # normalize to [0,1], lower is better

    def _pairwise_cls_cost(
        self, pred_cls_logit: torch.Tensor, gt_cls_onehot: torch.Tensor
    ) -> torch.Tensor:
        """
        pred_cls_logit: [K, C]
        gt_cls_onehot: [N, C] (multi-label one-hot)
        Returns cost matrix [K, N] using focal BCE for consistency with final loss.
        Lower is better.
        """
        if gt_cls_onehot.numel() == 0:
            return torch.zeros(pred_cls_logit.size(0), 0, device=pred_cls_logit.device)

        logits = pred_cls_logit[:, None, :].expand(
            -1, gt_cls_onehot.size(0), -1
        )  # [K,N,C]
        targets = gt_cls_onehot[None, :, :].expand(
            pred_cls_logit.size(0), -1, -1
        )  # [K,N,C]

        # Use same focal loss as final loss for consistency
        focal = focal_bce_with_logits(
            logits,
            targets,
            alpha=self.cls_focal_alpha,
            gamma=self.cls_focal_gamma,
            reduction="none",
            pos_weight=self.cls_pos_weight,
        )  # [K,N,C]

        # Simple average over classes - no additional weighting
        cost = focal.mean(dim=-1)  # [K,N]
        return cost

    @staticmethod
    def _robust_normalize(cost_matrix: torch.Tensor) -> torch.Tensor:
        """
        Min-max normalize cost matrix to [0, 1] range.
        More stable than z-score normalization, especially with few sources.
        """
        min_val = cost_matrix.min()
        max_val = cost_matrix.max()
        range_val = max_val - min_val

        # If all costs are identical, return zeros (no preference)
        if range_val < 1e-6:
            return torch.zeros_like(cost_matrix)

        return (cost_matrix - min_val) / range_val

    def _hungarian(self, pred_loc, pred_cls_logit, gt_loc, gt_cls):
        """
        Returns matched indices (pred_indices, gt_indices) for one sample without objectness.
        """
        N = gt_loc.size(0)
        if N == 0:
            device = pred_loc.device
            return (
                torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.long, device=device),
            )

        with torch.no_grad():
            loc_cost = self._pairwise_loc_cost(pred_loc, gt_loc)  # [K,N]
            cls_cost = self._pairwise_cls_cost(pred_cls_logit, gt_cls)  # [K,N]

            # Use robust min-max normalization instead of unstable z-score
            loc_cost_norm = self._robust_normalize(loc_cost)
            cls_cost_norm = self._robust_normalize(cls_cost)

            total_cost = (
                self.loc_cost_weight * loc_cost_norm
                + self.cls_cost_weight * cls_cost_norm
            )

            cost_np = total_cost.detach().cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(cost_np)

            pred_inds = torch.as_tensor(
                row_ind, dtype=torch.long, device=pred_loc.device
            )
            gt_inds = torch.as_tensor(col_ind, dtype=torch.long, device=pred_loc.device)

        return pred_inds, gt_inds

    def forward(self, outputs, targets):
        """
        outputs: tuple (loc_out, cls_logit)
          - loc_out: [B, K, D]
          - cls_logit: [B, K, C]
        targets: list[dict] of length B
          - 'loc': [N_i, D]
          - 'cls': [N_i, C]
        """
        loc_out, cls_logit = outputs
        B, K = loc_out.shape[:2]

        device = loc_out.device
        total_loc = torch.zeros((), device=device)
        total_cls = torch.zeros((), device=device)
        samples_with_sources = 0

        for b in range(B):
            gt_loc = targets[b]["loc"].to(loc_out.device)  # [N,D]
            gt_cls = targets[b]["cls"].to(loc_out.device)  # [N,C]

            pred_loc_b = loc_out[b]  # [K,D]
            pred_cls_b = cls_logit[b]  # [K,C]

            N = gt_loc.size(0)
            if N > 0:
                samples_with_sources += 1
                pred_idx, gt_idx = self._hungarian(
                    pred_loc_b, pred_cls_b, gt_loc, gt_cls
                )

                # Localization loss on matched
                loc_loss_b = self.loc_criterion(pred_loc_b[pred_idx], gt_loc[gt_idx])

                # Classification loss on matched (multi-label focal BCE)
                cls_loss_b = focal_bce_with_logits(
                    pred_cls_b[pred_idx],
                    gt_cls[gt_idx],
                    alpha=self.cls_focal_alpha,
                    gamma=self.cls_focal_gamma,
                    reduction="mean",
                    pos_weight=self.cls_pos_weight,
                )
            else:
                # No sources: no loc/cls loss contribution
                loc_loss_b = torch.tensor(0.0, device=loc_out.device)
                cls_loss_b = torch.tensor(0.0, device=loc_out.device)

            total_loc += loc_loss_b
            total_cls += cls_loss_b

        denom = max(samples_with_sources, 1)
        loc_loss = total_loc / denom
        cls_loss = total_cls / denom

        if (
            getattr(self, "task_weighter", None) is not None
            and samples_with_sources > 0
        ):
            total = self.task_weighter({"loc": loc_loss, "cls": cls_loss})
        else:
            total = self.loc_weight * loc_loss + self.cls_weight * cls_loss

        return {
            "total": total,
            "loc": loc_loss.detach(),
            "cls": cls_loss.detach(),
        }


def get_localization_criterion(name: str):
    """
    Helper to get localization criterion by name.
    - "MSE": Mean squared error on coordinates
    - "AD": Angular distance loss (2D deg -> 3D unit vec, or 3D Cartesian)
    - "MIX": MSE + Angular distance
    """
    if name == "MSE":
        return nn.MSELoss()
    if name == "AD":
        return AngularLossWithCartesianCoordinate()
    if name == "MIX":
        return MixWithCartesianCoordinate()
    raise ValueError("Unknown localization loss")
