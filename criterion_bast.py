import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

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




def focal_bce_with_logits(logits, targets, alpha=0.25, gamma=2.0, reduction="mean"):
    """
    Optional focal loss for multi-label classification.
    """
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
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
    Combines:
      - Hungarian matching
      - Localization loss (your chosen criterion)
      - Classification BCE (per matched slot)
      - Objectness BCE (matched=1, unmatched=0)
    """

    def __init__(
        self,
        loc_criterion: nn.Module,
        num_classes: int,
        loc_weight=1.0,
        cls_weight=1.0,
        obj_weight=1.0,
        cls_cost_weight=0.25,
        loc_cost_weight=1.0,
        obj_cost_weight=0.1,
        max_sources=4,
    ):
        super().__init__()
        self.loc_criterion = loc_criterion
        self.num_classes = num_classes
        self.loc_weight = loc_weight
        self.cls_weight = cls_weight
        self.obj_weight = obj_weight
        # Matching cost weights (can differ from loss weights)
        self.cls_cost_weight = cls_cost_weight
        self.loc_cost_weight = loc_cost_weight
        self.obj_cost_weight = obj_cost_weight
        self.max_sources = max_sources

    @staticmethod
    def _pairwise_loc_cost(pred_loc, gt_loc):
        if gt_loc.numel() == 0:
            return torch.zeros(pred_loc.size(0), 0, device=pred_loc.device)
        # pred_loc, gt_loc: [K,2] / [N,2] (az_deg 0..360, el_deg -90..90)
        pred = pred_loc[:, None, :]  # [K,1,2]
        tgt = gt_loc[None, :, :]  # [1,N,2]
        # Convert to radians
        az_p = pred[..., 0] * (torch.pi / 180)
        el_p = pred[..., 1] * (torch.pi / 180)
        az_t = tgt[..., 0] * (torch.pi / 180)
        el_t = tgt[..., 1] * (torch.pi / 180)

        # 3D vectors
        def to_vec(az, el):
            x = torch.cos(el) * torch.cos(az)
            y = torch.cos(el) * torch.sin(az)
            z = torch.sin(el)
            return torch.stack([x, y, z], dim=-1)

        vp = to_vec(az_p, el_p)
        vt = to_vec(az_t, el_t)
        vp = vp / vp.norm(dim=-1, keepdim=True)
        vt = vt / vt.norm(dim=-1, keepdim=True)
        cos_sim = (vp * vt).sum(dim=-1).clamp(-1 + 1e-6, 1 - 1e-6)  # [K,N]
        # Angular distance in radians -> convert to degrees (optional) or keep radians
        ang = torch.acos(cos_sim)  # radians
        return ang  # lower = better

    @staticmethod
    def _pairwise_cls_cost(pred_cls_logit, gt_cls_onehot, pos_weight=7.0):
        """
        pred_cls_logit: [K,C]
        gt_cls_onehot: [N,C] (usually one 1 per row)
        Returns classification cost matrix [K,N] using summed BCE per gt source.
        """
        if gt_cls_onehot.numel() == 0:
            return torch.zeros(pred_cls_logit.size(0), 0, device=pred_cls_logit.device)
        p = torch.sigmoid(pred_cls_logit)  # [K,C]
        p = torch.clamp(p, 1e-6, 1 - 1e-6)
        # BCE per class: -(y*log p + (1-y)*log (1-p))
        # Expand dims to broadcast: [K,1,C] vs [1,N,C]
        p_exp = p[:, None, :]
        y_exp = gt_cls_onehot[None, :, :]
        pos_term = -(y_exp * torch.log(p_exp))
        neg_term = -((1 - y_exp) * torch.log(1 - p_exp))
        cost = pos_weight * pos_term + neg_term
        return cost.sum(-1)

    def _hungarian(self, pred_loc, pred_obj_logit, pred_cls_logit, gt_loc, gt_cls):
        """
        Returns matched indices (pred_indices, gt_indices) for one sample.
        """
        K = pred_loc.size(0)
        N = gt_loc.size(0)
        if N == 0:
            return torch.empty(
                0, dtype=torch.long, device=pred_loc.device
            ), torch.empty(0, dtype=torch.long, device=pred_loc.device)

        with torch.no_grad():
            loc_cost = self._pairwise_loc_cost(pred_loc, gt_loc)  # [K,N]
            cls_cost = self._pairwise_cls_cost(pred_cls_logit, gt_cls)  # [K,N]
            obj_prob = torch.sigmoid(pred_obj_logit).unsqueeze(1)  # [K,1]
            obj_cost = -obj_prob.expand(K, N)  # encourage higher objectness

            total_cost = (
                self.loc_cost_weight * loc_cost
                + self.cls_cost_weight * cls_cost
                + self.obj_cost_weight * obj_cost
            )

            # Hungarian expects CPU numpy
            cost_np = total_cost.detach().cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(cost_np)
            # Keep only assignments where row<K and col<N (they always are)
            pred_inds = torch.as_tensor(
                row_ind, dtype=torch.long, device=pred_loc.device
            )
            gt_inds = torch.as_tensor(col_ind, dtype=torch.long, device=pred_loc.device)
        return pred_inds, gt_inds

    def forward(self, outputs, targets):
        """
        outputs: (loc_out, obj_logit, cls_logit)
          loc_out: [B,K,2], obj_logit: [B,K], cls_logit: [B,K,C]
        targets: list of dicts length B, each:
          {'loc': [N_i,2], 'cls': [N_i,C]}
        """
        loc_out, obj_logit, cls_logit = outputs
        B, K = loc_out.shape[0], loc_out.shape[1]

        total_loc = 0.0
        total_cls = 0.0
        total_obj = 0.0
        samples_with_sources = 0

        for b in range(B):
            gt_loc = targets[b]["loc"].to(loc_out.device)  # [N,2]
            gt_cls = targets[b]["cls"].to(loc_out.device)  # [N,C]

            pred_loc_b = loc_out[b]  # [K,2]
            pred_obj_b = obj_logit[b]  # [K]
            pred_cls_b = cls_logit[b]  # [K,C]

            N = gt_loc.size(0)
            if N > 0:
                samples_with_sources += 1
                pred_idx, gt_idx = self._hungarian(
                    pred_loc_b, pred_obj_b, pred_cls_b, gt_loc, gt_cls
                )

                # Localization loss on matched
                loc_loss_b = self.loc_criterion(pred_loc_b[pred_idx], gt_loc[gt_idx])

                # # Classification loss on matched
                # cls_loss_b = F.binary_cross_entropy_with_logits(
                #     pred_cls_b[pred_idx], gt_cls[gt_idx], reduction="mean"
                # )
                # Classification loss on matched
                cls_loss_b = focal_bce_with_logits(
                    pred_cls_b[pred_idx],
                    gt_cls[gt_idx],
                    alpha=0.25,
                    gamma=2.0,
                    reduction="mean",
                )

                # Objectness:
                obj_target = torch.zeros_like(pred_obj_b)
                obj_target[pred_idx] = 1.0
                obj_loss_b = F.binary_cross_entropy_with_logits(
                    pred_obj_b, obj_target, reduction="mean"
                )
            else:
                # No sources: all objectness targets = 0
                loc_loss_b = torch.tensor(0.0, device=loc_out.device)
                cls_loss_b = torch.tensor(0.0, device=loc_out.device)
                obj_loss_b = F.binary_cross_entropy_with_logits(
                    obj_logit[b], torch.zeros_like(obj_logit[b]), reduction="mean"
                )

            total_loc += loc_loss_b
            total_cls += cls_loss_b
            total_obj += obj_loss_b

        denom = max(samples_with_sources, 1)
        loc_loss = total_loc / denom
        cls_loss = total_cls / denom
        obj_loss = total_obj / B  # count objectness over all samples

        total = (
            self.loc_weight * loc_loss
            + self.cls_weight * cls_loss
            + self.obj_weight * obj_loss
        )

        return {
            "total": total,
            "loc": loc_loss.detach(),
            "cls": cls_loss.detach(),
            "obj": obj_loss.detach(),
        }


# Helper to get localization criterion
def get_localization_criterion(name: str):
    if name == "MSE":
        return nn.MSELoss()
    if name == "AD":
        return AngularLossWithCartesianCoordinate()
    if name == "MIX":
        return MixWithCartesianCoordinate()
    raise ValueError("Unknown localization loss")
