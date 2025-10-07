import torch

from criterion_bast import SetCriterionBAST

import os

import glob


def build_target_list(loc_lists, cls_lists):
    return [{"loc": l, "cls": c} for l, c in zip(loc_lists, cls_lists)]


def compute_batch_metrics(
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    targets: list[dict[str, torch.Tensor]],
    criterion: "SetCriterionBAST",
    cls_threshold: float,
) -> dict[str, float | int]:
    """
    Computes:
      - avg 3D location error (Euclidean distance in Cartesian coords) over matched pairs
      - exact match accuracy (all class bits correct) over matched pairs
      - element-wise accuracy over matched pairs
    """
    loc_out, obj_logit, cls_logit = outputs
    B = int(loc_out.shape[0])
    loc_err_sum = 0.0
    matched_pairs = 0
    exact_match = 0
    elem_correct = 0
    elem_total = 0
    with torch.no_grad():
        for b in range(B):
            gt_loc = targets[b]["loc"].to(loc_out.device)
            gt_cls = targets[b]["cls"].to(loc_out.device)
            N = int(gt_loc.size(0))
            if N == 0:
                continue
            pred_loc_b = loc_out[b]
            pred_obj_b = obj_logit[b]
            pred_cls_b = cls_logit[b]
            pred_idx, gt_idx = criterion._hungarian(
                pred_loc_b, pred_obj_b, pred_cls_b, gt_loc, gt_cls
            )
            if pred_idx.numel() == 0:
                continue
            pl = pred_loc_b[pred_idx]  # [M, 3] predicted Cartesian (x,y,z)
            gl = gt_loc[gt_idx]  # [M, 3] ground-truth Cartesian (x,y,z)
            # 3D Euclidean distance per matched pair
            loc_err = torch.linalg.norm(pl - gl, dim=-1)
            loc_err_sum += loc_err.sum().item()
            matched_pairs += int(pl.size(0))

            # Classification metrics on matched pairs
            pred_cls_prob = torch.sigmoid(pred_cls_b[pred_idx])
            pred_bin = (pred_cls_prob >= cls_threshold).float()
            gt_cls_m = gt_cls[gt_idx]
            exact_match += pred_bin.eq(gt_cls_m).all(dim=1).sum().item()
            elem_correct += pred_bin.eq(gt_cls_m).sum().item()
            elem_total += int(pred_bin.numel())
    metrics: dict[str, float | int] = {
        "loc_err": (loc_err_sum / matched_pairs) if matched_pairs else 0.0,
        "cls_exact": (exact_match / matched_pairs) if matched_pairs else 0.0,
        "cls_elem_acc": (elem_correct / elem_total) if elem_total else 0.0,
        "matched_pairs": matched_pairs,
    }
    return metrics



def find_latest_checkpoint(directory: str, pattern: str = "*.pt") -> str | None:
    matches = glob.glob(os.path.join(directory, pattern))
    if not matches:
        return None
    matches.sort(key=os.path.getmtime, reverse=True)
    return matches[0]


def strip_module_prefix(state_dict: dict) -> dict:
    """
    Handles loading a DataParallel-saved state_dict into a non-parallel model.
    """
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
