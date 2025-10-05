import torch
import torch.nn.functional as F


def init_objectness_bias(model, prior_pos_fraction=0.25):
    """
    Initialize objectness head bias so initial sigmoid(obj) ~ prior_pos_fraction.
    Assumes model has attribute obj_head that ends with a Linear layer.
    """
    logit = torch.log(
        torch.tensor(prior_pos_fraction) / (1 - torch.tensor(prior_pos_fraction))
    )
    for module in model.modules():
        # If you wrapped heads in nn.Sequential as suggested
        if isinstance(module, torch.nn.Sequential):
            last = module[-1]
            if (
                isinstance(last, torch.nn.Linear)
                and last.out_features == model.max_sources
            ):
                if last.bias is not None:
                    with torch.no_grad():
                        last.bias.fill_(logit.item())
    return logit.item()


def grad_norm(module):
    total = 0.0
    for p in module.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return total**0.5


def collect_cls_prob_stats(outputs, targets, criterion):
    """
    Returns:
      pos_sum, pos_cnt, neg_sum, neg_cnt
    Based only on matched pairs (Hungarian).
    """
    loc_out, obj_logit, cls_logit = outputs
    B = loc_out.size(0)
    pos_sum = neg_sum = 0.0
    pos_cnt = neg_cnt = 0
    with torch.no_grad():
        for b in range(B):
            gt_loc = targets[b]["loc"].to(loc_out.device)
            gt_cls = targets[b]["cls"].to(loc_out.device)
            if gt_loc.size(0) == 0:
                continue
            pred_loc_b = loc_out[b]
            pred_obj_b = obj_logit[b]
            pred_cls_b = cls_logit[b]
            pred_idx, gt_idx = criterion._hungarian(
                pred_loc_b, pred_obj_b, pred_cls_b, gt_loc, gt_cls
            )
            if pred_idx.numel() == 0:
                continue
            probs = torch.sigmoid(pred_cls_b[pred_idx])  # [M,C]
            gt_sel = gt_cls[gt_idx]  # [M,C]
            pos_mask = gt_sel == 1
            neg_mask = gt_sel == 0
            if pos_mask.any():
                pos_sum += probs[pos_mask].sum().item()
                pos_cnt += pos_mask.sum().item()
            if neg_mask.any():
                neg_sum += probs[neg_mask].sum().item()
                neg_cnt += neg_mask.sum().item()
    return pos_sum, pos_cnt, neg_sum, neg_cnt


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


def overfit_subset(dataloader, model, criterion, device, steps=200, lr=1e-3):
    """
    Quick overfit test on a tiny dataloader (e.g., first 32 samples).
    """
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for step in range(1, steps + 1):
        try:
            specs, loc_lists, cls_lists, n_list = next(iter(dataloader))
        except StopIteration:
            break
        specs = specs.to(device)
        outputs = model(specs)
        targets = [{"loc": l, "cls": c} for l, c in zip(loc_lists, cls_lists)]
        loss_dict = criterion(outputs, targets)
        loss = loss_dict["total"]
        opt.zero_grad()
        loss.backward()
        # inside overfit loop after loss.backward()
        opt.step()
        if step % 20 == 0:
            print(
                f"[OVERFIT] Step {step} Total {loss_dict['total']:.4f} Loc {loss_dict['loc']:.4f} "
                f"Cls {loss_dict['cls']:.4f} Obj {loss_dict['obj']:.4f}"
            )
    print("[OVERFIT] Done.")
