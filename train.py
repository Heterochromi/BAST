# %%
"""
BAST Multitask Training Script with added metrics:
 - Exact match (per-source multi-label classification)
 - Element-wise classification accuracy
 - Average azimuth error (deg, circular)
 - Average elevation error (deg, linear)

Added (at bottom):
 - Single-sample inference utility: load a saved model checkpoint and run one WAV file.
"""

# %%
from network.BAST import (
    BAST_Variant,
    AngularLossWithCartesianCoordinate,
    MixWithCartesianCoordinate,
    # SphericalVectorLoss,
)
from data_loading import MultiSourceSpectrogramDataset
import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from datetime import datetime

# %%
# initialization params from model commented here
# image_size,  # e.g., (129, 61) - (freq, time)
# patch_size,  # e.g., 16
# patch_overlap,  # e.g., 10
# num_coordinates_output,  # e.g., 2 (azimuth, elevation) or 3 (x,y,z)
# dim,  # embedding dimension, e.g., 512
# num_encoder_layers=6,
# num_decoder_layers=3,
# heads,  # number of attention heads, e.g., 8
# mlp_dim,  # MLP dimension, e.g., 1024
# num_encoder_layers=6,
# num_decoder_layers=3,
# channels=2,
# dim_head=64,
# dropout=0.2,
# emb_dropout=0.0,
# binaural_integration="CROSS_ATTN",
# max_sources=4,
# num_classes_cls=1,

# Configuration
CSV_PATH = "tensor_metadata.csv"
SPECTROGRAM_SIZE = [64, 19]  # [Freq (n_mels), Time frames]
PATCH_SIZE = 8
PATCH_OVERLAP = 4
NUM_OUTPUT = 3
EMBEDDING_DIM = 256
TRANSFORMER_ENCODER_DEPTH = 4
TRANSFORMER_DECODER_DEPTH = 2
TRANSFORMER_HEADS = 8
TRANSFORMER_MLP_DIM = 512
TRANSFORMER_DIM_HEAD = 32
INPUT_CHANNEL = 2
DROPOUT = 0.2
EMB_DROPOUT = 0.2
TRANSFORMER_POOL = "conv"

EPOCHS = 60
BATCH_SIZE = 1800
LEARNING_RATE = 0.0001
TEST_SPLIT = 0.3
VAL_SPLIT = 0.3
SEED = 42

LOC_WEIGHT = 0.1
CLS_WEIGHT = 6
OBJ_WEIGHT = 0.1
BINAURAL_INTEGRATION = "CROSS_ATTN"
SHARE_WEIGHTS = False
MAX_SOURCES = 4
LOSS_TYPE = "MIX"

CLS_THRESHOLD = 0.5

GPU_LIST = [0] if torch.cuda.is_available() else []
MODEL_SAVE_DIR = "./output/models/"
MODEL_NAME = "BAST"
NUM_WORKERS = 4

# %%


# Helper to get localization criterion
def get_localization_criterion(name: str):
    if name == "MSE":
        return nn.MSELoss()
    if name == "AD":
        return AngularLossWithCartesianCoordinate()
    if name == "MIX":
        return MixWithCartesianCoordinate()
    raise ValueError("Unknown localization loss")


# %%

torch.manual_seed(SEED)
np.random.seed(SEED)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

dataset = MultiSourceSpectrogramDataset(CSV_PATH, tensor_dir="output_tensors")
num_classes = dataset.num_classes
print(
    f"[{datetime.now()}] (MultiSource) Samples: {len(dataset)} | Classes: {num_classes}"
)

test_size = int(len(dataset) * TEST_SPLIT)
remaining_size = len(dataset) - test_size
remaining_ds, test_ds = random_split(
    dataset, [remaining_size, test_size], generator=torch.Generator().manual_seed(SEED)
)
val_size = int(remaining_size * VAL_SPLIT)
train_size = remaining_size - val_size
train_ds, val_ds = random_split(
    remaining_ds,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED + 1),
)

# %%


def multisource_collate(batch):
    specs, loc_lists, cls_lists, n_list = zip(*batch)
    specs = torch.stack(specs, dim=0)
    return specs, loc_lists, cls_lists, torch.tensor(n_list, dtype=torch.long)


train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    collate_fn=multisource_collate,
)
val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    collate_fn=multisource_collate,
)
test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    collate_fn=multisource_collate,
)

print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

# %%

print(f"[{datetime.now()}] Building model ...")
net = BAST_Variant(
    image_size=SPECTROGRAM_SIZE,
    patch_size=PATCH_SIZE,
    patch_overlap=PATCH_OVERLAP,
    num_coordinates_output=NUM_OUTPUT,
    dim=EMBEDDING_DIM,
    num_encoder_layers=TRANSFORMER_ENCODER_DEPTH,
    num_decoder_layers=TRANSFORMER_DECODER_DEPTH,
    heads=TRANSFORMER_HEADS,
    mlp_dim=TRANSFORMER_MLP_DIM,
    channels=INPUT_CHANNEL,
    dim_head=TRANSFORMER_DIM_HEAD,
    dropout=DROPOUT,
    emb_dropout=EMB_DROPOUT,
    binaural_integration=BINAURAL_INTEGRATION,
    max_sources=MAX_SOURCES,
    num_classes_cls=num_classes,
)
# %%
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
# %%
if use_cuda and GPU_LIST:
    net = nn.DataParallel(net, device_ids=GPU_LIST).to(device)
else:
    net = net.to(device)

print(f"Model built successfully. Using device: {device}")
print(f"Model parameters: {sum(p.numel() for p in net.parameters())}")

# %%
from criterion_bast import SetCriterionBAST

criterion = SetCriterionBAST(
    loc_criterion=get_localization_criterion(LOSS_TYPE),
    num_classes=num_classes,
    loc_weight=LOC_WEIGHT,
    cls_weight=CLS_WEIGHT,
    obj_weight=OBJ_WEIGHT,
    cls_cost_weight=5,
    loc_cost_weight=0.1,
    obj_cost_weight=0.1,
    max_sources=MAX_SOURCES,
)

optimizer = torch.optim.AdamW(net.parameters(), lr=LEARNING_RATE, weight_decay=0.0)

# %%


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


def validate(epoch):
    net.eval()
    running_loss = {"total": 0.0, "loc": 0.0, "cls": 0.0, "obj": 0.0, "batches": 0}
    metric_accum = {
        "loc_err": 0.0,
        "cls_exact": 0.0,
        "cls_elem_acc": 0.0,
        "matched_pairs": 0,
        "batches": 0,
    }
    with torch.no_grad():
        for specs, loc_lists, cls_lists, n_list in val_loader:
            specs = specs.to(device)
            outputs = net(specs)
            targets = build_target_list(loc_lists, cls_lists)
            losses = criterion(outputs, targets)
            for k in ("total", "loc", "cls", "obj"):
                running_loss[k] += float(losses[k])
            running_loss["batches"] += 1
            batch_metrics = compute_batch_metrics(
                outputs, targets, criterion, CLS_THRESHOLD
            )
            mp = batch_metrics["matched_pairs"]
            if mp > 0:
                for mk in ("loc_err", "cls_exact", "cls_elem_acc"):
                    metric_accum[mk] += batch_metrics[mk] * mp
                metric_accum["matched_pairs"] += mp
            metric_accum["batches"] += 1
    for k in ("total", "loc", "cls", "obj"):
        running_loss[k] /= max(running_loss["batches"], 1)
    if metric_accum["matched_pairs"] > 0:
        for mk in ("loc_err", "cls_exact", "cls_elem_acc"):
            metric_accum[mk] /= metric_accum["matched_pairs"]
    print(
        f"[VAL] Epoch {epoch} | Total {running_loss['total']:.4f} | Loc {running_loss['loc']:.4f} | "
        f"Cls {running_loss['cls']:.4f} | Obj {running_loss['obj']:.4f} | "
        f"loc_err {metric_accum['loc_err']:.3f} |"
        f"ClsExact {metric_accum['cls_exact']:.3f} | ClsElem {metric_accum['cls_elem_acc']:.3f}"
    )
    return {**running_loss, **metric_accum}


# %%
best_val = float("inf")
for epoch in range(1, EPOCHS + 1):
    net.train()
    epoch_losses = {"total": 0.0, "loc": 0.0, "cls": 0.0, "obj": 0.0, "batches": 0}
    metric_epoch = {
        "loc_err": 0.0,
        "cls_exact": 0.0,
        "cls_elem_acc": 0.0,
        "matched_pairs": 0,
        "batches": 0,
    }
    for specs, loc_lists, cls_lists, n_list in train_loader:
        specs = specs.to(device)
        outputs = net(specs)
        targets = build_target_list(loc_lists, cls_lists)
        loss_dict = criterion(outputs, targets)
        loss = loss_dict["total"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_losses["total"] += float(loss_dict["total"])
        epoch_losses["loc"] += float(loss_dict["loc"])
        epoch_losses["cls"] += float(loss_dict["cls"])
        epoch_losses["obj"] += float(loss_dict["obj"])
        epoch_losses["batches"] += 1
        batch_metrics = compute_batch_metrics(
            outputs, targets, criterion, CLS_THRESHOLD
        )
        mp = batch_metrics["matched_pairs"]
        if mp > 0:
            for mk in ("loc_err", "cls_exact", "cls_elem_acc"):
                metric_epoch[mk] += batch_metrics[mk] * mp
            metric_epoch["matched_pairs"] += mp
        metric_epoch["batches"] += 1
    for k in ("total", "loc", "cls", "obj"):
        epoch_losses[k] /= max(epoch_losses["batches"], 1)
    if metric_epoch["matched_pairs"] > 0:
        for mk in ("loc_err", "cls_exact", "cls_elem_acc"):
            metric_epoch[mk] /= metric_epoch["matched_pairs"]
    print(
        f"[TRAIN] Epoch {epoch} | Total {epoch_losses['total']:.4f} | Loc {epoch_losses['loc']:.4f} | "
        f"Cls {epoch_losses['cls']:.4f} | Obj {epoch_losses['obj']:.4f} | "
        f"loc_err {metric_epoch['loc_err']:.3f}      |"
        f"ClsExact {metric_epoch['cls_exact']:.3f} | ClsElem {metric_epoch['cls_elem_acc']:.3f} |"
    )
    val_metrics = validate(epoch)
    if val_metrics["total"] < best_val:
        best_val = val_metrics["total"]
        save_path = os.path.join(
            MODEL_SAVE_DIR,
            f"{MODEL_NAME}_{BINAURAL_INTEGRATION}_{LOSS_TYPE}_DET_{'SP' if SHARE_WEIGHTS else 'NSP'}_best.pt",
        )
        torch.save(net.state_dict(), save_path)
        print(f"  -> New best model saved to {save_path}")
# End of training

# ---------------------- Test one sample (Single WAV Inference) ------------------------------#
# %%
"""
This section:
 1. Locates a saved model checkpoint (best *.pt).
 2. Loads the weights into a fresh model instance (handling DataParallel prefixes).
 3. Converts a target WAV file into the expected mel-spectrogram tensor.
 4. Pads/crops time dimension to match SPECTROGRAM_SIZE (Freq=64, Time=8).
 5. Runs a forward pass and prints per-slot predictions:
      - Objectness probability
      - (x, y, z) Cartesian coordinates
      - Per-class probabilities (and top active classes)
Adjust SINGLE_WAV_PATH below to point to a test stereo WAV file.
"""

# User-adjustable parameters for single-sample inference
DO_SINGLE_SAMPLE_TEST = True
OBJECTNESS_THRESHOLD = 0.4  # filter low-confidence detection slots
TOP_K_CLASSES = 4  # how many top class probabilities to display

from mel_spec_tensor import generate_mel_spectrogram_torch_tensor


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


def build_model_for_inference(num_classes: int) -> BAST_Variant:
    model = BAST_Variant(
        image_size=SPECTROGRAM_SIZE,
        patch_size=PATCH_SIZE,
        patch_overlap=PATCH_OVERLAP,
        num_coordinates_output=NUM_OUTPUT,
        dim=EMBEDDING_DIM,
        depth=TRANSFORMER_DEPTH,
        heads=TRANSFORMER_HEADS,
        mlp_dim=TRANSFORMER_MLP_DIM,
        channels=INPUT_CHANNEL,
        dim_head=TRANSFORMER_DIM_HEAD,
        dropout=DROPOUT,
        emb_dropout=EMB_DROPOUT,
        binaural_integration=BINAURAL_INTEGRATION,
        max_sources=MAX_SOURCES,
        num_classes_cls=num_classes,
    )
    return model


def load_checkpoint_into_model(model: nn.Module, ckpt_path: str, device: torch.device):
    raw_state = torch.load(ckpt_path, map_location=device)
    if isinstance(raw_state, dict) and "state_dict" in raw_state:
        raw_state = raw_state["state_dict"]
    raw_state = strip_module_prefix(raw_state)
    missing, unexpected = model.load_state_dict(raw_state, strict=False)
    print(f"[Inference] Loaded weights from {ckpt_path}")
    if missing:
        print(f"  -> Missing keys ({len(missing)}): {missing}")
    if unexpected:
        print(f"  -> Unexpected keys ({len(unexpected)}): {unexpected}")


def prepare_spectrogram_tensor(mel_tensor: torch.Tensor) -> torch.Tensor:
    """
    Ensure shape [2, 64, 8]: (Channels, Freq, Time)
    If time frames < target, pad with zeros at end; if > target, center-crop.
    """
    if mel_tensor.ndim != 3 or mel_tensor.shape[0] != 2:
        raise ValueError(
            f"Expected raw mel tensor shape [2, F, T], got {tuple(mel_tensor.shape)}"
        )
    target_f, target_t = SPECTROGRAM_SIZE
    # Frequency sanity (if mismatch, we can interpolate or error)
    if mel_tensor.shape[1] != target_f:
        raise ValueError(
            f"Frequency bins mismatch: got {mel_tensor.shape[1]}, expected {target_f}"
        )
    cur_t = mel_tensor.shape[2]
    if cur_t == target_t:
        return mel_tensor
    if cur_t < target_t:
        pad_amt = target_t - cur_t
        pad_tensor = torch.zeros((2, target_f, pad_amt), dtype=mel_tensor.dtype)
        mel_tensor = torch.cat([mel_tensor, pad_tensor], dim=2)
    else:
        # center crop
        start = max((cur_t - target_t) // 2, 0)
        mel_tensor = mel_tensor[:, :, start : start + target_t]
    return mel_tensor


def run_single_wav_inference(
    wav_path: str,
    checkpoint_path: str | None,
    objectness_threshold: float = 0.5,
    top_k_classes: int = 3,
):
    if not os.path.exists(wav_path):
        print(f"[Inference] WAV file not found: {wav_path}")
        return
    print(f"[Inference] Preparing model for single-sample test...")
    inf_model = build_model_for_inference(num_classes=dataset.num_classes).to(device)
    inf_model.eval()

    # If no explicit checkpoint path, try to auto-find latest
    if checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint(MODEL_SAVE_DIR, "*.pt")
        if checkpoint_path is None:
            print("[Inference] No checkpoint found in model directory.")
            return

    load_checkpoint_into_model(inf_model, checkpoint_path, device)

    print(f"[Inference] Generating mel spectrogram for: {wav_path}")
    with torch.no_grad():
        mel_tensor = generate_mel_spectrogram_torch_tensor(
            wav_path, output_path=None, return_tensor=True
        )
        mel_tensor = prepare_spectrogram_tensor(mel_tensor)  # shape [2, 64, 8]
        batch = mel_tensor.unsqueeze(0).to(device)  # [1, 2, 64, 8]
        loc_out, obj_logit, cls_logit = inf_model(batch)
        # Shapes: loc_out [1,K,3], obj_logit [1,K], cls_logit [1,K,C]
        loc_out = loc_out[0]
        obj_prob = torch.sigmoid(obj_logit[0])
        cls_prob = torch.sigmoid(cls_logit[0])

        # Sort by objectness
        sorted_indices = torch.argsort(obj_prob, descending=True)
        print("\n[Inference] Predictions (sorted by objectness):")
        print(
            f"{'Slot':<4} {'ObjProb':>8} | {'x':>8} {'y':>8} {'z':>8} | Top classes (prob)"
        )
        print("-" * 70)
        for rank, idx in enumerate(sorted_indices.tolist()):
            p_obj = obj_prob[idx].item()
            if p_obj < objectness_threshold:
                continue
            x, y, z = loc_out[idx].tolist()
            cls_vec = cls_prob[idx]
            # Top-k classes
            topk = torch.topk(cls_vec, k=min(top_k_classes, cls_vec.shape[0]))
            class_entries = []
            for c_idx, c_prob in zip(topk.indices.tolist(), topk.values.tolist()):
                cname = dataset.index_to_class.get(c_idx, f"id{c_idx}")
                class_entries.append(f"{cname}:{c_prob:.2f}")
            class_str = " ".join(class_entries)
            print(f"{idx:<4} {p_obj:8.3f} | {x:8.2f} {y:8.2f} {z:8.2f} | {class_str}")
        print("-" * 70)
        print("Done.")


# %%
# Execute single-sample inference if enabled
if DO_SINGLE_SAMPLE_TEST:
    SINGLE_WAV_PATH = (
        "dataset_parallel_100ms/sample_0057.wav"  # <-- Replace with your test WAV path
    )
    print("\n================ Single WAV Inference ================")
    # You can hardcode a checkpoint path here if desired:
    EXPLICIT_WEIGHTS_PATH = (
        None  # e.g., "./output/models/BAST_SUB_AZEL_DET_NSP_vanilla_best.pt"
    )
    run_single_wav_inference(
        wav_path=SINGLE_WAV_PATH,
        checkpoint_path=EXPLICIT_WEIGHTS_PATH,
        objectness_threshold=OBJECTNESS_THRESHOLD,
        top_k_classes=TOP_K_CLASSES,
    )
    print("======================================================\n")


# %%
def test(epoch):
    net.eval()
    running_loss = {"total": 0.0, "loc": 0.0, "cls": 0.0, "obj": 0.0, "batches": 0}
    metric_accum = {
        "loc_err": 0.0,
        "cls_exact": 0.0,
        "cls_elem_acc": 0.0,
        "matched_pairs": 0,
        "batches": 0,
    }
    with torch.no_grad():
        for specs, loc_lists, cls_lists, n_list in train_loader:
            specs = specs.to(device)
            outputs = net(specs)
            targets = build_target_list(loc_lists, cls_lists)
            losses = criterion(outputs, targets)
            for k in ("total", "loc", "cls", "obj"):
                running_loss[k] += float(losses[k])
            running_loss["batches"] += 1
            batch_metrics = compute_batch_metrics(outputs, targets, criterion, 0.3)
            mp = batch_metrics["matched_pairs"]
            if mp > 0:
                for mk in ("loc_err", "cls_exact", "cls_elem_acc"):
                    metric_accum[mk] += batch_metrics[mk] * mp
                metric_accum["matched_pairs"] += mp
            metric_accum["batches"] += 1
    for k in ("total", "loc", "cls", "obj"):
        running_loss[k] /= max(running_loss["batches"], 1)
    if metric_accum["matched_pairs"] > 0:
        for mk in ("loc_err", "cls_exact", "cls_elem_acc"):
            metric_accum[mk] /= metric_accum["matched_pairs"]
    print(
        f"[VAL] Epoch {epoch} | Total {running_loss['total']:.4f} | Loc {running_loss['loc']:.4f} | "
        f"Cls {running_loss['cls']:.4f} | Obj {running_loss['obj']:.4f} | "
        f"loc_err {metric_accum['loc_err']:.3f}"
        f"ClsExact {metric_accum['cls_exact']:.3f} | ClsElem {metric_accum['cls_elem_acc']:.3f}"
    )
    return {**running_loss, **metric_accum}


# %%
test(1)


# # %%
# def inspect_one_batch():
#     specs, loc_lists, cls_lists, n_list = next(iter(test_loader))
#     specs = specs.to(device)
#     outputs = net(specs).to(device)
#     targets = build_target_list(loc_lists, cls_lists)
#     loc_out, obj_logit, cls_logit = outputs
#     b = 0
#     gt_loc = targets[b]["loc"]
#     gt_cls = targets[b]["cls"]
#     pred_loc_b = loc_out[b]
#     pred_obj_b = obj_logit[b]
#     pred_cls_b = cls_logit[b]
#     pi, gi = criterion._hungarian(pred_loc_b, pred_obj_b, pred_cls_b, gt_loc, gt_cls)
#     print("Batch0 matches (pred_idx -> gt_idx):")
#     for p, g in zip(pi.tolist(), gi.tolist()):
#         print(
#             f"  pred_slot {p} -> gt {g}  | pred_loc {pred_loc_b[p].detach().cpu().numpy()}  gt_loc {gt_loc[g].numpy()}"
#         )
#         pc = torch.sigmoid(pred_cls_b[p]).detach().cpu().numpy()
#         gc = gt_cls[g].numpy()
#         print("    cls probs:", pc, "gt one-hot:", gc)


# # %%
# inspect_one_batch()


# %%
#
from torch.utils.data import DataLoader, Subset
from debug_utils import overfit_subset

mini_indices = list(range(32))
mini_subset = Subset(train_ds, mini_indices)
mini_loader = DataLoader(
    mini_subset,
    batch_size=32,
    shuffle=True,
    num_workers=0,
    collate_fn=multisource_collate,
)

overfit_subset(mini_loader, net, criterion, device, steps=160, lr=1e-3)
# %%
import math

print("=== Dataset Sanity ===")
sample_idxs = [0, len(dataset) // 2, len(dataset) - 1]
for idx in sample_idxs:
    spec, loc_t, cls_t, n = dataset[idx]
    print(f"Idx {idx} n={n} spec={tuple(spec.shape)}")
    if n > 0:
        print("  Loc head (first 3):", loc_t[:3])
        print("  Class rows (first 3):", cls_t[:3])
print("Classes total:", dataset.num_classes)
