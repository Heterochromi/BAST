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
from network.BAST import BAST_Variant
from data_loading import MultiSourceSpectrogramDataset, multisource_collate
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from datetime import datetime
from utils import *
from ax.api.client import Client
from ax.api.configs import ChoiceParameterConfig, RangeParameterConfig
from criterion_bast import SetCriterionBAST, get_localization_criterion


# Configuration
CSV_PATH = "tensor_metadata.csv"
SPECTROGRAM_SIZE = [64, 19]  # [Freq (n_mels), Time frames]
PATCH_SIZE = 8
PATCH_OVERLAP = 4  # change this to ZERO later
NUM_OUTPUT = 3
EMBEDDING_DIM = 256
TRANSFORMER_ENCODER_DEPTH = 4
TRANSFORMER_DECODER_DEPTH = 2
TRANSFORMER_HEADS = 8
TRANSFORMER_MLP_DIM = 512
TRANSFORMER_DIM_HEAD = 32
DROPOUT = 0.2
EMB_DROPOUT = 0.2
LEARNING_RATE = 0.0001
CLS_COST_WEIGHT_HUNGARIAN = 5
LOC_COST_WEIGHT_HUNGARIAN = 0.1
OBJ_COST_WEIGHT_HUNGARIAN = 0.1
LOC_WEIGHT = 0.4
CLS_WEIGHT = 6
OBJ_WEIGHT = 1

INPUT_CHANNEL = 2
EPOCHS = 1
BATCH_SIZE = 210
TEST_SPLIT = 0.3
VAL_SPLIT = 0.3
SEED = 42

BINAURAL_INTEGRATION = "CROSS_ATTN"
MAX_SOURCES = 4
LOSS_TYPE = "MIX"

CLS_THRESHOLD = 0.5

GPU_LIST = [0] if torch.cuda.is_available() else []
MODEL_SAVE_DIR = "./output/models/"
MODEL_NAME = "BAST"
NUM_WORKERS = 4

# %%
HPO_client = Client()
# %%
#
# CLS_COST_WEIGHT_HUNGARIAN = 5
# LOC_COST_WEIGHT_HUNGARIAN = 0.1
# OBJ_COST_WEIGHT_HUNGARIAN = 0.1
# LOC_WEIGHT = 0.1
# CLS_WEIGHT = 6
# OBJ_WEIGHT = 0.1
parameters = [
    RangeParameterConfig(
        name="CLS_WEIGHT",
        bounds=(1, 10),
        parameter_type="float",
        scaling="linear",
    ),
    # RangeParameterConfig(
    #     name="LOC_WEIGHT",
    #     bounds=(0.1, 1),
    #     parameter_type="float",
    #     scaling="linear",
    # ),
    # RangeParameterConfig(
    #     name="OBJ_WEIGHT",
    #     bounds=(0.1, 1),
    #     parameter_type="float",
    #     scaling="linear",
    # ),
    RangeParameterConfig(
        name="CLS_COST_WEIGHT_HUNGARIAN",
        bounds=(0.1, 1),
        parameter_type="float",
        scaling="linear",
    ),
    RangeParameterConfig(
        name="LOC_COST_WEIGHT_HUNGARIAN",
        bounds=(0.1, 1),
        parameter_type="float",
        scaling="linear",
    ),
    RangeParameterConfig(
        name="OBJ_COST_WEIGHT_HUNGARIAN",
        bounds=(0.1, 1),
        parameter_type="float",
        scaling="linear",
    ),
    RangeParameterConfig(
        name="LEARNING_RATE",
        bounds=(1e-5, 1e-2),
        parameter_type="float",
        scaling="log",
    ),
    RangeParameterConfig(
        name="TRANSFORMER_ENCODER_DEPTH",
        bounds=(3, 9),
        parameter_type="int",
        scaling="linear",
    ),
    RangeParameterConfig(
        name="TRANSFORMER_DECODER_DEPTH",
        bounds=(2, 6),
        parameter_type="int",
        scaling="linear",
    ),
    ChoiceParameterConfig(
        name="EMBEDDING_DIM",
        values=[
            240,
            384,
            528,
            768,
            1536,
            2064,
        ],  # [3360, 4464,5808, 6672, 8688]  ALL EMBEDDING_DIM must be divisible by ALL TRANSFORMER_HEADS
        parameter_type="int",
        is_ordered=True,
    ),
    ChoiceParameterConfig(
        name="TRANSFORMER_HEADS",
        values=[4, 6, 8, 12, 16],  # Common divisors
        parameter_type="int",
        is_ordered=True,
    ),
    RangeParameterConfig(
        name="TRANSFORMER_MLP_RATIO",
        bounds=(2, 4),
        parameter_type="float",
        scaling="linear",
    ),
    RangeParameterConfig(
        name="EMB_DROPOUT",
        bounds=(0.0, 0.2),
        parameter_type="float",
        scaling="linear",
    ),
    RangeParameterConfig(
        name="DROPOUT",
        bounds=(0.0, 0.2),
        parameter_type="float",
        scaling="linear",
    ),
    RangeParameterConfig(
        name="WEIGHT_DECAY",
        bounds=(1e-6, 1e-2),
        parameter_type="float",
        scaling="log",
    ),
]

# %%
HPO_client.configure_experiment(
    parameters=parameters,
    name="Binaural SELD Experiment",
)


# %%
HPO_client.configure_optimization(
    objective="cls_exact,-loc_err,-total,cls_elem_acc,-cls,-loc,-obj",
    outcome_constraints=[
        "cls_exact >= 0.70",
        "loc_err <= 0.40",
        "cls_elem_acc >= 0.75",
    ],
)
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
def load_model_with_hpo_parameters(hpo_parameters):
    net = BAST_Variant(
        image_size=SPECTROGRAM_SIZE,
        patch_size=PATCH_SIZE,
        patch_overlap=PATCH_OVERLAP,
        num_coordinates_output=NUM_OUTPUT,
        dim=hpo_parameters["EMBEDDING_DIM"],
        num_encoder_layers=hpo_parameters["TRANSFORMER_ENCODER_DEPTH"],
        num_decoder_layers=hpo_parameters["TRANSFORMER_DECODER_DEPTH"],
        mlp_ratio=hpo_parameters["TRANSFORMER_MLP_RATIO"],
        heads=hpo_parameters["TRANSFORMER_HEADS"],
        dropout=hpo_parameters["DROPOUT"],
        emb_dropout=hpo_parameters["EMB_DROPOUT"],
        binaural_integration=BINAURAL_INTEGRATION,
        max_sources=MAX_SOURCES,
        num_classes_cls=num_classes,
    )

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda and GPU_LIST:
        net = nn.DataParallel(net, device_ids=GPU_LIST).to(device)
    else:
        net = net.to(device)

    print(f"Model built successfully. Using device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in net.parameters())}")
    return net, device


# %%
def load_criterion_with_hpo(hpo_parameters):
    criterion = SetCriterionBAST(
        loc_criterion=get_localization_criterion(LOSS_TYPE),
        num_classes=num_classes,
        loc_weight=LOC_WEIGHT,
        cls_weight=hpo_parameters["CLS_WEIGHT"],
        obj_weight=OBJ_WEIGHT,
        cls_cost_weight=hpo_parameters["CLS_COST_WEIGHT_HUNGARIAN"],
        loc_cost_weight=hpo_parameters["LOC_COST_WEIGHT_HUNGARIAN"],
        obj_cost_weight=hpo_parameters["OBJ_COST_WEIGHT_HUNGARIAN"],
        max_sources=MAX_SOURCES,
    )
    return criterion


# %%
def load_optimizer_with_hpo(net, hpo_hyperparameter):
    optimizer = torch.optim.AdamW(
        net.parameters(),
        lr=hpo_hyperparameter["LEARNING_RATE"],
        weight_decay=hpo_hyperparameter["WEIGHT_DECAY"],
    )
    return optimizer


# %%
def validate(net, epoch, criterion, device):
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
for _ in range(2):
    trials = HPO_client.get_next_trials(max_trials=1)
    torch.cuda.empty_cache()
    for trial_i, parameters in trials.items():
        # training set up
        best_val = float("inf")
        net, device = load_model_with_hpo_parameters(parameters)
        criterion = load_criterion_with_hpo(parameters)
        optimizer = load_optimizer_with_hpo(net, parameters)
        is_early_stopped = False
        for epoch in range(1, EPOCHS + 1):
            net.train()
            epoch_losses = {
                "total": 0.0,
                "loc": 0.0,
                "cls": 0.0,
                "obj": 0.0,
                "batches": 0,
            }
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
            val_metrics = validate(net, epoch, criterion, device)
            if val_metrics["total"] < best_val:
                best_val = val_metrics["total"]
                save_path = os.path.join(
                    MODEL_SAVE_DIR,
                    f"{MODEL_NAME}_{BINAURAL_INTEGRATION}_{LOSS_TYPE}_DET_{trial_i}.pt",
                )

                torch.save(net.state_dict(), save_path)
                print(f"  -> New best model saved to {save_path}")
            HPO_client.attach_data(
                trial_index=trial_i,
                raw_data=val_metrics,
                progression=epoch,
            )
            if HPO_client.should_stop_trial_early(trial_index=trial_i):
                is_early_stopped = True
                HPO_client.mark_trial_early_stopped(trial_index=trial_i)
                break
        if is_early_stopped:
            break
        if not is_early_stopped:
            HPO_client.complete_trial(
                trial_index=trial_i,
            )


# %%
best_parameters, prediction, index, name = HPO_client.get_best_parameterization()
print("Best Parameters:", best_parameters)
print("Prediction (mean, variance):", prediction)
# %%
# Save best parameters to disk
import json, os, datetime

save_dir = "artifacts"
os.makedirs(save_dir, exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
save_path = os.path.join(save_dir, f"best_hpo_params_{timestamp}.json")

payload = {
    "best_parameters": best_parameters,
    "prediction": prediction,
    "trial_index": int(index),
    "arm_name": str(name),
}

with open(save_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)

print(f"Saved best parameters to {save_path}")


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


def build_model_for_inference(num_classes: int) -> BAST_Variant:
    model = BAST_Variant(
        image_size=SPECTROGRAM_SIZE,
        patch_size=PATCH_SIZE,
        patch_overlap=PATCH_OVERLAP,
        num_coordinates_output=NUM_OUTPUT,
        dim=EMBEDDING_DIM,
        num_encoder_layers=TRANSFORMER_ENCODER_DEPTH,
        num_decoder_layers=TRANSFORMER_DECODER_DEPTH,
        heads=TRANSFORMER_HEADS,
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
    load_checkpoint_into_model(net, "", device)
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


# %%

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
