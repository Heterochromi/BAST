# %%
from network.BASTCONV import BAST_CONV
from data_loading import MultiSourceSpectrogramDataset, multisource_collate
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from datetime import datetime
from utils import *
from criterion_bast import SetCriterionBAST, get_localization_criterion
from torch.optim.lr_scheduler import CyclicLR


# %%
# Configuration (manual control over params)
CSV_PATH = "tensor_metadata.csv"
SPECTROGRAM_SIZE = [64, 19]  # [Freq (n_mels), Time frames]
NUM_OUTPUT = 3  # e.g., (x, y, z)
EMBEDDING_DIM = 256
TRANSFORMER_ENCODER_DEPTH = 4
TRANSFORMER_DECODER_DEPTH = 2
TRANSFORMER_HEADS = 8
TRANSFORMER_MLP_RATIO = 2
DROPOUT = 0.1
EMB_DROPOUT = 0.1

# Binaural integration and detection
BINAURAL_INTEGRATION = "CROSS_ATTN"
MAX_SOURCES = 4
LOSS_TYPE = "MIX"

# Hungarian matching and loss weights
CLS_COST_WEIGHT_HUNGARIAN = 0.5
LOC_COST_WEIGHT_HUNGARIAN = 0.5
OBJ_COST_WEIGHT_HUNGARIAN = 0.5
LOC_WEIGHT = 0.4
CLS_WEIGHT = 6.0
OBJ_WEIGHT = 1.0

# Optimization / training
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 60
BATCH_SIZE = 128
TEST_SPLIT = 0.3
VAL_SPLIT = 0.3
SEED = 42
NUM_WORKERS = 4
CLS_THRESHOLD = 0.5

# Model saving / device
GPU_LIST = [0] if torch.cuda.is_available() else []
MODEL_SAVE_DIR = "./output/models/"
MODEL_NAME = "BASTCONV_MANUAL"

# Tokenizer / conv front-end for BAST_CONV
N_CONV_INPUT_CHANNELS = 1  # BAST_CONV internally splits stereo to two 1-ch streams
TOKEN_KERNEL_SIZE = 3
TOKEN_STRIDE = 1
TOKEN_PADDING = 1
POOL_KERNEL_SIZE = 3
POOL_STRIDE = 2
POOL_PADDING = 1
N_CONV_LAYERS = 1

# %%
# Data setup
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
# Dataloaders
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
# Build model, criterion, optimizer (manual params, no HPO)
def build_model_manual():
    net = BAST_CONV(
        image_size=SPECTROGRAM_SIZE,
        num_coordinates_output=NUM_OUTPUT,
        dim=EMBEDDING_DIM,
        heads=TRANSFORMER_HEADS,
        num_encoder_layers=TRANSFORMER_ENCODER_DEPTH,
        num_decoder_layers=TRANSFORMER_DECODER_DEPTH,
        mlp_ratio=TRANSFORMER_MLP_RATIO,
        dropout=DROPOUT,
        emb_dropout=EMB_DROPOUT,
        binaural_integration=BINAURAL_INTEGRATION,
        max_sources=MAX_SOURCES,
        num_classes_cls=num_classes,
        n_conv_input_channels=N_CONV_INPUT_CHANNELS,
        kernel_size=TOKEN_KERNEL_SIZE,
        stride=TOKEN_STRIDE,
        padding=TOKEN_PADDING,
        pooling_kernel_size=POOL_KERNEL_SIZE,
        pooling_stride=POOL_STRIDE,
        pooling_padding=POOL_PADDING,
        n_conv_layers=N_CONV_LAYERS,
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


def build_criterion_manual():
    criterion = SetCriterionBAST(
        loc_criterion=get_localization_criterion(LOSS_TYPE),
        num_classes=num_classes,
        loc_weight=LOC_WEIGHT,
        cls_weight=CLS_WEIGHT,
        obj_weight=OBJ_WEIGHT,
        cls_cost_weight=CLS_COST_WEIGHT_HUNGARIAN,
        loc_cost_weight=LOC_COST_WEIGHT_HUNGARIAN,
        obj_cost_weight=OBJ_COST_WEIGHT_HUNGARIAN,
        max_sources=MAX_SOURCES,
    )
    return criterion


def build_optimizer_manual(net):
    optimizer = torch.optim.AdamW(
        net.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
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
        f"[VAL] Epoch {epoch} | Total {running_loss['total']:.4f} | Loc {running_loss['loc']:.4f} | Cls {running_loss['cls']:.4f} | Obj {running_loss['obj']:.4f} | loc_err {metric_accum['loc_err']:.3f} | ClsExact {metric_accum['cls_exact']:.3f} | ClsElem {metric_accum['cls_elem_acc']:.3f}"
    )
    return {**running_loss, **metric_accum}


# %%
# Training loop (manual)
best_val = float("inf")
net, device = build_model_manual()
criterion = build_criterion_manual()
optimizer = build_optimizer_manual(net)

# %%
batches_per_epoch = len(train_loader)

total_epochs = 60
num_cycles = 6

steps_per_cycle = EPOCHS * batches_per_epoch
step_size_up = steps_per_cycle // 2

scheduler = CyclicLR(
    optimizer,
    base_lr=1e-5,
    max_lr=1e-2,
    step_size_up=step_size_up,
    # mode="triangular2",
    cycle_momentum=False,
)
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
        scheduler.step()

        # Accumulate losses
        epoch_losses["total"] += float(loss_dict["total"])
        epoch_losses["loc"] += float(loss_dict["loc"])
        epoch_losses["cls"] += float(loss_dict["cls"])
        epoch_losses["obj"] += float(loss_dict["obj"])
        epoch_losses["batches"] += 1

        # Metrics
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
        f"[TRAIN] Epoch {epoch} | Total {epoch_losses['total']:.4f} | Loc {epoch_losses['loc']:.4f} | Cls {epoch_losses['cls']:.4f} | Obj {epoch_losses['obj']:.4f} | loc_err {metric_epoch['loc_err']:.3f} | ClsExact {metric_epoch['cls_exact']:.3f} | ClsElem {metric_epoch['cls_elem_acc']:.3f} |"
    )

    # Validate and save best
    val_metrics = validate(net, epoch, criterion, device)
    if val_metrics["total"] < best_val:
        best_val = val_metrics["total"]
        save_path = os.path.join(
            MODEL_SAVE_DIR,
            f"{MODEL_NAME}_{BINAURAL_INTEGRATION}_{LOSS_TYPE}_best.pt",
        )
        torch.save(net.state_dict(), save_path)
        print(f"  -> New best model saved to {save_path}")

# %%
# Optional: Single-sample inference utilities (using BAST_CONV)
# Set DO_SINGLE_SAMPLE_TEST = True and adjust SINGLE_WAV_PATH to try it.
DO_SINGLE_SAMPLE_TEST = False
OBJECTNESS_THRESHOLD = 0.4
TOP_K_CLASSES = 4

# (import moved inside run_single_wav_inference)


def build_model_for_inference(num_classes: int) -> BAST_CONV:
    model = BAST_CONV(
        image_size=SPECTROGRAM_SIZE,
        num_coordinates_output=NUM_OUTPUT,
        dim=EMBEDDING_DIM,
        heads=TRANSFORMER_HEADS,
        num_encoder_layers=TRANSFORMER_ENCODER_DEPTH,
        num_decoder_layers=TRANSFORMER_DECODER_DEPTH,
        mlp_ratio=TRANSFORMER_MLP_RATIO,
        dropout=DROPOUT,
        emb_dropout=EMB_DROPOUT,
        binaural_integration=BINAURAL_INTEGRATION,
        max_sources=MAX_SOURCES,
        num_classes_cls=num_classes,
        n_conv_input_channels=N_CONV_INPUT_CHANNELS,
        kernel_size=TOKEN_KERNEL_SIZE,
        stride=TOKEN_STRIDE,
        padding=TOKEN_PADDING,
        pooling_kernel_size=POOL_KERNEL_SIZE,
        pooling_stride=POOL_STRIDE,
        pooling_padding=POOL_PADDING,
        n_conv_layers=N_CONV_LAYERS,
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
    if mel_tensor.ndim != 3 or mel_tensor.shape[0] != 2:
        raise ValueError(
            f"Expected raw mel tensor shape [2, F, T], got {tuple(mel_tensor.shape)}"
        )
    target_f, target_t = SPECTROGRAM_SIZE
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
    use_cuda = torch.cuda.is_available()
    dev = torch.device("cuda" if use_cuda else "cpu")
    inf_model = build_model_for_inference(num_classes=dataset.num_classes).to(dev)
    inf_model.eval()

    if checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint(MODEL_SAVE_DIR, "*.pt")
        if checkpoint_path is None:
            print("[Inference] No checkpoint found in model directory.")
            return

    load_checkpoint_into_model(inf_model, checkpoint_path, dev)

    print(f"[Inference] Generating mel spectrogram for: {wav_path}")
    from mel_spec_tensor import generate_mel_spectrogram_torch_tensor

    with torch.no_grad():
        mel_tensor = generate_mel_spectrogram_torch_tensor(
            wav_path, output_path=None, return_tensor=True
        )
        mel_tensor = prepare_spectrogram_tensor(mel_tensor)  # shape [2, 64, T]
        batch = mel_tensor.unsqueeze(0).to(dev)  # [1, 2, 64, T]
        loc_out, obj_logit, cls_logit = inf_model(batch)
        loc_out = loc_out[0]
        obj_prob = torch.sigmoid(obj_logit[0])
        cls_prob = torch.sigmoid(cls_logit[0])

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
        "dataset_parallel_100ms/sample_0057.wav"  # Replace with your test WAV path
    )
    print("\n================ Single WAV Inference ================")
    EXPLICIT_WEIGHTS_PATH = None
    run_single_wav_inference(
        wav_path=SINGLE_WAV_PATH,
        checkpoint_path=EXPLICIT_WEIGHTS_PATH,
        objectness_threshold=OBJECTNESS_THRESHOLD,
        top_k_classes=TOP_K_CLASSES,
    )
    print("======================================================\n")


# %%
def test(epoch):
    # Implemented similarly to validate(), but you can load a specific checkpoint first if needed.
    # load_checkpoint_into_model(net, "<PATH_TO_CKPT>.pt", device)
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
        for specs, loc_lists, cls_lists, n_list in test_loader:
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
        f"[TEST] Epoch {epoch} | Total {running_loss['total']:.4f} | Loc {running_loss['loc']:.4f} | Cls {running_loss['cls']:.4f} | Obj {running_loss['obj']:.4f} | loc_err {metric_accum['loc_err']:.3f} | ClsExact {metric_accum['cls_exact']:.3f} | ClsElem {metric_accum['cls_elem_acc']:.3f}"
    )
    return {**running_loss, **metric_accum}
