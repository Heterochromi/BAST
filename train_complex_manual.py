# %%
from network.BAST_CRV import BAST_CRV
from data_loading import MultiSourceSpectrogramDataset, multisource_collate
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from datetime import datetime
from utils import *
from criterion_bast import (
    SetCriterionBAST,
    get_localization_criterion,
    UncertaintyWeighter,
)
from torch.optim.lr_scheduler import CyclicLR


# %%
# Configuration (manual control over params)
CSV_PATH = "tensor_metadata_complex.csv"
SPECTROGRAM_SIZE = [129, 18]  # [Freq (n_mels), Time frames]
NUM_OUTPUT = 3  # e.g., (x, y, z)
EMBEDDING_DIM = 512
TRANSFORMER_ENCODER_DEPTH = 6
TRANSFORMER_DECODER_DEPTH = 3
TRANSFORMER_HEADS = 4
TRANSFORMER_MLP_RATIO = 2
DROPOUT = 0.05
EMB_DROPOUT = 0.05
PATCH_SIZE = 6
# Binaural integration
BINAURAL_INTEGRATION = "CROSS_ATTN"
MAX_SOURCES = 4
LOSS_TYPE = "MIX"

# Hungarian matching and loss weights
CLS_COST_WEIGHT_HUNGARIAN = 3
LOC_COST_WEIGHT_HUNGARIAN = 1
# LOC_WEIGHT = 0.1
# CLS_WEIGHT = 0.5

# Optimization / training
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 60
BATCH_SIZE = 400
TEST_SPLIT = 0.1
VAL_SPLIT = 0.3
SEED = 42
NUM_WORKERS = 4
CLS_THRESHOLD = 0.5

# Model saving / device
GPU_LIST = [0] if torch.cuda.is_available() else []
MODEL_SAVE_DIR = "./output/models/"
MODEL_NAME = "BASTCONV_MANUAL"
CHECKPOINT_PATH = None  # Set to path to resume training, e.g., "./output/models/checkpoint_epoch_30.pt"
START_EPOCH = 1  # Will be overridden if resuming from checkpoint
# %%
# Data setup
torch.manual_seed(SEED)
np.random.seed(SEED)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

dataset = MultiSourceSpectrogramDataset(
    CSV_PATH,
    tensor_dir="output_tensors_complex",
    preserve_complex=False,
    split_real_imag=True,
)
num_classes = dataset.num_classes
print(
    f"[{datetime.now()}] (MultiSource) Samples: {len(dataset)} | Classes: {num_classes}"
)

# %%
print(f"dataset shape {dataset[0][0].shape}")

# %%

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
# Verify input shapes - CRITICAL for debugging
print("\n" + "=" * 60)
print("INPUT SHAPE VERIFICATION")
print("=" * 60)
for specs, loc_lists, cls_lists, n_list in train_loader:
    print(f"✓ Batch specs shape: {specs.shape}")
    print(
        f"✓ Expected shape: [B={BATCH_SIZE}, channel=2, component=2, freq=129, time=18]"
    )
    print(f"✓ Single sample shape: {specs[0].shape}")
    print(f"✓ Expected single: [channel=2, component=2, freq=129, time=18]")
    print(f"✓ Specs dtype: {specs.dtype}")
    print(f"✓ Specs device: {specs.device}")

    # Verify dimensions match expected
    expected_shape = (BATCH_SIZE, 2, 2, 129, 18)
    if specs.shape[1:] != expected_shape[1:]:
        print(f"\n⚠️  WARNING: Shape mismatch!")
        print(f"   Got:      {specs.shape}")
        print(f"   Expected: {expected_shape}")
        raise ValueError("Input shape does not match model expectations!")
    else:
        print(f"✓ Shape verification PASSED")

    print(f"\n✓ Number of sources in batch: {n_list.tolist()[:5]}... (showing first 5)")
    break
print("=" * 60 + "\n")


# %%
# Build model, criterion, optimizer (manual params, no HPO)
def build_model_manual():
    net = BAST_CRV(
        image_size=SPECTROGRAM_SIZE,
        patch_size=PATCH_SIZE,
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
    cls_focal_alpha = None
    cls_focal_gamma = 2.0
    cls_pos_weight = 6

    # Task weighter (uncertainty weighting for loc/cls losses)
    weighter = UncertaintyWeighter(init_log_vars={"loc": 0.0, "cls": 0.0})

    criterion = SetCriterionBAST(
        loc_criterion=get_localization_criterion(LOSS_TYPE),
        num_classes=num_classes,
        # Hungarian matching weights
        cls_cost_weight=CLS_COST_WEIGHT_HUNGARIAN,
        loc_cost_weight=LOC_COST_WEIGHT_HUNGARIAN,
        max_sources=MAX_SOURCES,
        # Exposed hyperparameters for classification focal loss
        cls_focal_alpha=cls_focal_alpha,
        cls_focal_gamma=cls_focal_gamma,
        cls_pos_weight=cls_pos_weight,
        # Learned task weighting for final loss
        task_weighter=weighter,
    )
    return criterion


def build_optimizer_manual(net, criterion=None):
    # Include criterion params (e.g., UncertaintyWeighter) in optimizer if available
    params = list(net.parameters())
    if criterion is not None and isinstance(criterion, nn.Module):
        params += list(criterion.parameters())
    optimizer = torch.optim.AdamW(
        params,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    return optimizer


# %%
def validate(net, epoch, criterion, device):
    net.eval()
    running_loss = {"total": 0.0, "loc": 0.0, "cls": 0.0, "batches": 0}
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
            for k in ("total", "loc", "cls"):
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
    for k in ("total", "loc", "cls"):
        running_loss[k] /= max(running_loss["batches"], 1)
    if metric_accum["matched_pairs"] > 0:
        for mk in ("loc_err", "cls_exact", "cls_elem_acc"):
            metric_accum[mk] /= metric_accum["matched_pairs"]
    print(
        f"[VAL] Epoch {epoch} | Total {running_loss['total']:.4f} | Loc {running_loss['loc']:.4f} | Cls {running_loss['cls']:.4f} | loc_err {metric_accum['loc_err']:.3f} | ClsExact {metric_accum['cls_exact']:.3f} | ClsElem {metric_accum['cls_elem_acc']:.3f}"
    )
    return {**running_loss, **metric_accum}


# %%
# Training loop (manual) with checkpoint support
best_val = float("inf")
net, device = build_model_manual()
criterion = build_criterion_manual()
criterion = criterion.to(device)
# Move criterion parameters to device explicitly
for param in criterion.parameters():
    param.data = param.data.to(device)
optimizer = build_optimizer_manual(net, criterion)

# Load checkpoint if resuming
if CHECKPOINT_PATH is not None and os.path.exists(CHECKPOINT_PATH):
    print(f"\n{'=' * 60}")
    print(f"RESUMING FROM CHECKPOINT: {CHECKPOINT_PATH}")
    print(f"{'=' * 60}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

    # Check if this is an old checkpoint (just model weights) or new checkpoint (full state)
    is_old_checkpoint = (
        "model_state_dict" not in checkpoint
        and "optimizer_state_dict" not in checkpoint
    )

    if is_old_checkpoint:
        print("⚠️  Old checkpoint format detected (model weights only)")
        print("   Optimizer and scheduler will start from scratch")
        print("")
        print("   💡 IMPORTANT: You need to manually set START_EPOCH!")
        print(f"   Currently set to: {START_EPOCH}")
        print("   If you trained 30 epochs, set START_EPOCH = 31 in the config section")
        print("")

        # Old checkpoint: directly contains state_dict
        if isinstance(net, nn.DataParallel):
            net.module.load_state_dict(checkpoint)
        else:
            net.load_state_dict(checkpoint)

        print(f"✓ Model weights loaded")
        print(f"✓ Starting from epoch {START_EPOCH}")
        print(f"✓ Note: Training will continue but optimizer state is reset")
    else:
        # New checkpoint: contains full training state
        print("✓ New checkpoint format detected (full training state)")

        # Load model state
        if isinstance(net, nn.DataParallel):
            net.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            net.load_state_dict(checkpoint["model_state_dict"])

        # Load optimizer state
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print(f"✓ Optimizer state loaded")

        # Load criterion state (for uncertainty weighter)
        if "criterion_state_dict" in checkpoint:
            criterion.load_state_dict(checkpoint["criterion_state_dict"])
            print(f"✓ Criterion state loaded (uncertainty weights preserved)")

        # Load training state
        if "epoch" in checkpoint:
            START_EPOCH = checkpoint["epoch"] + 1
        if "best_val" in checkpoint:
            best_val = checkpoint["best_val"]

        print(f"✓ Resumed from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"✓ Best validation loss: {best_val:.4f}")
        print(f"✓ Continuing from epoch {START_EPOCH}")

    print(f"{'=' * 60}\n")
else:
    print(f"\n{'=' * 60}")
    print("STARTING FRESH TRAINING")
    print(f"{'=' * 60}\n")

# %%
batches_per_epoch = len(train_loader)

total_epochs = 60
num_cycles = 6

steps_per_cycle = (EPOCHS // num_cycles) * batches_per_epoch
step_size_up = steps_per_cycle // 2

scheduler = CyclicLR(
    optimizer,
    base_lr=1e-5,
    max_lr=1e-4,  # Lowered from 3e-4 for better stability with transformers
    step_size_up=step_size_up,
    mode="triangular2",
    cycle_momentum=False,
)

# If resuming from new checkpoint, advance scheduler to correct position
if CHECKPOINT_PATH is not None and os.path.exists(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    # Only advance scheduler if we have epoch information (new checkpoint format)
    if "epoch" in checkpoint:
        steps_taken = (checkpoint["epoch"]) * batches_per_epoch
        for _ in range(steps_taken):
            scheduler.step()
        print(f"✓ Scheduler advanced by {steps_taken} steps\n")
    else:
        print(f"⚠️  Scheduler starting from beginning (old checkpoint format)\n")
# %%
# Initialize tracking for debugging
prev_total_loss = None
print(f"\n{'=' * 60}")
print("TRAINING STARTED")
print(f"{'=' * 60}")
print(f"Initial LR: {optimizer.param_groups[0]['lr']:.2e}\n")

for epoch in range(START_EPOCH, EPOCHS + 1):
    net.train()
    epoch_losses = {
        "total": 0.0,
        "loc": 0.0,
        "cls": 0.0,
        "batches": 0,
    }
    metric_epoch = {
        "loc_err": 0.0,
        "cls_exact": 0.0,
        "cls_elem_acc": 0.0,
        "matched_pairs": 0,
        "batches": 0,
    }

    # Track gradient norms
    grad_norms = []

    for batch_idx, (specs, loc_lists, cls_lists, n_list) in enumerate(train_loader):
        specs = specs.to(device)
        outputs = net(specs)
        targets = build_target_list(loc_lists, cls_lists)
        loss_dict = criterion(outputs, targets)
        loss = loss_dict["total"]
        optimizer.zero_grad()
        loss.backward()

        # Calculate gradient norm before clipping
        total_norm = 0.0
        for p in net.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm**0.5
        grad_norms.append(total_norm)

        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Accumulate losses
        epoch_losses["total"] += loss_dict["total"].detach().item()
        epoch_losses["loc"] += float(loss_dict["loc"])
        epoch_losses["cls"] += float(loss_dict["cls"])

        epoch_losses["batches"] += 1

        # Detailed logging for first few batches of first 3 epochs
        if epoch <= 3 and batch_idx < 2:
            print(f"\n--- Epoch {epoch}, Batch {batch_idx} ---")
            print(f"  Total Loss: {loss_dict['total'].item():.4f}")
            print(f"  Loc Loss: {loss_dict['loc'].item():.4f}")
            print(f"  Cls Loss: {loss_dict['cls'].item():.4f}")
            print(f"  Grad Norm (before clip): {total_norm:.4f}")
            print(f"  Current LR: {optimizer.param_groups[0]['lr']:.2e}")
            print(
                f"  Num sources in batch: min={n_list.min().item()}, max={n_list.max().item()}, mean={n_list.float().mean().item():.2f}"
            )

            # Check outputs
            loc_out, cls_logit = outputs
            print(
                f"  Pred loc range: [{loc_out.min().item():.3f}, {loc_out.max().item():.3f}]"
            )
            print(
                f"  Pred cls logit range: [{cls_logit.min().item():.3f}, {cls_logit.max().item():.3f}]"
            )

            # Check first target
            if len(targets) > 0 and targets[0]["loc"].numel() > 0:
                print(
                    f"  GT loc range: [{targets[0]['loc'].min().item():.3f}, {targets[0]['loc'].max().item():.3f}]"
                )
                print(
                    f"  GT cls range: [{targets[0]['cls'].min().item():.3f}, {targets[0]['cls'].max().item():.3f}]"
                )

            # Check matching quality
            with torch.no_grad():
                for b_idx in range(min(2, len(targets))):
                    if targets[b_idx]["loc"].numel() > 0:
                        pred_idx, gt_idx = criterion._hungarian(
                            outputs[0][b_idx],
                            outputs[1][b_idx],
                            targets[b_idx]["loc"],
                            targets[b_idx]["cls"],
                        )
                        print(
                            f"  Sample {b_idx}: matched {len(pred_idx)} sources - Pred slots: {pred_idx.tolist()}, GT indices: {gt_idx.tolist()}"
                        )

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

    for k in ("total", "loc", "cls"):
        epoch_losses[k] /= max(epoch_losses["batches"], 1)
    if metric_epoch["matched_pairs"] > 0:
        for mk in ("loc_err", "cls_exact", "cls_elem_acc"):
            metric_epoch[mk] /= metric_epoch["matched_pairs"]

    # Calculate average gradient norm
    avg_grad_norm = sum(grad_norms) / len(grad_norms) if grad_norms else 0.0

    print(f"\n{'=' * 60}")
    print(f"EPOCH {epoch}/{EPOCHS} SUMMARY")
    print(f"{'=' * 60}")
    print(
        f"[TRAIN] Total {epoch_losses['total']:.4f} | Loc {epoch_losses['loc']:.4f} | Cls {epoch_losses['cls']:.4f} | loc_err {metric_epoch['loc_err']:.3f} | ClsExact {metric_epoch['cls_exact']:.3f} | ClsElem {metric_epoch['cls_elem_acc']:.3f}"
    )
    print(f"  Avg Grad Norm: {avg_grad_norm:.4f}")
    print(f"  Final LR this epoch: {optimizer.param_groups[0]['lr']:.2e}")

    # Check uncertainty weighter
    if hasattr(criterion, "task_weighter") and criterion.task_weighter is not None:
        log_var_loc = criterion.task_weighter.log_vars["loc"].item()
        log_var_cls = criterion.task_weighter.log_vars["cls"].item()
        weight_loc = torch.exp(-torch.tensor(log_var_loc)).item()
        weight_cls = torch.exp(-torch.tensor(log_var_cls)).item()
        effective_loc = weight_loc * epoch_losses["loc"]
        effective_cls = weight_cls * epoch_losses["cls"]

        print(f"  🎯 Task weights - Loc: {weight_loc:.4f}, Cls: {weight_cls:.4f}")
        if weight_cls < 0.3:  # Cls weight dropping too low
            print(
                "  🚨 WARNING: Cls weight dropping! Consider freezing uncertainty weighter"
            )
        print(
            f"  📊 Effective contribution - Loc: {effective_loc:.4f}, Cls: {effective_cls:.4f}"
        )
        print(f"  📈 Log vars - Loc: {log_var_loc:.4f}, Cls: {log_var_cls:.4f}")

    # Early warnings
    if epoch == START_EPOCH:
        if epoch_losses["total"] > 100:
            print(
                "\n⚠️  WARNING: Loss is very high! Check data normalization and targets."
            )
        if avg_grad_norm < 1e-4:
            print("\n⚠️  WARNING: Gradients are tiny! Model might not be learning.")
        if avg_grad_norm > 100:
            print("\n⚠️  WARNING: Gradients are huge! Might need stronger clipping.")

    # Track improvement
    if prev_total_loss is not None:
        improvement = (prev_total_loss - epoch_losses["total"]) / prev_total_loss * 100
        print(f"  📉 Improvement from prev epoch: {improvement:.2f}%")
        if abs(improvement) < 0.5 and epoch > 5:
            print("  ⚠️  Less than 0.5% improvement - learning might be stalling")

    # Validate and save best
    val_metrics = validate(net, epoch, criterion, device)

    # Save checkpoint every 10 epochs
    if epoch % 10 == 0:
        checkpoint_path = os.path.join(
            MODEL_SAVE_DIR,
            f"checkpoint_epoch_{epoch}.pt",
        )
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": net.module.state_dict()
                if isinstance(net, nn.DataParallel)
                else net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "criterion_state_dict": criterion.state_dict(),
                "best_val": best_val,
                "train_loss": epoch_losses["total"],
                "val_loss": val_metrics["total"],
            },
            checkpoint_path,
        )
        print(f"  💾 Checkpoint saved to {checkpoint_path}")

    if val_metrics["total"] < best_val:
        best_val = val_metrics["total"]
        save_path = os.path.join(
            MODEL_SAVE_DIR,
            f"{MODEL_NAME}_{BINAURAL_INTEGRATION}_{LOSS_TYPE}_best.pt",
        )
        torch.save(net.state_dict(), save_path)
        print(f"  ✨ New best model saved to {save_path} (val loss: {best_val:.4f})")

    print(f"{'=' * 60}\n")

    # Update previous loss for next iteration
    prev_total_loss = epoch_losses["total"]

# Final save
final_checkpoint_path = os.path.join(MODEL_SAVE_DIR, f"final_epoch_{EPOCHS}.pt")
torch.save(
    {
        "epoch": EPOCHS,
        "model_state_dict": net.module.state_dict()
        if isinstance(net, nn.DataParallel)
        else net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "criterion_state_dict": criterion.state_dict(),
        "best_val": best_val,
    },
    final_checkpoint_path,
)
print(f"\n{'=' * 60}")
print(f"TRAINING COMPLETE!")
print(f"Final checkpoint saved to {final_checkpoint_path}")
print(f"Best validation loss: {best_val:.4f}")
print(f"{'=' * 60}\n")

# %%
# Optional: Single-sample inference utilities (using BAST_CONV)
# Set DO_SINGLE_SAMPLE_TEST = True and adjust SINGLE_WAV_PATH to try it.
DO_SINGLE_SAMPLE_TEST = False
# OBJECTNESS_THRESHOLD removed (no objectness head)
TOP_K_CLASSES = 4

# (import moved inside run_single_wav_inference)


def build_model_for_inference(num_classes: int) -> BAST_CRV:
    model = BAST_CRV(
        image_size=SPECTROGRAM_SIZE,
        num_coordinates_output=NUM_OUTPUT,
        dim=EMBEDDING_DIM,
        patch_size=4,
        heads=TRANSFORMER_HEADS,
        num_encoder_layers=TRANSFORMER_ENCODER_DEPTH,
        num_decoder_layers=TRANSFORMER_DECODER_DEPTH,
        mlp_ratio=TRANSFORMER_MLP_RATIO,
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
        loc_out, cls_logit = inf_model(batch)
        loc_out = loc_out[0]
        cls_prob = torch.sigmoid(cls_logit[0])

        print("\n[Inference] Predictions:")
        print(f"{'Slot':<4} | {'x':>8} {'y':>8} {'z':>8} | Top classes (prob)")
        print("-" * 70)
        num_slots = loc_out.shape[0]
        for idx in range(num_slots):
            coords = loc_out[idx].tolist()
            coords = (coords + [0.0, 0.0, 0.0])[:3]  # pad for consistent printing
            x, y, z = coords
            cls_vec = cls_prob[idx]
            topk = torch.topk(cls_vec, k=min(top_k_classes, cls_vec.shape[0]))
            class_entries = []
            for c_idx, c_prob in zip(topk.indices.tolist(), topk.values.tolist()):
                cname = dataset.index_to_class.get(c_idx, f"id{c_idx}")
                class_entries.append(f"{cname}:{c_prob:.2f}")
            class_str = " ".join(class_entries)
            print(f"{idx:<4} | {x:8.2f} {y:8.2f} {z:8.2f} | {class_str}")
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
        top_k_classes=TOP_K_CLASSES,
    )
    print("======================================================\n")


# %%
def test(epoch):
    # Implemented similarly to validate(), but you can load a specific checkpoint first if needed.
    # load_checkpoint_into_model(net, "<PATH_TO_CKPT>.pt", device)
    net.eval()
    running_loss = {"total": 0.0, "loc": 0.0, "cls": 0.0, "batches": 0}
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
            for k in ("total", "loc", "cls"):
                running_loss[k] += float(losses[k])
            running_loss["batches"] += 1
            batch_metrics = compute_batch_metrics(outputs, targets, criterion, 0.3)
            mp = batch_metrics["matched_pairs"]
            if mp > 0:
                for mk in ("loc_err", "cls_exact", "cls_elem_acc"):
                    metric_accum[mk] += batch_metrics[mk] * mp
                metric_accum["matched_pairs"] += mp
            metric_accum["batches"] += 1
    for k in ("total", "loc", "cls"):
        running_loss[k] /= max(running_loss["batches"], 1)
    if metric_accum["matched_pairs"] > 0:
        for mk in ("loc_err", "cls_exact", "cls_elem_acc"):
            metric_accum[mk] /= metric_accum["matched_pairs"]
    print(
        f"[TEST] Epoch {epoch} | Total {running_loss['total']:.4f} | Loc {running_loss['loc']:.4f} | Cls {running_loss['cls']:.4f} | loc_err {metric_accum['loc_err']:.3f} | ClsExact {metric_accum['cls_exact']:.3f} | ClsElem {metric_accum['cls_elem_acc']:.3f}"
    )
    return {**running_loss, **metric_accum}
