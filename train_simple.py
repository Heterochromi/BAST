# %%
from network.BAST_CRV import BAST_CRV_Simple
from data_loading import MultiSourceSpectrogramDataset, multisource_collate
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from datetime import datetime
from utils import *
from torch.optim.lr_scheduler import CyclicLR
from raw_spec import generate_raw_spectrogram_torch_tensor


# %%
# Configuration for simple (non-DETR) model
CSV_PATH = "tensor_metadata_complex.csv"
SPECTROGRAM_SIZE = [129, 18]  # [Freq (n_mels), Time frames]
NUM_OUTPUT = 3  # e.g., (x, y, z)
EMBEDDING_DIM = 512
TRANSFORMER_ENCODER_DEPTH = 6
TRANSFORMER_HEADS = 4
TRANSFORMER_MLP_RATIO = 2
DROPOUT = 0.05
EMB_DROPOUT = 0.05
PATCH_SIZE = 6
BINAURAL_INTEGRATION = "CROSS_ATTN"

# Optimization / training
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 60
BATCH_SIZE = 200
TEST_SPLIT = 0.1
VAL_SPLIT = 0.3
SEED = 42
NUM_WORKERS = 4
CLS_THRESHOLD = 0.5

# Model saving / device
GPU_LIST = [0] if torch.cuda.is_available() else []
MODEL_SAVE_DIR = "./output/models_simple/"
MODEL_NAME = "BAST_CRV_SIMPLE"
CHECKPOINT_PATH = None  # Set to path to resume training
START_EPOCH = 1

# Loss weights
LOC_WEIGHT = 1.0
CLS_WEIGHT = 1.0

# %%
# Data setup
torch.manual_seed(SEED)
np.random.seed(SEED)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# Use filter_duplicate_classes=True to filter out samples with repeated classes
dataset = MultiSourceSpectrogramDataset(
    CSV_PATH,
    tensor_dir="output_tensors_complex",
    preserve_complex=False,
    split_real_imag=True,
    filter_duplicate_classes=True,  # NEW: Filter duplicate classes
)
num_classes = dataset.num_classes
print(
    f"[{datetime.now()}] (Simple Model) Samples: {len(dataset)} | Classes: {num_classes}"
)

# %%
print(f"dataset shape {dataset[0][0].shape}")
# %%
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
# Verify input shapes
print("\n" + "=" * 60)
print("INPUT SHAPE VERIFICATION")
print("=" * 60)
for specs, loc_lists, cls_lists, n_list in train_loader:
    print(f"✓ Batch specs shape: {specs.shape}")
    print(
        f"✓ Expected shape: [B={BATCH_SIZE}, channel=2, component=2, freq=129, time=18]"
    )
    print(f"✓ Number of samples: {len(loc_lists)}")
    print(f"✓ Sources per sample (first 5): {n_list[:5].tolist()}")
    print(f"✓ First sample loc shape: {loc_lists[0].shape}")
    print(f"✓ First sample cls shape: {cls_lists[0].shape}")
    print(f"✓ Specs dtype: {specs.dtype}")
    break
print("=" * 60 + "\n")


# %%
def build_model():
    net = BAST_CRV_Simple(
        image_size=SPECTROGRAM_SIZE,
        patch_size=PATCH_SIZE,
        num_coordinates_output=NUM_OUTPUT,
        dim=EMBEDDING_DIM,
        heads=TRANSFORMER_HEADS,
        num_encoder_layers=TRANSFORMER_ENCODER_DEPTH,
        mlp_ratio=TRANSFORMER_MLP_RATIO,
        dropout=DROPOUT,
        emb_dropout=EMB_DROPOUT,
        binaural_integration=BINAURAL_INTEGRATION,
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
def build_criterion():
    """Simple criterion without Hungarian matching"""
    loc_criterion = nn.MSELoss()
    cls_criterion = nn.BCEWithLogitsLoss()
    return loc_criterion, cls_criterion


# %%
def compute_loss(outputs, loc_lists, cls_lists, loc_criterion, cls_criterion):
    """
    Compute loss for multi-source model.
    outputs: (loc_out, cls_logit)
        loc_out: [B, num_classes, 3]
        cls_logit: [B, num_classes]
    loc_lists: tuple of [N_i, 3] tensors
    cls_lists: tuple of [N_i, num_classes] tensors
    """
    loc_out, cls_logit = outputs
    B = loc_out.size(0)

    total_loc_loss = 0.0
    total_cls_loss = 0.0
    num_valid_samples = 0

    for i in range(B):
        gt_locs = loc_lists[i]  # [N_i, 3]
        gt_cls = cls_lists[i]  # [N_i, num_classes]

        if gt_locs.size(0) == 0:
            continue

        # For each ground truth source, find best matching predicted class
        pred_locs = loc_out[i]  # [num_classes, 3]
        pred_cls = cls_logit[i]  # [num_classes]

        # Take first N sources (where N = number of GT sources)
        n_sources = min(gt_locs.size(0), pred_locs.size(0))

        # Simple matching: use first n_sources predictions for first n_sources GTs
        loc_loss = loc_criterion(
            pred_locs[:n_sources], gt_locs[:n_sources].to(pred_locs.device)
        )
        cls_loss = cls_criterion(
            pred_cls[:n_sources].unsqueeze(0),
            gt_cls[:n_sources].to(pred_cls.device).max(dim=1)[0].unsqueeze(0),
        )

        total_loc_loss += (
            loc_loss.item() if isinstance(loc_loss, torch.Tensor) else loc_loss
        )
        total_cls_loss += (
            cls_loss.item() if isinstance(cls_loss, torch.Tensor) else cls_loss
        )
        num_valid_samples += 1

    if num_valid_samples > 0:
        total_loc_loss /= num_valid_samples
        total_cls_loss /= num_valid_samples

    total_loss = LOC_WEIGHT * total_loc_loss + CLS_WEIGHT * total_cls_loss

    return {
        "total": torch.tensor(total_loss, requires_grad=True)
        if num_valid_samples > 0
        else torch.tensor(0.0, requires_grad=True),
        "loc": total_loc_loss,
        "cls": total_cls_loss,
    }


# %%
def compute_metrics(outputs, loc_lists, cls_lists, threshold=0.5):
    """
    Compute evaluation metrics for multi-source.
    """
    loc_out, cls_logit = outputs
    B = loc_out.size(0)

    total_loc_err = 0.0
    total_cls_exact = 0.0
    total_cls_elem_acc = 0.0
    matched_pairs = 0

    for i in range(B):
        gt_locs = loc_lists[i]  # [N_i, 3]
        gt_cls = cls_lists[i]  # [N_i, num_classes]

        if gt_locs.size(0) == 0:
            continue

        pred_locs = loc_out[i]  # [num_classes, 3]
        pred_cls = cls_logit[i]  # [num_classes]

        n_sources = min(gt_locs.size(0), pred_locs.size(0))

        # Localization error
        loc_err = (
            torch.norm(
                pred_locs[:n_sources] - gt_locs[:n_sources].to(pred_locs.device), dim=1
            )
            .mean()
            .item()
        )
        total_loc_err += loc_err * n_sources

        # Classification accuracy
        cls_pred = (torch.sigmoid(pred_cls[:n_sources]) > threshold).float()
        gt_cls_binary = (gt_cls[:n_sources].to(cls_pred.device).sum(dim=1) > 0).float()
        cls_exact = (
            (cls_pred == gt_cls_binary.unsqueeze(1)).all(dim=1).float().mean().item()
        )
        cls_elem_acc = (cls_pred == gt_cls_binary.unsqueeze(1)).float().mean().item()

        total_cls_exact += cls_exact * n_sources
        total_cls_elem_acc += cls_elem_acc * n_sources
        matched_pairs += n_sources

    if matched_pairs > 0:
        return {
            "loc_err": total_loc_err / matched_pairs,
            "cls_exact": total_cls_exact / matched_pairs,
            "cls_elem_acc": total_cls_elem_acc / matched_pairs,
            "matched_pairs": matched_pairs,
        }
    else:
        return {
            "loc_err": 0.0,
            "cls_exact": 0.0,
            "cls_elem_acc": 0.0,
            "matched_pairs": 0,
        }


# %%
def validate(net, epoch, loc_criterion, cls_criterion, device):
    net.eval()
    running_loss = {"total": 0.0, "loc": 0.0, "cls": 0.0, "batches": 0}
    running_metrics = {
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
            losses = compute_loss(
                outputs, loc_lists, cls_lists, loc_criterion, cls_criterion
            )

            for k in ("total", "loc", "cls"):
                running_loss[k] += float(losses[k])
            running_loss["batches"] += 1

            metrics = compute_metrics(outputs, loc_lists, cls_lists, CLS_THRESHOLD)
            mp = metrics["matched_pairs"]
            if mp > 0:
                for k in ("loc_err", "cls_exact", "cls_elem_acc"):
                    running_metrics[k] += metrics[k] * mp
                running_metrics["matched_pairs"] += mp
            running_metrics["batches"] += 1

    # Average losses and metrics
    for k in ("total", "loc", "cls"):
        running_loss[k] /= max(running_loss["batches"], 1)
    if running_metrics["matched_pairs"] > 0:
        for k in ("loc_err", "cls_exact", "cls_elem_acc"):
            running_metrics[k] /= running_metrics["matched_pairs"]

    print(
        f"[VAL] Epoch {epoch} | Total {running_loss['total']:.4f} | "
        f"Loc {running_loss['loc']:.4f} | Cls {running_loss['cls']:.4f} | "
        f"loc_err {running_metrics['loc_err']:.3f} | "
        f"ClsExact {running_metrics['cls_exact']:.3f} | "
        f"ClsElem {running_metrics['cls_elem_acc']:.3f}"
    )
    return {**running_loss, **running_metrics}


# %%
# Build model and criterion
best_val = float("inf")
net, device = build_model()
loc_criterion, cls_criterion = build_criterion()
optimizer = torch.optim.AdamW(
    net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)

# Load checkpoint if resuming
if CHECKPOINT_PATH is not None and os.path.exists(CHECKPOINT_PATH):
    print(f"\n{'=' * 60}")
    print(f"RESUMING FROM CHECKPOINT: {CHECKPOINT_PATH}")
    print(f"{'=' * 60}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

    if isinstance(net, nn.DataParallel):
        net.module.load_state_dict(checkpoint["model_state_dict"])
    else:
        net.load_state_dict(checkpoint["model_state_dict"])

    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"✓ Optimizer state loaded")

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
# Learning rate scheduler
batches_per_epoch = len(train_loader)
num_cycles = 6
steps_per_cycle = (EPOCHS // num_cycles) * batches_per_epoch
step_size_up = steps_per_cycle // 2

scheduler = CyclicLR(
    optimizer,
    base_lr=1e-5,
    max_lr=1e-4,
    step_size_up=step_size_up,
    mode="triangular2",
    cycle_momentum=False,
)

# Advance scheduler if resuming
if CHECKPOINT_PATH is not None and os.path.exists(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    if "epoch" in checkpoint:
        steps_taken = checkpoint["epoch"] * batches_per_epoch
        for _ in range(steps_taken):
            scheduler.step()
        print(f"✓ Scheduler advanced by {steps_taken} steps\n")

# %%
# Training loop
print(f"\n{'=' * 60}")
print("TRAINING STARTED (SIMPLE MODEL - NO DETR)")
print(f"{'=' * 60}")
print(f"Initial LR: {optimizer.param_groups[0]['lr']:.2e}\n")

prev_total_loss = None

for epoch in range(START_EPOCH, EPOCHS + 1):
    net.train()
    epoch_losses = {"total": 0.0, "loc": 0.0, "cls": 0.0, "batches": 0}
    epoch_metrics = {
        "loc_err": 0.0,
        "cls_exact": 0.0,
        "cls_elem_acc": 0.0,
        "matched_pairs": 0,
        "batches": 0,
    }
    grad_norms = []

    for batch_idx, (specs, loc_lists, cls_lists, n_list) in enumerate(train_loader):
        specs = specs.to(device)

        outputs = net(specs)
        loss_dict = compute_loss(
            outputs, loc_lists, cls_lists, loc_criterion, cls_criterion
        )
        loss = loss_dict["total"]

        # Ensure loss is a tensor before calling backward
        if not isinstance(loss, torch.Tensor):
            loss = torch.tensor(loss, requires_grad=True)

        optimizer.zero_grad()
        loss.backward()

        # Calculate gradient norm
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
        epoch_losses["total"] += float(
            loss_dict["total"].detach().item()
            if isinstance(loss_dict["total"], torch.Tensor)
            else loss_dict["total"]
        )
        epoch_losses["loc"] += float(loss_dict["loc"])
        epoch_losses["cls"] += float(loss_dict["cls"])
        epoch_losses["batches"] += 1

        # Detailed logging for first few batches
        if epoch <= 3 and batch_idx < 2:
            print(f"\n--- Epoch {epoch}, Batch {batch_idx} ---")
            total_loss_val = (
                loss_dict["total"].item()
                if isinstance(loss_dict["total"], torch.Tensor)
                else loss_dict["total"]
            )
            loc_loss_val = (
                loss_dict["loc"]
                if not isinstance(loss_dict["loc"], torch.Tensor)
                else loss_dict["loc"].item()
            )
            cls_loss_val = (
                loss_dict["cls"]
                if not isinstance(loss_dict["cls"], torch.Tensor)
                else loss_dict["cls"].item()
            )
            print(f"  Total Loss: {total_loss_val:.4f}")
            print(f"  Loc Loss: {loc_loss_val:.4f}")
            print(f"  Cls Loss: {cls_loss_val:.4f}")
            print(f"  Grad Norm: {total_norm:.4f}")
            print(f"  Current LR: {optimizer.param_groups[0]['lr']:.2e}")

            loc_out, cls_logit = outputs
            print(
                f"  Pred loc range: [{loc_out.min().item():.3f}, {loc_out.max().item():.3f}]"
            )
            print(
                f"  Pred cls range: [{cls_logit.min().item():.3f}, {cls_logit.max().item():.3f}]"
            )
            print(f"  Number of sources in batch: {n_list.sum().item()}")

        # Metrics
        metrics = compute_metrics(outputs, loc_lists, cls_lists, CLS_THRESHOLD)
        mp = metrics["matched_pairs"]
        if mp > 0:
            for k in ("loc_err", "cls_exact", "cls_elem_acc"):
                epoch_metrics[k] += metrics[k] * mp
            epoch_metrics["matched_pairs"] += mp
        epoch_metrics["batches"] += 1

    # Average losses and metrics
    for k in ("total", "loc", "cls"):
        epoch_losses[k] /= max(epoch_losses["batches"], 1)
    if epoch_metrics["matched_pairs"] > 0:
        for k in ("loc_err", "cls_exact", "cls_elem_acc"):
            epoch_metrics[k] /= epoch_metrics["matched_pairs"]

    avg_grad_norm = sum(grad_norms) / len(grad_norms) if grad_norms else 0.0

    print(f"\n{'=' * 60}")
    print(f"EPOCH {epoch}/{EPOCHS} SUMMARY")
    print(f"{'=' * 60}")
    print(
        f"[TRAIN] Total {epoch_losses['total']:.4f} | "
        f"Loc {epoch_losses['loc']:.4f} | Cls {epoch_losses['cls']:.4f} | "
        f"loc_err {epoch_metrics['loc_err']:.3f} | "
        f"ClsExact {epoch_metrics['cls_exact']:.3f} | "
        f"ClsElem {epoch_metrics['cls_elem_acc']:.3f}"
    )
    print(f"  Avg Grad Norm: {avg_grad_norm:.4f}")
    print(f"  Final LR this epoch: {optimizer.param_groups[0]['lr']:.2e}")

    # Track improvement
    if prev_total_loss is not None:
        improvement = (prev_total_loss - epoch_losses["total"]) / prev_total_loss * 100
        print(f"  📉 Improvement from prev epoch: {improvement:.2f}%")

    # Validate
    val_metrics = validate(net, epoch, loc_criterion, cls_criterion, device)

    # Save checkpoint every 10 epochs
    if epoch % 10 == 0:
        checkpoint_path = os.path.join(MODEL_SAVE_DIR, f"checkpoint_epoch_{epoch}.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": net.module.state_dict()
                if isinstance(net, nn.DataParallel)
                else net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val": best_val,
                "train_loss": epoch_losses["total"],
                "val_loss": val_metrics["total"],
            },
            checkpoint_path,
        )
        print(f"  💾 Checkpoint saved to {checkpoint_path}")

    # Save best model
    if val_metrics["total"] < best_val:
        best_val = val_metrics["total"]
        save_path = os.path.join(MODEL_SAVE_DIR, f"{MODEL_NAME}_best.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": net.module.state_dict()
                if isinstance(net, nn.DataParallel)
                else net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val": best_val,
            },
            save_path,
        )
        print(f"  ✨ New best model saved to {save_path} (val loss: {best_val:.4f})")

    print(f"{'=' * 60}\n")
    prev_total_loss = epoch_losses["total"]

# %%
# Final save
final_checkpoint_path = os.path.join(MODEL_SAVE_DIR, f"final_epoch_{EPOCHS}.pt")
torch.save(
    {
        "epoch": EPOCHS,
        "model_state_dict": net.module.state_dict()
        if isinstance(net, nn.DataParallel)
        else net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
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
def test():
    """Test the model on test set"""
    net.eval()
    running_loss = {"total": 0.0, "loc": 0.0, "cls": 0.0, "batches": 0}
    running_metrics = {
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
            losses = compute_loss(
                outputs, loc_lists, cls_lists, loc_criterion, cls_criterion
            )

            for k in ("total", "loc", "cls"):
                running_loss[k] += float(losses[k])
            running_loss["batches"] += 1

            metrics = compute_metrics(outputs, loc_lists, cls_lists, CLS_THRESHOLD)
            mp = metrics["matched_pairs"]
            if mp > 0:
                for k in ("loc_err", "cls_exact", "cls_elem_acc"):
                    running_metrics[k] += metrics[k] * mp
                running_metrics["matched_pairs"] += mp
            running_metrics["batches"] += 1

    # Average
    for k in ("total", "loc", "cls"):
        running_loss[k] /= max(running_loss["batches"], 1)
    if running_metrics["matched_pairs"] > 0:
        for k in ("loc_err", "cls_exact", "cls_elem_acc"):
            running_metrics[k] /= running_metrics["matched_pairs"]

    print(f"\n{'=' * 60}")
    print("TEST RESULTS")
    print(f"{'=' * 60}")
    print(
        f"[TEST] Total {running_loss['total']:.4f} | "
        f"Loc {running_loss['loc']:.4f} | Cls {running_loss['cls']:.4f} | "
        f"loc_err {running_metrics['loc_err']:.3f} | "
        f"ClsExact {running_metrics['cls_exact']:.3f} | "
        f"ClsElem {running_metrics['cls_elem_acc']:.3f}"
    )
    print(f"{'=' * 60}\n")
    return {**running_loss, **running_metrics}


# %%
# Uncomment to run test after training
# test_results = test()


# %%
# ============================================================
# CHECKPOINT LOADING AND SINGLE SAMPLE TESTING
# ============================================================


def load_checkpoint_for_inference(checkpoint_path, device=None):
    """
    Load a trained model from checkpoint for inference.

    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load model on (defaults to cuda if available)

    Returns:
        model: Loaded model in eval mode
        device: Device the model is on
        checkpoint_info: Dictionary with checkpoint metadata
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'=' * 60}")
    print(f"LOADING CHECKPOINT FOR INFERENCE")
    print(f"{'=' * 60}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")

    # Build model
    model = BAST_CRV_Simple(
        image_size=SPECTROGRAM_SIZE,
        patch_size=PATCH_SIZE,
        num_coordinates_output=NUM_OUTPUT,
        dim=EMBEDDING_DIM,
        heads=TRANSFORMER_HEADS,
        num_encoder_layers=TRANSFORMER_ENCODER_DEPTH,
        mlp_ratio=TRANSFORMER_MLP_RATIO,
        dropout=DROPOUT,
        emb_dropout=EMB_DROPOUT,
        binaural_integration=BINAURAL_INTEGRATION,
        num_classes_cls=8,
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Extract checkpoint info
    checkpoint_info = {
        "epoch": checkpoint.get("epoch", "unknown"),
        "best_val": checkpoint.get("best_val", "unknown"),
        "train_loss": checkpoint.get("train_loss", "unknown"),
        "val_loss": checkpoint.get("val_loss", "unknown"),
    }

    print(f"✓ Model loaded successfully")
    print(f"  Epoch: {checkpoint_info['epoch']}")
    print(f"  Best Val Loss: {checkpoint_info['best_val']}")
    print(f"{'=' * 60}\n")

    return model, device, checkpoint_info


def test_single_sample(model, sample_idx, dataset, device, cls_threshold=0.5):
    """
    Test the model on a single sample from the dataset.

    Args:
        model: Trained model
        sample_idx: Index of sample in dataset
        dataset: Dataset to get sample from
        device: Device to run inference on
        cls_threshold: Threshold for classification predictions

    Returns:
        Dictionary with predictions and ground truth
    """
    model.eval()

    # Get sample
    spec, loc_target, cls_target, n_sources = dataset[sample_idx]

    # Prepare input
    spec_input = spec.unsqueeze(0).to(device)  # Add batch dimension

    # Get prediction
    with torch.no_grad():
        loc_out, cls_logit = model(spec_input)
        loc_out = loc_out.squeeze(1).squeeze(0)  # Remove batch and seq dims -> [3]
        cls_logit = cls_logit.squeeze(1).squeeze(0)  # -> [num_classes]
        cls_prob = torch.sigmoid(cls_logit)
        cls_pred = (cls_prob > cls_threshold).float()

    # Move to CPU for display
    loc_out = loc_out.cpu().numpy()
    cls_prob = cls_prob.cpu().numpy()
    cls_pred = cls_pred.cpu().numpy()
    loc_target = (
        loc_target.numpy() if isinstance(loc_target, torch.Tensor) else loc_target
    )
    cls_target = (
        cls_target.numpy() if isinstance(cls_target, torch.Tensor) else cls_target
    )

    # Compute metrics
    loc_error = np.linalg.norm(loc_out - loc_target)
    cls_exact_match = np.all(cls_pred == cls_target)
    cls_element_acc = np.mean(cls_pred == cls_target)

    # Get predicted class names
    predicted_classes = np.where(cls_pred > 0)[0]
    target_classes = np.where(cls_target > 0)[0]

    # Print results
    print(f"\n{'=' * 60}")
    print(f"SINGLE SAMPLE TEST - Sample Index: {sample_idx}")
    print(f"{'=' * 60}")
    print(f"\n📍 LOCALIZATION:")
    print(f"  Predicted: [{loc_out[0]:6.3f}, {loc_out[1]:6.3f}, {loc_out[2]:6.3f}]")
    print(
        f"  Ground Truth: [{loc_target[0]:6.3f}, {loc_target[1]:6.3f}, {loc_target[2]:6.3f}]"
    )
    print(f"  L2 Error: {loc_error:.4f}")

    print(f"\n🏷️  CLASSIFICATION:")
    print(f"  Predicted Classes: {predicted_classes.tolist()}")
    print(f"  Ground Truth Classes: {target_classes.tolist()}")
    print(f"  Exact Match: {'✓' if cls_exact_match else '✗'}")
    print(f"  Element Accuracy: {cls_element_acc:.3f}")

    print(f"\n📊 TOP 5 CLASS PROBABILITIES:")
    top5_indices = np.argsort(cls_prob)[-5:][::-1]
    for idx in top5_indices:
        marker = "✓" if cls_target[idx] > 0 else " "
        print(f"  {marker} Class {idx:3d}: {cls_prob[idx]:.4f}")

    print(f"{'=' * 60}\n")

    return {
        "sample_idx": sample_idx,
        "loc_pred": loc_out,
        "loc_target": loc_target,
        "loc_error": loc_error,
        "cls_pred": cls_pred,
        "cls_target": cls_target,
        "cls_prob": cls_prob,
        "cls_exact_match": cls_exact_match,
        "cls_element_acc": cls_element_acc,
    }


def preprocess_audio_for_inference(audio_path, max_duration=0.1):
    """
    Load and preprocess a raw audio file for inference.

    Args:
        audio_path: Path to the audio file
        max_duration: Maximum duration in seconds (default: 0.1s = 100ms)

    Returns:
        Preprocessed spectrogram tensor ready for model input [2, freq, time, 2]
        Format: [channels, freq, time, real/imag]
    """
    print(f"\n{'=' * 60}")
    print(f"PREPROCESSING AUDIO FILE")
    print(f"{'=' * 60}")
    print(f"File: {audio_path}")

    # Generate complex spectrogram using raw_spec.py
    complex_spec = generate_raw_spectrogram_torch_tensor(
        audio_path, return_tensor=True, max_duration=max_duration
    )

    print(f"Raw spectrogram shape: {complex_spec.shape}")
    print(f"Expected: [2, freq_bins, time_frames]")

    # Split real and imaginary parts to match data_loading.py format
    # complex_spec shape: [2, freq, time] (stereo complex)
    # We need: [2, freq, time, 2] (channels, freq, time, real/imag)

    # Stack real and imaginary as last dimension for each channel
    preprocessed = torch.stack([complex_spec.real, complex_spec.imag], dim=-1)

    print(f"Preprocessed shape: {preprocessed.shape}")
    print(f"Expected model input: [2, {SPECTROGRAM_SIZE[0]}, {SPECTROGRAM_SIZE[1]}, 2]")

    # Resize if needed to match expected dimensions
    current_shape = preprocessed.shape[1:3]  # [freq, time]
    if current_shape != tuple(SPECTROGRAM_SIZE):
        print(f"⚠️  Shape mismatch! Resizing from {current_shape} to {SPECTROGRAM_SIZE}")
        # Use interpolation to resize
        import torch.nn.functional as F

        # Reshape to [2*2, freq, time] for interpolation
        batch_size = preprocessed.shape[0] * preprocessed.shape[3]
        preprocessed = preprocessed.permute(0, 3, 1, 2).reshape(
            batch_size, *current_shape
        )

        # Resize
        preprocessed = F.interpolate(
            preprocessed,
            size=SPECTROGRAM_SIZE,
            mode="bilinear",
            align_corners=False,
        )

        # Reshape back to [2, freq, time, 2]
        preprocessed = preprocessed.reshape(
            2, 2, SPECTROGRAM_SIZE[0], SPECTROGRAM_SIZE[1]
        )
        preprocessed = preprocessed.permute(0, 2, 3, 1)
        print(f"✓ Resized to: {preprocessed.shape}")

    print(f"{'=' * 60}\n")
    return preprocessed


def test_audio_file(model, audio_path, device, cls_threshold=0.5, class_names=None):
    """
    Test the model on a single audio file loaded from disk.

    Args:
        model: Trained model
        audio_path: Path to the audio file
        device: Device to run inference on
        cls_threshold: Threshold for classification predictions
        class_names: Optional list of class names for better output

    Returns:
        Dictionary with predictions
    """
    model.eval()

    # Preprocess the audio file
    spec = preprocess_audio_for_inference(audio_path)

    # Prepare input and permute to match model expected format
    # Dataset format: [channel, freq, time, component] -> [2, H, W, 2]
    # Model expects: [batch, channel, component, freq, time] -> [B, 2, 2, H, W]
    spec_input = (
        spec.permute(0, 3, 1, 2).unsqueeze(0).to(device)
    )  # [2, H, W, 2] -> [2, 2, H, W] -> [1, 2, 2, H, W]

    # Get prediction
    with torch.no_grad():
        loc_out, cls_logit = model(spec_input)
        loc_out = loc_out.squeeze(1).squeeze(0)  # Remove batch and seq dims -> [3]
        cls_logit = cls_logit.squeeze(1).squeeze(0)  # -> [num_classes]
        cls_prob = torch.sigmoid(cls_logit)
        cls_pred = (cls_prob > cls_threshold).float()

    # Move to CPU for display
    loc_out = loc_out.cpu().numpy()
    cls_prob = cls_prob.cpu().numpy()
    cls_pred = cls_pred.cpu().numpy()

    # Get predicted class indices
    predicted_classes = np.where(cls_pred > 0)[0]

    # Print results
    print(f"\n{'=' * 60}")
    print(f"AUDIO FILE INFERENCE RESULTS")
    print(f"{'=' * 60}")
    print(f"File: {audio_path}")

    print(f"\n📍 PREDICTED LOCALIZATION:")
    print(f"  Azimuth (x):   {loc_out[0]:7.3f}")
    print(f"  Elevation (y): {loc_out[1]:7.3f}")
    print(f"  Distance (z):  {loc_out[2]:7.3f}")

    print(f"\n🏷️  PREDICTED CLASSES (threshold={cls_threshold}):")
    if len(predicted_classes) > 0:
        for idx in predicted_classes:
            class_name = (
                class_names[idx]
                if class_names and idx < len(class_names)
                else f"Class {idx}"
            )
            print(f"  ✓ {class_name}: {cls_prob[idx]:.4f}")
    else:
        print(f"  (No classes above threshold)")

    print(f"\n📊 TOP 10 CLASS PROBABILITIES:")
    top10_indices = np.argsort(cls_prob)[-10:][::-1]
    for idx in top10_indices:
        class_name = (
            class_names[idx]
            if class_names and idx < len(class_names)
            else f"Class {idx}"
        )
        bar_length = int(cls_prob[idx] * 30)
        bar = "█" * bar_length + "░" * (30 - bar_length)
        print(f"  {class_name:20s} {bar} {cls_prob[idx]:.4f}")

    print(f"{'=' * 60}\n")

    return {
        "audio_path": audio_path,
        "loc_pred": loc_out,
        "cls_pred": cls_pred,
        "cls_prob": cls_prob,
        "predicted_classes": predicted_classes,
    }


# %%
# EXAMPLE USAGE:
# Uncomment the code below to load a checkpoint and test on audio files

# 1. Load checkpoint
INFERENCE_CHECKPOINT = "./output/models_simple/BAST_CRV_SIMPLE_best.pt"
model_loaded, device_loaded, info = load_checkpoint_for_inference(INFERENCE_CHECKPOINT)
# %%
# 2. Test on a raw audio file
TEST_AUDIO_PATH = "dataset_parallel_100ms/sample_0033.wav"
result = test_audio_file(
    model_loaded, TEST_AUDIO_PATH, device_loaded, cls_threshold=0.5
)

# 3. Test on multiple audio files
# audio_files = [
#     "dataset_parallel_100ms/sample_0025.wav",
#     "dataset_parallel_100ms/sample_0100.wav",
#     "dataset_parallel_100ms/sample_0200.wav",
# ]
# for audio_path in audio_files:
#     result = test_audio_file(model_loaded, audio_path, device_loaded)

# %%
# ALTERNATIVE: Test on samples from the dataset
# Uncomment to use dataset samples instead of raw audio files

# # Load checkpoint
# INFERENCE_CHECKPOINT = "./output/models_simple/BAST_CRV_SIMPLE_best.pt"
# model_loaded, device_loaded, info = load_checkpoint_for_inference(INFERENCE_CHECKPOINT)

# # Test on a single sample from test set
# test_sample_idx = 0  # Index within the test_ds
# original_idx = test_ds.indices[test_sample_idx]
# result = test_single_sample(model_loaded, original_idx, dataset, device_loaded)

# # Test on multiple samples
# for i in range(5):
#     original_idx = test_ds.indices[i]
#     result = test_single_sample(model_loaded, original_idx, dataset, device_loaded)

# %%
