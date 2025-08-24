from network.BAST import BAST_Variant, AngularLossWithCartesianCoordinate, MixWithCartesianCoordinate
from data_loading import SpectrogramDataset
from conf import *
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from datetime import datetime


def build_args():
    parser = argparse.ArgumentParser(description="Multitask training for BAST: class + azimuth + elevation")
    parser.add_argument('--csv', type=str, required=True, help='Path to dataset CSV with spectrogram file paths')
    parser.add_argument('--backbone', type=str, default='vanilla', choices=['vanilla'], help='Transformer variant')
    parser.add_argument('--integ', type=str, default='SUB', choices=['SUB', 'ADD', 'CONCAT'], help='Binaural integration method')
    parser.add_argument('--shareweights', action='store_true', help='Share weights between left/right branches')
    parser.add_argument('--loss', type=str, default='MIX', choices=['MSE', 'AD', 'MIX'], help='Localization loss')
    parser.add_argument('--epochs', type=int, default=EPOCH)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--lr', type=float, default=LR)
    parser.add_argument('--val_split', type=float, default=0.2)
    parser.add_argument('--cls_weight', type=float, default=1.0)
    parser.add_argument('--elev_weight', type=float, default=0.1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def get_localization_criterion(name: str):
    if name == 'MSE':
        return nn.MSELoss()
    if name == 'AD':
        return AngularLossWithCartesianCoordinate()
    if name == 'MIX':
        return MixWithCartesianCoordinate()
    raise ValueError('Unknown localization loss')


def main():
    args = build_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(MODEL_SAVE, exist_ok=True)

    print(f"[{datetime.now()}] Loading dataset from {args.csv} ...")
    dataset = SpectrogramDataset(args.csv)
    num_classes = len(dataset.class_to_index)
    print(f"[{datetime.now()}] Samples: {len(dataset)} | Classes: {num_classes}")

    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    print(f"[{datetime.now()}] Building model ...")
    net = BAST_Variant(
        image_size=SPECTROGRAM_SIZE,
        patch_size=PATCH_SIZE,
        patch_overlap=PATCH_OVERLAP,
        num_classes=NUM_OUTPUT,
        dim=EMBEDDING_DIM,
        depth=TRANSFORMER_DEPTH,
        heads=TRANSFORMER_HEADS,
        mlp_dim=TRANSFORMER_MLP_DIM,
        pool=TRANSFORMER_POOL,
        channels=INPUT_CHANNEL,
        dim_head=TRANSFORMER_DIM_HEAD,
        dropout=DROPOUT,
        emb_dropout=EMB_DROPOUT,
        binaural_integration=args.integ,
        share_params=args.shareweights,
        transformer_variant=args.backbone,
        classify_sound=True,
        num_classes_cls=num_classes,
        regress_elevation=True,
    )

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    if use_cuda and GPU_LIST:
        net = nn.DataParallel(net, device_ids=GPU_LIST).to(device)
    else:
        net = net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=0.0)
    criterion_loc = get_localization_criterion(args.loss)
    if num_classes > 1:
        criterion_cls = nn.CrossEntropyLoss()
    else:
        criterion_cls = nn.BCEWithLogitsLoss()
    criterion_elev = nn.MSELoss()

    model_save_name = f"{MODEL_NAME}_{args.integ}_{args.loss}_MT_{'SP' if args.shareweights else 'NSP'}_{args.backbone}"

    print(f"[{datetime.now()}] Start training ...")
    best_val = float('inf')
    train_losses = []
    val_losses = []
    for epoch in range(args.epochs):
        net.train()
        running = 0.0
        n_train = 0
        for batch in train_loader:
            specs, loc_xy, cls_idx, az_el_deg = batch
            specs = specs.to(device, non_blocking=True)
            loc_xy = loc_xy.to(device, non_blocking=True)
            cls_idx = cls_idx.to(device, non_blocking=True).squeeze(1)
            az_el_deg = az_el_deg.to(device, non_blocking=True)

            outputs = net(specs)
            # outputs: (loc_out, cls_out, elev_out)
            loc_out = outputs[0]
            cls_out = outputs[1]
            elev_out = outputs[2]

            loss_loc = criterion_loc(loc_out, loc_xy)
            if num_classes > 1:
                loss_cls = criterion_cls(cls_out, cls_idx)
            else:
                # BCE expects float targets with shape [B,1]
                target_bce = cls_idx.float().unsqueeze(1)
                loss_cls = criterion_cls(cls_out, target_bce)
            # elevation is a single scalar in degrees
            loss_elev = criterion_elev(elev_out.squeeze(1), az_el_deg[:, 1])

            loss = loss_loc + args.cls_weight * loss_cls + args.elev_weight * loss_elev

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bsz = specs.size(0)
            running += loss.item() * bsz
            n_train += bsz

        avg_tr = running / max(1, n_train)
        train_losses.append(avg_tr)

        # Validation
        net.eval()
        running_val = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                specs, loc_xy, cls_idx, az_el_deg = batch
                specs = specs.to(device, non_blocking=True)
                loc_xy = loc_xy.to(device, non_blocking=True)
                cls_idx = cls_idx.to(device, non_blocking=True).squeeze(1)
                az_el_deg = az_el_deg.to(device, non_blocking=True)

                outputs = net(specs)
                loc_out = outputs[0]
                cls_out = outputs[1]
                elev_out = outputs[2]

                loss_loc = criterion_loc(loc_out, loc_xy)
                if num_classes > 1:
                    loss_cls = criterion_cls(cls_out, cls_idx)
                else:
                    target_bce = cls_idx.float().unsqueeze(1)
                    loss_cls = criterion_cls(cls_out, target_bce)
                loss_elev = criterion_elev(elev_out.squeeze(1), az_el_deg[:, 1])

                loss = loss_loc + args.cls_weight * loss_cls + args.elev_weight * loss_elev
                bsz = specs.size(0)
                running_val += loss.item() * bsz
                n_val += bsz

        avg_val = running_val / max(1, n_val)
        val_losses.append(avg_val)

        print(f"[{datetime.now()}] Epoch {epoch+1:03d}/{args.epochs} | train {avg_tr:.4f} | val {avg_val:.4f}")

        # Save best
        if avg_val < best_val:
            best_val = avg_val
            state = net.module.state_dict() if isinstance(net, nn.DataParallel) else net.state_dict()
            torch.save({
                'epoch': epoch,
                'state_dict': state,
                'best_loss': best_val,
                'log': {'training': train_losses, 'validation': val_losses},
                'conf': {
                    'image_size': SPECTROGRAM_SIZE,
                    'patch_size': PATCH_SIZE,
                    'patch_overlap': PATCH_OVERLAP,
                    'num_classes': NUM_OUTPUT,
                }
            }, os.path.join(MODEL_SAVE, model_save_name + '_best.pkl'))

        # Save last
        state = net.module.state_dict() if isinstance(net, nn.DataParallel) else net.state_dict()
        torch.save({
            'epoch': epoch,
            'state_dict': state,
            'best_loss': best_val,
            'log': {'training': train_losses, 'validation': val_losses},
        }, os.path.join(MODEL_SAVE, model_save_name + '_last.pkl'))


if __name__ == '__main__':
    main()


