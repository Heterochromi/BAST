from network.BAST import BAST_Variant, AngularLossWithCartesianCoordinate, MixWithCartesianCoordinate
from utils import *
from datetime import datetime
from conf import *
import argparse
import torch
import torch.nn as nn
import sys


parser = argparse.ArgumentParser()
parser.add_argument("-b", "--backbone", help="mamba/vanilla/swin")
parser.add_argument("-i", "--integ", help="SUB/ADD/CONCAT")
parser.add_argument("-l", "--loss", help="MSE/AD/MIX")
parser.add_argument("-s", "--shareweights", help="Share weights", type=bool, default=False)
parser.add_argument("-e", "--env", help="RI/RI01/RI02")
parser.add_argument("--classify", help="Enable binary sound classification head", action='store_true')
parser.add_argument("--cls_weight", help="Weight for classification loss", type=float, default=1.0)
args = parser.parse_args()

BINAURAL_INTEGRATION = args.integ
LOSS = args.loss
SHARE_PARAMS = args.shareweights
DATA_ENV = args.env
MODEL_TYPE = args.backbone
ENABLE_CLASSIFY = args.classify
CLS_WEIGHT = args.cls_weight

if 'SNR' in DATA_ENV:
    LR = 0.001

"""Loading the training and BAST datasets"""
print('\n[{}] Loading data...'.format(datetime.now()))
d_x, d_y = load_dataset(DATA_ENV, train=True, raw_path=DATA_DIR, converted_path=CONVERTED_DATA_DIR)
d_x[np.isinf(d_x)] = 0
data_manager = DataManager(d_x, d_y)
tr_x, tr_y, val_x, val_y = data_manager.split_data_large([TRAINING_PERCENT, VALIDATION_PERCENT])

tr_x = torch.Tensor(tr_x)
tr_y = torch.Tensor(tr_y)

val_x = torch.Tensor(val_x)
val_y = torch.Tensor(val_y)

"""Initializing the network"""
print('\n[{}] Initializing the network...'.format(datetime.now()))
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
    binaural_integration=BINAURAL_INTEGRATION,
    share_params=SHARE_PARAMS,
    transformer_variant=MODEL_TYPE,
    classify_sound=ENABLE_CLASSIFY,
    num_classes_cls=1,
)

"""Parallelize the network"""
if GPU_LIST:
    net = nn.DataParallel(net, device_ids=GPU_LIST).cuda()

"""Initializing settings"""
print('[{}] Initializing optimizer...'.format(datetime.now()))
optimizer = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=0)
print('[{}] Initializing loss function...'.format(datetime.now()))
if LOSS == 'MSE':
    criterion = nn.MSELoss()
elif LOSS == 'AD':
    criterion = AngularLossWithCartesianCoordinate()
elif LOSS == 'MIX':
    criterion = MixWithCartesianCoordinate()
criterion_cls = nn.BCEWithLogitsLoss() if ENABLE_CLASSIFY else None
conf = {
    'SPECTROGRAM_SIZE': SPECTROGRAM_SIZE,
    'PATCH_SIZE': PATCH_SIZE,
    'PATCH_OVERLAP': PATCH_OVERLAP,
    'NUM_OUTPUT': NUM_OUTPUT,
    'EMBEDDING_DIM': EMBEDDING_DIM,
    'TRANSFORMER_DEPTH': TRANSFORMER_DEPTH,
    'TRANSFORMER_HEADS': TRANSFORMER_HEADS,
    'TRANSFORMER_MLP_DIM': TRANSFORMER_MLP_DIM,
    'TRANSFORMER_POOL': TRANSFORMER_POOL,
    'TRANSFORMER_DIM_HEAD': TRANSFORMER_DIM_HEAD,
    'INPUT_CHANNEL': INPUT_CHANNEL,
    'DROPOUT': DROPOUT,
    'EMB_DROPOUT': EMB_DROPOUT,
    'DATA_ENV': DATA_ENV,
    'LR': LR,
    'EPOCH': EPOCH,
    'BATCH_SIZE': BATCH_SIZE,
    'BINAURAL_INTEGRATION': BINAURAL_INTEGRATION,
    'LOSS': LOSS,
    'SHARE_PARAMS': SHARE_PARAMS
}
num_tr = tr_y.shape[0]
num_batches = int(np.ceil(num_tr / BATCH_SIZE))
num_val = val_y.shape[0]
num_batches_val = int(np.ceil(num_val / BATCH_SIZE))
tr_loss = []
val_loss = []
min_val_loss = 10
start_epoch = 0
end_epoch = start_epoch + EPOCH

model_save_name = MODEL_NAME + '_' + BINAURAL_INTEGRATION + '_' + LOSS + '_XY' + '_' + (
    'SP' if SHARE_PARAMS else 'NSP') + '_' + MODEL_TYPE
if DATA_ENV != 'RI':
    model_save_name += '_' + DATA_ENV
if ENABLE_CLASSIFY:
    model_save_name += '_CLS'

"""Starting training"""
print('[{}] Start training...'.format(datetime.now()))
for epoch in range(start_epoch, end_epoch):
    batch_loss = 0
    for batch in range(num_batches):
        net.train()
        idx_s = BATCH_SIZE * batch
        idx_e = BATCH_SIZE * (batch + 1)
        idx_e = np.min([idx_e, num_tr])
        input_x = tr_x[idx_s:idx_e, :, :, :].cuda()
        target = tr_y[idx_s:idx_e, :].cuda()
        output = net(input_x)
        if ENABLE_CLASSIFY:
            loc_out, cls_out = output
            # Expect classification target to be in last channel if provided; else derive a dummy zero vector
            if tr_y.shape[1] > 2:
                target_loc = target[:, :2]
                target_cls = target[:, 2].unsqueeze(1)
            else:
                target_loc = target
                target_cls = torch.zeros_like(cls_out)
            loss_loc = criterion(loc_out, target_loc)
            loss_cls = criterion_cls(cls_out, target_cls)
            loss = loss_loc + CLS_WEIGHT * loss_cls
        else:
            loss = criterion(output, target)
        loss_v = loss.item()
        batch_loss += loss_v * (idx_e - idx_s)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(
            '\r[{}] [TRAINING] Epoch: {}, Batch: {}, Curr Loss: {:.6f}, Avg Loss: {:.6f}'.format(datetime.now(), epoch,
                                                                                                 batch, loss_v,
                                                                                                 batch_loss / idx_e),
            end='')
    avg_batch_loss = batch_loss / num_tr
    tr_loss.append(avg_batch_loss)
    print('\r[{}] [TRAINING] Epoch: {}, Batch: {}, Curr Loss: {:.6f}, Avg Loss: {:.6f}'.format(datetime.now(), epoch,
                                                                                               batch, loss_v,
                                                                                               avg_batch_loss))

    batch_loss_val = 0
    with torch.no_grad():
        for batch in range(num_batches_val):
            net.eval()
            idx_s = BATCH_SIZE * batch
            idx_e = BATCH_SIZE * (batch + 1)
            idx_e = np.min([idx_e, num_val])
            input_x = val_x[idx_s:idx_e, :, :, :].cuda()
            target = val_y[idx_s:idx_e, :].cuda()
            output = net(input_x)
            if ENABLE_CLASSIFY:
                loc_out, cls_out = output
                if val_y.shape[1] > 2:
                    target_loc = target[:, :2]
                    target_cls = target[:, 2].unsqueeze(1)
                else:
                    target_loc = target
                    target_cls = torch.zeros_like(cls_out)
                loss_loc = criterion(loc_out, target_loc)
                loss_cls = criterion_cls(cls_out, target_cls)
                loss = loss_loc + CLS_WEIGHT * loss_cls
            else:
                loss = criterion(output, target)
            loss_v = loss.item()
            batch_loss_val += loss_v * (idx_e - idx_s)
            print('\r[{}] [VALIDATION] Epoch: {}, Batch: {}, Curr Loss: {:.6f}, Avg Loss: {:.6f}'.format(datetime.now(),
                                                                                                         epoch, batch,
                                                                                                         loss_v,
                                                                                                         batch_loss_val / idx_e),
                  end='')
        avg_batch_loss_val = batch_loss_val / num_val
        val_loss.append(avg_batch_loss_val)
        print('\r[{}] [VALIDATION] Epoch: {}, Batch: {}, Curr Loss: {:.6f}, Avg Loss: {:.6f}'.format(datetime.now(),
                                                                                                     epoch, batch,
                                                                                                     loss_v,
                                                                                                     avg_batch_loss_val))
        print(tr_loss)
        print(val_loss)

    # Save the best network
    if min_val_loss > avg_batch_loss_val:
        min_val_loss = avg_batch_loss_val
        torch.save({'epoch': epoch, 'state_dict': net.module.state_dict(), 'best_loss': min_val_loss,
                    # 'optimizer': optimizer.state_dict(),
                    'log': {'training': tr_loss,
                            'validation': val_loss},
                    'conf': conf},
                   MODEL_SAVE + model_save_name + '_best.pkl')

    # Save the last network
    torch.save({'epoch': epoch, 'state_dict': net.module.state_dict(), 'best_loss': min_val_loss,
                # 'optimizer': optimizer.state_dict(),
                'log': {'training': tr_loss,
                        'validation': val_loss},
                'conf': conf},
               MODEL_SAVE + model_save_name + '_last.pkl')
sys.exit()
