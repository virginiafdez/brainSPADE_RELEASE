import os
import numpy as np
import random
import time
import torch
from torch.utils.data import TensorDataset, DataLoader,Dataset
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
import monai
from options.modisc_options import ModiscOptions
from torch.utils.tensorboard import SummaryWriter
from data.spadenai_v2_sliced import SpadeNaiSlice
from modality_classifier.modisc_net import Modisc
from tqdm import tqdm

# Callable functions
def transform_to_ohe_numerical(name_vals, batch):
    out = np.zeros([len(batch), len(name_vals)])
    batch_ = np.asarray(batch)
    for ind, i in enumerate(name_vals):
        out[:, ind] = (batch_ == i).astype(float)
    return torch.from_numpy(out)

def train_epoch(model, loader, optimizer, criterion, sequences, datasets, device,
                len_cont):
    model.train()
    train_loss = []
    for ind, data_i in enumerate(loader):
        gt_seq = transform_to_ohe_numerical(sequences, data_i['this_seq']).to(device)
        gt_dat = transform_to_ohe_numerical(datasets, data_i['this_dataset']).to(device)
        img = data_i['image'].to(device)
        optimizer.zero_grad()
        logits_mod, logits_dat = model(img)
        loss = criterion(logits_mod, gt_seq) + criterion(logits_dat, gt_dat)
        loss.backward()
        optimizer.step()
        loss_np = loss.detach().cpu().numpy()
        train_loss.append(loss_np)
        if ind%100 == 0:
            print("Iteration %d / %d\n" %(ind, len_cont))
    return np.mean(train_loss)

def val_epoch(model, loader, criterion, sequences, datasets, device, len_cont):
    model.eval()
    val_loss = []
    LOGITS_MODS = []
    LOGITS_DATA = []
    TARGET_MODS = []
    TARGET_DATA = []
    with torch.no_grad():
        for data_i in loader:
            gt_seq = transform_to_ohe_numerical(sequences, data_i['this_seq']).to(device)
            gt_dat = transform_to_ohe_numerical(datasets, data_i['this_dataset']).to(device)
            img = data_i['image'].to(device)
            optimizer.zero_grad()
            logits_mod, logits_dat = model(img)
            LOGITS_MODS.append(logits_mod.detach().cpu())
            LOGITS_DATA.append(logits_dat.detach().cpu())
            TARGET_MODS.append(gt_seq.detach().cpu())
            TARGET_DATA.append(gt_dat.detach().cpu())

        val_loss = criterion(torch.cat(LOGITS_MODS), torch.cat(TARGET_MODS)).numpy() + \
                   criterion(torch.cat(LOGITS_DATA), torch.cat(TARGET_DATA)).numpy()
        PROBS_MOD = torch.softmax(torch.cat(LOGITS_MODS), dim=1).numpy().squeeze()
        PROBS_DAT = torch.softmax(torch.cat(LOGITS_DATA), dim=1).numpy().squeeze()
        LOGITS_MODS = torch.cat(LOGITS_MODS).cpu().numpy()  # Dimension: B x (NCLASSES)
        LOGITS_DATA = torch.cat(LOGITS_DATA).cpu().numpy()  # Dimension: B x (NCLASSES)vv
        TARGET_MODS = torch.cat(TARGET_MODS).cpu().numpy()  # Dimension: B x (NCLASSES)
        TARGET_DATA = torch.cat(TARGET_DATA).cpu().numpy()  # Dimension: B x (NCLASSES)

    acc_mod = (np.argmax(PROBS_MOD, 1) == np.argmax(TARGET_MODS, 1)).mean() * 100.0
    acc_dat = (np.argmax(PROBS_DAT, 1) == np.argmax(TARGET_DATA, 1)).mean() * 100.0
    try:
        auc_mod = roc_auc_score(TARGET_MODS, LOGITS_MODS)
    except:
        auc_mod = 0.0
    try:
        auc_dat = roc_auc_score(TARGET_DATA, LOGITS_DATA)
    except:
        auc_dat = 0.0

    return float(val_loss), acc_mod, auc_mod, acc_dat, auc_dat

# Define criterion (loss)
def define_criterion():
    return nn.BCEWithLogitsLoss()


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    print('Seeded!')

def findLRGamma(startingLR, endingLR, num_epochs):
    '''
    Gamma getter. Based on a minimum and maximum learning rate, calculates the Gamma
    necessary to go from minimum to maimum in num_epochs.
    :param startingLR: First Learning Rate.
    :param endingLR: Final Learning Rate.
    :param num_epochs: Number of epochs.
    :return:
    '''

    gamma = np.e ** (np.log(endingLR / startingLR) / num_epochs)
    return gamma

# RUNNABLE LOOP
set_seed(0)
options = ModiscOptions().parse()
model = Modisc(n_mods = len(options.sequences), n_datasets=len(options.datasets),
               dropout=0.2, spatial_dims=2, in_channels=1)
if options.continue_train:
    epoch_last = model.findLastModel(options.checkpoints_dir,)
else:
    epoch_last = 0

# Define dataset
dataset_container = SpadeNaiSlice(options, mode = 'train')
dataset_val_container = SpadeNaiSlice(options, mode = 'validation', store_and_use_slices=True)
dataloader = DataLoader(dataset_container.sliceDataset,
                        batch_size=options.batchSize, shuffle=False,
                        num_workers=int(options.nThreads), drop_last=options.isTrain,
                        )
dataloader_val = DataLoader(dataset_val_container.sliceDataset,
                            batch_size=options.batchSize, shuffle=False,
                            num_workers=int(options.nThreads), drop_last=False,
                            )

# Define folders
checkpoints = os.path.join(options.checkpoints_dir, options.name)
if not os.path.isdir(checkpoints):
    os.makedirs(checkpoints)
log_dir = os.path.join(checkpoints, "logs")
if not os.path.isdir(log_dir):
    os.makedirs(log_dir)

# Define logs
tboard = SummaryWriter(log_dir=log_dir)
if options.continue_train:
    write_flag = 'a'
else:
    write_flag = 'w'
with open(os.path.join(checkpoints, 'log.txt'), write_flag) as f:
    f.write("Epoch\tTrain loss\tVal loss\tAccuracy-Mod\tAccuracy-Dat\tAUC-Mod\tAUC-Dat\n")
    f.close()

# Define learning rate, optimizer and so on
gamma_lr = findLRGamma(options.LR_init, options.LR_end, options.num_epochs)
if options.continue_train:
    lr_now = options.LR_init * np.exp(-gamma_lr*epoch_last)
else:
    lr_now = options.LR_init
optimizer = torch.optim.Adam(model.parameters(), lr_now, betas = (0.9, 0.999))
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma_lr)

# Criterion
criterion = define_criterion()

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training loop
model.to(device)
for epoch in range(epoch_last, options.num_epochs):
    print("Epoch %d/%d" %(epoch, options.num_epochs))
    train_loss = train_epoch(model, dataloader, optimizer, criterion, options.sequences, options.datasets,
                             device, len_cont = len(dataset_container))
    print("Training loss: %.3f" %train_loss)
    val_loss, acc_mod, auc_mod, acc_dat, auc_dat = val_epoch(model, dataloader_val, criterion,
                                                             options.sequences, options.datasets,
                                                             device, len_cont = len(dataset_container))

    print("Val. loss: %.3f, Acc. mod %.3f, Acc. dat %.3f., AUC mod %.3f, AUC dat %.3f\n"
          %(val_loss, acc_mod, acc_dat, auc_mod, auc_dat))

    with open(os.path.join(checkpoints, 'log.txt'), 'a') as f:
        f.write("%d\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n"
                %(epoch, train_loss, val_loss, acc_mod, acc_dat, auc_mod, auc_dat))
        f.close()

    if epoch % options.epoch_save == 0 and epoch > 0:
        model.saveLatestModel(checkpoints, epoch)

    scheduler.step()
    for param_group in optimizer.param_groups:
        lr_ = param_group['lr']
    print("Learning rate: %.5f" % lr_)

    if options.use_tboard:
        tboard.add_scalar("training/training loss", train_loss, epoch)
        tboard.add_scalar("validation/val loss", val_loss, epoch)
        tboard.add_scalar("validation/mod accuracy", acc_mod, epoch)
        tboard.add_scalar("validation/dat accuracy", acc_dat, epoch)
        tboard.add_scalar("validation/mod AUC", auc_mod, epoch)
        tboard.add_scalar("validation/dat AUC", auc_dat, epoch)

print("Finished training at epoch %d" %epoch)
torch.cuda.empty_cache()
dataset_container.clearCache()
dataset_val_container.clearCache()
model.saveLatestModel(checkpoints, epoch)
tboard.flush()
