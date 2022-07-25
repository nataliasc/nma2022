"""
This file contains the code for saliency prediction algorithm.
Creator: Lucy, Kaitlyn, Maria, Linas
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torch.nn as nn
import wandb
import seaborn as sns
# tqdm is a library for smart loops in ML used by neuromatch tutors
from audtorch.metrics import PearsonR

from utils_saliency import set_device, set_seed
from simple_fcn_gaze_pred import SimpleFCN

# set device and random seed
DEVICE = set_device()

SEED = 2022
set_seed(seed=SEED)

# %%
#################
# data preprocessing
#################
dataset = torch.load('processed_data/data.pt')
total_samples = len(dataset)
split = [int(.8 * total_samples), int(.1 * total_samples), int(.1 * total_samples)]
train_set, val_set, test_set = data.random_split(dataset, split, generator=torch.Generator().manual_seed(SEED))

# %%
#################
# data set preparation
#################
wandb.login(key='25f10546ef384a6f1ab9446b42d7513024dea001')
wandb.init(project="saliency-prediction", entity="nma2022")

config = wandb.config
config.batch_size = 100
config.lr = 1e-3
config.epoch = 100
config.log_freq = 200
config.val_freq = 200

train_loader = data.DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=2)
val_loader = data.DataLoader(val_set, batch_size=config.batch_size, shuffle=True, num_workers=2)
test_loader = data.DataLoader(test_set, batch_size=config.batch_size, shuffle=True, num_workers=2)

# %%
#################
# network class
#################
net = SimpleFCN(config.batch_size, DEVICE)
net.float().to(DEVICE)
#criterion = nn.KLDivLoss(reduction="batchmean", log_target=True)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=config.lr)

wandb.watch(net, log_freq=100)


# %%
#################
# evaluation function
#################
def pearson_r_batchmean(x, y):
    """
    compute pearson correlation between predicted heat maps
    :param x: batch * 84 * 84  prediction tensor
    :param y: batch * 84 * 84  target tensor
    :return: pearson r, fill nan with 0 tensor
    """
    assert x.shape == y.shape
    dim = -1  # batch first
    centered_x = x - x.mean(dim=dim, keepdim=True)
    centered_y = y - y.mean(dim=dim, keepdim=True)

    covariance = (centered_x * centered_y).sum(dim=dim, keepdim=True)

    bessel_corrected_covariance = covariance / (x.shape[dim] - 1)

    x_std = x.std(dim=dim, keepdim=True)
    y_std = y.std(dim=dim, keepdim=True)

    corr = bessel_corrected_covariance / (x_std * y_std)

    corr = torch.nan_to_num(corr, nan=0.0, posinf=1, neginf=-1)  # get rid of nan values

    return corr.mean()  # batch mean


def plot_map(pred_map):
    """
    plot predicted saliency density in log space during validation
    :param pred_map: tensor 84*84 containing saliency density
    :return: fig object for logging on wandb
    """
    pred_map = pred_map.numpy()

    fig, ax = plt.subplots()
    sns.heatmap(pred_map, ax=ax)

    return fig


def eval_model(model, data_loader=test_loader, device=DEVICE):
    """
    evaluates the performance of saliency prediction by giving separate losses
    :arg model: defined network object
    :arg data_loader: dataloader object containing either validation or test set
    :param device: cpu or gpu

    :return: mean of each of three evaluation metrics across all batches:
    KL divergence, Pearson's correlation coefficient, Normalized Scanpath Saliency
    """

    # list containing metric for each batch
    kl_log = []
    corr_log = []

    with torch.no_grad():
        for batch_id, batch in enumerate(data_loader):
            # Extract minibatch data
            data, target = batch[0].float().to(device), batch[1].float().to(device)
            # Evaluate model and loss on minibatch
            preds = model(data)
            preds = torch.squeeze(preds)  # raw pred dim: batch*1*84*84, target dim: batch*84*84
            log_target = torch.log(target)  # convert target to log space for comparison
            # compute batch mean kl div
            kl_div = F.kl_div(preds, log_target.float(), reduction='batchmean', log_target=False).cpu().item()
            kl_log.append(kl_div)
            # compute batch mean pearsonr
            pear_corr = pearson_r_batchmean(torch.flatten(preds, start_dim=1), torch.flatten(log_target,
                                                                                             start_dim=1)).cpu().item()  # reshape pred and target to batch_size*num_pixels to fit PearsonR() class
            corr_log.append(pear_corr)

            if batch_id == len(data_loader) - 1:
                pred_map = plot_map(torch.squeeze(preds[0, :, :].cpu()))
                wandb.log({'val predicted saliency density': wandb.Image(pred_map)})
                target_map = plot_map(torch.squeeze(log_target[0, :, :].cpu()))
                wandb.log({'val target saliency density': wandb.Image(target_map)})

    return np.mean(kl_log), np.mean(corr_log)


#################
# training loop
#################

def train(model, train_loader, val_loader, optimizer, loss_function, eval_model,
          EPOCHS=config.epoch,
          LOG_FREQ=config.log_freq,
          VAL_FREQ=config.val_freq,
          device=DEVICE):
    """
  trains the model
  :arg model: defined network object
  :arg train_loader: dataloader object containing the training set
  :arg val_loader: dataloader object containing the validation set
  :arg optimizer: optimizer for the network (torch.optim object)
  :arg loss_function: loss function used in the network (may need to specify if used for CPU or GPU)
  :arg eval_model: evaluation model used
  :arg EPOCHS: number of epochs used to train the model
  :arg LOG_FREQ (int): model prints training statistics every LOG_FREQ batches
  :arg VAL_FREQ (int): frequency for evaluating the validation metrics (measured in batches)
  :arg device (str, 'cpu' or 'cuda:0'): what device the network is trained on
  :return: trained model
  """

    # define metrics
    metrics = {'train_loss': [],
               'val_kl': [],
               'val_pearson_r': [],
               'val_idx': []}

    # step_idx is the counter of BATCHES within an epoch
    step_idx = 0

    # iterate over each epoch (full dataset)
    for epoch in range(EPOCHS):

        # at the start of the epoch, training loss is 0
        running_loss = 0.0

        # within an epoch, loop over batches of data
        for batch_id, batch in enumerate(train_loader):

            # add 1 to the current count of batches
            step_idx += 1

            # get data: Extract minibatch data and labels, make sure the rest runs on the right device
            data, labels = batch[0].float().to(device), batch[1].float().to(device)

            # 1. set the parameter gradients to zero
            optimizer.zero_grad()

            # 2. run forward prop (apply the convnet on the input data)
            preds = model(data)

            # 3. define loss by our criterion (e.g. cross entropy loss)
            # 1st arg: predictions, 2nd arg: data
            # loss = loss_function(torch.squeeze(preds), labels)  loss for kl div
            loss = loss_function(torch.squeeze(preds), torch.log(labels))

            # 4. calculate the gradients
            loss.backward()

            # 5. update the parameter weights based on the gradients
            optimizer.step()

            # add metrics for plotting
            metrics['train_loss'].append(loss.cpu().item())

            # Every 1 in VAL_FREQ iterations, get validation metrics and print them
            # calling eval_model
            if batch_id % VAL_FREQ == (VAL_FREQ - 1):
                kl_div, pearson_r = eval_model(net, val_loader, device=device)
                wandb.log({
                    'val kl div': kl_div,
                    'val pearson r': pearson_r
                })

                metrics['val_idx'].append(step_idx)
                metrics['val_kl'].append(kl_div)
                metrics['val_pearson_r'].append(pearson_r)

            # print statistics
            running_loss += loss.cpu().item()

            # Print results every LOG_FREQ minibatches
            if batch_id % LOG_FREQ == (LOG_FREQ - 1):
                print(f"[TRAIN] Epoch {epoch + 1} - Batch {batch_id + 1} - "
                      f"Loss: {running_loss / LOG_FREQ:.3f} - ")

                wandb.log({f'avg loss over {LOG_FREQ} batches': running_loss / LOG_FREQ,
                           'epoch': epoch})

                # reset loss
                running_loss = 0.0

                # PLOT THE RESULTS
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    ax[0].plot(range(len(metrics['train_loss'])), metrics['train_loss'],
               alpha=0.8, label='Train')
    ax[0].plot(metrics['val_idx'], metrics['val_loss'], label='Valid')
    ax[0].set_xlabel('Iteration')
    ax[0].set_ylabel('Loss')
    ax[0].legend()

    plt.tight_layout()
    plt.show()

    # Export the model to torchscript
    model_scripted = torch.jit.script(model)
    # Save the model
    model_scripted.save('trained_sali_pred/model_scripted.pt')

    return model


# %%
trained_model = train(net, train_loader, val_loader, optimizer, criterion, eval_model)
