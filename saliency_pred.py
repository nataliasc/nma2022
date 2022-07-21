"""
This file contains the code for saliency prediction algorithm.
Creator: Lucy, Kaitlyn, Maria, Linas
"""
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
# tqdm is a library for smart loops in ML used by neuromatch tutors
import tqdm
from audtorch.metrics import PearsonR

from utils_saliency import set_device, set_seed

# set device and random seed
DEVICE = set_device()

SEED = 2022
set_seed(seed=SEED)


#################
# data preprocessing
#################


#################
# data set preparation
#################


#################
# network class
#################


#################
# evaluation function
#################

def eval_model(model, data_loader, device=DEVICE):
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
            data, target = batch[0].to(device), batch[1].to(device)
            # Evaluate model and loss on minibatch
            preds = model(data)
            # compute batch mean kl div
            kl_div = F.kl_div(preds, target, reduction='batchmean').item()
            kl_log.append(kl_div)
            # compute batch mean pearsonr
            metric_pr = PearsonR(reduction='mean', batch_first=True)
            pear_corr = metric_pr(torch.flatten(preds, start_dim=1), torch.flatten(target,
                                                                                   start_dim=1)).item()  # reshape pred and target to batch_size*num_pixels to fit PearsonR() class
            corr_log.append(pear_corr)

    return np.mean(kl_log), np.mean(corr_log)


#################
# training loop
#################

def train(model, train_loader, val_loader, optimizer, loss_function, eval_model,
          MAX_EPOCHS=2,
          LOG_FREQ=200,
          VAL_FREQ=200,
          device=DEVICE):
    """
  trains the model
  :arg model: defined network object
  :arg train_loader: dataloader object containing the training set
  :arg val_loader: dataloader object containing the validation set
  :arg optimizer: optimizer for the network (torch.optim object)
  :arg loss_function: loss function used in the network (may need to specify if used for CPU or GPU)
  :arg eval_model: evaluation model used
  :arg MAX_EPOCHS: number of epochs used to train the model
  :arg LOG_FREQ (int): model prints training statistics every LOG_FREQ batches
  :arg VAL_FREQ (int): frequency for evaluating the validation metrics (measured in batches)
  :arg device (str, 'cpu' or 'cuda:0'): what device the network is trained on
  :return: trained model
  """

    # define metrics
    metrics = {'train_loss': [],
               'val_loss': [],
               'val_idx': []}

    # step_idx is the counter of BATCHES within an epoch
    step_idx = 0

    # iterate over each epoch (full dataset)
    # tqdm is a library for loops in ML
    for epoch in tqdm(range(MAX_EPOCHS)):

        # at the start of the epoch, training loss is 0
        running_loss = 0.0

        # within an epoch, loop over batches of data
        for batch_id, batch in enumerate(train_loader):

            # add 1 to the current count of batches
            step_idx += 1

            # get data: Extract minibatch data and labels, make sure the rest runs on the right device
            data, labels = batch[0].to(device), batch[1].to(device)

            # 1. set the parameter gradients to zero
            optimizer.zero_grad()

            # 2. run forward prop (apply the convnet on the input data)
            preds = model(data)

            # 3. define loss by our criterion (e.g. cross entropy loss)
            # 1st arg: predictions, 2nd arg: data
            loss = loss_function(preds, labels)

            # 4. calculate the gradients
            loss.backward()

            # 5. update the parameter weights based on the gradients
            optimizer.step()

            # add metrics for plotting
            metrics['train_loss'].append(loss.cpu().item())

            # Every 1 in VAL_FREQ iterations, get validation metrics and print them
            # calling eval_model
            if batch_id % VAL_FREQ == (VAL_FREQ - 1):
                val_loss = eval_model(model, val_loader, num_batches=100, device=device)

                metrics['val_idx'].append(step_idx)
                metrics['val_loss'].append(val_loss)

                print(f"[VALID] Epoch {epoch + 1} - Batch {batch_id + 1} - "
                      f"Loss: {val_loss:.3f}")

            # print statistics
            running_loss += loss.cpu().item()

            # Print results every LOG_FREQ minibatches
            if batch_id % LOG_FREQ == (LOG_FREQ - 1):
                print(f"[TRAIN] Epoch {epoch + 1} - Batch {batch_id + 1} - "
                      f"Loss: {running_loss / LOG_FREQ:.3f} - ")

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
    model_scripted.save('model_scripted.pt')

    return model
