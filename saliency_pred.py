"""
This file contains the code for saliency prediction algorithm.
Creator: Lucy, Kaitlyn, Maria, Linas
"""
import torch.nn.functional as F
from audtorch.metrics import PearsonR

from utils_saliency import *

DEVICE = set_device()

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
# training loop
#################


#################
# evaluation function
#################

def eval_model(model, data_loader, device=DEVICE):
    '''
    evaluates the performance of saliency prediction by giving separate losses
    :arg model: defined network object
    :arg data_loader: dataloader object containing either validation or test set

    :return: mean of each of three evaluation metrics across all batches:
    KL divergence, Pearson's correlation coefficient, Normalized Scanpath Saliency
    '''

    with torch.no_grad():
        for batch_id, batch in enumerate(data_loader):
            # Extract minibatch data
            data, target = batch[0].to(device), batch[1].to(device)
            # Evaluate model and loss on minibatch
            preds = model(data)
            # compute batch mean kl div
            kl_div = F.kl_div(preds, target, reduction='batchmean').item()
            # compute batch mean pearsonr
            metric_pr = PearsonR(reduction='mean', batch_first=True)
            pear_corr = metric_pr(preds, target)

    return kl_div, pear_corr

# this metric requires different target data format,
# def norm_scanpath_saliency(pred_map, target_fixation_map):
#     '''
#     computes Normalized Scanpath Saliency which measures the average normalized saliency between two fixation maps
#
#     :arg pred_map: network output
#     :arg target_map: ground truth fixation map containing binary values 0 and 1
#     :return: nss
#     '''
#
#     return norm_ss
