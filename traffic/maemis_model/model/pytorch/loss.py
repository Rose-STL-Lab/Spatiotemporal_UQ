import torch

def maemis_loss(y_pred, y_true):
    mask = (y_true != 0).float()
    mask /= mask.mean()
    pho = 0.05
    loss0 = torch.abs(y_pred.T[2].T - y_true)
    loss1 = torch.max(y_pred.T[0].T-y_pred.T[1].T,torch.tensor([0.]).cuda())
    loss2 = torch.max(y_pred.T[1].T-y_true,torch.tensor([0.]).cuda())*2/pho
    loss3 = torch.max(y_true-y_pred.T[0].T,torch.tensor([0.]).cuda())*2/pho
    loss = loss0+loss1+loss2+loss3
    loss = loss * mask
    loss[loss != loss] = 0
    return loss.mean()
