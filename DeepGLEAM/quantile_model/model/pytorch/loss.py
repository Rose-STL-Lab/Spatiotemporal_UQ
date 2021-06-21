import torch


#def masked_mae_loss(y_pred, y_true):
    # mask = (y_true != 0).float()
    # mask /= mask.mean()
#    loss = torch.abs(y_pred - y_true)
    # loss = loss * mask
    # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
#    loss[loss != loss] = 0
#    return loss.mean()

def quantile_loss(y_pred, y_true):
    quantiles = [0.025, 0.5, 0.975]
    losses = []
    for i, q in enumerate(quantiles):
        errors =  y_true - torch.unbind(y_pred,3)[i]
        errors[errors != errors] = 0
        losses.append(torch.max((q-1) * errors, q * errors).unsqueeze(0))
    loss = torch.mean(torch.sum(torch.cat(losses, dim=0), dim=0))
    return loss
