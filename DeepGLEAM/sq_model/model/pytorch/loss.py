import torch
import torch.nn as nn
import torch.nn.functional as F

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


def spline_quantile_loss(y_pred, y_true):

    y_true_1 = torch.unsqueeze(y_true, -1)
    gamma = y_pred[:,:,:,:1]
    beta = y_pred[:,:,:,1:6]
    m = nn.Softmax(dim =3)
    delta = m(y_pred[:,:,:,6:11])

    b = beta - F.pad(beta, (1, 0))[:, :, :, :-1]
    d = F.pad(torch.cumsum(delta, dim=3), (1, 0))[:, :, :, :-1]  # d term for piecewise-linear functions

    value_knot = torch.add(F.pad(torch.cumsum(beta*delta, dim=3), (1, 0)),gamma)

    mask_1 = torch.where(value_knot >= 
        y_true_1,torch.zeros(value_knot.shape).cuda(),torch.ones(value_knot.shape).cuda())
    mask1 = mask_1[:, :, :, :-1]

    a_tilde_1 = (y_true_1-gamma+ torch.sum(mask1*b*d,3,keepdim=True))/ (1e-10+torch.sum(mask1*b,3,keepdim=True))
    a_tilde = torch.max(torch.min(a_tilde_1,torch.ones(a_tilde_1.shape).cuda()),torch.zeros(a_tilde_1.shape).cuda())

    coeff = (1.0 - torch.pow(d, 3)) / 3.0 - d - torch.pow(torch.max(a_tilde,d),2) + 2 * torch.max(a_tilde,d) * d
    crps = (2 * a_tilde - 1) * y_true_1 + (1 - 2 * a_tilde) * gamma + torch.sum(b * coeff,3,keepdim=True)

    loss = crps
# trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0

    return loss.mean()
