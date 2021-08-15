import torch
import torch.nn as nn 


def mae_loss(y_pred, y_true):
    loss = torch.abs(y_pred - y_true)
    loss[loss != loss] = 0
    return loss.mean()

# def rmse_loss(y_pred, y_true):
#     mse = nn.MSELoss()
#     eps = torch.tensor([1e-6]).cuda()
#     loss = torch.sqrt(mse(y_pred, y_true)+eps)
#     #loss[loss != loss] = 0
#     return loss
