import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda')

def feed_model_data_dev(model, grid_data_seq):
    hidden_states = None

    _, hidden_states = model(grid_data_seq, hidden_states)

    return model, hidden_states

def generate_forecasts_dev(model, hidden_states, init_grid_data, seq_len=48):
    prev_grid = init_grid_data
    meo_forecasts = []

    for i in range(seq_len):
        output, hidden_states = model(prev_grid[:,:,:6], hidden_states)
        weather = output[:,:,:5]
        gamma = output[:,:,5:6]
        beta = output[:,:,6:11]
        delta = output[:,:,11:16]
        m = nn.Softmax(dim =2)
        delta = m(delta)
        b = beta - F.pad(beta, (0, 0, 0, 0, 1, 0))[:, :, :-1]
        d_origin = torch.cumsum(delta, dim=2) 
        d = F.pad(d_origin, (0, 0, 0, 0, 1, 0))[:, :, :-1]
        median = 0.5
        mask2 = torch.where(median-d>0,median-d,torch.zeros(d.shape).cuda())
        prev_grid = torch.cat([weather,torch.unsqueeze(torch.sum(mask2*b,2),2) + gamma],dim=2)
        meo_forecasts.append(output)

    return torch.cat(meo_forecasts, 1)

def sq_loss(y_pred, y_true):
    y_true_1 = torch.unsqueeze(y_true, dim = 2)
    gamma = y_pred[:,:,:1]
    beta = y_pred[:,:,1:6]
    m = nn.Softmax(dim =2)
    delta = m(y_pred[:,:,6:11])

    b = beta - F.pad(beta, (0, 0, 0, 0, 1, 0))[:, :, :-1]
    d = F.pad(torch.cumsum(delta, dim=2), (0, 0, 0, 0, 1, 0))[:, :, :-1]

    value_knot = torch.add(F.pad(torch.cumsum(beta*delta, dim=2), (0, 0, 0, 0, 1, 0)),gamma)

    mask_1 = torch.where(value_knot >= 
        y_true_1,torch.zeros(value_knot.shape).cuda(),torch.ones(value_knot.shape).cuda())
    mask1 = mask_1[:, :, :-1]

    a_tilde_1 = (y_true_1-gamma+ torch.sum(mask1*b*d,2,keepdim=True))/ (1e-10+torch.sum(mask1*b,2,keepdim=True))
    a_tilde = torch.max(torch.min(a_tilde_1,torch.ones(a_tilde_1.shape).cuda()), torch.zeros(a_tilde_1.shape).cuda())

    coeff = (1.0 - torch.pow(d, 3)) / 3.0 - d - torch.pow(torch.max(a_tilde,d),2) + 2 * torch.max(a_tilde,d) * d
    crps = (2 * a_tilde - 1) * y_true_1 + (1 - 2 * a_tilde) * gamma + torch.sum(b * coeff,2,keepdim=True)

    crps[crps != crps] = 0
    return crps.mean()

def compute_dev_set_loss(model, input_seqs, target_meo_seqs):
    loss_function = nn.L1Loss()
    losses = []
    with torch.no_grad():
        for i in range(len(input_seqs)):
            input_seq = torch.tensor(
                input_seqs[i][:,:-1],
                dtype=torch.float32,
                device=device,
            )

            start_point = torch.tensor(
                input_seqs[i][:,-1:],
                dtype=torch.float32,
                device=device,
            )

            target_meo = torch.tensor(
                target_meo_seqs[i],
                dtype=torch.float32,
                device=device,
            )

            model, hidden_states = feed_model_data_dev(model, input_seq)
            meo_forecasts = generate_forecasts_dev(model, hidden_states, start_point)
            loss_meo = loss_function(meo_forecasts[:,:,:5], target_meo[:,:,:5])/5
            loss_pm = sq_loss(meo_forecasts[:,:,5:], target_meo[:,:,-1])
            loss = loss_meo+loss_pm
            losses.append(loss.item())
    return np.mean(losses)

def prediction(model, input_seqs):
    with torch.no_grad():
        for i in range(len(input_seqs)):
            input_seq = torch.tensor(
                input_seqs[i][:,:-1],
                dtype=torch.float32,
                device=device,
            )

            start_point = torch.tensor(
                input_seqs[i][:,-1:],
                dtype=torch.float32,
                device=device,
            )

            model, hidden_states = feed_model_data_dev(model, input_seq)
            meo_forecasts = generate_forecasts_dev(model, hidden_states, start_point)

            pred = meo_forecasts.detach().cpu().numpy()

    return pred
