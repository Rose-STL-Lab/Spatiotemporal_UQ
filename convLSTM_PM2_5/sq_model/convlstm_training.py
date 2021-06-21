import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import load_data as ld
import convlstm as md
import evaluation as ev
import torch.nn.functional as F
import copy

device = torch.device('cuda')

def feed_model_data(model, grid_data_seq):
    _, hidden_states = model(grid_data_seq)

    return model, hidden_states

def generate_forecasts(model, hidden_states, init_grid_data, seq_len=48):
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

def seq_preprocessing(grid_seqs):
    """

    :param grid_seqs: list of (m, Tx, n_c, n_h, n_w)
    :param aqi_seqs: list of (m, Tx, n_c)
    :return:
    """
    input_seqs = []
    target_meo_seqs = []
    avg_grids = []
    std_grids = []

    for data in grid_seqs:
        m, Tx, _, _, _ = data.shape
        avg = np.reshape(np.average(data, axis=(1, 3, 4)), (m, 1, 6, 1, 1))
        std = np.reshape(np.std(data, axis=(1, 3, 4)), (m, 1, 6, 1, 1))
        avg_grids.append(avg)
        std_grids.append(std)

    for i in range(len(grid_seqs)):
        grid_seq = grid_seqs[i]
        grid_seq = (grid_seq - avg_grids[i]) / std_grids[i]

        input_seq = grid_seq[:, :24, :, :, :]  # Remove the last from the input seq
        target_meo = grid_seq[:, 24:, :, :, :]

        input_seqs.append(input_seq)
        target_meo_seqs.append(target_meo)

    assert len(input_seqs) == len(target_meo_seqs)
    return input_seqs, target_meo_seqs, avg_grids[0], std_grids[0]


def train(
        model,
        input_seqs,
        target_meo_seqs,
        dev_input_seqs,
        dev_target_meo_seqs,
        snapshots,
        iterations=100,
        lr=0.01,
        clipping_norm=1e-5):
    loss_function = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr)

    for epoch in range(iterations):
        losses = []
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

            model.zero_grad()
            model, hidden_states = feed_model_data(model, input_seq)
            meo_forecasts = generate_forecasts(model, hidden_states, start_point)
            loss_meo = loss_function(meo_forecasts[:,:,:5], target_meo[:,:,:5])/5
            loss_pm = sq_loss(meo_forecasts[:,:,5:], target_meo[:,:,-1])
            loss = loss_meo+loss_pm
            loss.backward()

            losses.append(loss.item())
            nn.utils.clip_grad_norm_(model.parameters(), clipping_norm)
            optimizer.step()

        loss = np.mean(losses)

        dev_loss = ev.compute_dev_set_loss(model, dev_input_seqs, dev_target_meo_seqs)
        snapshots.append((loss, dev_loss))
        #torch.save(model.state_dict(), 'models/3x3-2-256-2loss_{}.md'.format(dev_loss))

    return model, loss, dev_loss
