import numpy as np
import torch
import torch.nn as nn

device = torch.device('cuda')

def feed_model_data_dev(model, grid_data_seq):
    hidden_states = None

    _, hidden_states = model(grid_data_seq, hidden_states)

    return model, hidden_states

def generate_forecasts_dev(model, hidden_states, init_grid_data, seq_len=48):
    prev_grid = init_grid_data
    meo_forecasts = []

    for i in range(seq_len):
        prev_grid, hidden_states = model(prev_grid[:,:,:6], hidden_states)
        meo_forecasts.append(prev_grid)

    return torch.cat(meo_forecasts, 1)

def maemis_loss(y_pred, y_true):
    pho = 0.05
    loss0 = torch.abs(y_pred[:,:,0] - y_true)
    loss1 = torch.max(y_pred[:,:,2]-y_pred[:,:,1],torch.tensor([0.]).cuda())
    loss2 = torch.max(y_pred[:,:,1]-y_true,torch.tensor([0.]).cuda())*2/pho
    loss3 = torch.max(y_true-y_pred[:,:,2],torch.tensor([0.]).cuda())*2/pho
    loss = loss0+loss1+loss2+loss3
    loss[loss != loss] = 0
    return loss.mean()

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
            loss_pm = maemis_loss(meo_forecasts[:,:,5:], target_meo[:,:,-1])
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
