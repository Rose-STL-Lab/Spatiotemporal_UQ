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
        prev_grid, hidden_states = model(prev_grid, hidden_states)
        meo_forecasts.append(prev_grid)

    return torch.cat(meo_forecasts, 1)

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
            loss_pm = loss_function(meo_forecasts[:,:,5:], target_meo[:,:,5:])
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
