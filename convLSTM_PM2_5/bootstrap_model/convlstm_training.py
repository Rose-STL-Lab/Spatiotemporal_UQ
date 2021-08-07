import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import load_data as ld
import convlstm as md
import evaluation as ev
import copy

device = torch.device('cuda')

def feed_model_data(model, grid_data_seq):
    _, hidden_states = model(grid_data_seq)

    return model, hidden_states

def generate_forecasts(model, hidden_states, init_grid_data, seq_len=48):
    prev_grid = init_grid_data
    meo_forecasts = []

    for i in range(seq_len):
        prev_grid, hidden_states = model(prev_grid, hidden_states)
        meo_forecasts.append(prev_grid)

    return torch.cat(meo_forecasts, 1)

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
    # loss_mask = torch.tensor(
    #     [0.2,0.2,0.2,0.2,0.2,1.],
    #     dtype=torch.float32,
    #     device=device,
    # )
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
            loss_pm = loss_function(meo_forecasts[:,:,5:], target_meo[:,:,5:])
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
