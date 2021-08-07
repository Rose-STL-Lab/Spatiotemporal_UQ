import numpy as np
import load_data as ld
import convlstm as md
import torch
import convlstm_training as tr
import evaluation as ev
import torch.nn as nn
from sgnht import Sampler

# Random seed

def main():
    # Training data
    grid_seqs = ld.load_batch_seq_data()
    input_seqs, target_meo_seqs, _, _ = tr.seq_preprocessing(grid_seqs)
    # Dev data
    dev_grid_seqs = ld.load_batch_dev_seq_data()
    dev_input_seqs, dev_target_meo_seqs, _, _ = \
        tr.seq_preprocessing(dev_grid_seqs)
    # Test data
    test_grid_seqs = ld.load_batch_test_seq_data()
    test_input_seqs, test_target_meo_seqs, avg_grid, std_grid = \
        tr.seq_preprocessing(test_grid_seqs)
    np.save('avg.npy', avg_grid)
    np.save('std.npy', std_grid)

    model = md.ConvLSTMForecast2L((21, 31), 256, 3, 1).cuda() #256
    snapshots = []
    losses = []
    dev_losses = []
    test_losses = []
    sampler = Sampler(model, nn.L1Loss(), momentum=0.9, lr=0.0005, wdecay=0.0, total=10000, A = 10.0)
    for i in range (60): #20
        model, loss, dev_loss = tr.train(
            model, input_seqs, target_meo_seqs, dev_input_seqs, dev_target_meo_seqs, 
            snapshots, iterations=1, lr=0.001, sampler=sampler)

        test_loss = ev.compute_dev_set_loss(
            model,
            test_input_seqs,
            test_target_meo_seqs)

        losses.append(loss)
        dev_losses.append(dev_loss)
        test_losses.append(test_loss)

        print('Epoch: {}, Train loss: {}, Dev loss: {}, Test loss: {}'.format(
            i, loss, dev_loss, test_loss))
        if i >= 50:
            outputs = ev.prediction(model,test_input_seqs)
            outputs = {'prediction': outputs, 'truth': test_target_meo_seqs}
            np.savez_compressed('./models/point_epo{}.npz'.format(i), **outputs)
            print('Predictions saved as epo{}.npz'.format(i))

    # for i in range (70): #50
    #     model, loss, dev_loss = tr.train(
    #         model, input_seqs, target_meo_seqs, dev_input_seqs, dev_target_meo_seqs, 
    #         snapshots, iterations=1, lr=0.0001)

    #     test_loss = ev.compute_dev_set_loss(
    #         model,
    #         test_input_seqs,
    #         test_target_meo_seqs)

    #     losses.append(loss)
    #     dev_losses.append(dev_loss)
    #     test_losses.append(test_loss)

    #     print('Epoch: {}, Train loss: {}, Dev loss: {}, Test loss: {}'.format(
    #         i, loss, dev_loss, test_loss))
    #     if (i+1)%10 == 0:
    #         outputs = ev.prediction(model,test_input_seqs)
    #         np.save('./point_epo{}.npy'.format(i+120), outputs)
    #         print('Predictions saved as epo{}.npy'.format(i+120))

    losses = np.array(losses)
    dev_losses = np.array(dev_losses)
    test_losses = np.array(test_losses)
    np.save('loss.npy', np.stack([losses,dev_losses,test_losses]))

if __name__ == "__main__":
    main()
