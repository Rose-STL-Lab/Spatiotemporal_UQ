import numpy as np
import load_data as ld
import convlstm as md
import torch
import convlstm_training as tr
import evaluation as ev
import os

def main():
    os.mkdir('./models')
    os.mkdir('../loss_maemis')
    for k in range (10):
        # Random seed
        random_seed = k
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        np.random.seed(random_seed)
        # Create directory
        directory = 'seed'+str(random_seed)
        path = os.path.join('./', directory) 
        os.mkdir(path)
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

        model = md.ConvLSTMForecast2L((21, 31), 256, 3, 1).cuda() #256
        snapshots = []
        losses = []
        dev_losses = []
        test_losses = []

        for i in range (50): 
            model, loss, dev_loss = tr.train(
                model, input_seqs, target_meo_seqs, dev_input_seqs, dev_target_meo_seqs, 
                snapshots, iterations=1, lr=0.001)

            test_loss = ev.compute_dev_set_loss(
                model,
                test_input_seqs,
                test_target_meo_seqs)

            losses.append(loss)
            dev_losses.append(dev_loss)
            test_losses.append(test_loss)

            print('Epoch: {}, Train loss: {}, Dev loss: {}, Test loss: {}'.format(
                i, loss, dev_loss, test_loss))

        outputs = ev.prediction(model,test_input_seqs)
        np.save(os.path.join(path,'maemis_epo{}.npy'.format(i)), outputs)
        print('Predictions saved as epo{}.npy'.format(i))

        min_val_loss = float('inf')
        wait = 0
        patience = 10
        for i in range (50): 
            model, loss, dev_loss = tr.train(
                model, input_seqs, target_meo_seqs, dev_input_seqs, dev_target_meo_seqs, 
                snapshots, iterations=1, lr=0.0001)

            test_loss = ev.compute_dev_set_loss(
                model,
                test_input_seqs,
                test_target_meo_seqs)

            losses.append(loss)
            dev_losses.append(dev_loss)
            test_losses.append(test_loss)
            
            print('Epoch: {}, Train loss: {}, Dev loss: {}, Test loss: {}'.format(
                i+50, loss, dev_loss, test_loss))

            if dev_loss < min_val_loss:
                wait = 0
                min_val_loss = dev_loss
            elif dev_loss >= min_val_loss:
                wait += 1
                if wait == patience:
                    outputs = ev.prediction(model,test_input_seqs)
                    np.save(os.path.join(path,'maemis_epo{}.npy'.format(i+50)), outputs)
                    print('Predictions saved as epo{}.npy'.format(i+50))
                    print('Early stopping at epoch: {}'.format(i+50))
                    break
            if (i+1)%10 == 0:
                outputs = ev.prediction(model,test_input_seqs)
                np.save(os.path.join(path,'maemis_epo{}.npy'.format(i+50)), outputs)
                print('Predictions saved as epo{}.npy'.format(i+50))

        torch.save(model.state_dict(), 'models/seed{}_epo{}.md'.format(k,i))
        losses = np.array(losses)
        dev_losses = np.array(dev_losses)
        test_losses = np.array(test_losses)
        np.save('../loss_maemis/seed{}.npy'.format(k), np.stack([losses,dev_losses,test_losses]))

if __name__ == "__main__":
    main()
