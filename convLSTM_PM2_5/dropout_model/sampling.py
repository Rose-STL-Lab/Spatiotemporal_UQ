import numpy as np
import load_data as ld
import convlstm as md
import torch
import convlstm_training as tr
import evaluation as ev
import os

def main():
    os.mkdir('./dropout_pred')
    path = './dropout_pred'
    # Test data
    test_grid_seqs = ld.load_batch_test_seq_data()
    test_input_seqs, test_target_meo_seqs, avg_grid, std_grid = \
        tr.seq_preprocessing(test_grid_seqs)
    for k in range (10):
        # Random seed
        random_seed = k
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        np.random.seed(random_seed)
        # Create directory
        # directory = 'seed'+str(random_seed)
        # path = os.path.join('./sampling', directory) 
        # os.mkdir(path)
        path1 = os.path.join('./models_sampling', str(random_seed)+'.md')
        model = md.ConvLSTMForecast2L((21, 31), 256, 3, 1).cuda() #256
        model.load_state_dict(torch.load(path1))

        #output list for 50 samples
        outputs = []
        for i in range (50): 
            output = ev.prediction(model,test_input_seqs)
            outputs.append(output[:,:,-1:])

        output_ensemble = np.concatenate(outputs, axis=2)
        mean = np.mean(output_ensemble,axis=2)
        std = np.std(output_ensemble,axis=2)
        print(mean.shape)
        print(std.shape)
        np.save(os.path.join(path,'mean{}.npy'.format(k)), mean)
        np.save(os.path.join(path,'std{}.npy'.format(k)), std)
        print('Predictions saved as mean{}.npy'.format(k))
        print('Predictions saved as std{}.npy'.format(k))

if __name__ == "__main__":
    main()
