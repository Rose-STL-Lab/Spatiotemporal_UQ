import argparse
import numpy as np
import os
import sys
import yaml

from lib.utils import load_graph_data
from model.pytorch.dcrnn_supervisor import DCRNNSupervisor


def run_dcrnn(args):
    with open(args.config_filename) as f:
        supervisor_config = yaml.load(f)

        graph_pkl_filename = supervisor_config['data'].get('graph_pkl_filename')
        sensor_ids, sensor_id_to_ind, adj_mx = load_graph_data(graph_pkl_filename)

        #supervisor = DCRNNSupervisor(adj_mx=adj_mx, **supervisor_config)
        #mean_score, outputs = supervisor.evaluate('test')
        #np.savez_compressed(args.output_filename, **outputs)
        #print("MAE : {}".format(mean_score))
        #print('Predictions saved as {}.'.format(args.output_filename))
        resultlist = []
        for i in range (50):
            supervisor = DCRNNSupervisor(adj_mx=adj_mx, **supervisor_config)
            mean_score, outputs = supervisor.evaluate('test')
            output_pred = outputs['prediction']
            resultlist.append(output_pred)
        result = np.stack(resultlist,axis = 0)
        print("SHAPE : {}".format(result.shape))
        print("SHAPE_MEAN : {}".format(np.mean(result,axis=0).shape))
        print("SHAPE_STD : {}".format(np.std(result,axis=0).shape))
        print(repr(np.round(np.mean(result,axis=0)[0][0],2)))
        print(repr(np.round(np.std(result,axis=0)[0][0],2)))
        new_output = {'mean': np.mean(result,axis=0), 'std': np.std(result,axis=0), 'truth': outputs['truth']}
        np.savez_compressed(args.output_filename, **new_output)
        print('Predictions saved as {}.'.format(args.output_filename))

if __name__ == '__main__':
    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cpu_only', default=False, type=str, help='Whether to run tensorflow on cpu.')
    parser.add_argument('--config_filename', default='data/model/pretrained/METR-LA/config.yaml', type=str,
                        help='Config file for pretrained model.')
    parser.add_argument('--output_filename', default='dropout_result/seed0.npz')
    args = parser.parse_args()
    run_dcrnn(args)
