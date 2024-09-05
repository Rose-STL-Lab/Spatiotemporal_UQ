from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import yaml
import random

from lib.utils import load_graph_data
from model.pytorch.dcrnn_supervisor import DCRNNSupervisor

import numpy as np
import os

def main(args):
    with open(args.config_filename) as f:
        supervisor_config = yaml.load(f)

        data_path = os.path.join(supervisor_config['data'].get('dataset_dir'), 'train.npz')
        data = np.load(data_path)
        for i in range (1):
            index = np.random.choice(25, 23, replace=True)
            np.savez_compressed(
                        os.path.join(supervisor_config['data'].get('dataset_dir'), "train_"+str(i)+".npz"),
                        x=data['x'][index],
                        y=data['y'][index],
                        x_offsets=data['x_offsets'],
                        y_offsets=data['y_offsets'],
                    )

        graph_pkl_filename = supervisor_config['data'].get('graph_pkl_filename')
        sensor_ids, sensor_id_to_ind, adj_mx = load_graph_data(graph_pkl_filename)
        seed = random.randint(0, 1000)
        supervisor = DCRNNSupervisor(random_seed = seed, adj_mx=adj_mx, **supervisor_config)
        supervisor.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', default=None, type=str,
                        help='Configuration filename for restoring the model.')
    parser.add_argument('--use_cpu_only', default=False, type=bool, help='Set to true to only use cpu.')
    args = parser.parse_args()
    main(args)
