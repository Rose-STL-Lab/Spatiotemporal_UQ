#!/bin/bash
python3 dcrnn_train_pytorch.py --config_filename=data/model/dcrnn_la.yaml
bash auto_result_collection.sh
