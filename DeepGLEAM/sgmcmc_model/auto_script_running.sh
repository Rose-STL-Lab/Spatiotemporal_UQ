#!/bin/bash
rm -rf models
python3 dcrnn_train_pytorch.py --config_filename=data/model/dcrnn_cov.yaml
bash auto_result_collection.sh

rm -rf models
python3 dcrnn_train_pytorch.py --config_filename=data/model/dcrnn_cov.yaml
bash auto_result_collection.sh

