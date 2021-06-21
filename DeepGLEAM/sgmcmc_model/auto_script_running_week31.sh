#!/bin/bash
rm -rf models
python3 dcrnn_train_pytorch.py --config_filename=data/model/dcrnn_cov_week31.yaml
bash auto_result_collection_week31.sh

rm -rf models
python3 dcrnn_train_pytorch.py --config_filename=data/model/dcrnn_cov_week31.yaml
bash auto_result_collection_week31.sh

rm -rf models
python3 dcrnn_train_pytorch.py --config_filename=data/model/dcrnn_cov_week31.yaml
bash auto_result_collection_week31.sh


