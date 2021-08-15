#!/bin/bash
rm -rf models
python3 dcrnn_train_pytorch.py --config_filename=data/model/dcrnn_cov_week33.yaml
bash auto_result_collection_week33.sh

rm -rf models
python3 dcrnn_train_pytorch.py --config_filename=data/model/dcrnn_cov_week33.yaml
bash auto_result_collection_week33.sh

