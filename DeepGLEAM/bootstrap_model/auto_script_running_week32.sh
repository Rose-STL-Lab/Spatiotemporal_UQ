#!/bin/bash
rm -rf models_seed0
python3 data/autoreg_shuffled_data/week32/sample_train.py
python3 dcrnn_train_pytorch.py --config_filename=data/model/dcrnn_cov_week32.yaml
bash auto_result_collection_week32.sh

rm -rf models_seed0
python3 data/autoreg_shuffled_data/week32/sample_train.py
python3 dcrnn_train_pytorch.py --config_filename=data/model/dcrnn_cov_week32.yaml
bash auto_result_collection_week32.sh

rm -rf models_seed0
python3 data/autoreg_shuffled_data/week32/sample_train.py
python3 dcrnn_train_pytorch.py --config_filename=data/model/dcrnn_cov_week32.yaml
bash auto_result_collection_week32.sh

rm -rf models_seed0
python3 data/autoreg_shuffled_data/week32/sample_train.py
python3 dcrnn_train_pytorch.py --config_filename=data/model/dcrnn_cov_week32.yaml
bash auto_result_collection_week32.sh

rm -rf models_seed0
python3 data/autoreg_shuffled_data/week32/sample_train.py
python3 dcrnn_train_pytorch.py --config_filename=data/model/dcrnn_cov_week32.yaml
bash auto_result_collection_week32.sh
