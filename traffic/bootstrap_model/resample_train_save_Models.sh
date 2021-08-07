#!/bin/bash
# Since Bootstrap requires the inference under randomly resampled data, 
# the following code will repeat the 4 steps described below. 

# First time: 
# 1: removed previously trained models
rm -rf models
# 2: resample (in Bootstrap steps)
python3 data/METR-LA/sample_train.py
# 3: train model
python3 dcrnn_train_pytorch.py --config_filename=data/model/dcrnn_la.yaml
# 4: collect trained models to desired folder 
bash auto_result_collection.sh

# Second time: 
rm -rf models
python3 data/METR-LA/sample_train.py
python3 dcrnn_train_pytorch.py --config_filename=data/model/dcrnn_la.yaml
bash auto_result_collection.sh

# Third time: 
rm -rf models
python3 data/METR-LA/sample_train.py
python3 dcrnn_train_pytorch.py --config_filename=data/model/dcrnn_la.yaml
bash auto_result_collection.sh

# 4th time: 
rm -rf models
python3 data/METR-LA/sample_train.py
python3 dcrnn_train_pytorch.py --config_filename=data/model/dcrnn_la.yaml
bash auto_result_collection.sh

# 5th time: 
rm -rf models
python3 data/METR-LA/sample_train.py
python3 dcrnn_train_pytorch.py --config_filename=data/model/dcrnn_la.yaml
bash auto_result_collection.sh

# 6th time: 
rm -rf models
python3 data/METR-LA/sample_train.py
python3 dcrnn_train_pytorch.py --config_filename=data/model/dcrnn_la.yaml
bash auto_result_collection.sh

# 7th time: 
rm -rf models
python3 data/METR-LA/sample_train.py
python3 dcrnn_train_pytorch.py --config_filename=data/model/dcrnn_la.yaml
bash auto_result_collection.sh
