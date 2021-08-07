#!/bin/bash
# this line to generate result using run_demo_pytorch.py
python3 run_demo_pytorch.py --config_filename=data/model/pretrained/METR-LA/config.yaml --loading_epoch=`ls -1 ./models | sort -n -k1.4 | tail -1`

# please specify your own folder to move dcrnn_predictions.npz
# the following code will automatically assign dcrnn_predictions1.npz, dcrnn_predictions2.npz, dcrnn_predictions3.npz, ...
# incrementally to the desired folder
mv data/dcrnn_predictions.npz ../CI_plotting/data/bootstrap/`ls ../CI_plotting/data/bootstrap/*.npz | wc -l`.npz


