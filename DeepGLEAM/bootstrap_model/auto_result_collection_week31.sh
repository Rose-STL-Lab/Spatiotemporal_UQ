#!/bin/bash
python3 run_demo_pytorch.py --config_filename=data/model/pretrained/COV-19/config_week31.yaml --loading_epoch=`ls -1 ./models_seed0 | sort -n -k1.4 | tail -1`
mv data/alltest_deepgleam_point_week33_seed1_epo69.npz ../../ensemble_model/analysis/week31/autoreg_bootstrap/`ls ../../ensemble_model/analysis/week31/autoreg_bootstrap/*.npz | wc -l`.npz

