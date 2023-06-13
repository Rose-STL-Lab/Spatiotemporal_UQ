## Quantifying Uncertainty in Deep Spatiotemporal Forecasting
## Paper: 
Dongxia Wu, Liyao Gao, Xinyue Xiong, Matteo Chinazzi, Alessandro Vespignani, Yi-An Ma, Rose Yu, [Quantifying Uncertainty in Deep Spatiotemporal Forecasting](https://arxiv.org/abs/2105.11982), KDD 2021

## Requirements
* torch
* scipy>=0.19.0
* numpy>=1.12.1
* pandas>=0.19.2
* pyyaml
* statsmodels
* tensorflow>=1.3.0
* torch
* tables
* future

Dependency can be installed using the following command:
```bash
pip install -r requirements.txt
```

## [Dataset](https://drive.google.com/drive/folders/102QfowJq7zmyR3W5LjF1K_eNZ0eAA8RL?usp=sharing)
1. Download DeepGLEAM/autoreg_shuffled_data to DeepGLEAM/*uq_model*/data folder (*uq_model* includes dropout_model, maemis_model, quantile_model, and sq_model). 
2. Download traffic/METR-LA to traffic/*uq_model*/data folder (*uq_model* includes dropout_model, maemis_model, quantile_model, and sq_model). 
3. Download convLSTM_PM2_5/data to convLSTM_PM2_5 folder.


## Abstract:
Deep learning is gaining increasing popularity for spatiotemporal forecasting. However, prior works have mostly focused on point estimates without quantifying the uncertainty of the predictions. In high stakes domains, being able to generate probabilistic forecasts with confidence intervals is critical to risk assessment and decision making. Hence, a systematic study of uncertainty quantification (UQ) methods for spatiotemporal forecasting is missing in the community. In this paper, we describe two types of spatiotemporal forecasting problems: regular grid-based and graph-based. Then we analyze UQ methods from both the Bayesian and the frequentist point of view, casting in a unified framework via statistical decision theory. Through extensive experiments on real-world road network traffic, epidemics, and air quality forecasting tasks, we reveal the statistical and computational trade-offs for different UQ methods: Bayesian methods are typically more robust in mean prediction, while confidence levels obtained from frequentist methods provide more extensive coverage over data variations. Computationally, quantile regression type methods are cheaper for a single confidence interval but require re-training for different intervals. Sampling based methods generate samples that can form multiple confidence intervals, albeit at a higher computational cost.


## Description
1. DeepGLEAM/: Five UQ methods for experiments on epidemics forecasting tasks.
2. traffic/: Five UQ methods for experiments on road network traffic forecasting tasks.
3. convLSTM_PM25/: Five UQ methods for experiments on air quality forecasting tasks.


## Model Training and Evaluation
*uq_model* includes dropout_model, maemis_model, quantile_model, sq_model, and sgmcmc_model.
```bash
# DeepGLEAM
cd DeepGLEAM/uq_model
python dcrnn_train_pytorch.py --config_filename=data/model/dcrnn_cov.yaml
./test.sh

# traffic
cd traffic/uq_model
python dcrnn_train_pytorch.py --config_filename=data/model/dcrnn_la.yaml
python run_demo_pytorch.py

# air quality
cd convLSTM_PM2_5/uq_model
python main.py
```


## Cite
```
@article{wu2021quantifying,
  title={Quantifying Uncertainty in Deep Spatiotemporal Forecasting},
  author={Wu, Dongxia and Gao, Liyao and Xiong, Xinyue and Chinazzi, Matteo and Vespignani, Alessandro and Ma, Yi-An and Yu, Rose},
  journal={Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery \& Data Mining},
  year={2021}
}
```
