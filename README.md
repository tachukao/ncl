# Natural continual learning

This repository contains an implementation of natural continual learning (NCL) for recurrent networks develop in the following paper:

```
@inproceedings{
    kao2021natural,
    title = {Natural continual learning: success is a journey, not (just) a destination},
    author = {Ta-Chu Kao and Kristopher T Jensen and Gido Martijn van de Ven and Alberto Bernacchia and Guillaume Hennequin},
    booktitle = {Thirty-Fifth Conference on Neural Information Processing Systems},
    year = {2021},
    url = {https://openreview.net/forum?id=W9250bXDgpK}
}
```

The experiments in feedforward networks were implemented in a fork of the following repository and will be released shortly at this [link](https://github.com/GMvandeVen/continual-learning).

Please feel free to reach out to any of the NCL authors, if you would like access to our feedforward implementation before the public release.

## Getting started
### Extracting SMNIST data 

```sh
cd data
tar -xzvf smnist.tar.gz
```
### Install NCL package

```sh
pip install -r requirements.txt # cpu support
pip install -r requirements-gpu.txt # gpu support
```
### Training models

```sh
mkdir results # create results directory

python run.py --learner ncl --task smnist --train_batch_size 256 --n_hidden 30 --learning_rate 0.01 --results_dir results/ --max_steps 500000 --data_dir data

python run.py --learner ncl --task ryang --train_batch_size 32 --n_hidden 256 --learning_rate 0.001 --results_dir results/ --max_steps 500000
```