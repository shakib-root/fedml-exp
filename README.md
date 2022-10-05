# fedml-exp

## Clone the Repository
```sh
git clone --recurse-submodules https://github.com/shakib-root/fedml-exp.git
```

## Environment set-up
The code is tested on Ubuntu 20.04 in a conda environment
```sh
$ conda env create -f env.yml
$ conda activate fedml
```

## Features

- Basic Fed-ML simulations
- Fed-ML simulations with SAM optimizer(currently supports Fed-avg algorithm)

## Run Experiments
```sh
$ python basic_runner.py --cf [configuration file path] # Run basic FedML experiments
$ python sam_runner.py --cf [configuration file path]   # Fed-ML simulations with SAM optimizer
```