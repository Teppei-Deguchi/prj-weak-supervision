# Data-efficient protein mutational effect prediction with weak supervision by molecular simulation and protein language models
<img width="5760" height="3953" alt="Image" src="https://github.com/user-attachments/assets/a3992a10-a0ee-4f1b-b90c-4b452f42bf53" />

## Required packages
The following is the required packages and programs

* python (3.10.0)  
* scikit-learn (1.5.2)  
* numpy (1.26.4)  
* torch (2.5.1)
* pandas (2.3.2)
* tqdm (4.67.1)
* bio (1.8.0)
* fair-esm (2.0)
* scikit-learn-intelex (2025.8.0) (optional) 

## ML example
The directory contains example scripts for users to perform machine learning combining their experimental data and weak training data prepared by the protocol given in make_calc_data directory below.

## make_calc_data
The directory contains example scripts for preparing weak training data using Rosetta and ESM-2 zero-shot prediction.

## Benchmark
The directory contains scripts and datasets to reproduce the results reported in [Deguchi et al. BioRxiv. 2025](https://www.biorxiv.org/content/10.1101/2025.04.08.647800v2).
