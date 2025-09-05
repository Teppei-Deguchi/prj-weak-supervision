# Data-efficient protein mutational effect prediction with weak supervision by molecular simulation and protein language models
<img width="5760" height="3953" alt="Image" src="https://github.com/user-attachments/assets/a3992a10-a0ee-4f1b-b90c-4b452f42bf53" />

## Required packages
The following is the required package and programs

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
This provides exampel script and dataset for our method. APH(3')II enzymatic activity and GRB2-SH3 binding affinity dataset is available.

## make_calc_data
This provide a consistent protocol that encompasses ESM-2 zero-shot predictions, Rosetta ddG monomer, and Rosetta flex ddG calculations, leading up to the construction of a sequence-calculated value dataset for ML.

## Benchmark
The directory includes scripts and datasets to reproduce the results reported in Deguchi et al. BioRxiv. 2025.
