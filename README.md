# Data-efficient protein mutational effect prediction with weak supervision by molecular simulation and protein language models

## Required packages
The following is the required package and programs

* python (3.10.0)  
* scikit-learn (1.5.2)  
* numpy (1.26.4)  
* pytorch (2.5.1)  

## scripts
~~~
python3 binding_single_SVR_RFR_bind_fold.py --res_number residue_number --model_type model_type(svr or rfr) --function_type function_type (binding or abundance) --n_estimaters n_estimaters(optional) --metric spearman --sequence_embedding amino_acid_embedding --dataset dataset --outputfile output_filename  --split_data_b 1 --split_data_e 100
~~~

