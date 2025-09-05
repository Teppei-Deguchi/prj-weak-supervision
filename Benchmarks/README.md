# Data-efficient protein mutational effect prediction with weak supervision by molecular simulation and protein language models

## Scripts
### 1. Make amino acid embedding
~~~
python3 make_embedding_descriptor.py --input_csv input_filename(ML_eval/single or double/.csv) --output_pkl output_filename(.pkl) --feature_type str #descriptor
python3 make_embedding_esm2.py --input_csv input_filename(ML_eval/single or double/.csv) --output_pkl output_filename(.pkl) #ESM-2
~~~

### 2. Conduct ML calculation
Calculate model accuracy with single residue mutants. For binding affinity and protein abundance.
~~~
python3 single_ML_eval.py --res_number protein_residue_length --model_type model_type(svr or rfr) --function_type function_type(binding or abundance) --C float(if model_type=svr) --gamma float(if model_type=svr) --epsilon float(if model_type=svr) --n_estimaters int(if model_type=rfr) --metric metric --sequence_embedding amino_acid_embedding(.pkl) --dataset dataset(ML_eval/single/.csv) --outputfile output_filename  --split_data_b int --split_data_e int
~~~
Calculate model accuracy with double residue mutants. For binding affinity and protein abundance.
~~~
python3 double_ML_eval.py --res_number protein_residue_length --model_type model_type(svr or rfr) --function_type function_type(binding or abundance) --C float(if model_type=svr) --gamma float(if model_type=svr) --epsilon float(if model_type=svr) --n_estimaters int(if model_type=rfr) --metric metric --sequence_embedding amino_acid_embedding(.pkl) --dataset dataset(ML_eval/single/.csv) --test_dataset dataset(ML_eval/double/.csv) --test_sequence_embedding test_amino_acid_embedding(.pkl) --outputfile output_filename  --split_data_b int --split_data_e int
~~~
Calculate model accuracy on enzymatic activity
~~~
python3 enzyme_ML_eval.py --res_number protein_residue_length --model_type model_type(svr or rfr) --function_type function_type (binding or abundance) --C float(if model_type=svr) --gamma float(if model_type=svr) --epsilon float(if model_type=svr) --n_estimaters int(if model_type=rfr) --metric metric --sequence_embedding amino_acid_embedding(.pkl) --dataset dataset(ML_eval/single/.csv) --outputfile output_filename  --split_data_b int --split_data_e int
~~~
In ML calculation, n% of experimental training data is extracted, and 100-n% of experimental training data is replaced with calculated data. split_data_b and split_data_e specify the rate n. The ML calculation is conducted with n in range of split_data_b <= n <= split_data_e.
