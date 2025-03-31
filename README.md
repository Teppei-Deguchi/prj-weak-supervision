# Data-efficient protein mutational effect prediction with weak supervision by molecular simulation and protein language models

## Required packages
The following is the required package and programs

* python (3.10.0)  
* scikit-learn (1.5.2)  
* numpy (1.26.4)  
* pytorch (2.5.1)  

## Scripts
Calculate model accuracy with single residue mutants. For binding affinity and protein abundance.
~~~
python3 single_ML_eval.py --res_number residue_number --model_type model_type(svr or rfr) --function_type function_type(binding or abundance) --C float(if model_type=svr) --gamma float(if model_type=svr) --epsilon float(if model_type=svr) --n_estimaters int(if model_type=rfr) --metric metric --sequence_embedding amino_acid_embedding --dataset dataset --outputfile output_filename  --split_data_b int --split_data_e int
~~~
Calculate model accuracy with double residue mutants. For binding affinity and protein abundance.
~~~
python3 double_ML_eval.py --res_number residue_number --model_type model_type(svr or rfr) --function_type function_type(binding or abundance) --C float(if model_type=svr) --gamma float(if model_type=svr) --epsilon float(if model_type=svr) --n_estimaters int(if model_type=rfr) --metric metric --sequence_embedding amino_acid_embedding --dataset dataset --test_sequence_embedding sequence_embedding(double residue dataset) --test_dataset dataset(double residue dataset) --outputfile output_filename  --split_data_b int --split_data_e int
~~~
Calculate model accuracy on enzymatic activity
~~~
python3 enzyme_ML_eval.py --res_number residue_number --model_type model_type(svr or rfr) --function_type function_type (binding or abundance) --C float(if model_type=svr) --gamma float(if model_type=svr) --epsilon float(if model_type=svr) --n_estimaters int(if model_type=rfr) --metric metric --sequence_embedding amino_acid_embedding --dataset dataset --outputfile output_filename  --split_data_b int --split_data_e int
~~~

Calculate accuracy of computational value, and ML model trained only conputationa data with single residue mutants.
~~~
python3 calc_eval_single.py --res_number residue_number --model_type model_type(svr or rfr) --function_type function_type(binding or abundance) --C float(if model_type=svr) --gamma float(if model_type=svr) --epsilon float(if model_type=svr) --n_estimaters int(if model_type=rfr) --metric metric --sequence_embedding amino_acid_embedding --dataset dataset --outputfile output_filename
~~~
Calculate accuracy of computational value with double residue mutants.
~~~
python3 calc_eval_double.py --res_number residue_number --function_type function_type(binding or abundance) --metric metric --test_dataset dataset(double residue dataset) --outputfile output_filename
~~~
Calculate accuracy of ML model trained only conputationa data with double residue mutants.
~~~
python3 calc_ML_double.py  --res_number residue_number --model_type model_type(svr or rfr) --function_type function_type(binding or abundance) --C float(if model_type=svr) --gamma float(if model_type=svr) --epsilon float(if model_type=svr) --n_estimaters int(if model_type=rfr) --metric metric --sequence_embedding amino_acid_embedding --dataset dataset --test_sequence_embedding sequence_embedding(double residue dataset) --test_dataset dataset(double residue dataset) --outputfile output_filename
~~~
Calculate model accuracy on enzymatic activity
~~~
python3 calc_eval_enzyme.py --res_number residue_number --model_type model_type(svr or rfr) --C float(if model_type=svr) --gamma float(if model_type=svr) --epsilon float(if model_type=svr) --n_estimaters int(if model_type=rfr) --metric metric --sequence_embedding amino_acid_embedding --dataset dataset --outputfile output_filename
~~~

Make amino acid embedding
~~~
python3 make_embedding_descriptor.py --input_csv input_filename --output_pkl output_filename --descriptor_type str #descriptor
python3 make_embedding_esm2.py --input_csv input_filename --output_pkl output_filename #ESM-2
~~~

