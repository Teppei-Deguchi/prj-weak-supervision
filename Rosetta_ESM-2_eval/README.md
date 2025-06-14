## Accuracy of ML model trained on only computational data and computational data itself.
Calculate accuracy of computational value, and ML model trained only conputationa data with single residue mutants.
~~~
python3 calc_eval_single.py --res_number residue_number --model_type model_type(svr or rfr) --function_type function_type(binding or abundance) --C float(if model_type=svr) --gamma float(if model_type=svr) --epsilon float(if model_type=svr) --n_estimaters int(if model_type=rfr) --metric metric --sequence_embedding amino_acid_embedding --dataset dataset(ML_eval/single/.csv) --outputfile output_filename
~~~
Calculate accuracy of computational value with double residue mutants.
~~~
python3 calc_eval_double.py --res_number residue_number --function_type function_type(binding or abundance) --metric metric --test_dataset dataset(Rosetta_ESM-2_eval/double_calc_values/.csv) --outputfile output_filename
~~~
Calculate accuracy of ML model trained only conputationa data with double residue mutants.
~~~
python3 calc_ML_double.py  --res_number residue_number --model_type model_type(svr or rfr) --function_type function_type(binding or abundance) --C float(if model_type=svr) --gamma float(if model_type=svr) --epsilon float(if model_type=svr) --n_estimaters int(if model_type=rfr) --metric metric --sequence_embedding amino_acid_embedding --dataset dataset(ML_eval/single/.csv) --test_sequence_embedding test_amino_acid_embedding --test_dataset dataset(ML_eval/double/.csv) --outputfile output_filename
~~~
Calculate accuracy of computational value, and ML model trained only conputationa data on enzymatic activity.
~~~
python3 calc_eval_enzyme.py --res_number residue_number --model_type model_type(svr or rfr) --C float(if model_type=svr) --gamma float(if model_type=svr) --epsilon float(if model_type=svr) --n_estimaters int(if model_type=rfr) --metric metric --sequence_embedding amino_acid_embedding --dataset dataset(ML_eval/single/.csv) --outputfile output_filename
~~~

