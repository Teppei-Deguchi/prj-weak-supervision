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
Calculate accuracy of computational value, and ML model trained only conputationa data on enzymatic activity.
~~~
python3 calc_eval_enzyme.py --res_number residue_number --model_type model_type(svr or rfr) --C float(if model_type=svr) --gamma float(if model_type=svr) --epsilon float(if model_type=svr) --n_estimaters int(if model_type=rfr) --metric metric --sequence_embedding amino_acid_embedding --dataset dataset --outputfile output_filename
~~~

