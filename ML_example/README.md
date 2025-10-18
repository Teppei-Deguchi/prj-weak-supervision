# Machine learning calculation example
ML calculation with APH(3')II enzymatic activity (dataset_activity) and GRB2-SH3 Protein-Protein binding affinity dataset (dataset_PPB).

### 1. Make required directories
~~~
mkdir embedding
mkdir result
~~~

### 2. Make amino acid embedding
~~~
dataset_dir="dataset_activity" 
for i in test train_exp train_calc
do
python3 make_embedding_esm2.py --input_csv ${dataset_dir}/${i}.csv --output_pkl embedding/${i}.pkl
done
~~~

### 3. Conduct ML calculation
Main ML calculation can be done by following command. If you have experimental data for test data, and want to evaluate model accuracy, set eval_testdata=yes. Sequence-activity dataset file like dataset_**/test.csv is required.
~~~
python3 run_ML.py  --res_number protein_residue_length --model_type model_type(svr or rfr) --C float(if model_type=svr) --epsilon float(if model_type=svr) --gamma float(if model_type=svr) --n_estimaters int(if model_type=rfr) --metric metric --sequence_embedding embedding_directory --dataset_dir dataset_directory --calc_mode calculation mode (rosetta, esm2, or hybrid) --eval_testdata (yes or no)

(example enzymatic activity) python3 run_ML.py  --res_number 255 --model_type svr --C 1 --epsilon 1e-06 --gamma 1e-06 --metric spearman --sequence_embedding embedding --dataset_dir dataset_activity --calc_mode hybrid --PPI no --eval_testdata yes
(example PPB) python3 run_ML.py  --res_number 56 --model_type svr --C 1000 --epsilon 0.0001 --gamma 0.0001 --metric spearman --sequence_embedding embedding --dataset_dir dataset_PPB --calc_mode hybrid --PPI yes --eval_testdata yes

~~~

