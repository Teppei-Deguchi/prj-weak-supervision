# Make calculated data

## 1.Make mutation data file
Firstly, following commnad will generate all saturation mutations. Mutational data for test data can be excluded if you have. The mutation data will used following steps.
~~~
cd dataset
python make_mut_data.py \
    --wt-seq TYVQALFDFDPQEDGELGFRRGDFIHVMDNSDPNWWKGACHGQTGMFPRNYVTPVN \
    --test test.csv (optional) \
    --output mutation.csv
~~~

## 2.ESM-2 zero-shot prediction
[ESM-2](https://github.com/facebookresearch/esm) zero-shot prediction is done with following command.
~~~
cd ESM-2_zero-shot
python predict.py --model-location esm2_t36_3B_UR50D \
    --sequence TYVQALFDFDPQEDGELGFRRGDFIHVMDNSDPNWWKGACHGQTGMFPRNYVTPVN \
    --dms-input ../dataset/mutation.csv \
    --mutation-col mutation \
    --dms-output ../result/ESM-2_zero-shot_value.csv \
    --offset-idx 1 \
    --scoring-strategy wt-marginals
~~~

## 3.Rosetta ddG monomer calculation
Foldng free energy calcualtion can be done in following step. [Rosetta installation](https://docs.rosettacommons.org/demos/latest/tutorials/install_build/install_build) is required. As a preliminary step, energy minimization and the preparation of the required files are necessary.

Energy minimizatoin and create constraint file.
~~~
cd Rosetta_ddg-monomer

/Rosetta-PATH/main/source/bin/minimize_with_cst.linuxgccrelease -in:file:l protein_list.txt  -in:file:fullatom -ignore_unrecognized_res -fa_max_dis 9.0 -database /ROSETTA-PATH/main/database/ -ddg::harmonic_ca_tether 0.5 -ddg::constraint_weight 1.0 -ddg::out_pdb_prefix input/min_cst_0.5 -ddg::sc_min_only false > cstmin.log

bash ./convert_to_cst_file.sh cstmin.log > ca_dist_restraints.cst

sed -E '/^AtomPair[[:space:]]+CA[[:space:]]+[0-9]+[[:space:]]+CA[[:space:]]+[0-9]+[[:space:]]+HARMONIC[[:space:]]+-?[0-9]+(\.[0-9]+)?[[:space:]]+-?[0-9]+(\.[0-9]+)?$/!d' \
  ca_dist_restraints.cst > ca_dist_restraints_clean.cst
~~~
Run [ddG monomer](https://docs.rosettacommons.org/docs/latest/application_documentation/analysis/ddg-monomer) calculation. Use energy minimized structure. High or Low resolution protocol is defined by --select_protocol. Set ROSETTA_PATH variable in run_ddgmonomer_parallel.py beforehand. 
~~~
python3 run_ddgmonomer_parallel.py \
    --input_file ../dataset/mutation.csv \
    --minimized_pdb input/min_cst_0.5.2vwf_monomer_0001.pdb \
    --select_protocol high-resolution \
    --rosetta_path ROSETTA_PATH
~~~
Collect ddGf values from result files.
~~~
python3 get_ddGf.py
~~~

## 4.Rosetta Flex ddG calculation (Protien-Protein Binding)
Protien-Protein Binding free energy calcualtion with Roestta [Flex ddG](https://github.com/Kortemme-Lab/flex_ddG_tutorial). Complex structure is required on pdb/. The residue numbering in the PDB file must start from 1.
~~~
cd Rosetta_flex-ddg
mkdir output
mkdir ddg_result
mkdir analysis_output
python3 run_flexddg.py　\
    --input_file ../dataset/mutation.csv \
    --pdb pdb/2vwf.pdb \
    --target_chain A \
    --rosetta_path ROSETTA_PATH
~~~
Collect ddGf values from result files.
~~~
python3 get_ddGb.py
~~~

## 5.Combine calculated data
The calculated data is added to training experimental dataset. Calculated dataset for data expansion is also generated. Mutational information is converted to amino-acid sequence.
~~~
cd result
python3 merge_mutantion_data.py --esm2 ESM-2_zero-shot_value.csv --ddgf ddGf_value.csv --ddgb ddGb_value.csv (optional) --wt-seq TYVQALFDFDPQEDGELGFRRGDFIHVMDNSDPNWWKGACHGQTGMFPRNYVTPVN --train_data ../dataset/train_exp.csv
~~~

## Acknowledgement
- [ESM-2](https://github.com/facebookresearch/esm)
- [Rosetta](https://rosettacommons.org/software/)
