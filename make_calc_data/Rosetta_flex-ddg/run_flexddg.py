import os
import csv
import re
import argparse
from run_example import flex_ddg
from analyze_flex_ddG import ddg_analysis

parser = argparse.ArgumentParser(description='Command line argument examples')
parser.add_argument('--input_file', type=str)
parser.add_argument('--pdb', type=str)
parser.add_argument('--target_chain', type=str)
parser.add_argument('--rosetta_path', type=str)
args = parser.parse_args()

input_file = args.input_file
pdb = args.pdb
target_chain = args.target_chain
rosetta_path = args.rosetta_path

data = []
pattern = re.compile(r'^\s*([A-Za-z])\s*(\d+)\s*([A-Za-z])\s*$')

with open(input_file, 'r', newline='') as file:
    csv_reader = csv.reader(file)
    header = next(csv_reader, None) 
    mut_idx = 0
    if header:
        for j, name in enumerate(header):
            if str(name).strip().lower() == "mutation":
                mut_idx = j
                break

    for row in csv_reader:
        if not row:
            continue
        txt = row[mut_idx]
        m = pattern.match(txt)
        if not m:
            print(f"Skipping malformed mutation: {txt}")
            continue
        wt, pos, mut = m.groups()
        data.append([int(pos), mut.upper(), wt.upper()]) 

num_rows = len(data)
output_directory = 'inputs'
os.makedirs(output_directory, exist_ok=True)

for i in range(num_rows):
    pos = data[i][0]       
    mut = data[i][1]       
    wt  = data[i][2]       
    basename = f"{wt}{pos}{mut}"

    resfile_path = os.path.join(output_directory, f"{basename}.resfile")
    with open(resfile_path, 'w') as resfile:
        resfile.write("NATAA\n")
        resfile.write("start\n")
        resfile.write(f"{pos} {target_chain} PIKAA {mut}\n")

    # flex ddg calculation
    flex_ddg(basename,rosetta_path,pdb,target_chain)

    # get ddg value
    ddg_analysis(f"output/{basename}", basename)

