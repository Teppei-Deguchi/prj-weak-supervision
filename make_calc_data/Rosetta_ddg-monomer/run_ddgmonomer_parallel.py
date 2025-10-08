import os
import subprocess
import pandas as pd
from multiprocessing import Pool
import shutil
import argparse

curr_dir = os.getcwd()
parser = argparse.ArgumentParser(description='Command line argument examples')
parser.add_argument('--input_file', type=str,help="input mutational information file")
parser.add_argument('--minimized_pdb', type=str,help="input energy minimized pdb file")
parser.add_argument('--select_protocol', type=str, choices=["high-resolution", "low-resolution"], help="ddG monomer protocol")
parser.add_argument('--rosetta_path', type=str,help="set rosett path")
args = parser.parse_args()
input_file = args.input_file
minimized_pdb = args.minimized_pdb
ROSETTA_PATH = args.rosetta_path

pdb_file = curr_dir + "/" + minimized_pdb
cst_file = curr_dir + "/ca_dist_restraints_clean.cst"
ddg_exec = ROSETTA_PATH + "/main/source/bin/ddg_monomer.linuxgccrelease"
mut_csv = curr_dir + "/" + input_file
nstruct = 50
cores = 32

output_dir = curr_dir + "/calculation_result"
os.makedirs(output_dir, exist_ok=True)

mutants_df = pd.read_csv(mut_csv, skiprows=1, header=None, names=["mutation"])  

#get mutation information (e.g. L10A → wt_residue = L, position = 10, mutant = A)
mutants_df[["wt_residue", "position", "mutant"]] = (
    mutants_df["mutation"].str.extract(r"([A-Z])(\d+)([A-Z])")
)

mutants_df["position"] = mutants_df["position"].astype(int)

df = mutants_df

def run_ddg(row):
    original_position = int(row['position'])
    corrected_position = original_position
    wt_residue = row['wt_residue']
    mutant = row['mutant']
    mut_id = f"{wt_residue}{original_position}{mutant}"
    mut_dir = os.path.join(output_dir, mut_id)
    os.makedirs(mut_dir, exist_ok=True)

    # make Resfile
    resfile_path = os.path.join(mut_dir, "mut.resfile")
    print("resfile_path",resfile_path)
    with open(resfile_path, "w") as f:
        f.write("total 1\n")
        f.write("1\n")
        f.write(f"{wt_residue} {corrected_position} {mutant}\n") 

    res_dir = curr_dir + "/output"
    
    #command for hihg resolution protocol
    cmd_high_res = [
            ddg_exec,
        "-in:file:s", pdb_file,
        "-ddg:weight_file", "soft_rep_design",
        "-database", ROSETTA_PATH + "/main/database/",
        "-ddg::iterations", str(nstruct),
        "-ddg::dump_pdbs", "true",
        "-ddg:mut_file", resfile_path,
        "-ddg::local_opt_only", "false",
        "-ddg::min_cst", "true",
        "-constraints::cst_file", cst_file,
        "-ddg::suppress_checkpointing", "true",
        "-multithreading::total_threads", "0",
        "-in::file::fullatom",
        "-ddg::mean", "false",
        "-ddg::min", "true",
        "-ddg::sc_min_only", "false",
        "-ddg::ramp_repulsive", "true",
        "-out:file:scorefile", os.path.join(mut_dir, "ddg.sc"),
        "-ddg::output_silent", "true",
        "-out:path:pdb", os.path.join(res_dir,f"{mut_id}.out")
    ]

    #command for low resolution protocol
    cmd_low_res = [
            ddg_exec,
        "-in:file:s", pdb_file,
        "-ddg:weight_file", "soft_rep_design",
        "-database", ROSETTA_PATH +"/main/database/",
        "-ddg::iterations", str(nstruct),
        "-ddg::dump_pdbs", "true",
        "-ddg:mut_file", resfile_path,
        "-ddg::local_opt_only", "true",
        "-ddg::min_cst", "true",
        "-constraints::cst_file", cst_file,
        "-ddg::suppress_checkpointing", "true",
        "-multithreading::total_threads", "0",
        "-in::file::fullatom",
        "-ddg::mean", "true",
        "-ddg::min", "false",
        "-out:file:scorefile", os.path.join(mut_dir, "ddg.sc"),
        "-ddg::output_silent", "true",
        "-out:path:pdb", os.path.join(res_dir,f"{mut_id}.out")
    ]
    
    if args.select_protocol == "high-resolution":
        cmd = cmd_high_res
    elif args.select_protocol == "low-resolution":
        cmd = cmd_low_res

    #ddg monomer calculation
    try:
        subprocess.run(cmd, check=True, cwd=mut_dir)
    except subprocess.CalledProcessError as e:
        print(f"Error in mutation {mut_id}: {e}")

# parallel calculation
if __name__ == "__main__":
    with Pool(cores) as p:
        p.map(run_ddg, [row for _, row in df.iterrows()])

