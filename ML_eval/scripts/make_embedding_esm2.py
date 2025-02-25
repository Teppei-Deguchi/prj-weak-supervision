import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import pickle
import esm
import os
import pandas as pd

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()

def batch_split(data, batch_size=32):
    i = 0
    while i < len(data):
        yield data[i:min(i+batch_size, len(data))]
        i += batch_size

def process_data(data,batch_size=32):
    embeddings = []
    batch_com = []
    for batch in batch_split(data, batch_size=batch_size):
        model.cuda()
        batch_converter = alphabet.get_batch_converter()
        model.eval()
        batch = [(protein_num, protein_seq) for protein_num,protein_seq in batch]
        batch_labels, batch_strs,batch_tokens = batch_converter(batch)
        batch_tokens = batch_tokens.to(device="cuda", non_blocking=True)
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[6])
        embeddings_tensor = torch.Tensor(results["representations"][6].detach().cpu())
        embeddings_list = embeddings_tensor.tolist()
        embeddings = embeddings + embeddings_list
    return embeddings

def process_alldata(input_file):
    csv_path = os.path.join(input_file)
    df = pd.read_csv(csv_path, header=0)
    num = range(len(df))
    id_smi_seq = list(zip(num,df['sequence']))
    return id_smi_seq

parser = argparse.ArgumentParser(description="Amino acid embedding with ESM-2")
parser.add_argument("--input_csv", type=str, required=True, help="Path to input csv file")
parser.add_argument("--output_pkl", type=str, required=True, help="Path to output pkl file")
args = parser.parse_args()

data = process_alldata(args.input_file)
p_data = process_data(data)
with open(args.output_file,'wb') as file:
    pickle.dump(p_data, file)
