import argparse
import os
import pandas as pd
import pickle

def load_feature_dict(feature_file):
    feature_dict = {}
    with open(feature_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) > 1:
                feature_dict[parts[0]] = list(map(float, parts[1:])) 
    return feature_dict

def encode_sequences(input_csv, feature_dict):
    df = pd.read_csv(input_csv, header=0)  
    sequences = df.iloc[:, 0].astype(str).tolist()
    
    encoded_list = []
    for seq in sequences:
        encoded_vector = []
        for char in seq:
            if char in feature_dict:
                encoded_vector.extend(feature_dict[char])
            else:
                raise ValueError(f"Character '{char}' not found in feature dictionary")
        encoded_list.append(encoded_vector)

    return encoded_list

def main():
    parser = argparse.ArgumentParser(description="Convert protein sequences to feature vectors and save as a pkl file.")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--output_pkl", type=str, required=True, help="Path to output pkl file")
    parser.add_argument("--feature_type", type=str, required=True, choices=["BLOSUM", "FASGAI", "MS-WHIM", 
                                                                             "ProtFP-Feature", "ProtFP", 
                                                                             "ST-scale", "T-scale", 
                                                                             "VHSE-4", "VHSE", "Z-scale"], 
                        help="Feature type to use for encoding")
    args = parser.parse_args()

    feature_file = os.path.join("aafeat", f"{args.feature_type}.txt")

    if not os.path.exists(feature_file):
        raise FileNotFoundError(f"Feature file '{feature_file}' not found.")

    feature_dict = load_feature_dict(feature_file)

    encoded_data = encode_sequences(args.input_csv, feature_dict)

    with open(args.output_pkl, 'wb') as f:
        pickle.dump(encoded_data, f)

    print(f"Encoded data successfully saved to {args.output_pkl}")

if __name__ == "__main__":
    main()

