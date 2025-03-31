import sys
import csv
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer,r2_score, mean_squared_error
from scipy.stats import spearmanr, pearsonr
from scipy.optimize import curve_fit
import pickle
import torch
import subprocess
import argparse
import concurrent.futures
import os

def pearsonr_metric(y_true, y_pred):
    r = pearsonr(x=y_true, y=y_pred)
    return r[0] 

def spearmanr_metric(y_true, y_pred):
    r = spearmanr(a=y_true, b=y_pred)
    return r[0] 

def set_scoring(metric):
    if metric == 'r2':
        return 'r2'
    elif metric == 'rmse':
        return 'neg_root_mean_squared_error'
    elif metric == 'pearson':
        return make_scorer(pearsonr_metric)
    elif metric == 'spearman':
        return make_scorer(spearmanr_metric)
    else:
        print('wrong metric', metric)
        exit()

def fitting_function_linr(x, a, b):
   return a*x + b

def fitting_function_linr_two(xy, a, b,c):
   x, y = xy
   return a*x + b*y + c

T=303.15
R=0.0019872041
def calc_Ffb(b,f):
   return 1/(1+np.exp(b/(R*T))*(1+np.exp(f/(R*T))))
def calc_Ff(f):
   return 1/(1+np.exp(f/(R*T)))

target_mean = 0.0
target_std_dev = 3.0

def standardization(x):
    normalized_sequence = (x - np.mean(x)) / np.std(x)
    return normalized_sequence * target_std_dev + target_mean

def model_estimate(metric,y_exp_test,y_pred):
    if metric == 'r2':
        s = r2_score(y_exp_test, y_pred)
    elif metric == 'rmse':
        s = - np.sqrt(mean_squared_error(y_exp_test, y_pred, squared=False))
    elif metric == 'pearson':
        s = pearsonr_metric(y_exp_test, y_pred)
    elif metric == 'spearman':
        s = spearmanr_metric(y_exp_test, y_pred)
    return s

# calculation of rosetta functional calue
def make_rosetta_value(dataset,function_type,comp_value):
    F_calc_scale = np.array([])
    ddGf_rosetta = dataset["ddGf_rosetta"]                            
    ddGf_rosetta_scale = standardization(ddGf_rosetta)

    if function_type == "binding":
        ddGb_rosetta = dataset["ddGb_rosetta"]
        ddGb_rosetta_scale = standardization(ddGb_rosetta)
        F_calc = calc_Ffb(ddGb_rosetta_scale,ddGf_rosetta_scale)
    elif function_type == "abundance":
        F_calc = calc_Ff(ddGf_rosetta_scale)
    return F_calc

parser = argparse.ArgumentParser(description='Command line argument examples')
parser.add_argument('--res_number', type=int)
parser.add_argument("--function_type", choices=["binding", "abundance"], help="Type of functional value: binding or abundance")
parser.add_argument('--metric', type=str)
parser.add_argument('--test_dataset', type=str)
parser.add_argument('--outputfile', type=str)
args = parser.parse_args()

res_number = args.res_number
function_type = args.function_type
metric = args.metric
test_dataset_name = args.test_dataset
outputfile = args.outputfile
scoring = set_scoring(metric)

dataset_test = np.genfromtxt(test_dataset_name, delimiter=',', dtype=None, names=True, encoding='utf-8')

test_size=0.1
# Main calculation
def compute(j,comp_value):
    random_state = j
    dataset_test1, dataset_test2 = train_test_split(dataset_test, test_size=test_size, random_state=random_state)
    score_array = []

    if function_type == "binding":
        y_test2 = dataset_test2["Ffb_exp"] 
    elif function_type == "abundance":
        y_test2 = dataset_test2["Ff_exp"] 

    if comp_value =="rosetta":
        F_calc_test = make_rosetta_value(dataset_test2,function_type,comp_value)
    elif comp_value =="zero-shot":
        F_calc_test = dataset_test2["ESM2_zeroshot"] 
    
    score_calc = model_estimate(metric,y_test2,F_calc_test)
    score_array.append((score_calc))

    return score_array,j,comp_value

def parallel_computation():
    numbers = range(1, 11) # Random seeds
    score_list_1_orig = [[] for _ in range(1,11)]
    score_list_2_orig = [[] for _ in range(1,11)]
    score_array_1 = []
    score_array_2 = []

    # Submit parallel computation with each random seeds and calculation value types
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(compute, num, "zero-shot"): num for num in numbers}
        futures.update({executor.submit(compute, num, "rosetta"): num for num in numbers})
        for future in concurrent.futures.as_completed(futures):
            result,num, phase = future.result()
            if phase == "zero-shot":
                score_list_1_orig[num-1].append(result)
            elif phase == "rosetta":
                score_list_2_orig[num-1].append(result)
    return score_list_1_orig, score_list_2_orig


if __name__ == "__main__":
    score_list_1_orig,score_list_2_orig = parallel_computation()

    score_list_1 = []
    score_list_2 = []
    for i in range(0,10):
        score_list_1.append(score_list_1_orig[i][0])
        score_list_2.append(score_list_2_orig[i][0])

    # Caluclate average and standard deviations
    score_array_ave_1 = np.mean(score_list_1, axis=0)
    score_array_std_dev_1 = np.std(score_list_1, axis=0, ddof=1)
    score_array_ave_2 = np.mean(score_list_2, axis=0)
    score_array_std_dev_2 = np.std(score_list_2, axis=0, ddof=1)

    # Combine data
    combined_data = np.column_stack((score_array_ave_1,score_array_ave_2,score_array_std_dev_1,score_array_std_dev_2))

    # Save as csv file
    os.makedirs("results", exist_ok=True)
    csv_filename = "results/" + outputfile + ".csv"
    with open(csv_filename, mode="w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["zero-shot calc","Rosetta calc","zero-shot calc std","Rosetta calc std"])
        csv_writer.writerows(combined_data.tolist())
    
    # Save non-averaged results
    os.makedirs("results/split_results", exist_ok=True)
    for i in range(1,11):
        csv_filename = "results/split_results/" + outputfile + "_" +str(i)+".csv"
        with open(csv_filename, mode="w", newline="") as csv_file:
            combined_data = np.column_stack((score_list_1_orig[i-1][0],score_list_2_orig[i-1][0]))
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["zero-shot","Rosetta"])
            csv_writer.writerows(combined_data.tolist())

