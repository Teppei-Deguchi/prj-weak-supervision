import sys
import csv
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold,train_test_split
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

def train_pred(X_train,y_train,X_test,sample_weight,model_type,random_state,**parameters):
    if model_type == "svr":
        model = SVR(C = parameters["C"], gamma = parameters["gamma"],epsilon = parameters["epsilon"])
    elif model_type == "rfr":
        model = RandomForestRegressor(n_estimators=parameters["n_estimaters"],random_state=random_state, n_jobs=-1)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model.fit(X_train, y_train, sample_weight=sample_weight)
    y_pred = model.predict(X_test)
    return y_pred

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

# calculation of ESM-2 zero-shot, rosetta, and hybrid functional value
def make_calc_functional_value(dataset_train1,dataset_train2,ddGf_rosetta_train1_scale,ddGf_rosetta_train2_scale,comp_value,i):
    
    F_calc_scale = np.array([])
    F_exp_train1,ESM2_zeroshot_train1 = dataset_train1["activity"],dataset_train1["ESM2_zeroshot"]
    F_exp_train2,ESM2_zeroshot_train2 = dataset_train2["activity"],dataset_train2["ESM2_zeroshot"]
    
    initial_parameters = [0.0,0.0]
    initial_parameters_hybrid = [0.0,0.0,0.0]

    if comp_value == "zero-shot": # Scaling of ESM-2 zero-shot value
        optimized_parameters ,covariance = curve_fit(fitting_function_linr, ESM2_zeroshot_train2, F_exp_train2, p0=initial_parameters,maxfev = 1000000)
        a_fit, b_fit = optimized_parameters
        F_calc_scale = fitting_function_linr(ESM2_zeroshot_train1, a_fit, b_fit)
    
    elif comp_value == "rosetta": # Calculate Rosetta Ffb or Ff value       
        F_calc1 = calc_Ff(ddGf_rosetta_train1_scale)
        F_calc2 = calc_Ff(ddGf_rosetta_train2_scale)

        optimized_parameters ,covariance = curve_fit(fitting_function_linr, F_calc2, F_exp_train2, p0=initial_parameters,maxfev = 1000000)
        a_fit, b_fit = optimized_parameters
        F_calc_scale = fitting_function_linr(F_calc1, a_fit, b_fit)

    elif comp_value == "hybrid": # Calculate hybrid value
        F_calc1 = calc_Ff(ddGf_rosetta_train1_scale)      
        F_calc2 = calc_Ff(ddGf_rosetta_train2_scale)

        xdata2 = np.vstack((F_calc2, ESM2_zeroshot_train2))
        optimized_parameters ,covariance = curve_fit(fitting_function_linr_two, xdata2 , F_exp_train2, p0=initial_parameters_hybrid,maxfev = 1000000)
        a_fit, b_fit,c_fit = optimized_parameters
        xdata1 = np.vstack((F_calc1, ESM2_zeroshot_train1))
        F_calc_scale = fitting_function_linr_two(xdata1, a_fit, b_fit,c_fit)
    return F_calc_scale

# ML with only experimental data
def ML_exp_only(X_train,y_train,X_test,model_type,random_state,**parameters):
    sample_weight = np.ones(len(y_train)) * (np.exp(len(y_train)/res_number))
    y_pred = train_pred(X_train,y_train,X_test,sample_weight,model_type,random_state,**parameters)
    return y_pred

# ML with data augmentation
def ML_data_augment(X_train2,y_train2,X_train1,F_calc_scale,X_test,model_type,random_state,**parameters):
    X_train_cat = np.concatenate((X_train2, X_train1))
    y_train_cat = np.concatenate((y_train2, F_calc_scale))
    
    # Weight adjustment
    weights_exp = np.ones(len(y_train2)) * (np.exp(len(y_train2)/res_number))
    weights_calc = np.ones(len(F_calc_scale)) * (np.exp(-len(y_train2)/res_number))
    sample_weight = np.concatenate((weights_exp, weights_calc))
    
    y_pred = train_pred(X_train_cat,y_train_cat,X_test,sample_weight,model_type,random_state,**parameters)
    return y_pred

def weight_adj_decision(X_train2,y_train2,X_train1,F_calc_scale,X_test,model_type,random_state,**parameters):
    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
    X_train2_array = np.array(X_train2)
    results = []
    score_fold = np.zeros((5))

    #5-fold cross validation with only experimental value
    for fold, (train_index, test_index) in enumerate(kf.split(X_train2_array)):
        X_train3, X_valid = X_train2_array[train_index], X_train2_array[test_index]
        y_train3, y_valid = y_train2[train_index], y_train2[test_index]
        sample_weight = np.ones(len(y_train3))
        y_pred = train_pred(X_train3,y_train3,X_valid,sample_weight,model_type,random_state,**parameters)
        score_fold[fold] = model_estimate(metric,y_valid,y_pred)
    
    score_eval = np.zeros((5))
    k = 0
    #5-fold cross validation with data augumentation
    for fold, (train_index, test_index) in enumerate(kf.split(X_train2_array)):
        X_train3, X_valid = X_train2_array[train_index], X_train2_array[test_index]
        y_train3, y_valid = y_train2[train_index], y_train2[test_index]
        y_pred = ML_data_augment(X_train3,y_train3,X_train1,F_calc_scale,X_valid,model_type,random_state,**parameters)
        score_eval[fold] = model_estimate(metric,y_valid,y_pred)

    mask = score_eval < score_fold #Compare two estimates
    count = np.sum(mask)
    if count >= 3:
        k = k + 1

    if k == 0:
        y_pred = ML_data_augment(X_train2,y_train2,X_train1,F_calc_scale,X_test,model_type,random_state,**parameters)
    if k == 1:
        y_pred = ML_exp_only(X_train2,y_train2,X_test,model_type,random_state,**parameters)
    return y_pred

test_size=0.2
parser = argparse.ArgumentParser(description='Command line argument examples')
parser.add_argument('--res_number', type=int)
parser.add_argument("--model_type", choices=["svr", "rfr"], help="Type of regression model: svr or rfr")
parser.add_argument('--C', type=float)
parser.add_argument('--gamma', type=float)
parser.add_argument('--epsilon', type=float)
parser.add_argument('--n_estimaters', type=int)
parser.add_argument('--metric', type=str)
parser.add_argument('--split_data_b', type=int)
parser.add_argument('--split_data_e', type=int)
parser.add_argument('--sequence_embedding', type=str)
parser.add_argument('--dataset', type=str)
parser.add_argument('--outputfile', type=str)
args = parser.parse_args()

res_number = args.res_number
model_type = args.model_type
parameters = {"C": args.C, "gamma": args.gamma,"epsilon": args.epsilon} if model_type == "svr" else {"n_estimaters": args.n_estimaters}
metric = args.metric
split_data_b = args.split_data_b
split_data_e = args.split_data_e
sequence_embedding = args.sequence_embedding
dataset_name = args.dataset
outputfile = args.outputfile
scoring = set_scoring(metric)

# Get amino-acid embeddings
with open(sequence_embedding, 'rb') as file:
    X = pickle.load(file)
X = torch.tensor(X)
X = X.reshape(X.size(0), -1)
X = X.tolist()

dataset = np.genfromtxt(dataset_name, delimiter=',', dtype=None, names=True, encoding='utf-8')

# Main calculation
def compute(j,comp_value):
    random_state = j
    X_train,X_test,dataset_train, dataset_test = train_test_split(X,dataset, test_size=test_size, random_state=random_state)
    score_array = []

    y_test = dataset_test["activity"] 

    for i in range(split_data_b,split_data_e):  
        exp_rate = i / 100 
        ddGf_rosetta_train = dataset_train["ddGf_rosetta"]
        ddGf_rosetta_train_scale = standardization(ddGf_rosetta_train)
        X_train1,X_train2,dataset_train1,dataset_train2,ddGf_rosetta_train1_scale,ddGf_rosetta_train2_scale = train_test_split(X_train,dataset_train,ddGf_rosetta_train_scale,test_size=exp_rate, random_state=random_state)
        F_calc_scale = make_calc_functional_value(dataset_train1,dataset_train2,ddGf_rosetta_train1_scale,ddGf_rosetta_train2_scale,comp_value,i)
        
        y_train1 = dataset_train1["activity"]
        y_train2 = dataset_train2["activity"]
        
        if comp_value == "exp_only":
            y_pred = ML_exp_only(X_train2,y_train2,X_test,model_type,random_state,**parameters)
            score = model_estimate(metric,y_test,y_pred)
        else:
            y_pred = weight_adj_decision(X_train2,y_train2,X_train1,F_calc_scale,X_test,model_type,random_state,**parameters)
            score = model_estimate(metric,y_test,y_pred)

        if comp_value == "exp_only":
            score_array.append((exp_rate,score))
        else:
            score_array.append((score))
    return score_array,j,comp_value

def parallel_computation():
    numbers = range(1, 11) # Random seeds
    score_list_1_orig = [[] for _ in range(1,11)]
    score_list_2_orig = [[] for _ in range(1,11)]
    score_list_3_orig = [[] for _ in range(1,11)]
    score_list_4_orig = [[] for _ in range(1,11)]
    score_array_1 = []
    score_array_2 = []
    score_array_3 = []
    score_array_4 = []

    # Submit parallel computation with each random seeds and calculation value types
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(compute, num, "exp_only"): num for num in numbers}
        futures.update({executor.submit(compute, num, "zero-shot"): num for num in numbers})
        futures.update({executor.submit(compute, num, "rosetta"): num for num in numbers})
        futures.update({executor.submit(compute, num, "hybrid"): num for num in numbers})
        for future in concurrent.futures.as_completed(futures):
            result,num, phase = future.result()
            if phase == "exp_only":
                score_list_1_orig[num-1].append(result)
            elif phase == "zero-shot":
                score_list_2_orig[num-1].append(result)
            elif phase == "rosetta":
                score_list_3_orig[num-1].append(result)
            elif phase == "hybrid":
                score_list_4_orig[num-1].append(result)
    return score_list_1_orig, score_list_2_orig, score_list_3_orig,score_list_4_orig


if __name__ == "__main__":
    score_list_1_orig,score_list_2_orig,score_list_3_orig,score_list_4_orig = parallel_computation()

    score_list_1 = []
    score_list_2 = []
    score_list_3 = []
    score_list_4 = []
    for i in range(0,10):
        score_list_1.append(score_list_1_orig[i][0])
        score_list_2.append(score_list_2_orig[i][0])
        score_list_3.append(score_list_3_orig[i][0])
        score_list_4.append(score_list_4_orig[i][0])

    # Caluclate average and standard deviations
    score_array_ave_1 = np.mean(score_list_1, axis=0)
    score_array_std_dev_1 = np.std(score_list_1, axis=0, ddof=1)
    score_array_ave_2 = np.mean(score_list_2, axis=0)
    score_array_std_dev_2 = np.std(score_list_2, axis=0, ddof=1)
    score_array_ave_3 = np.mean(score_list_3, axis=0)
    score_array_std_dev_3 = np.std(score_list_3, axis=0, ddof=1)
    score_array_ave_4 = np.mean(score_list_4, axis=0)
    score_array_std_dev_4 = np.std(score_list_4, axis=0, ddof=1)

    # Combine data
    combined_data = np.column_stack((score_array_ave_1[:,0],score_array_ave_1[:,1],score_array_ave_2,score_array_ave_3,score_array_ave_4,score_array_std_dev_1[:,1],score_array_std_dev_2,score_array_std_dev_3,score_array_std_dev_4))

    # Save as csv file
    os.makedirs("results", exist_ok=True)
    csv_filename = "results/" + outputfile + ".csv"
    with open(csv_filename, mode="w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["exp rate","exp","zero-shot","Rosetta","Hybrid","exp std","zero-shot std","Rosetta std","Hybrid std"])
        csv_writer.writerows(combined_data.tolist())
    
    # Save non-averaged results
    os.makedirs("results/split_results", exist_ok=True)
    for i in range(1,11):
        csv_filename = "results/split_results/" + outputfile + "_" +str(i)+".csv"
        with open(csv_filename, mode="w", newline="") as csv_file:
            combined_data = np.column_stack((score_list_1_orig[i-1][0],score_list_2_orig[i-1][0],score_list_3_orig[i-1][0],score_list_4_orig[i-1][0]))
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["exp rate","exp","zero-shot","Rosetta","Hybrid"])
            csv_writer.writerows(combined_data.tolist())

