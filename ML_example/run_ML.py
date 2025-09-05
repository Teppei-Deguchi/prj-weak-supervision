import sys
import numpy as np
"""　#optimization library for intel CPU (if required)
from sklearnex import patch_sklearn
patch_sklearn()
"""
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold,train_test_split
from sklearn.metrics import make_scorer,r2_score, mean_squared_error
from scipy.stats import spearmanr, pearsonr
from scipy.optimize import curve_fit
import pickle
import torch
import argparse
import concurrent.futures
import os
from pathlib import Path
import pandas as pd

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

# calculation of rosetta functional calue
def make_rosetta_value(dataset_calc,dataset_exp,PPI,comp_value):
    F_calc_scale = np.array([])

    ddGf_rosetta_train_exp = dataset_train_exp["ddGf_rosetta"]
    ddGf_rosetta_train_calc = dataset_train_calc["ddGf_rosetta"]
    ddGf_rosetta_train_combined = np.concatenate([ddGf_rosetta_train_exp,ddGf_rosetta_train_calc])

    ddGf_rosetta_train_scale = standardization(ddGf_rosetta_train_combined)
    ddGf_rosetta_train_exp_scale = ddGf_rosetta_train_scale[:len(ddGf_rosetta_train_exp)]
    ddGf_rosetta_train_calc_scale = ddGf_rosetta_train_scale[len(ddGf_rosetta_train_exp):]

    if PPI == "yes":
        ddGb_rosetta_train_exp = dataset_train_exp["ddGb_rosetta"]
        ddGb_rosetta_train_calc = dataset_train_calc["ddGb_rosetta"]
        ddGb_rosetta_train_combined = np.concatenate([ddGb_rosetta_train_exp,ddGb_rosetta_train_calc])

        ddGb_rosetta_train_scale = standardization(ddGb_rosetta_train_combined)
        ddGb_rosetta_train_exp_scale = ddGb_rosetta_train_scale[:len(ddGb_rosetta_train_exp)]
        ddGb_rosetta_train_calc_scale = ddGb_rosetta_train_scale[len(ddGb_rosetta_train_exp):]
        
        F_calc = calc_Ffb(ddGb_rosetta_train_calc_scale,ddGf_rosetta_train_calc_scale)
        F_exp = calc_Ffb(ddGb_rosetta_train_exp_scale,ddGf_rosetta_train_exp_scale)
    else:
        F_calc = calc_Ff(ddGf_rosetta_train_calc_scale)
        F_exp = calc_Ff(ddGf_rosetta_train_exp_scale)
    return F_calc,F_exp

# calculation of ESM-2 zero-shot, rosetta, and hybrid functional value
def make_calc_functional_value(dataset_train1,dataset_train2,PPI,comp_value):
    
    F_calc_scale = np.array([])
    ESM2_zeroshot_train1 = dataset_train1["ESM2_zeroshot"]
    F_exp_train2,ESM2_zeroshot_train2 = dataset_train2["activity"],dataset_train2["ESM2_zeroshot"]
    
    initial_parameters = [0.0,0.0]
    initial_parameters_hybrid = [0.0,0.0,0.0]

    if comp_value == "esm2": # Scaling of ESM-2 zero-shot value
        optimized_parameters ,covariance = curve_fit(fitting_function_linr, ESM2_zeroshot_train2, F_exp_train2, p0=initial_parameters,maxfev = 1000000)
        a_fit, b_fit = optimized_parameters
        F_calc_scale = fitting_function_linr(ESM2_zeroshot_train1, a_fit, b_fit)
    
    elif comp_value == "rosetta": # Calculate Rosetta Ffb or Ff value       
        F_calc1,F_calc2 = make_rosetta_value(dataset_train1,dataset_train2,PPI,comp_value)

        optimized_parameters ,covariance = curve_fit(fitting_function_linr, F_calc2, F_exp_train2, p0=initial_parameters,maxfev = 1000000)
        a_fit, b_fit = optimized_parameters
        F_calc_scale = fitting_function_linr(F_calc1, a_fit, b_fit)

    elif comp_value == "hybrid": # Calculate hybrid value
        F_calc1,F_calc2 = make_rosetta_value(dataset_train1,dataset_train2,PPI,comp_value)

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

# Get amino-acid embeddings
def load_embedding(sequence_embedding):
    with open(sequence_embedding, 'rb') as file:
        X = pickle.load(file)
    X = torch.tensor(X)
    X = X.reshape(X.size(0), -1)
    X = X.tolist()
    return X

def load_dataset(dataset_name):
    dataset = np.genfromtxt(dataset_name, delimiter=',', dtype=None, names=True, encoding='utf-8')
    return dataset

parser = argparse.ArgumentParser(description='Command line argument examples')
parser.add_argument('--res_number', type=int)
parser.add_argument("--model_type", choices=["svr", "rfr"], help="Type of regression model: svr or rfr")
parser.add_argument('--C', type=float, help="Set if model_type=svr")
parser.add_argument('--gamma', type=float, help="Set if model_type=svr")
parser.add_argument('--epsilon', type=float, help="Set if model_type=svr")
parser.add_argument('--n_estimaters', type=int, help="Set if model_type=rfr")
parser.add_argument('--eval_testdata', choices=["yes", "no"], help="Chooe yes if you want to evaluate prediction accuracy with test data")
parser.add_argument('--metric', type=str,help="Select from r2, rmse, pearson, spearman")
parser.add_argument('--sequence_embedding_dir', type=str, help="Path to embedding directory")
parser.add_argument('--dataset_dir', type=str, help="Path to dataset directory")
parser.add_argument('--PPI', choices=["yes", "no"], help="Chooe yes if you want to predict protein-protien binding affinity")
parser.add_argument('--calc_mode', type=str, help="Select from exp_only, rosetta, esm2, hybrid")
args = parser.parse_args()

res_number = args.res_number
model_type = args.model_type
parameters = {"C": args.C, "gamma": args.gamma,"epsilon": args.epsilon} if model_type == "svr" else {"n_estimaters": args.n_estimaters}
metric = args.metric
eval_testdata = args.eval_testdata
sequence_embedding_dir = args.sequence_embedding_dir
dataset_dir = args.dataset_dir
PPI = args.PPI
calc_mode = args.calc_mode
scoring = set_scoring(metric)

sequence_embedding_dir = Path(args.sequence_embedding_dir)
dataset_dir = Path(args.dataset_dir)

X_train_exp = load_embedding(sequence_embedding_dir / "train_exp.pkl")
X_train_calc = load_embedding(sequence_embedding_dir / "train_calc.pkl")
X_test = load_embedding(sequence_embedding_dir /"test.pkl")
dataset_train_exp = load_dataset(dataset_dir / "train_exp.csv")
dataset_train_calc = load_dataset(dataset_dir / "train_calc.csv")
dataset_test = load_dataset(dataset_dir / "test.csv")

random_state = 42
# Main calculation
def compute(comp_value):
    F_calc_scale = make_calc_functional_value(dataset_train_calc,dataset_train_exp,PPI,comp_value)
        
    y_train_exp = dataset_train_exp["activity"]
        
    if comp_value == "exp_only":
        y_pred = ML_exp_only(X_train_exp,y_train_exp,X_test,model_type,random_state,**parameters)
    else:
        y_pred = weight_adj_decision(X_train_exp,y_train_exp,X_train_calc,F_calc_scale,X_test,model_type,random_state,**parameters)
        
    
    if eval_testdata == "yes":
        y_test = dataset_test["activity"]
        score = model_estimate(metric,y_test,y_pred)
        print("Test Data " + metric + ": ",score)

    return y_pred

y_pred = compute(calc_mode)

df = pd.DataFrame({
    "sequence": dataset_test["sequence"],
    "prediction": y_pred
})

df.to_csv("result/output.csv", index=False)



