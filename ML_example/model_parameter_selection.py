import os
"""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
"""
import pickle
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearnex import patch_sklearn,config_context
patch_sklearn()
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer
import argparse
import torch

parser = argparse.ArgumentParser(description='Command line argument examples')
parser.add_argument('--embedding', type=str)
parser.add_argument('--exp_data', type=str)
args = parser.parse_args()

# --- Data Loading ---
# Get amino-acid embeddings
with open(args.embedding, 'rb') as file:
    X = pickle.load(file)
X = torch.tensor(X)
X = X.reshape(X.size(0), -1)
X = X.tolist()

y = pd.read_csv(args.exp_data).iloc[:, 1].values  # 2nd column (index=1)

def spearman_scorer(y_true, y_pred):
    return spearmanr(y_true, y_pred).correlation

scorer = make_scorer(spearman_scorer, greater_is_better=True)

# --- Model Settings ---
models = {
    "Linear": {
        "model": Pipeline([("scaler", StandardScaler()), ("regressor", LinearRegression())]),
        "params": {
            # LinearRegression has no hyperparameters to tune
        }
    },
    "GP": {
        "model": Pipeline([("scaler", StandardScaler()), ("regressor", GaussianProcessRegressor())]),
        "params": {
            "regressor__alpha": [1e-10, 1e-5, 1e-2],
            "regressor__kernel": [C(1.0) * RBF(1.0), C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))]
        }
    },
    "SVR": {
        "model": Pipeline([("scaler", StandardScaler()), ("regressor", SVR())]),
        "params": {
            "regressor__C": [0.1, 1, 10],
            "regressor__epsilon": [0.01, 0.1, 1],
            "regressor__gamma": [0.0001,0.01,1.0]
        }
    },
    "RF": {
        "model": Pipeline([("scaler", StandardScaler()), ("regressor", RandomForestRegressor(random_state=42))]),
        "params": {
            "regressor__n_estimators": [50, 100],
        }
    }
}

# Hyperparameter tuning after model selection
models_rfr = {
    "RF": {
        "model": Pipeline([("scaler", StandardScaler()), ("regressor", RandomForestRegressor(random_state=42))]),
        "params": {
            "regressor__n_estimators": [1000, 1500,2000,2500,3000],
        }
    }
}

models_svr = {
        "SVR": {
        "model": Pipeline([("scaler", StandardScaler()), ("regressor", SVR())]),
        "params": {
            "regressor__C": [0.01,0.1, 1, 10],
            "regressor__epsilon": [0.0001,0.001,0.01, 0.1, 1],
            "regressor__gamma": [0.0001,0.001,0.01,0.1,1.0]
        }
    }
}

# --- Nested CV Setup ---
outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)

# --- Execution ---
best_model_name = None
best_score = -np.inf
best_model = None


# Model selection phase: compare multiple models with a coarse parameter grid.
# Switch to models_rfr or models_svr for fine-grained tuning once the best model is identified.
model_set = models

print(model_set)
#for name, config in models.items():
for name, config in model_set.items():
    print(f"\n🔍 Evaluating model: {name}")
    grid = GridSearchCV(
        estimator=config["model"],
        param_grid=config["params"],
        cv=inner_cv,
        scoring=scorer,
        n_jobs=-1
    )

    # Nested CV: model evaluation on outer fold
    nested_score = cross_val_score(grid, X, y, cv=outer_cv, scoring=scorer, n_jobs=1)
    mean_score = np.mean(nested_score)
    print(f"Average sparman score: {mean_score:.4f}")

    if mean_score > best_score:
        best_score = mean_score
        best_model_name = name
        best_model = grid

# --- Retrain with best model + optimal parameters ---
print(f"\n🏆 Best model: {best_model_name} with score {best_score:.4f}")
best_model.fit(X, y)
print(f"Best hyperparameters: {best_model.best_params_}")


