import argparse

import os
import numpy as np
import pandas as pd

# import warnings
# warnings.filterwarnings('error')
import datetime


os.chdir('/home-nfs/raphaelr/lacqr')

from uacqr import uacqr 
from helper import QuantileRegressionNN
from experiment import experiment
from datasets import GetDataset
from sklearn.model_selection import KFold, ParameterGrid, ParameterSampler
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, ShuffleSplit
from scipy.stats.distributions import expon

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', '-d', help="int or string for dataset", default= 'community')
parser.add_argument('--runs', '-r', help="int for number of runs", default= 20,  type=int)
parser.add_argument('--log_response', '-l', help="perform log transform on response?", action='store_true')
parser.add_argument('--randomized_conformal', '-rc', help="use randomized conformal to break ties?", action='store_true')
parser.add_argument('--use_iqr', '-iqr', help="use iqr instead of std for uacqrs", action='store_true')
parser.add_argument('--standardize_response', '-s', help="divide responses by mean absolute response", action='store_true')
parser.add_argument('--model', '-m', help="model type", default='rfqr', type=str)
parser.add_argument('--cv_iters', '-cv', help="number of cv iterations for neural_net", default=100, type=int)
parser.add_argument('--two_sided_conformal', '-t', help="conformalize upper and lower separately", action='store_true')
parser.add_argument('--uacqrs_bagging', '-ub', help="use bagging for uacqrs, not full model", action='store_true')
parser.add_argument('--conditional_coverage', '-c', help="use conditional coverage as the metric", action='store_true')


args = parser.parse_args()

dataset = args.dataset

runs = args.runs

log_response = args.log_response

randomized_conformal = args.randomized_conformal

use_iqr = args.use_iqr

standardize_response = args.standardize_response

model_type = args.model

cv_iters = args.cv_iters

two_sided_conformal = args.two_sided_conformal

uacqrs_bagging = args.uacqrs_bagging

conditional_coverage = args.conditional_coverage

date_today = datetime.date.today().strftime("%m.%d.%Y")

if conditional_coverage:
    metric = 'conditional_coverage'
else:
    metric = 'interval_score_loss'

if use_iqr:
    agg = 'iqr'
else:
    agg = 'std'

if dataset.isdigit():
    dataset_names = ['bike','bio', 'community','concrete','homes','star', 'blog_data','facebook_1','facebook_2',
                     'meps_21','meps_20','meps_19']
    # dataset_names = ["community", "bike", "concrete", "blog_data", "bio", "facebook_1", "facebook_2", 
                    #  "star", "homes", "bulldozer","meps_21", "meps_20", "meps_19"]
    dataset = dataset_names[int(dataset)-1]



fractions = {"community":1, "bike":1, "concrete":1, "blog_data":1/4, "bio":1/4, 
"facebook_1":1/4, "facebook_2":1/8, "star": 1, "homes": 1/2, "bulldozer":0.03, "meps_21":1, "meps_20":1, "meps_19":1}

# fractions = {"community":1/3, "bike":1/3, "concrete":1/3, "blog_data":1/12, "bio":1/12, 
# "facebook_1":1/12, "facebook_2":1/24, "star": 1/3, "homes": 1/6, "bulldozer":0.01, "meps_21":1/3, "meps_20":1/3, "meps_19":1/3}

# sizes = {"community":1994, "bike":10886, "concrete":1030, "blog_data":52397, "bio":45730, 
# "facebook_1":40948, "facebook_2":81311, "star": 2161, "homes": 21613, "meps_21", "meps_20", "meps_19"}

X, y = GetDataset(dataset)

X = np.array(X)
y = np.array(y)

if np.mean(y==0)<0.25:
    two_sided_conformal=False

print(np.mean(np.abs(y)))
if log_response:
    y = np.log(1+y)


df = pd.DataFrame(columns=['params','uacqrp','uacqrs','cqr','cqrr'])


param_dist = {'dropout':[0, 0.001, 0.01, 0.05], 'epochs':[101, 151, 201], 'hidden_size':[100, 200], 
              'lr':[1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3],
         'batch_size':[8,16,32,64], 'normalize':[True], 'weight_decay':[0,1e-7,1e-6], 
         'epoch_model_tracking':[False, True], 'drop_last':[True], 'undo_quantile_crossing':[True],
         'nuisance':expon()}
# Set up cross-validation
kf = ShuffleSplit(n_splits=runs, random_state=42, train_size=0.8*fractions[dataset], test_size=0.2*fractions[dataset])

# Perform hyperparameter tuning
best_score_uacqrp = float('inf')
best_params_uacqrp = None

best_score_uacqrs = float('inf')
best_params_uacqrs = None

best_score_cqr = float('inf')
best_params_cqr = None

best_score_cqrr = float('inf')
best_params_cqrr = None

save_string = f"results/{dataset}_nn_experiment_{agg}_{date_today}.pkl"

t=0
for params in ParameterSampler(param_dist, cv_iters, random_state=42):
    del params["nuisance"]
    scores_uacqrp = []
    scores_uacqrs = []
    scores_cqr = []
    scores_cqrr = []
    i = 0
    for train_index, test_index in kf.split(X):
        print(t)
        X_train_fold, X_test_fold = X[train_index], X[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]
        X_train_fold, X_val_fold, y_train_fold, y_val_fold = train_test_split(X_train_fold, y_train_fold, test_size=0.5, random_state=42)
        
        
        if standardize_response:
            y_test_fold = y_test_fold / np.mean(np.abs(y_train_fold))
            y_val_fold = y_val_fold / np.mean(np.abs(y_train_fold))
            y_train_fold = y_train_fold / np.mean(np.abs(y_train_fold))
        
        B = params["epochs"] - 1
        
        
        model = uacqr(params,
                     bootstrapping_for_uacqrp=False, B=B, random_state=i, uacqrs_agg=agg, randomized_conformal=randomized_conformal,
                     uacqrs_bagging=False, model_type='neural_net')
        model.fit(X_train_fold, y_train_fold)
        model.calibrate(X_val_fold, y_val_fold)
        model.evaluate(X_test_fold, y_test_fold)

        scores_uacqrp.append(model.uacqrp_average_length_test)
        scores_uacqrs.append(model.uacqrs_average_length_test)
        scores_cqr.append(model.cqr_average_length_test)
        scores_cqrr.append(model.cqrr_average_length_test)
        
        df.loc[t] = [str(params), model.uacqrp_average_length_test, model.uacqrs_average_length_test,
                     model.cqr_average_length_test, model.cqrr_average_length_test]
        
        t+=1
        i+=1
        if i % 5 == 0:
            df.to_pickle(save_string)
            avg_score_uacqrp = np.mean(scores_uacqrp)
            avg_score_uacqrs = np.mean(scores_uacqrs)
            avg_score_cqr = np.mean(scores_cqr)
            avg_score_cqrr = np.mean(scores_cqrr)
            
            uacqrp_flag = avg_score_uacqrp > 1.05 * best_score_uacqrp
            uacqrs_flag = avg_score_uacqrs > 1.05 * best_score_uacqrs
            cqr_flag = avg_score_cqr > 1.05 * best_score_cqr
            cqrr_flag = avg_score_cqrr > 1.05 * best_score_cqrr
            
            if uacqrp_flag and uacqrs_flag and cqr_flag and cqrr_flag:
                break
       
        if i == runs:
            df.to_pickle(save_string)

    avg_score_uacqrp = np.mean(scores_uacqrp)
    avg_score_uacqrs = np.mean(scores_uacqrs)
    avg_score_cqr = np.mean(scores_cqr)
    avg_score_cqrr = np.mean(scores_cqrr)
    
    if avg_score_uacqrp < best_score_uacqrp:
        best_score_uacqrp = avg_score_uacqrp
        best_params_uacqrp = params
    if avg_score_uacqrs < best_score_uacqrs:
        best_score_uacqrs = avg_score_uacqrs
        best_params_uacqrs = params
    if avg_score_cqr < best_score_cqr:
        best_score_cqr = avg_score_cqr
        best_params_cqr = params
    if avg_score_cqrr < best_score_cqrr:
        best_score_cqrr = avg_score_cqrr
        best_params_cqrr = params
df.to_pickle(save_string)