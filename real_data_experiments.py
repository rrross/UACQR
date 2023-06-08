import argparse

import os
import numpy as np

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

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', '-d', help="int or string for dataset", default= 'community')
parser.add_argument('--runs', '-r', help="int for number of runs", default= 2,  type=int)
parser.add_argument('--log_response', '-l', help="perform log transform on response?", action='store_true')
parser.add_argument('--randomized_conformal', '-rc', help="use randomized conformal to break ties?", action='store_true')
parser.add_argument('--use_iqr', '-iqr', help="use iqr instead of std for uacqrs", action='store_true')
parser.add_argument('--standardize_response', '-s', help="divide responses by mean absolute response", action='store_true')
parser.add_argument('--model', '-m', help="model type", default='rfqr', type=str)
parser.add_argument('--cv_iters', '-cv', help="number of cv iterations for neural_net", default=20, type=int)
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


if conditional_coverage:
    metric = 'conditional_coverage'
else:
    metric = 'interval_score_loss'


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
# if standardize_response:
#     y = y / np.mean(np.abs(y))

if model_type == 'rfqr':
    params = dict()
    params["max_features"] = 'sqrt'
    B=100
elif model_type == 'neural_net':
    param_dist = {'dropout':[0], 'epochs':[101, 151, 201], 'hidden_size':[100, 200], 
                  'lr':[1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3],
             'batch_size':[8,16,32,64], 'normalize':[True], 'weight_decay':[0,1e-7,1e-6], 
             'epoch_model_tracking':[True]}
    # Set up cross-validation
    kf = ShuffleSplit(n_splits=2, random_state=42, train_size=0.4*fractions[dataset], test_size=0.2*fractions[dataset])

    # Perform hyperparameter tuning
    best_score = float('inf')
    best_params = None
    for params in ParameterSampler(param_dist, cv_iters, random_state=42):
        scores = []
        for train_index, val_index in kf.split(X):
            X_train_fold, X_val_fold = X[train_index], X[val_index]
            y_train_fold, y_val_fold = y[train_index], y[val_index]

            model = QuantileRegressionNN(quantiles=[0.05, 'mean', 0.95], **params, random_state=42)
            model.fit(X_train_fold, y_train_fold)

            scores.append(model.score(X_val_fold, y_val_fold))

        avg_score = np.mean(scores)
        if avg_score < best_score:
            best_score = avg_score
            best_params = params
    
    params = best_params
    B = params['epochs']-1
elif model_type == 'linear':
    params = dict()
    B=100

if use_iqr:
    agg = 'iqr'
else:
    agg = 'std'

if model_type == 'rfqr':
    results = experiment(X=X, y=y, S=runs, B=B, fixed_model_params=params, model_type=model_type,
                     empirical_data_fraction=fractions[dataset], randomized_conformal=randomized_conformal, uacqrs_agg=agg,
                     uacqrs_bagging=uacqrs_bagging, metric=metric,
                     max_normalization=standardize_response)
elif model_type =='neural_net':
    results = experiment(X=X, y=y, S=runs, B=B, fixed_model_params=params, model_type=model_type,
                     empirical_data_fraction=fractions[dataset], randomized_conformal=randomized_conformal, uacqrs_agg=agg,
                     extraneous_quantiles=['mean'], var_name='batch_norm', var_list=['True'], 
                     uacqrs_bagging=uacqrs_bagging, metric=metric, max_normalization=standardize_response)
    with open(f'results/{dataset}_nn_params.txt', 'w') as f:
        print(params, file=f)
elif model_type =='linear':
    results = experiment(X=X, y=y, S=runs, B=B, fixed_model_params=params, model_type=model_type, fast_uacqr=False,
                     empirical_data_fraction=fractions[dataset], randomized_conformal=randomized_conformal, uacqrs_agg=agg,
                     uacqrs_bagging=uacqrs_bagging, metric=metric,
                     max_normalization=standardize_response)
date_today = datetime.date.today().strftime("%m.%d.%Y")
results.save(f"results/{dataset}_{model_type}_{runs}runs_{round(fractions[dataset],2)}fraction_log{log_response}_randomConformal{randomized_conformal}_cqrbagg{agg}_twosidedconformal{two_sided_conformal}_B={B}_{date_today}.pkl")
