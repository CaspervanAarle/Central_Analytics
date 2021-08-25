# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 13:59:21 2021

@author: Casper
"""
import numpy as np
import json
import pickle
import pandas as pd

from sklearn.linear_model import LogisticRegression, LinearRegression

def load(model_dir, model_name): 
    model, config = _load_model(model_dir, model_name)
    if 'data_dir' in config:
        data_dir = config['data_dir']
    else:
        data_dir = input("what is the data source?\n")
    X,y = _load_data(config, data_dir)
    print(config)
    if('mu_list' in config and 'sigma_list' in config):
        X_mu = config['mu_list'][:np.array(X).shape[1]]
        X_sigma = config['sigma_list'][:np.array(X).shape[1]]
        X = _standardize(X, X_mu, X_sigma)
        print("input standardized")
        ### logistic regression only input standardized; linear regression all standardized
        if isinstance(model, LinearRegression):
            print("output standardized")
            y = _standardize(y, config['mu_list'][-1], config['sigma_list'][-1])
            
    y = np.array(y).flatten()
    return model, X, y, config
    

def _load_model(model_dir, name):
    with open(model_dir + name + '.json') as json_file:
        config = json.load(json_file)
        print("Loading model with hyperparameters:")
        print(config)
    return pickle.load(open(model_dir + name + '.sav', 'rb')), config


def _load_data(config, data_dir):    
    X = pd.read_csv(data_dir)[config['input_vars']].values.tolist()[:config['n_nodes']]
    y = pd.read_csv(data_dir)[config['target_vars']].values.tolist()[:config['n_nodes']]
    print("input dim: {}".format(np.array(X).shape))
    print("target dim: {}".format(np.array(y).shape))
    return X,y
    

def _standardize(X, mu, sigma):
    return (np.array(X) - np.array(mu)) / np.array(sigma)