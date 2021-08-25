# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 11:14:09 2021

@author: Casper
"""

import pickle
import json
import pandas as pd
import numpy as np

from metrics import r_squared, adjusted_r_squared, odds_ratio, AUC, ROC_curve 
from metrics import confidence_interval_linreg, confidence_interval_logreg2
from metrics import show_coef_, show_intercept_
from load_model import load

model_name = "experiment_20210824_115600"

FEDERATED_DIR = "C:\\Users\\Casper\\Projects\\MasterScriptie\\custom_projects\\editing\\PHT_Server\\results\\"

if __name__ == "__main__":
    model, X, y, config = load(FEDERATED_DIR, model_name)
    #AUC(model, X, y)
    #ROC_curve(model, X, y)
    
    
    #odds_ratio(model)
    r2_score = r_squared(model, X, y)
    #adj_r2_score = adjusted_r_squared(model, X, y)
    show_coef_(model)