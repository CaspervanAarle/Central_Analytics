# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 12:22:17 2021

@author: Casper
"""
import math
import matplotlib.pyplot as plt 
from datetime import datetime

from sklearn.metrics import r2_score, auc, roc_curve, RocCurveDisplay
from sklearn.linear_model import LogisticRegression, LinearRegression

import numpy as np

def r_squared(model, X, y_true):
    assert isinstance(model, LinearRegression)
    
    """ metric for linear regression """
    y_pred = model.predict(X)
    score = r2_score(y_true, y_pred)
    print("R-squared score: ", score)
    return score

def adjusted_r_squared(model, X, y_true):
    assert isinstance(model, LinearRegression)
    
    """ metric for linear regression """
    y_pred = model.predict(X)
    r2_sk = r2_score(y_true,y_pred)
    N=len(y_true)
    p=X.shape[1]
    x = (1-r2_sk)
    y = (N-1) / (N-p-1)
    adj_rsquared = (1 - (x * y))
    print("Adjusted-R-squared score: " , adj_rsquared)
    
    
    
    
def AUC(model, X, y_true):
    assert isinstance(model, LogisticRegression)
    """ metric for logistic regression """
    
    y_pred = model.predict_proba(X)[:,1]
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
    auc_score = auc(fpr, tpr)
    print("AUC score: ", auc_score)
    
def odds_ratio(model):
    assert isinstance(model, LogisticRegression)
    """ metric for logistic regression """
    
    ORs = []
    for coef in model.coef_:
        ORs.append(math.exp(coef))
    if(model.coef_.shape[0] > 1):
        print("Adjusted Odds Ratio: ", ORs)
    else:
        print("Crude Odds Ratio: ", ORs)
    return ORs
        
def ROC_curve(model, X, y_true):
    assert isinstance(model, LogisticRegression)
    """ metric for logistic regression """
    
    y_pred = model.predict_proba(X)[:,1]
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
    auc_score = auc(fpr, tpr)
    display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc_score)
    display.plot()  
    plt.savefig('results_ROCcurve\\ROC_' +  datetime.now().strftime("%Y%m%d_%H%M%S") + '.png')
    plt.show() 
    
def confidence_interval_linreg(model, config, X, y_true):
    for i, coef in enumerate(model.coef_):
        S = np.sqrt(np.sum((model.predict(X)-y_true)**2))
        CI_upper = coef + 1.96*S/np.sqrt(config["n_nodes"])
        CI_lower = coef - 1.96*S/np.sqrt(config["n_nodes"])
        print("Confidence Interval for variable: ", i, "(", str(CI_upper), str(CI_lower), ")")
        
def confidence_interval_logreg(model, config, X, y_true):
    for i, coef in enumerate(model.coef_[0]):
        S = np.sqrt(np.sum((model.predict(X)-y_true)**2))
        CI_upper = np.exp(coef + 1.96*S)
        CI_lower = np.exp(coef - 1.96*S)
        print("Confidence Interval for variable: ", i+1, "(", str(CI_upper), str(CI_lower), ")")
        
def confidence_interval_logreg2(model, config, X, y_true):
    for i, coef in enumerate(model.coef_[0]):
        S = np.sqrt(np.sum((model.predict(X)-y_true)**2))
        CI_upper = np.exp(coef + 1.96   )
        CI_lower = np.exp(coef - 1.96*S/np.sqrt(config["n_nodes"]))
        print("Confidence Interval for variable: ", i+1, "(", str(CI_upper), str(CI_lower), ")")
        
        
def show_coef_(model):
    coef = model.coef_
    print(coef)
    if len(np.array(model.coef_).shape) > 1:
        coef = model.coef_[0]
    print("Model coefficients:")
    for c in coef:
        print(c)


def show_intercept_(model):
    print(model.intercept_)



















