'''
This tests the methods defined in spca_model_ml.py
'''

# Author : Austin Talbot <austin.talbot1993@gmail.com>
# Corey Keller
# Scott Linderman

import numpy as np
import numpy.random as rand
import numpy.linalg as la
import tensorflow as tf
from scipy import stats as st
from sklearn import decomposition as dp
import matplotlib.pyplot as plt

import sys,os

rand.seed(1993)

sys.path.append('/Users/austin/Statistical_Dropout/Code/Comparison_Models')
from spca_model_ml import PPCA

sys.path.append('/home/austin/Utilities/Code/Numpy')
sys.path.append('/Users/austin/Utilities/Code/Numpy')
sys.path.append('/home/austin/Utilities/Code/Miscellaneous')
sys.path.append('/Users/austin/Utilities/Code/Miscellaneous')
from utils_metrics_np import cosine_similarity_np
from utils_unitTest import greater_than,tolerance
from utils_gaussian_np import mvn_loglikelihood_np

def generate_data():
    N = 10000
    L = 4
    p = 50
    sigma = 1.0

    W_base = st.ortho_group.rvs(p)
    W = W_base[:L]
    lamb = rand.gamma(1,1,size=L) + 1
    for l in range(L):
        W[l] = W[l]*lamb[l]

    S_train = rand.randn(N,L)
    X_hat = np.dot(S_train,W)
    X_noise = sigma*rand.randn(N,p)
    X_train = X_hat + X_noise

    return W,sigma,X_train,S_train


def test_init():
    print('###############################')
    print('# Testing PPCA init and print #')
    print('###############################')
    print('Initializing model')
    model = PPCA(4)
    print('Model initialized')
    print(model)


def test_fit(W,sigma,X_train,S_train):
    print('####################')
    print('# Testing PPCA fit #')
    print('####################')
    print('Fitting model')
    training_dict = {'n_iterations':5000}
    training_dict2 = {'n_iterations':5}
    L = 4
    model = PPCA(L,training_dict=training_dict)
    model2 = PPCA(L,training_dict=training_dict2)
    model.fit(X_train)
    model2.fit(X_train)

    W_est = model.W_
    sigma_ = model.sigma_

    model2.W_ = W
    model2.sigma_ = sigma*np.ones(X_train.shape[1])
    tolerance(model.sigma_[0]-sigma,1e-3,'Conditional variance check')
    ll_est = model.score(X_train)
    ll_true = model2.score(X_train)
    greater_than(ll_est,ll_true-.1,'Log Likelihood check')

if __name__ == "__main__":
    W,sigma,X_train,S_train = generate_data()
    test_init()
    test_fit(W,sigma,X_train,S_train)



