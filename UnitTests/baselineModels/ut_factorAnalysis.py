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
from spca_model_ml import SPCA
from spca_model_ml import FactorAnalysis

sys.path.append('/home/austin/Utilities/Code/Numpy')
sys.path.append('/Users/austin/Utilities/Code/Numpy')
sys.path.append('/home/austin/Utilities/Code/Miscellaneous')
sys.path.append('/Users/austin/Utilities/Code/Miscellaneous')
from utils_metrics_np import cosine_similarity_np
from utils_unitTest import greater_than,tolerance
from utils_gaussian_np import mvn_loglikelihood_np

def generate_data():
    N = 100000
    L = 4
    p = 30
    alpha = 4
    beta = 4
    lamb = .25*st.gamma(3,1,3).rvs(4)
    lamb = np.sort(lamb)
    lamb = lamb[::-1]
    S = rand.randn(N,L)
    W_ = st.ortho_group.rvs(p)
    W_ = W_[:L]
    W_ = np.transpose(W_.T*lamb)
    W_ = 20*W_
    sigmas = st.invgamma(3,0,3).rvs(p)
    ig = st.invgamma(alpha,0,beta)
    phi = ig.rvs(p)#*.01
    X_noise = rand.randn(N,p)*phi
    X_hat = np.dot(S,W_)
    X = X_hat + X_noise
    return X,S,W_,phi

def test_factor_init():
    print('#########################################')
    print('# Testing FactorAnalysis init and print #')
    print('#########################################')
    print('Initializing model')
    training_dict = {'n_iterations':7500}
    model = FactorAnalysis(4,training_dict=training_dict)
    print('Model initialized')
    print(model)

def test_factor_fit(X_factor,S_factor,W_,phi):
    print('##############################')
    print('# Testing FactorAnalysis fit #')
    print('##############################')
    print('Fitting model')
    training_dict = {'n_iterations':7500}
    L = 4

    model = FactorAnalysis(L,training_dict=training_dict)
    var_dict = 'sklearn'
    model.fit(X_factor,var_dict=var_dict)
    sigma2 = model.sigma_
    diff = sigma2-phi**2

    #Check noise variables
    tolerance(np.mean(np.abs(diff)),1e-2,'FactorAnalysis noise check')

    cov_true = np.dot(W_.T,W_) + np.diag(phi**2)
    cov_est = model.get_covariance()
    diff = cov_est - cov_true

    #Check factor orientations
    print('>>>>>')
    for i in range(L):
        csi = cosine_similarity_np(model.W_[i],W_[i])
        tolerance(np.abs(1-np.abs(csi)),1e-3,'Factor %d Csim check'%i)

    #Check likelihood
    print('>>>>>')
    est_ll = model.score(X_factor)
    model.sigma_ = phi**2
    model.W_ = W_
    true_ll = model.score(X_factor)
    tolerance(np.abs(true_ll-est_ll),1e-1,'Likelihood check')

if __name__ == "__main__":
    X_factor,S_factor,W_,phi = generate_data()

    test_factor_init()
    test_factor_fit(X_factor,S_factor,W_,phi)











