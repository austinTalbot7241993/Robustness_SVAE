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
sys.path.append('/home/austin/Linderman_Lab/Statistical_Dropout/Code/Comparison_Models')
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

def generate_data(L=4,sigmax=1.0,sigmay=0.5,N=10000):
    p = 30
    q = 5

    W_x_base = st.ortho_group.rvs(p)
    W_y_base = st.ortho_group.rvs(q)
    Wx = W_x_base[:L]
    Wy = W_y_base[:L]
    lamb_x = np.sort(rand.gamma(1,1,size=L))[::-1] + 1
    lamb_y = rand.gamma(1,1,size=L) + 1
    for l in range(L):
        Wx[l] = Wx[l]*lamb_x[l]
        Wy[l] = Wy[l]*lamb_y[l]

    S_train = rand.randn(N,L)

    X_hat = np.dot(S_train,Wx)
    Y_hat = np.dot(S_train,Wy)

    X_noise = sigmax*rand.randn(N,p)
    Y_noise = sigmay*rand.randn(N,q)

    X_train = X_hat + X_noise
    Y_train = Y_hat + Y_noise

    return Wx,Wy,sigmax,sigmay,X_train,Y_train,S_train

def test_single_init():
    print('######################################')
    print('# Testing SPCA_single init and print #')
    print('######################################')
    print('Initializing model')
    model = SPCA(4)
    print('Model initialized')
    print(model)


def test_single_fit():
    print('###########################')
    print('# Testing SPCA_single fit #')
    print('###########################')
    print('Fitting model')
    Wx,Wy,sigmax,sigmay,X_train,Y_train,S_train = generate_data()
    training_dict = {'n_iterations':5000}
    L = 4
    model = SPCA(L,training_dict=training_dict)
    XX = np.hstack((X_train,Y_train))
    mod = dp.PCA(L)
    mod.fit(XX)
    #var_dict = {'W_':mod.components_,'sigmal_':-1.0}
    idxs = np.zeros(35)
    idxs[30:] = 1
    model.fit(XX,idxs)#,var_dict)
    W_est = model.W_
    sigma_ = model.sigma_
    print(sigma_)
    print('>>>>>>>')
    print('>>>>>>>')
    print(model.score(XX))
    print('>>>>>>>')
    print('>>>>>>>')
    model.sigma_[:30] = 1.0
    model.sigma_[30:] = .25
    model.W_[:,:30] = Wx
    model.W_[:,30:] = Wy
    print('>>>>>>>')
    print('>>>>>>>')
    print(model.score(XX))
    print('>>>>>>>')
    print('>>>>>>>')
    return model #Notice this returns the true parameters

def test_single_transform():
    print('#################################')
    print('# Testing SPCA_single transform #')
    print('#################################')
    Wx,Wy,sigmax,sigmay,X_train,Y_train,S_train = generate_data(L=4,
                                                sigmax=.1,sigmay=.1)
    XX = np.hstack((X_train,Y_train))
    idxs = np.zeros(35)
    idxs[30:] = 1
    L = 4
    training_dict = {'n_iterations':50}
    model = SPCA(L,training_dict=training_dict)
    model.fit(XX,idxs)

    model.sigma_[:30] = 1.0
    model.sigma_[30:] = .25
    model.W_[:,:30] = Wx
    model.W_[:,30:] = Wy
    
    S_hat = model.transform(XX)

    mdiff = np.mean((S_train-S_hat))
    mvar = np.mean((S_train-S_hat)**2)

    tolerance(np.abs(mdiff),1e-2,'Score mean check')
    tolerance(np.abs(mvar),2e-2,'Score variance check')

def test_single_miscellaneous():
    print('#####################################')
    print('# Testing SPCA_single other methods #')
    print('#####################################')
    Wx,Wy,sigmax,sigmay,X_train,Y_train,S_train = generate_data(N=1000000)
    L = 4
    XX = np.hstack((X_train,Y_train))
    idxs = np.zeros(35)
    idxs[30:] = 1
    W_ = np.hstack((Wx,Wy))

    cov_data = np.cov(XX.T)
    prec_data = la.inv(cov_data)

    training_dict = {'n_iterations':5}
    model = SPCA(L,training_dict=training_dict)
    model.fit(XX,idxs)

    model.sigma_[:30] = 1.0
    model.sigma_[30:] = .25
    model.W_ = W_

    # Checking get precision and covariance
    cov_est = model.get_covariance()
    prec_est = model.get_precision()
    prod = np.dot(prec_est,cov_data)
    cov_mean = np.mean(np.abs(cov_data-cov_est))
    prec_mean = np.mean(np.abs(prod-np.eye(35)))
    tolerance(cov_mean,1e-2,'get_covariance check')
    tolerance(prec_mean,1e-2,'get_precision check')

    # Checking transform subset, predict




if __name__ == "__main__":
    test_single_init()
    test_single_fit()
    #test_single_transform()
    #test_single_reconstruction()
    #test_single_miscellaneous()











