import numpy as np
import pickle
import matplotlib.pyplot as plt
import numpy.random as rand
import numpy.linalg as la
import tensorflow as tf
from scipy import stats as st

import sys,os

sys.path.append('/home/austin/Linderman_Lab/Statistical_Dropout/Code/Comparison_Models')
sys.path.append('/Users/austin/Statistical_Dropout/Code/Comparison_Models')
from pls_model import PLS
sys.path.append('/home/austin/Utilities/Code/Numpy')
sys.path.append('/Users/austin/Utilities/Code/Numpy')
sys.path.append('/home/austin/Utilities/Code/Miscellaneous')
sys.path.append('/Users/austin/Utilities/Code/Miscellaneous')
from utils_metrics_np import cosine_similarity_np
from utils_gaussian_np import mvn_loglikelihood_np
from utils_unitTest import greater_than,tolerance
from utils_gaussian_np import mvn_conditional_distribution_np
from utils_matrix_np import subset_square_matrix_np

from sklearn.linear_model import LinearRegression as LR

rand.seed(1993)

def generate_data(Ls,Lx,eps=.3):
    N = 1000000
    p = 60
    q = 20

    B_true_x = rand.randn(Lx,p)*.5
    W_true_x = rand.randn(Ls,p)
    W_true_y = rand.randn(Ls,q)
    S_shared = rand.randn(N,Ls)
    S_x = rand.randn(N,Lx)

    X_hat_s = np.dot(S_shared,W_true_x)
    Y_hat_s = np.dot(S_shared,W_true_y)
    X_hat_x = np.dot(S_x,B_true_x)

    X_hat = X_hat_s + X_hat_x
    Y_hat = Y_hat_s 

    X_noise = eps*rand.randn(N,p)
    Y_noise = eps*rand.randn(N,q)
    
    X = X_hat + X_noise
    Y = Y_hat + Y_noise

    V = np.hstack((X,Y))

    cov = np.cov(V.T)
    sigma = eps**2
    return X,Y,W_true_x,W_true_y,B_true_x,cov,sigma,Y_hat

def test_init():
    print('###############################')
    print('# Testing PPCA init and print #')
    print('###############################')
    print('Initializing model')
    model = PLS(1)
    print('Model initialized')
    print(model)

def test_fit():
    print('####################')
    print('# Testing PPCA fit #')
    print('####################')
    Ls = 1
    Lx = 1
    X,Y,W_true_x,W_true_y,B_true_x,cov,sigma,Y_hat = generate_data(Ls,Lx)

    training_dict = {'n_iterations':15000,'learning_rate':6e-4}
    var_dict = {}
    var_dict['W_'] = np.vstack((W_true_x.T,W_true_y.T))
    var_dict['B_x'] = B_true_x.T

    model = PLS(Ls,n_components_x=Lx,training_dict=training_dict)
    model.fit(X,Y)#var_dict=var_dict)

    B_est_x = np.squeeze(model.B_x_)
    W_est_x = np.squeeze(model.W_[:X.shape[1]])
    W_est_y = np.squeeze(model.W_[X.shape[1]:])

    csm_bx = cosine_similarity_np(B_est_x,B_true_x)
    csm_wx = cosine_similarity_np(W_est_x,W_true_x)
    csm_wy = cosine_similarity_np(W_est_y,W_true_y)
    diff_s = np.abs(model.sigma2_-sigma)
    tolerance(np.abs(1-np.abs(csm_bx)),3e-2,'Fitting Bx check')
    tolerance(np.abs(1-np.abs(csm_wx)),3e-2,'Fitting Wx check')
    tolerance(np.abs(1-np.abs(csm_wy)),3e-2,'Fitting Wy check')
    tolerance(diff_s,1e-2,'Fitting sigma2 check')

def test_transform():
    print('#########################')
    print('# Testing PLS transform #')
    print('#########################')
    Ls = 1
    Lx = 1
    X,Y,W_true_x,W_true_y,B_true_x,cov,sigma,Y_hat = generate_data(Ls,Lx)
    W_tot = np.vstack((W_true_x.T,W_true_y.T))
    model = PLS(Ls,n_components_x=Lx,training_dict=training_dict)
    model.W_ = W_tot
    model.sigma2_ = sigma
    model.B_x_ = B_true_x.T

def test_score():
    print('###################################################')
    print('# Testing PLS score, scoreX, scoreY, score_shared #')
    print('###################################################')
    Ls = 2
    Lx = 2
    X,Y,W_true_x,W_true_y,B_true_x,cov,sigma,Y_hat = generate_data(Ls,Lx)
    model = PLS(Ls,n_components_x=Lx)
    XX = np.hstack((X,Y))
    W_tot = np.vstack((W_true_x.T,W_true_y.T))
    model = PLS(Ls,n_components_x=Lx)
    model.dims = np.array([X.shape[1],Y.shape[1]])
    model.W_ = W_tot
    model.sigma2_ = sigma
    model.B_x_ = B_true_x.T

    eyep = np.eye(cov.shape[0])
    cov_shared = sigma*eyep + np.dot(W_tot,W_tot.T)

    idx_x = np.ones(XX.shape[1])
    idx_x[X.shape[1]:] = 0

    mod_score = model.score(X,Y)
    score_true = mvn_loglikelihood_np(XX,cov)
    diff = np.abs(mod_score-score_true)
    tolerance(diff,1e-2,'score check')

    mod_scorex = model.scoreX(X)
    mod_scorey = model.scoreY(Y)
    scorex_true = mvn_loglikelihood_np(X,
                                subset_square_matrix_np(cov,idx_x))
    scorey_true = mvn_loglikelihood_np(Y,
                                subset_square_matrix_np(cov,1-idx_x))

    tolerance(np.abs(mod_scorex-scorex_true),1e-2,'scoreX check')
    tolerance(np.abs(mod_scorey-scorey_true),1e-2,'scoreY check')

    mod_score_shared = model.score_shared(X,Y)
    score_shared_true = mvn_loglikelihood_np(XX,cov_shared)
    diff = np.abs(score_shared_true-mod_score_shared)
    tolerance(diff,1e-2,'score_shared check')


def test_covariance():
    print('#####################################################')
    print('# Testing PLS get_covariance, get_covariance_shared #')
    print('#             get_precision, get_precision_shared   #')
    print('#####################################################')
    Ls = 2
    Lx = 2
    X,Y,W_true_x,W_true_y,B_true_x,cov,sigma,Y_hat = generate_data(Ls,Lx)
    model = PLS(Ls,n_components_x=Lx)
    XX = np.hstack((X,Y))
    W_tot = np.vstack((W_true_x.T,W_true_y.T))
    model = PLS(Ls,n_components_x=Lx)
    model.dims = np.array([X.shape[1],Y.shape[1]])
    model.W_ = W_tot
    model.sigma2_ = sigma
    model.B_x_ = B_true_x.T

    cov_est = model.get_covariance()
    cov_est_shared = model.get_covariance_shared()
    prec_est = model.get_precision()
    prec_est_shared = model.get_precision_shared()

    cp_prod = np.dot(cov_est,prec_est)
    cps_prod = np.dot(cov_est_shared,prec_est_shared)
    eyep = np.eye(cov.shape[0])
    cov_shared = sigma*eyep + np.dot(W_tot,W_tot.T)

    tolerance(np.mean(np.square(cov_est-cov)),1e-5,'get_covariance check')
    tolerance(np.mean(np.square(eyep-cp_prod)),1e-5,'get_precision check')
    tolerance(np.mean(np.square(cov_est_shared-cov_shared)),1e-5,
                                    'get_covariance_shared check')
    tolerance(np.mean(np.square(eyep-cps_prod)),1e-5,
                                    'get_precision_shared check')

def test_predictY():
    print('########################')
    print('# Testing PLS predictY #')
    print('########################')
    Ls = 2
    Lx = 2
    X,Y,W_true_x,W_true_y,B_true_x,cov,sigma,Y_hat = generate_data(Ls,Lx)
    model = PLS(Ls,n_components_x=Lx)
    XX = np.hstack((X,Y))
    W_tot = np.vstack((W_true_x.T,W_true_y.T))
    model = PLS(Ls,n_components_x=Lx)
    model.dims = np.array([X.shape[1],Y.shape[1]])
    model.W_ = W_tot
    model.sigma2_ = sigma
    model.B_x_ = B_true_x.T

    Y_hat_m = model.predictY(X)

    model_sklearn = LR()
    model_sklearn.fit(X,Y)
    Y_pred = model_sklearn.predict(X)

    diff = np.mean(np.square(Y_hat_m-Y_pred))
    tolerance(diff,1e-4,'predictY check')

if __name__ == "__main__":
    test_init()
    test_fit()
    #test_transform()
    test_score()
    test_covariance()
    test_predictY()












