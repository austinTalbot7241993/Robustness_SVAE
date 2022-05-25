'''
This is the unit tests for canonical correlation analysis with more than 
two groups.

'''

# Author : Austin Talbot <austin.talbot1993@gmail.com>
# Corey Keller
# Scott Linderman

import numpy as np
import numpy.random as rand
import numpy.linalg as la
import tensorflow as tf
from scipy import stats as st
import sys,os
from scipy.linalg import block_diag
from numpy.random import multivariate_normal
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as LR

sys.path.append('/Users/austin/Utilities/Code/Miscellaneous')
sys.path.append('/home/austin/Utilities/Code/Miscellaneous')
from utils_unitTest import tolerance,greater_than,message

sys.path.append('/home/austin/Utilities/Code/Numpy')
sys.path.append('/Users/austin/Utilities/Code/Numpy')
from utils_metrics_np import cosine_similarity_np
from utils_gaussian_np import mvn_loglikelihood_np
from utils_gaussian_np import mvn_conditional_distribution_np
from utils_matrix_np import subset_square_matrix_np
from utils_boxcar_np import boxcarAverage

sys.path.append('../Code/Comparison_Models')
from cca_model import CCA_multi

rand.seed(1993)

def mat_L1(X,Y):
    diff = np.abs(X-Y)
    return np.mean(diff)

def generate_data(n_groups=4):
    N = 1000000 #Samples per group
    eps = .3
    p_list = rand.randint(10,high=60,size=n_groups)
    Ls = 1 # Shared components
    Li = 1 # Individual components
    p = int(np.sum(p_list))
    B_true_list = [.2*rand.randn(Li,p_list[i]) for i in range(n_groups)]
    W_true = rand.randn(Ls,int(np.sum(p_list)))
    cov = eps*np.eye(p)
    cov = cov + np.dot(W_true.T,W_true)
    p_list2 = np.zeros(n_groups+1)
    p_list2[1:] = p_list
    for i in range(n_groups):
        st = int(np.sum(p_list2[:i+1]))
        nd = int(np.sum(p_list2[:(i+2)]))
        a = cov[st:nd,st:nd]
        bbx = np.dot(B_true_list[i].T,B_true_list[i])
        cov[st:nd,st:nd] = a + bbx
    X = multivariate_normal(np.zeros(p),cov,size=N)
    idxs = np.zeros(p)
    count = 0
    for i in range(n_groups):
        idxs[count:count+p_list[i]] = i
        count += p_list[i]

    return X,W_true,B_true_list,cov,idxs,eps
    
def test_init():
    print('####################################')
    print('# Testing CCA_multi init and print #')
    print('####################################')
    model = CCA_multi(1)
    print('Model initialized')
    print(model)

def test_CCA_multi_fit(X,W_true,B_true_list,cov,idxs,eps):
    print('###########################')
    print('# Testing SPCA_single fit #')
    print('###########################')
    #training_dict = {'n_iterations':5000}
    training_dict = {'n_iterations':50}
    model = CCA_multi(1,training_dict=training_dict)
    ngroups = len(np.unique(idxs))
    model.fit(X,idxs)#,comp_list=np.ones(ngroups))

    W_ = model.W_
    B_list = model.B_list_
    sigma2 = model.sigma2_

    '''
    ba = boxcarAverage(model.likelihood_tot)
    plt.plot(ba)
    plt.show()
    plt.plot(ba[-1000:])
    plt.show()
    #'''

    csim = cosine_similarity_np(W_,W_true)
    #tolerance(np.abs(csim)-1,1e-2,'Cosine similarity W check')
    diff_abs = np.abs(la.norm(W_)-la.norm(W_true))
    #print(la.norm(W_)-la.norm(W_true),la.norm(W_))
    #tolerance(diff_abs,1e-2,'Norm W check')

    for i in range(len(B_list)):
        csim = cosine_similarity_np(B_list[i],B_true_list[i])
        myStr = 'Cosine similarity space %d check'%i
        #tolerance(np.abs(csim)-1,1e-2,myStr)
        diff_abs = np.abs(la.norm(B_list[i])-la.norm(B_true_list[i]))
        myStr = 'Norm space %d check'%i
        #tolerance(diff_abs,1e-2,myStr)
        print(la.norm(B_list[i])-la.norm(B_true_list[i]),la.norm(B_list[i]))
    
    diff = np.abs(sigma2-eps)
    #tolerance(diff,1e-2,'Conditional noise check')
    return model

def test_CCA_multi_prec_cov(model,X,W_true,B_true_list,cov,idxs,eps):
    print('############################################')
    print('# Testing get_precision and get_covariance #')
    print('############################################')
    print(model.score_total(X))
    p = X.shape[1]
    cov_total = model.get_total_covariance()
    cov_shared_t = np.dot(W_true.T,W_true) + eps*np.eye(p)
    cov_shared = model.get_shared_covariance()

    prec_total = model.get_total_precision()
    prec_shared = model.get_shared_precision()

    ecvt = mat_L1(cov_total,cov)
    prod_total = np.dot(prec_total,cov)
    eye_true = np.eye(p)
    epct = mat_L1(eye_true,prod_total)
    tolerance(ecvt,1e-2,'Covariance total check')
    tolerance(epct,1e-2,'Precision total check')

    ecvs = mat_L1(cov_shared,cov_shared_t)
    prod_shared = np.dot(prec_shared,cov_shared_t)
    epcs = mat_L1(eye_true,prod_shared)
    tolerance(ecvs,1e-2,'Covariance shared check')
    tolerance(epcs,1e-2,'Precision shared check')


def test_CCA_multi_transform(model,X,W_true,B_true_list,cov,idxs,eps):
    print('#############################')
    print('# Testing transform methods #')
    print('#############################')

def test_CCA_multi_score(model,X,W_true,B_true_list,cov,idxs,eps):
    print('#########################')
    print('# Testing score methods #')
    print('#########################')
    p = X.shape[1]
    cov_shared_t = np.dot(W_true.T,W_true) + eps*np.eye(p)

    score_t = model.score_total(X)
    score_t_true = mvn_loglikelihood_np(X,cov)

    score_s = model.score_shared(X)
    score_s_true = mvn_loglikelihood_np(X,cov_shared_t)

    diff = np.abs(score_t-score_t_true)
    tolerance(diff,1e-2,'Score total check')

    diff = np.abs(score_t-score_t_true)
    tolerance(diff,1e-2,'Score total check')


def test_CCA_multi_predict(model,X,W_true,B_true_list,cov,idxs,eps):
    print('###########################')
    print('# Testing predict methods #')
    print('###########################')

    p = X.shape[1]
    model.W_ = W_true.T
    #model.B_list_ = B_true_list
    for i in range(len(B_true_list)):
        model.B_list_[i] = B_true_list[i].T

    model.sigma2_ = eps

    idx_sub1 = np.zeros(p)
    idx_sub2 = np.zeros(p)
    idx_sub1[:10] = 1
    idx_sub2[10:] = 1

    model_sklearn = LR()
    model_sklearn.fit(X[:,idx_sub1==0],X[:,idx_sub1==1])
    Y_hat1 = model_sklearn.predict(X[:,idx_sub1==0])

    model_sklearn2 = LR()
    model_sklearn2.fit(X[:,idx_sub2==0],X[:,idx_sub2==1])
    Y_hat2 = model_sklearn2.predict(X[:,idx_sub2==0])

    Y_hat1_model = model.predict(X[:,idx_sub1==0],idx_sub1)
    Y_hat2_model = model.predict(X[:,idx_sub2==0],idx_sub2)

    diff1 = np.mean(np.abs(Y_hat1-Y_hat1_model))
    diff2 = np.mean(np.abs(Y_hat2-Y_hat2_model))
    tolerance(diff1,1e-2,'First prediction check')
    tolerance(diff2,1e-2,'First prediction check')



if __name__ == "__main__":
    X,W_true,B_true_list,cov,idxs,eps = generate_data()
    test_init()
    model = test_CCA_multi_fit(X,W_true,B_true_list,cov,idxs,eps)
    test_CCA_multi_prec_cov(model,X,W_true,B_true_list,cov,idxs,eps)
    test_CCA_multi_transform(model,X,W_true,B_true_list,cov,idxs,eps)
    test_CCA_multi_score(model,X,W_true,B_true_list,cov,idxs,eps)
    test_CCA_multi_predict(model,X,W_true,B_true_list,cov,idxs,eps)











