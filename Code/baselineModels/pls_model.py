'''

Author : Austin Talbot <austin.talbot1993@gmail.com>

'''
from math import log,sqrt
import numbers
import logging

import numpy as np
from scipy import linalg as la
from scipy.special import gammaln
from numpy import random as rand
from tqdm import trange

from sklearn.utils import check_array, check_random_state
from sklearn.utils.extmath import randomized_svd,fast_logdet,svd_flip
from sklearn.utils.extmath import stable_cumsum
from sklearn.utils.validation import check_is_fitted
from sklearn import decomposition as dp
from sklearn import linear_model as lm

from datetime import datetime as dt
import os,sys,time
from scipy.io import savemat
import matplotlib.pyplot as plt 

import tensorflow as tf
from tensorflow import keras
import pickle
from pathlib import Path

homeDir = str(Path.home())
sys.path.append(homeDir + '/Utilities/Code/Miscellaneous')
sys.path.append(homeDir + '/Utilities/Code/Numpy')
sys.path.append(homeDir + '/Utilities/Code/Tensorflow')
from utils_misc_tf import limitGPU,return_optimizer_tf
from utils_misc import fill_dict,pretty_string_dict
from utils_batcher_np import simple_batcher_X,simple_batcher_XY
from utils_losses_tf import L2_loss_tf
from utils_gaussian_np import mvn_conditional_distribution_np
from utils_gaussian_tf import mvn_loglikelihood_tf
from utils_matrix_tf import block_diagonal_square_tf
from utils_gaussian_np import mvn_loglikelihood_np
from utils_matrix_np import woodbury_inverse_sym_np
from utils_matrix_np import subset_square_matrix_np

from cca_base import base

version = "1.0"

class PLS(base):
    '''
    Partial least squares as described in Murphy 2010

    '''
    
    def __init__(self,n_components_s,n_components_x=1,gpu_memory=1024,
                    training_dict={},prior_dict={}):
        '''
        Paramters
        ---------
        n_components_s : int
            Shared latent space dimension

        n_components_x : int
            Latent space dimension reserved for X

        '''
        self.n_components_s = int(n_components_s)
        self.n_components_x = int(n_components_x)

        self.training_dict = self._fillTrainingOpsDict(training_dict)
        self.prior_dict = self._fillPriorOptsDict(prior_dict)

        self.version = version
        self.creationDate = dt.now()
        self.gpu_memory = int(gpu_memory)

    def __repr__(self):
        out_str = 'PLS object\n'
        out_str = out_str + 'n_components_s=%d\n'%self.n_components_s
        out_str = out_str + 'n_components_x=%d\n'%self.n_components_x
        out_str = out_str + '>>>>>>>>>>>>\n'
        out_str = out_str + 'Training Parameters:\n'
        out_str = out_str + pretty_string_dict(self.training_dict)
        out_str = out_str + '>>>>>>>>>>>>\n'
        out_str = out_str + 'Prior Parameters:\n'
        out_str = out_str + pretty_string_dict(self.prior_dict)
        return out_str

    def _fillPriorOptsDict(self,prior_dict):
        '''
        Fills in parameters used for prior of parameters

        Paramters
        ---------
        prior_dict: dictionary
            The prior parameters used to specify the prior

        Options
        -------
        sig_BX: float, L2 penalization on Bx
        sig_WX: float, L2 penalization on Wx
        sig_BY: float, L2 penalization on By
        sig_WY: float, L2 penalization on Wy

        '''
        return {'sig_BX':0.01,'sig_WX':0.01,'sig_WY':0.01}

    def _prior(self,training_variables):
        '''
        Evaluates the prior of the parameters

        Paramters
        ---------
        training_variables:
            List of variables we are optimizing

        Returns
        -------
        log_prior: tf.float
            The prior of the parameters

        '''
        pd = self.prior_dict
        term1 = pd['sig_BX']*L2_loss_tf(training_variables[0])
        #term2 = pd['sig_WX']*L2_loss_tf(training_variables[1])
        #term3 = pd['sig_WY']*L2_loss_tf(training_variables[3])
        #prior = -1*(term1 + term2 + term3)
        prior = -1*term1
        return prior

    def _initializeVars(self,data=None,var_dict=None):
        '''
        Initializes the variables of the model

        var_dict: keys {W_x,W_y,B_x,B_y}
            optional dictionary of initialized matrices

        Returns
        -------
        W_x : tf.Variable, array-like (n_features_x,n_components_s)
            The shared loadings for x

        W_y : tf.Variable, array-like (n_features_y,n_components_s)
            The shared loadings for y

        B_x : tf.Variable, array-like (n_features_x,n_components_)
            The exclusive loadings for x

        '''
        Ls,Lx = self.n_components_s,self.n_components_x
        if var_dict is None:
            p,q = self.dims
            W_ = tf.Variable(rand.randn(p+q,Ls).astype(np.float32))
            B_x = tf.Variable(rand.randn(p,Lx).astype(np.float32))
        else:
            W_ = tf.Variable(var_dict['W_'].astype(np.float32))
            B_x = tf.Variable(var_dict['B_x'].astype(np.float32))
        sigmal_ = tf.Variable(0.0)
        return W_,B_x,sigmal_

    def fit(self,X,Y,var_dict=None):
        '''
        Fits the model given X, Y and optional initialization parameters

        Paramters
        ---------
        X : array-like (n_samples,p)
            Data from population 1

        Y : array-like (n_samples,q)
            Data from population 2

        var_dict: keys {W_x,W_y,B_x}
            optional dictionary of initialized matrices
        '''
        limitGPU(self.gpu_memory)
        X,Y = self._convertData([X,Y])
        V = np.hstack((X,Y))
        N,p,q = X.shape[0],self.dims[0],self.dims[1]
        Ls,Lx=self.n_components_s,self.n_components_x
        td = self.training_dict

        trainable_variables = self._initializeVars(var_dict=var_dict)
        W_ = trainable_variables[0]
        B_x = trainable_variables[1]
        sigmal_ = trainable_variables[2]

        #Constants for constructing the matrix on (12.94) Murphy 2010
        zeros_Y = tf.constant(np.zeros((q,q)).astype(np.float32))
        eye = tf.constant(np.eye(np.sum(self.dims)).astype(np.float32))

        optimizer = return_optimizer_tf(td['method'],td['learning_rate'])
        self._initializeSavedVariables()

        for i in trange(td['n_iterations']):
            V_batch = simple_batcher_X(td['batch_size'],V)
            with tf.GradientTape() as tape:
                Bsquare = tf.matmul(B_x,tf.transpose(B_x))
                B_block = block_diagonal_square_tf([Bsquare,zeros_Y])
                WWT = tf.matmul(W_,tf.transpose(W_))
                sigma2 = tf.nn.softplus(sigmal_)
                Sigma = B_block + WWT + sigma2*eye

                like_prior = self._prior(trainable_variables)

                like_tot = mvn_loglikelihood_tf(V_batch,Sigma)
                posterior = like_tot + 1/N*like_prior

                Sigma_X = Sigma[:p,:p]
                Sigma_Y = Sigma[p:,p:]

                like_x = mvn_loglikelihood_tf(V_batch[:,:p],Sigma_X)
                like_y = mvn_loglikelihood_tf(V_batch[:,p:],Sigma_Y)

                loss = -1*posterior
            
            gradients = tape.gradient(loss,trainable_variables)
            optimizer.apply_gradients(zip(gradients,trainable_variables))

            self._saveLosses(i,like_prior,like_tot,posterior,like_x,
                                            like_y,sigma2)

        self._saveVariables(trainable_variables)

    def _initializeSavedVariables(self):
        td = self.training_dict
        self.likelihood_prior = np.zeros(td['n_iterations'])
        self.likelihood_tot = np.zeros(td['n_iterations'])
        self.likelihood_X = np.zeros(td['n_iterations'])
        self.likelihood_Y = np.zeros(td['n_iterations'])
        self.list_posterior = np.zeros(td['n_iterations'])
        self.sigmas = np.zeros(td['n_iterations'])
        
    def _saveLosses(self,i,like_prior,like_tot,posterior,like_x,
                                            like_y,sigma2):
        self.likelihood_prior[i] = like_prior.numpy()
        self.likelihood_tot[i] = like_tot.numpy()
        self.likelihood_X[i] = like_x.numpy()
        self.likelihood_Y[i] = like_y.numpy()
        self.list_posterior[i] = posterior.numpy()
        self.sigmas[i] = sigma2.numpy()
    
    def _saveVariables(self,tv):
        self.W_ = tv[0].numpy()
        self.B_x_ = tv[1].numpy()
        sigma2 = tf.nn.softplus(tv[2])
        self.sigma2_ = sigma2.numpy()

    def _constructWTot(self):
        '''

        '''
        p = np.sum(self.dims)
        dimX = np.sum(self.n_components_s+self.n_components_x)
        W_tot = np.zeros((p,dimX))
        if self.n_components_s == 1:
            W_tot[:,0] = np.squeeze(self.W_)
        else:
            W_tot[:,:self.n_components_s] = self.W_

        W_tot[:self.dims[0],self.n_components_s:] = self.B_x_
        return W_tot

    def get_covariance(self):
        '''
        Gets the covariance matrix WW^T + sigma^2I defined by the model

        Returns
        -------
        cov : np.array-like (sum(),sum())
            The covariance matrix
        '''
        W = self._constructWTot()
        p = np.sum(self.dims)
        cov = self.sigma2_*np.eye(p) + np.dot(W,W.T)
        return cov

    def get_precision(self):
        W = self._constructWTot()
        p = np.sum(self.dims)
        Dinv = 1/self.sigma2_*np.eye(p)
        precision = woodbury_inverse_sym_np(Dinv,W)
        return precision
        
    def get_covariance_shared(self):
        p = np.sum(self.dims)
        cov = self.sigma2_*np.eye(p) + np.dot(self.W_,self.W_.T)
        return cov

    def get_precision_shared(self):
        p = np.sum(self.dims)
        Dinv = 1/self.sigma2_*np.eye(p)
        precision = woodbury_inverse_sym_np(Dinv,self.W_)
        return precision

    def score(self,X,Y):
        XX = np.hstack((X,Y))
        cov = self.get_covariance()
        score = mvn_loglikelihood_np(XX,cov)
        return score

    def scoreX(self,X):
        p = self.dims[0]
        Wx = self.W_[:p,:]
        Bx = self.B_x_
        cov = self.sigma2_*np.eye(p) + np.dot(Wx,Wx.T) + np.dot(Bx,Bx.T)
        score = mvn_loglikelihood_np(X,cov)
        return score

    def scoreY(self,Y):
        p = self.dims[0]
        Wy = self.W_[p:,:]
        cov = self.sigma2_*np.eye(self.dims[1]) + np.dot(Wy,Wy.T)
        score = mvn_loglikelihood_np(Y,cov)
        return score

    def score_shared(self,X,Y):
        cov = self.get_covariance_shared()
        XX = np.hstack((X,Y))
        score = mvn_loglikelihood_np(XX,cov)
        return score

    def predictY(self,X):
        idx_miss = np.ones(np.sum(self.dims))
        idx_miss[:self.dims[0]] = 0
        coef = self._get_predictiveCoefficients(idx_miss)
        Y_hat = np.dot(X,coef.T)
        return Y_hat

    def _get_predictiveCoefficients(self,idx_miss,idx_obs=None):
        '''
        Gets the predictive coefficients predicting idx_miss using 
        idx_obs

        Parameters
        ----------
        idx_miss : np.array-like(p,)
            The variables to predict, entries 0,1

        idx_miss : np.array-like(p,) or None
            The observed variables, default to not predicted

        Returns
        -------
        coef : 
            The regression coefficients
        '''
        if idx_obs == None:
            idx_obs = 1 - idx_miss

        idx_tot = idx_obs + idx_miss

        cov = self.get_covariance()
        cov_sub = subset_square_matrix_np(cov,idx_tot==1)

        idx_o = idx_obs[idx_tot==1]
        coef,_ = mvn_conditional_distribution_np(cov_sub,idx_o)
        return coef

