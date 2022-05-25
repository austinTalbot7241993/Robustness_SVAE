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
from utils_batcher_np import simple_batcher_X,simple_batcher_XY
from utils_gaussian_np import mvn_conditional_distribution_np
from utils_matrix_np import woodbury_inverse_sym_np
from utils_gaussian_np import mvn_loglikelihood_np
from utils_matrix_np import subset_square_matrix_np

sys.path.append(homeDir + '/Utilities/Code/Tensorflow')
from utils_misc_tf import limitGPU,return_optimizer_tf
from utils_misc import fill_dict,pretty_string_dict
from utils_losses_tf import L2_loss_tf
from utils_gaussian_tf import mvn_loglikelihood_tf
from utils_matrix_tf import block_diagonal_square_tf

from cca_base import base

version = "1.0"

class CCA_multi(base):
    '''
    The multi-group extension of cannonical correlation anaylsis. Note that
    the number of group-specific components is passed at training time

    '''
    
    def __init__(self,n_components_s,training_dict={},
                                prior_dict={},gpu_memory=1024):
        '''
        Paramters
        ---------
        n_components_s : int
            Shared latent space dimension

        '''
        self.n_components_s = int(n_components_s)

        self.training_dict = self._fillTrainingOpsDict(training_dict)
        self.prior_dict = self._fillPriorOptsDict(prior_dict)
        self.gpu_memory = int(gpu_memory)

        self.version = version
        self.creationDate = dt.now()

    def __repr__(self):
        out_str = 'CCA_multi object\n'
        out_str = out_str + 'n_components_s=%d\n'%self.n_components_s
        out_str = out_str + 'Training Parameters:\n'
        out_str = out_str + pretty_string_dict(self.training_dict)
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
        sig_W: float, L2 penalization on Wx

        '''
        return {'sig_B':0.01,'sig_W':0.01}

    def _initializeVars(self,X,var_dict=None):
        '''
        Initializes the variables of the model

        X :

        var_dict: dict, keys {W_init,B_list}, 'sklearn' or 'random'
            optional dictionary of initialized matrices or string. sklearn
            uses sklearn to initialize components while random is a 
            random initialization

        Returns
        -------
        var_list : list 
            List of tensorflow variables to optimize
        '''
        Ls = self.n_components_s
        ncomp = self.n_comps
        if var_dict == 'random':
            wi = rand.randn(self.p,Ls).astype(np.float32)
            W_ = tf.Variable(wi)
            Bs = []
            for i in range(self.n_groups):
                bi = rand.randn(self.qs[i],ncomp[i]).astype(np.float32)
                Bs.append(tf.Variable(bi))
        elif var_dict ==  'sklearn':
            pca_mod = dp.PCA(self.n_components_s)
            S_est = pca_mod.fit_transform(X)
            X_recon = pca_mod.inverse_transform(S_est)
            W_ = tf.Variable(pca_mod.components_.T.astype(np.float32))
            diff = X - X_recon
            Bs = []
            for i in range(self.n_groups):
                diff_sub = diff[:,self.groups==i]
                pca_sub = dp.PCA(int(self.n_comps[i]))
                pca_sub.fit(diff_sub)
                bi = pca_sub.components_.T.astype(np.float32)
                Bs.append(tf.Variable(bi))
        else:
            W_ = tf.Variable(var_dict['W_init'].astype(np.float32))
            Bs = [tf.Variable(var_dict['B_list'][i].astype(np.float32)) for i in range(self.n_groups)]
            
        sigmal_ = tf.Variable(-3.0)
        var_list = [W_] + Bs + [sigmal_]
        return var_list

    def fit(self,X,groups,comp_list=None,var_dict='sklearn'):
        '''
        Fits the model given X, Y and optional initialization parameters

        Paramters
        ---------
        X : array-like (n_samples,p)
            Data from all populations 

        groups : 

        comp_list : list

        var_dict: keys {Winit,B_list}
            optional dictionary of initialized matrices
        '''
        limitGPU(self.gpu_memory)
        self.n_groups = len(np.unique(groups))
        self.groups = groups
        self.qs = np.zeros(self.n_groups)
        for i in range(self.n_groups):
            self.qs[i] = np.sum(groups==i)
        X = X.astype(np.float32)
        N,self.p = X.shape
        if comp_list is None:
            self.n_comps = np.ones(self.n_groups)
        else:
            self.n_comps = np.squeeze(np.array([comp_list])).astype(int)
            
        trainable_variables = self._initializeVars(X,var_dict)
        W_ = trainable_variables[0]
        sigmal_ = trainable_variables[-1]
        Blist = trainable_variables[1:-1]

        eye = tf.constant(np.eye(self.p).astype(np.float32))

        td = self.training_dict
        optimizer = return_optimizer_tf(td['method'],td['learning_rate'])

        self._initializeSavedVariables()

        for i in trange(td['n_iterations']):
            X_batch = simple_batcher_X(td['batch_size'],X)
            with tf.GradientTape() as tape:
                Bsquares = [tf.matmul(B,tf.transpose(B)) for B in Blist]
                B_block = block_diagonal_square_tf(Bsquares)
                WWT = tf.matmul(W_,tf.transpose(W_)) 
                sigma2 = tf.nn.softplus(sigmal_)
                Sigma = B_block + WWT + sigma2*eye

                like_prior = self._prior(trainable_variables)

                like_tot = mvn_loglikelihood_tf(X_batch,Sigma)
                posterior = like_tot #+ 1/N*like_prior

                #like_list = [mvn_loglikelihood_tf(X_batch[:,groups==i],Sigma[groups==i,groups==i]) for i in range(self.n_groups)]
                like_list = []

                loss = -1*posterior
            
            gradients = tape.gradient(loss,trainable_variables)
            optimizer.apply_gradients(zip(gradients,trainable_variables))

            self._saveLosses(i,like_prior,like_tot,posterior,like_list,
                                            sigma2)

        self._saveVariables(trainable_variables)

    def _prior(self,trainable_variables):
        return 0.0

    def _initializeSavedVariables(self):
        td = self.training_dict
        self.likelihood_prior = np.zeros(td['n_iterations'])
        self.likelihood_tot = np.zeros(td['n_iterations'])
        self.likelihood_subspace = np.zeros((td['n_iterations'],
                                                        self.n_groups))
        self.list_posterior = np.zeros(td['n_iterations'])
        self.sigmas = np.zeros(td['n_iterations'])
        
    def _saveLosses(self,i,like_prior,like_tot,posterior,like_list,sigma2):
        #self.likelihood_prior[i] = like_prior.numpy()
        self.likelihood_tot[i] = like_tot.numpy()
        self.list_posterior[i] = posterior.numpy()
        self.sigmas[i] = sigma2.numpy()
        #for j in range(self.n_groups):
        #    self.likelihood_subspace[i,j] = like_list[j].numpy()
    
    def _saveVariables(self,tv):
        '''

        '''
        self.W_ = tv[0].numpy()
        b_vals = tv[1:-1]
        self.B_list_ = [b_vals[i].numpy() for i in range(len(b_vals))]
        sigma2 = tf.nn.softplus(tv[-1])
        self.sigma2_ = sigma2.numpy()

    def _constructWTot(self):
        '''
        This constructs the multigroup extension of the matrix in Murphy
        (12.93).

        Returns
        -------
        W_tot : np.array_like(self.p,sum(subspace)+shared)
            Total W with conveniently placed zeros
        '''
        p = self.p
        dimX = int(np.sum(self.n_comps) + self.n_components_s)
        W_tot = np.zeros((p,dimX))
        if self.n_components_s == 1:
            W_tot[:,0] = np.squeeze(self.W_)
        else:
            W_tot[:,:self.n_components_s] = self.W_
        count = self.n_components_s
        count2 = 0
        for i in range(self.n_groups):
            stx = int(count2)
            edx = int(count2 + self.qs[i])
            sty = int(count)
            edy = int(count + self.n_comps[i])
            W_tot[stx:edx,sty:edy] = self.B_list_[i]
            count += self.n_comps[i]
            count2 += self.qs[i]

        return W_tot

    def get_total_covariance(self):
        '''
        Returns the covariance matrix that depends on all latent spaces

        Returns 
        -------
        Sigma : np.matrix(p,p)
            The covariance matrix using total space
        '''
        W = self._constructWTot()
        cov = self.sigma2_*np.eye(self.p) + np.dot(W,W.T)
        return cov

    def get_shared_covariance(self):
        '''
        Returns the covariance matrix that depends on the shared latent 
        space

        Returns 
        -------
        Sigma : np.matrix(p,p)
            The covariance matrix using total space
        '''
        cov = self.sigma2_*np.eye(self.p) + np.dot(self.W_,self.W_.T)
        return cov

    def get_total_precision(self):
        W = self._constructWTot()
        Dinv = 1/self.sigma2_*np.eye(self.p)
        precision = woodbury_inverse_sym_np(Dinv,W)
        return precision

    def get_shared_precision(self):
        Dinv = 1/self.sigma2_*np.eye(self.p)
        precision = woodbury_inverse_sym_np(Dinv,self.W_)
        return precision
        
    def transform_shared(self,X):
        '''
        This returns the latent variable estimates given X

        Parameters
        ----------
        X : np array-like,(N_samples,\sum idxs)
            The data to transform

        Returns
        -------
        S : np.array-like,(N_samples,n_components)
            The factor estimates
        '''
        prec = self.get_shared_precision()
        coefs = np.dot(self.W_,prec)
        mu_bar = np.dot(X,coefs.T)
        return mu_bar

    def transform_shared_subset(self,X,idxs):
        '''

        '''
        raise NotImplementedError('Sorry')


    def transform_individual(self,X,group_num):
        '''

        group_num : which group to use

        Returns
        -------
        Scores_shared : 


        '''
        raise NotImplementedError('Sorry')

    def score_total(self,X):
        '''
        Computes the marginal likelihood given the shared and individual
        subspaces. Marginal likelihood of 

        X_0      |0| |BB^T+WW^T+s2I W^T           W^T          |
        X_1  ~ N(|0|,|W             BB^T+WW^T+s2I W^T          |)
        X_n      |0| |W             W             BB^T+WW^T+s2I|

        Parameters
        ----------
        X : np.array-like,(N,p)
            The data to evaluate the likelihood

        Returns
        -------
        score : float
            Average log likelihood
        '''
        cov = self.get_total_covariance()
        score = mvn_loglikelihood_np(X,cov)
        return score

    def score_subset_total(self,X,idxs):
        '''
        Computes the marginal likelihood given the shared and individual
        subspaces. Marginal likelihood of 

        X_0      |0| |BB^T+WW^T+s2I W^T           W^T          |
        X_1  ~ N(|0|,|W             BB^T+WW^T+s2I W^T          |)
        X_n      |0| |W             W             BB^T+WW^T+s2I|

        Parameters
        ----------
        X : np.array-like,(N,p)
            The data to evaluate the likelihood

        Returns
        -------
        score : float
            Average log likelihood
        '''
        cov = self.get_total_covariance()
        cov_sub = cov[np.ix_(idxs==1,idxs==1)]
        score = mvn_loglikelihood_np(X,cov_sub)
        return score

    def score_shared(self,X):
        '''
        Computes the marginal likelihood given only the shared subspace
        with distribution

        X_0      |0| |WW^T+s2I W^T           W^T     |
        X_1  ~ N(|0|,|W             WW^T+s2I W^T     |)
        X_n      |0| |W             W        WW^T+s2I|

        Parameters
        ----------
        X : np.array-like,(N,p)
            The data to evaluate the likelihood

        Returns
        -------
        score : float
            Average log likelihood
        '''
        cov = self.get_shared_covariance()
        score = mvn_loglikelihood_np(X,cov)
        return score

    def score_subset_shared(self,X,idxs):
        '''
        Computes the marginal likelihood given only the shared subspace
        with distribution

        X_0      |0| |WW^T+s2I W^T           W^T     |
        X_1  ~ N(|0|,|W             WW^T+s2I W^T     |)
        X_n      |0| |W             W        WW^T+s2I|

        Parameters
        ----------
        X : np.array-like,(N,p)
            The data to evaluate the likelihood
        
        idxs : np.array-like,(p,)
            Vector of ones and zeros indicating which data X corresponds
            to

        Returns
        -------
        score : float
            Average log likelihood
        '''
        cov = self.get_shared_covariance()
        cov_sub = cov[np.ix_(idxs==1,idxs==1)]
        score = mvn_loglikelihood_np(X,cov_sub)
        return score
        
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
        if idx_obs is None:
            idx_obs = 1 - idx_miss
        idx_tot = idx_obs + idx_miss

        cov = self.get_total_covariance()
        cov_sub = subset_square_matrix_np(cov,idx_tot==1)

        idx_o = idx_obs[idx_tot==1]
        coef,_ = mvn_conditional_distribution_np(cov_sub,idx_o)
        return coef

    def predict(self,X,idx_miss):
        '''

        '''
        idx_obs = 1 - idx_miss

        coef = self._get_predictiveCoefficients(idx_miss,idx_obs)
        Y_hat = np.dot(X,coef.T)
        return Y_hat




