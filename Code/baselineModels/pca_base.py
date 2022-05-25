'''
This implements the base class used for any gaussian linear models. It 
contains most basic methods, except for fit, prior definitions, and which
training variables to save. ALL methods have been unit tested to ensure 
functionality.

Authors : Austin Talbot <austin.talbot1993@gmail.com>
# Scott Linderman
# Corey Keller

Objects:
PCA_base(base)
    def _demean(self,X,idxs=None):
        This demeans the data. The first time this is run, saves mean as 
        attribute and demeans. Otherwise (transform) uses attribute to 
        demean

    def transform(self,X):
        This returns the latent variable estimates given X

    def get_covariance(self):
        Gets the covariance matrix WW^T + sigma^2I defined by the model

    def get_precision(self):
        This returns the precision matrix (inverse of covariance matrix). 
        More efficient than la.inv(self.get_covariance()) because uses 
        Sherman-Woodbury Matrix inversion identity.

    def transform_subset(self,X,idxs):
        Estimates the latent variables given only a subset of the data.

    def predict(self,X,idxs):
        Predicts missing data using observed data.

    def inverseTransform(self,S):
        Given latent variable estimates, provides reconstruction of the data

    def reconstruct(self,X):
        Creates a reconstruction of the data.

    def score(self,X):
        Returns the log-liklihood of data. 

    def conditional_score(self,X,idxs):
        Returns the predictive log-likelihood of a subset of data.

    def marginal_subset_score(self,X,idxs):
        Returns the marginal log-likelihood of a subset of data

    def _get_predictiveCoefficients(self,idx_miss,idx_obs=None):
        Gets the predictive coefficients predicting idx_miss using 
        idx_obs

    def marginal_entropy(self):
        Returns the entropy of the defined model

    def marginal_entropy_subset(self,idxs1):
        Returns the entropy of the defined model corresponding to a subset
        of covariates

    def mutual_information(self,idxs1,idxs2):
        This computes the mutual information bewteen the two sets of 
        covariates based on the model.

Creation Date: 12/01/2021

Version History:

Further reading:
    Yu et al 2006
    West 2003
    Bishop 2006
    Murphy 2010

'''
import numpy as np
from scipy import linalg as la
from numpy import random as rand
from tqdm import trange

from sklearn.utils import check_array, check_random_state
from sklearn.utils.extmath import randomized_svd,fast_logdet,svd_flip
from sklearn.utils.extmath import stable_cumsum,svd_flip
from sklearn.utils.validation import check_is_fitted
from sklearn import decomposition as dp
from sklearn import linear_model as lm

from datetime import datetime as dt
import os,sys,time
from scipy.io import savemat
import matplotlib.pyplot as plt 

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

import pickle
from pathlib import Path
from cca_base import base

homeDir = str(Path.home())
sys.path.append(homeDir + '/Utilities/Code/Miscellaneous')
sys.path.append(homeDir + '/Utilities/Code/Numpy')
sys.path.append(homeDir + '/Utilities/Code/Tensorflow')
from utils_misc_tf import limitGPU,return_optimizer_tf
from utils_misc import fill_dict,pretty_string_dict
from utils_batcher_np import simple_batcher_X,simple_batcher_XY
from utils_losses_tf import L2_loss_tf,gamma_loss_tf
from utils_gaussian_tf import mvn_loglikelihood_tf

from utils_gaussian_np import mvn_conditional_distribution_np
from utils_gaussian_np import mvn_loglikelihood_np
from utils_gaussian_np import mvn_entropy_np
from utils_gaussian_np import mvn_mutual_information_np
from utils_activations_np import softplus_inverse_np
from utils_matrix_np import woodbury_inverse_sym_np,subset_square_matrix_np

version = "1.0"

class PCA_base(base):
    '''
    Base class 
    '''

    def __init__(self,n_components,training_dict={},prior_dict={},
                                    gpu_memory=1024):
        self.n_components = int(n_components)

        self.training_dict = self._fillTrainingOpsDict(training_dict)
        self.prior_dict = self._fillPriorOptsDict(prior_dict)

        self.version = version
        self.creationDate = dt.now()
        self.gpu_memory = int(gpu_memory)

    def _initializeSavedVariables(self):
        '''
        Creates the attributes that store the loss values over training
        '''
        td = self.training_dict
        self.evals_prior = np.zeros(td['n_iterations'])
        self.evals_likelihood = np.zeros(td['n_iterations'])
        self.evals_posterior = np.zeros(td['n_iterations'])

    def _saveLosses(self,i,like_prior,like_tot,posterior):
        '''
        Saves the training values at iteration i

        Parameters
        ----------
        i : int
            Iteration

        like_prior : tf.Float
            Prior loglikelihood

        like_tot : tf.Float
            Data loglikelihood

        posterior : tf.Float
            Log posterior of data
        '''
        self.evals_prior[i] = like_prior.numpy()
        self.evals_likelihood[i] = like_tot.numpy()
        self.evals_posterior[i] = posterior.numpy()

    def _demean(self,X,idxs=None):
        '''
        This demeans the data. The first time this is run, saves mean as 
        attribute and demeans. Otherwise (transform) uses attribute to 
        demean

        Parameters
        ----------
        X : np array-like,(N_samples,\sum p_i)
            The data to demean

        idxs : np.array-like,(sum(p_i),)
            The associated group label

        Returns 
        -------
        X_mean : np array-like,(N_samples,\sum p_i)
            Demeaned data
        '''
        if hasattr(self,'mean_'):
            X_mean = X - self.mean_
            return X_mean
        else:
            if idxs is not None:
                self.n_groups = len(np.unique(idxs))
                ng = self.n_groups
                self.ps = np.array([np.sum(idxs==i) for i in range(ng)])
            self.mean_ = np.mean(X,axis=0)
            X_mean = X - self.mean_
            return X_mean

    def _demean_subset(self,X,idxs):
        '''
        Demeans data when only a subset is provided. Note that idxs has the
        same dimension as original data while X is reduced dataset.

        Parameters
        ----------
        X : np array-like,(N_samples,\sum idxs)
            The data to demean

        idxs : {0,1} array-like, (p,)
            1 Indicates observed data
        Returns 
        -------
        X_mean : np array-like,(N_samples,\sum idxs)
            Demeaned data
        '''
        check_is_fitted(self,'mean_')
        mean_ = self.mean_[idxs==1]
        X_mean = X - mean_
        return X_mean

    def transform(self,X):
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
        check_is_fitted(self,'mean_')
        X = self._demean(X)

        #Normally this would be stupid, taking an inverse and then 
        #multiplying it. However, when computing the inverse we use the 
        # Woodbury identity which actually makes this better. 
        prec = self.get_precision()
        coefs = np.dot(self.W_,prec)

        mu_bar = np.dot(X,coefs.T)
        return mu_bar
        
    def get_covariance(self):
        '''
        Gets the covariance matrix WW^T + sigma^2I defined by the model

        Returns
        -------
        cov : np.array-like (sum(),sum())
            The covariance matrix
        '''
        check_is_fitted(self,'mean_')
        cov = np.dot(self.W_.T,self.W_) + np.diag(self.sigma_)
        return cov

    def get_precision(self):
        '''
        This returns the precision matrix (inverse of covariance matrix). 
        More efficient than la.inv(self.get_covariance()) because uses 
        Sherman-Woodbury Matrix inversion identity.

        Returns
        -------
        precision : np.array-like(sum(p),sum(p))
            The inverse of the covariance matrix
        '''
        check_is_fitted(self,'mean_')
        Dinv = np.diag(1/self.sigma_)
        precision = woodbury_inverse_sym_np(Dinv,self.W_.T)
        return precision

    def transform_subset(self,X,idxs):
        '''
        Estimates the latent variables given only a subset of the data.

        Parameters
        ----------
        X : np array-like,(N_samples,\sum idxs)
            The data to transform

        idxs: np.array-like,(sum(p),)
            The observation locations 

        Returns 
        -------
        S : np.array-like,(N_samples,n_components)
            The factor estiamtes
        '''
        check_is_fitted(self,'mean_')
        X = X - self.mean_[idxs==1]
        D_sub = np.diag(1/self.sigma_[idxs==1])
        W_sub = self.W_[:,idxs==1]
        precision = woodbury_inverse_sym_np(D_sub,W_sub.T)
        coefs = np.dot(W_sub,precision)

        mu_bar = np.dot(X,coefs.T)
        return mu_bar

    def predict(self,X,idx_obs):
        '''
        Predicts missing data using observed data.

        Parameters
        ----------
        X : np array-like,(N_samples,\sum idxs)
            The data to transform

        idxs: np.array-like,(sum(p),)
            The observation locations 

        Returns
        -------
        preds : np.array-like,(N_samples,p-\sum idxs)
            The predicted values
        '''
        S = self.transform_subset(X,idx_obs)
        W_miss = self.W_[:,idx_obs==0]
        preds = np.dot(S,W_miss) + self.mean_[idx_obs==0]
        return preds

    def predictive_distribution(self,X,idx_obs):
        cov = self.get_covariance()
        preds,sig = mvn_conditional_distribution_np(cov,idx_obs)
        mu = np.dot(X,preds.T) + self.mean_[idx_obs==0]
        return mu,sig

    def inverseTransform(self,S):
        '''
        Given latent variable estimates, provides reconstruction of the data

        Parameters
        ----------
        S : np array-like,(N_samples,n_components)
            The latent variables.

        Returns
        -------
        X_recon : np.array-like,(N_samples,p)
            The predicted values
        '''
        X_recon = np.dot(S,self.W_) + self.mean_
        return X_recon

    def reconstruct(self,X):
        '''
        Creates a reconstruction of the data.

        Parameters
        ----------
        X : np array-like,(N_samples,p)
            The data to transform

        Returns
        -------
        X_recon : np.array-like,(N_samples,p)
            The predicted values
        '''
        S = self.transform(X)
        X_recon = np.dot(S,self.W_) + self.mean_
        return X_recon

    def score(self,X):
        '''
        Returns the log-liklihood of data. 

        Parameters
        ----------
        X : np.array-like,(N,sum(p))
            The data 

        Returns
        -------
        score : float
            Average log likelihood
        '''
        covariance = self.get_covariance()
        score = mvn_loglikelihood_np(X,covariance)
        return score
        
    def conditional_score(self,X,idxs):
        '''
        Returns the predictive log-likelihood of a subset of data.

        Parameters
        ----------
        X : np.array-like,(N,sum(idxs))
            The data 

        idxs: np.array-like,(sum(p),)
            The observation locations 

        Returns
        -------
        score : float
            Average log likelihood
        '''
        X_obs = X[:,idxs==1]
        X_miss = X[:,idxs==0]
        cov = self.get_covariance()
        coef_bar,Sig_bar = mvn_conditional_distribution_np(cov,idxs)
        diff = X_miss - np.dot(X_obs,coef_bar.T)
        score = mvn_loglikelihood_np(diff,Sig_bar)
        return score

    def marginal_subset_score(self,X,idxs):
        '''
        Returns the marginal log-likelihood of a subset of data

        Parameters
        ----------
        X : np.array-like,(N,sum(idxs))
            The data 

        idxs: np.array-like,(sum(p),)
            The observation locations 

        Returns
        -------
        score : float
            Average log likelihood
        '''
        cov = self.get_covariance()
        cov1 = cov[idxs==1]
        cov_sub = cov1[:,idxs==1]
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
        if idx_obs == None:
            idx_obs = 1 - idx_miss

        idx_tot = idx_obs + idx_miss

        cov = self.get_covariance()
        cov_sub = subset_square_matrix_np(cov,idx_tot==1)

        idx_o = idx_obs[idx_tot==1]
        coef,_ = mvn_conditional_distribution_np(cov_sub,idx_o)
        return coef

    def _saveVariables(self,trainable_variables,D):
        '''
        This saves two variables, the diagonal matrix Psi and the factors
        W

        Attributes
        ----------
        sigma_ : np.array-like (sum(p),)
            The diagonal matrix stored as a vector

        W_ : np.array-like (,)
            The loadings stored as a matrix
        '''
        self.W_ = trainable_variables[0].numpy()
        DD = D.numpy()
        self.sigma_ = np.zeros(DD.shape[0])
        for i in range(DD.shape[0]):
            self.sigma_[i] = DD[i,i]

    ##############################
    # Information theory section #
    ##############################
    def marginal_entropy(self):
        '''
        Returns the entropy of the defined model

        Returns
        -------
        entropy : float
            The entropy of the distribution
        '''
        cov = self.get_covariance()
        entropy = mvn_entropy_np(cov)
        return entropy

    def marginal_entropy_subset(self,idxs1):
        '''
        Returns the entropy of the defined model corresponding to a subset
        of covariates

        Parameters
        ----------
        idxs1 : np.array-like,(p,)
            First group of variables

        Returns
        -------
        entropy : float
            The entropy of the distribution
        '''
        cov = self.get_covariance()
        cov_subset = subset_square_matrix_np(cov,idxs)
        entropy = mvn_entropy_np(cov_subset)
        return entropy

    def mutual_information(self,idxs1,idxs2):
        '''
        This computes the mutual information bewteen the two sets of 
        covariates based on the model.

        Parameters
        ----------
        idxs1 : np.array-like,(p,)
            First group of variables

        idxs2 : np.array-like,(p,)
            Second group of variables

        Returns
        -------
        mutual_information : float
            The mutual information between the two variables 
        '''
        cov = self.get_covariance()
        idx_tot = idxs1 + idxs2
        cov_subset = subset_square_matrix_np(cov,idx_tot)
        idx_obs = idxs1[idx_tot==1]
        mutual_information = mvn_mutual_information_np(cov_subset,idx_obs)
        return mutual_information
