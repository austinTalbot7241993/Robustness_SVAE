from math import log,sqrt
import numbers
import logging

import numpy as np
from scipy import linalg as la
from scipy.special import gammaln
from scipy.sparse import issparse
from scipy.sparse.linalg import svds
from numpy import random as rand
from tqdm import trange
from abc import ABC,abstractmethod

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
from utils_misc import fill_dict
from utils_gaussian_np import mvn_conditional_distribution_np
from utils_batcher_np import simple_batcher_X,simple_batcher_XY
from utils_misc_tf import limitGPU,return_optimizer_tf
from utils_gaussian_tf import mvn_loglikelihood_tf

version = "1.0"

class base(ABC):
    
    def _convertData(self,X_list):
        '''
        Turns a list of matrices into float32 needed for tensorflow

        Paramters
        ---------
        X_list : list
            List of numpy arrays

        Returns
        -------
        X_converted : list
            List of numpy.float32 arrays

        '''
        X_converted = [X.astype(np.float32) for X in X_list]
        self.dims = [X.shape[1] for X in X_list]
        return X_converted

    def _fillTrainingOpsDict(self,training_dict):
        '''
        Fills in parameters used for learning algorithm

        Paramters
        ---------
        n_iterations: int, default=1000
            Number of iterations

        learning_rate : float, default=1e-4
            Learning rate of gradient descent

        method : string {'Nadam'}, default='Nadam'
            The learning algorithm

        batch_size : int, default=128
            The number of observations to use at each iteration
        '''
        default_dict = {'n_iterations':3000,'learning_rate':1e-3,
                        'method':'Nadam','batch_size':128}
        return fill_dict(training_dict,default_dict)


    @abstractmethod
    def _fillPriorOptsDict(self,prior_dict):
        '''
        Fills in parameters used for prior of parameters
        '''
        pass

    @abstractmethod
    def _prior(self,training_variables):
        '''
        Evaluates the prior of the parameters
        '''
        pass

    @abstractmethod
    def _initializeVars(self,data=None,var_dict=None):
        '''
        Initializes the variables of the model
        '''
        pass

    @abstractmethod
    def _initializeSavedVariables(self):
        '''
        Creates training loss attributes
        '''
        pass
        
    @abstractmethod
    def _saveLosses(self,i,like_prior,like_tot,posterior,loss_list=None):
        '''
        Saves losses
        '''
        pass
    
    @abstractmethod
    def _saveVariables(self,trainable_variables):
        '''
        Saves the variables 
        '''
        pass

