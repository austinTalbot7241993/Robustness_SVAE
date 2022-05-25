'''
Authors : Austin Talbot <austin.talbot1993@gmail.com>
# Scott Linderman
# Corey Keller

This implements supervised probabilistic principal component analysis, also
known as Bayesian factor regression. This generalizes both of these papers,
since we allow for more than one group. This uses Tensorflow to solve. I 
noticed that neither paper had analytic ML solutions, and some effort on 
my part makes me think that they don't exist. So we use gradient descent.
I know this isn't convex, ...

I'm pretty sure I could get around that by estimating the components one 
at a time and then figuring out the noise at the end, but too complicated.
Anyways the same criticism could be leveled at the EM formulation that they
advocate for so bite me.

And we can add priors too so obviously better.

Model:

                p(s)   ~ N(0,I_L)
                p(X|s) ~ N(0, WW^T + Psi)

There are two forms that we implement here. The first is Bayesian factor
regression, where Psi has a variance associated with each ``group'' of
variables. Within each group is isotropic noise. The other is factor 
analysis where each element on the diagonal gets its own variance.

We place different priors for W and Psi in the models, with various 
options provided.

Objects:

PPCA
SPCA
FactorAnalysis

Creation Date: 11/01/2021

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
from pca_base import PCA_base

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

class PPCA_analytic(PCA_base):
    '''
    This implments probabilistic PCA relying on analytic solutions and no
    priors on anything besides latent variables.

    '''

    def __init__(self,*args,**kwargs):
        super(PPCA_analytic,self).__init__(*args,**kwargs)

    def __repr__(self):
        out_str = 'PPCA_analytic object\n'
        out_str += 'n_components=%d\n'%self.n_components
        return out_str

    def fit(self,X):
        self.mean_ = np.mean(X,axis=0)
        X = X - self.mean_
        N,self.p = X.shape
        L,p = self.n_components,self.p

        U,s,V = la.svd(X,full_matrices=False)
        evals = s**2/(N-1)

        var = 1.0/(p-L)*(np.sum(evals)-np.sum(evals[:L]))

        L_m = np.diag((evals[:L]-np.ones(L)*var)**.5)
        W = np.dot(V[:L].T,L_m)
        self.W_ = W.T
        self.sigma_ = var*np.ones(p)
    
    def _fillPriorOptsDict(self,prior_dict):
        return None
    def _initializeVars(self,training_dict):
        return None
    def _prior(self,training_variables):
        return None


class PPCA(PCA_base):
    '''
    This implments probabilistic PCA using tensorflow gradient descent. 

    p(s) ~ N(0,I)
    p(x|s) ~ N(Ws,sigma^2I)


    Parameters
    ----------

    Attributes
    ----------

    Examples
    --------
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

    '''
    
    def __init__(self,*args,**kwargs):
        super(PPCA,self).__init__(*args,**kwargs)

    def __repr__(self):
        out_str = 'PPCA object\n'
        out_str += 'n_components=%d\n'%self.n_components
        out_str = out_str + '>>>>>>>>>>>>\n'
        out_str = out_str + 'Training Parameters:\n'
        out_str = out_str + pretty_string_dict(self.training_dict)
        out_str = out_str + '>>>>>>>>>>>>\n'
        out_str = out_str + 'Prior Parameters:\n'
        out_str = out_str + pretty_string_dict(self.prior_dict)
        return out_str

    def fit(self,X,var_dict=None):
        '''
        Fit the model given X and the groupings.

        Parameters
        ----------
        X : np.array-like,(N,sum(p))
            The data 

        var_dict : dictionary
            Used if a non-random initialization is used
        '''
        limitGPU(self.gpu_memory)
        td = self.training_dict
        X = self._demean(X)
        X = X.astype(np.float32)
        N,p = X.shape
        self.p = p

        W_,sigmal_ = self._initializeVars(X,var_dict=var_dict)
        trainable_variables = [W_,sigmal_] 

        self._initializeSavedVariables()

        optimizer = return_optimizer_tf(td['method'],td['learning_rate'])
        eye = tf.constant(np.eye(p).astype(np.float32))

        for i in trange(td['n_iterations']):
            X_batch = simple_batcher_X(td['batch_size'],X)

            with tf.GradientTape() as tape:
                sigma = tf.nn.softplus(sigmal_)
                WWT = tf.matmul(tf.transpose(W_),W_)

                Sigma = WWT + sigma*eye

                like_prior = self._prior(trainable_variables)
                like_tot = mvn_loglikelihood_tf(X_batch,Sigma)
                posterior = like_tot + 1/N*like_prior

                loss = -1*posterior

            gradients = tape.gradient(loss,trainable_variables)
            optimizer.apply_gradients(zip(gradients,trainable_variables))

            self._saveLosses(i,like_prior,like_tot,posterior)
        
        self._saveVariables(trainable_variables)

    def _demean(self,X):
        '''
        This demeans the data. The first time this is run, saves mean as 
        attribute and demeans. Otherwise (transform) uses attribute to 
        demean

        Parameters
        ----------
        X : np array-like,(N_samples,\sum p_i)
            The data to demean

        Returns 
        -------
        X_mean : np array-like,(N_samples,\sum p_i)
            Demeaned data
        '''
        if hasattr(self,'mean_'):
            X_mean = X - self.mean_
            return X_mean
        else:
            self.mean_ = np.mean(X,axis=0)
            X_mean = X - self.mean_
            return X_mean

    def _initializeVars(self,X,var_dict=None):
        '''
        Initializes the variables of the model

        Parameters
        ----------
        X : np.array-like,(n_samples,p)
            The data

        var_dict : {None,dict,'sklearn'}
            The dictionary containing (potential) choices for intial
            guesses

        Returns
        -------
        W_ : tf.Variable-like,(n_components,p)
            The loadings of our latent factor model

        sigmal_ : list
            List of variables of variances associated with each group
        '''
        if var_dict is None:
            sigmal_ = tf.Variable(0.0)
            W_ = tf.Variable(rand.randn(self.n_components,
                                    len(self.mean_)).astype(np.float32))
        elif var_dict == 'sklearn':
            model = dp.PCA(self.n_components)
            S_hat = model.fit_transform(X)
            W_init = model.components_.astype(np.float32)
            W_ = tf.Variable(W_init)
            X_recon = np.dot(S_hat,W_init)
            diff = np.mean((X-X_recon)**2)
            sigmal_ = tf.Variable(diff)
        else:
            W_ = tf.Variable(var_dict['W_'].astype(np.float32))
            sinv = softplus_inverse_np(var_dict['sigma_'])
            sigmal_ = tf.Variable(sinv[0])
        return W_,sigmal_

    def _prior(self,tv):
        '''
        This computes the prior on the parameters

        Parameters
        ----------

        tv : list
            List of tensorflow variables corresponding to parameters

        Returns
        -------
        log_prior : tf.Float
            The log density of the prior
        '''
        if self.prior_dict['type'] == 'gamma':
            #Gamma distribution
            alpha = self.prior_dict['alpha']
            beta = self.prior_dict['beta']
            sigma = tf.nn.softplus(tv[1])
            log_prior = gamma_loss_tf(sigma,alpha,beta)
        elif self.prior_dict['type'] == 'laplace':
            lambd = self.prior_dict['lambda']
            term1 = -1/2*self.p*self.n_components*np.log(lambd)
            const = -np.sqrt(2/lambd)
            term2 = const*tf.reduce_sum(tf.abs(tf[0]))
            log_pror = term1 + term2
        else:
            raise NotImplementedError('Sorry')
        return log_prior

    def _fillPriorOptsDict(self,prior_dict):
        '''
        Fills in parameters used for prior of parameters

        Paramters
        ---------
        prior_dict: dictionary
            The prior parameters used to specify the prior

        '''
        if 'type' not in prior_dict:
            prior_dict['type'] = 'gamma'

        if prior_dict['type'] == 'gamma':
            defaults = {'alpha':1.0,'beta':1.0}
        elif prior_dict['type'] == 'laplace':
            defaults = {'lambda':1.0}
        else:
            raise NotImplementedError('Sorry')

        new_dict = fill_dict(prior_dict,defaults)
        return new_dict

    def _saveVariables(self,trainable_variables):
        '''
        This saves two variables, the diagonal matrix Psi and the factors

        Attributes
        ----------
        sigma_ : np.array-like (sum(p),)
            The diagonal matrix stored as a vector

        W_ : np.array-like (,)
            The loadings stored as a matrix
        '''
        self.W_ = trainable_variables[0].numpy()
        sigma = tf.nn.softplus(trainable_variables[1])
        self.sigma_ = sigma.numpy()*np.ones(self.p)

class SPCA(PCA_base):
    '''
    Supervised probabilistic principal component analysis generalized to
    multiple groups, rather than the two groups advocated for by Yu et al
    and West.

    Parameters
    ----------

    Attributes
    ----------

    Examples
    --------

    ## Generate data
    N = 1000
    L = 4
    p = 30
    q = 5
    sigmax = 1.0
    sigmay = .5

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

    ## Fit the model
    model = SPCA(L,training_dict=training_dict)
    XX = np.hstack((X_train,Y_train))
    idxs = np.zeros(35)
    model.fit(XX,idxs)

    ## Analyze the results
    for i in range(L):
        print(cosine_similarity_np(W_est[i,:30],Wx[i]))
    plt.plot(model.evals_likelihood)
    plt.show()

    '''
    def __init__(self,*args,**kwargs):
        super(SPCA,self).__init__(*args,**kwargs)

    def __repr__(self):
        out_str = 'SPCA object\n'
        out_str += 'n_components=%d\n'%self.n_components
        out_str = out_str + '>>>>>>>>>>>>\n'
        out_str = out_str + 'Training Parameters:\n'
        out_str = out_str + pretty_string_dict(self.training_dict)
        out_str = out_str + '>>>>>>>>>>>>\n'
        out_str = out_str + 'Prior Parameters:\n'
        out_str = out_str + pretty_string_dict(self.prior_dict)
        return out_str

    def fit(self,X,idxs,var_dict=None):
        '''
        Fit the model given X and the groupings.

        Parameters
        ----------
        X : np.array-like,(N,sum(p))
            The data 

        idxs : np.array-like(sum(p))
            The groupings of the data

        var_dict : dictionary
            Used if a non-random initialization is used
        '''
        limitGPU(self.gpu_memory)
        td = self.training_dict
        X = self._demean(X,idxs)
        X = X.astype(np.float32)
        ng = len(np.unique(idxs))
        N = X.shape[0]

        W_,slist = self._initializeVars(X,idxs,var_dict=var_dict)
        trainable_variables = [W_] + slist

        clist = self._getConstList(idxs)

        self._initializeSavedVariables()

        optimizer = return_optimizer_tf(td['method'],td['learning_rate'])

        for i in trange(td['n_iterations']):
            X_batch = simple_batcher_X(td['batch_size'],X)

            with tf.GradientTape() as tape:
                WWT = tf.matmul(tf.transpose(W_),W_)

                D = tf.linalg.diag(tf.add_n([tf.nn.softplus(slist[i])*clist[i] for i in range(ng)]))

                Sigma = WWT + D

                like_prior = self._prior(trainable_variables)
                like_tot = mvn_loglikelihood_tf(X_batch,Sigma)
                posterior = like_tot + 1/N*like_prior

                loss = -1*posterior

            gradients = tape.gradient(loss,trainable_variables)
            optimizer.apply_gradients(zip(gradients,trainable_variables))

            self._saveLosses(i,like_prior,like_tot,posterior)
        
        self._saveVariables(trainable_variables,D)

    def _prior(self,tv):
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
        alpha = self.prior_dict['alpha']
        beta = self.prior_dict['beta']
        ng = self.n_groups
        sigmas = [tf.nn.softplus(tv[i+1]) for i in range(ng)]
        prior_list =[gamma_loss_tf(sigmas[i],alpha,beta) for i in range(ng)]
        log_prior = tf.math.add_n(prior_list)
        return log_prior

    def _fillPriorOptsDict(self,prior_dict):
        '''
        Fills in parameters used for prior of parameters

        Paramters
        ---------
        prior_dict: dictionary
            The prior parameters used to specify the prior

        Options
        -------
        alpha : default=1.
            Gamma prior on sigma.  

        beta : default=1.
            Gamma prior on sigma.  
        '''
        defaults = {'alpha':1.0,'beta':1.0}
        new_dict = fill_dict(prior_dict,defaults)
        return new_dict

    def _initializeVars(self,X,idxs,var_dict=None):
        '''
        Initializes the variables of the model

        Parameters
        ----------
        X : np.array-like,(n_samples,p)
            The data

        idxs : np.array-like(sum(p))
            The groupings of the data

        var_dict : {None,dict,'sklearn'}
            The dictionary containing (potential) choices for intial
            guesses

        Returns
        -------
        W_ : tf.Variable-like,(n_components,p)
            The loadings of our latent factor model

        sigmal_ : list
            List of variables of variances associated with each group
        '''
        ng = self.n_groups
        if var_dict is None:
            sigmal_ = [tf.Variable(0.0) for i in range(ng)]
            W_ = tf.Variable(rand.randn(self.n_components,
                                        len(self.mean_)).astype(np.float32))
        elif var_dict == 'sklearn':
            model = dp.PCA(self.n_components)
            S_hat = model.fit_transform(data)
            W_init = model.components_.astype(np.float32)
            W_ = tf.Variable(W_init)
            X_recon = np.dot(S_hat,W_init)
            diff = np.mean((X_recon-data)**2,axis=0)
            diff_list = [np.mean(diff[idxs==i]) for i in range(ng)]
            six = [softmax_inverse_np(diff_list[i]) for i in range(ng)]
            sigmal_ = [tf.Variable(six[i]) for i in range(ng)]
        else:
            W_ = tf.Variable(var_dict['W_'].astype(np.float32))
            sigmal_ = [tf.Variable(var_dict['sigmal_']) for i in range(ng)]

        return W_,sigmal_

    def _getConstList(self,idxs):
        '''
        Parameters
        ----------

        Returns
        -------

        '''
        myList = []
        for i in range(self.n_groups):
            numpy_array = np.zeros(len(self.mean_))
            numpy_array[idxs==i] = 1
            myList.append(tf.constant(numpy_array.astype(np.float32)))
        return myList

class FactorAnalysis(PCA_base):
    '''
    This implements Factor Analysis, which is basically the same thing as
    probabilistic principal component analysis except it allows for non-
    isotropic noise in the model. There are tons of different 
    implementations, we place a specific prior in the mode

    \sigma_i ~ 
    \W_i ~ 
    X|W,\sigma ~ N(0,WWT+Sigma)

    We then perform MAP estimation using stochastic gradient descent.

    Further reading
    ---------------

    Parameters
    ----------

    Attributes
    ----------

    Methods
    -------

    Examples
    --------
    '''

    def __init__(self,*args,**kwargs):
        super(FactorAnalysis,self).__init__(*args,**kwargs)

    def __repr__(self):
        out_str = 'FactorAnalysis object\n'
        out_str = out_str + 'n_components=%d\n'%self.n_components
        out_str = out_str + '>>>>>>>>>>>>\n'
        out_str = out_str + 'Training Parameters:\n'
        out_str = out_str + pretty_string_dict(self.training_dict)
        out_str = out_str + '>>>>>>>>>>>>\n'
        out_str = out_str + 'Prior Parameters:\n'
        out_str = out_str + pretty_string_dict(self.prior_dict)
        return out_str

    def fit(self,X,var_dict=None):
        '''
        Fit the model given X 

        Parameters
        ----------
        X : np.array-like,(N,sum(p))
            The data 

        var_dict : dictionary
            Used if a non-random initialization is used
        '''
        limitGPU(self.gpu_memory)
        td = self.training_dict
        X = self._demean(X)
        X = X.astype(np.float32)
        N = X.shape[0]

        W_,sigmal_ = self._initializeVars(X,var_dict=var_dict)

        trainable_variables = [W_,sigmal_]
        trainable_variables_init = [sigmal_]

        self._initializeSavedVariables()

        optimizer = return_optimizer_tf(td['method'],td['learning_rate'])
        optimizer_init = return_optimizer_tf(td['method'],
                                                    10*td['learning_rate'])
        for i in trange(1000):
            X_batch = simple_batcher_X(td['batch_size'],X)

            with tf.GradientTape() as tape:
                WWT = tf.matmul(tf.transpose(W_),W_)

                sigmas = tf.nn.softplus(sigmal_)
                D = tf.linalg.diag(sigmas)
                Sigma = WWT + D

                like_prior = self._prior(trainable_variables)
                like_tot = mvn_loglikelihood_tf(X_batch,Sigma)
                posterior = like_tot + 1/N*like_prior

                loss = -1*posterior

            gradients = tape.gradient(loss,trainable_variables_init)
            optimizer_init.apply_gradients(zip(gradients,
                                            trainable_variables_init))
            
        for i in trange(td['n_iterations']):
            X_batch = simple_batcher_X(td['batch_size'],X)

            with tf.GradientTape() as tape:
                WWT = tf.matmul(tf.transpose(W_),W_)

                sigmas = tf.nn.softplus(sigmal_)
                D = tf.linalg.diag(sigmas)
                Sigma = WWT + D

                like_prior = self._prior(trainable_variables)
                like_tot = mvn_loglikelihood_tf(X_batch,Sigma)
                posterior = like_tot #+ 1/N*like_prior

                loss = -1*posterior

            gradients = tape.gradient(loss,trainable_variables)
            optimizer.apply_gradients(zip(gradients,trainable_variables))

            self._saveLosses(i,like_prior,like_tot,posterior)
        
        self._saveVariables(trainable_variables,D)

    
    def _initializeVars(self,X,var_dict=None):
        '''
        Initializes the variables of the model

        Parameters
        ----------
        X : np.array-like,(n_samples,p)
            The data

        var_dict : {None,dict,'sklearn'}
            The dictionary containing (potential) choices for intial
            guesses

        Returns
        -------
        W_ : tf.Variable-like,(n_components,p)
            The loadings of our latent factor model

        sigmal_ : tf.Variable-like,(p,)
            Sigma2 is softplus(sigmal_)
        '''
        p = len(self.mean_)
        if var_dict is None:
            sigmal_ = tf.Variable(.1*rand.randn(p).astype(np.float32))
            W_ = tf.Variable(rand.randn(self.n_components,
                                        p).astype(np.float32))
        elif var_dict == 'sklearn':
            model = dp.PCA(self.n_components)
            S_hat = model.fit_transform(X)
            W_init = model.components_.astype(np.float32)
            for i in range(self.n_components):
                W_init[i] = np.std(S_hat[:,i])*W_init[i]
                S_hat[:,i] = 1/np.std(S_hat[:,i])
            W_ = tf.Variable(W_init)
            nv = model.noise_variance_
            X_recon = np.dot(S_hat,W_init)
            sigma_est = softplus_inverse_np(nv*np.ones(p))
            sigmal_ = tf.Variable(sigma_est.astype(np.float32))
        else:
            W_ = tf.Variable(var_dict['W_'].astype(np.float32))
            sigmal_ = tf.Variable(var_dict['sigmal_'].astype(np.float32))

        return W_,sigmal_

    def _prior(self,training_variables):
        '''
        Evaluates the prior of the parameters

        Paramters
        ---------
        training_variables:
            List of variables we are optimizing

        Returns
        -------
        loss : tf.float
            The prior of the parameters
        '''
        alpha = self.prior_dict['alpha']
        beta = self.prior_dict['beta']
        sigmas = tf.nn.softplus(training_variables[1])
        log_prior = gamma_loss_tf(sigmas,alpha,beta)
        loss = tf.reduce_mean(log_prior)
        return loss 

    def _fillPriorOptsDict(self,prior_dict):
        '''
        Fills in parameters used for prior of parameters

        Paramters
        ---------
        prior_dict: dictionary
            The prior parameters used to specify the prior

        Returns
        -------
        new_dict : dict
            dictionary of prior parameters
        '''
        defaults = {'alpha':1.0,'beta':1.0}
        new_dict = fill_dict(prior_dict,defaults)
        return new_dict

