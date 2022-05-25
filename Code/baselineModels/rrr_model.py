'''
Implements reduced rank regression

'''


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

