
# coding: utf-8

# In[ ]:

import numpy as np
from datetime import *
import json
import array
import math
import sys
import scipy.io as sio
import random
import pickle
from scipy.stats import multivariate_normal
import time
from functools import wraps
import errno 
import os
import signal
import roadrunner


# In[ ]:

def CalculateWeightMultiVariateNormalDensity(paramNum,currentPop,priorPop,priorWeights):
    # initialie new weights
    newWeights = np.zeros(priorWeights.shape)
    for i in range(len(newWeights)):
        # calcualte the mean and covaraince 
        meanMat = np.array([0]*paramNum)
        covMat = np.identity(paramNum)
        diffMat =multivariate_normal.pdf(priorPop - np.tile(currentPop[i],(len(priorPop),1) ),mean = meanMat,cov = covMat)
        newWeights[i] = np.matmul(priorWeights,diffMat**(1/float(paramNum)))
    newWeights = newWeights/sum(newWeights)
    return newWeights


# In[ ]:

def UniformWeight(population):
    numberOfParticles = len(population)
    returnWeights = np.array([1/float(numberOfParticles)]*len(population))
    return returnWeights

