
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

def SimpleSampleUniform(paramNum,order=2):
    # prior sampling based on log scale
    return np.array((np.random.uniform(size = paramNum) -0.5)*order)


# In[ ]:

def MultiVariateNormalCovariateSample(paramNum,priorPop,priorWeights,covscaling,order=2):
    arraySort = np.argsort(priorWeights)
    reverseArraySort = arraySort[::-1]
    priorWeightsSorted = [priorWeights[i] for i in reverseArraySort]
    priorPopSorted = [priorPop[i] for i in reverseArraySort]
    # randomly select a particle based on their associated weights
    totalWeight = sum(priorWeightsSorted)
    particleRand = totalWeight*random.random()
    #print('particleRand = '+str(particleRand))
    itIsHere = 0
    while itIsHere < len(priorWeightsSorted):
        #print('itIsHere = '+ str(itIsHere))
        if sum(priorWeightsSorted[0:itIsHere+1]) >= particleRand:
            break
        itIsHere +=1
    returnParticle = priorPopSorted[itIsHere]  
    particleOK = False
    while particleOK == False:
        rv = multivariate_normal(mean = None,cov = np.multiply( np.cov(priorPop,rowvar = False),covscaling),allow_singular=True)
        randomVar = rv.rvs(size=1)
        returnParticlePerturbed = returnParticle + randomVar
        if all(abs(i)<=order/2 for i in returnParticlePerturbed):
            particleOK = True
    return returnParticlePerturbed

