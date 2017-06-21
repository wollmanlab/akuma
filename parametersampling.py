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

def SimpleSampleUniform(paramNum,order=2):
    """
    The method samples parameter vectors from a log uniform interval 
    Input-
        paramNum: The number of parameters
        order: The span of log order of the parameers 
    Output-
        A parameter vector in the log range

    """
    # prior sampling based on log scale
    return np.array((np.random.uniform(size = paramNum) -0.5)*order)

def MultiVariateNormalCovariateSample(paramNum,priorPop,priorWeights,covscaling,order=2):
    """
    The method resamples from the posterior of the previous rejection sampling and perturbs the particles by calculating the covariance of the parameters
    Inputs-
        paramNum: Number of parameters
        priorPop: The parameter population of the prior rejection sampling iteration
        priorWeights: The re-sampling weights for prior population of posterior 
        covscaling: The covariance scaling factor 
        order: The total span of log order 
    Outputs-
        returnParticlePerturbed: The re-sampled and perturbed parameter vector particle  
    """
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

