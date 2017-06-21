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

def CalculateWeightMultiVariateNormalDensity(paramNum,currentPop,priorPop,priorWeights):
    """
    The function assigns the re-sampling weights of the population. The function assumes the parameter posterior follows a multivariate normal density function and assignes the weights based on the density  
    Inputs-
        paramNum: number of parameters
        currentPop: The current iteration of parameter vector population. 
        priorPop: The parameter vector population of the previous iteration of rejection sampling
        priorWeights: The re-sampling weights of the previous iteration of rejection sampling 
    Output-
        newWeights: The calculated sampling weights of the current parameter vector population
    """
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

def UniformWeight(population):
    """
    The function assigns equal re-sampling weights to all the parameter vectors in the pouplation 
    Input-
    population: Populaiton of parameter vectors
    Output-
    returnWeights: The calculated sampling weights of the current parameter vector population 
    """
    numberOfParticles = len(population)
    returnWeights = np.array([1/float(numberOfParticles)]*len(population))
    return returnWeights

