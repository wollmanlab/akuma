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

class FncOutput:    
   """
    The class to record the results of model simulation with a given set of parameters 
    Attributes:
        results = The simulated results of the state variables 
        resultsEquilibrium = The simulated results of the state variables simulating toward equilibrium prior to perturbation
        obs = The name of the state variable used as the observable
    """
    def __init__(self,
                 results=None,
                 resultsEquilibrium=None,
                 obs=None,
                 ):
        self.results = results
        self.resultsEquilibrium = resultsEquilibrium
        self.obs = obs
    def GetResults(self):
        return self.results
    def SetResults(self,results):
        self.results = results
    def GetResultsEquilibrium(self):
        return self.resultsEquilibrium
    def SetResultsEquilibrium(self,resultsEquilibrium):
        self.resultsEquilibrium = resultsEquilibrium
    def GetObs(self):
        return self.obs
    def SetObs(self,obs):
        self.obs = obs

def SimpleEval(score,threshold=1):
    """
    The method evaluates the score based on a given threshold and returns True if the score is below the threshold, False otherwise
    """
    return score < threshold

def CheckConstraints(outputResult, constraintDict):
    """
    The method checks if the simulated results fullfill the constraints
    Inputs-
        outputResult: Results of model simulation
        constraintDict: The dictionary of constraints  
    Output-
        outputCheck: The output of constraint checking. True if all constraints are met and false otherwise
    """
    if 'resultArray' in outputResult:
        result = outputResult['resultArray']
    if 'resultEquilibriumArray' in outputResult:
        resultEquilibrium = outputResult['resultEquilibriumArray']    
    if 'varList' in outputResult:
        varList = outputResult['varList']
    if 'varsIniBoundList' in constraintDict.keys():
        varsIniBoundList = constraintDict['varsIniBoundList']
        gIni = [True]*len(varsIniBoundList)
        for i in range(len(gIni)):
            currBound = varsIniBoundList[i]
            if currBound:
                lower = True
                upper = True
                if currBound[0] != None:
                    lower = result[i][0] >= currBound[0]
                if currBound[1] != None:
                    upper = result[i][0] <= currBound[1]
                gIni[i] = lower and upper
    else:
        gIni = [True]
    if 'varsBoundList' in constraintDict.keys():
        varsBoundList = constraintDict['varsBoundList']
        gResult = [True]*len(varsBoundList)   
        for i in range(len(gResult)):
            currBound = varsBoundList[i]
            if currBound:
                lower = True
                upper = True
                if currBound[0] != None:
                    lower = result[i][0] >= currBound[0]
                if currBound[1] != None:
                    upper = result[i][0] <= currBound[1]
                gResult[i] = lower and upper
    else:
        gResult = [True]
    if 'varsEquilibriumBoundList' in constraintDict.keys():
        varsEquilibriumBoundList = constraintDict['varsEquilibriumBoundList']
        gResultEqui = [True]*len(varsEquilibriumBoundList)
        for i in range(len(gResultEqui)):
            currBound = varsEquilibriumBoundList[i]
            if currBound:
                lower = True
                upper = True
                if currBound[0] != None:
                    lower = min(resultEquilibrium[i]) >= currBound[0]
                if currBound[1] != None:
                    upper = max(resultEquilibrium[i]) <= currBound[1]
                gResultEqui[i] = lower and upper
    else:
        gResultEqui = [True]
    outputCheck = all(gIni+gResult+gResultEqui) 
    return outputCheck

def SimulateODERoadRunner(xmlFile,observableName,varList,data,simulationStart,simulationEnd,simulationSteps,equilibriumEnd=None,equilibriumSteps=None,paramDict=None,perturbParamDict=None,varInitDict=None):
    """
    The function simulates ODE using the libroadRunner package
    Input:
        xmlFile: The sbml file of the model
        observableName: The string representation of the observable 
        varList: The list of strings of the variables to consider for constraints
        data: The data to be fitted 
        x0: The initial conditions for all state variables
        simulationDuration: simulation time duration
        simulationSteps: The number of time steps for simulation
        equilibriumEnd: The ending time point for equilibriumEnd
        equilibriumSteps: The number of time steps in simulating toward equilibrium
        paramDict: The key-value pairs of parameters 
        perturbParamDict: The key-value pairs of parameters to be perturbed
        varInitDict: key-value pairs of state variables 
    output: 
        key-value pairs of simulated results
    """
    thisFncOutput = FncOutput()
    rr = roadrunner.RoadRunner(xmlFile)
    # set initial conditions
    if varInitDict:
        keyList = varInitDict.keys()
        valueList = varInitDict.values()
        for i in range(len(keyList)):
            setattr(rr,keyList[i],valueList[i])
    # set parameter values
    if paramDict:
        keyList = paramDict.keys()
        valueList = paramDict.values()
        for i in range(len(keyList)):
            setattr(rr,keyList[i],valueList[i])
    # perturb the parameter if must
    if perturbParamDict:
        #print(perturbParamDict)
        keyList = perturbParamDict.keys()
        valueList = perturbParamDict.values()
        for i in range(len(keyList)):
            #print(keyList[i])
            #print(valueList[i][0])
            setattr(rr,keyList[i],valueList[i][0])
    #print(keyList)
    #print(valueList)
    rr.timeCourseSelections = ['time']+ varList
    #print(rr.timeCourseSelections)
    # run to equilibrium if must
    needRunToEqui = False
    if equilibriumEnd and equilibriumSteps:
        #print('Run to equilibirum')
        needRunToEqui = True
        start_time = time.time()
        resultEquilibrium = rr.simulate(end = equilibriumEnd,start= 0,steps = equilibriumSteps)   
        timeInterval = time.time() - start_time
    # change parameter values for perturbation if must
    if perturbParamDict:
        #print(perturbParamDict)
        keyList = perturbParamDict.keys()
        valueList = perturbParamDict.values()
        for i in range(len(keyList)):
            #print(keyList[i])
            #print(valueList[i][1])
            setattr(rr,keyList[i],valueList[i][1])
    # simulate 
    start_time = time.time()
    result = rr.simulate(end=simulationEnd,start=simulationStart,steps=simulationSteps)
    timeInterval = time.time() - start_time
    # transpose the results
    resultArray = []
    for i in range(len(varList)):
        if i == 0:
            resultArray = result[varList[i]]
        else:
            resultArray = np.column_stack((resultArray,result[varList[i]]))
    if needRunToEqui:
        resultEquilibriumArray = []
        for i in range(len(varList)):
            if i == 0:
                resultEquilibriumArray = resultEquilibrium[varList[i]]
            else:
                resultEquilibriumArray = np.column_stack((resultEquilibriumArray,resultEquilibrium[varList[i]]))    
    # check if the user chooses to simulate state variables going to equilibria
    resultArray = np.transpose(resultArray)
    # observable
    #print(observableName)
    #print(result[observableName[0]])
    obs = np.transpose(result[observableName[0]])
    if needRunToEqui:
        resultEquilibriumArray = np.transpose(resultEquilibriumArray)
        return {'obs':obs,'varList':varList,'resultArray':resultArray,'resultEquilibriumArray':resultEquilibriumArray}
    else:
        return {'obs':obs,'varList':varList,'resultArray':resultArray}

def FncSimpleScore(data,observable,scoringFncParams,option=3):        
    """
    The scoring function takes in the single observable from the simulation that also passed the constraints tests
    
    Inputs-
        data: real data
        observable: The observable of the model to be scored against the data
        scoringFncParams: The key-value pairs of parameters associated with fitness function   
        option: choice of fitness function. 1 = sum of squares, 2 = sum of squares of differential, 3 = sum of 1 and 2 
    Output-
        fitness: The fitness score 
    """
    if 'f1scale' in scoringFncParams.keys():
        f1scale = scoringFncParams['f1scale']
    else:
        f1scale = 1 
    if 'f0scale' in scoringFncParams.keys():
        f0scale = scoringFncParams['f0scale']
    else:
        f0scale = 1
    diffTraj = np.diff(observable)
    diffSim = np.diff(data) 
    diffVec = np.subtract(diffSim , diffTraj)**2
    F1 = sum(diffVec)/float(f1scale)
    F0 = sum(np.square(data-observable))/float(f0scale)
    if option == 1:
        fitness = F0
    elif option ==2:
        fitness = F1
    else:
        fitness = F0 + F1
    return fitness