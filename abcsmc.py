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
import simulationandscoring
"""
abcsmc.py includes two classes: FittingData and ModelFitting. 
FittingData encapsulates all the data associated with the results of fitting to one single data using ABC-SMC algorithm.
ModelFitting encaptulates all the parameter settings of the ABC-SMC algorithm along with the the FittingData object as the results
"""
class FittingData:
    """
    The class to record the input data and the output to model fitting. 
    Attributes:
        data: time course data. The datatype should be 1D numpy ndarray
        simulatedData: Simulated data of posteriors from all iterations of the algorithm. The data is a list and each of the list elements is the simulated data from the accepted posterior for the rejection sampling iteration. Each population of simulated data is a 2D ndarray, with rows spanning individual simulated data and the columns spanning the number of time points or dimensions of the data    
        distributionData: The data for the fitted parameters of the accepted posteriors. The data is a list and each of the list elements is the population of the parameter vectors from the accepted posterior. Each population of parameter vectors is a 2D ndarray, with rows spanning individual parameter vectors and the columns spanning the number of dimenisons of the parameter vector.      
        timePoints: The time points vector. The data is 1D ndarray.  
        simID: The numerical ID for this particular FittingData object
        referenceVec: The reference parameter vector that is used to construct the samppling range. The values for parameters in the distributionData field are in log10 scale of multiplier with respect to the reference data. 
        scoreList: The goodness of fit for the accepted posteriors. The data is a list and each of the list elements is the population of the goodness of fit from the accepted posterior. Each population of parameter vectors is a 1D ndarray of scores.  
        runningTimeList: The list of running time in each of the iterations of rejection sampling, in hours    
        finalRunTime: The total running time of the sequential Monte Carlo algorithm for this instance of data fitting
        weightList: The data for the re-sampling weights for the accepted posteriors. The data is a list and each of the list elements is the population of the re-sampling weights calculated for the accepted posterior. Each population of weights is a 1D ndarray. 
        success: A boolean value to indicate if the algorithm successfully terminates
    """
    def __init__(self,
                 data=None,
                 simulatedData=None,
                 distributionData=None,
                 timePoints=None,
                 simID=None,
                 referenceVec=None,
                 scoreList = None,
                 runningTimeList = None,
                 finalRunTime = None,
                 weightList = None,
                 success = None,
                 ):
        self.data = data
        self.simulatedData = simulatedData
        self.distributionData = distributionData
        self.timePoints = timePoints
        self.simID = simID
        self.referenceVec = referenceVec   
        self.scoreList = scoreList
        self.runningTimeList = runningTimeList
        self.finalRunTime = finalRunTime
        self.weightList = weightList
        self.success = success
    def GetData(self):
        return self.data
    def SetData(self,data):
        self.data = data
        return 
    def GetSimulatedData(self):
        return self.simulatedData
    def SetSimulatedData(self,simulatedData):
        self.simulatedData = simulatedData
    def AddSimulatedData(self,newSimData):
        """
        Append new data to the existing data attributeb
        """
        if self.simulatedData:
            self.simulatedData.append(newSimData)
        else:
            self.simulatedData = []
            self.simulatedData.append(newSimData)
    def GetDistributionData(self):
        return self.distributionData 
    def SetDistributionData(self,distributionData):
        self.distributionData = distributionData
    def AddDistributionData(self,currDistr):  
        if self.distributionData:
            self.distributionData.append(currDistr)
        else:
            self.distributionData = []
            self.distributionData.append(currDistr)
    def GetTimePoints(self):
        return self.timePoints
    def SetTimePoints(self,timePoints):
        self.timePoints = timePoints
    def GetSimID(self):
        return self.simID
    def SetSimID(self,simID):
        self.simID = simID
    def GetReferenceVec(self):
        return self.referenceVec
    def SetReferenceVec(self,referenceVec):    
        self.referenceVec = referenceVec  
    def GetScoresList(self):
        return self.scoreList 
    def SetScoresList(self,scoreList):
        self.scoreList = scoreList
    def AddScoresList(self,currScores):  
        if self.scoreList:
            self.scoreList.append(currScores)
        else:
            self.scoreList = []
            self.scoreList.append(currScores)
    def GetRunningTimeList(self):
        return self.runningTimeList 
    def SetRunningTimeList(self,runningTimeList):
        self.runningTimeList = runningTimeList
    def AddRunningTimeList(self,currRunningTime):  
        if self.runningTimeList:
            self.runningTimeList.append(currRunningTime)
        else:
            self.runningTimeList = []
            self.runningTimeList.append(currRunningTime)  
    def GetFinalRunTime(self):
        return self.finalRunTime
    def SetFinalRunTime(self,finalRunTime):
        self.finalRunTime = finalRunTime        
    def GetWeightList(self):
        return self.weightList
    def SetWeightList(self,weightList):
        self.weightList = weightList        
    def AddWeightList(self,currWeights):  
        if self.weightList:
            self.weightList.append(currWeights)
        else:
            self.weightList = []
            self.weightList.append(currWeights) 
    def GetSuccess(self):
        return self.success
    def SetSuccess(self,success):
        self.success = success 
    def ReturnFittingData(self):
        returnDict = {'success':self.success,'data':self.data,'simulatedData':self.simulatedData,'distributionData':self.distributionData, 'timePoints': self.timePoints,'simID': self.simID , 'referenceVec':self.referenceVec,'scoreList':self.scoreList,'runningTimeList':self.runningTimeList,'finalRunTime':self.finalRunTime,'weightList':self.weightList }
        return returnDict 
        
class ModelFitting:
    """
    Carries out the Sequential Monte Carlo Approximate Bayesian Computation 
    Attributes:
        fittingData= The object of the FittingData class
        paramList = The list of parameter names that need to be optimized. It is a list of strings. 
        varList = The list of state variables as a list of strings 
        observableName = The state variable name that is the observable and to be used to calculate the fitneess score 
        logFileName= The name of the log file
        previousGenPop = The parameter vector population for a previously terminated run of sequential Monte Carlo   
        previousGenWeights= The assigned re-sampling weights for a previously terminated run of sequential Monte Carlo   
        previousGenScores= The goodness of fit scores for a previously terminated run of sequential Monte Carlo 
        needToGeneratePrior= a boolean flag for whether the algorithm will need to generate a prior population of parameter vectors
        scoreFunc= The name of scoring function between the simulated data and the real data
        scoringFncParams= The dictionary of parameters for the selected simulated data scoring function 
        priorSampleFunc= The name of sampling function for the first prior of parameters    
        posteriorSampleFunc= The name of re-sampling function to produce posteriors for each succession of rejection sampling
        calculateFirstGenWeight= The name of function to calculate the re-sampling weights for the posterior of first iteration of rejection sampling
        calculateWeightFunc= The name of function to calculate the re-sampling weights for the posterior of each succession of rejection sampling
        firstThreshold= The threshold for termination of the first iteration of rejection sampling
        finalThreshold= The threshold for successful termination of the final iteration of rejection sampling 
        popSize= The size of the parameter vector population collected for each iteration of rejection sampling 
        maxRunTime= The time limit on the running of sequential Monte Carlo algorithm 
        option= Numerical indicator for the choice of termination threshold of sequential Monte Carlo. 1 = using a schedule of epsilon. 2 = using a schedule of alpha fraction. 3 = using a single alpha fraction
        alpha= The fraction for relative threshold. The parameter vector will have to at least score within the top alpha fraction of the prior of the current rejection sampling iteration to be considered as part of the posterior  
        epsilonSchedule= The schedule of epsilon threshold 
        alphaSchedule= The schedule of alpha 
        fEvalPreliminary= The name of the fit evaluation function for the posterior from the first iteration of rejection sampling. The function determines whether the parameter vector will be selected as part of the posterior. 
        firstRejectionSamplingTime= The time limit for the first rejection sampling
        terminationFraction= The fraction of the posterior that has to meet the final termination threshold for the iteration to be the terminating iteration of the sequential Monte Carlo algorithm
        covscaling = The covariance scaling factor of the re-sampling function 
        order = The total span of log order. If the order = 2 and the reference value of parameter is x, then the parameter is sampled from a range of [x*10^-1 , x*10]  
        xmlFile = The name of the SBML file of the mathematical model
        simulationStart= The starting time point of the simulation 
        simulationEnd= The ending time point of the simulation
        simulationSteps= The number of time point steps in the simulation 
        equilibriumEnd= The ending time point of simulation for state variables to reach equilibrium
        equilibriumSteps= The number of steps time points of simulation for state variables to reach equilibrium
        perturbParamDict= The list of names for the parameters to be perturbed  
        varInitVals = The list of initial values for state variables 
        varsIniBoundList = The list of boundaries for the state variable initial values 
        varsBoundList = The list of boundaries for the state variables time course values
        varsEquilibriumBoundList = The list of boundaries for the state variables when simulating towarding initial equilibrium before perturbations
        constraintDict = The dictionary that has the list of initial values of state variables, list of boundaries of initial values of state variables, list of boundaries for time course of state variables, list of boundaries for time course of state variables of simulating to equilibrium prior to perturbation 
        CheckConstraintFunc = The handle of function to check for constraints in simulated data 
        ODESolveFunc = The name of the function of ODE solver
    """
    def __init__(self,
                 fittingData= None,
                 paramList = None,
                 varList = None,
                 observableName = None,
                 logFileName=None,
                 previousGenPop =None,
                 previousGenWeights=None,
                 previousGenScores=None,
                 needToGeneratePrior=True,
                 scoreFunc=None,
                 scoringFncParams=None,
                 priorSampleFunc=None,
                 posteriorSampleFunc=None,
                 calculateFirstGenWeight=None,
                 calculateWeightFunc=None,
                 firstThreshold=None,
                 finalThreshold=None,
                 popSize=None,
                 maxRunTime=None,
                 option=None,
                 alpha=None,
                 epsilonSchedule=None,
                 alphaSchedule=None,
                 fEvalPreliminary=None,
                 firstRejectionSamplingTime=None,
                 terminationFraction=None,
                 covscaling=None,
                 order =None,
                 xmlFile = None,
                 simulationStart=None,
                 simulationEnd=None,
                 simulationSteps=None,
                 equilibriumEnd=None,
                 equilibriumSteps=None,
                 perturbParamDict=None,
                 varInitVals=None,
                 varsIniBoundList =None,
                 varsBoundList = None,
                 varsEquilibriumBoundList = None,
                 firstIterTime = None,
                 constraintDict = None,
                 CheckConstraintFunc=None,
                 ODESolveFunc = None,
                 ):
        self.fittingData = fittingData      
        self.varList = varList
        self.observableName = observableName
        self.logFileName = logFileName
        self.paramList = paramList
        self.previousGenPop = previousGenPop
        self.previousGenWeights = previousGenWeights
        self.previousGenScores = previousGenScores
        self.needToGeneratePrior = needToGeneratePrior
        self.scoreFunc = scoreFunc
        self.scoringFncParams = scoringFncParams
        self.priorSampleFunc = priorSampleFunc
        self.posteriorSampleFunc = posteriorSampleFunc
        self.calculateFirstGenWeight = calculateFirstGenWeight
        self.calculateWeightFunc = calculateWeightFunc
        self.firstThreshold = firstThreshold
        self.finalThreshold = finalThreshold
        self.popSize = popSize
        self.maxRunTime = maxRunTime
        self.option = option
        self.alpha = alpha 
        self.epsilonSchedule = epsilonSchedule
        self.alphaSchedule = alphaSchedule
        self.fEvalPreliminary = fEvalPreliminary
        self.firstRejectionSamplingTime = firstRejectionSamplingTime
        self.terminationFraction = terminationFraction
        self.covscaling = covscaling
        self.order = order
        self.xmlFile = xmlFile
        self.simulationStart = simulationStart
        self.simulationEnd = simulationEnd
        self.simulationSteps = simulationSteps
        self.equilibriumEnd = equilibriumEnd
        self.equilibriumSteps = equilibriumSteps
        self.perturbParamDict = perturbParamDict
        self.varInitVals = varInitVals
        self.varsIniBoundList = varsIniBoundList
        self.varsBoundList = varsBoundList
        self.varsEquilibriumBoundList = varsEquilibriumBoundList
        self.constraintDict = constraintDict
        self.CheckConstraintFunc = CheckConstraintFunc
        self.ODESolveFunc = ODESolveFunc
    def OutputFittingData(self):
        returnFittingData = self.fittingData.ReturnFittingData()
        pickle.dump(returnFittingData,open("instance"+ str(self.fittingData.simID) + "_fittingData.p","wb"))
        # return the name of the file 
        return "instance"+ str(self.fittingData.simID) + "_fittingData.p"
    def GetFittingData(self):
        return self.fittingData
    def SetFittingData(self,newFittingData):
        self.fittingData = newFittingData
    def GetVarList(self):
        return self.varList
    def SetVarList(self,varList):
        self.varList = varList
    def GetObservableName(self):
        return self.observableName
    def SetObservableName(self,obs):
        self.observableName = observableName
    def GetLogFileName(self):
        return self.logFileName
    def SetLogFileName(self,logFileName):
        self.logFileName = logFileName
    def GetParamList(self):
        return self.paramList
    def SetParamList(self,paramList):
        self.paramList = paramList
    def GetPreviousGenPop(self):
        return self.previousGenPop
    def SetPreviousGenPop(self,previousGenPop):
        self.previousGenPop = previousGenPop
    def GetPreviousGenWeights(self):
        return self.previousGenWeights
    def SetPreviousGenWeights(self,previousGenWeights):
        self.previousGenWeights = previousGenWeights
    def GetPreviousGenScores(self):
        return self.previousGenScores
    def SetPreviousGenScores(self,previousGenScores):
        self.previousGenScores = previousGenScores
    def GetNeedToGeneratePrior(self):
        return self.needToGeneratePrior
    def SetNeedToGeneratePrior(self,needToGeneratePrior):
        self.needToGeneratePrior = needToGeneratePrior
    def GetScoreFunc(self):
        return self.scoreFunc
    def SetScoreFunc(self,scoreFunc):
        self.scoreFunc = scoreFunc
    def GetScoringFncParams(self):
        return self.scoringFncParams
    def SetScoringFncParams(self,scoringFncParams):
        self.scoringFncParams = scoringFncParams
    def GetPriorSampleFunc(self):
        return self.priorSampleFunc
    def SetPriorSampleFunc(self,priorSampleFunc):
        self.priorSampleFunc = priorSampleFunc
    def GetPosteriorSampleFunc(self):
        return self.posteriorSampleFunc
    def SetPosteriorSampleFunc(self,posteriorSampleFunc):
        self.posteriorSampleFunc = posteriorSampleFunc
    def GetCalculateFirstGenWeight(self):
        return self.calculateFirstGenWeight
    def SetCalculateFirstGenWeight(self,calculateFirstGenWeight):
        self.calculateFirstGenWeight = calculateFirstGenWeight
    def GetCalculateWeightFunc(self):
        return self.calculateWeightFunc
    def SetCalculateWeightFunc(self,calculateWeightFunc):
        self.calculateWeightFunc = calculateWeightFunc
    def GetFirstThreshold(self):
        return self.firstThreshold
    def SetFirstThreshold(self,firstThreshold):
        self.firstThreshold = firstThreshold
    def GetFinalThreshold(self):
        return self.finalThreshold
    def SetFinalThreshold(self,finalThreshold):
        self.finalThreshold = finalThreshold
    def GetPopSize(self):
        return self.popSize
    def SetPopSize(self,popSize):
        self.popSize = popSize
    def GetMaxRunTime(self):
        return self.maxRunTime
    def SetMaxRunTime(self,maxRunTime):
        self.maxRunTime = maxRunTime 
    def GetOption(self):
        return self.option
    def SetOption(self,option):
        self.option = option 
    def GetAlpha(self):
        return self.alpha
    def SetAlpha(self,alpha):
        self.alpha = alpha       
    def GetEpsilonSchedule(self):
        return self.epsilonSchedule
    def SetEpsilonSchedule(self,epsilonSchedule):
        self.epsilonSchedule = epsilonSchedule
    def GetAlphaSchedule(self):
        return self.alphaSchedule
    def SetAlphaSchedule(self,alphaSchedule):
        self.alphaSchedule = alphaSchedule
    def GetFEvalPreliminary(self):
        return self.fEvalPreliminary
    def SetFEvalPreliminary(self,fEvalPreliminary):
        self.fEvalPreliminary = fEvalPreliminary
    def GetFirstRejectionSamplingTime(self):
        return self.firstRejectionSamplingTime
    def SetFirstRejectionSamplingTime(self,firstRejectionSamplingTime):
        self.firstRejectionSamplingTime = firstRejectionSamplingTime
    def GetTerminationFraction(self):
        return self.terminationFraction
    def SetTerminationFraction(self,terminationFraction):
        self.terminationFraction = terminationFraction
    def GetCovscaling(self):
        return self.covscaling
    def SetCovscaling(self,covscaling):
        self.covscaling = covscaling
    def GetOrder(self):
        return self.order
    def SetOrder(self,order):
        self.order = order
    def GetXmlFile(self):
        return self.xmlFile
    def SetXmlFile(self):
        self.xmlFile = xmlFile
    def GetSimulationStart(self):
        return self.simulationStart
    def SetSimulationStart(self,simulationStart):
        self.simulationStart = simulationStart
    def GetSimulationEnd(self):
        return self.simulationEnd
    def SetSimulationEnd(self,simulationEnd):
        self.simulationEnd = simulationEnd
    def GetSimulationSteps(self):
        return self.simulationSteps
    def SetSimulationSteps(self,simulationSteps):
        self.simulationSteps = simulationSteps 
    def GetEquilibriumEnd(self):
        return self.equilibriumEnd
    def SetEquilibriumEnd(self,equilibriumEnd):
        self.equilibriumEnd = equilibriumEnd           
    def GetEquilibriumSteps(self):
        return self.equilibriumSteps
    def SetEquilibriumSteps(self,equilibriumSteps):
        self.equilibriumSteps = equilibriumSteps
    def GetPerturbParamDict(self):
        return self.perturbParamDict
    def SetPerturbParamDict(self,perturbParamDict):
        self.perturbParamDict = perturbParamDict
    def GetVarInitVals(self):
        return self.varInitVals
    def SetVarInitVals(self,varInitVals):
        self.varInitVals = varInitVals 
    def GetVarsIniBoundList(self):
        return self.varsIniBoundList
    def SetVarsIniBoundList(self,varsIniBoundList):
        self.varsIniBoundList = varsIniBoundList
    def GetVarsBoundList(self):
        return self.varsBoundList
    def SetVarsBoundList(self,varsBoundList):
        self.varsBoundList = varsBoundList
    def GetVarsEquilibriumBoundList(self):
        return self.varsEquilibriumBoundList
    def SetVarsEquilibriumBoundList(self,varsEquilibriumBoundList):
        self.varsEquilibriumBoundList = varsEquilibriumBoundList          
    def GetConstraintDict(self):
        return self.constraintDict
    def SetConstraintDict(self,constraintDict):
        self.constraintDict = constraintDict
    def GetCheckConstraintFunc(self):
        return self.CheckConstraintFunc
    def SetCheckConstraintFunc(self,CheckConstraintFunc):
        self.CheckConstraintFunc = CheckConstraintFunc
    def GetODESolveFunc(self):
        return self.ODESolveFunc
    def SetODESolveFunc(self,ODESolveFunc):
        self.ODESolveFunc = ODESolveFunc
    def RejectionSampling(self,ODESolveFunc,fSampleFunc,fAccept,paramList=None,fScore=None,popSize=None,rejectionTime=None,fileObj=None,CheckConstraintFunc=None): 
        """
        Perform a single iteration of rejection sampling 
        Input-
        ODESolveFunc: The ode sovler function that takes in a list of parameters and output the dictionary of results and possibly results from simulation towards initial equilibrium     
        paramNum: The number of dimensions in the parameter space 
        fScore: The handle of scoring function 
        fSampleFunc: The handle of sampling function 
        fAccept: The handle of function to accept the particles 
        popSize: The population size of the estimated posterior 
        rejectiontime: Time limit of rejeciton sampling 
        fileObj:file object
        CheckConstraintFunc : The function to check for the constraints 
        """
        if not paramList:
            if self.GetParamList():
                paramList = self.GetParamList()
            else:
                raise Exception('Missing list of parameters')
        if not fScore:
            if self.GetScoreFunc():
                fScore = self.GetScoreFunc()
            else:
                raise Exception('Missing scoring function')
        if not popSize:
            if self.GetPopSize():
                popSize = self.GetPopSize()
            else:
                raise Exception('Need to specify size of population')   
        if not rejectionTime:
            rejectionTime = 0.5
        if not fileObj:
            fileObj = open('logfile.txt','w+')
        if not CheckConstraintFunc:
            if self.GetCheckConstraintFunc():
                #print('Set constraint')
                CheckConstraintFunc = self.GetCheckConstraintFunc()
        fileObj.write('Started rejection sampling\n')
        #print('Started rejection sampling')
        # initialize array
        paramNum = len(paramList)
        finalScores = np.full((popSize,1),np.inf)
        finalPop = np.full((popSize,paramNum),np.inf)
        finalSimData = np.array([])
        totalAttempts = 0
        t0 = datetime.now()
        #import pdb; pdb.set_trace()
        fileObj.write('about to ener into while loop\n')
        #print('about to ener into while loop')
        count = 0
        fileObj.write('\r')
        #print('\r')
        # the flag to determine whether the optimization is successful or not
        success  = True 
        while (count < popSize) and success == True:
            #print('Inside while loop')
            # Use fPriorSample to generate a single sample from the prior distribution 
            priorP = fSampleFunc()
            #
            #print('Got a new priorP')
            #print(priorP)
            K = self.GetFittingData().GetReferenceVec()*10**(priorP)
            #print('Got K')
            #print(K)
            # simulate the model with the given parameters
            outputODESolver = ODESolveFunc(K)
            #print('Finished ODE simulation')
            #print(outputODESolver)
            # check for the constraints 
            outputCheck = True
            if CheckConstraintFunc:
                #print('Check constraint')
                outputCheck = CheckConstraintFunc(outputODESolver)
            # score the simulation            
            if outputCheck: 
                #print("Use constraint")
                score = fScore(outputODESolver['obs'])
            else:
                score = np.inf
            # fScore returns a True or False boolean value
            outfEval = fAccept(score)
            if outfEval == True:
                finalPop[count,:] = priorP 
                finalScores[count] = score
                if len(finalSimData)>0:
                    finalSimData = np.row_stack((finalSimData,outputODESolver['obs'] ))
                else:
                    finalSimData = outputODESolver['obs']                
                count +=1
                fileObj.write('count = '+str(count)+'elapsed time = '+str((datetime.now() -t0).total_seconds()/3600)+' hours\n')
                fileObj.write('score = '+ str(score)+ '\n')                
                #print('count = '+str(count)+' elapsed time = '+str((datetime.now() -t0).total_seconds()/3600)+' hours. score ='+ str(score)+'\r')
            totalAttempts += 1
            if rejectionTime < (datetime.now() -t0).total_seconds()/3600:
                success = False
        finalTime = (datetime.now() - t0).total_seconds()/3600 
        fileObj.write('\r')
        #print('\r')
        fileObj.write('Finished rejection sampling\n')
        #print('Finished rejection sampling')
        fileObj.write('elapsed time = '+str((datetime.now() -t0).total_seconds()/3600)+' hours\n')
        #print('elapsed time = '+str((datetime.now() -t0).total_seconds()/3600)+' hours')
        return {'success':success, 'finalSimData':finalSimData,'finalPop':finalPop,'finalScores':finalScores,'totalAttempts':totalAttempts,'finalTime':finalTime,'fileObj':fileObj}
    
    def SequentialMonteCarlo(self,**kwargs):
        """
        Perform sequential Monte Carlo algorithm  
        Input-
        varLIst = The list of state variables as a list of strings 
        logfile = The name of the log file
        paramList = The list of parameter names that need to be optimized. It is a list of strings. 
        previousGenPop = The parameter vector population for a previously terminated run of sequential Monte Carlo   
        previousGenWeights = The assigned re-sampling weights for a previously terminated run of sequential Monte Carlo
        previousGenScores = The goodness of fit scores for a previously terminated run of sequential Monte Carlo 
        needToGeneratePrior = a boolean flag for whether the algorithm will need to generate a prior population of parameter vectors
        simID = The numerical ID for this particular FittingData object
        ODESolveFunc = The name of the function of ODE solver
        scoreFunc = The name of scoring function between the simulated data and the real data
        priorSampleFunc = The name of sampling function for the first prior of parameters 
        posteriorSampleFunc = The name of re-sampling function to produce posteriors for each succession of rejection sampling
        calculateFirstGenWeight = The name of function to calculate the re-sampling weights for the posterior of first iteration of rejection sampling
        calculateWeightFunc = The name of function to calculate the re-sampling weights for the posterior of each succession of rejection sampling
        firstThreshold = The threshold for termination of the first iteration of rejection sampling
        finalThreshold = The threshold for successful termination of the final iteration of rejection sampling 
        popSize = The size of the parameter vector population collected for each iteration of rejection sampling 
        maxRunTime = The time limit on the running of sequential Monte Carlo algorithm 
        option= Numerical indicator for the choice of termination threshold of sequential Monte Carlo. 1 = using a schedule of epsilon. 2 = using a schedule of alpha fraction. 3 = using a single alpha fraction
        epsilonSchedule = The schedule of epsilon threshold 
        alphaSchedule = The schedule of alpha
        alpha= The fraction for relative threshold. The parameter vector will have to at least score within the top alpha fraction of the prior of the current rejection sampling iteration to be considered as part of the posterior  
        fEvalPreliminary= The name of the fit evaluation function for the posterior from the first iteration of rejection sampling. The function determines whether the parameter vector will be selected as part of the posterior. 
        firstRejectionSamplingTime = The time limit for the first rejection sampling 
        terminationFraction = The fraction of the posterior that has to meet the final termination threshold for the iteration to be the terminating iteration of the sequential Monte Carlo algorithm
        covscaling = The covariance scaling factor of the re-sampling function 
        CheckConstraintFunc = The handle of function to check for constraints in simulated data 
        """
        t0 = datetime.now()
        if 'varList' in kwargs:
            varList = kwargs['varList']
        elif self.GetVarList():
            varList = self.GetVarList()
        else: 
            raise Exception('Need to specify the list of variables to calculate for fitness and constraints')
        if 'logfile' in kwargs:
            logFileName = kwargs['logfile'] 
        elif self.GetLogFileName(): 
            logFileName = self.GetLogFileName()
        else:
            logFileName = 'logfile'
        if 'paramList' in kwargs:
            paramList = kwargs['paramList']
        else:
            if self.GetParamList():
                paramList = self.GetParamList()
            else:
                raise Exception('Missing list of parameters')
        if 'previousGenPop' in kwargs:
            previousGenPop = kwargs['previousGenPop']
        elif self.GetPreviousGenPop():
            previousGenPop= self.GetPreviousGenPop()
        else:
            needToGeneratePrior = True
        if 'previousGenWeights' in kwargs:
            previousGenWeights = kwargs['previousGenWeights']
        elif self.GetPreviousGenWeights():
            previousGenWeights = self.GetPreviousGenWeights()
        else:
            needToGeneratePrior = True
            
        if 'previousGenScores' in kwargs:
            previousGenScores = kwargs['previousGenScores']
        elif self.GetPreviousGenScores():
            previousGenScores = self.GetPreviousGenScores()
        else:    
            needToGeneratePrior = True
        if 'needToGeneratePrior' in kwargs:
            needToGeneratePrior = kwargs['needToGeneratePrior']
        elif self.GetNeedToGeneratePrior():
            needToGeneratePrior = self.GetNeedToGeneratePrior()
        else:
            needToGeneratePrior = True
        if 'simID' in kwargs:
            simID = kwargs['simID']
        else:
            if self.GetFittingData().GetSimID()!= None:
                simID = self.GetFittingData().GetSimID()
            else:
                raise Exception('Missing simID')
        if 'ODESolveFunc' in kwargs: 
            ODESolveFunc = kwargs['ODESolveFunc'] 
        elif self.GetODESolveFunc():
            #print('Get ODE from self')
            ODESolveFunc = self.GetODESolveFunc()
        else: 
            raise Exception('Missing the ODE solver option')
        if 'scoreFunc' in kwargs:
            scoreFunc = kwargs['scoreFunc']
        else:
            if self.GetScoreFunc():
                scoreFunc = self.GetScoreFunc()
            else:
                raise Exception('Missing scoreFunc')
        if 'priorSampleFunc' in kwargs:
            priorSampleFunc = kwargs['priorSampleFunc']
        else:
            if self.GetPriorSampleFunc():
                priorSampleFunc = self.GetPriorSampleFunc()
        # Sample data points given weights from the previous iteration
        if 'posteriorSampleFunc' in kwargs:
            posteriorSampleFunc = kwargs['posteriorSampleFunc']
        else:
            if self.GetPosteriorSampleFunc():
                posteriorSampleFunc = self.GetPosteriorSampleFunc()
            else:
                raise Exception('Missing posteriorSampleFunc')  
        # Calculate weights of the first population
        if 'calculateFirstGenWeight' in kwargs:
            calculateFirstGenWeight = kwargs['calculateFirstGenWeight']       
        else:
            if self.GetCalculateFirstGenWeight():
                calculateFirstGenWeight = self.GetCalculateFirstGenWeight()
        # Calculate new weights given current weights
        if 'calculateWeightFunc' in kwargs:
            calculateWeightFunc = kwargs['calculateWeightFunc']
        else: 
            if self.GetCalculateWeightFunc():
                calculateWeightFunc = self.GetCalculateWeightFunc()
            else:
                raise Exception('calculateWeightFunc missing')                
        if 'firstThreshold' in kwargs:
            firstThreshold = kwargs['firstThreshold']
        else:
            if self.GetFirstThreshold():
                firstThreshold = self.GetFirstThreshold()
            else:
                if needToGeneratePrior:
                    raise Exception('firstThreshold missing')
        if 'finalThreshold' in kwargs:
            finalThreshold = kwargs['finalThreshold']
        else:
            if self.GetFinalThreshold():
                finalThreshold = self.GetFinalThreshold()
            else:
                raise Exception('finalThreshold missing')
        if 'popSize' in kwargs:
            popSize = kwargs['popSize']
        else:
            if self.GetPopSize(): 
                popSize = self.GetPopSize()
            else:
                raise Exception('popSize missing')
        if 'maxRunTime' in kwargs:   
            maxRunTime = kwargs['maxRunTime']
        else:
            if self.GetMaxRunTime(): 
                maxRunTime =  self.GetMaxRunTime()
            else:
                raise Exception('maxRunTime missing')
        if 'option' in kwargs:
            option = kwargs['option']
        else:
            if self.GetOption():
                option = self.GetOption()
            else:
                raise Exception('Need to specify the schedule option')                
        # option =1 means epsilonSchedule; option =2 means alphaSchdule; option =3 means = alpha
        # check for input epsilonSchedule
        if 'epsilonSchedule' in kwargs:
            epsilonSchedule = kwargs['epsilonSchedule']
        else:
            if self.GetEpsilonSchedule():
                epsilonSchedule = self.GetEpsilonSchedule()
        # The schedule of alpha. The potential particle has to score better than the alpha fraction of the previous iteration 
        if 'alphaSchedule' in kwargs:
            alphaSchedule = kwargs['alphaSchedule']
        else:
            if self.GetAlphaSchedule():
                alphaSchedule = self.GetAlphaSchedule()
        # One constant alpha throughout the iterations. If using this option then the termination fraction has to be specified. 
        if 'alpha' in kwargs:
            alpha = kwargs['alpha']
        else:
            if self.GetAlpha(): 
                alpha = self.GetAlpha()
        # check if the schedule information is set 
        if not ('epsilonSchedule' in vars() or 'epsilonSchedule' in globals()) and not('alphaSchedule' in vars() or 'alphaSchedule' in globals()) and not('alpha' in vars() or 'alpha' in globals() ):
            raise Exception('Need to provide either epsilonSchedule or alphaSchedule or alpha')
        # the evaluation function for first population 
        if 'fEvalPreliminary' in kwargs:
            fEvalPreliminary = kwargs['fEvalPreliminary']
        else:
            if self.GetFEvalPreliminary():
                fEvalPreliminary = self.GetFEvalPreliminary()
            else:
                if needToGeneratePrior:  
                    raise Exception( 'fEvalPreliminary missing')           
        if 'firstRejectionSamplingTime' in kwargs:
            firstRejectionSamplingTime = kwargs['firstRejectionSamplingTime']
        else:
            if self.GetFirstRejectionSamplingTime(): 
                firstRejectionSamplingTime = self.GetFirstRejectionSamplingTime()
            else:
                if needToGeneratePrior: 
                    firstRejectionSamplingTime = 0.5
                    raise Exception('firstRejectionSamplingTime missing')  
        # termination fraction is the fraction of the posterior population that has to meet the final threshold
        if 'terminationFraction' in kwargs:
            terminationFraction = kwargs['terminationFraction']
        else:
            if self.GetTerminationFraction():
                terminationFraction = self.GetTerminationFraction()
            else:
                terminationFraction = 1        
        # 
        if 'covscaling' in kwargs:
            covscaling = kwargs['covscaling']
        else:
            if self.GetCovscaling():
                covscaling = self.GetCovscaling()
            else:
                covscaling = 1            
        #  
        if 'CheckConstraintFunc' in kwargs:
            CheckConstraintFunc = kwargs['CheckConstraintFunc']
        elif self.GetCheckConstraintFunc():
            #print('Get CheckConstraintFunc from self')
            CheckConstraintFunc = self.GetCheckConstraintFunc() 
        elif self.GetConstraintDict():    
            CheckConstraintFunc = lambda x: simulationandscoring.CheckConstraints(x,self.GetConstraintDict())    
        # initialize the object for output data
        if not needToGeneratePrior and (not ('firstGenPop' in kwargs) or not ('firstGenWeights' in kwargs) or not ('firstGenScores' in kwargs)):
            raise Exception('firstGenPop and firstGenWeights and firstGenScores have to be all specified')
        # Check if the information needed for first generation weights is present
        if ('firstGenWeights' in kwargs) and ('calculatePriorWeightFunc' in kwargs):
            raise Exception('Both firstGenWeights and calculatePriorWeightFunc are not present. Need at least one present to proceed with rejection sampling')
        f1 = open(logFileName+'.txt','w+')
        #print('About to run the initial rejection sampling')
        f1.write('About to run the initial rejection sampling\n')    
        # Get the fittingData object in its current state 
        thisFittingData = self.GetFittingData()
        success = True
        # Run rejection sampling from prior
        if needToGeneratePrior: 
            #import pdb; pdb.set_trace()
            rejectionSamplingOut = self.RejectionSampling(ODESolveFunc,priorSampleFunc,fEvalPreliminary,paramList=paramList,fScore=scoreFunc,popSize=popSize,rejectionTime=firstRejectionSamplingTime,fileObj=f1,CheckConstraintFunc=CheckConstraintFunc)                      
            # Get the output of rejection sampling
            success = rejectionSamplingOut['success']
            firstGenPop = rejectionSamplingOut['finalPop']
            firstGenScores = rejectionSamplingOut['finalScores']
            firstRunningTime = rejectionSamplingOut['finalTime']  
            firstSimData = rejectionSamplingOut['finalSimData']
            f1 = rejectionSamplingOut['fileObj']
            # Calculate the weights of the first population 
            firstGenWeights = calculateFirstGenWeight(firstGenPop) 
        #print('Finished with initial rejection sampling')
        f1.write('Finished with initial rejection sampling\n')
        # Append the fit results to the fitting daa object
        #import pdb; pdb.set_trace()
        thisFittingData.AddDistributionData(firstGenPop)
        thisFittingData.AddSimulatedData(firstSimData)
        thisFittingData.AddScoresList(firstGenScores)
        thisFittingData.AddRunningTimeList(firstRunningTime)
        thisFittingData.AddWeightList(firstGenWeights)
        # Initialize the list for all populaitons
        AllPops = []
        AllPops.append(firstGenPop)
        # Initialize the list for all scores
        AllScores = []
        AllScores.append(firstGenScores)
        # Initialize the list for all weights
        AllWeights = []
        AllWeights.append(firstGenWeights)
        # Initialize the list for all running times 
        AllRunnnigTimes = []
        AllRunnnigTimes.append(firstRunningTime)
        # Initialize the list fo all simulated data
        AllSimData = []
        AllSimData.append(firstSimData)
        self.SetFittingData(thisFittingData)
        pickle.dump(AllPops,open("AllPops.p","wb"))
        pickle.dump(AllScores,open("AllScores.p","wb"))
        pickle.dump(AllWeights,open("AllWeights.p","wb"))
        pickle.dump(AllSimData,open("AllSimData.p","wb"))
        pickle.dump(AllRunnnigTimes,open("AllRunnnigTimes.p","wb"))
        pickle.dump(thisFittingData,open("FittingDataSnapShot.p","wb"))
        f1.write('Appended the information of initial rejection sampling\n')
        #print('Appended the information of initial rejection sampling')
        f1.write('Mean Score of Current Iteration '+ str(np.mean(rejectionSamplingOut['finalScores']))+'\n' )
        #print('Mean Score of Current Iteration ' + str(np.mean(rejectionSamplingOut['finalScores'])))
        # canTerminate is the boolean variable to determine whether to terminate optimization 
        canTerminate = False
        # initialize the counter for Bayeisan fitting iteration 
        iterationCounter = 0 
        finalRunTime = 0
        # Check if the optimization timed out at the preliminary rejection sampling 
        if success == True: 
            #import pdb; pdb.set_trace()
            # Iteratively sample for the posterios
            while (~canTerminate) and success == True:
                # calculate the time limit for this iteration of rejection sampling
                f1.write('iteration = ' + str(iterationCounter)+'\n')
                #print('iteration = ' + str(iterationCounter))
                rejectionTime = maxRunTime - (datetime.now() - t0).total_seconds()/3600 
                # initialize the fEval for this iteration depending on the input choices and the current iteration 
                f1.write('About to start new iteration of rejection sampling\n')
                #print('About to start new iteration of rejection sampling')      
                if option == 1:
                    # Construct new evaluation function t
                    if iterationCounter > len(epsilonSchedule)-1:
                        iterationCounter = len(epsilonSchedule)-1
                    fEvalCurr = lambda inputScore: inputScore < epsilonSchedule[iterationCounter]
                    rejectionSamplingOutput = self.RejectionSampling(ODESolveFunc,lambda: posteriorSampleFunc(AllPops[iterationCounter],AllWeights[iterationCounter]),fEvalCurr,paramList=paramList,fScore=scoreFunc,popSize=popSize,rejectionTime=rejectionTime,fileObj=f1,CheckConstraintFunc=CheckConstraintFunc)
                elif option == 2:
                    # the score from top alpha fraction from the rejection sampling output
                    if iterationCounter > len(alphaSchedule)-1:
                        iterationCounter = len(alphaSchedule)-1                
                    thresholdScore = np.percentile(AllScores[iterationCounter],alphaSchedule[iterationCounter])
                    fEvalCurr = lambda inputScore: inputScore < thresholdScore
                    rejectionSamplingOutput = self.RejectionSampling(ODESolveFunc,lambda: posteriorSampleFunc(AllPops[iterationCounter],AllWeights[iterationCounter]),fEvalCurr,paramList=paramList,fScore=scoreFunc,popSize=popSize,rejectionTime=rejectionTime,fileObj=f1,CheckConstraintFunc=CheckConstraintFunc)        
                else:
                    #print('option 3')
                    #print('iterationCounter = ' + str(iterationCounter))
                    #print('alpha = ' + str(alpha))
                    thresholdScore = np.percentile(AllScores[iterationCounter],alpha*100)
                    #print('current iteration threshold = '+str(thresholdScore))
                    fEvalCurr = lambda inputScore: inputScore < thresholdScore
                    rejectionSamplingOutput = self.RejectionSampling(ODESolveFunc,lambda: posteriorSampleFunc(AllPops[iterationCounter],AllWeights[iterationCounter]),fEvalCurr,paramList=paramList,fScore=scoreFunc,popSize=popSize,rejectionTime=rejectionTime,fileObj=f1,CheckConstraintFunc=CheckConstraintFunc)                
                f1 = rejectionSamplingOut['fileObj']
                f1.write('Finished with rejection sampling\n')
                #print('Finished with rejection sampling')
                f1.write('Calculate new weights\n')
                #print('Calculate new weights')
                # calculate the weights for next iteration 
                newWeights = calculateWeightFunc(rejectionSamplingOutput['finalPop'],AllPops[iterationCounter],AllWeights[iterationCounter])
                # Update the populations, scores and weights
                AllPops.append(rejectionSamplingOutput['finalPop']) 
                AllScores.append(rejectionSamplingOutput['finalScores'])
                AllWeights.append(newWeights)
                AllRunnnigTimes.append(rejectionSamplingOutput['finalTime'])
                AllSimData.append(rejectionSamplingOut['finalSimData'])            
                thisFittingData.AddDistributionData(rejectionSamplingOutput['finalPop'])
                thisFittingData.AddSimulatedData(rejectionSamplingOutput['finalSimData'])
                thisFittingData.AddScoresList(rejectionSamplingOutput['finalScores'])
                thisFittingData.AddRunningTimeList(rejectionSamplingOutput['finalTime'])    
                thisFittingData.AddWeightList(newWeights)  
                success = rejectionSamplingOutput['success']
                f1.write('Appended the results of current iteration\n')
                #print('Appended the results of current iteration')
                # update the loop counter 
                iterationCounter += 1
                # determine if the loop can be terminated
                f1.write('Check termination criterion\n')
                #print('Check termination criterion')
                if option == 1:
                    lastScores = AllScores[-1]
                    finalThreshold = epsilonSchedule[-1]
                    currFraction = sum(s< finalThreshold for s in lastScores)/float(len(lastScores))
                    #print('currFraction = '+ str(currFraction))
                    canTerminate = currFraction >= terminationFraction            
                elif option == 2:
                    lastScores = AllScores[-1]
                    currFraction = sum(s < finalThreshold for s in lastScores)/float(len(lastScores))
                    #print('currFraction = '+ str(currFraction))
                    canTerminate = currFraction >= alphaSchdule[-1]                      
                else:
                    lastScores = AllScores[-1]            
                    currFraction = sum(s < finalThreshold for s in lastScores)/float(len(lastScores))
                    #print('currFraction = '+ str(currFraction))
                    canTerminate = currFraction >= terminationFraction      
                # dump the current variables 
                f1.write('Save current variables\n')
                #print('Save current variables')
                self.SetFittingData(thisFittingData)
                pickle.dump(AllPops,open("AllPops.p","wb"))
                pickle.dump(AllScores,open("AllScores.p","wb"))
                pickle.dump(AllWeights,open("AllWeights.p","wb"))
                pickle.dump(AllRunnnigTimes,open("AllRunnnigTimes.p","wb"))
                pickle.dump(AllSimData,open("AllSimData.p","wb"))
                pickle.dump(self.fittingData,open("FittingDataSnapShot.p","wb"))
                f1.write('Finished iteration number '+ str(iterationCounter)+'\n')        
                #print('Finished iteration number '+ str(iterationCounter))
                f1.write('Mean Score of Current Iteration ' + str(np.mean(rejectionSamplingOutput['finalScores']))+'\n')
                #print('Mean Score of Current Iteration ' + str(np.mean(rejectionSamplingOutput['finalScores'])))
                f1.write('Final Threshold '+str(finalThreshold)+'\n')
                #print('Final Threshold '+str(finalThreshold))
                f1.write('Current running time = '+str((datetime.now() -t0).total_seconds()/3600)+' hours\n')
                #print('Current running time = '+str((datetime.now() -t0).total_seconds()/3600),' hours')
        self.fittingData.SetSuccess(success)
        #print('Finished ABCSMC')
        finalRunTime = (datetime.now() -t0).total_seconds()/3600
        self.fittingData.SetFinalRunTime(finalRunTime)
        return {'success':success,'AllPops':AllPops,'AllScores':AllScores,'AllWeights':AllWeights,'finalRunTime':finalRunTime,'simID':simID}
