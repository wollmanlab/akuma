# Read the inputfile and execute fitting 
''' 
This notebook tests the correctness of rejection sampling with the Yao calcium model
'''
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
import ast
import abcsmc
import parametersampling as ps
import sampleweighting as sw
import simulationandscoring as sc
# Read in the input file name 
inputFileName = sys.argv[1]
f = open('hasstarted','wb')
f.write('simulation has started')
f.close()
inputDict = {}
with open(inputFileName,"r") as f:
    for line in f:
        lineStrip = line.lstrip()
        if bool(lineStrip):
            if lineStrip[0] != "#" and lineStrip[0] != '\n':
                parsedList = lineStrip.split("=") 
                varName = parsedList[0].strip()
                #print(varName)
                #print parsedList
                inputDict[varName] = ast.literal_eval(parsedList[1].strip())


# In[ ]:

# Get the single cell data needed from the matrix
# population data
if 'needToGeneratePrior' in inputDict.keys() and inputDict['needToGeneratePrior']:
    dataMat = pickle.load(open(inputDict['dataMat'],"rb"))
    # simulation ID's
    simIDList =np.array(pickle.load(open(inputDict['simIDList'],"rb")))
    # 
    thresholdList =np.array(pickle.load(open(inputDict['thresholdList'],'rb')))
    # reference parameter vector
    referenceVec = inputDict['referenceVec']
    timePoints =np.array(pickle.load(open(inputDict['timePoints'],'rb')))  
    simulationEnd = int(timePoints[-1])
    simulationSteps = int(len(timePoints)-1)
    simulationStart =int(timePoints[0])
    # read the index
    index = pickle.load(open('index','rb'))
    simID = simIDList[index]
    thisIndex = np.where(simIDList == simID)[0][0]
    thisData = dataMat[thisIndex,:]
    thisThreshold = thresholdList[thisIndex]
    # create the fitting data object
    thisFittingData = abcsmc.FittingData(data=thisData,timePoints=timePoints,simID = simID,referenceVec=referenceVec)
    # get the finalThreshold for the current cell
    finalThreshold = thisThreshold
    # create the model fitting object    
    modelFitting = abcsmc.ModelFitting(fittingData=thisFittingData,paramList=inputDict['paramList'],varList=inputDict['varList'],observableName=inputDict['observableName'],needToGeneratePrior=True,firstThreshold= inputDict['firstThreshold'],finalThreshold=thisThreshold,popSize= inputDict['popSize'], option = inputDict['option'],xmlFile= inputDict['modelFile'],simulationEnd= simulationEnd,simulationSteps=simulationSteps,simulationStart= simulationStart,varInitVals =inputDict['varInitVals'])


# In[ ]:

if modelFitting.GetOption() == 1:
    modelFitting.SetEpsilonSchedule(inputDict['epsilonSchedule'])
elif modelFitting.GetOption() == 2:
    modelFitting.SetAlphaSchedule(inputDict['alphaSchedule'])
elif modelFitting.GetOption() == 3:
    modelFitting.SetAlpha(inputDict['alpha'])
else:
    raise ValueError('option has to be either 1,2 or 3')
# 
if 'scoringFncParams' in inputDict.keys():
    modelFitting.SetScoringFncParams(inputDict['scoringFncParams'])
else:
    modelFitting.SetScoringFncParams(None)
if 'terminationFraction' in inputDict.keys():
    modelFitting.SetTerminationFraction(inputDict['terminationFraction'])
else: 
    modelFitting.SetTerminationFraction(1)
if 'covscaling' in inputDict.keys():
    modelFitting.SetCovscaling(inputDict['covscaling'])
else:
    modelFitting.SetCovscaling(1)
if 'order' in inputDict.keys():
    modelFitting.SetOrder(inputDict['order'])
else:
    modelFitting.SetOrder(2)
if 'equilibriumEnd' in inputDict.keys():
    modelFitting.SetEquilibriumEnd(int(inputDict['equilibriumEnd']))
else:
    modelFitting.SetEquilibriumEnd(None)
if 'equilibriumSteps' in inputDict.keys():
    modelFitting.SetEquilibriumSteps(int(inputDict['equilibriumSteps']))
else:
    modelFitting.SetEquilibriumSteps(None)
if 'perturbParamList' in inputDict.keys() and 'perturbParamValueList' in inputDict.keys():
    perturbParamDict = dict((inputDict['perturbParamList'][i],inputDict['perturbParamValueList'][i]) for i in range(len(inputDict['perturbParamList'])))
    modelFitting.SetPerturbParamDict(perturbParamDict)
else:
    modelFitting.SetPerturbParamDict(None)
if 'maxRunTime' in inputDict.keys():
    modelFitting.SetMaxRunTime(inputDict['maxRunTime'])
else:
    modelFitting.SetMaxRunTime(1)
if 'varsIniBoundList' in inputDict.keys():
    modelFitting.SetVarsIniBoundList(inputDict['varsIniBoundList'])
if 'varsBoundList' in inputDict.keys():
    modelFitting.SetVarsBoundList(inputDict['varsBoundList'])
if 'varsEquilibriumBoundList' in inputDict.keys(): 
    modelFitting.SetVarsEquilibriumBoundList(inputDict['varsEquilibriumBoundList'])
if 'firstRejectionSamplingTime' in inputDict.keys():
    modelFitting.SetFirstRejectionSamplingTime(inputDict['firstRejectionSamplingTime'] )
else:
    modelFitting.SetFirstRejectionSamplingTime(0.5)


# In[ ]:

# Create preliminary evaluation funciton 
prelim_eval_method_to_call = getattr(sc,inputDict['scoringPrelim'])
if inputDict['scoringPrelim'] == "SimpleEval":
    prelimEvalHandle = lambda x: prelim_eval_method_to_call(x,modelFitting.GetFirstThreshold())
    modelFitting.SetFEvalPreliminary(prelimEvalHandle)


# In[ ]:

# Create function handles 
# Create scoring function handle
scoring_method_to_call = getattr(sc,inputDict['scoringFnc'])
if inputDict['scoringFnc'] == 'FncSimpleScore':
    thisFittingData = modelFitting.GetFittingData()
    if inputDict['scoringOption']:
        scoringOption = inputDict['scoringOption']
    else:
        scoringOption = 3
    if 'scoringFncParams' in inputDict.keys():
        scoringFncHandle = lambda x: scoring_method_to_call(thisFittingData.GetData(),x,inputDict['scoringFncParams'],scoringOption)
    else:
        scoringFncHandle = lambda x: scoring_method_to_call(thisFittingData.GetData(),x,option=scoringOption)
    modelFitting.SetScoreFunc(scoringFncHandle)


# In[ ]:

# Create prior sampling function handle
priorsampling_method_to_call = getattr(ps,inputDict['priorSampleFnc'])
if inputDict['priorSampleFnc'] == 'SimpleSampleUniform':
    thisParamNum = len(modelFitting.GetParamList())
    priorSamplingFncHandle = lambda: priorsampling_method_to_call(thisParamNum,modelFitting.GetOrder()) 
    modelFitting.SetPriorSampleFunc(priorSamplingFncHandle)


# In[ ]:

# create weighted posterior sampling function handle
posteriorsampling_method_to_call = getattr(ps,inputDict['posteriorSampleFnc'])
if inputDict['posteriorSampleFnc'] == 'MultiVariateNormalCovariateSample':
    thisParamNum = len(inputDict['paramList'])
    thisCovscaling = inputDict['covscaling'] 
    posteriorSamplingFncHandle = lambda x,y: posteriorsampling_method_to_call(thisParamNum,x,y,thisCovscaling,inputDict['order'])
    modelFitting.SetPosteriorSampleFunc(posteriorSamplingFncHandle)


# In[ ]:

# Create function handle to calculate weights of the first sampled population 
calculateFirstGenWeight_method_to_call = getattr(sw,inputDict['calculateFirstGenWeightsFnc'])
if inputDict['calculateFirstGenWeightsFnc'] == 'UniformWeight':
    calculateFirstGenWeightsFncHandle = lambda x: calculateFirstGenWeight_method_to_call(x)
    modelFitting.SetCalculateFirstGenWeight(calculateFirstGenWeightsFncHandle)


# In[ ]:

# Create the function handle to calculate weights of the posterior population 
calculatePosteriorGenWeights_method_to_call = getattr(sw,inputDict['calculatePosteriorGenWeightsFnc'])
if inputDict['calculatePosteriorGenWeightsFnc'] == 'CalculateWeightMultiVariateNormalDensity':
    thisParamNum = len(inputDict['paramList'])
    calculatePosteriorGenWeightsFncHandle = lambda x,y,z: calculatePosteriorGenWeights_method_to_call(thisParamNum,x,y,z)
    modelFitting.SetCalculateWeightFunc(calculatePosteriorGenWeightsFncHandle)


# In[ ]:

# Create the function handle to the objective function  
if 'ODESolveFunc' in inputDict.keys():
    ODESolveFunc_method_to_call = getattr(sc,inputDict['ODESolveFunc'])
else:
    ODESolveFunc_method_to_call = getattr(sc,'SimulateODERoadRunner')
ODESolveFuncHandle = lambda paramArray: ODESolveFunc_method_to_call(modelFitting.GetXmlFile(),modelFitting.GetObservableName(),modelFitting.GetVarList(),modelFitting.GetFittingData().GetData(),modelFitting.GetSimulationStart(),modelFitting.GetSimulationEnd(),modelFitting.GetSimulationSteps(),modelFitting.GetEquilibriumEnd(),modelFitting.GetEquilibriumSteps(),dict( (modelFitting.GetParamList()[i],paramArray[i]) for i in range(len(modelFitting.GetParamList())) ),modelFitting.GetPerturbParamDict(), dict((modelFitting.GetVarList()[i],modelFitting.GetVarInitVals()[i]) for i in range(len(modelFitting.GetVarList()))))
modelFitting.SetODESolveFunc(ODESolveFuncHandle)


# In[ ]:

# Make function handle of the check constraints function 
if  'varsIniBoundList' in inputDict.keys() or 'varsBoundList' in inputDict.keys() or 'varsEquilibriumBoundList' in inputDict.keys():
    constraintDict = {}
    if 'varsIniBoundList' in inputDict.keys() :
        constraintDict['varsIniBoundList'] = inputDict['varsIniBoundList']
    if 'varsBoundList' in inputDict.keys() :
        constraintDict['varsBoundList'] = inputDict['varsBoundList']
    if 'varsEquilibriumBoundList' in inputDict.keys() :
        constraintDict['varsEquilibriumBoundList'] = inputDict['varsEquilibriumBoundList']
    modelFitting.SetConstraintDict(constraintDict)
    CheckConstraintFncHandle =lambda x: sc.CheckConstraints(x,modelFitting.GetConstraintDict())
    modelFitting.SetCheckConstraintFunc(CheckConstraintFncHandle) 


# In[ ]:

modelFitting.SequentialMonteCarlo()


# In[ ]:

outputFileName = modelFitting.OutputFittingData()
outputFolderName = 'outputFolder'

# upload the output file to the service account
from oauth2client.service_account import ServiceAccountCredentials
from oauth2client.client import GoogleCredentials
import base64,httplib2
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
credentials = ServiceAccountCredentials.from_json_keyfile_name('client_secrets.json',scopes="https://www.googleapis.com/auth/drive")
credentials.authorize(httplib2.Http())
gauth = GoogleAuth()
gauth.credentials = credentials
drive = GoogleDrive(gauth)


# In[ ]:

# Upload files to the folder
file_list = drive.ListFile({'q':"'root' in parents and trashed=false"}).GetList()
filterList = filter(lambda ele:ele['title'] == outputFolderName,file_list)
thisFolder = filterList[0]
fid = thisFolder['id']
f = drive.CreateFile({"parents": [{"kind":"drive#fileLink","id":fid}],'title':outputFileName })
f.SetContentFile(outputFileName)
f.Upload()

