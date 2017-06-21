# This script checks whether all the instances have been initialized
# termination time is defined in units of hour
import sys
from datetime import *
import time
import copy 
import pickle
import os 
import numpy as np
import fnmatch
import ast
t0 = datetime.now()
# take in the simIDList 
simIDListName = sys.argv[1]
inputFileName = sys.argv[2]
durationStr = sys.argv[3]
# parse the input file
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
# Get the maixmum run time limit in the 
maxRunTimeAlgo = inputDict['maxRunTime']
# Convert to seconds and add 5 more minutes to the run time as the upperbound for the waiting interval 
maxWaitingIntervalInMinutes = maxRunTimeAlgo*60 + 5
#maxWaitingIntervalInMinutes =  4
# This is the running time of the SubmitJobs.sh currently 
duration = int(durationStr)
simIDList = pickle.load(open(simIDListName,'rb'))
simIDList = np.array(simIDList)
print('Start monitoring for optimization results')
terminationTime = 10 
readyIndexList = []
readyContentList = []
readyFlag = False
# read in the instance names 
instaneNameList = []
with open("instanceNameList","r") as f:
    instanceCount = sum(1 for _ in f)
# Read in the content of the 
with open("instanceNameList","r") as f:
    content = f.readlines()
indArray = range(len(content))
content = [x.strip() for x in content]
totalResultsCount = 0
# pause time in between updates
pauseTimeInSeconds = 60 
# Build the original list of resutls output names 
resultsFileNameList = [None]*instanceCount
# All the list of instances that should return results
for i in range(len(resultsFileNameList)):
    currSimID = simIDList[i]
    resultsFileNameList[i] = 'instance'+str(currSimID) + '_fittingData.p'
# This variable counts the number of minutes that no results have been returned. If the count exceeds maxWaitingIntervalInMinutes then terminate the while loop
sameCount = 0 
while(readyFlag == False):   
    time.sleep(pauseTimeInSeconds)
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
    file_list = drive.ListFile({'q':"'root' in parents and trashed=false"}).GetList()
    # define output file folder
    outputFolderName = 'outputFolder'
    # find the id of the output folder
    file_list = drive.ListFile({'q':"'root' in parents and trashed=false"}).GetList()
    filterList = filter(lambda ele:ele['title'] == outputFolderName,file_list)
    # Get the file object of the output folder
    thisFolder = filterList[0]
    fid = thisFolder['id']
    # Get the list of results currently finished
    fileListInFolder = drive.ListFile({'q':"'%s' in parents and trashed = false" %fid}).GetList()
    # Get the list of the titles of finished results files
    fileTitleList = []
    for i in range(len(fileListInFolder)):
        thisFile = fileListInFolder[i]
        fileTitleList.append(thisFile['title'])
    # Get the overlap of the file titles     
    overlapTitleList = list(set(fileTitleList).intersection(resultsFileNameList))
    if len(overlapTitleList) > 0:    
        print('Length of overlap bigger than zero')
        sameCount = 0
        # Download the overlapping files
        for i in range(len(overlapTitleList)):
            thisTitle = overlapTitleList[i]
            filterList = filter(lambda ele:ele['title'] == thisTitle,fileListInFolder)
            thisFile = filterList[0]
            thisDownloadFile = drive.CreateFile({'id':thisFile['id']})
            thisFileTitle = thisFile['title']
            thisDownloadFile.GetContentFile(thisFile['title'])
            resultsFileNameList.remove(thisFileTitle)
        totalDuration = (datetime.now() - t0).total_seconds() + duration
    else:
        print('Add sameCount')
        sameCount = sameCount +1
        print('sameCount = ' + str(sameCount))
    resultsDownloadCount = instanceCount - len(resultsFileNameList)
    elapsedTime = (datetime.now() - t0).total_seconds()/60
    sys.stdout.write("\rInstances %s      out of %s     finished.      %0.1f  minutes elapsed." %(resultsDownloadCount,instanceCount,elapsedTime ) )
    sys.stdout.flush()
    if sameCount > maxWaitingIntervalInMinutes:
        readyFlag = True
        print('\r')
        print('Optimizations are prematurely returned because ' + str(len(resultsFileNameList)) + ' instances fail to proeprly initialize')    
    # check if all rppesults have been downloaded 
    if len(resultsFileNameList) == 0:
        readyFlag = True
        print('\r')
        print('All optimizationresults have been returned')    
# Write the output duration to file
totalDuration = (datetime.now() - t0).total_seconds() + duration
f = open('duration','w')
f.write(str(totalDuration))
f.close()