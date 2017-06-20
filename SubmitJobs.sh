#!/bin/bash 
# PreStaging1 installs all the necessary software and libraries
while getopts f:n:i:d:s:x:t:z: option
do
	case "${option}"
	in
		# the number of instances
		f) INPUT=${OPTARG};;
		
		n) NUM=${OPTARG};;

		# simIDList		
		i) SIM=${OPTARG};;		
		
		# dataMat
		d) DATA=${OPTARG};;
		
		# scoreThresholdList
		s) SCORE=${OPTARG};;
		
		# XML
		x) XML=${OPTARG};;
			
		# timePoints 		
		t) TIME=${OPTARG};; 

		z) ZONE=${OPTARG};;
	esac
done
start=$SECONDS
# The script makes the new simID and pass them along to individual instances
pythonCreateUploadFolder="
# Refresh the folder for the files to be uploaded  
from oauth2client.service_account import ServiceAccountCredentials
from oauth2client.client import GoogleCredentials
import base64,httplib2
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
credentials = ServiceAccountCredentials.from_json_keyfile_name('client_secrets.json',scopes=\"https://www.googleapis.com/auth/drive\")
credentials.authorize(httplib2.Http())
gauth = GoogleAuth()
gauth.credentials = credentials
drive = GoogleDrive(gauth)
file_list = drive.ListFile({'q':\"'root' in parents and trashed=false\"}).GetList()
# Create a folder 
outputFolderName = 'outputFolder'
filterList = filter(lambda ele:ele['title'] == outputFolderName,file_list)
# if folder does not exist, create the folder; else delete the previous output folder and re-create it 
if len(filterList) > 0:
    thisFolder = filterList[0]
    thisFolder.Delete()
thisFolder = drive.CreateFile({'title':outputFolderName,\"mimeType\":\"application/vnd.google-apps.folder\"})
thisFolder.Upload()
"
MULTILINE="
sudo chown ${USER} SingleCellFitting;
chmod 755 SingleCellFitting;
cp index /home/${USER}/SingleCellFitting;
cp ${SIM} /home/${USER}/SingleCellFitting;
cp ${DATA} /home/${USER}/SingleCellFitting;
cp ${SCORE} /home/${USER}/SingleCellFitting;
cp ${XML} /home/${USER}/SingleCellFitting;
cp ${TIME} /home/${USER}/SingleCellFitting;
cp ${INPUT} /home/${USER}/SingleCellFitting;
cp client_secrets.json /home/${USER}/SingleCellFitting;
cd /home/${USER}/SingleCellFitting;
sudo /home/${USER}/anaconda2/bin/python ReadInputAndRunABCSMC.py ${INPUT};
"
echo "Authentication Needed"
eval `ssh-agent`
ssh-add ~/.ssh/google_compute_engine
# Make startup script
cat > StartupInstall.sh << EOF
#!/bin/bash
sudo apt-get update
cd /home/${USER}
wget https://repo.continuum.io/archive/Anaconda2-4.2.0-Linux-x86_64.sh
bash Anaconda2-4.2.0-Linux-x86_64.sh -b -p /home/${USER}/anaconda2
echo "export PATH=/home/${USER}/anaconda2/bin:\$PATH">>/home/${USER}/.bashrc
y | sudo /home/${USER}/anaconda2/bin/conda install -c sys-bio roadrunner
sudo apt-get -y install git 
git clone https://github.com/jasoncyao/SingleCellFitting.git
sudo /home/${USER}/anaconda2/bin/pip install PyDrive
sudo chown ${USER} SingleCellFitting
chmod 755 SingleCellFitting
cp index /home/${USER}/SingleCellFitting
cp ${SIM} /home/${USER}/SingleCellFitting
cp ${DATA} /home/${USER}/SingleCellFitting
cp ${SCORE} /home/${USER}/SingleCellFitting
cp ${XML} /home/${USER}/SingleCellFitting
cp ${TIME} /home/${USER}/SingleCellFitting
cp ${INPUT} /home/${USER}/SingleCellFitting
cp client_secrets.json /home/${USER}/SingleCellFitting
cd /home/${USER}/SingleCellFitting
sudo /home/${USER}/anaconda2/bin/python ReadInputAndRunABCSMC.py ${INPUT}
EOF
chmod +x StartupInstall.sh

# Create template
gcloud compute instance-templates create example-template \
--machine-type n1-standard-1 \
--image-family ubuntu-1404-lts \
--image-project ubuntu-os-cloud \
--boot-disk-size 15GB \
--preemptible
# Create instances
gcloud compute instance-groups managed create example-managed-instance-group \
--zone $ZONE \
--template example-template \
--size $NUM

gcloud compute instance-groups managed list-instances example-managed-instance-group --zone=$ZONE > instanceList
tail -n +2 < instanceList | cut -d ' ' -f1 > instanceNameList
echo "Please wait for instance initilization"
sleep 1m
# Refresh the creation of the upload folder 
python -c "$pythonCreateUploadFolder" 
echo "Transfer the necessary files" 
# Copy necessary files and client_secrets.json into each of the VM instances 
currInd=0
while read p
do 
	echo $currInd	
	# Save the current index 
	python -c "import sys;import pickle;index = int(sys.argv[1]);pickle.dump(index,open('index','wb'))" $currInd		
	# copy the simIDList, dataMat, scoreThresholdList, XML, timePoints, input file  
	gcloud compute copy-files StartupInstall.sh index $SIM $DATA $SCORE $XML $TIME $INPUT client_secrets.json $p:~ --zone $ZONE  		
	currInd=$(($currInd+1))
	echo ""
done < instanceNameList
# Monitor for the initialization of the instances
#python CheckInstanceInitialized.py

# After the initialization is done, run the optimization
while read p
do 
	ssh $p -o StrictHostKeyChecking=no "./StartupInstall.sh" 2>/dev/null &
done < instanceNameList
echo "Calcualte durationAfterSubmission"
durationAfterSubmission=$((SECONDS - start))
python CheckResultsDownload.py $SIM $INPUT $durationAfterSubmission
duration=$((SECONDS - start))
