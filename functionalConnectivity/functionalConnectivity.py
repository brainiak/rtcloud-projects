"""-----------------------------------------------------------------------------
The below portion of code simply imports modules and sets up path directories.
-----------------------------------------------------------------------------"""
# Importing modules and setting up path directories
import os
import sys
import warnings
import argparse
import json
import tempfile
import time
import nibabel as nib
import pandas as pd

from subprocess import call
from pathlib import Path
from datetime import datetime, date
from scipy.stats import zscore
from nilearn.image import get_data

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
print('************************************************')
cwd = os.getcwd()
print(f"rt-cloud folder: {cwd}")
sys.path.append(cwd)

from rtCommon.utils import loadConfigFile, stringPartialFormat
from rtCommon.clientInterface import ClientInterface
from rtCommon.imageHandling import readRetryDicomFromDataInterface, convertDicomImgToNifti, saveAsNiftiImage
from rtCommon.bidsArchive import BidsArchive
from rtCommon.bidsRun import BidsRun


'''project directory'''
projectDir = os.path.dirname(os.path.realpath(__file__)) #'.../rt-cloud/projects/project_name'
print(f"project folder: {projectDir}")

'''load the config file'''
defaultConfig = os.path.join(projectDir, f'conf/{Path(__file__).stem}.toml')
argParser = argparse.ArgumentParser()
argParser.add_argument('--config', '-c', default=defaultConfig, type=str,
                        help='experiment config file (.json or .toml)')
args = argParser.parse_args(None)
cfg = loadConfigFile(args.config)

'''collect and format the subject and session number'''
sub = cfg.subjectNum
subj = f'sub-RP{sub:03d}'

ses = cfg.subjectDay
sess = f'ses-{ses}'

'''find the dicom folder, which needs the datestring in the format of YYYYMMDD'''
dicomRoot = cfg.dicomRoot
if dicomRoot == 'None':
    dicomRoot = f'{projectDir}/dicomDir'

dateString = cfg.dateString
if dateString == "None":
    today = date.today()
    dateString = today.strftime('%Y%m%d')
dicomPath = f'{dicomRoot}/{dateString}.sub_RP{sub:03d}.sub_RP{sub:03d}'
print(f'dicom folder: {dicomPath}')

'''this is the output from the processing, where i can also put the files i make'''
# outDir = f'/Data1/realtime_outputs/ah7700/rt-press/sub-{subj}'
outRoot = cfg.outRoot
if outRoot == 'None':
    outRoot = f'{projectDir}/outDir'
outDir = f'{outRoot}/{subj}/{sess}'
#make the realtime path
if not os.path.exists(f"{outDir}"):
    os.makedirs(f"{outDir}")
print(f'output folder: {outDir}')

'''specify directory for temporary files'''
# tmpPath = tempfile.gettempdir()
tmpPath = f'{outRoot}/tmp'
if os.path.exists(tmpPath):
    os.system(f'rm -r {tmpPath}')
    os.makedirs(tmpPath)
else:
    os.makedirs(tmpPath)
print(f"temp folder: {tmpPath}")

print('************************************************')
print(f"\n----Starting project: {cfg.title}----\n")

# Prep starting run and scan number 
# These are 1-indexed because that's how the DICOM file names are written
# Note: cfg variables altered in web interface will be strings!
startRun = int(cfg.runNum[0])
numRuns = int(cfg.numRuns)
num_TRs_per_run = int(cfg.num_total_TRs)
num_runs_before_rt = int(cfg.num_runs_before_rt)
disdaqs = 10 # num volumes at start of run to discard for MRI to reach steady state
hrf_delay = 0 #currently, this project is designed for continuous feedback, so we are not shifting to account for HRF

# Initialize the remote procedure call (RPC) for the data_analyser
# (aka projectInferface). This will give us a dataInterface for retrieving 
# files, a subjectInterface for giving feedback, a webInterface
# for updating what is displayed on the experimenter's webpage,
# and enable BIDS functionality
clientInterfaces = ClientInterface(rpyc_timeout=999999)
webInterface  = clientInterfaces.webInterface
bidsInterface = clientInterfaces.bidsInterface
subjInterface = clientInterfaces.subjInterface
subjInterface.subjectRemote = True

#we could create a bids archive to save afterwards
# archive = BidsArchive(f'{outDir}/bidsDataset')

"""====================REAL-TIME ANALYSIS GOES BELOW====================
Use the below section to program whatever real-time analysis that you want  
performed on your scanning data. In this example, for each TR,
we transform the DICOM data into a Nifti file and then apply motion correction 
and spatial smoothing. We then mask voxels and save the activations 
for later training of the multivoxel classifier.
===================================================================="""
# clear existing web browser plots if there are any
try:
    webInterface.clearAllPlots()
except:
    pass

'''things we need to load in from offline runs'''
'''these are made in the initialize.py script'''
refvol = f'{outDir}/{subj}_{sess}_refvol.nii.gz'
refvol_brain_mask = f'{outDir}/{subj}_{sess}_refvol_brain_mask.nii.gz'

MFG_mask = get_data(f'{outDir}/{subj}_{sess}_MFG_mask.nii.gz').ravel().astype(bool)
HC_mask = get_data(f'{outDir}/{subj}_{sess}_HC_mask.nii.gz').ravel().astype(bool)

for curRun in range(startRun,numRuns+1):
    feedback_total = 0
    # prep stream of DICOMS -> BIDS 
    # "anonymize" removes participant specific fields from each DICOM header.
    dicomScanNamePattern = stringPartialFormat(cfg.dicomNamePattern, 'RUN', curRun+num_runs_before_rt)#plus 6 here to account for the shift in things from the auto-scout
    streamId = bidsInterface.initDicomBidsStream(dicomPath,dicomScanNamePattern,
                                                cfg.minExpectedDicomSize, 
                                                anonymize=False,
                                                **{'subject':cfg.subjectNum,
                                                'run':curRun,
                                                'task':cfg.taskName})


    # prep BIDS-Run, which will store each BIDS-Incremental in the current run
    currentBidsRun = BidsRun()

    # reset the first x-axis plot location for Data Plot
    point_idx=-1 
    
    '''point to the run specific output folder,
    and check to see if its empty and properly warn'''
    run_outDir = f'{outDir}/run-{curRun}'
    if not os.path.exists(run_outDir):
        os.makedirs(run_outDir)
    elif len([i for i in os.listdir(run_outDir)]) > 3:
        print('************************************************')
        print('LOOKS LIKE YOU ARE RERUNNING A RUN THAT ALREADY HAS DATA')
        print('WAITING 5 SECONDS AND THEN OVERWRITING EXISTING OUTPUT')
        print('************************************************')
        time.sleep(5)
        os.system(f'rm -r {run_outDir}')
        os.makedirs(run_outDir)

    # TR will be our run-specific TR counter
    for TR in range(1,num_TRs_per_run): #if we wanted to shift for HRF we would add that here
        print(f'--- Run {curRun} | TR {TR} ---')    
        
        #start the bids stream
        bidsIncremental = bidsInterface.getIncremental(streamId,volIdx=TR,
                                        timeout=999999,demoStep=cfg.demoStep)
        currentBidsRun.appendIncremental(bidsIncremental)

        #count the time between TRs for diagnostic purposes
        if TR == 1:
            previous_TR = datetime.now().timestamp()
        else:
            this_TR = datetime.now().timestamp()
            print(f'time between TRs = {this_TR - previous_TR:.4f}')
            previous_TR = this_TR

        if TR in range(1,disdaqs+hrf_delay):
            # Skip analysis of first disdaqs volumes, hrf_delay is currently 0

            feedback = False
            MFG_activity_now = 0
            HC_activity_now = 0

        else:
            #otherwise get the image
            niftiObject = bidsIncremental.image
        
            # save Nifti to temporary location
            nib.save(niftiObject, tmpPath+"/temp.nii")

            # Motion correct to this session's functional reference
            command = f"mcflirt -in {tmpPath+'/temp.nii'} -reffile {refvol} -out {tmpPath+'/temp_aligned'}"
            A = datetime.now().timestamp(); call(command,shell=True); B = datetime.now().timestamp()
            print(f"Motion correction time: {B-A:.4f}")

            # Spatial smoothing
            fwhm = 5 # 5mm full-width half-maximum smoothing kernel (dividing by 2.3548 converts from standard dev. to fwhm)
            command = f'fslmaths {tmpPath+"/temp_aligned"} -kernel gauss {fwhm/2.3548} -fmean {tmpPath+"/temp_aligned_smoothed"}'
            A = datetime.now().timestamp(); call(command,shell=True); B = datetime.now().timestamp()
            print(f"Smooth time: {B-A:.4f}")

            '''apply brain mask'''
            command = f'fslmaths {tmpPath+"/temp_aligned_smoothed"} -mas {refvol_brain_mask} {tmpPath+"/temp_aligned_smoothed_masked"}'
            A = datetime.now().timestamp(); call(command,shell=True); B = datetime.now().timestamp()
            print(f"Masking time: {B-A:.4f}")

            # Load nifti data as img variable
            img = get_data(tmpPath+'/temp_aligned_smoothed_masked.nii.gz').ravel()
            
            #zscore across voxels to remove global signal
            img = zscore(img)

            #take the activity in our masks
            MFG_activity_now = img[MFG_mask].mean()
            HC_activity_now = img[HC_mask].mean()

            try:
                #take the difference between last TR and now
                MFG_diff = MFG_activity_now - MFG_activity_before
                HC_diff = HC_activity_now - HC_activity_before

                #if MFG is upward trend, and HS is downward, then provide feedback
                feedback = (MFG_diff > 0) & (HC_diff < 0)

            except:
                #if the difference fails, print what TR it is (expected to fail on first TR after discard volumes)
                print(f'difference failed, and its TR {TR}; (expected to fail on first TR after initialization)')
                feedback = False

            #save activity values for next TR
            MFG_activity_before = MFG_activity_now
            HC_activity_before = HC_activity_now
        
        #move the plot along
        point_idx+=1
        #plot if feedback was given, 1 = Yes, 0 = No
        webInterface.plotDataPoint(curRun, point_idx, int(feedback))
        #count the total feedback given
        feedback_total += int(feedback)

        #write out feedback through the analysis listener
        subjInterface.setResultDict(name=f'{subj}/{sess}/run-{curRun}/TR{TR:03d}',
                            values={'feedback': str(feedback),
                                    'MFG_activity':str(MFG_activity_now),
                                    'HC_activity':str(HC_activity_now)})

    print(f"-----------------------------------------------------------------------\n"
    f"END OF RUN {curRun}! feedback total = {feedback_total} \n"
    "-----------------------------------------------------------------------")
    bidsInterface.closeStream(streamId)

print("-----------------------------------------------------------------------\n"
"REAL-TIME EXPERIMENT COMPLETE!\n"
"-----------------------------------------------------------------------")
sys.exit(0)
