"""-----------------------------------------------------------------------------
--- TEMPLATE PROJECT ---
The purpose of this script is to show an example RT-Cloud project and to
act as a template for you to wrap RT-Cloud into your own project. This
template loosely follows the pipeline implemented in Mennen et al. (2021) 
(https://doi.org/10.1016/j.bpsc.2020.10.006). 

This template project contains two runs of fMRI scanning consisting of 
242 functional volumes each (1 TR = 2 seconds). These volumes are 
contained in the template/DICOMs folder, and RT-Cloud will simulate the 
processing of these volumes as if this were a real real-time scan. 

--- PREPPING FOR A REALTIME SESSION ---
We recommend first scanning participants in a prelimary session
prior to the real-time scanning session if you want to use predefined
anatomical templates and/or brain regions of interest (ROIs). 
For this template project, we have preprocessed such a preliminary scan 
using fMRIPrep (https://fmriprep.org), with outputs stored in the 
template/fmriprep folder. This is also a good time to ensure that your
participant behaves well in the scanner and produces high-quality data,
as high-quality data is even more essential for real-time neurofeedback.

--- IMAGING SEQUENCE ---
Scanning was acquired with a 3T Siemens Prisma MRI scanner with sequences
matched to those in deBettencourt et al. (2015)
(https://www.nature.com/articles/nn.3940.pdf) as closely as possible. 

--- STUDY DESIGN ---
Each of the two runs included in this project contains 8 blocks. 
The first 4 blocks (stable blocks) showed overlaid face/scene stimuli with 
constant opacity and served as training for the face-versus-scene classifier. 
For the last 4 blocks (neurofeedback blocks), the attended category was 
scenes and the distractor category was negative faces. For neurofeedback 
blocks, the opacity changed depending on the relative degree of neural 
representation of scenes versus faces indicated by the classifier. At the
start of each block, participants were given a cue that indicated the
block type (face or scene) and go category. For instance, the cue 
“indoor scenes” indicated that participants should press for indoor scenes 
(90% go trials) and refrain from pressing when seeing outdoor scenes 
(10% no-go trials). Participants continuously ignored the overlaid irrelevant 
category stimuli (e.g., faces). The participant instructions (e.g., press 
button for female faces) showed up on the 1st TR of each block. 

--- REAL-TIME PROCESSING PIPELINE ---
Due to hemodynamic lag, we will use the associated brain activations from 
TR X+2 for a stimulus presented on TR X. That is, the brain activations
elicited from the scanner 4 seconds after the initial presentation of the
stimulus. 

PRIOR TO STABLE BLOCKS: 10 discarded TRs (disdaqs)

STABLE BLOCK PIPELINE (first 4 blocks):
TRs 1: Task instruction. Use this TR as the functional reference volume for 
    the rest of the scan. Motion correct the fMRIPrep brain mask to the
    functional reference volume and mask out the non-brain voxels.
TRs 2-26: 
    1. Transform DICOM to Nifti 
    1. Motion correct to the functional reference
    2. Spatial smoothing using 5mm Gaussian kernel (FWHM)
    3. Use brain mask (EPI space) to mask out non-brain voxels
TRs 27-29: Inter-block interval (IBI) and, if this is the final stable
    block, train the multivoxel pattern classifier. 

NEUROFEEDBACK BLOCK PIPELINE (last 4 blocks):
TRs 1: Task instruction.
TRs 2-26:
    1. Transform DICOM to Nifti 
    1. Motion correct to the functional reference
    2. Spatial smoothing using 5mm Gaussian kernel (FWHM)
    3. Z-score using mean and sd from current run's stable blocks
    4. Use brain mask (EPI space) to mask out non-brain voxels
    5. Input voxels to the trained classifier to obtain model prediction 
    for participant's attention towards either scenes or faces, save this
    output as a json file to be read by the presentation software. An increase
    in attention to (task-irrelevant) negative faces will trigger an increase
    in face visibility (i.e., punishing participants by making the task harder),
    as gauged by a moving window over the previous 3 TR classifier outputs. 
TRs 27-29: Inter-block interval (IBI)

-----------------------------------------------------------------------------"""

"""-----------------------------------------------------------------------------
The below portion of code simply imports modules and sets up path directories.
-----------------------------------------------------------------------------"""
# Importing modules and setting up path directories
import os
import sys
import warnings
import argparse
from subprocess import call
import tempfile
from pathlib import Path
from datetime import datetime
from copy import deepcopy
import numpy as np
import nibabel as nib
from sklearn.linear_model import LogisticRegression
import pdb # use pdb.set_trace() for debugging
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)

tmpPath = tempfile.gettempdir() # specify directory for temporary files
currPath = os.path.dirname(os.path.realpath(__file__)) #'.../rt-cloud/projects/project_name'
rootPath = os.path.dirname(os.path.dirname(currPath)) #'.../rt-cloud'
dicomPath = currPath+'/dicomDir' #'.../rt-cloud/projects/project_name/dicomDir/'
print("Location of subject's dicoms: \n" + dicomPath + "\n")
outPath = rootPath+'/outDir' #'.../rt-cloud/outDir/'

# add the path for the root directory to your python path
sys.path.append(rootPath)

from rtCommon.utils import loadConfigFile, stringPartialFormat
from rtCommon.clientInterface import ClientInterface
from rtCommon.imageHandling import readRetryDicomFromDataInterface, convertDicomImgToNifti, saveAsNiftiImage
from rtCommon.bidsArchive import BidsArchive
from rtCommon.bidsRun import BidsRun

"""-----------------------------------------------------------------------------
We will now initialize some important variables which will be used
for later parts of the code. 

When starting a new project, you will need
to define some variables beforehand in the .toml file located in the 
"/conf" folder of this project. For now, the current variables already 
defined in "conf/template.toml" will work for this example. 

Note: if you changed the name of this project from "template" to something
else, you will need to rename "template.py" and "template.toml" to match
the new project name.
-----------------------------------------------------------------------------"""
# obtain the full path for the configuration toml file
# if the toml variables have been changed in the web interface 
# then use those altered variables instead (note: does not overwrite the toml)
defaultConfig = os.path.join(currPath, f'conf/{Path(__file__).stem}.toml')
argParser = argparse.ArgumentParser()
argParser.add_argument('--config', '-c', default=defaultConfig, type=str,
                        help='experiment config file (.json or .toml)')
args = argParser.parse_args(None)
cfg = loadConfigFile(args.config)

print(f"\n----Starting project: {cfg.title}----\n")

# Prep starting run and scan number 
# These are 1-indexed because that's how the DICOM file names are written
# Note: cfg variables altered in web interface will be strings!
startRun = int(cfg.runNum[0])
startBlock = int(cfg.blockNum)
numRuns = int(cfg.numRuns)
numBlocks = int(cfg.numBlocks)
num_TRs_per_run = int(cfg.num_total_TRs)
num_TRs_per_block = int(cfg.num_TRs_per_block)
disdaqs = 10 # num volumes at start of run to discard for MRI to reach steady state
hrf_delay = 2 # assuming relevant brain activations emerge 2 TRs later (1 TR = 2 sec)

# Load experimental design for participant. Below code loads a text file that
# contains a list of what stimulus type was instructed to be attended to 
# for every every TR (0=none, 1=scene, 2=face).
run1_design = np.loadtxt(currPath+'/study_design/run1.txt')
run2_design = np.loadtxt(currPath+'/study_design/run2.txt')
run_designs = np.concatenate([[run1_design, run2_design]])
print("run_designs.shape",run_designs.shape)

"""-----------------------------------------------------------------------------
The below section initiates the clientInterface that enables communication 
between the three RTCloud components, which may be running on different 
machines.
-----------------------------------------------------------------------------"""
# Initialize the remote procedure call (RPC) for the data_analyser
# (aka projectInferface). This will give us a dataInterface for retrieving 
# files, a subjectInterface for giving feedback, a webInterface
# for updating what is displayed on the experimenter's webpage,
# and enable BIDS functionality
clientInterfaces = ClientInterface()
dataInterface = clientInterfaces.dataInterface
subjInterface = clientInterfaces.subjInterface
webInterface  = clientInterfaces.webInterface
bidsInterface = clientInterfaces.bidsInterface
archive = BidsArchive(tmpPath+'/bidsDataset')

"""-----------------------------------------------------------------------------
Locate your pre-existing reference scans. You don't need to 
collect a prior session of scanning before RT-fMRI, but it can be useful for
things like determining which voxels belong to certain brain parcellations or 
regions of interest. In this example, we will show how to transform a predefined 
mask (here a whole-brain mask from fMRIPrep's functional reference space) into 
the current run's native/EPI space for masking voxels in real-time.
-----------------------------------------------------------------------------"""
# fMRIPrep's T1w_to_scanner transform
T1w_to_scanner = currPath+'/fmriprep/func/sub-01_ses-001_task-func_run-1_from-T1w_to-scanner_mode-image_xfm.txt'

# brain mask (ROI)
brainmask_T1w = currPath+'/fmriprep/anat/sub-01_ses-001_desc-brain_mask.nii.gz'

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

# # VNC Viewer: if on cloud server, use VNC to show GUIs in web interface (cannot use "--test" mode)
# call("DISPLAY=:1 xeyes",shell=True) # can replace xeyes with any GUI (like fsleyes)

for curRun in range(startRun,numRuns+1):
    # prep stream of DICOMS -> BIDS 
    # "anonymize" removes participant specific fields from each DICOM header.
    if cfg.dsAccessionNumber=='None': 
        dicomScanNamePattern = stringPartialFormat(cfg.dicomNamePattern, 'RUN', curRun)
        streamId = bidsInterface.initDicomBidsStream(dicomPath,dicomScanNamePattern,
                                                    cfg.minExpectedDicomSize, 
                                                    anonymize=True,
                                                    **{'subject':cfg.subjectNum,
                                                    'run':curRun,
                                                    'task':cfg.taskName})
    else:
        # For OpenNeuro replay, initialize a BIDS stream using the dataset's Accession Number
        streamId = bidsInterface.initOpenNeuroStream(cfg.dsAccessionNumber,
                                                        **{'subject':cfg.subjectNum,
                                                        'run':f"0{curRun}",
                                                        'task':cfg.taskName})

    # prep BIDS-Run, which will store each BIDS-Incremental in the current run
    currentBidsRun = BidsRun()

    # reset the first x-axis plot location for Data Plot
    point_idx=-1 

    # Skip analysis of first disdaqs volumes, add 2 TRs to account for hrf delay
    # TR will be our run-specific TR counter
    for TR in range(1,disdaqs+hrf_delay+1): 
        print(f'--- Run {curRun} | TR {TR} ---')
        # Note that these variables are 1-indexed (starting at one, not zero)!
        bidsIncremental = bidsInterface.getIncremental(streamId,volIdx=TR,
                                        demoStep=cfg.demoStep)
        currentBidsRun.appendIncremental(bidsIncremental)
        
    for curBlock in range(startBlock,numBlocks+1):
        for blockTR in range(1,num_TRs_per_block+1):
            # set TR based on the current block (since block # could change in web interface)
            TR = disdaqs+hrf_delay+((curBlock-1)*num_TRs_per_block)+blockTR
            print(f'--- Run {curRun} | Block {curBlock} | Block TR {blockTR} | TR {TR} ---')

            # ensure we dont try to analyze the final two TRs of the session, because we
            # are accounting for an hrf delay of 2 TRs, analyzing the last 2 TRs would error
            if TR <= num_TRs_per_run:
                bidsIncremental = bidsInterface.getIncremental(streamId,volIdx=TR,
                                                demoStep=cfg.demoStep)
                currentBidsRun.appendIncremental(bidsIncremental)
                niftiObject = bidsIncremental.image
                
                # get the true class for this TR (0=N/A, 1=scene, 2=face)
                # -1 to account for 1-indexing a Python array, another -2 due to hrf delay
                actual_class = int(run_designs[curRun-1,TR-1-hrf_delay]) 

                if curRun==1 and curBlock==1 and TR==13: 
                    # this is our first non-disdaq functional volume for the scanning session
                    # we will use it as a reference scan for all subsequent functional volumes this session

                    # save Nifti to temporary location 
                    nib.save(niftiObject, tmpPath+"/funcRef_scanner.nii")
                    print(f"Temp. nifti location: {tmpPath+'/funcRef_scanner.nii'}")

                    # transform precollected brain mask from T1w space to native/EPI/BOLD space using this TR as reference
                    command = f"antsApplyTransforms --input {brainmask_T1w} \
                    --interpolation NearestNeighbor \
                    --output {tmpPath+'/brainmask_scanner.nii'} \
                    --reference-image {tmpPath+'/funcRef_scanner.nii'} \
                    --transform {T1w_to_scanner}"
                    A = datetime.now().timestamp(); call(command,shell=True); B = datetime.now().timestamp()
                    print(f"Brain mask transform time: {B-A:.4f}, saved to {tmpPath+'/brainmask_scanner.nii'}")
                
                # save Nifti to temporary location 
                nib.save(niftiObject, tmpPath+"/temp.nii")
                print(f"Temp. nifti location: {tmpPath+'/temp.nii'}")

                # Motion correct to this run's functional reference
                command = f"mcflirt -in {tmpPath+'/temp.nii'} -reffile {tmpPath+'/funcRef_scanner.nii'} -out {tmpPath+'/temp_aligned'}"
                A = datetime.now().timestamp(); call(command,shell=True); B = datetime.now().timestamp()
                print(f"Motion correction time: {B-A:.4f}, saved to {tmpPath+'/temp_aligned'}")

                # Spatial smoothing
                fwhm = 5 # 5mm full-width half-maximum smoothing kernel (dividing by 2.3548 converts from standard dev. to fwhm)
                command = f'fslmaths {tmpPath+"/temp_aligned"} -kernel gauss {fwhm/2.3548} -fmean {tmpPath+"/temp_aligned_smoothed"}'
                A = datetime.now().timestamp(); call(command,shell=True); B = datetime.now().timestamp()
                print(f"Smooth time: {B-A:.4f}, saved to {tmpPath+'/temp_aligned_smoothed'}")

                # Masking voxels outside the brain 
                command = f'fslmaths {tmpPath+"/temp_aligned_smoothed"} -mas {tmpPath+"/brainmask_scanner"} {tmpPath+"/temp_aligned_smoothed_masked"}'
                A = datetime.now().timestamp(); call(command,shell=True); B = datetime.now().timestamp()
                print(f"Masking time: {B-A:.4f}")

                # Load nifti data as img variable
                img = nib.load(tmpPath+'/temp_aligned_smoothed_masked.nii.gz').get_fdata().flatten()
                
                # If stable block, save activations in preparation for model fitting before neurofeedback blocks
                if curBlock<=4:
                    try: # add train_activations from current TR to saved matrix
                        train_activations = np.vstack([train_activations, img])
                        train_classes = np.hstack([train_classes, actual_class])
                    except: # if train_activations not yet defined (or corrupted), then define the variable
                        train_activations = img
                        train_classes = actual_class
                
                # if neurofeedback block and TR can be used to predict stimulus (1=scene, 2=face)
                if curBlock>4 and actual_class!=0: 
                    #img = nib.load(tmpPath+'/temptemp.nii.gz').get_fdata().flatten()
                    # z-score the input TR based on the mean/std of the training activations
                    zscored_img = ((img-train_mean) / train_std)
                    point_idx+=1
                    # predict class using trained classifiers
                    scene_class_prob = scene_classifier.predict_proba(zscored_img[None,:]).flatten()[1]
                    face_class_prob = face_classifier.predict_proba(zscored_img[None,:]).flatten()[1]
                    scene_minus_face = scene_class_prob - face_class_prob
                    """
                    We will use webInterface.plotDataPoint() to send the result to the RTCloud web interface 
                    to be plotted in the Data Plots tab. Each run will have its own data plot. 
                    Make sure that your .toml file is configured to work with the plots you are trying to make.
                    IMPORTANT: the inputs MUST be python integers/floats, not numpy!
                    """
                    webInterface.plotDataPoint(curRun, point_idx, scene_minus_face)
                    # Send the model outputs back to the computer running analysis_listener. 
                    subjInterface.setResultDict(name=f'run{curRun}_TR{TR}',
                                                values={'values': str(scene_minus_face)})
                else: 
                    # To demonstrate analyis_listener functionality without having to wait for neurofeedback 
                    # blocks to begin, output to the computer running analysis_listener the avg. voxel
                    # activation for each TR until then.
                    subjInterface.setResultDict(name=f'run{curRun}_TR{TR}',
                                                values={'values': str(np.round(np.mean(img),2))})

            # Check if end of block
            if blockTR == num_TRs_per_block:
                print(f"==END OF BLOCK {curBlock}!==\n")
                # if end of stable blocks, train scene-classifier and face-classifier
                if curBlock==4: 
                    A = datetime.now().timestamp()
                    # z-score the training activations
                    train_mean = np.mean(train_activations)
                    train_std = np.std(train_activations)
                    zscored_activations = (train_activations-train_mean) / train_std
                    scene_classes = deepcopy(train_classes)
                    # define classes as correct (1) or incorrect (2) for scene and face model
                    scene_classes[scene_classes==2]=0
                    face_classes = deepcopy(train_classes)
                    face_classes[face_classes==1]=0
                    face_classes[face_classes==2]=1
                    # fit classifiers according to respective class labels
                    # for quick demonstration purposes im using a strict max_iter cutoff
                    scene_classifier = LogisticRegression(penalty='l2', max_iter=100, random_state=0)
                    face_classifier = LogisticRegression(penalty='l2', max_iter=100, random_state=0)
                    scene_classifier.fit(zscored_activations, scene_classes)
                    face_classifier.fit(zscored_activations, face_classes)
                    print(f"Fit classifier time: {datetime.now().timestamp()-A:.4f}")

    print(f"==END OF RUN {curRun}!==\n")
    archive.appendBidsRun(currentBidsRun)
    bidsInterface.closeStream(streamId)

print("-----------------------------------------------------------------------\n"
"REAL-TIME EXPERIMENT COMPLETE!\n"
"-----------------------------------------------------------------------")
sys.exit(0)

"""-----------------------------------------------------------------------------
You are now ready to conduct your own RT-Cloud enabled project!

Additional note: you can create an "initialize.py" and/or "finalize.py" file, which
are scripts that can run by clicking the respective buttons on the RT-Cloud web 
browser (under the Session tab). This can be helpful when needing to run specific 
code at the beginning or end of your experiment.
-----------------------------------------------------------------------------"""