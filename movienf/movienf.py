"""
"""
# FIXME add doc stirng
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
import ants
import joblib
import pandas as pd
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
import re
import templatematch
import time

date_today = str(datetime.today()).split()[0].replace('-','')
MOCK=False

# TODO check that cfg variables are valid
# set useful paths
# ----------------------------------------------
# directory for storing dicoms/bids data (set by TMPDIR or TEMP environemnt variable)
tmpPath = tempfile.gettempdir() 
# get project directory
currPath = os.path.dirname(os.path.realpath(__file__)) #'.../rt-cloud/projects/project_name'
rootPath = os.path.dirname(os.path.dirname(currPath)) #'.../rt-cloud'
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



subject = cfg.subjectNum
session = cfg.subjectDay
run = cfg.runNum
if type(run) in [list, tuple, np.ndarray] and len(run)==1:
    run = run[0]  # if run is a list, just take the first element
project_dir = cfg.rootDirAnalysis
nf_run_dir = cfg.runDirAnalysis
disdaqs = cfg.disdaqs # num volumes at start of run to discard for MRI to reach steady state
hrf_delay = cfg.hrf_delay # assuming relevant brain activations emerge 2 TRs later (1 TR = 1 sec)
# load stations
feedback_delay = 10+10+2 # feedback wait time + feedback display time +continue display time
stations_path = os.path.join(project_dir, 'clf','stations.csv')
stations_df = pd.read_csv(stations_path)
# merge all stations within a scene into one station
stations_df['station_index'] = stations_df['scene']
models_dir = os.path.join(project_dir, 'clf','scene_models')
# get nested list of TRs for each station
stations = []
station_models = []
feedback_trs = []
restart_nf_trs = []
feedback_delay = cfg.feedbackTimeTotal
for i, _station in enumerate(stations_df.station_index.unique()):
    _delay = i*feedback_delay + disdaqs
    model_fname = 'scene_model_scene-{}.joblib'.format(float(_station))
    model = joblib.load(os.path.join(models_dir, model_fname))
    _stations_df = stations_df[stations_df.station_index==_station]
    station_trs=[]
    for row_i, row in _stations_df.iterrows():
        TRs = list(range(row['start']+_delay, row['end']+1+_delay))
        station_trs+= TRs
    
    # get TR corresponding to end of scene
    feedback_start_tr = int(_stations_df.scene_end_time.max()) + _delay+ 1 
    feedback_end_tr  =feedback_start_tr + feedback_delay
    _feedback_trs = list(range(feedback_start_tr, feedback_end_tr+1))

    # if feedback tr is in station_trs, remove it from feedback trs
    # _feedback_trs = [tr for tr in _feedback_trs if tr not in station_trs]
    restart_nf_tr = _feedback_trs[-1] + 1 # TR to restart neurofeedback
    
    # print('feedback TRs: {}'.format(_feedback_trs))
    feedback_trs.append(_feedback_trs)
    restart_nf_trs.append(restart_nf_tr)
    
    stations.append(station_trs)
    station_models.append(model)
    # print('station_index: {}, template shape:{}, station length:{}'.format(_station, (model.templates[0].shape, model.templates[1].shape), len(station_trs)))

    # print('station: {}, station_trs: {}'.format(_station, station_trs))
    # print('station_index: {}, feedback_trs: {}, restart_nf_tr: {}'.format(_station, _feedback_trs, restart_nf_tr))

# for each feedback tr check that it is not in any station
for i, _feedback_trs in enumerate(feedback_trs):
    for j, _station_trs in enumerate(stations):
        if i==j:
            continue
        for _feedback_tr in _feedback_trs:
            if _feedback_tr in _station_trs:
                # remove _feedback_tr from _feedback_trs
                print('removing feedback TR {} from station {} TRs'.format(_feedback_tr, j))
                _feedback_trs.remove(_feedback_tr)
                # update restart_nf_trs
                restart_nf_trs[i] = _feedback_trs[-1] + 1

all_trs = np.arange(1,cfg.total_trs+1)
nonstation_trs = list(set(all_trs) - set(np.concatenate(stations)) - set(np.concatenate(feedback_trs)))


# check for necessary initialized files
ref_files_to_check = [cfg.refvol, cfg.refvol_brain, cfg.refvol_brain_mask, cfg.subject_nf_mask]
missing_files = [f for f in ref_files_to_check if not os.path.exists(f)]
if len(missing_files) > 0:
    print('\nThe following reference files are missing:')
    for f in missing_files:
        print(f)
    print('Make sure you go to the "Session" tab and run "Initialize session" before beginning the task')
    sys.exit(1)
else:
    print('\nAll reference files found, proceeding with real-time analysis...')


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

# FIXME this needs to be on skyport
# path for feedback scores
feedback_path = os.path.join(nf_run_dir, 'sub-{}_ses-{}_run-{}_feedback_scores.csv'.format(subject, session, run))
print('--------------------------------')
print('getting feedback output path')
print('run number for feedback scores: {}'.format(run))
print('feedback output path: {}'.format(feedback_path))

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


dicomPath = cfg.dicomDirAnalysis
refvol = os.path.join(cfg.referenceDirAnalysis, 'funcRef_scanner.nii.gz')
print('real time dicom directory:', dicomPath)
print('real time dicom directory exists?:', os.path.exists(dicomPath))

# get most recent run number in dicom directory
add_sbref = True # set to False for testing, set to True for actual task
pattern = '001_([0-9]+)_([0-9]+).dcm'
runNums = [int(re.search(pattern, f).group(1)) for f in os.listdir(dicomPath) if re.search(pattern, f)]
# account for sbref by adding 1
if runNums:
    curRun = max(runNums)+1 
    if add_sbref:
        curRun += 1
elif add_sbref:
    curRun = 2
else:
    curRun = 1

print('current run number: {}'.format(curRun))


# FIXME adjust this to match our incoming dicom naming convention
# print('dicom directory: {}'.format(dicomPath))
dicomScanNamePattern = stringPartialFormat(cfg.dicomNamePattern, 'RUN', curRun)
print('dicom scan name pattern: {}'.format(dicomScanNamePattern))
streamId = bidsInterface.initDicomBidsStream(dicomPath,dicomScanNamePattern,
                                            cfg.minExpectedDicomSize, 
                                            anonymize=True,
                                            **{'subject':cfg.subjectNum,
                                            'run':curRun,
                                            'task':cfg.taskName})



# prep BIDS-Run, which will store each BIDS-Incremental in the current run
currentBidsRun = BidsRun()

# reset the first x-axis plot location for Data Plot
point_idx=-1 

# first TR to start analyzing (account for disdaqs and hrf delay)
first_TR = disdaqs+hrf_delay+1

# initialize feedback scores for each station
feedback_scores = np.array([np.nan for _ in stations])

# checkpoint to make sure computers are synced 
#----------------------------------------------
# save file called analysis_ready.txt to communicate with display computer
analysis_ready_path = os.path.join(nf_run_dir, 'analysis_ready.txt')
with open(analysis_ready_path, 'w') as f:
    f.write('analysis ready')

# while loop to check for display_ready.txt
print('\n************************')
print('Waiting for display computer to be ready')
print('************************')
while not os.path.exists(os.path.join(nf_run_dir, 'display_ready.txt')):
    # sleep
    time.sleep(0.1)
    pass
print('\nDisplay computer is ready, beginning real-time analysis...')
display_start_time =  time.time()



# skip TR's if we fall too far behind in real-time
lag_tolerance_nonstation = 4 # seconds
lag_tolerance_station=4 # seconds
TR_duration = 1
tr_durations = []
skipped_station_trs = []
skipped_nonstation_trs = []
max_motion_correction_time = 0
max_masking_time = 0
in_station = False
_mean = None
_m2 = None
# iterate TRs
for TR in np.arange(1,cfg.total_trs+1):
    # approximate display time (to check if we are falling behind)
    display_time = time.time() - display_start_time
    lag = display_time - TR*TR_duration

    print('----------------------------------------------------')
    print('\nbeginning processing TR {} of {} (lag behind display time: {:.2f} seconds)'.format(TR, cfg.total_trs, lag))
    if (lag > lag_tolerance_nonstation) and TR in np.array(nonstation_trs):
        print(f"Skipping TR {TR} due to lag tolerance exceeded: {display_time-TR:.2f} seconds")
        skipped_nonstation_trs.append(TR)
        continue

    # if we are behind by more than the duration of the feedback delay, then we need to skip this TR, even if it is a station TR

    _start_tr=time.time()
    # skip initial disdaq TRs
    if TR==1:#first_TR:
        print(f'getting BIDS Incremental for TR {TR} of {cfg.total_trs}...')
        bidsIncremental = bidsInterface.getIncremental(streamId,volIdx=TR,timeout=20)
        # currentBidsRun.appendIncremental(bidsIncremental)
        niftiObject = bidsIncremental.image
        image_found = True
        _end_time = time.time()
        print(f"Time to get BIDS Incremental for TR {TR}: {_end_time-_start_tr:.4f} seconds")
        continue
    elif TR< first_TR:
        print('skipping TR {} for disdaq'.format(TR))
        continue
    elif TR== first_TR:
        # wait until the last feedback TR arrives
        print(f'waiting for TR {TR} to arrive before restarting neurofeedback')
        _wait_tr = TR-1
        _wait_filename = f'001_{curRun:06}_{_wait_tr:06}.dcm'
        _wait_path = os.path.join(dicomPath, _wait_filename)
        _start_time = time.time()
        while not os.path.exists(_wait_path):
            # print(f'Waiting for TR {_wait_tr} to arrive...')
            time.sleep(0.1)
        _end_time = time.time()
        print(f"Time to wait for TR {_wait_tr}: {_end_time-_start_time:.4f} seconds")

    # skip TRs while feedback is being displayed (this helps catch up if we are behind)
    if TR in np.concatenate(feedback_trs):
        print('skipping TR {} for feedback'.format(TR))
        # skip TRs that are for feedback
        continue

    # TRs where we restart real-time processing after feedback display
    elif TR in np.array(restart_nf_trs):
        # wait until the last feedback TR arrives
        print(f'waiting for TR {TR} to arrive before restarting neurofeedback')
        _wait_tr = TR-1
        _wait_filename = f'001_{curRun:06}_{_wait_tr:06}.dcm'
        _wait_path = os.path.join(dicomPath, _wait_filename)
        _start_time = time.time()
        while not os.path.exists(_wait_path):
            # print(f'Waiting for TR {_wait_tr} to arrive...')
            time.sleep(0.1)
        _end_time = time.time()
        print(f"Time to wait for TR {_wait_tr}: {_end_time-_start_time:.4f} seconds")

    # try getting nifti object for current TR
    try:
        if lag>lag_tolerance_station and TR in np.concatenate(stations):
            print(f"Skipping station TR {TR} due to lag tolerance exceeded: {display_time-TR:.2f} seconds")
            image_found=False
            skipped_station_trs.append(TR)
        else:
            print(f'getting BIDS Incremental for TR {TR} of {cfg.total_trs}...')
            _start_tr= time.time()
            bidsIncremental = bidsInterface.getIncremental(streamId,volIdx=TR,timeout=5)
            # currentBidsRun.appendIncremental(bidsIncremental)
            niftiObject = bidsIncremental.image
            image_found = True
            _end_time = time.time()
            print(f"Time to get BIDS Incremental for TR {TR}: {_end_time-_start_tr:.4f} seconds")
    except:
        print(f'BIDS Incremental for TR {TR} not found')
        image_found = False
    
    # further processing steps
    if TR>= first_TR and image_found:
        start_time = time.time()
        
        # save Nifti to temporary location 
        nib.save(niftiObject, tmpPath+"/temp.nii")

        # Motion correct to this run's functional reference
        command = f"mcflirt -in {tmpPath+'/temp.nii'} -reffile {cfg.refvol} -out {tmpPath+'/temp_aligned'}"
        A = datetime.now().timestamp(); call(command,shell=True); B = datetime.now().timestamp()
        # print(f"Motion correction time: {B-A:.4f}, saved to {tmpPath+'/temp_aligned'}")

        smoothing=False
        masking=False
        if smoothing:
            # Spatial smoothing
            fwhm = 5 # 5mm full-width half-maximum smoothing kernel (dividing by 2.3548 converts from standard dev. to fwhm)
            command = f'fslmaths {tmpPath+"/temp_aligned"} -kernel gauss {fwhm/2.3548} -fmean {tmpPath+"/temp_aligned_smoothed"}'
            A = datetime.now().timestamp(); call(command,shell=True); B = datetime.now().timestamp()
            # print(f"Smooth time: {B-A:.4f}, saved to {tmpPath+'/temp_aligned_smoothed'}")
            if masking:
                # Masking voxels outside the brain 
                command = f'fslmaths {tmpPath+"/temp_aligned_smoothed"} -mas {tmpPath+"/brainmask_scanner"} {tmpPath+"/temp_aligned_smoothed_masked"}'
                A = datetime.now().timestamp(); call(command,shell=True); B = datetime.now().timestamp()
                # print(f"Masking time: {B-A:.4f}")
                _img = tmpPath+"/temp_aligned_smoothed_masked"
            else:
                _img = tmpPath+"/temp_aligned_smoothed"

        elif masking:
            # Masking voxels outside the brain 
            # print('mask path:', cfg.refvol_brain_mask)
            command = f'fslmaths {tmpPath+"/temp_aligned"} -mas {cfg.refvol_brain_mask} {tmpPath+"/temp_aligned_smoothed_masked"}'
            A = datetime.now().timestamp(); call(command,shell=True); B = datetime.now().timestamp()
            # print(f"Masking time: {B-A:.4f}")
            _img = tmpPath+"/temp_aligned_masked"
        else:
            # No smoothing or masking
            _img = tmpPath+"/temp_aligned"

        # streaming zscore
        output = _img+"_zscored"
        img_path = _img
        zscore_online=True
        if zscore_online:

            if TR >= first_TR and _mean is None:
                # load _img as numpy array
                _img = nib.load(_img+'.nii.gz').get_fdata()
                _mean = np.zeros_like(_img)
                _m2 = np.ones_like(_img)

            elif TR> first_TR:
                _start_zscore = time.time()
                n = TR-first_TR+1
                _img = nib.load(_img+'.nii.gz').get_fdata()
                _mean_new = _mean + (_img - _mean) / n
                _diff_old = _img - _mean
                _diff_new = _img - _mean_new
                _product = _diff_old * _diff_new
                _m2 = _m2 + _product
                _std = np.sqrt(_m2 / (n-1))
                _mean = _mean_new
                _zscore = (_img - _mean) / _std
                _img = _zscore
                # print('TR {} zscoring took {} seconds'.format(TR, time.time()-_start_zscore))
            
            img_path = output
        else:
            # print('skipping z-score online')
            img_path = _img
        
        end_time = time.time()
        tr_durations.append(end_time-_start_tr)
        print(f"TR {TR} took {end_time-_start_tr:.4f} seconds")
        print('average TR duration: {}'.format(np.mean(tr_durations)))

    # NF transform and classifier for stations
    _start_station = time.time()
    for station_i, station in enumerate(stations):
        # detect station onset
        if TR == station[0]:
            transformed_signal =[]
        # process each TR in station
        if TR in station:
            if in_station==False:
                transformed_signal = []
                in_station=True
                print(f"Entering station {station_i} at TR {TR}")
            print(f"Processing station {station_i} TR {TR}")

            # linear transform in subject space
            nf_weight_mask_scanner_path = cfg.subject_nf_mask

            # check that the nf classifier mask was applied correctly
            if image_found:
                # print('appyling nf transform')
                _start = time.time()
                _img1 = _img
                _img2 = nib.load(nf_weight_mask_scanner_path).get_fdata()
                _img3 = _img1*_img2
                nf_value = _img3.sum()
                _end = time.time()
                # print(f"NF transform time: {_end-_start:.4f} seconds")
            else:
                nf_value = np.nan

            transformed_signal.append(nf_value)
            print(f"NF value: {nf_value} added to transformed signal: {transformed_signal}")
        
        # if last TR in station, compute classifier accuracy
        if TR==station[-1]:
            print(f'applying classifier to station {station_i}')
            # classifier input
            X = np.array(transformed_signal).reshape(1,-1)
            # load classifier
            clf = station_models[station_i]
            score = clf.nf_score(X)[0] # (corr2-corr1)/(abs(corr2)+abs(corr1)+1e-10)
            if score<0.:
                score=0
            elif score>1.:
                score=1
            elif np.isnan(score):
                score=0

            score = np.round(score*100)

            # add to feedback file
            feedback_scores[station_i] = score
            feedback_df = pd.DataFrame({'feedback_score': feedback_scores, 'station_idx': range(len(feedback_scores))}).dropna()

            # write score and transformed signal to file
            feedback_txt_path = os.path.join(nf_run_dir, 'sub-{}_ses-{}_run-{}_station-{}_feedback_score.txt'.format(subject, session,run, station_i))
            with open(feedback_txt_path, 'w') as f:
                f.write(str(score))

            # saved transformed signal to file
            transformed_txt_path = feedback_txt_path.replace('feedback_score', 'transformed_signal')
            np.savetxt(transformed_txt_path, transformed_signal, fmt='%.6f')

            # save skipped trs
            skipped_trs_path = os.path.join(nf_run_dir, 'sub-{}_ses-{}_run-{}_skipped_trs.txt'.format(subject, session, run))
            all_skipped_trs = np.array(skipped_station_trs + skipped_nonstation_trs)
            np.savetxt(skipped_trs_path, all_skipped_trs)


            # save feedback scores
            _end_station = time.time()
            print('******************\n\n')
            print(f"Feedback score for station {station_i}: {score}, took {_end_station-_start_station:.4f} seconds")
            print('\n\n******************')

            in_station=False
            transformed_signal = []

print(f"==END OF RUN {curRun}!==\n")
# archive.appendBidsRun(currentBidsRun)
bidsInterface.closeStream(streamId)

# copy reference files for this subject to their skyport data directory
#-----------------------------------------------------------------------
src_dir = cfg.referenceDirLocal
tgt_dir = cfg.referenceDirAnalysis
# tgt_dir = os.path.join(nf_run_dir, 'reference_scans')
if not os.path.exists(tgt_dir):
    os.makedirs(tgt_dir)
for f in os.listdir(src_dir):
    src_path = os.path.join(src_dir, f)
    tgt_path = os.path.join(tgt_dir, f)
    if os.path.exists(tgt_path):
        print(f"File {tgt_path} already exists, skipping copy")
    else:
        call(['cp', src_path, tgt_path])
        print(f"Copied {src_path} to {tgt_path}")

print("-----------------------------------------------------------------------\n"
"REAL-TIME EXPERIMENT COMPLETE!\n"
"-----------------------------------------------------------------------")
sys.exit(0)