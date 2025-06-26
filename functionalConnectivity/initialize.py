#standard imports and utilities
import os
import sys
import argparse
from subprocess import call
from pathlib import Path
from datetime import datetime, date

#print diagnositc information
print('************************************************')
cwd = os.getcwd()
print(f"rt-cloud folder: {cwd}")
sys.path.append(cwd)
from rtCommon.utils import loadConfigFile

projectDir = os.path.dirname(os.path.realpath(__file__))
print(f'project folder: {projectDir}')

# obtain the full path for the configuration toml file
# if the toml variables have been changed in the web interface 
# then use those altered variables instead (note: does not overwrite the toml)
defaultConfig = os.path.join(cwd, f'conf/{Path(__file__).stem}.toml')
argParser = argparse.ArgumentParser()
argParser.add_argument('--config', '-c', default=defaultConfig, type=str,
                        help='experiment config file (.json or .toml)')
args = argParser.parse_args(None)
cfg = loadConfigFile(args.config)

#collect and format the subject number and session
sub = cfg.subjectNum
subj = f'sub-RP{sub:03d}' #RP is the study abbreviation using a bids format, can be changed

ses = cfg.subjectDay
sess = f'ses-{ses}'

#find the dicom folder, which needs the datestring in the format of YYYYMMDD
dicomRoot = cfg.dicomRoot
if dicomRoot == 'None':
    dicomRoot = f'{projectDir}/dicomDir'

dateString = cfg.dateString
if dateString == 'None':
    today = date.today()
    dateString = today.strftime('%Y%m%d')
dicomPath = f'{dicomRoot}/{dateString}.sub_RP{sub:03d}.sub_RP{sub:03d}'
dicomPath_tmp = f'{dicomPath}/tmp'
print(f'dicom folder: {dicomPath}')

#this is the output from the processing, also where the files that are made in this script are put 
# outDir = f'/Data1/realtime_outputs/ah7700/rt-press/sub-{subj}'
outRoot = cfg.outRoot
if outRoot == 'None':
    outRoot = f'{projectDir}/outDir'

outDir = f'{outRoot}/{subj}/{sess}' #path for this subject/session

#make the realtime path
os.makedirs(f"{outDir}",exist_ok=True)
print(f'output folder: {outDir}')
print('************************************************')

'''also making run folders here'''
for run in range(1,int(cfg.numRuns)+1):
    os.makedirs(f'{outDir}/run-{run}',exist_ok=True)

#copy the ref dicoms to a tmp folder
#this is just for the demo to work smoothly, as normally you wouldnt have all the dicoms when running this script
os.system(f'mkdir -p {dicomPath_tmp}')
os.system(f'cp {dicomPath}/001_000001_***.dcm {dicomPath_tmp}/')

# convert dicoms to nii.gz files
#change dicomPath_tmp to just dicomPath if you want to convert all the dicoms
command = f'dcm2niix -z y -f {subj}_%p -o {outDir}/ {dicomPath_tmp}/'
A = datetime.now().timestamp(); call(command,shell=True); B = datetime.now().timestamp()
print(f"dcm2niix Time: {B-A:.4f}\n")

#remove the tmp folder since we just created to get dcm2niix to just convert the one run we need for this script
os.system(f'rm -r {dicomPath_tmp}') #this is just for the demo to work smoothly

#these are files that we are going to create in the process of registering subjects
#refvol to standard space, and then registering masks to subject space
standard = f'{projectDir}/standard/MNI152NLin2009cAsym_T1_1mm.nii.gz'
standard_brain = f'{projectDir}/standard/MNI152NLin2009cAsym_T1_1mm_brain.nii.gz'

refseries = f'{outDir}/{subj}_{sess}_task-refseries_bold.nii.gz'
refvol = f'{outDir}/{subj}_{sess}_refvol.nii.gz'
refvol_brain = f'{outDir}/{subj}_{sess}_refvol_brain.nii.gz'
refvol_brain_mask = f'{outDir}/{subj}_{sess}_refvol_brain_mask.nii.gz'

std2anat = f'{outDir}/std_to_anat.mat'
std2ref = f'{outDir}/std_to_refvol.mat'
ref2std = f'{outDir}/refvol_to_std.mat'

#take the mean of the refseries
command = f'fslmaths {refseries} -Tmean {refvol}'
A = datetime.now().timestamp(); call(command,shell=True); B = datetime.now().timestamp()
print(f"Refvol creation time: {B-A:.4f}\n")

# skull strip to create a brain mask
command = f'bet {refvol} {refvol_brain} -m -R'
A = datetime.now().timestamp(); call(command,shell=True); B = datetime.now().timestamp()
print(f"Skull Strip Time: {B-A:.4f}\n")

#directly register the refvol to the standard, this time we can use the brain extracted images for cleaner results
command = f'flirt -in {refvol_brain} -ref {standard_brain} -omat {ref2std}'
A = datetime.now().timestamp(); call(command,shell=True); B = datetime.now().timestamp()
print(f"FLIRT 1 Time: {B-A:.4f}\n")

#invert it
command = f'convert_xfm -omat {std2ref} -inverse {ref2std}'
A = datetime.now().timestamp(); call(command,shell=True); B = datetime.now().timestamp()
print(f"FLIRT 1.5 Time: {B-A:.4f}\n")

#convert the masks
for mask in ['HC','MFG']:
    in_mask = f'{projectDir}/standard/{mask}_1mm.nii.gz'
    out_mask = f'{outDir}/{subj}_{sess}_{mask}_mask.nii.gz'

    command = f'flirt -in {in_mask} -out {out_mask} -ref {refvol_brain} -applyxfm -init {std2ref} -interp nearestneighbour'
    A = datetime.now().timestamp(); call(command,shell=True); B = datetime.now().timestamp()
    print(f"FLIRT {mask} mask pt1 Time: {B-A:.4f}\n")

    #mask them with the brain mask just to be careful
    command = f'fslmaths {out_mask} -mas {refvol_brain_mask} {out_mask}'
    A = datetime.now().timestamp(); call(command,shell=True); B = datetime.now().timestamp()
    print(f"FLIRT {mask} mask pt2 Time: {B-A:.4f}\n")

print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
print('finished init.py, close the loop!')
print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')