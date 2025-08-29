# movienf
## Overview
This is an example of real-time fMRI neurofeedback during a movie stimulus.
A pre-training sample of two groups of participants watching the movie was used to learn a linear voxel transform that reduced the dimensionality of the BOLD data to a 1D time series and performed well on group classification with a correlation-based classifier. 

For delivering real-time feedback, the movie was divided into stations, where each station corresponded to segments with good classifier performance in the pre-training sample and a template signal with at least 10 TRs of monotonic increase/decrease. 
A separate classifier was pre-trained for scoring each station, where scores are computed by correlating the real-time 1D signal with the average templates for each pre-training group.

During movie scenes with stations to be analyzed, a green border appears around the movie to indicate that it is being scored.  To display feedback with minimal disruption to the movie narrative, the movie is paused at cuts between major scenes for the feedback from the previous scene to be displayed

## Real-time protocol
### Reference scan
At the beginning of each visit, a 30 second resting scan is collected to be used as a reference for motion correction and transforming pre-trained voxel weights from MNI to subject functional space. For the purposes of the demo, the reference volume and weights are provided in subject functional space in the referenceDir
### Real-time analysis
    * real-time dicoms are detected and bidsified with rt-cloud
    * motion correction to the reference volume
    * Z-score, based on a running mean and standard deviation
    * If a real-time TR is within a station, the pre-trained linear voxel transform is applied to yield a scalar, which is appended to the 1D time series for the current station
    * At the end of the station, the resulting 1D time series is passed to a pre-trained classifier for that station and a feedback score is computed based on correlation with the pre-trained target group template
    * At the end of the scene, the movie is paused to display the feedback scores
## Dataset
* dicomDir
    * 1229 dicoms with TR=1s
* referenceDir 
    * reference nifti for motion correction
    * Pre-trained transform weights in subject space nifti
* clf
    * station_models
        * .joblib files containing pre-trained templatematch classifier objects for each station
        * stations.csv
            * Onset and offset TRs and movie frames for each station/scene

## Run instructions
The project assumes there is a task machine for displaying the PsychoPy task and a separate analysis machine for running real-time analysis of fMRI data.  The task and analysis machines communicate by reading and writing feedback.txt files on a shared network drive

1. On the task machine, open a terminal, navigate to the project directory, and run the psychopy task
    * python movienf_task.py
    * subject = test
    * session=1
    * run=1
2. On analysis machine, open a terminal, navigate to the rt-cloud directory, and run_movienf.py
    * python run_movienf.py
        * Should be run from the directory that contains the rt-cloud install
        * Sets up config file, creates data directory for current subject-session-run and calls data-analyzer
    * subject=test
    * session=1
    * run=1
    * Once rt-cloud initializes, you will see a link for  localhost:8888 in the terminal, open this link in a browser
    * login with username=test,  password=test
    * Click Run

## Requirements
* rt-cloud installed locally
* Task machine (PsychoPy)
    * python 2.7 
    * psychopy 1.84.2
* Analysis machine
    * Python 3.7.4
    * rt-cloud
