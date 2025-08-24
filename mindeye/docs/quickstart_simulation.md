# Quickstart for Simulated Real-Time MindEye
Set up and run simulated real-time image reconstruction using MindEye on pre-collected data.

## Introduction
This quickstart guide will walk you through the setup to start producing real-time reconstructions using MindEye with a GPU. This notebook uses pre-collected data to reproduce the real-time-compatible preprocessing and analysis previously performed in a real-time scan. This isolates the MindEye functionality from RT-Cloud, which enables real-time data streaming from an fMRI scanner, and the fMRI scanner itself. We recommend starting here to understand the real-time preprocessing and analysis pipeline before trying to integrate everything with RT-Cloud, which requires a more involved setup.

This document assumes you have completed the setup instructions in the [README](../README.md).  

## Detailed Pipeline
This is a description of the algorithm that we use to perform real-time image reconstructions and retrievals. We perform real-time analysis on Princeton subject 005 session 06. Our training data consists of subject 005 sessions 01-03 (sessions 04-05 were used for other purposes).

For the simulated real-time code provided here, note that all of step 1 has been done for you and the fully processed data are provided. (RISHAB MAYBE YOU CAN MAKE THIS SOUND BETTER)

1. Prior to the real-time session:
    1. MindEye was pre-trained on multiple subjects from NSD 7T data
    2. We collected 3 sessions of 3T data from subject 005 (sessions 01-03). Each session involved viewing XX images (SHOULD WE INCLUDE INFO ABOUT # OF TRIALS, # OF IMAGES, # OF REPEATS?)
    3. Subject 005's data was preprocessed using fMRIPrep; all three sessions were preprocessed together resulting in all functional data in alignment with each other
        * We used the outputs in the subject's native T1 space for all analyses (space-T1w_bold)
    4. Each session's preprocessed data was input to GLMsingle (independent of the other session RISHAB CAN YOU CHECK THAT THIS IS CORRECT) to obtain single-trial response estimates (betas) for each image-viewing trial
    5. We estimated which voxels responded most reliably to visual images in each session, and then created a "union mask" with the most reliable voxels from each session: 
        * First, we took the NSDgeneral mask in MNI space provided with the NSD dataset and resampled to the subject's native T1w space
        * For each voxel within the subject's NSDgeneral mask, we computed reliability: the correlation of beta values across repeated image presentations (a "reliable voxel" should have consistent responses to the same image presented at different times)
        * Computed the across-session correlation of voxel reliabilities at varying reliability thresholds (i.e., we correlated session 01 and 03 voxel reliabilities and separately correlated session 02 and 03 voxel reliabilities) 
        * For sessions 01 and 02, we independently choose the reliability threshold that maximized the correlation with session 03, resulting in two masks of voxels
        * We took the union of the two masks from the previous step; this "union mask" included all voxels that will be used to fine-tune MindEye and to be analyzed in a new real-time session
    6. Apply the union mask to the betas from sessions 01-03 and fine-tune MindEye

The simulated real-time code will perform the following pipeline steps: 

2. In real-time, stream in the functional data TR-by-TR. 
    1. To be done once at the beginning of the session: Use FLIRT to register the first functional volume of the real-time session to a BOLD reference volume from session 01 in space-T1w
    2. At each TR, motion-correct the new functional volume to the first functional volume of the real-time session and then apply the previously calculated registration (NOTE TO RISHAB: DOES THIS HAPPEN FOR EACH TR OR ONLY TRs SELECTED FOR RECONSTRUCTION?)
    3. If the current TR corresponds to an image (after accounting for the hemodynamic response function (HRF) NOTE: HOW MANY TRs or SECONDS AFTER IMAGE ONSET IS THIS?), run a simple GLM (implemented with nilearn) to deconvolve the HRF and produce a single-trial beta and append this to a running list
    4. If the current TR's data should be reconstructed:
        * Z-score voxel-wise over all available betas (using a growing list of betas across all runs, so the mean and standard deviation become more stable as the session progresses)
        * Run a forward pass through MindEye to generate retrievals and reconstructions based on the z-scored beta
        * Plot MindEye's outputs

## How to run
To run with minimal setup using uv (no IDE required): `uv run --with jupyter jupyter lab`, which opens a localhost instance of Jupyter Lab using the uv environment we installed previously 
* Defaults to http://localhost:8898 which you can open from your web browser
* Otherwise, open the link that it outputs in your web browser, which might look something like this: `http://localhost:8888/lab?token=3a57676d6590bf560852b39fe091183c520c7563db59acea`
* Open the notebook and select Run All

Alternatively, if you prefer using an IDE like Visual Studio Code, you can just open the notebook and press "Run all". Make sure the uv environment is active.

You have succeeded when you see an output like this:     
![alt text](https://github.com/brainiak/rtcloud-projects/raw/main/mindeye/docs/sample_jupyter_output.png "Sample Jupyter Output")
