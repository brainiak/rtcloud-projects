# Quickstart for Simulated Real-Time MindEye
Set up and run simulated real-time image reconstruction using MindEye on pre-collected data.

## Introduction
This quickstart guide will walk you through the setup to start producing real-time reconstructions using MindEye with a GPU. This notebook uses pre-collected data to reproduce the real-time-compatible preprocessing and analysis previously performed in a real-time scan. This isolates the MindEye functionality from RT-Cloud, which enables real-time data streaming from an fMRI scanner.

This assumes you have completed the setup instructions in the [README](../README.md). We recommend starting here to understand the preprocessing and analysis pipeline before proceeding to integrate MindEye with RT-Cloud, which requires a more involved setup. 

## Detailed Pipeline
This is a description of the algorithm that we use to perform real-time image reconstructions and retrievals. We perform real-time analysis on Princeton subject 005 session 06. Our training data consists of subject 005 sessions 01-03 (sessions 04-05 were used for other purposes).

1. Prior to the real-time session:
    1. MindEye is pre-trained on multiple subjects from NSD 7T data
    2. We collect 3 sessions of 3T data from the target subject (subject 005 sessions 01-03)
    3. This new data is preprocessed using fMRIPrep and single-trial response estimates (betas) are acquired using GLMsingle
        * The data remains in the subject's native T1 space for all analyses
    4. We predict the voxels that will be task-responsive in the real-time session by generating a "union mask" with the following procedure: 
        * First, apply the NSDgeneral mask, a liberal visual cortex mask from NSD
        * For each voxel within the NSDgeneral mask, compute reliability: the correlation of beta values across repeated image presentations (a "reliable voxel" should have consistent responses to the same image presented at different times)
        * Compute the across-session correlation of voxel reliabilities at varying reliability thresholds (comparing session 01 to 03 and session 02 to 03)
        * For session 01 and 02, independently choose the reliability threshold that maximizes the correlation with session 03, resulting in two masks of voxels
        * Going forward, we take the union of the two previously calculated masks to define all voxels that will be analyzed in real-time
    5. Apply the union mask to the betas from sessions 01-03 and fine-tune MindEye

2. In real-time, stream in the functional data TR-by-TR. 
    1. Once at the beginning of the session, use FLIRT to register the first functional volume of the real-time session to a BOLD reference volume from session 01 in native T1 space
    2. At each TR, motion-correct the new functional volume to the first functional volume of the real-time session and then apply the previously calculated registration
    3. If the current TR corresponds to an image (after accounting for the hemodynamic response function (HRF)), run a simple GLM (implemented with nilearn) to deconvolve the HRF and produce a single-trial beta and append this to a running list
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
