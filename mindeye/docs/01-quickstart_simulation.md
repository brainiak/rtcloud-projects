# Quickstart for Simulated Real-Time MindEye
Set up and run simulated real-time image reconstruction using MindEye on pre-collected data.

## Introduction
This quickstart guide will walk you through the setup to start producing real-time reconstructions using MindEye with a GPU. 

This notebook uses pre-collected data to reproduce the real-time-compatible preprocessing and MindEye analysis steps that we had previously performed during one of our real-time scans at Princeton; importantly, the analysis steps in this notebook can be run without installing RT-Cloud (if you wanted to conduct actual real-time data streaming from an fMRI scanner, you would have to install RT-Cloud). 

We recommend starting here to understand the real-time preprocessing and analysis pipeline before trying to integrate everything with RT-Cloud, which requires a more involved setup.

## Prerequisites
This document assumes you have completed the setup instructions in the [README](../README.md). Refer to [00-pipeline.md](00-pipeline.md) for a detailed description of the MindEye pipeline and its components.

## How to run
To run with minimal setup using uv (no IDE required): `uv run --with jupyter jupyter lab`, which opens a localhost instance of Jupyter Lab using the uv environment we installed previously 
* Defaults to http://localhost:8898 which you can open from your web browser
* Otherwise, open the link that it outputs in your web browser, which might look something like this: `http://localhost:8888/lab?token=3a57676d6590bf560852b39fe091183c520c7563db59acea`
* Open the notebook and select Run All

Alternatively, if you prefer using an IDE like Visual Studio Code, you can just open the notebook and press "Run all". Make sure the uv environment is active.

You have succeeded when you see an output like this:     
![alt text](https://github.com/brainiak/rtcloud-projects/raw/main/mindeye/docs/sample_jupyter_output.png "Sample Jupyter Output")
