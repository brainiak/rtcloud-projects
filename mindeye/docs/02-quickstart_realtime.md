# Quickstart for MindEye with RT-Cloud
Stream pre-collected data with RT-Cloud and perform preprocessing and analysis in real-time.

## Introduction
This quickstart guide will walk you through the process of integrating MindEye with RT-Cloud to stream data and work with the different components of RT-Cloud. Similar to [01-quickstart_simulation.md](01-quickstart_simulation.md), we will use pre-collected data to reproduce the real-time-compatible preprocessing and analysis previously performed in a real-time scan. 

## Prerequisites
You must have completed the setup instructions in the [README](../README.md). You should also be able to run MindEye; see [01-quickstart_simulation.md](01-quickstart_simulation.md). The instructions here will rely on file paths and other terms defined in these documents.

You should follow the RT-Cloud installation instructions and be able to run the sample project and the template project listed in the [RT-Cloud documentation](https://github.com/brainiak/rt-cloud/tree/master?tab=readme-ov-file#realtime-fmri-cloud-framework).

Note that MindEye requires a GPU to run, so the data analyser component of RT-Cloud must be hosted on a GPU-enabled computer. 

## Setup
### Data analyser
1. Check if [apptainer](https://apptainer.org/) is installed: `apptainer --version`. If not, see [documentation](https://apptainer.org/docs/user/main/quick_start.html).
2. Check that the RT-Cloud docker container `rtcloud_latest.sif` exists in `<path/to/rt_all_data>`. Alternatively, pull the latest rtcloud docker image using apptainer: `apptainer pull docker://brainiak/rtcloud:latest`. 
3. Download the rt-cloud repository into your GPU-enabled computer like you did locally in (3). TODO do we have to copy the rtcloud-projects/mindeye folder into rt-cloud/projects? isn't it easier without? if not, leave it as is and make sure the projects/mindeye directory is ignored by rt-cloud for potential future pushes from rt-cloud on local
4. Verify that `scripts/mindeye.py` exists on your GPU-enabled computer. Ensure that `config/config.json` has accurate paths for your system and is referenced accurately by the Python script.
5. From this /mindeye/ folder, get the bidsRun.py file and replace the bidsRun.py file within rt-cloud/rtCommon on your GPU-enabled computer with this new version of the bidsRun.py file set up particularly for this project. TODO again, figure out if we can get away without. if not, should add bidsRun.py to gitignore in rt-cloud
6. On the GPU-enabled computer, create a conda environment by downloading the rt_mindEye2.yml file from this /mindeye/ folder then run: ```conda env create -f rt_mindEye2.yml``` TODO replace with uv env from previous notebook and test that it works. TODO delete all refs to this conda env and ensure it isn't secretly being called anywhere in the repo, for example by the bashrc_mindeye
7. Copy the file called bashrc_mindeye.py from this /mindeye/ folder to rt-cloud/ in the GPU-enabled computer. TODO same as above

### PsychoPy (to display reconstructions)
This refers to the PsychoPy instance that watches for reconstruction outputs written by RT-Cloud and displays the reconstructions to the experimenters. Currently, the reconstructions are not displayed to the participant in the scanner, which is an area for improvement. 

Note: This means that in the current setup, there are two different instances of PsychoPy running in the real-time scan. First is the instance that displays reconstructions to the experimenter, described here. Second is the instance that displays images to the participant in the scanner. In this guide, we focus on setting up RT-Cloud using pre-collected data, so the latter is not needed. Instructions for running a real participant are located in [03-experiment_guide.md](03-experiment_guide.md), which we suggest proceeding to once you are comfortable with the contents of both [01-quickstart_simulation.md](01-quickstart_simulation.md) and this document.

1. On your local laptop/computer which you are going to display PsychoPy on, clone the rtcloud-projects repository: `cd <path/to/directory>` and `git clone https://github.com/brainiak/rtcloud-projects.git`
2. Download this /mindeye/ projects folder from the rt-cloud-projects repository and place it in the projects folder of the rt-cloud repository both locally on the PsychoPy display computer and on the GPU-enabled data analyzer computer.
3. Change the absolute_path variable in the rt-cloud/projects/mindeye/psychopy_example/rtcloud_psychopy.py file on your PsychoPy display local computer to the absolute path location of your rt-cloud folder you set up locally in (3). TODO change this to consolidate repos, rn it's using rt_mindeye/rt-cloud/outDir locally (?) which is pretty useless. probably should make the output dir in 3t derivatives so it can be saved to HF.

## Run it!
1) On the GPU-enabled computer,
   - Go to the directory with the rtcloud_latest.sif file created in set up step (2).
   - Enter the apptainer (the --nv flag connects to the gpu): `apptainer exec --nv <path/to/rtcloud_latest.sif> bash`
   - cd into the rt-cloud directory
   - Run the bash script we modified earlier: `source bashrc_mindeye`. This will set up fsl and the anaconda environment within the apptainer.
   - Then run the project server/data analyzer
     ```
     bash scripts/data_analyser.sh -p mindeye --port 8898 --subjectRemote --test
     ```
3) Run the following on your local computer to enable port-forwarding between the GPU-enabled computer and your local laptop
   used for the analysis listener, web-interface and PsychoPy task display:
   ```
   ssh -L 8892:hostname:8898 [username]@[server-name]
   ```
   where hostname is the hostname of the GPU-enabled compute node. This is how we allow the data analyzer on the GPU-enabled computer on a server to send information to your local computer via the analysis listener which then becomes input for PsychoPy.
4) In terminal on your PsychoPy display computer locally, cd into the rt-cloud directory from set up (3). Activate the local rtcloud anaconda environment. Run the analysis listener start up:
   ```
   WEB_IP=localhost
   bash scripts/analysis_listener.sh -s $WEB_IP:8892  --test
   ```

5) Open a new terminal locally on your PsychoPy display computer. cd into the rt-cloud directory from set up (3) and then go to the following path: ```projects/mindeye/psychopy_example/```. Activate the rtcloud environment and run ```python rtcloud_psychopy.py```. This will open up the PsychoPy program which starts waiting for incoming output from the data analyzer that reaches your computer through the analysis listener. Output coming in from the real-time analysis of the fMRI data can then modify the stimuli given to your participant through the PsychoPy program. 

6) On your local computer, open the webinterface by going to [http://localhost:8882/](http://localhost:8882/). Then enter "test" for both the username and password and press run. Files will populate the outDir in the local rt-cloud directory via the analysis listener and the PsychoPy program will start displaying images once it receives the first TR's information.

TODO include picture of connected rt-cloud browser

TODO include picture of PsychoPy waiting screen and reconstruction display
