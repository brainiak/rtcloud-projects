# Quickstart for MindEye with RT-Cloud
Stream pre-collected data with RT-Cloud to perform preprocessing and analysis in real-time.

## Introduction


## Prerequisites
You must have completed the setup instructions in the [README](../README.md). You should also be able to run MindEye; see [`quickstart_simulation.md`](quickstart_simulation.md).

This assumes you are familiar with the components of RT-Cloud. For example, you should know what the "data analyser" refers to. You should be able to run the sample project and the template project; see [documentation](https://github.com/brainiak/rt-cloud/tree/master?tab=readme-ov-file#realtime-fmri-cloud-framework).

Note that MindEye requires a GPU to run, so the data analyser component of RT-Cloud must be hosted on a GPU-enabled computer. 

## Setup
1. On a GPU-enabled computer, go to the directory where you want to place the rt-cloud folder with this project inside it, and then download [apptainer](https://apptainer.org/). Next, pull the latest rtcloud docker image into an apptainer file: 
    ```
    apptainer pull docker://brainiak/rtcloud:latest
    ```
    Alternatively, you could get the .sif file, which the above command creates, from the hugging face data set described in set up step (6) below and place the rtcloud_latest.sif file in the directory instead of doing this apptainer pull command.
3) On your local laptop/computer which you are going to display PsychoPy on, do the rt-cloud [local installation](https://github.com/brainiak/rt-cloud/tree/master?tab=readme-ov-file#local-installation) consisting of getting the rtcloud anaconda environment set up and cloning the rt-cloud repository on your local laptop/computer.
4) Download the rt-cloud repository into your GPU-enabled computer like you did locally in (3).
5) Download this /mindeye/ projects folder from the rt-cloud-projects repository and place it in the projects folder of the rt-cloud repository both locally on the PsychoPy display computer and on the GPU-enabled data analyzer computer.
6) Change the absolute_path variable in the rt-cloud/projects/mindeye/psychopy_example/rtcloud_psychopy.py file on your PsychoPy display local computer to the absolute path location of your rt-cloud folder you set up locally in (3).
7) Download the hugging face dataset [here](https://huggingface.co/datasets/rkempner/rt-cloud-mindeye) into your GPU-enabled computer. In the mindeye.py script in this /mindeye/ folder, set the path of data_and_model_storage_path to be where you place this hugging face dataset. 
8) From this /mindeye/ folder, get the bidsRun.py file and replace the bidsRun.py file within rt-cloud/rtCommon on your GPU-enabled computer with this new version of the bidsRun.py file set up particularly for this project.
9) On the GPU-enabled computer, create a conda environment by downloading the rt_mindEye2.yml file from this /mindeye/ folder then run: ```conda env create -f rt_mindEye2.yml```
10) Copy the file called bashrc_mindeye.py from this /mindeye/ folder to rt-cloud/ in the GPU-enabled computer.
11) Copy the BidsDir from the hugging face dataset into rt-cloud/projects/mindeye/ in the GPU-enabled computer.

**How to run**
1) On the GPU-enabled computer,
   - Go to the directory with the rtcloud_latest.sif file created in set up step (2).
   - Enter the apptainer (the --nv flag connects to the gpu)
     ```
     apptainer exec --nv ~/rtcloud_latest.sif bash
     ```
   - cd into the rt-cloud directory
   - Run the bash set up file you created in setup
     ```
     source bashrc_mindeye
     ```
     which will set up fsl and the anaconda environment within the apptainer.
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
When you run the project server via the webinterface and run PsychoPy which is waiting for input via the analysis listener, you will get images like below where we can see the actual ground truth image shown to participants compared to the top 5 retrievals. The MindEye2 model's most likely candidate image is directly adjacent to the grouth truth image. The model is retrieving the most likely candidate images from the pool of images given in the 63 stimuli trials in the run.
