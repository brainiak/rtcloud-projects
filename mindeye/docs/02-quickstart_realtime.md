# Quickstart for MindEye with RT-Cloud
Stream pre-collected data with RT-Cloud and perform preprocessing and analysis in real-time.

## Introduction
This quickstart guide will walk you through the process of integrating MindEye with RT-Cloud to stream data and work with the different components of RT-Cloud. Similar to [01-quickstart_simulation.md](01-quickstart_simulation.md), we will use pre-collected data to reproduce the real-time-compatible preprocessing and analysis previously performed in a real-time scan. 

## Prerequisites
You must have completed the setup instructions in the [README](../README.md). You should also be able to run MindEye; see [01-quickstart_simulation.md](01-quickstart_simulation.md). The instructions here will rely on file paths and other terms defined in those documents.

You should follow the RT-Cloud installation instructions and be able to run the sample project and the template project listed in the [RT-Cloud documentation](https://github.com/brainiak/rt-cloud/tree/master?tab=readme-ov-file#realtime-fmri-cloud-framework).

Note that MindEye requires a GPU to run, so the data analyser component of RT-Cloud must be hosted on a GPU-enabled computer. 

## Setup
### Data analyser
These instructions should be performed on the GPU-enabled computer from which you will be hosting the data analyser (doing all of the real-time analysis). 

1. Check if [apptainer](https://apptainer.org/) is installed: `apptainer --version`. If not, see [instructions](https://apptainer.org/docs/user/main/quick_start.html).
2. Check that the RT-Cloud docker container `rtcloud_latest.sif` exists in `<path/to/rt_all_data>`. Alternatively, pull the latest rtcloud docker image using apptainer: `apptainer pull docker://brainiak/rtcloud:latest`.
3. Download the rt-cloud repository into your GPU-enabled computer: `git clone https://github.com/brainiak/rt-cloud`
4. Set up a symbolic link from `<path/to/rt-cloud/projects/mindeye>` to `<path/to/rtcloud-projects/mindeye>` on your GPU-enabled computer: 
   ```
   cd </path/to/rt-cloud/projects>
   ln -s </path/to/rtcloud-projects/mindeye> mindeye
   # Now rt-cloud/projects/mindeye is a link to rtcloud-projects/mindeye
   ```
   > **Important**: If you write data to `rt-cloud/projects/mindeye` you would really be making changes to the destination directory, `rtcloud-projects/mindeye`. Be careful when using relative paths and deleting files, since they will affect the destination. For an explainer on symbolic links, see [here](https://www.freecodecamp.org/news/symlink-tutorial-in-linux-how-to-create-and-remove-a-symbolic-link/).
5. Ensure that `</path/to/rtcloud-projects/mindeye/config/config.json>` contains accurate paths for your system.
6. Copy `</path/to/rtcloud-projects/mindeye/scripts/bidsRun.py>` and replace the bidsRun.py file located in `</path/to/rt-cloud/rtCommon/>`.
7. Update the file paths for FSL and your uv environment in `</path/to/rtcloud-projects/mindeye/config/bashrc_mindeye>` based on your system. To double check the paths:

   ```
   apptainer exec --nv <path/to/rtcloud_latest.sif> bash  # Open the container
   source </path/to/rtcloud-projects/mindeye/config/bashrc_mindeye>  # Run your bashrc_mindeye file
   ```
   > You should see something like "(mindeye) Apptainer>" at the command prompt if the environment was activated. Run `flirt -version` to verify that FSL commands are available.

### PsychoPy (to display reconstructions)
This refers to the PsychoPy instance that waits for RT-Cloud to write the reconstruction output json files, reads and decodes them, and then displays the reconstructions to the experimenters. Currently, the reconstructions are not displayed to the participant in the scanner, which is an area for improvement.

> Note: This means that in the current setup, there are two different instances of PsychoPy running in the real-time scan. First is the instance that displays reconstructions to the experimenter, described here. Second is the instance that displays images to the participant in the scanner. In this guide, we focus on setting up RT-Cloud using pre-collected data, so the latter is not needed. 

These instructions should be performed on the computer on which you would like to view the reconstructions. For example, this could be your laptop which is ssh'd into the GPU-enabled computer. This would allow you to monitor the data analyser and view the reconstructions on the same laptop.

1. Clone the rtcloud-projects repository: 
   ```
   cd </path/to/directory>  # Navigate to the directory where you want to clone the repo
   git clone https://github.com/brainiak/rtcloud-projects.git
   ```
2. Change the absolute_path variable in the rt-cloud/projects/mindeye/psychopy_example/rtcloud_psychopy.py file on your PsychoPy display local computer to the absolute path location of your rt-cloud folder you set up locally in (3). TODO change this to consolidate repos, rn it's using rt_mindeye/rt-cloud/outDir locally (?) which is pretty useless. probably should make the output dir in 3t derivatives so it can be saved to HF.

## Run it!
1) On the GPU-enabled computer,
   * Enter the apptainer: `apptainer exec --nv <path/to/rtcloud_latest.sif> bash`
      * The --nv flag stands for "NVIDIA" and allows the container to access the GPU.
   * Go to the rt-cloud directory: `cd </path/to/rt-cloud>`
   * Run the bash script we modified earlier: `source bashrc_mindeye`
      * This is necessary to set up the FSL path and will also activate 
   * Then run the project server/data analyzer
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
