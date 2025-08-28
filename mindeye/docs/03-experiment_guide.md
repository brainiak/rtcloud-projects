# Running the Real-time MindEye experiment
Collect data from a new participant, preprocess and fine-tune MindEye, and administer a real-time reconstruction scan. 

## Introduction
This document will provide an overview of the necessary components for running the real-time MindEye experiment. We will not include detailed instructions on how to run the MindEye analysis or integrate MindEye with RT-Cloud, which are respectively covered in [01-quickstart_simulation.md](01-quickstart_simulation.md) and [02-quickstart_realtime.md](02-quickstart_realtime.md). Rather, we will focus on the surrounding steps, which include preparing the stimuli to be shown to the participant in the scanner, data preprocessing and analysis, and considerations for the real-time scan.

Specific implementation-level details (such as code snippets) may be mentioned in this document or in the linked repositories, some of which reference the specific setup used at Princeton, such as computing clusters and MRI configurations. These are included for reference as usage examples, but institutional specifics should be changed accordingly. 

## Prerequisites
You must have completed the setup instructions in the [README](../README.md) and the quickstart guides: [01-quickstart_simulation.md](01-quickstart_simulation.md) and [02-quickstart_realtime.md](02-quickstart_realtime.md). We assume familiarity with the concepts from these documents including references to file paths. You should be familiar with the pipeline; see [00-pipeline.md](00-pipeline.md) for a refresher.

The data analyser component of RT-Cloud must be hosted on a GPU-enabled computer. You must have a way to stream files from your MRI scanner onto this computer. This has been tested at Princeton using a Siemens Prisma 3T scanner with a physical connection to a GPU workstation (located in the control room). However, many other configurations are possible, including the use of cloud-based analysis. 

## Preparing for a new participant
Refer to the [mindeye_task](https://github.com/PrincetonCompMemLab/mindeye_task) repository for information on task design and stimulus presentation. The repository provides scripts to copy existing conditions files or generate new image curricula for each participant. It also includes setup instructions for running the task in PsychoPy and standardized instructions to read to participants. 

## Preprocessing and fine-tuning MindEye
Refer to the [mindeye_preproc](https://github.com/PrincetonCompMemLab/mindeye_preproc/tree/main) repository for sample command-line snippets and scripts for preprocessing with fMRIPrep and GLMsingle on Princeton research clusters. This includes instructions on generating the subject-specific NSDgeneral mask and getting the betas to fine-tune MindEye.

Refer to the [mindeye_offline](https://github.com/PrincetonCompMemLab/mindeye_offline) repository for instructions on making the union mask from multiple sessions and fine-tuning MindEye based on that mask.

## Preparing for the real-time scan
We strongly recommend running a real-time test scan using a dummy such as an MRI phantom. Based on the connection between the MRI machine and your analysis computer, you may need to identify where newly streamed DICOM volumes are being sent, how to access them, and any scanner-specific naming schemes for these volumes. 

For example, at Princeton, the scanner interface has an "RT Start" option which mounts a drive from the GPU workstation on the MRI reconstruction computer. Subsequently, new DICOM volumes will appear in these directories during the scan (located at `/home/scontrol/` on Prisma and `/Data1/subjects/` on Skyra).



## Running the real-time scan

### 1. Before you start: Edit mindeye.py and set session variables
Before launching the data analyser, open `mindeye.py` and check/update the following variables for your session. Note that exact values (such as `dicomNamePattern`) might differ depending on your institution.

* `model_name`
* `num_voxels` (e.g., 8627 if using union mask from ses-01-02)
* `sub`, `session`
* `dicomNamePattern`
* `filename` (must match the participant's last name and subject ID as registered on the scanner console)
* `dicomDir` (update the date in the format `YYYYMMDD`, e.g., `20250729`)
* `demoStep` (should be 0 for no manual delay)


> **Tip:** If you need to edit `mindeye.py` after starting the data analyser, stop the server (in Terminal 1), make your changes, save, and re-launch the server.

### 2. Start the Data Analyser (on GPU-enabled computer)
1. Connect to VPN and SSH into the real-time computer:
	```bash
	ssh ri4541@pni-jom9144.princeton.edu
	cd /home/ri4541@pu.win.princeton.edu
	```
2. Start the container with GPU support:
	```
	apptainer exec -B /home/scontrol:/home/scontrol --nv ~/rtcloud-projects/mindeye/rt_all_data/rtcloud_latest.sif bash
	```
	* `-B /home/scontrol:/home/scontrol` binds the `/home/scontrol` directory from the host system (GPU workstation) into the same path inside the container. This allows the container to access the DICOM volumes.
3. Launch the data analyser server:
	```bash
	cd /home/ri4541@pu.win.princeton.edu/rt-cloud
	source bashrc_mindeye
	source rtcloud/bin/activate
	bash scripts/data_analyser.sh -p mindeye --port 8898 --subjectRemote --test
	```
4. **Success:** You should see `Listening on: http://localhost:8898`.
	* The server is now running and ready to accept connections.
	* Underlying code comes from `/home/ri4541@pu.win.princeton.edu/rt-cloud/projects/mindeye/mindeye.py`
> **Note:** Wait to click **Run** in the browser interface until the scout scan is finished and you see a new folder containing DICOMs in `/home/scontrol/`. For now, continue to the next step.

### 3. Set Up Port Forwarding (on your local computer)

1. Open a new terminal window.
2. Run the following command to forward local port 8892 to the remote data analyser on port 8898:
	```
	ssh -L 8892:localhost:8898 ri4541@pni-jom9144.princeton.edu
	```
	- This allows your local machine to access the data analyser server via `localhost:8892`.
3. There will be no output if successful.
	* Optional: test with `lsof -i:8898` to check the connection. If you see output, it means the port is in use. If not, the port is available.

### 4. Start the Analysis Listener (on your local computer)

1. Open a new terminal window.
2. Mount the shared volume:
	- In Finder, press `Cmd + K` and connect to: `smb://cup.pni.princeton.edu/norman`
3. Navigate to the analysis directory:
	```bash
	cd /Volumes/norman/rsiyer/rt_mindeye/rt-cloud
	```
4. Start the analysis listener:
	```bash
	WEB_IP=localhost
	bash scripts/analysis_listener.sh -s $WEB_IP:8892 --test
	```
5. **Success:** You should see `Connected to: ws://localhost:8892/wsSubject`.


### 5. Open the rt-cloud Interface (Browser)

1. In your web browser, go to: [http://localhost:8892](http://localhost:8892). 
2. Log in if prompted:
	- **Username:** test
	- **Password:** test
3. In the upper right, confirm all 3 components are connected:
	- browser: connected
	- dataConn: connected
	- subjConn: connected

> **Note:** The first two (browser and dataConn) should show "connected" after starting the server (i.e., after completing steps in terminal 1 for the data analyser and terminal 2 for port forwarding). The third (subjConn) will connect after you start the analysis listener.
4. For Run # and Scan #, you can leave the default value. These variables are not used.
5. Click **Run** to start the session.

### 6. Start PsychoPy Display (on your local computer)

1. Open a new terminal window.
2. Activate the rtcloud conda environment:
	```bash
	conda activate rtcloud
	```
3. Navigate to the directory containing the PsychoPy task:
	```bash
	cd /Volumes/norman/rsiyer/rtcloud-projects/mindeye/psychopy_example
	```
4. Make sure the output directory is empty:
	```
	/Volumes/norman/rsiyer/rt_mindeye/rt-cloud/outDir
	```

	> **Note:** At each TR, a `.json` file will be written to `outDir`. If the file contains "pass", PsychoPy will do nothing; otherwise, it will display the retrievals/reconstructions.

6. Start PsychoPy:
	```bash
	python rtcloud_psychopy.py
	```
	* This will open a PsychoPy window to display reconstructions.
	* The window should say "Waiting for scanner..." until a file is read.

### 7. Start the scan!
1. Run the scout scan. After it finishes, verify that `/home/scontrol/` contains a new folder with the scout DICOMs.
2. When ready, press **Run** on the browser RT-Cloud interface.
3. Monitor Terminal 1 for messages (e.g., `Waiting for 001_00005_...`).
4. Start the functional run on the scanner.
5. Monitor progress in Terminal 1 and the PsychoPy display. Good luck! 

## Troubleshooting and Tips

* **Editing `mindeye.py` during a session:**
	* Stop the data analyser server (Terminal 1).
	* Make your changes and save the file.
	* Restart the server as described above.

* **DICOM run mapping:**
	* Functional runs 1–4: DICOMs 5–8
	* T1w: DICOM 9
	* Runs 5–8: DICOMs 10–13
	* T2w: DICOM 14
	* Runs 9–11: DICOMs 15–17

* **Restarting a functional run:**
	* If you stop and restart a functional run, the DICOM number will increment by 1 each time.
	* For example, functional run 1 starts at `001_000005_xx.dcm`, but will become `001_000006_xx.dcm` if you restart once, `001_000007_xx.dcm` if you restart twice, etc.
	* To adjust for this, edit `mindeye.py`:
		* Change the line:
			```python
			dicomScanNamePattern = stringPartialFormat(dicomNamePattern, 'RUN', run_to_dicom[run_num])
			```
			to use `[run_num+1]` or the appropriate increment.
		* `run_to_dicom` is a dictionary mapping `run_num` to the corresponding DICOM number used in the pattern search.
