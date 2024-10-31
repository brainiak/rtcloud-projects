# RT-Cloud Projects
This repository contains example and template projects using the BrainIAK RT-Cloud software.

For installation and basic usage instructions, please see the main repository: https://github.com/brainiak/rt-cloud

---

# Included projects:

Please see the README within each project for a more detailed description and instructions. 

- *amygActivation*: Example project for providing feedback based on a univariate signal within a given region of interest (e.g., the amgydala), based on work from Kymberly Young (e.g., [Young et al., 2018, _PCN_](https://onlinelibrary.wiley.com/doi/full/10.1111/pcn.12665)). This demo contains a full RT-Cloud project and an example display in PsychToolbox (MATLAB). 
- *dicomBidsStream*: Toy example of how RT-Cloud can be used to stream DICOM images using our custom real-time BIDS implementation.
- *functionalConnectivity*: Example project for providing feedback based on functional connectivity between two regions, based on [Ramot et al., (2017) _eLife_](https://elifesciences.org/articles/28974). Includes a working example of the "puzzle" feedback display from the referenced paper in PsychoPy.
- *induction*: Example project for how to run an "induction" based design using RT-Cloud, including a PsychoPy display. Currently the feedback is based on functional connectivity, but could be adapted to any multivariate or univariate signal.
- *mindeye*: Example of a computationally intensive project in RT-Cloud, using a GPU for real-time image reconstruction.
- *openNeuroClient*: Example of how to stream a published data set from OpenNeuro through RT-Cloud.
- *sample*: A small example project designed to test that your installation of RT-Cloud is working correctly, a useful tutorial.
- *syntheticDataSample*: Another small example project, including an example of how to generate synthetic data for testing.
- *template*: A stripped down template project. *Looking to create your own Rt-Cloud project? Start here!*