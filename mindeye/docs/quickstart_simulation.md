# Quickstart for Simulated Real-Time MindEye
Quickly set up and run simulated real-time image reconstruction using MindEye on pre-collected data.

## Introduction
This quickstart guide will walk you through the minimal setup to start producing real-time reconstructions using MindEye. This guide uses a standalone Jupyter notebook `rtcloud-projects/mindeye/scripts/mindeye.ipynb` that isolates the real-time MindEye code from the RT-Cloud framework. 

This requires basic familiarity with Python, Git, and the command line. Specific code snippets that you should run will be formatted like `this text`. Within code snippets, paths that might differ on your computer will be formatted like `<this>`.

This assumes you have completed the setup instructions in the [README](../README.md). You have successfully completed this when you are able to run the main analysis loop and it begins generating image reconstructions.

## Running the notebook in simulated real-time
At this point, everything should be ready to go!
1. Run the Jupyter notebook `rtcloud-projects/mindeye/scripts/mindeye.ipynb`
    * To run with minimal setup using uv: `uv run --with jupyter jupyter lab`, which opens a localhost instance of Jupyter Lab using the uv environment we installed previously 
        * Defaults to http://localhost:8898 which you can enter in your web browser
        * Otherwise, enter the link that it outputs
    * Select Run All
    * You have succeeded when you see an output like this: 
    
    ![alt text](https://github.com/brainiak/rtcloud-projects/raw/main/mindeye/docs/sample_jupyter_output.png "Sample Jupyter Output")

## Next steps

{% comment %}
Provide a quick recap of what has been accomplished in the quick start as a means of transitioning to next steps. Include 2-3 actionable next steps that the user take after completing the quickstart. Always link to conceptual content on the feature or product. You can also link off to other related information on docs.github.com or in GitHub Skills.
{% endcomment %}

Include citation for github style guide docs and quickstart template
