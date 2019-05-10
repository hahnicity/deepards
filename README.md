# deepards
Deep Learning for ARDS Detection with Ventilator Waveform Data

## Install
Install using anaconda. First install anaconda on your machine then setup a new environment

    conda create -n deepards python=2.7

We are using python 2.7 because our software stack only supports 2.7 at this time. We plan to
upgrade though. Next install things that really only get installed well with anaconda

    source activate deepards
    conda install pytorch torchvision matplotlib -c PyTorch

After that install everything else with pip, because for some reason numpy and pandas dependencies
get all messed up in anaconda. Also need to install our private python pkg, ucdpvanalysis.

    pip install -r requirements.txt
    pip install ucdpvanalysis-1.3.tar.gz
