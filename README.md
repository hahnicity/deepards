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

Now go to the deepards directory and make a results directory so that your model can store its
results for future analysis.

    cd deepards
    mkdir results

## Quickstart

Make sure you have the ardsdetection dataset in a known path. If the path is unknown to you then you can
always download it from our [data repository.]() Once you have the dataset you can run training. For now
we will just showcase a single model with a single network. For this model, we will feed each breath individually
to a CNN, and then output its processed features to an LSTM. The LSTM will then be able to make a determination
on whether the breath looks like it belongs to an ARDS patient or not. We will use some hyperparameters with the model,
including: batch size of 32, learning rate of .001 and we will run for 5 kfolds.

    cd deepards
    python train_ards_detector.py -dp /path/to/ards/dataset -n cnn_lstm --cuda -b 32 -lr .001 --kfolds 5 -dt padded_breath_by_breath

By default, after each epoch the model will run a testing epoch to evaluate training progress. If you don't
want this to happen in the future you can provide the `--no-test-after-epochs` to the training CLI.
You may have to wait for awhile for the model to finish. So feel free to do other work while it runs.

## Dataset Types
