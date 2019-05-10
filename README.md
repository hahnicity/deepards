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

    pip install ucdpvanalysis-1.3.tar.gz
    pip install -e .

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

You may have noticed the `-dt` flag we specified above. This stands for `--dataset-type`. Different
dataset types correspond with different ways for modeling the ARDS detection problem. These different methods
all use relatively the same data, but in a variety of different ways. As of 2019-05-10 types are:

    * **padded_breath_by_breath** - Simplest dataset. Each breath extracted, and then padded if necessary to a certain sequence size (usually 224 for resnet). Sequential breaths are then clustered together into a sub-batch of a certain size
    * **stretched_breath_by_breath** - Instead of padding, breaths are stretched to the maximum extent of a certain sequence size via upsampling.
    * **spaced_padded_breath_by_breath** - Stretches the breath similar to above, but instead of upsampling, we just pad with 0's. Probably one of the weirder modeling techniques and didn't yield great results
    * **unpaded_sequences** - Instead of having 1 single breath fed into a CNN, we utilize sequences of multiple breaths until we cannot fit any more breaths in the sequence.
    * **padded_breath_by_breath_with_full_bm_target** - Utilizes the padded breath by breath method, but sets breath metadata as its target instead of an ARDS/no ARDS classification.
    * **padded_breath_by_breath_with_limited_bm_target** - Utilizes the padded breath by breath method, but sets a limited set of breath metadata as its target instead of an ARDS/no ARDS classification.
    * **padded_breath_by_breath_with_flow_time_features** - Utilizes the padded breath by breath method, and adds breath metadata so the model can use that for classification decisions as well.

I understand that this might be a bit to understand in writing so I have also added images to help visualize how data is being processed.

### Padded Breath By Breath
