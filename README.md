# deepards
Deep Learning for ARDS Detection with Ventilator Waveform Data

## Install
Install using anaconda. First install anaconda on your machine then setup a new environment

    conda create -n deepards python=3.7
    source activate deepards
    conda install pytorch torchvision matplotlib -c PyTorch

After that install everything else with pip, because for some reason numpy and pandas dependencies
get all messed up in anaconda. Also need to install our private python pkg, ucdpvanalysis.

    pip install ucdpvanalysis-1.5.tar.gz
    pip install -e .

Now go to the deepards directory and make a results directory so that your model can store its
results for future analysis.

    cd deepards
    mkdir results

## Quickstart

Make sure you have the ardsdetection dataset in a known path. If the path is unknown to you then you can
always download it from our [data repository.](https://ucdavis.app.box.com/folder/67693436232) Once you have the dataset you can run training. For now
we will just showcase a single model with a single network. For this model, we will feed each breath individually
to a CNN, and then output its processed features to an LSTM. The LSTM will then be able to make a determination
on whether the breath looks like it belongs to an ARDS patient or not. We will use some hyperparameters with the model,
including: batch size of 32, learning rate of .001 and we will run for 5 kfolds.

    cd deepards
    python train_ards_detector.py -dp /path/to/ards/dataset -n cnn_lstm --cuda -b 32 -lr .001 --kfolds 5 -dt padded_breath_by_breath

By default, after each epoch the model will run a testing epoch to evaluate training progress. If you don't
want this to happen in the future you can provide the `--no-test-after-epochs` to the training CLI.
You may have to wait for awhile for the model to finish. So feel free to do other work while it runs.

If you want to run an experiment to test a number of different parameters you should do so by passing the `-exp` flag to the `train_ards_detector.py` script.
This will allow the results reporting to understand that you wish to compare different runs in the same cluster of experiments to each other. Example

    python train_ards_detector.py -lr .001 --kfolds 5 -exp test_learning_rate_changes
    python train_ards_detector.py -lr .0001 --kfolds 5 -exp test_learning_rate_changes
    python train_ards_detector.py -lr .00001 --kfolds 5 -exp test_learning_rate_changes

## Dataset Types

You may have noticed the `-dt` flag we specified above. This stands for `--dataset-type`. Different
dataset types correspond with different ways for modeling the ARDS detection problem. These different methods
all use relatively the same data, but in a variety of different ways. As of 2019-05-10 types are:

* **padded_breath_by_breath** - Simplest dataset. Each breath extracted, and then padded if necessary to a certain sequence size (usually 224 for resnet). Sequential breaths are then clustered together into a sub-batch of a certain size
* **stretched_breath_by_breath** - Instead of padding, breaths are stretched to the maximum extent of a certain sequence size via upsampling.
* **spaced_padded_breath_by_breath** - Stretches the breath similar to above, but instead of upsampling, we just pad with 0's. Probably one of the weirder modeling techniques and didn't yield great results
* **unpadded_sequences** - Instead of having 1 single breath fed into a CNN, we utilize sequences of multiple breaths until we cannot fit any more breaths in the sequence.
* **padded_breath_by_breath_with_full_bm_target** - Utilizes the padded breath by breath method, but sets breath metadata as its target instead of an ARDS/no ARDS classification. This is used for pretraining a CNN and then later you can apply it to ARDS detection.
* **padded_breath_by_breath_with_limited_bm_target** - Utilizes the padded breath by breath method, but sets a limited set of breath metadata as its target instead of an ARDS/no ARDS classification. This is used for pretraining a CNN and then later you can apply it to ARDS detection.
* **padded_breath_by_breath_with_flow_time_features** - Utilizes the padded breath by breath method, and adds breath metadata so the model can use that for classification decisions as well.

I understand that this might be a bit to understand in writing so I have also added images to help visualize how data is being processed.

![](img/padded_breath_by_breath.png)

![](img/stretched_breath_by_breath.png)

Note that the spaced padding is zoomed so you can see what's happening.
![](img/spaced_padded_breath_by_breath.png)

![](img/unpadded_sequences.png)

### Best Performers
Currently its unclear if unpadded_sequences or padded_breath_by_breath performs best. I have been using padded_breath_by_breath more
consistently because it was just the first thing I coded and it was unclear if unpadded_sequences hurt performance
or not. More experiments will need to be done however to validate which performs best. A further possible advantage
of padded_breath_by_breath is that you can pretrain CNNs using breath metadata regressors and then apply it to ARDS detection.
I will discuss this in a later section.

## Visualizing Results
Now that you've run everything you will want to visualize your results in an informative way. You can do so using the `visualize_results.py` script. You can use this
a few different ways.

 1. Pass in the starting time of a run. This starting time is given to you via the CLI

        python visualize_results.py -st 1561106349

 2. Pass in the experiment name you wish to compare. This will allow you to compare different parameters in an experiment

        python visualize_results.py -exp test_learning_rate_changes

## Pretraining
It has been noted that training a neural network on a related task and then applying it to your dataset in question has
either improved results of the classifier or improved the speed at which a classifier converges. We can do this using the ARDS
dataset by training the neural network to determine specific metadata properties of a breath. These properties can be computed
computationally using non-ML algorithms, but the advantage of using them in a neural network is that learning these properties
can teach the network to recognize breaths before it is applied to the ARDS dataset.

### Dataset
First we need a dataset of breaths and corresponding breath metadata. We can set this up by using the `create_separate_breath_meta_dataset.py` script. We should ensure the
script trains on a dataset of patients that **do not** have relation to the current ARDS
cohort. Of course evaluating the performance of the regressor on ARDS dataset patients
is OK though. This script operates by K-means clustering and then randomly picks a number
of breaths from the cluster. This helps to ensure that we have a heterogeneous sampling
of breaths to draw from so that we are not overtraining our regressor on a single breath type. The script will ensure that a given number of breaths are chosen from each cluster.

First ensure the path to patient directories has no relation to the current ARDS dataset.

You can create the dataset:

    python create_separate_breath_meta_dataset.py -dp /path/to/patient/dirs --clusters 75 -bp 100

This implementation will find 75 clusters of breaths from each patient in our dataset and
then will extract a maximum of 100 breaths from each of these clusters. At maximum this
will ensure we get 7500 breaths from each patient.

### Pretraining
Once the dataset is created you can train your regressor on it.

    python train_ards_detector.py -n cnn_regressor --cuda --save-model pretrained-model.pth -dp /path/to/breath/meta/dataset -dt padded_breath_by_breath_with_limited_bm_target

This will save a deep CNN model that you can use to analyze your breath data in future runs.
