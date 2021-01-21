FROM nvcr.io/nvidia/l4t-ml:r32.4.4-py3
USER root

ENV TZ=America/Los_Angeles
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone


# install all dependencies
RUN apt-get update
RUN apt-get install -y python3-pip python3-matplotlib python3-seaborn python3-setuptools build-essential python3-opencv python3-tk
RUN pip3 install -U pip setuptools
COPY static/ucdpvanalysis-1.5.tar.gz ucdpvanalysis.tar.gz
RUN pip3 install ventmap ipython pathlib ucdpvanalysis.tar.gz prettytable pyyaml pandas "scipy>=1.1" scikit-learn imbalanced-learn mock

# copy install dependencies and finish deepards install
COPY static/deepards-2.0.tar.gz deepards-2.0.tar.gz
RUN pip3 install deepards-2.0.tar.gz

# setup ML
COPY static/unpadded_centered_sequences-nb20-kfold.pkl unpadded_centered_sequences-nb20-kfold.pkl
RUN mkdir config
COPY src/evaluate.py evaluate.py
RUN mkdir -p saved_models/experiment_files_unpadded_centered_nb20_cnn_linear
COPY models/* saved_models/experiment_files_unpadded_centered_nb20_cnn_linear/
COPY static/unpadded_centered_nb20_cnn_linear.yml config/
