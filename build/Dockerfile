FROM nvcr.io/nvidia/l4t-pytorch:r32.4.4-pth1.6-py3
USER root

ENV TZ=America/Los_Angeles
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# copy install dependencies
COPY static/ucdpvanalysis-1.5.tar.gz ucdpvanalysis.tar.gz
COPY static/deepards-2.0.tar.gz deepards-2.0.tar.gz
COPY static/dtwco-master.zip dtwco-master.zip

# install all dependencies
RUN apt-get update
RUN apt-get install -y python3-pip python3-matplotlib python3-seaborn python3-setuptools unzip build-essential
RUN unzip dtwco-master.zip
RUN pip3 install -U pip setuptools
RUN pip3 install ventmap ipython pathlib cython ucdpvanalysis.tar.gz prettytable pyyaml pandas numpy "scipy>=1.1" scikit-learn imbalanced-learn
RUN (cd dtwco-master/ && python3 setup.py install)
RUN pip3 install deepards-2.0.tar.gz

# setup ML
COPY static/unpadded_centered_sequences-nb20-kfold.pkl unpadded_centered_sequences-nb20-kfold.pkl
RUN mkdir config
COPY src/evaluate.py evaluate.py
RUN mkdir -p saved_models/experiment_files_unpadded_centered_nb20_cnn_linear
COPY models/* saved_models/experiment_files_unpadded_centered_nb20_cnn_linear/
COPY static/unpadded_centered_nb20_cnn_linear.yml config/
