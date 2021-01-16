import argparse

import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from deepards.config import Configuration
from deepards.dataset import ARDSRawDataset
from deepards.train_ards_detector import build_parser, network_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-co', '--config-override', required=True, help='override file')
    parser_args = parser.parse_args()

    model_args = build_parser().parse_args([])
    model_args.config_override = parser_args.config_override
    args = Configuration(model_args)

    dataset = pd.read_pickle(args.train_from_pickle)
    test_dataset = ARDSRawDataset.make_test_dataset_if_kfold(dataset)
    cls = network_map[args.network](args)
    saved_model_dir = Path(__file__).parent.joinpath('saved_models/{}'.format(args.experiment_name))

    for fold in range(5):
        test_dataset.set_kfold_indexes_for_fold(fold)
        test_loader = DataLoader(test_dataset, 16, True)
        device = torch.device('cuda:0')
        model = torch.load(str(saved_model_dir.joinpath(args.models[fold]))).to(device)
        cls.run_test_epoch(0, model, test_dataset, test_loader, fold)


if __name__ == "__main__":
    main()
