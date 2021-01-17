import argparse

import pandas as pd
from pathlib import Path
from prettytable import PrettyTable
from sklearn.metrics import accuracy_score, roc_auc_score
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
    device = torch.device('cuda:0')

    for fold in range(5):
        test_dataset.set_kfold_indexes_for_fold(fold)
        test_loader = DataLoader(test_dataset, 16, True)
        for i, model_name in enumerate(args.models[fold]):
            model = torch.load(str(saved_model_dir.joinpath(model_name))).to(device)
            # this is a bit of a hack to be able to aggregate across multiple different
            # runs (having each model be an individual epoch)
            cls.run_test_epoch(i, model, test_dataset, test_loader, fold)

    print('\nMean Results')
    table = PrettyTable()
    table.field_names = ['Fold', 'Accuracy', 'AUC']
    for i, fold_df in cls.results.results.groupby('fold_num'):
        accuracy = round(accuracy_score(fold_df.patho.tolist(), fold_df.prediction.tolist()), 4)
        auc = round(roc_auc_score(fold_df.patho.tolist(), fold_df.pred_frac.tolist()), 4)
        table.add_row([i, accuracy, auc])
    print(table)

    print("\nAggregated Results")
    cls.results.aggregate_classification_results()


if __name__ == "__main__":
    main()
