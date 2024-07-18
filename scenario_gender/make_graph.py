import argparse
import logging
import os
from datetime import datetime

import dgl
import numpy as np
import pandas as pd
import torch

from ptls_extension_2024_research.graphs.graph_construction.gender import GenderGraphBuilder
from scenario_gender.make_dataset_help_no_ptls import encode_col

logger = logging.getLogger(__name__)

def parse_args(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--trx_file')
    parser.add_argument('--use_weights', action='store_true')

    parser.add_argument('--data_path', type=os.path.abspath)
    parser.add_argument('--col_client_id', type=str)
    parser.add_argument('--col_item_id', type=str)
    parser.add_argument('--cols_log_norm', nargs='*', default=[])
    parser.add_argument('--test_ids_path', type=os.path.abspath)

    parser.add_argument('--output_train_graph_path', type=os.path.abspath)
    parser.add_argument('--output_full_graph_path', type=os.path.abspath)
    parser.add_argument('--output_client_id2graph_id_path', type=os.path.abspath)
    parser.add_argument('--output_item_id2graph_id_path', type=os.path.abspath)
    parser.add_argument('--log_file', type=os.path.abspath)

    args = parser.parse_args(args)
    return args


def configure_logger(config):
    if config.log_file is not None:
        handlers = [logging.StreamHandler(), logging.FileHandler(config.log_file, mode='w')]
    else:
        handlers = None
    logging.basicConfig(level=logging.INFO, format='%(funcName)-20s   : %(message)s', handlers=handlers)


def preprocess_df(df_data, config):
    df_data[config.col_item_id] = encode_col(df_data[config.col_item_id])
    for col in config.cols_log_norm:
        df_data[col] = np.log1p(abs(df_data[col])) * np.sign(df_data[col])
        df_data[col] /= abs(df_data[col]).max()
    return df_data


def main_create_graph(config):
    path_to_transactions = os.path.join(config.data_path, config.trx_file)
    df_data = pd.read_csv(path_to_transactions)
    df_data = preprocess_df(df_data, config)
    g, client_id2graph_id, item_id2graph_id = GenderGraphBuilder().build(df=df_data,
                                                                         client_col=config.col_client_id,
                                                                         item_col=config.col_item_id,
                                                                         use_weights=config.use_weights,
                                                                         )
    dgl.save_graphs(config.output_full_graph_path, [g])
    torch.save(client_id2graph_id, config.output_client_id2graph_id_path)
    torch.save(item_id2graph_id, config.output_item_id2graph_id_path)


if __name__ == '__main__':
    _start = datetime.now()
    config = parse_args()
    configure_logger(config)
    logger.info('Parsed args:\n' + '\n'.join([f'  {k:15}: {v}' for k, v in vars(config).items()]))
    main_create_graph(config)
    _duration = datetime.now() - _start
    logger.info(f'Data collected in {_duration.seconds} sec ({_duration})')