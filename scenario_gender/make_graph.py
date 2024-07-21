import sys
import os
sys.path.append(os.path.abspath('..'))

import argparse
import logging
from datetime import datetime

import dgl
import numpy as np
import pandas as pd
import torch

from ptls_extension_2024_research.graphs.graph_construction.gender import GenderGraphBuilder
from ptls_extension_2024_research.graphs.utils import create_subgraph_with_all_neighbors
from scenario_gender.make_dataset_help_no_ptls import encode_col

logger = logging.getLogger(__name__)

def parse_args(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=os.path.abspath)
    parser.add_argument('--trx_file', type=str)
    parser.add_argument('--col_client_id', type=str)
    parser.add_argument('--col_item_id', type=str)
    parser.add_argument('--cols_log_norm', nargs='*', default=[])
    parser.add_argument('--test_ids_path', type=os.path.abspath)

    parser.add_argument('--output_graph_path', type=os.path.abspath)
    parser.add_argument('--output_full_graph_file', type=str)
    parser.add_argument('--output_train_graph_file', type=str)
    parser.add_argument('--output_client_id2full_graph_id_file', type=str)
    parser.add_argument('--output_item_id2full_graph_id_file', type=str)
    parser.add_argument('--output_client_id2train_graph_id_file', type=str)
    parser.add_argument('--output_item_id2train_graph_id_file', type=str)
    parser.add_argument('--log_file', type=os.path.abspath)
    parser.add_argument('--use_weights', action='store_true', default=False)

    args = parser.parse_args(args)
    logger.info('Parsed args:\n' + '\n'.join([f'  {k:15}: {v}' for k, v in vars(args).items()]))
    return args


def preprocess_df(df_data, config):
    df_data[config.col_item_id] = encode_col(df_data[config.col_item_id])

    for col in config.cols_log_norm:
        df_data[col] = np.log1p(abs(df_data[col])) * np.sign(df_data[col])
        df_data[col] /= abs(df_data[col]).max()
    return df_data


def get_train_clients(df_data, config):
    all_clients = set(df_data[config.col_client_id])

    test_clients = pd.read_csv(config.test_ids_path)
    train_clients = all_clients - set(test_clients)
    return train_clients


def main_create_graph(config):
    os.makedirs(config.output_graph_path, exist_ok=True)

    df_data = pd.read_csv(os.path.join(config.data_path, config.trx_file))
    df_data = preprocess_df(df_data, config)
    full_g, client_id2full_graph_id, item_id2full_graph_id = GenderGraphBuilder().build(df=df_data,
                                                                         client_col=config.col_client_id,
                                                                         item_col=config.col_item_id,
                                                                         use_weights=config.use_weights,
                                                                         )
    dgl.save_graphs(os.path.join(config.output_graph_path, config.output_full_graph_file), [full_g])
    torch.save(client_id2full_graph_id,
               os.path.join(config.output_graph_path, config.output_client_id2full_graph_id_file))
    torch.save(item_id2full_graph_id,
               os.path.join(config.output_graph_path, config.output_item_id2full_graph_id_file))

    print(client_id2full_graph_id)

    # create train graph
    train_clients = get_train_clients(df_data, config)
    train_g = \
        create_subgraph_with_all_neighbors(full_g, node_ids=client_id2full_graph_id[torch.LongTensor(sorted(train_clients))])

    # 1st part - clients, then - items
    new_id2old_client_id = train_g.ndata['_ID'][:len(train_clients)]
    new_id2old_item_id = train_g.ndata['_ID'][len(train_clients):]

    client_id2train_graph_id = torch.zeros(new_id2old_client_id.max() + 1, dtype=torch.long)
    client_id2train_graph_id[new_id2old_client_id] = torch.arange(len(train_clients))

    item_id2train_graph_id = torch.zeros(new_id2old_item_id.max() + 1, dtype=torch.long)
    item_id2train_graph_id[new_id2old_item_id] = torch.arange(len(new_id2old_item_id))

    dgl.save_graphs(os.path.join(config.output_graph_path, config.output_train_graph_file), [train_g])
    torch.save(client_id2train_graph_id,
               os.path.join(config.output_graph_path, config.output_client_id2train_graph_id_file))
    torch.save(item_id2train_graph_id,
               os.path.join(config.output_graph_path, config.output_item_id2train_graph_id_file))

    """
    client_id2graph_id: [0, 3,4,5]
    NID: [3, 4, 5] -> [0,0,0,0,1,2]
    хотим: [0]
    """


if __name__ == '__main__':
    _start = datetime.now()
    config = parse_args()
    main_create_graph(config)
    _duration = datetime.now() - _start
    logger.info(f'Data collected in {_duration.seconds} sec ({_duration})')