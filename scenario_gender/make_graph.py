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

logger = logging.getLogger(__name__)

def parse_args(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=os.path.abspath)
    parser.add_argument('--trx_file', type=str)
    parser.add_argument('--col_client_id', type=str)
    parser.add_argument('--col_item_id', type=str)
    parser.add_argument('--item_map_file_path', type=str)
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
    # args.__dict__['data_path'] = 'data'
    # args.__dict__['trx_file'] = 'transactions.csv'
    # args.__dict__['col_client_id'] = 'customer_id'
    # args.__dict__['col_item_id'] = "mcc_code"
    # args.__dict__['cols_log_norm'] =[ 'amount']
    # args.__dict__['test_ids_path'] = "data/test_ids.csv"
    # args.__dict__['output_graph_path'] = 'data/graphs/weighted'
    # args.__dict__['output_train_graph_file'] = "train_graph.bin"
    # args.__dict__['output_full_graph_file'] = "full_graph.bin"
    # args.__dict__['output_client_id2full_graph_id_file'] = "client_id2full_graph_id.pt"
    # args.__dict__['output_item_id2full_graph_id_file'] = "item_id2full_graph_id.pt"
    # args.__dict__['output_client_id2train_graph_id_file'] = "client_id2train_graph_id.pt"
    # args.__dict__['output_item_id2train_graph_id_file'] = "item_id2train_graph_id.pt"
    # args.__dict__['log_file'] = "results/dataset_gender.txt"
    # args.__dict__['use_weights'] = "true"
    
    logger.info('Parsed args:\n' + '\n'.join([f'  {k:15}: {v}' for k, v in vars(args).items()]))
    return args


def encode_item_ids(df_data: pd.DataFrame, config) -> pd.DataFrame:
    ITEM_MAP_ORIG_COL_NAME = f'_orig_{config.col_item_id}'
    ITEM_MAP_NULL_TOKEN = '#EMPTY'
    
    item_map = pd.read_csv(config.item_map_file_path)

    df = df_data.rename(columns={config.col_item_id: ITEM_MAP_ORIG_COL_NAME})
    df[ITEM_MAP_ORIG_COL_NAME] = df[ITEM_MAP_ORIG_COL_NAME].fillna(ITEM_MAP_NULL_TOKEN)
    df = df.merge(item_map, on=ITEM_MAP_ORIG_COL_NAME, how='left')
    df = df.drop(columns = [ITEM_MAP_ORIG_COL_NAME])
    return df


def preprocess_df(df_data, config):
    df_data = encode_item_ids(df_data, config)

    logger.info(f"df_data.head() after item_encoding: \n{df_data.head()}")

    # for col in config.cols_log_norm:
    #     df_data[col] = np.log1p(abs(df_data[col])) * np.sign(df_data[col])
    #     df_data[col] /= abs(df_data[col]).max()
    return df_data


def get_train_clients(df_data, config):
    all_clients = set(df_data[config.col_client_id])

    test_clients = pd.read_csv(config.test_ids_path)['customer_id']
    train_clients = all_clients - set(test_clients)
    return train_clients


def main_create_graph(config):
    os.makedirs(config.output_graph_path, exist_ok=True)

    df_data = pd.read_csv(os.path.join(config.data_path, config.trx_file))
    df_data = preprocess_df(df_data, config)
    full_g, client_id2full_graph_id, item_id2full_graph_id, real_items_cnt = GenderGraphBuilder().build(df=df_data,
                                                                         client_col=config.col_client_id,
                                                                         item_col=config.col_item_id,
                                                                         use_weights=config.use_weights,
                                                                         )
    dgl.save_graphs(os.path.join(config.output_graph_path, config.output_full_graph_file), [full_g])
    torch.save(client_id2full_graph_id,
               os.path.join(config.output_graph_path, config.output_client_id2full_graph_id_file))
    torch.save(item_id2full_graph_id,
               os.path.join(config.output_graph_path, config.output_item_id2full_graph_id_file))

    # print(client_id2full_graph_id)

    # create train graph
    train_clients = get_train_clients(df_data, config)
    train_clients = torch.LongTensor(sorted(train_clients))
    train_g = \
        create_subgraph_with_all_neighbors(full_g, node_ids=client_id2full_graph_id[train_clients])

    # 1st part - clients, then - items
    new_id2old_client_id = train_g.ndata['_ID'][:len(train_clients)]

    full_graph_id2train_graph_id = torch.zeros(train_g.ndata['_ID'].max() + 1, dtype=torch.long)
    full_graph_id2train_graph_id[train_g.ndata['_ID']] = train_g.nodes()

    # set of items is the same for both graphs
    print(len(train_g.nodes()))
    print(len(train_clients))
    print(real_items_cnt)
    assert len(train_g.nodes()) - len(train_clients) == real_items_cnt

    client_id2train_graph_id = torch.zeros(len(client_id2full_graph_id), dtype=torch.long)
    client_id2train_graph_id[train_clients] = full_graph_id2train_graph_id[client_id2full_graph_id[train_clients]]

    item_id2train_graph_id = torch.zeros(len(item_id2full_graph_id), dtype=torch.long)
    all_items = torch.arange(len(item_id2full_graph_id))
    item_id2train_graph_id[all_items] = full_graph_id2train_graph_id[item_id2full_graph_id[all_items]]

    # new_id2old_item_id = train_g.ndata['_ID'][len(train_clients):]

    # client_id2train_graph_id = torch.zeros(new_id2old_client_id.max() + 1, dtype=torch.long)
    # client_id2train_graph_id[new_id2old_client_id] = torch.arange(len(train_clients))
    #
    # item_id2train_graph_id = torch.zeros(new_id2old_item_id.max() + 1, dtype=torch.long)
    # item_id2train_graph_id[new_id2old_item_id] = torch.arange(len(new_id2old_item_id)) + len(train_clients)

    dgl.save_graphs(os.path.join(config.output_graph_path, config.output_train_graph_file), [train_g])
    torch.save(client_id2train_graph_id,
               os.path.join(config.output_graph_path, config.output_client_id2train_graph_id_file))
    torch.save(item_id2train_graph_id,
               os.path.join(config.output_graph_path, config.output_item_id2train_graph_id_file))


if __name__ == '__main__':
    _start = datetime.now()
    config = parse_args()
    main_create_graph(config)
    _duration = datetime.now() - _start
    logger.info(f'Data collected in {_duration.seconds} sec ({_duration})')

