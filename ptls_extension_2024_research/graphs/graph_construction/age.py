from abc import ABC, abstractmethod

import numpy as np

from ptls_extension_2024_research.graphs.graph_construction.base import GraphBuilder


class AgeGraphBuilder(GraphBuilder):
    def preprocess(self, df):
        df['amount_rur'] = df['amount_rur'].apply(abs)
        return df

    def _build_weighted_edge_df(self, df, client_col, item_col):
        # df = df[[client_col, item_col, 'amount_rur']].copy()
        df = df[[client_col, item_col, 'amount_rur']]
        df = self.preprocess(df)
        df_type = df.groupby([client_col, item_col]).agg(sum).reset_index()
        df_type = df_type.rename({'amount_rur': 'amount_rur_type_mcc'}, axis=1)
        df_cl = df[[client_col, 'amount_rur']].groupby([client_col]).agg(sum).reset_index()
        df_cl = df_cl.rename({'amount_rur': 'amount_rur_client_total'}, axis=1)
        df_merged = df_type.merge(df_cl, on=client_col, how='left')
        df_merged['amount_rur_tf'] = df_merged['amount_rur_type_mcc'] / df_merged['amount_rur_client_total']

        df_mean_mcc_type = df[[item_col, 'amount_rur']].groupby([item_col]).agg('mean').reset_index()
        df_mean_mcc_type['amount_rur'] = np.sqrt(df_mean_mcc_type['amount_rur'].sum() / df_mean_mcc_type['amount_rur'])
        df_mean_mcc_type = df_mean_mcc_type.rename({'amount_rur': 'mcc_type_norm'}, axis=1)

        df_merged = df_merged.merge(df_mean_mcc_type, on=[item_col], how='left')
        df_merged['amount_rur_tf'] = df_merged['amount_rur_tf'] * df_merged['mcc_type_norm']

        unique_clients_cnt = len(df[client_col].unique())
        df_client_cnt = df[[client_col, item_col]].drop_duplicates().groupby([item_col]).agg(len).reset_index()
        df_client_cnt = df_client_cnt.rename({client_col: 'unique_customers'}, axis=1)
        df_client_cnt['amount_rur_idf'] = np.log(unique_clients_cnt / df_client_cnt['unique_customers'])
        df_merged = df_merged.merge(df_client_cnt, on=[item_col], how='left')
        df_merged['weight'] = df_merged['amount_rur_tf'] * df_merged['amount_rur_idf']
        assert all(df_merged['weight'] != 0)
        return df_merged, client_col, item_col, 'weight'

    def _build_simple_edge_df(self, df, client_col, item_col):
        return df.drop_duplicates([client_col, item_col]), client_col, item_col