from abc import ABC, abstractmethod

import numpy as np


class GenderGraphBuilder(ABC):

    def preprocess(self, df):
        df['amount'] = df.amount.apply(abs)
        return df

    @abstractmethod
    def _build_weighted_edge_df(self, df, src_col, dst_col):
        df = df[[src_col, dst_col, 'amount']].copy()
        df = self.preprocess(df)
        df_type = df.groupby([src_col, dst_col]).agg(sum).reset_index()
        df_type = df_type.rename({'amount': 'amount_type_mcc'}, axis=1)
        df_cl = df[[src_col, 'amount']].groupby([src_col]).agg(sum).reset_index()
        df_cl = df_cl.rename({'amount': 'amount_client_total'}, axis=1)
        df_merged = df_type.merge(df_cl, on=src_col, how='left')
        df_merged['amount_tf'] = df_merged['amount_type_mcc'] / df_merged['amount_client_total']

        df_mean_mcc_type = df[[dst_col, 'amount']].groupby([dst_col]).agg('mean').reset_index()
        df_mean_mcc_type['amount'] = np.sqrt(df_mean_mcc_type['amount'].sum() / df_mean_mcc_type['amount'])
        df_mean_mcc_type = df_mean_mcc_type.rename({'amount': 'mcc_type_norm'}, axis=1)

        df_merged = df_merged.merge(df_mean_mcc_type, on=[dst_col], how='left')
        df_merged['amount_tf'] = df_merged['amount_tf'] * df_merged['mcc_type_norm']

        unique_clients_cnt = len(df[src_col].unique())
        df_client_cnt = df[[src_col, dst_col]].drop_duplicates().groupby([dst_col]).agg(len).reset_index()
        df_client_cnt = df_client_cnt.rename({src_col: 'unique_customers'}, axis=1)
        df_client_cnt['amount_idf'] = np.log(unique_clients_cnt / df_client_cnt['unique_customers'])
        df_merged = df_merged.merge(df_client_cnt, on=[dst_col], how='left')
        df_merged['weight'] = df_merged['amount_tf'] * df_merged['amount_idf']
        assert all(df_merged['weight'] != 0)
        return df_merged, src_col, dst_col, 'weight'

    @abstractmethod
    def _build_simple_edge_df(self, df, src_col, dst_col):
        return df.drop_duplicates([src_col, dst_col]), src_col, dst_col