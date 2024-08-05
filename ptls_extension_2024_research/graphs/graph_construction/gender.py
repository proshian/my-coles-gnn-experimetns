from abc import ABC, abstractmethod

import numpy as np

from ptls_extension_2024_research.graphs.graph_construction.base import GraphBuilder


class GenderGraphBuilder(GraphBuilder):
    def preprocess(self, df, client_col, item_col):
        df = df[[client_col, item_col, 'amount']]
        df['amount'] = df['amount'].apply(abs)
        df['amount'] = np.log1p(df['amount'])
        return df

    def _build_weighted_edge_df(self, df, client_col, item_col):
        df = self.preprocess(df, client_col, item_col)

        grouped_edges = df.groupby([client_col, item_col]).agg(sum)
        edge2sum_amount = dict(zip(grouped_edges.index, grouped_edges['amount']))

        grouped_item_weights = df.groupby([item_col]).agg(sum)
        item2sum = dict(zip(grouped_item_weights.index, grouped_item_weights['amount']))

        df_total = df[[client_col, item_col]].drop_duplicates()

        df_total['weight'] = df_total.apply(lambda row:
                                            edge2sum_amount[(row[client_col], row[item_col])] / item2sum[
                                                row[item_col]], axis=1)
        assert all(df_total['weight'] > 0) and all(df_total['weight'] <= 1)
        return df_total, client_col, item_col, 'weight'

    def _build_simple_edge_df(self, df, client_col, item_col):
        return df.drop_duplicates([client_col, item_col]), client_col, item_col