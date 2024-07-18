#!/usr/bin/env bash

PYTHONPATH=.. python make_graph.py \
--data_path data/ \
--use_weights \
--trx_file transactions.csv \
--col_client_id customer_id \
--col_item_id mcc_code \
--cols_log_norm "amount" \
--test_ids_path "data/test_ids.csv" \
--output_train_graph_path "data/graphs/weighted/train_graph.bin" \
--output_full_graph_path "data/graphs/weighted/full_graph.bin" \
--output_client_id2graph_id_path "data/graphs/weighted/client_id2graph_id.pt" \
--output_item_id2graph_id_path "data/graphs/weighted/item_id2graph_id.pt" \
--log_file "results/weighted_graph_gender.txt" \
# --cols_event_time "#gender" "tr_datetime" \


# 152 sec with    --print_dataset_info
#  52 sec without --print_dataset_info