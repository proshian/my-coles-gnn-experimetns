#!/usr/bin/env bash

python make_graph.py \
--data_path data/ \
--trx_file transactions.csv \
--col_client_id customer_id \
--cols_event_time "#gender" "tr_datetime" \
--col_item_id mcc_code \
--cols_log_norm "amount" \
--test_ids_path "data/test_ids.csv" \
--output_train_graph_path "data/train_graph.bin" \
--output_full_graph_path "data/full_graph.bin" \
--output_client_id2graph_id_path "data/client_id2graph_id.pt" \
--output_item_id2graph_id_path "data/item_id2graph_id.pt" \
--log_file "results/dataset_gender.txt" \

# 152 sec with    --print_dataset_info
#  52 sec without --print_dataset_info