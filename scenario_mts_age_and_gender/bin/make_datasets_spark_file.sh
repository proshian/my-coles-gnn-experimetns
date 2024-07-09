#!/usr/bin/env bash

export PYTHONPATH="../../"
SPARK_LOCAL_IP="127.0.0.1" spark-submit \
    --master local[8] \
    --name "Age Make Dataset" \
    --driver-memory 8G \
    --conf spark.sql.shuffle.partitions=100 \
    --conf spark.sql.parquet.compression.codec="snappy" \
    --conf spark.ui.port=4041 \
    --conf spark.local.dir="data/.spark_local_dir" \
    make_dataset.py \
    --data_path data/original_data/ \
    --trx_files competition_data_final.parquet \
    --col_client_id "user_id" \
    --cols_event_time "#mts" "date" "part_of_day" \
    --cols_category "region_name" "city_name" "cpe_model_name" "cpe_manufacturer_name" "part_of_day" "cpe_type_cd" "cpe_model_os_type" \
    --cols_log_norm "price" \
    --cols_to_float "request_cnt" \
    --target_files public_train.parquet \
    --col_target "is_male" "age" \
    --test_size 0.1 \
    --output_train_path "data/train_trx_file.parquet" \
    --output_test_path "data/test_trx_file.parquet" \
    --output_test_ids_path "data/test_ids_file.csv" \
    --log_file "results/dataset_age_pred_file.txt"

# 654 sec with    --print_dataset_info
# 144 sec without --print_dataset_info
