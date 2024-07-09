#!/usr/bin/env bash

mkdir data
cd data
mkdir original_data
cd original_data

curl -OL 'https://huggingface.co/datasets/proshian/mts-ml-cup-2023/resolve/main/competition_data_final_pqt.tar.gz'
curl -OL 'https://huggingface.co/datasets/proshian/mts-ml-cup-2023/resolve/main/public_train.pqt.gz'

gunzip -f *.pqt.gz
tar -xzf *.tar.gz
rm *.tar.gz


# # Loop through all .pqt files in the current directory
# for file in *.pqt; do
#   # Check if any .pqt files exist
#   if [ -e "$file" ]; then
#     # Rename the file to .parquet
#     mv -- "$file" "${file%.pqt}.parquet"
#   fi
# done

mv competition_data_final_pqt competition_data_final.parquet
mv public_train.pqt public_train.parquet