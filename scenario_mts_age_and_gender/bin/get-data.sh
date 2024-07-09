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