# Docker
## Dependencies to be installed on host
Before building the dockerfile you should
1. Install cuda toolkit 12.1.0 or later on your host machine ([follow official instructions](https://developer.nvidia.com/cuda-12-1-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=18.04&target_type=deb_local))
2. Install NVIDIA Container Toolkit on your host machine (instructions are below)

NVIDIA Container Toolkit installation instrucitons: 

```python
# Add the package repositories
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install the NVIDIA Docker package
sudo apt-get update
sudo apt-get install -y nvidia-docker2

# Restart the Docker daemon to apply the changes
sudo systemctl restart docker

```

## Build
```sh
docker build -t proshian/ptls-experiments .
```
## Run
```sh
docker run --gpus all --ipc=host -it --rm -p 6006:6006 -p 8082:8082 -p 4041:4041 proshian/ptls-experiments 
```


# Known problems

Хоть `bin/make_datasets_spark_file.sh` корректно отрабатывает, процесс невозможно отследить через `localhost:4041`. При запуске вне контейнера по этому порту доступен UI, демонстррующий статус задачи.

# Nano documentation :)

Since all experiments are similar we'll consider scenario_age_pred

* bin/get-data.sh downloads original data. It's usually a tebale with each row representing a single transaction
* bin/make_datasets_spark_file.sh uses spark to convert the dataset to a ptls format. The result are two files: train_trx_file.parquet, test_trx_file.parquet and test_ids_file.csv. It's a parket file where keys store sequences for a user. So each element represents a user and each colun is a 1d nd.array with activity info by the user. This script actually launches ptls.make_datasets_spark.DatasetConverter().run(). An example can be seen in CoLES-investigation repo.
* `scenario_run_all` runs data splitting before running scenarios. The result is stored in `embeddings_validation.work/folds` Info on data spliting is below

## Data splitting
```python -m embeddings_validation \
    --config-dir conf --config-name embeddings_validation_baselines_supervised +workers=10 +total_cpu_count=20 \
    +split_only=True
```

```
Splits data into n folds (train, validation, test),
saves them to files and creates folds.json file with paths to these files.
The saved files are dumped TargetFile objects that contain 
ids and target values (subset of the original data).
Use TargetFile.load(path) to load theese files.
TargetFile = embeddings_validation.file_reader.TargetFile
Main interface of TargetFile are pseudo properties:
* .ids_values
* .target_values
The Target (Luigi Target) of this task is folds.json file with folds information.
The split can be done in two ways (self.conf.validation_schema):
* VALID_TRAIN_TEST
* VALID_CROSS_VAL
For both split types the folds Dict has the same structure:
Keys represent fold number and values are dictionaries with keys:
- 'train': dictionary with keys 'path' and 'shape' representing path to the train data and its shape
- 'valid': dictionary with keys 'path' and 'shape' representing path to the validation data and its shape
- 'test': dictionary with keys 'path' and 'shape' representing path to the test data and its shape

Test data is optional. If it is not provided, the 'test' property is None.
If it's providded, regardless of the split type, the 'test' 
property is always the same (same path and shape for all folds).
If validation_schema is VALID_CROSS_VAL:
* Fold numbers are integers from 0 to self.conf['split']['cv_split_count'] - 1
* Each fold contains train and validation data randomly split from the train data
If validation_schema is VALID_TRAIN_TEST:
* Fold numbers are integers from 0 to self.conf['split']['n_iteration'] - 1
* Each fold is exactly the same (Uses given train, validation and test ids); 
    same path and shape for all folds
```





<br>
<br>
<br>
<br>
<br>


# Original README

Experiments on public datasets for `pytorch-lifestream` library

# Setup and test using pipenv

```sh
# Ubuntu 18.04

sudo apt install python3.8 python3-venv
pip3 install pipenv

pipenv sync  --dev # install packages exactly as specified in Pipfile.lock
pipenv shell
pytest

# run luigi server
luigid
# check embedding validation progress at `http://localhost:8082/`

# use tensorboard for metrics exploration
tensorboard --logdir lightning_logs/ 
# check tensorboard metrics at `http://localhost:6006/`

```

# Run scenario
 We check 5 datasets as separate experiments. See `README.md` files in experiments folder:
 - [Age](scenario_age_pred/README.md)
 - [Churn](scenario_rosbank/README.md)
 - [Assess](scenario_bowl2019/README.md)
 - [Retail](scenario_x5/README.md)
 - [Scoring](scenario_alpha_battle/README.md)
 - [Small demo dataset](scenario_gender/README.md)

# Notebooks

Full scenarious are console scripts configured by hydra yaml configs.
If you like jupyter notebooks you can see an example for AgePred dataset in [AgePred notebooks](scenario_age_pred/notebooks/)

# Results

All results are stored in `*/results` folder.

Unsupervised learned embeddings with LightGBM model downstream evaluations:
|                         |     mean $\pm$ std      |
|-------------------------|-------------------------|
|    **Gender**           |  **AUROC**              |
|        baseline         |    0.877 $\pm$ 0.010    |
|        cpc_embeddings   |    0.851 $\pm$ 0.006    |
|        mles2_embeddings |    0.882 $\pm$ 0.006    |
|        mles_embeddings  |    0.881 $\pm$ 0.006    |
|        nsp_embeddings   |    0.852 $\pm$ 0.011    |
|        random_encoder   |    0.593 $\pm$ 0.020    |
|        rtd_embeddings   |    0.855 $\pm$ 0.008    |
|        sop_embeddings   |    0.785 $\pm$ 0.007    |
|        barlow_twins     |    0.865 $\pm$ 0.007    |
| **Age group (age_pred)**|  **Accuracy**           |
|        baseline         |    0.629 $\pm$ 0.002    |
|        cpc_embeddings   |    0.602 $\pm$ 0.004    |
|        mles2_embeddings |    0.643 $\pm$ 0.003    |
|        mles_embeddings  |    0.640 $\pm$ 0.004    |
|        mles_longformer  |    0.630 $\pm$ 0.003    |
|        nsp_embeddings   |    0.621 $\pm$ 0.005    |
|        random_encoder   |    0.375 $\pm$ 0.003    |
|        rtd_embeddings   |    0.631 $\pm$ 0.006    |
|        sop_embeddings   |    0.512 $\pm$ 0.002    |
|        barlow_twins     |    0.634 $\pm$ 0.003    |
|        coles_transformer|    0.646 $\pm$ 0.003    |
|    **Churn (rosbank)**  |  **AUROC**              |
|        baseline         |    0.827  $\pm$ 0.010   |
|        cpc_embeddings   |    0.792  $\pm$ 0.015   |
|        mles2_embeddings |    0.837  $\pm$ 0.006   |
|        mles_embeddings  |    0.841  $\pm$ 0.010   |
|        nsp_embeddings   |    0.828  $\pm$ 0.012   |
|        random_encoder   |    0.725  $\pm$ 0.013   |
|        rtd_embeddings   |    0.771  $\pm$ 0.016   |
|        sop_embeddings   |    0.780  $\pm$ 0.012   |
|        barlow_twins     |    0.839  $\pm$ 0.010   |
|**Assessment (bowl2019)**|  **Accuracy**           |
|        barlow_twins     |    0.595 $\pm$ 0.005    |    
|        baseline         |    0.592 $\pm$ 0.004    |    
|        cpc_embeddings   |    0.593 $\pm$ 0.004    |    
|        mles2_embeddings |    0.588 $\pm$ 0.008    |    
|        mles_embeddings  |    0.597 $\pm$ 0.001    |    
|        nsp_embeddings   |    0.579 $\pm$ 0.002    |    
|        random_encoder   |    0.574 $\pm$ 0.004    |
|        rtd_embeddings   |    0.574 $\pm$ 0.004    |
|        sop_embeddings   |    0.567 $\pm$ 0.005    |    
|    **Retail (x5)**      |  **Accuracy**           |
|        baseline         |    0.547 $\pm$ 0.001    |
|        cpc_embeddings   |    0.525 $\pm$ 0.001    |
|        mles_embeddings  |    0.539 $\pm$ 0.001    |
|        nsp_embeddings   |    0.425 $\pm$ 0.002    |
|        rtd_embeddings   |    0.520 $\pm$ 0.001    |
|        sop_embeddings   |    0.428 $\pm$ 0.001    |
|**Scoring (alpha battle)**| **AUROC**              |
|        baseline         |    0.7792 $\pm$ 0.0006  |
|        random_encoder   |    0.6456 $\pm$ 0.0009  |
|        barlow_twins     |    0.7878 $\pm$ 0.0009  |
|        cpc              |    0.7919 $\pm$ 0.0004  |
|        mles             |    0.7921 $\pm$ 0.0003  |
|        nsp              |    0.7655 $\pm$ 0.0006  |
|        rtd              |    0.7910 $\pm$ 0.0006  |
|        sop              |    0.7238 $\pm$ 0.0010  |
|        mlmnsp           |    0.7591 $\pm$ 0.0044  |
|        tabformer        |    0.7862 $\pm$ 0.0042  |
|        gpt              |    0.7737 $\pm$ 0.0032  |
|   coles_transformer     |    0.7968 $\pm$ 0.0007  |

Supervised finetuned encoder with MLP head evaluation:
|                         |     mean $\pm$ std      |
|-------------------------|-------------------------|
|    **Gender**           |  **AUROC**              |
|        barlow_twins     |    0.865 $\pm$ 0.011    |
|        cpc_finetuning   |    0.865 $\pm$ 0.007    |
|        mles_finetuning  |    0.879 $\pm$ 0.007    |
|        rtd_finetuning   |    0.868 $\pm$ 0.006    |
|        target_scores    |    0.867 $\pm$ 0.008    |
|**Age group (age_pred)** |  **Accuracy**           |
|        barlow_twins     |    0.619 $\pm$ 0.004    |
|        cpc_finetuning   |    0.625 $\pm$ 0.005    |
|        mles_finetuning  |    0.624 $\pm$ 0.005    |
|        rtd_finetuning   |    0.622 $\pm$ 0.003    |
|        target_scores    |    0.620 $\pm$ 0.006    |
|    **Churn (rosbank)**  |  **AUROC**              |
|        barlow_twins     |    0.830 $\pm$ 0.006    |
|        cpc_finetuning   |    0.804 $\pm$ 0.017    |
|        mles_finetuning  |    0.819 $\pm$ 0.011    |
|        nsp_finetuning   |    0.806 $\pm$ 0.010    |
|        rtd_finetuning   |    0.791 $\pm$ 0.016    |
|        target_scores    |    0.818 $\pm$ 0.005    |
|**Assessment (bowl2019)**|  **Accuracy**           |
|        barlow_twins     |    0.561 $\pm$ 0.007    |    
|        cpc_finetuning   |    0.594 $\pm$ 0.002    |    
|        mles_finetuning  |    0.577 $\pm$ 0.007    |    
|        rtd_finetuning   |    0.571 $\pm$ 0.003    |    
|        target_scores    |    0.585 $\pm$ 0.002    |
|    **Retail (x5)**      |  **Accuracy**           |
|        cpc_finetuning   |    0.549 $\pm$ 0.001    |
|        mles_finetuning  |    0.552 $\pm$ 0.001    |
|        rtd_finetuning   |    0.544 $\pm$ 0.002    |
|        target_scores    |    0.542 $\pm$ 0.001    |

# Other experiments

- [Data Fusion Contest 2024, 2-st place on the Churn Task](https://github.com/warofgam/Sber-AI-Lab---datafusion) (in Russian) 
- [Data Fusion Contest 2024, Ivan Alexandrov](https://github.com/Ivanich-spb/datafusion_2024_churn) (in Russian)
- [Data Fusion Contest 2022. 1-st place on the Matching Task](https://github.com/ivkireev86/datafusion-contest-2022)
- [Alpha BKI dataset](experiments/scenario_alpha_rnn_vs_transformer/README.md)
- [American Express - Default Prediction](https://www.kaggle.com/competitions/amex-default-prediction)
    - [Supervised training with RNN](https://www.kaggle.com/code/ivkireev/amex-ptls-baseline-supervised-neural-network)
    - [Supervised training with Transformer](https://www.kaggle.com/code/ivkireev/amex-transformer-network-train-with-ptls)
    - [CoLES Embedding preparation](https://www.kaggle.com/code/ivkireev/amex-contrastive-embeddings-with-ptls-coles)
    - [CoLES Embedding usage as extra features for catboost](https://www.kaggle.com/code/ivkireev/catboost-classifier-with-coles-embeddings)
- [Softmax loss](experiments/softmax_loss_vs_contrastive_loss/readme.md) - try CoLES with Softmax loss.
- [Random features](experiments/random_features/readme.md) - how CoLES works with slowly changing features which helps to distinguish clients.
- [Small prretrain](experiments/mles_experiments_supervised_only/README.md) - check the CoLES quality depends on prertain size.
- [COTIC](https://github.com/VladislavZh/COTIC) - `pytorch-lifestream` is used in experiment for Continuous-time convolutions model of event sequences.

