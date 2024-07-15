# Check COLEs with split_count=2

# 41031 iterations (training batches) per epoch
# HYDRA_FULL_ERROR=1 PYTHONPATH=.. python -m ptls.pl_train_module \
HYDRA_FULL_ERROR=1 PYTHONPATH=.. python -m ptls.pl_train_module \
    --config-dir conf --config-name mles_params_with_url_iterable \
    data_module.train_data.splitter.split_count=2 \
    data_module.valid_data.splitter.split_count=2 \
    pl_module.validation_metric.K=1 \
    pl_module.lr_scheduler_partial.step_size=60 \
    model_path="models/mles_model_small_batch_2.p" \
    logger_name="mles_model_small_batch_2" \
    data_module.train_batch_size=16 \
    data_module.train_num_workers=4 \
    data_module.valid_batch_size=16 \
    data_module.valid_num_workers=4 \
    # trainer.max_epochs=1 


echo "\n\n\n==== Inference start \n\n\n"


# ! For some resaon even batch_size 50 takes over 2 Gb on inference  !
HYDRA_FULL_ERROR=1 PYTHONPATH=.. python -m ptls.pl_inference    \
    model_path="models/mles_model_small_batch_2.p" \
    embed_file_name="mles_model_small_batch_2_embeddings" \
    inference.batch_size=30 \
    --config-dir conf --config-name mles_params_with_url_iterable 
    