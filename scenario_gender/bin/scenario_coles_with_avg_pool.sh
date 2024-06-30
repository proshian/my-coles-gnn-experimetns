# Check COLEs with split_count=2
# was 0.637
# python -m ptls.pl_train_module \

# PYTHONPATH is set to make ptls_extension_2024_research module available
PYTHONPATH=.. python -m ptls.pl_train_module \
    --config-dir conf --config-name mles_average_pool_params \
    data_module.train_data.splitter.split_count=2 \
    data_module.valid_data.splitter.split_count=2 \
    pl_module.validation_metric.K=1 \
    pl_module.lr_scheduler_partial.step_size=60 \
    model_path="models/coles_avg_pool_model_2.p" \
    logger_name="coles_avg_pool_model_2"  \
    # trainer.max_epochs=1 \


PYTHONPATH=.. python -m ptls.pl_inference    \
    model_path="models/coles_avg_pool_model_2.p" \
    embed_file_name="coles_avg_pool_model_embeddings" \
    inference.batch_size=100 \
    --config-dir conf --config-name mles_average_pool_params