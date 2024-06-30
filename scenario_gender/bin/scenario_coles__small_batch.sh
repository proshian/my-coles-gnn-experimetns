# Check COLEs with split_count=2
# was 0.637
# python -m ptls.pl_train_module \
python -m ptls.pl_train_module \
    --config-dir conf --config-name mles_params \
    data_module.train_data.splitter.split_count=2 \
    data_module.valid_data.splitter.split_count=2 \
    pl_module.validation_metric.K=1 \
    pl_module.lr_scheduler_partial.step_size=60 \
    model_path="models/mles_model2.p" \
    logger_name="mles_model2" \
    data_module.train_batch_size=100 \
    data_module.train_num_workers=4 \
    data_module.valid_batch_size=128 \
    data_module.valid_num_workers=4
    # trainer.max_epochs=300 \

    
python -m ptls.pl_inference    \
    model_path="models/mles_model2.p" \
    embed_file_name="mles2_embeddings" \
    inference.batch_size=40 \
    --config-dir conf --config-name mles_params 
    