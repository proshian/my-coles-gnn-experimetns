# Check COLEs with split_count=2
# was 0.637
# python -m ptls.pl_train_module \

# PYTHONPATH is set to make ptls_extension_2024_research module available
PYTHONPATH=.. python -m ptls.pl_train_module \
    --config-dir conf --config-name coles_gnn_params \
    data_module.train_data.splitter.split_count=2 \
    data_module.valid_data.splitter.split_count=2 \
    pl_module.validation_metric.K=1 \
    pl_module.lr_scheduler_partial.step_size=60 \
    model_path="models/coles_gnn_model_for_check_2.p" \
    logger_name="coles_gnn_model_for_check_2"  \
    data_module.train_batch_size=100 \
    data_module.train_num_workers=4 \
    data_module.valid_batch_size=128 \
    data_module.valid_num_workers=4  \
    trainer.max_epochs=5 
    
# PYTHONPATH=.. python -m ptls.pl_inference    \
#     model_path="models/coles_gnn_model_for_check_2.p" \
#     embed_file_name="coles_gnn_model_for_check_2_embeddings" \
#     inference.batch_size=40 \
#     --config-dir conf --config-name coles_gnn_params.yaml 
