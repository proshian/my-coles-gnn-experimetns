# start with working directory: scenario_age_pred
# dataset should be prepared before this script
echo "==== Folds split"
rm -r lightning_logs/
rm -r conf/embeddings_validation.work/
python -m embeddings_validation \
    --config-dir conf --config-name embeddings_validation_baselines_supervised +workers=10 +total_cpu_count=4 \
    +split_only=True

echo "==== Device cuda:${CUDA_VISIBLE_DEVICES} will be used"

echo ""
echo "==== Start coles run"
PYTHONPATH=.. python -m pl_train_module \
    --config-dir conf --config-name mles_average_pool_experimeny__params \
    data_module.train_data.splitter.split_count=2 \
    data_module.valid_data.splitter.split_count=2 \
    pl_module.validation_metric.K=1 \
    trainer.max_epochs=300 \
    pl_module.lr_scheduler_partial.step_size=60 \
    model_path="models/mles_model2.p" \
    logger_name="mles_model2"  \

    
# PYTHONPATH=.. python -m pl_inference    \
#     model_path="models/mles_model2.p" \
#     embed_file_name="mles2_embeddings" \
#     --config-dir conf --config-name coles_gnn_params__for_check.yaml