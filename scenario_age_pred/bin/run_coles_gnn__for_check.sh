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
sh bin/scenario_coles_gnn__for_check.sh
