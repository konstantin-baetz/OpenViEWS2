#!/usr/bin/env bash
set -e
eval "$(conda shell.bash hook)"
conda initiate
conda activate views2 

echo "loading data"

python runners/import_data.py --fetch

echo "starting script"

python projects/prediction_competition/konstanz_models.py

echo "estimations sucessful"
