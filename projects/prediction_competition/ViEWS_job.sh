#!/usr/bin/env bash

conda activate views2 

echo "loading data"

python runners/import_data.py --fetch

echo "starting script"

python projects/prediction_competition/konstanz_models.py

echo "estimations sucessful"