#!/usr/bin/env python
# coding: utf-8
# Logging imports
import sys
import json
import logging
import views

import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.ensemble import RandomForestRegressor
from joblib import dump, load

import views
from views import Period, Model, Downsampling, DATASETS, Ensemble
from views.utils.data import assign_into_df
from views.utils import db, io, data as datautils
from views.apps.transforms import lib as translib
from views.apps.evaluation import lib as evallib, feature_importance as fi
from views.apps.model import api
from views.apps.extras import extras
from views.specs.models import cm as model_specs_cm, pgm as model_specs_pgm
from views.specs.periods import get_periods, get_periods_by_name
logging.basicConfig(
    level=logging.DEBUG,
    format=views.config.LOGFMT,
    handlers=[
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# These are the core models defined in the ViEWS pipeline
# These are defined in
from views.apps.pipeline.models_cm import all_cm_models_by_name
# from views.apps.pipeline.models_pgm import all_pgm_models_by_name
# these are not needed for country month analysis

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

models = [model_baseline]

for model in models:
    model.evaluate(df)
