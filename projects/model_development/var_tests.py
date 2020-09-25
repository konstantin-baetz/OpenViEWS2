
import sys
import os
import logging

import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.ensemble import RandomForestRegressor
from joblib import dump, load

import views
from views import Period, Model, Downsampling
from views.utils.data import assign_into_df
from views.apps.transforms import lib as translib
from views.apps.evaluation import lib as evallib, feature_importance as fi
from views.apps.model import api
from views.apps.extras import extras

logging.basicConfig(
    level=logging.DEBUG,
    format=views.config.LOGFMT,
    handlers=[
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

#from views.utils import io
path = "~/OpenViEWS2/storage/data/datasets/manual.parquet"  # change to your path
cm_global_imp_0 = io.parquet_to_df(path)



df = dataset.df
vars = list(df.columns.values)
print(*vars, sep = "\n")
#var = tlag_8_ged_dummy_sb
if 'imfweo_bca_ngdpd_tmin1' in df.columns: 
    print("variable exists. we good.")
else:
    print("variable does not exist. probably a problem.")
