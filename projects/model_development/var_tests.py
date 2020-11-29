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

from views.utils import io

path = "~/OpenViEWS2/storage/data/datasets/manual.parquet"  # change to your path
cm_global_imp_0 = io.parquet_to_df(path)
df = cm_global_imp_0

df_mdums = pd.get_dummies(df["month"], prefix="mdum")
df_ydums = pd.get_dummies(df["year"], prefix="ydum")

df = df.join(df_mdums)
df = df.join(df_ydums)

konstanz_df = pd.read_csv("~/OpenViEWS2/storage/data/konstanz/konstanz.csv", low_memory=False)
# konstanz_df.head()
list(konstanz_df.columns)
# konstanz_df.index

konstanz_df = konstanz_df.set_index(["month_id", "country_id"])
df = df.join(konstanz_df)
cdums = sorted([col for col in df.columns if "cdum" in col], key=lambda x: int(x.split("_")[1]))
mdums = sorted([col for col in df.columns if "mdum" in col], key=lambda x: int(x.split("_")[1]))
ydums = sorted([col for col in df.columns if "ydum" in col], key=lambda x: int(x.split("_")[1]))

testing_sample = df.loc[490:495]

#vars = list(df.columns.values)
#print(*vars, sep="\n")
# var = tlag_8_ged_dummy_sb

testing_sample.dropna(
    axis=1,
    how='all',
    thresh=None,
    subset=None,
    inplace=True
)

vars = ["reign_anticipation",
    "reign_couprisk",
    "reign_delayed",
    "kn_relative_age",
    "kn_leader_age",
    "imfweo_pcpie_tcurrent",
    "imfweo_pcpie_tmin1",
    "imfweo_pcpie_tplus1",
    "imfweo_ngdp_d_tcurrent",
    "imfweo_ngdp_d_tmin1",
    "imfweo_ngdp_d_tplus1",
    "imfweo_ngdp_d_tplus2",
    "wdi_fp_cpi_totl",
    "vdem_v2x_polyarchy",
    "kn_oilprice",
    "kn_ramadan",
    "kn_food_price",
    "kn_death_mil",
    "kn_case_mil",
    "kn_hosp_1k",
    "surkn_n_actors",
    "surkn_pow_var",
    "sur_pos_avg",
    "sur_pos_avg_pw",
    "sur_pos_std",
    "sur_pos_std_pw",
    "sur_hhi"]

for var in vars:
    if var in df.columns:
        print(var)
        print("variable exists. we good.")
    else:
        print(var)
        print("variable does not exist. probably a problem.")
