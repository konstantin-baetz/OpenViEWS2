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
from views.utils import io
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

# Do you wish to fetch the latest public data? If so, change False to True and run this cell
# Cells below will fail if this is not run if you haven't imported data yourself yet.
if False:
    path_zip = views.apps.data.public.fetch_latest_zip_from_website(path_dir_destination=views.DIR_SCRATCH)
    views.apps.data.public.import_tables_and_geoms(tables=views.TABLES, geometries=views.GEOMETRIES, path_zip=path_zip)
# set global variables for choice of models and time structure
testing_mode = False
task = 1
delta_models = True
level = "cm"
if delta_models:
    delta_sig = "DELTA"
else:
    delta_sig = "NORM"

model_path = "./models/{sub}"
out_paths = {
    "evaluation": model_path.format(sub="evaluation"),
    "features": model_path.format(sub="features")
}
for k, v in out_paths.items():
    if not os.path.isdir(v):
        os.makedirs(v)

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

# set periods automatically based on the task variable defined above
# Periods task 1:
period_calib_t1 = api.Period(
    name="calib",
    train_start=121,   # 1990-01
    train_end=488,     # 2013.12
    predict_start=487, # 2014.01
    predict_end=489,   # 2016.12
)
# True forecasts
period_true_t1 = api.Period(
    name="true",
    train_start=121,   # 1990.01
    train_end=489,     # 2020.08
    predict_start=490, # 2020.10
    predict_end=495,   # 2021.03
)

# Periods task 2:
period_calib_t2 = api.Period(
    name="calib",
    train_start=121,  # 1990-01
    train_end=408,  # 2013.12
    predict_start=409,  # 2014.01
    predict_end=444,  # 2016.12
)
period_test_t2 = api.Period(
    name="test",
    train_start=121,  # 1990-01
    train_end=444,  # 2016.12
    predict_start=445,  # 2017.01
    predict_end=480,  # 2019.12
)

# Periods task 3:
period_calib_t3 = api.Period(
    name="calib",
    train_start=121,  # 1990-01
    train_end=366,  # 2010.12
    predict_start=367,  # 2011.01
    predict_end=402,  # 2013.12
)
period_test_t3 = api.Period(
    name="test",
    train_start=121,  # 1990-01
    train_end=402,  # 2013.12
    predict_start=403,  # 2014.01
    predict_end=444,  # 2016.12
)

# Periods task 4:
period_calib_t4 = api.Period(
    name="calib",
    train_start=121,  # 1990-01
    train_end=484,  # 2020.03
    predict_start=484,  # 2020.04
    predict_end=486,  # 2020.06
)
period_test_t4 = api.Period(
    name="test",
    train_start=121,  # 1990-01
    train_end=486,  # 2020.06
    predict_start=487,  # 2020.07
    predict_end=490,  # 2020.10
)

# Periods task 5:
period_calib_t5 = api.Period(
    name="calib",
    train_start=121,  # 1990-01
    train_end=366,  # 2010.12
    predict_start=367,  # 2011.01
    predict_end=402,  # 2013.12
)
period_test_t5 = api.Period(
    name="test",
    train_start=121,  # 1990-01
    train_end=402,  # 2013.12
    predict_start=403,  # 2014.01
    predict_end=444,  # 2016.12
)

if task == 1:
    periods = [period_calib_t1, period_true_t1]
elif task == 2:
    periods = [period_calib_t2, period_test_t2]
elif task == 3:
    periods = [period_calib_t3, period_test_t3]
elif task == 4:
    periods = [period_calib_t4, period_test_t4]
elif task == 5:
    periods = [period_calib_t5, period_test_t5]

if testing_mode == True:
    steps = [1]
else:
    if task < 4:
        steps = [1, 2, 3, 4, 5, 6, 7]
    elif task == 4:
        steps = [1, 2, 3, 4]
    elif task == 5:
        steps = [1, 2, 3, 4]

basic_features = [
    'splag_1_1_acled_count_ns',
    'splag_1_1_acled_count_os',
    'splag_1_1_acled_count_pr',
    'splag_1_1_acled_count_sb',
    'splag_1_1_ged_best_ns',
    'splag_1_1_ged_best_os',
    'splag_1_1_ged_best_sb',
    'time_since_acled_dummy_ns',
    'time_since_acled_dummy_os',
    'time_since_acled_dummy_pr',
    'time_since_acled_dummy_sb',
    'time_since_ged_dummy_ns',
    'time_since_ged_dummy_os',
    'time_since_ged_dummy_sb',
    'time_since_greq_100_splag_1_1_ged_best_sb',
    'time_since_greq_25_ged_best_ns',
    'time_since_greq_25_ged_best_os',
    'time_since_greq_25_ged_best_sb',
    'time_since_greq_500_ged_best_ns',
    'time_since_greq_500_ged_best_os',
    'time_since_greq_500_ged_best_sb',
    'time_since_splag_1_1_acled_dummy_ns',
    'time_since_splag_1_1_acled_dummy_os',
    'time_since_splag_1_1_acled_dummy_pr',
    'time_since_splag_1_1_acled_dummy_sb',
    'time_since_splag_1_1_ged_dummy_ns',
    'time_since_splag_1_1_ged_dummy_os',
    'time_since_splag_1_1_ged_dummy_sb']

structural_variables = [
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
    "kn_food_price"]

political_variables = [
    "reign_anticipation",
    "reign_couprisk",
    "reign_delayed",
    "kn_relative_age",
    "kn_leader_age"]

corona_variables = [
    "kn_death_mil",
    "kn_case_mil",
    "kn_hosp_1k"]

survey_variables = [
    "surkn_n_actors",
    "surkn_pow_var",
    "sur_pos_avg",
    "sur_pos_avg_pw",
    "sur_pos_std",
    "sur_pos_std_pw",
    "sur_hhi"]
#define the features:
features_m0 = basic_features + political_variables
if task == 1 or task == 4:
    features_m1 = basic_features + structural_variables + corona_variables
    features_m2 = basic_features + structural_variables + corona_variables + political_variables
    features_m3 = basic_features + structural_variables + corona_variables + political_variables
elif task == 2:
    features_m1 = basic_features + structural_variables
    features_m2 = basic_features + structural_variables + political_variables
    features_m3 = basic_features + structural_variables + political_variables + survey_variables
elif task == 3:
    features_m1 = basic_features + structural_variables
    features_m2 = basic_features + structural_variables + political_variables
    features_m3 = basic_features + structural_variables + political_variables

#number of estimator
estimators = 200

#normal models
model_0 = api.Model(
    name="basic_model ",
    col_outcome="ln_ged_best_sb",
    cols_features=features_m0,
    steps=steps,
    periods=periods,
    outcome_type="real",
    estimator=RandomForestRegressor(n_jobs=-1, criterion="mse", n_estimators=estimators),
    tags=["sb"]
)

model_1 = api.Model(
    name="structural_model ",
    col_outcome="ln_ged_best_sb",
    cols_features=features_m1,
    steps=steps,
    periods=periods,
    outcome_type="real",
    estimator=RandomForestRegressor(n_jobs=-1, criterion="mse", n_estimators=estimators),
    tags=["sb"]
)

model_2 = api.Model(
    name="Model_with_elections ",
    col_outcome="ln_ged_best_sb",
    cols_features=features_m2,
    steps=steps,
    periods=periods,
    outcome_type="real",
    estimator=RandomForestRegressor(n_jobs=-1, criterion="mse", n_estimators=estimators),
    tags=["sb"]
)

model_3 = api.Model(
    name="model_with_survey ",
    col_outcome="ln_ged_best_sb",
    cols_features=features_m3,
    steps=steps,
    periods=periods,
    outcome_type="real",
    estimator=RandomForestRegressor(n_jobs=-1, criterion="mse", n_estimators=estimators),
    tags=["sb"]
)
#delta models

model_d0 = api.Model(
    name="model_0_delta ",
    col_outcome="ln_ged_best_sb",
    cols_features=features_m0,
    steps=steps,
    periods=periods,
    outcome_type="real",
    estimator=RandomForestRegressor(n_jobs=-1, criterion="mse", n_estimators=estimators),
    delta_outcome=True,
    tags=["sb"]
)

model_d1 = api.Model(
    name="structural_model_delta ",
    col_outcome="ln_ged_best_sb",
    cols_features=features_m1,
    steps=steps,
    periods=periods,
    outcome_type="real",
    estimator=RandomForestRegressor(n_jobs=-1, criterion="mse", n_estimators=estimators),
    delta_outcome=True,
    tags=["sb"]
)

model_d2 = api.Model(
    name="model_with_elections_delta ",
    col_outcome="ln_ged_best_sb",
    cols_features=features_m2,
    steps=steps,
    periods=periods,
    outcome_type="real",
    estimator=RandomForestRegressor(n_jobs=-1, criterion="mse", n_estimators=estimators),
    delta_outcome=True,
    tags=["sb"]
)

model_d3 = api.Model(
    name="model_with_survey_delta ",
    col_outcome="ln_ged_best_sb",
    cols_features=features_m3,
    steps=steps,
    periods=periods,
    outcome_type="real",
    estimator=RandomForestRegressor(n_jobs=-1, criterion="mse", n_estimators=estimators),
    delta_outcome=True,
    tags=["sb"]
)
if task == 2:
    if delta_models == True:
        models = [model_d0, model_d1, model_d2, model_d3]
    elif delta_models == False:
        models = [model_0, model_1, model_2, model_3]
else:
    if delta_models == True:
        models = [model_d0, model_d1, model_d2]
    elif delta_models == False:
        models = [model_0, model_1, model_2]

#models = [model_0, model_1, model_2]


# Train all models
for model in models:
    model.fit_estimators(df)

df = df.loc[df.in_africa == 1]

# df_save = df
period_calib = periods[0]
period_test = periods[1]
for model in models:
    df_predictions = model.predict(df)
    df = assign_into_df(df, df_predictions)
    df_predictions = model.predict_calibrated(
        df=df,
        period_calib=period_calib,
        period_test=period_test
    )
    df = assign_into_df(df, df_predictions)

for model in models:
    model.save()

if delta_models == True:
    if task == 1:
        prediction_data = df.loc[490:495]
        prediction_data.to_csv("/pfs/work7/workspace/scratch/kn_pop503398-ViEWS-0/forecasts_t1_delta_cas.csv")
    elif task == 2:
        prediction_data = df.loc[445:480]
        prediction_data.to_csv("/pfs/work7/workspace/scratch/kn_pop503398-ViEWS-0/forecasts_t2_delta_cas.csv")
    elif task == 3:
        prediction_data = df.loc[403:444]
        prediction_data.to_csv("/pfs/work7/workspace/scratch/kn_pop503398-ViEWS-0/forecasts_t3_delta_cas.csv")
    elif task == 4:
        prediction_data = df.loc[487:490]
        prediction_data.to_csv("/pfs/work7/workspace/scratch/kn_pop503398-ViEWS-0/forecasts_t4_delta_cas.csv")
elif delta_models == False:
    if task == 1:
        prediction_data = df.loc[490:495]
        prediction_data.to_csv("/pfs/work7/workspace/scratch/kn_pop503398-ViEWS-0/forecasts_t1_cas.csv")
    elif task == 2:
        prediction_data = df.loc[445:480]
        prediction_data.to_csv("/pfs/work7/workspace/scratch/kn_pop503398-ViEWS-0/forecasts_t2_cas.csv")
    elif task == 3:
        prediction_data = df.loc[403:444]
        prediction_data.to_csv("/pfs/work7/workspace/scratch/kn_pop503398-ViEWS-0/forecasts_t3_cas.csv")
    elif task == 4:
        prediction_data = df.loc[487:490]
        prediction_data.to_csv("/pfs/work7/workspace/scratch/kn_pop503398-ViEWS-0/forecasts_t4_cas.csv")


for model in models:
    model.evaluate(df)

partition = "test"

for model in models:
    for calib in ["uncalibrated", "calibrated"]:
        scores = {
            "Step": [],
            "MSE": [],
            "R2": []
        }
        if model.delta_outcome:
            scores.update({"TADDA": []})

        for key, value in model.scores[partition].items():
            if key != "sc":
                scores["Step"].append(key)
                scores["MSE"].append(value[calib]["mse"])
                scores["R2"].append(value[calib]["r2"])
                if model.delta_outcome:
                    scores["TADDA"].append(value[calib]["tadda_score"])

        out = pd.DataFrame(scores)
        tex = out.to_latex(index=False)

        # Add meta.
        now = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
        meta = f"""
        %Output created by wb_models.ipynb.
        %Evaluation of {model.col_outcome} per step.
        %Run on selected {model.name} features at {level} level.
        %Produced on {now}, written to {out_paths["evaluation"]}.
        \\
        """
        tex = meta + tex
        path_out = os.path.join(
            out_paths["evaluation"],
            f"{model.name}_{level}_{calib}_t{task}scores.tex"
        )
        with open(path_out, "w") as f:
            f.write(tex)
        print(f"Wrote scores table to {path_out}.")

for model in models:
    model._get_feature_importances()
    fi_cm = featimp_by_steps(
        model=model,
        steps=steps,
        sort_step=sort_step,
        top=top,
        cols=model.cols_features
    )
    fi.write_fi_tex(
        pd.DataFrame(fi_cm),
        os.path.join(out_paths["features"], f"impurity_imp_{model.name}_{level}.tex")
    )

sort_step = 3
top = 30

for model in models:
    for step in steps:
        pi_dict = model.extras.permutation_importances["test"][step]["test"]
        step_df = pd.DataFrame(fi.reorder_fi_dict(pi_dict))
        step_df = step_df.rename(columns={"importance": f"s={step}"})
        step_df.set_index("feature", inplace=True)
        pi_df = pi_df.join(step_df) if step > steps[0] else step_df.copy()

    pi_df = pi_df.sort_values(by=[f"s={sort_step}"], ascending=False)
    pi_df = pi_df[0:top + 1]

    fi.write_fi_tex(
        pi_df,
        os.path.join(out_paths["features"], f"permutation_imp_{model.name}.tex")
    )
