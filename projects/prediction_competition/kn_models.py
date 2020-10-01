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
level = "cm"


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

df_mdums = pd.get_dummies(df["month"], prefix = "mdum")
df_ydums = pd.get_dummies(df["year"], prefix = "ydum")

df = df.join(df_mdums)
df = df.join(df_ydums)

import pandas as pd
konstanz_df = pd.read_csv("~/OpenViEWS2/storage/data/konstanz/konstanz.csv", low_memory = False)
#konstanz_df.head()
list(konstanz_df.columns)
#konstanz_df.index

konstanz_df = konstanz_df.set_index(["month_id", "country_id"])
df = df.join(konstanz_df)
cdums = sorted([col for col in df.columns if "cdum" in col], key = lambda x: int(x.split("_")[1]))
mdums = sorted([col for col in df.columns if "mdum" in col], key = lambda x: int(x.split("_")[1]))
ydums = sorted([col for col in df.columns if "ydum" in col], key = lambda x: int(x.split("_")[1]))

#Periods task 1:
period_calib_t1 = api.Period(
    name="calib", 
    train_start=121,   # 1990-01
    train_end=408,     # 2013.12
    predict_start=409, # 2014.01
    predict_end=489,   # 2016.12
)
period_test_t1 = api.Period(
    name="test", 
    train_start=121,   # 1990-01
    train_end=402,     # 2013.12
    predict_start=403, # 2014.01
    predict_end=444,   # 2016.12
)
periods_t1 = [period_calib_t1, period_test_t1]

#Periods task 2:
period_calib_t2 = api.Period(
    name="calib", 
    train_start=121,   # 1990-01
    train_end=408,     # 2013.12
    predict_start=409, # 2014.01
    predict_end=444,   # 2016.12
)
period_test_t2 = api.Period(
    name="test", 
    train_start=121,   # 1990-01
    train_end=444,     # 2016.12
    predict_start=445, # 2017.01
    predict_end=480,   # 2019.12
)
periods_t2 = [period_calib_t2, period_test_t2]

#Periods task 3:
period_calib_t3 = api.Period(
    name="calib", 
    train_start=121,   # 1990-01
    train_end=366,     # 2010.12
    predict_start=367, # 2011.01
    predict_end=402,   # 2013.12
)
period_test_t3 = api.Period(
    name="test", 
    train_start=121,   # 1990-01
    train_end=402,     # 2013.12
    predict_start=403, # 2014.01
    predict_end=444,   # 2016.12
)
periods_t3 = [period_calib_t3, period_test_t3]

steps = [1, 2, 3, 4, 5, 6]


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

##define the features:

##for task 1 models:

features_m0_t1 = basic_features
features_m1_t1 = basic_features + structural_variables + corona_variables
features_m2_t1 = basic_features + structural_variables + corona_variables + political_variables
features_m3_t1 = basic_features + structural_variables + corona_variables+ political_variables + survey_variables


##for task 2 models:

features_m0_t2 = basic_features
features_m1_t2 = basic_features + structural_variables
features_m2_t2 = basic_features + structural_variables + political_variables
features_m3_t2 = basic_features + structural_variables + political_variables + survey_variables
##for task 3 models:

features_m0_t3 = basic_features
features_m1_t3 = basic_features + structural_variables
features_m2_t3 = basic_features + structural_variables + political_variables
features_m3_t3 = basic_features + structural_variables + political_variables + survey_variables
estimators = 200


##task 1, normal models
model_0_t1 = api.Model(
    name = "basic_model_t1",
    col_outcome = "ged_dummy_sb",
    cols_features = features_m0_t1,
    steps = steps,
    periods = periods_t1,
    outcome_type = "real",
    estimator = RandomForestRegressor(n_jobs=-1, criterion="mse", n_estimators=estimators),
    tags=["sb"]
)

model_1_t1 = api.Model(
    name = "structural_model_t1",
    col_outcome = "ged_dummy_sb",
    cols_features = features_m1_t1,
    steps = steps,
    periods = periods_t1,
    outcome_type = "real",
    estimator = RandomForestRegressor(n_jobs=-1, criterion="mse", n_estimators=estimators),
    tags=["sb"]
)

model_2_t1 = api.Model(
    name = "Model_with_elections_t1",
    col_outcome = "ged_dummy_sb",
    cols_features = features_m2_t1,
    steps = steps,
    periods = periods_t1,
    outcome_type = "real",
    estimator = RandomForestRegressor(n_jobs=-1, criterion="mse", n_estimators=estimators),
    tags = ["sb"]
)

model_3_t1 = api.Model(
    name = "model_with_survey_t1",
    col_outcome = "ged_dummy_sb",
    cols_features = features_m3_t1,
    steps = steps,
    periods = periods_t1,
    outcome_type = "real",
    estimator = RandomForestRegressor(n_jobs=-1, criterion="mse", n_estimators=estimators),
    tags=["sb"]
)
##task 1, delta models

model_d0_t1 = api.Model(
    name = "model_d0_t1",
    col_outcome = "ged_dummy_sb",
    cols_features = features_m0_t1,
    steps = steps,
    periods = periods_t1,
    outcome_type = "real",
    estimator = RandomForestRegressor(n_jobs=-1, criterion="mse", n_estimators=estimators),
    delta_outcome=True,
    tags=["sb"]
)

model_d1_t1 = api.Model(
    name = "structural_model_t1",
    col_outcome = "ged_dummy_sb",
    cols_features = features_m1_t1,
    steps = steps,
    periods = periods_t1,
    outcome_type = "real",
    estimator = RandomForestRegressor(n_jobs=-1, criterion="mse", n_estimators=estimators),
    delta_outcome=True,
    tags=["sb"]
)

model_d2_t1 = api.Model(
    name = "Model_with_elections_t1",
    col_outcome = "ged_dummy_sb",
    cols_features = features_m2_t1,
    steps = steps,
    periods = periods_t1,
    outcome_type = "real",
    estimator = RandomForestRegressor(n_jobs=-1, criterion="mse", n_estimators=estimators),
    delta_outcome=True,
    tags = ["sb"]
)

model_d3_t1 = api.Model(
    name = "model_with_survey_t1",
    col_outcome = "ged_dummy_sb",
    cols_features = features_m3_t1,
    steps = steps,
    periods = periods_t1,
    outcome_type = "real",
    estimator = RandomForestRegressor(n_jobs=-1, criterion="mse", n_estimators=estimators),
    delta_outcome=True,
    tags=["sb"]
)

##task 2, normal models
model_d0_t2 = api.Model(
    name = "model_d0_t2",
    col_outcome = "ged_dummy_sb",
    cols_features = features_m0_t2,
    steps = steps,
    periods = periods_t2,
    outcome_type = "real",
    estimator = RandomForestRegressor(n_jobs=-1, criterion="mse", n_estimators=estimators),
    tags=["sb"]
)

model_1_t2 = api.Model(
    name = "model_1_t2",
    col_outcome = "ged_dummy_sb",
    cols_features = features_m1_t2,
    steps = steps,
    periods = periods_t2,
    outcome_type = "real",
    estimator = RandomForestRegressor(n_jobs=-1, criterion="mse", n_estimators=estimators),
    tags=["sb"]
)

model_2_t2 = api.Model(
    name = "model_2_t2",
    col_outcome = "ged_dummy_sb",
    cols_features = features_m2_t2,
    steps = steps,
    periods = periods_t2,
    outcome_type = "real",
    estimator = RandomForestRegressor(n_jobs=-1, criterion="mse", n_estimators=estimators),
    tags = ["sb"]
)

model_3_t2 = api.Model(
    name = "model_3_t2",
    col_outcome = "ged_dummy_sb",
    cols_features = features_m3_t2,
    steps = steps,
    periods = periods_t1,
    outcome_type = "real",
    estimator = RandomForestRegressor(n_jobs=-1, criterion="mse", n_estimators=estimators),
    tags=["sb"]
)
##task 2, delta models
model_d0_t2 = api.Model(
    name = "model_d0_t2",
    col_outcome = "ged_dummy_sb",
    cols_features = features_m0_t2,
    steps = steps,
    periods = periods_t2,
    outcome_type = "real",
    estimator = RandomForestRegressor(n_jobs=-1, criterion="mse", n_estimators=estimators),
    delta_outcome=True,
    tags=["sb"]
)

model_d1_t2 = api.Model(
    name = "model_d1_t2",
    col_outcome = "ged_dummy_sb",
    cols_features = features_m1_t2,
    steps = steps,
    periods = periods_t2,
    outcome_type = "real",
    estimator = RandomForestRegressor(n_jobs=-1, criterion="mse", n_estimators=estimators),
    delta_outcome=True,
    tags=["sb"]
)

model_d2_t2 = api.Model(
    name = "model_d2_t2",
    col_outcome = "ged_dummy_sb",
    cols_features = features_m2_t2,
    steps = steps,
    periods = periods_t2,
    outcome_type = "real",
    estimator = RandomForestRegressor(n_jobs=-1, criterion="mse", n_estimators=estimators),
    delta_outcome=True,
    tags = ["sb"]
)

model_d3_t2 = api.Model(
    name = "model_d3_t2",
    col_outcome = "ged_dummy_sb",
    cols_features = features_m3_t2,
    steps = steps,
    periods = periods_t1,
    outcome_type = "real",
    estimator = RandomForestRegressor(n_jobs=-1, criterion="mse", n_estimators=estimators),
    delta_outcome=True,
    tags=["sb"]
)

##task 3, normal models
model_0_t3 = api.Model(
    name = "model_0_t3",
    col_outcome = "ged_dummy_sb",
    cols_features = features_m0_t3,
    steps = steps,
    periods = periods_t1,
    outcome_type = "real",
    estimator = RandomForestRegressor(n_jobs=-1, criterion="mse", n_estimators=estimators),
    tags=["sb"]
)

model_1_t3 = api.Model(
    name = "model_1_t3",
    col_outcome = "ged_dummy_sb",
    cols_features = features_m1_t3,
    steps = steps,
    periods = periods_t3,
    outcome_type = "real",
    estimator = RandomForestRegressor(n_jobs=-1, criterion="mse", n_estimators=estimators),
    tags=["sb"]
)

model_2_t3 = api.Model(
    name = "model_2_t3",
    col_outcome = "ged_dummy_sb",
    cols_features = features_m2_t3,
    steps = steps,
    periods = periods_t3,
    outcome_type = "real",
    estimator = RandomForestRegressor(n_jobs=-1, criterion="mse", n_estimators=estimators),
    tags = ["sb"]
)

model_3_t3 = api.Model(
    name = "model_3_t3",
    col_outcome = "ged_dummy_sb",
    cols_features = features_m3_t3,
    steps = steps,
    periods = periods_t3,
    outcome_type = "real",
    estimator = RandomForestRegressor(n_jobs=-1, criterion="mse", n_estimators=estimators),
    tags=["sb"]
)
##task 3, delta models

model_d0_t3 = api.Model(
    name = "model_d0_t3",
    col_outcome = "ged_dummy_sb",
    cols_features = features_m0_t3,
    steps = steps,
    periods = periods_t1,
    outcome_type = "real",
    estimator = RandomForestRegressor(n_jobs=-1, criterion="mse", n_estimators=estimators),
    delta_outcome=True,
    tags=["sb"]
)

model_d1_t3 = api.Model(
    name = "model_d1_t3",
    col_outcome = "ged_dummy_sb",
    cols_features = features_m1_t3,
    steps = steps,
    periods = periods_t3,
    outcome_type = "real",
    estimator = RandomForestRegressor(n_jobs=-1, criterion="mse", n_estimators=estimators),
    delta_outcome=True,
    tags=["sb"]
)

model_d2_t3 = api.Model(
    name = "model_d2_t3",
    col_outcome = "ged_dummy_sb",
    cols_features = features_m2_t3,
    steps = steps,
    periods = periods_t3,
    outcome_type = "real",
    estimator = RandomForestRegressor(n_jobs=-1, criterion="mse", n_estimators=estimators),
    delta_outcome=True,
    tags = ["sb"]
)

model_d3_t3 = api.Model(
    name = "model_d3_t3",
    col_outcome = "ged_dummy_sb",
    cols_features = features_m3_t3,
    steps = steps,
    periods = periods_t3,
    outcome_type = "real",
    estimator = RandomForestRegressor(n_jobs=-1, criterion="mse", n_estimators=estimators),
    delta_outcome=True,
    tags=["sb"]
)


# Lists of models are convenient
models_t1 = [model_0_t1, model_1_t1, model_2_t1]
models_d_t1 = [model_d0_t1, model_d1_t1, model_d2_t1]
models_t2 = [model_0_t2, model_1_t2, model_2_t2]
models_d_t2 = [model_d0_t2, model_d1_t2, model_d2_t2]
models_t3 = [model_0_t3, model_1_t3, model_2_t3]
models_d_t3 = [model_d0_t3, model_d1_t3, model_d2_t3]

all_models = models_t1 + models_d_t1 + models_t2 + models_d_t2 + models_t3 + models_d_t3
#To chose which models to run, pick out of the above list:

models = models_t2 + models_d_t2 + models_t3 + models_d_t3

# Train all models
for model in models:
    model.fit_estimators(df)
	
df = df.loc[df.in_africa==1]

for model in models:
    df_predictions = model.predict(df)
    df = assign_into_df(df, df_predictions)
    df_predictions = model.predict_calibrated(
        df=df,
        period_calib = period_calib,
        period_test = period_test
    )
    df = assign_into_df(df, df_predictions)

for model in models:
    model.save()
	
for model in models:
    model.evaluate(df)
	
partition = "test"

for model in models:
    for calib in ["uncalibrated", "calibrated"]:
        scores = {
            "Step":[], 
            "MSE":[], 
            "R2":[]
        }
        if model.delta_outcome:
            scores.update({"TADDA":[]}) 
            
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
            f"{model.name}_{level}_{calib}_scores.tex"
        )
        with open(path_out, "w") as f:
            f.write(tex)
        print(f"Wrote scores table to {path_out}.")



for model in models:
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
