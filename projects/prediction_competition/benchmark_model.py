#!/usr/bin/env python
# coding: utf-8

# # ViEWS prediction competition benchmark
# 
# This is the ViEWS prediction competition benchmark model notebook.
# If you are a member of the public please run the following in your terminal to fetch the data needed to run this notebook.
# 
#     conda activate views2
#     cd /path/to/ViEWS2
#     python runners/import_data.py --fetch --dataset
#     
# Which will download the latest available data from the ViEWS website and construct the datasets used in this notebook.
# 
# ViEWS team members should be able to run the notebook without any additional steps after configuring database access.

# In[1]:


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


# In[2]:


logging.basicConfig(
    level=logging.DEBUG,
    format=views.config.LOGFMT,
    handlers=[
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# # Setup

# ## Fetch data?

# In[3]:


# Do you wish to fetch the latest public data? If so, change False to True and run this cell
# Cells below will fail if this is not run if you haven't imported data yourself yet.
if False:
    path_zip = views.apps.data.public.fetch_latest_zip_from_website(path_dir_destination=views.DIR_SCRATCH)
    views.apps.data.public.import_tables_and_geoms(tables=views.TABLES, geometries=views.GEOMETRIES, path_zip=path_zip)


# In[4]:


# Enter the level of analysis, and whether to predict Africa-only.
level = "cm"
pred_africa = True


# In[5]:


model_path = "./models/{sub}"
out_paths = {
    "evaluation": model_path.format(sub="evaluation"),
    "features": model_path.format(sub="features")
}
for k, v in out_paths.items():
    if not os.path.isdir(v):
        os.makedirs(v)


# In[6]:


if level == "pgm":
    dataset = views.DATASETS["pgm_africa_imp_0"]
elif level == "cm":
    dataset = views.DATASETS["cm_global_imp_0"]
df = dataset.df


# In[7]:


# Define our 2017.01-2019.12 development period
# Keeping periods in a list lets us easily expand this as the 
# updated data becomes available
period_calib = api.Period(
    name="calib", 
    train_start=121,   # 1990-01
    train_end=408,     # 2013.12
    predict_start=409, # 2014.01
    predict_end=444,   # 2016.12
)
period_test = api.Period(
    name="test", 
    train_start=121,   # 1990-01
    train_end=444,     # 2016.12
    predict_start=445, # 2017.01
    predict_end=480,   # 2019.12
)
periods = [period_calib, period_test]






# In[8]:


# The steps to train, predict and evaluate for.
steps = [1, 2, 3, 4, 5, 6]


# ## Feature selection and model setup

# In[9]:


cols_cm = [
    'cdum_1',
    'cdum_10',
    'cdum_100',
    'cdum_101',
    'cdum_102',
    'cdum_103',
    'cdum_104',
    'cdum_105',
    'cdum_106',
    'cdum_107',
    'cdum_108',
    'cdum_109',
    'cdum_11',
    'cdum_110',
    'cdum_111',
    'cdum_112',
    'cdum_113',
    'cdum_114',
    'cdum_115',
    'cdum_116',
    'cdum_117',
    'cdum_118',
    'cdum_119',
    'cdum_12',
    'cdum_120',
    'cdum_121',
    'cdum_122',
    'cdum_123',
    'cdum_124',
    'cdum_125',
    'cdum_126',
    'cdum_127',
    'cdum_128',
    'cdum_129',
    'cdum_13',
    'cdum_130',
    'cdum_131',
    'cdum_132',
    'cdum_133',
    'cdum_134',
    'cdum_135',
    'cdum_136',
    'cdum_137',
    'cdum_138',
    'cdum_139',
    'cdum_14',
    'cdum_140',
    'cdum_141',
    'cdum_142',
    'cdum_143',
    'cdum_144',
    'cdum_145',
    'cdum_146',
    'cdum_147',
    'cdum_148',
    'cdum_149',
    'cdum_15',
    'cdum_150',
    'cdum_151',
    'cdum_152',
    'cdum_153',
    'cdum_154',
    'cdum_155',
    'cdum_156',
    'cdum_157',
    'cdum_158',
    'cdum_159',
    'cdum_16',
    'cdum_160',
    'cdum_161',
    'cdum_162',
    'cdum_163',
    'cdum_164',
    'cdum_165',
    'cdum_166',
    'cdum_167',
    'cdum_168',
    'cdum_169',
    'cdum_17',
    'cdum_170',
    'cdum_171',
    'cdum_172',
    'cdum_173',
    'cdum_174',
    'cdum_175',
    'cdum_176',
    'cdum_177',
    'cdum_178',
    'cdum_179',
    'cdum_18',
    'cdum_180',
    'cdum_181',
    'cdum_182',
    'cdum_183',
    'cdum_184',
    'cdum_185',
    'cdum_186',
    'cdum_187',
    'cdum_188',
    'cdum_189',
    'cdum_19',
    'cdum_190',
    'cdum_191',
    'cdum_192',
    'cdum_193',
    'cdum_194',
    'cdum_195',
    'cdum_196',
    'cdum_197',
    'cdum_198',
    'cdum_199',
    'cdum_2',
    'cdum_20',
    'cdum_200',
    'cdum_201',
    'cdum_202',
    'cdum_203',
    'cdum_204',
    'cdum_205',
    'cdum_206',
    'cdum_207',
    'cdum_208',
    'cdum_209',
    'cdum_21',
    'cdum_210',
    'cdum_211',
    'cdum_212',
    'cdum_213',
    'cdum_214',
    'cdum_215',
    'cdum_216',
    'cdum_217',
    'cdum_218',
    'cdum_219',
    'cdum_22',
    'cdum_220',
    'cdum_221',
    'cdum_222',
    'cdum_223',
    'cdum_224',
    'cdum_225',
    'cdum_226',
    'cdum_227',
    'cdum_228',
    'cdum_229',
    'cdum_23',
    'cdum_230',
    'cdum_231',
    'cdum_232',
    'cdum_233',
    'cdum_234',
    'cdum_235',
    'cdum_236',
    'cdum_237',
    'cdum_238',
    'cdum_239',
    'cdum_24',
    'cdum_240',
    'cdum_241',
    'cdum_242',
    'cdum_243',
    'cdum_244',
    'cdum_245',
    'cdum_246',
    'cdum_247',
    'cdum_248',
    'cdum_249',
    'cdum_25',
    'cdum_250',
    'cdum_251',
    'cdum_252',
    'cdum_253',
    'cdum_254',
    'cdum_255',
    'cdum_26',
    'cdum_27',
    'cdum_28',
    'cdum_29',
    'cdum_3',
    'cdum_30',
    'cdum_31',
    'cdum_32',
    'cdum_33',
    'cdum_34',
    'cdum_35',
    'cdum_36',
    'cdum_37',
    'cdum_38',
    'cdum_39',
    'cdum_4',
    'cdum_40',
    'cdum_41',
    'cdum_42',
    'cdum_43',
    'cdum_44',
    'cdum_45',
    'cdum_46',
    'cdum_47',
    'cdum_48',
    'cdum_49',
    'cdum_5',
    'cdum_50',
    'cdum_51',
    'cdum_52',
    'cdum_53',
    'cdum_54',
    'cdum_55',
    'cdum_56',
    'cdum_57',
    'cdum_58',
    'cdum_59',
    'cdum_6',
    'cdum_60',
    'cdum_61',
    'cdum_62',
    'cdum_63',
    'cdum_64',
    'cdum_65',
    'cdum_66',
    'cdum_67',
    'cdum_68',
    'cdum_69',
    'cdum_7',
    'cdum_70',
    'cdum_71',
    'cdum_72',
    'cdum_73',
    'cdum_74',
    'cdum_75',
    'cdum_76',
    'cdum_77',
    'cdum_78',
    'cdum_79',
    'cdum_8',
    'cdum_80',
    'cdum_81',
    'cdum_82',
    'cdum_83',
    'cdum_84',
    'cdum_85',
    'cdum_86',
    'cdum_87',
    'cdum_88',
    'cdum_89',
    'cdum_9',
    'cdum_90',
    'cdum_91',
    'cdum_92',
    'cdum_93',
    'cdum_94',
    'cdum_95',
    'cdum_96',
    'cdum_97',
    'cdum_98',
    'cdum_99',
    'fvp_demo',
    'fvp_grgdpcap_nonoilrent',
    'fvp_grgdpcap_oilrent',
    'fvp_grpop200',
    'fvp_indepyear',
    'fvp_lngdp200',
    'fvp_lngdpcap_nonoilrent',
    'fvp_lngdpcap_oilrent',
    'fvp_lngdppercapita200',
    'fvp_population200',
    'fvp_prop_discriminated',
    'fvp_prop_dominant',
    'fvp_prop_excluded',
    'fvp_prop_irrelevant',
    'fvp_prop_powerless',
    'fvp_semi',
    'fvp_ssp2_edu_sec_15_24_prop',
    'fvp_ssp2_urban_share_iiasa',
    'fvp_timeindep',
    'fvp_timesincepreindepwar',
    'fvp_timesinceregimechange',
    'icgcw_alerts',
    'icgcw_deteriorated',
    'icgcw_improved',
    'icgcw_opportunities',
    'icgcw_unobserved',
    'in_africa',
    'reign_age',
    'reign_anticipation',
    'reign_change_recent',
    'reign_couprisk',
    'reign_defeat_recent',
    'reign_delayed',
    'reign_direct_recent',
    'reign_elected',
    'reign_election_now',
    'reign_election_recent',
    'reign_exec_ant',
    'reign_exec_recent',
    'reign_gov_dominant_party',
    'reign_gov_foreign_occupied',
    'reign_gov_indirect_military',
    'reign_gov_military',
    'reign_gov_military_personal',
    'reign_gov_monarchy',
    'reign_gov_oligarchy',
    'reign_gov_parliamentary_democracy',
    'reign_gov_party_military',
    'reign_gov_party_personal',
    'reign_gov_party_personal_military_hybrid',
    'reign_gov_personal_dictatorship',
    'reign_gov_presidential_democracy',
    'reign_gov_provisional_civilian',
    'reign_gov_provisional_military',
    'reign_gov_warlordism',
    'reign_indirect_recent',
    'reign_irreg_lead_ant',
    'reign_irregular',
    'reign_lastelection',
    'reign_lead_recent',
    'reign_leg_ant',
    'reign_leg_recent',
    'reign_loss',
    'reign_male',
    'reign_militarycareer',
    'reign_nochange_recent',
    'reign_pctile_risk',
    'reign_precip',
    'reign_pt_attempt',
    'reign_pt_suc',
    'reign_ref_ant',
    'reign_ref_recent',
    'reign_tenure_months',
    'reign_victory_recent',
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
    'time_since_splag_1_1_ged_dummy_sb',
    'tlag_12_vdem_v2x_accountability',
    'tlag_12_vdem_v2x_api',
    'tlag_12_vdem_v2x_civlib',
    'tlag_12_vdem_v2x_clphy',
    'tlag_12_vdem_v2x_clpol',
    'tlag_12_vdem_v2x_clpriv',
    'tlag_12_vdem_v2x_corr',
    'tlag_12_vdem_v2x_cspart',
    'tlag_12_vdem_v2x_delibdem',
    'tlag_12_vdem_v2x_diagacc',
    'tlag_12_vdem_v2x_divparctrl',
    'tlag_12_vdem_v2x_edcomp_thick',
    'tlag_12_vdem_v2x_egal',
    'tlag_12_vdem_v2x_egaldem',
    'tlag_12_vdem_v2x_elecoff',
    'tlag_12_vdem_v2x_elecreg',
    'tlag_12_vdem_v2x_ex_confidence',
    'tlag_12_vdem_v2x_ex_direlect',
    'tlag_12_vdem_v2x_ex_hereditary',
    'tlag_12_vdem_v2x_ex_military',
    'tlag_12_vdem_v2x_ex_party',
    'tlag_12_vdem_v2x_execorr',
    'tlag_12_vdem_v2x_feduni',
    'tlag_12_vdem_v2x_frassoc_thick',
    'tlag_12_vdem_v2x_freexp',
    'tlag_12_vdem_v2x_freexp_altinf',
    'tlag_12_vdem_v2x_gencl',
    'tlag_12_vdem_v2x_gencs',
    'tlag_12_vdem_v2x_gender',
    'tlag_12_vdem_v2x_genpp',
    'tlag_12_vdem_v2x_horacc',
    'tlag_12_vdem_v2x_hosabort',
    'tlag_12_vdem_v2x_hosinter',
    'tlag_12_vdem_v2x_jucon',
    'tlag_12_vdem_v2x_legabort',
    'tlag_12_vdem_v2x_libdem',
    'tlag_12_vdem_v2x_liberal',
    'tlag_12_vdem_v2x_mpi',
    'tlag_12_vdem_v2x_neopat',
    'tlag_12_vdem_v2x_partip',
    'tlag_12_vdem_v2x_partipdem',
    'tlag_12_vdem_v2x_polyarchy',
    'tlag_12_vdem_v2x_pubcorr',
    'tlag_12_vdem_v2x_regime',
    'tlag_12_vdem_v2x_regime_amb',
    'tlag_12_vdem_v2x_rule',
    'tlag_12_vdem_v2x_suffr',
    'tlag_12_vdem_v2x_veracc',
    'tlag_12_vdem_v2xcl_acjst',
    'tlag_12_vdem_v2xcl_disc',
    'tlag_12_vdem_v2xcl_dmove',
    'tlag_12_vdem_v2xcl_prpty',
    'tlag_12_vdem_v2xcl_rol',
    'tlag_12_vdem_v2xcl_slave',
    'tlag_12_vdem_v2xcs_ccsi',
    'tlag_12_vdem_v2xdd_cic',
    'tlag_12_vdem_v2xdd_dd',
    'tlag_12_vdem_v2xdd_i_or',
    'tlag_12_vdem_v2xdd_i_pi',
    'tlag_12_vdem_v2xdd_i_pl',
    'tlag_12_vdem_v2xdd_i_rf',
    'tlag_12_vdem_v2xdd_toc',
    'tlag_12_vdem_v2xdl_delib',
    'tlag_12_vdem_v2xeg_eqaccess',
    'tlag_12_vdem_v2xeg_eqdr',
    'tlag_12_vdem_v2xeg_eqprotec',
    'tlag_12_vdem_v2xel_elecparl',
    'tlag_12_vdem_v2xel_elecpres',
    'tlag_12_vdem_v2xel_frefair',
    'tlag_12_vdem_v2xel_locelec',
    'tlag_12_vdem_v2xel_regelec',
    'tlag_12_vdem_v2xex_elecleg',
    'tlag_12_vdem_v2xex_elecreg',
    'tlag_12_vdem_v2xlg_elecreg',
    'tlag_12_vdem_v2xlg_legcon',
    'tlag_12_vdem_v2xlg_leginter',
    'tlag_12_vdem_v2xme_altinf',
    'tlag_12_vdem_v2xnp_client',
    'tlag_12_vdem_v2xnp_pres',
    'tlag_12_vdem_v2xnp_regcorr',
    'tlag_12_vdem_v2xpe_exlecon',
    'tlag_12_vdem_v2xpe_exlgender',
    'tlag_12_vdem_v2xpe_exlgeo',
    'tlag_12_vdem_v2xpe_exlpol',
    'tlag_12_vdem_v2xpe_exlsocgr',
    'tlag_12_vdem_v2xps_party',
    'tlag_1_greq_1_ged_best_ns',
    'tlag_1_greq_1_ged_best_os',
    'tlag_1_greq_1_ged_best_sb',
    'tlag_1_splag_1_1_ged_best_ns',
    'tlag_1_splag_1_1_ged_best_os',
    'tlag_1_splag_1_1_ged_best_sb',
    'tlag_2_greq_1_ged_best_sb',
    'tlag_3_greq_1_ged_best_sb',
    'vdem_v2x_accountability',
    'vdem_v2x_api',
    'vdem_v2x_civlib',
    'vdem_v2x_clphy',
    'vdem_v2x_clpol',
    'vdem_v2x_clpriv',
    'vdem_v2x_corr',
    'vdem_v2x_cspart',
    'vdem_v2x_delibdem',
    'vdem_v2x_diagacc',
    'vdem_v2x_divparctrl',
    'vdem_v2x_edcomp_thick',
    'vdem_v2x_egal',
    'vdem_v2x_egaldem',
    'vdem_v2x_elecoff',
    'vdem_v2x_elecreg',
    'vdem_v2x_ex_confidence',
    'vdem_v2x_ex_direlect',
    'vdem_v2x_ex_hereditary',
    'vdem_v2x_ex_military',
    'vdem_v2x_ex_party',
    'vdem_v2x_execorr',
    'vdem_v2x_feduni',
    'vdem_v2x_frassoc_thick',
    'vdem_v2x_freexp',
    'vdem_v2x_freexp_altinf',
    'vdem_v2x_gencl',
    'vdem_v2x_gencs',
    'vdem_v2x_gender',
    'vdem_v2x_genpp',
    'vdem_v2x_horacc',
    'vdem_v2x_hosabort',
    'vdem_v2x_hosinter',
    'vdem_v2x_jucon',
    'vdem_v2x_legabort',
    'vdem_v2x_libdem',
    'vdem_v2x_liberal',
    'vdem_v2x_mpi',
    'vdem_v2x_neopat',
    'vdem_v2x_partip',
    'vdem_v2x_partipdem',
    'vdem_v2x_polyarchy',
    'vdem_v2x_pubcorr',
    'vdem_v2x_regime',
    'vdem_v2x_regime_amb',
    'vdem_v2x_rule',
    'vdem_v2x_suffr',
    'vdem_v2x_veracc',
    'vdem_v2xcl_acjst',
    'vdem_v2xcl_disc',
    'vdem_v2xcl_dmove',
    'vdem_v2xcl_prpty',
    'vdem_v2xcl_rol',
    'vdem_v2xcl_slave',
    'vdem_v2xcs_ccsi',
    'vdem_v2xdd_cic',
    'vdem_v2xdd_dd',
    'vdem_v2xdd_i_or',
    'vdem_v2xdd_i_pi',
    'vdem_v2xdd_i_pl',
    'vdem_v2xdd_i_rf',
    'vdem_v2xdd_toc',
    'vdem_v2xdl_delib',
    'vdem_v2xeg_eqaccess',
    'vdem_v2xeg_eqdr',
    'vdem_v2xeg_eqprotec',
    'vdem_v2xel_elecparl',
    'vdem_v2xel_elecpres',
    'vdem_v2xel_frefair',
    'vdem_v2xel_locelec',
    'vdem_v2xel_regelec',
    'vdem_v2xex_elecleg',
    'vdem_v2xex_elecreg',
    'vdem_v2xlg_elecreg',
    'vdem_v2xlg_legcon',
    'vdem_v2xlg_leginter',
    'vdem_v2xme_altinf',
    'vdem_v2xnp_client',
    'vdem_v2xnp_pres',
    'vdem_v2xnp_regcorr',
    'vdem_v2xpe_exlecon',
    'vdem_v2xpe_exlgender',
    'vdem_v2xpe_exlgeo',
    'vdem_v2xpe_exlpol',
    'vdem_v2xpe_exlsocgr',
    'vdem_v2xps_party',
    'wdi_ag_lnd_agri_zs',
    'wdi_ag_lnd_arbl_zs',
    'wdi_ag_lnd_frst_k2',
    'wdi_ag_lnd_prcp_mm',
    'wdi_ag_lnd_totl_k2',
    'wdi_ag_lnd_totl_ru_k2',
    'wdi_ag_prd_crop_xd',
    'wdi_ag_prd_food_xd',
    'wdi_ag_prd_lvsk_xd',
    'wdi_ag_srf_totl_k2',
    'wdi_ag_yld_crel_kg',
    'wdi_bg_gsr_nfsv_gd_zs',
    'wdi_bm_klt_dinv_wd_gd_zs',
    'wdi_bn_cab_xoka_gd_zs',
    'wdi_bx_gsr_ccis_zs',
    'wdi_bx_gsr_cmcp_zs',
    'wdi_bx_gsr_insf_zs',
    'wdi_bx_gsr_mrch_cd',
    'wdi_bx_gsr_tran_zs',
    'wdi_bx_gsr_trvl_zs',
    'wdi_bx_klt_dinv_cd_wd',
    'wdi_bx_klt_dinv_wd_gd_zs',
    'wdi_bx_trf_pwkr_dt_gd_zs',
    'wdi_dt_dod_dect_gn_zs',
    'wdi_dt_dod_pvlx_gn_zs',
    'wdi_dt_oda_oatl_kd',
    'wdi_dt_oda_odat_gn_zs',
    'wdi_dt_oda_odat_pc_zs',
    'wdi_dt_tds_dect_gn_zs',
    'wdi_eg_elc_accs_zs',
    'wdi_eg_use_elec_kh_pc',
    'wdi_eg_use_pcap_kg_oe',
    'wdi_en_pop_slum_ur_zs',
    'wdi_en_urb_mcty_tl_zs',
    'wdi_ep_pmp_desl_cd',
    'wdi_ep_pmp_sgas_cd',
    'wdi_fp_cpi_totl',
    'wdi_fr_inr_dpst',
    'wdi_fr_inr_lndp',
    'wdi_gc_dod_totl_gd_zs',
    'wdi_ic_bus_ease_xq',
    'wdi_iq_cpa_econ_xq',
    'wdi_iq_cpa_fisp_xq',
    'wdi_iq_cpa_gndr_xq',
    'wdi_iq_cpa_macr_xq',
    'wdi_iq_cpa_prop_xq',
    'wdi_iq_cpa_pubs_xq',
    'wdi_iq_cpa_soci_xq',
    'wdi_iq_cpa_trad_xq',
    'wdi_iq_cpa_tran_xq',
    'wdi_ne_con_prvt_pc_kd_zg',
    'wdi_ne_dab_totl_kd',
    'wdi_ne_dab_totl_zs',
    'wdi_ne_exp_gnfs_zs',
    'wdi_ne_gdi_totl_zs',
    'wdi_ne_imp_gnfs_kd',
    'wdi_ne_imp_gnfs_kd_zg',
    'wdi_ne_imp_gnfs_zs',
    'wdi_ne_rsb_gnfs_zs',
    'wdi_ne_trd_gnfs_zs',
    'wdi_nv_agr_empl_kd',
    'wdi_nv_agr_totl_cd',
    'wdi_nv_agr_totl_cn',
    'wdi_nv_agr_totl_kd',
    'wdi_nv_agr_totl_kd_zg',
    'wdi_nv_agr_totl_kn',
    'wdi_nv_agr_totl_zs',
    'wdi_nv_ind_empl_kd',
    'wdi_nv_ind_manf_cd',
    'wdi_nv_ind_manf_cn',
    'wdi_nv_ind_manf_kd',
    'wdi_nv_ind_manf_kd_zg',
    'wdi_nv_ind_manf_kn',
    'wdi_nv_ind_manf_zs',
    'wdi_nv_ind_totl_cd',
    'wdi_nv_ind_totl_cn',
    'wdi_nv_ind_totl_kd',
    'wdi_nv_ind_totl_kd_zg',
    'wdi_nv_ind_totl_kn',
    'wdi_nv_ind_totl_zs',
    'wdi_nv_mnf_chem_zs_un',
    'wdi_nv_mnf_fbto_zs_un',
    'wdi_nv_mnf_mtrn_zs_un',
    'wdi_nv_mnf_othr_zs_un',
    'wdi_nv_mnf_tech_zs_un',
    'wdi_nv_mnf_txtl_zs_un',
    'wdi_nv_srv_empl_kd',
    'wdi_nv_srv_totl_cd',
    'wdi_nv_srv_totl_cn',
    'wdi_nv_srv_totl_kd',
    'wdi_nv_srv_totl_kd_zg',
    'wdi_nv_srv_totl_kn',
    'wdi_nv_srv_totl_zs',
    'wdi_ny_adj_dfor_cd',
    'wdi_ny_adj_dmin_gn_zs',
    'wdi_ny_adj_dres_gn_zs',
    'wdi_ny_adj_ictr_gn_zs',
    'wdi_ny_adj_nnty_kd',
    'wdi_ny_adj_nnty_kd_zg',
    'wdi_ny_gdp_coal_rt_zs',
    'wdi_ny_gdp_defl_kd_zg',
    'wdi_ny_gdp_defl_kd_zg_ad',
    'wdi_ny_gdp_defl_zs',
    'wdi_ny_gdp_defl_zs_ad',
    'wdi_ny_gdp_disc_cn',
    'wdi_ny_gdp_disc_kn',
    'wdi_ny_gdp_fcst_cd',
    'wdi_ny_gdp_fcst_cn',
    'wdi_ny_gdp_fcst_kd',
    'wdi_ny_gdp_fcst_kn',
    'wdi_ny_gdp_frst_rt_zs',
    'wdi_ny_gdp_minr_rt_zs',
    'wdi_ny_gdp_mktp_cd',
    'wdi_ny_gdp_mktp_cn',
    'wdi_ny_gdp_mktp_cn_ad',
    'wdi_ny_gdp_mktp_kd',
    'wdi_ny_gdp_mktp_kd_zg',
    'wdi_ny_gdp_mktp_kn',
    'wdi_ny_gdp_mktp_pp_cd',
    'wdi_ny_gdp_mktp_pp_kd',
    'wdi_ny_gdp_ngas_rt_zs',
    'wdi_ny_gdp_pcap_cd',
    'wdi_ny_gdp_pcap_cn',
    'wdi_ny_gdp_pcap_kd',
    'wdi_ny_gdp_pcap_kd_zg',
    'wdi_ny_gdp_pcap_kn',
    'wdi_ny_gdp_pcap_pp_cd',
    'wdi_ny_gdp_pcap_pp_kd',
    'wdi_ny_gdp_petr_rt_zs',
    'wdi_ny_gdp_totl_rt_zs',
    'wdi_ny_gnp_mktp_kd',
    'wdi_ny_gnp_mktp_pp_kd',
    'wdi_per_si_allsi_cov_pop_tot',
    'wdi_per_si_allsi_cov_q1_tot',
    'wdi_per_si_allsi_cov_q2_tot',
    'wdi_per_si_allsi_cov_q3_tot',
    'wdi_per_si_allsi_cov_q4_tot',
    'wdi_per_si_allsi_cov_q5_tot',
    'wdi_se_adt_1524_lt_fe_zs',
    'wdi_se_adt_1524_lt_ma_zs',
    'wdi_se_adt_1524_lt_zs',
    'wdi_se_adt_litr_fe_zs',
    'wdi_se_adt_litr_ma_zs',
    'wdi_se_adt_litr_zs',
    'wdi_se_enr_prim_fm_zs',
    'wdi_se_enr_prsc_fm_zs',
    'wdi_se_prm_cmpt_zs',
    'wdi_se_prm_cuat_fe_zs',
    'wdi_se_prm_cuat_ma_zs',
    'wdi_se_prm_cuat_zs',
    'wdi_se_prm_enrr',
    'wdi_se_prm_nenr',
    'wdi_se_prm_tenr_fe',
    'wdi_se_prm_tenr_ma',
    'wdi_se_sec_cmpt_lo_zs',
    'wdi_se_sec_cuat_lo_fe_zs',
    'wdi_se_sec_cuat_lo_ma_zs',
    'wdi_se_sec_cuat_lo_zs',
    'wdi_se_sec_nenr',
    'wdi_se_ter_cuat_do_fe_zs',
    'wdi_se_ter_cuat_do_ma_zs',
    'wdi_se_ter_cuat_do_zs',
    'wdi_sg_gen_parl_zs',
    'wdi_sg_vaw_reas_zs',
    'wdi_sh_dyn_0514',
    'wdi_sh_dyn_mort',
    'wdi_sh_dyn_mort_fe',
    'wdi_sh_dyn_mort_ma',
    'wdi_sh_h2o_basw_ru_zs',
    'wdi_sh_h2o_basw_ur_zs',
    'wdi_sh_h2o_basw_zs',
    'wdi_sh_mmr_risk_zs',
    'wdi_sh_sta_bass_ru_zs',
    'wdi_sh_sta_bass_ur_zs',
    'wdi_sh_sta_bass_zs',
    'wdi_sh_sta_maln_fe_zs',
    'wdi_sh_sta_maln_ma_zs',
    'wdi_sh_sta_maln_zs',
    'wdi_sh_sta_mmrt',
    'wdi_sh_sta_mmrt_ne',
    'wdi_sh_sta_stnt_fe_zs',
    'wdi_sh_sta_stnt_ma_zs',
    'wdi_sh_sta_stnt_zs',
    'wdi_sh_sta_traf_p5',
    'wdi_sh_sta_wash_p5',
    'wdi_sh_svr_wast_fe_zs',
    'wdi_sh_svr_wast_ma_zs',
    'wdi_sh_svr_wast_zs',
    'wdi_si_dst_02nd_20',
    'wdi_si_dst_03rd_20',
    'wdi_si_dst_04th_20',
    'wdi_si_dst_05th_20',
    'wdi_si_dst_10th_10',
    'wdi_si_dst_frst_10',
    'wdi_si_dst_frst_20',
    'wdi_si_pov_dday',
    'wdi_si_pov_gaps',
    'wdi_si_pov_gini',
    'wdi_si_pov_lmic',
    'wdi_si_pov_umic',
    'wdi_sl_agr_empl_ma_zs',
    'wdi_sl_agr_empl_zs',
    'wdi_sl_ind_empl_zs',
    'wdi_sl_srv_empl_zs',
    'wdi_sl_tlf_totl_fe_zs',
    'wdi_sl_uem_advn_fe_zs',
    'wdi_sl_uem_advn_ma_zs',
    'wdi_sl_uem_advn_zs',
    'wdi_sl_uem_neet_fe_zs',
    'wdi_sl_uem_neet_ma_zs',
    'wdi_sl_uem_neet_zs',
    'wdi_sl_uem_totl_fe_zs',
    'wdi_sl_uem_totl_ma_zs',
    'wdi_sl_uem_totl_zs',
    'wdi_sm_pop_netm',
    'wdi_sm_pop_refg',
    'wdi_sm_pop_refg_or',
    'wdi_sm_pop_totl_zs',
    'wdi_sn_itk_defc_zs',
    'wdi_sp_dyn_amrt_fe',
    'wdi_sp_dyn_amrt_ma',
    'wdi_sp_dyn_imrt_fe_in',
    'wdi_sp_dyn_imrt_in',
    'wdi_sp_dyn_imrt_ma_in',
    'wdi_sp_dyn_le00_fe_in',
    'wdi_sp_dyn_le00_in',
    'wdi_sp_dyn_le00_ma_in',
    'wdi_sp_dyn_tfrt_in',
    'wdi_sp_dyn_wfrt',
    'wdi_sp_hou_fema_zs',
    'wdi_sp_pop_0014_fe_zs',
    'wdi_sp_pop_0014_ma_zs',
    'wdi_sp_pop_0014_to_zs',
    'wdi_sp_pop_1564_fe_zs',
    'wdi_sp_pop_1564_ma_zs',
    'wdi_sp_pop_1564_to_zs',
    'wdi_sp_pop_65up_fe_zs',
    'wdi_sp_pop_65up_ma_zs',
    'wdi_sp_pop_65up_to_zs',
    'wdi_sp_pop_dpnd',
    'wdi_sp_pop_dpnd_ol',
    'wdi_sp_pop_dpnd_yg',
    'wdi_sp_pop_grow',
    'wdi_sp_pop_totl',
    'wdi_sp_rur_totl_zg',
    'wdi_sp_rur_totl_zs',
    'wdi_sp_urb_grow',
    'wdi_sp_urb_totl_in_zs',
    'wdi_st_int_arvl',
    'wdi_st_int_rcpt_xp_zs',
    'wdi_tx_val_agri_zs_un',
    'wdi_tx_val_food_zs_un',
    'wdi_tx_val_fuel_zs_un',
    'wdi_tx_val_mmtl_zs_un',
    'wdi_tx_val_tech_mf_zs',
    'wdi_vc_idp_nwcv',
    'wdi_vc_idp_nwds',
    'wdi_vc_idp_tocv',
    'wdi_vc_pkp_totl_un'
]


# In[10]:


cols_pgm = [
    "time_since_ged_dummy_sb",
    "time_since_ged_dummy_ns",
    "time_since_ged_dummy_os",
    "tlag_1_ged_best_ns",
    "tlag_1_ged_best_os",
    "tlag_1_ged_best_sb",
    "time_since_splag_1_1_ged_dummy_sb",
    "time_since_splag_1_1_ged_dummy_ns",
    "time_since_splag_1_1_ged_dummy_os",
    "time_since_greq_25_ged_best_sb",
    "time_since_greq_25_ged_best_ns",
    "time_since_greq_25_ged_best_os",
    "time_since_greq_25_splag_1_1_ged_best_sb",
    "time_since_greq_25_splag_1_1_ged_best_ns",
    "time_since_greq_25_splag_1_1_ged_best_os",
    "time_since_greq_500_ged_best_sb",
    "time_since_greq_500_ged_best_ns",
    "time_since_greq_500_ged_best_os",
    "time_since_greq_500_splag_1_1_ged_best_sb",
    "time_since_greq_500_splag_1_1_ged_best_ns",
    "time_since_greq_500_splag_1_1_ged_best_os",
    "ged_best_sb",
    "ged_best_ns",
    "ged_best_os",
    "pgd_bdist3",
    "pgd_capdist",
    "pgd_agri_ih",
    "pgd_pop_gpw_sum",
    "pgd_ttime_mean",
    "spdist_pgd_diamsec",
    "pgd_pasture_ih",
    "pgd_savanna_ih",
    "pgd_forest_ih",
    "pgd_urban_ih",
    "pgd_barren_ih",
    "pgd_gcp_mer",
    "fvp_ssp2_edu_sec_15_24_prop",
    "ln_fvp_timeindep",
    "ln_fvp_timesincepreindepwar",
    "spdist_pgd_petroleum",
    # "pgd_droughtstart_spi", many years old, dont use
]


# In[11]:


if level == "cm":
    cols_features = cols_cm
elif level =="pgm":
    cols_features = cols_pgm


# In[12]:


# Specify an optional downsampling level
if level=="cm":
    # No downsampling.
    downsampling = api.Downsampling(share_positive=1, share_negative=1, threshold=0)
if level=="pgm":
    downsampling = api.Downsampling(share_positive=1, share_negative=0.01, threshold=0)


# In[13]:


# Specify number of estimators in RF estimator
n_estimators = 200


# In[14]:


# Define the benchmark models.
benchmark_delta = api.Model(
    name="benchmark_delta",                
    col_outcome="ln_ged_best_sb",    
    cols_features=cols_features,     
    steps=steps,                     
    outcome_type="real",             
    periods=periods,                 
    estimator=RandomForestRegressor( 
        n_estimators=n_estimators,
        criterion="mse",
        n_jobs=-1,
    ),
    delta_outcome=True,            
    downsampling=downsampling,
)

models = [benchmark_delta]


# ## Model fit, prediction, and evaluation

# In[15]:


#get_().run_cell_magic('time', '', '# Train all models\nfor model in models:\n    model.fit_estimators(df)')


# In[16]:


# If pred_africa, subset df to Africa
if pred_africa and level=="cm":
    df = df.loc[df.in_africa==1]


# In[17]:


# Store predictions and calibrated predictions for all models in our dataframe
for model in models:
    df_predictions = model.predict(df)
    df = assign_into_df(df, df_predictions)
    df_predictions = model.predict_calibrated(
        df=df,
        period_calib = period_calib,
        period_test = period_test
    )
    df = assign_into_df(df, df_predictions)


# In[18]:


# Save model objects to file
for model in models:
    model.save()


# In[19]:


# Evaluate all models. Scores are stored in the model object
for model in models:
    model.evaluate(df)


# ## Evaluation output

# ### Performance metrics by step

# In[20]:


# Select the partition here.
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


# ### Feature importances

# In[21]:


sort_step = 3
top = 30


# In[22]:


def featimp_by_steps(model, steps, sort_step, top, cols):
    """
    Return pd.DataFrame of top feature importances by selected steps.
    
    Args:
        model: Model to get importances for.
        steps: List of ordered steps to include in table.
        sort_step: Step to sort table by.
        top: Top number of features to include.
        cols: Feature list.
    """
    for step in steps:
        fi_dict = model.extras.feature_importances["test"][step]
        step_df = pd.DataFrame(fi.reorder_fi_dict(fi_dict))
        step_df = step_df.rename(columns={"importance": f"s={step}"})
        step_df.set_index("feature", inplace=True)
        df = df.join(step_df) if step > steps[0] else step_df.copy()

    df = df.sort_values(by=[f"s={sort_step}"], ascending=False)
    df = df[0:top + 1]
    
    return df


# In[23]:


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


# ### Permutation scores

# In[24]:


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


# In[25]:






