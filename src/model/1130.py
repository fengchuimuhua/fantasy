import xgboost
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from feature import *
from label import user_date_label

fea_user_fn = '../../fea/fea_user.csv'
fea_user_date_fn = '../../fea/fea_user_date.csv'
fea_limit_date_fn = '../../fea/fea_limit_date.csv'
fea_user_month_fn = '../../fea/fea_user_month.csv'
label_user_date_fn = '../../label/user_date_label.csv'

user_df = pd.read_csv(fea_user_fn)
user_date_df = pd.read_csv(fea_user_date_fn)
limit_date_df = pd.read_csv(fea_limit_date_fn)
user_month_df = pd.read_csv(fea_user_month_fn)
user_date_label_df = pd.read_csv(label_user_date_fn)

user_date_df = user_dim.merge(user_df, user_date_df)
user_date_df = limit_date_dim.merge(limit_date_df, user_date_df)
user_date_df = user_month_dim.merge(user_month_df, user_date_df)
user_date_df = user_date_label.merge(user_date_label_df, user_date_df)

feature_col_list = ['u_age', 'u_sex', 'u_limit', 'uid', 'date', 'ud_active_days', 'ud_clk_cnt_1d', 'ud_clk_cnt_3d', 'ud_clk_cnt_7d', 'ud_clk_cnt_14d', 'ud_clk_cnt_21d', 'ud_clk_cnt_30d', 'ud_clk_cnt_60d', 'ud_clk_cnt_90d', 'ud_ord_cnt_1d', 'ud_ord_cnt_3d', 'ud_ord_cnt_7d', 'ud_ord_cnt_14d', 'ud_ord_cnt_21d', 'ud_ord_cnt_30d', 'ud_ord_cnt_60d', 'ud_ord_cnt_90d', 'ud_ctr_1d', 'ud_ctr_3d', 'ud_ctr_7d', 'ud_ctr_14d', 'ud_ctr_21d', 'ud_ctr_30d', 'ud_ctr_60d', 'ud_ctr_90d', 'ud_loan_1d', 'ud_loan_1d', 'ud_loan_3d', 'ud_loan_7d', 'ud_loan_14d', 'ud_loan_21d', 'ud_loan_30d', 'ud_loan_60d', 'ud_loan_90d', 'ld_limit_cat', 'ld_loan_1d', 'ld_loan_3d', 'ld_loan_7d', 'ld_loan_14d', 'ld_loan_21d', 'ld_loan_30d', 'ld_loan_60d', 'ld_loan_90d', 'um_left_plannum','um_left_balance','um_month_need_pay','um_tuned_limit']
dtrain = user_date_df[feature_col_list][user_date_df['date'] <= '2016-10-29'].as_matrix()
dtest = user_date_df[feature_col_list][user_date_df['date'] == '2016-10-30'].as_matrix()
