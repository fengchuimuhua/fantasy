#encoding=utf-8
import common
import xgboost as xgb
import sys
sys.path.append('/Users/dingxuetao/Desktop/快盘/competition/2017jd/fantasy/src')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from feature import user_dim
from feature import limit_date_dim
from feature import user_month_dim
from feature import age_date_dim
from feature import sex_date_dim
from feature import age_sex_date_dim
from label import user_date_label
from datetime import datetime

st = datetime.now()
fea_user_fn = '../../fea/fea_user.csv'
fea_user_date_fn = '../../fea/fea_user_date.csv'
fea_limit_date_fn = '../../fea/fea_limit_date.csv'
fea_user_month_fn = '../../fea/fea_user_month.csv'
fea_age_date_fn = '../../fea/fea_age_date.csv'
fea_sex_date_fn = '../../fea/fea_sex_date.csv'
fea_age_sex_date_fn = '../../fea/fea_age_sex_date.csv'
label_user_date_fn = '../../label/user_date_label.csv'

user_df = pd.read_csv(fea_user_fn)
user_date_df = pd.read_csv(fea_user_date_fn)
limit_date_df = pd.read_csv(fea_limit_date_fn)
user_month_df = pd.read_csv(fea_user_month_fn)
# age_date_df = pd.read_csv(fea_age_date_fn)
# sex_date_df = pd.read_csv(fea_sex_date_fn)
# age_sex_date_df = pd.read_csv(fea_age_sex_date_fn)
user_date_label_df = pd.read_csv(label_user_date_fn)

print 'step 1. read done'

user_date_df = user_dim.merge(user_df, user_date_df)
user_date_df = limit_date_dim.merge(limit_date_df, user_date_df)
user_date_df = user_month_dim.merge(user_month_df, user_date_df)
# user_date_df = age_date_dim.merge(age_date_df, user_date_df)
# user_date_df = sex_date_dim.merge(sex_date_df, user_date_df)
# user_date_df = age_sex_date_dim.merge(age_sex_date_df, user_date_df)
user_date_df = user_date_label.merge(user_date_label_df, user_date_df)

print 'step 2. join data done'

feature_col_list = ['u_age', 'u_sex', 'u_limit', 'ud_active_days', 'ud_clk_cnt_1d', 'ud_clk_cnt_3d', 'ud_clk_cnt_7d', 'ud_clk_cnt_14d', 'ud_clk_cnt_21d', 'ud_clk_cnt_30d', 'ud_clk_cnt_60d', 'ud_clk_cnt_90d', 'ud_ord_cnt_1d', 'ud_ord_cnt_3d', 'ud_ord_cnt_7d', 'ud_ord_cnt_14d', 'ud_ord_cnt_21d', 'ud_ord_cnt_30d', 'ud_ord_cnt_60d', 'ud_ord_cnt_90d', 'ud_ctr_1d', 'ud_ctr_3d', 'ud_ctr_7d', 'ud_ctr_14d', 'ud_ctr_21d', 'ud_ctr_30d', 'ud_ctr_60d', 'ud_ctr_90d', 'ud_loan_1d', 'ud_loan_3d', 'ud_loan_7d', 'ud_loan_14d', 'ud_loan_21d', 'ud_loan_30d', 'ud_loan_60d', 'ud_loan_90d', 'ld_limit_cat', 'ld_loan_1d', 'ld_loan_3d', 'ld_loan_7d', 'ld_loan_14d', 'ld_loan_21d', 'ld_loan_30d', 'ld_loan_60d', 'ld_loan_90d', 'um_left_balance','um_month_need_pay','um_tuned_limit']

# label: is_loan_in_31d, loan_amount_in_31d

# valid set performance
train_x = user_date_df[user_date_df['date'] <= '2016-10-29'][feature_col_list]
train_y = user_date_df[user_date_df['date'] <= '2016-10-29']['loan_amount_in_31d']
valid_x = user_date_df[user_date_df['date'] == '2016-10-30'][feature_col_list]
valid_y = user_date_df[user_date_df['date'] == '2016-10-30']['loan_amount_in_31d']
train = xgb.DMatrix(train_x, label=train_y)
valid = xgb.DMatrix(valid_x, label=valid_y)

watchlist = [(valid, 'valid'), (train, 'train')]
param = {'max_depth': 7, 'eta': 0.2, 'silent': 1, 'objective': 'reg:linear'}
param['eval_metric'] = ['rmse']
evals_result = {}
bst = xgb.train(param, train, 100, watchlist, evals_result=evals_result)


# output prediction file
test_x = user_date_df[user_date_df['date'] == '2016-11-30'][feature_col_list]
test = xgb.DMatrix(test_x)

pred = bst.predict(test)

te = pd.DataFrame({'uid': user_date_df[user_date_df['date'] == '2016-11-30']['uid']})
te['loan_amount_in_31d'] = pred
te.to_csv("pred.csv", index=False)
et = datetime.now()
print 'time cost : ' + str(et - st)
#print sorted(bst1.get_fscore().iteritems(),key=lambda d:d[1], reverse=True)
print sorted(bst.get_fscore().iteritems(),key=lambda d:d[1], reverse=True)
