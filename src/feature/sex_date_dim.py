#encoding=utf8

import sys
import pandas as pd
import numpy as np
from datetime import datetime
from fea_utils import *
from sklearn import preprocessing

def merge(df_user_date, df_sex_date):
	res = pd.merge(df_user_date, df_sex_date, on=['uid', 'date'])
	return res

def gen_fea(user_fn, click_fn, order_fn, loan_fn, fea_fn):

	print("------ sex * date features generator starts  ------")

	user_df = pd.read_csv(user_fn)
	clk_df = pd.read_csv(click_fn)
	ord_df = pd.read_csv(order_fn)
	loan_df = pd.read_csv(loan_fn)
	loan_df['real_loan_amount'] = loan_df['loan_amount'].map(lambda la : to_real_loan(la))

	sexes = set(user_df.sex)
	sex_df = pd.DataFrame({'sex':list(sexes)})
	sex_df = sex_df.sort_values(['sex'])

	date_all = loan_df['loan_time'].str.split(' ', expand=True)[0]
	start_date = pd.to_datetime(date_all.min(), format='%Y-%m-%d')
	end_date = pd.to_datetime(date_all.max(), format='%Y-%m-%d')
	date_list = map(lambda d : datetime.strftime(d, '%Y-%m-%d'), pd.date_range(start_date, end_date).tolist())
	date_df = pd.DataFrame(data={'date' : date_list})

	sex_df['key'] = 1
	date_df['key'] = 1

	sex_date_df = pd.merge(sex_df, date_df, on=['key'], how='left')
	del sex_date_df['key']

	user_df['key'] = 1
	user_date_df = pd.merge(user_df, date_df, on=['key'], how='left')
	del user_date_df['key']

	##################################################################################################################
	#
	# 生成点击特征数据
    ##################################################################################################################

	print(" - Generating click features : ")
	
	clk_df['date'] = clk_df['click_time'].map(lambda ct: ct.split(' ')[0])
	user_sex_clk_df = pd.merge(user_df[['uid', 'sex']], clk_df, on=['uid'])
	sex_date_clk_df = pd.DataFrame({'sd_clk_cnt' : user_sex_clk_df.groupby(['sex', 'date']).size()}).reset_index()
	sex_date_fea_df = pd.merge(sex_date_df, sex_date_clk_df, on=['sex', 'date'], how='left')
	sex_date_fea_df['sd_clk_cnt'] = sex_date_fea_df['sd_clk_cnt'].fillna(value=0)

	gc = sex_date_fea_df.groupby(['sex']).sd_clk_cnt

	sex_date_fea_df['sd_clk_cnt_3d'] = gc.apply(lambda x: x.rolling(3).sum()).fillna(value=-1)
	sex_date_fea_df['sd_clk_cnt_7d'] = gc.apply(lambda x: x.rolling(7).sum()).fillna(value=-1)
	sex_date_fea_df['sd_clk_cnt_14d'] = gc.apply(lambda x: x.rolling(14).sum()).fillna(value=-1)
	sex_date_fea_df['sd_clk_cnt_21d'] = gc.apply(lambda x: x.rolling(21).sum()).fillna(value=-1)
	sex_date_fea_df['sd_clk_cnt_30d'] = gc.apply(lambda x: x.rolling(30).sum()).fillna(value=-1)
	sex_date_fea_df['sd_clk_cnt_60d'] = gc.apply(lambda x: x.rolling(60).sum()).fillna(value=-1)
	sex_date_fea_df['sd_clk_cnt_90d'] = gc.apply(lambda x: x.rolling(90).sum()).fillna(value=-1)				

	print(" ---- Click features have been generated. ")

	##################################################################################################################
	#
	# 生成订单特征数据
    ##################################################################################################################

	print(" - Generating order features : ")

	ord_df['date'] = ord_df['buy_time']
	user_sex_ord_df = pd.merge(user_df[['uid', 'sex']], ord_df, on=['uid'])
	sex_date_ord_df = pd.DataFrame({'sd_ord_cnt' : user_sex_ord_df.groupby(['sex', 'date']).size()}).reset_index()
	sex_date_fea_df = pd.merge(sex_date_fea_df, sex_date_ord_df, on=['sex', 'date'], how='left')
	sex_date_fea_df['sd_ord_cnt'] = sex_date_fea_df['sd_ord_cnt'].fillna(value=0)

	gc = sex_date_fea_df.groupby('sex').sd_ord_cnt

	sex_date_fea_df['sd_ord_cnt_3d'] = gc.apply(lambda x: x.rolling(3).sum()).fillna(value=-1)
	sex_date_fea_df['sd_ord_cnt_7d'] = gc.apply(lambda x: x.rolling(7).sum()).fillna(value=-1)
	sex_date_fea_df['sd_ord_cnt_14d'] = gc.apply(lambda x: x.rolling(14).sum()).fillna(value=-1)
	sex_date_fea_df['sd_ord_cnt_21d'] = gc.apply(lambda x: x.rolling(21).sum()).fillna(value=-1)
	sex_date_fea_df['sd_ord_cnt_30d'] = gc.apply(lambda x: x.rolling(30).sum()).fillna(value=-1)
	sex_date_fea_df['sd_ord_cnt_60d'] = gc.apply(lambda x: x.rolling(60).sum()).fillna(value=-1)
	sex_date_fea_df['sd_ord_cnt_90d'] = gc.apply(lambda x: x.rolling(90).sum()).fillna(value=-1)

	print(" ---- Order features have been generated. ")

	##################################################################################################################
	#
	# 生成点击率特征数据
    ##################################################################################################################

	print(" - Generating click ratio features : ")

	sex_date_fea_df['sd_ctr'] = (sex_date_fea_df['sd_ord_cnt'] + 0.1) / (sex_date_fea_df['sd_clk_cnt'] + 0.5)
	sex_date_fea_df['sd_ctr_3d'] = (sex_date_fea_df['sd_ord_cnt_3d'] + 0.1) / (sex_date_fea_df['sd_clk_cnt_3d'] + 0.5)
	sex_date_fea_df['sd_ctr_7d'] = (sex_date_fea_df['sd_ord_cnt_7d'] + 0.1) / (sex_date_fea_df['sd_clk_cnt_7d'] + 0.5)
	sex_date_fea_df['sd_ctr_14d'] = (sex_date_fea_df['sd_ord_cnt_14d'] + 0.1) / (sex_date_fea_df['sd_clk_cnt_14d'] + 0.5)
	sex_date_fea_df['sd_ctr_21d'] = (sex_date_fea_df['sd_ord_cnt_21d'] + 0.1) / (sex_date_fea_df['sd_clk_cnt_21d'] + 0.5)
	sex_date_fea_df['sd_ctr_30d'] = (sex_date_fea_df['sd_ord_cnt_30d'] + 0.1) / (sex_date_fea_df['sd_clk_cnt_30d'] + 0.5)
	sex_date_fea_df['sd_ctr_60d'] = (sex_date_fea_df['sd_ord_cnt_60d'] + 0.1) / (sex_date_fea_df['sd_clk_cnt_60d'] + 0.5)
	sex_date_fea_df['sd_ctr_90d'] = (sex_date_fea_df['sd_ord_cnt_90d'] + 0.1) / (sex_date_fea_df['sd_clk_cnt_90d'] + 0.5)

	sex_date_fea_df['sd_ctr'] = sex_date_fea_df.apply(lambda x: -1 if x['sd_ord_cnt'] < 0 or x['sd_clk_cnt'] < 0 else x['sd_ctr'], axis=1)
	sex_date_fea_df['sd_ctr_3d'] = sex_date_fea_df.apply(lambda x: -1 if x['sd_ord_cnt_3d'] < 0 or x['sd_clk_cnt_3d'] < 0 else x['sd_ctr_3d'], axis=1)
	sex_date_fea_df['sd_ctr_7d'] = sex_date_fea_df.apply(lambda x: -1 if x['sd_ord_cnt_7d'] < 0 or x['sd_clk_cnt_7d'] < 0 else x['sd_ctr_7d'], axis=1)
	sex_date_fea_df['sd_ctr_14d'] = sex_date_fea_df.apply(lambda x: -1 if x['sd_ord_cnt_14d'] < 0 or x['sd_clk_cnt_14d'] < 0 else x['sd_ctr_14d'], axis=1)
	sex_date_fea_df['sd_ctr_21d'] = sex_date_fea_df.apply(lambda x: -1 if x['sd_ord_cnt_21d'] < 0 or x['sd_clk_cnt_21d'] < 0 else x['sd_ctr_21d'], axis=1)
	sex_date_fea_df['sd_ctr_30d'] = sex_date_fea_df.apply(lambda x: -1 if x['sd_ord_cnt_30d'] < 0 or x['sd_clk_cnt_30d'] < 0 else x['sd_ctr_30d'], axis=1)
	sex_date_fea_df['sd_ctr_60d'] = sex_date_fea_df.apply(lambda x: -1 if x['sd_ord_cnt_60d'] < 0 or x['sd_clk_cnt_60d'] < 0 else x['sd_ctr_60d'], axis=1)
	sex_date_fea_df['sd_ctr_90d'] = sex_date_fea_df.apply(lambda x: -1 if x['sd_ord_cnt_90d'] < 0 or x['sd_clk_cnt_90d'] < 0 else x['sd_ctr_90d'], axis=1)

	print(" ---- Click ratio features have been generated. ")

	##################################################################################################################
	#
	# 生成贷款特征数据
    ##################################################################################################################

	print(" - Generating loan features : ")

	loan_df['date'] = loan_df['loan_time'].map(lambda lt: lt.split(' ')[0])
	user_sex_loan_df = pd.merge(user_df[['uid', 'sex']], loan_df, on=['uid'])
	sex_date_loan_df = pd.DataFrame({'sd_loan' : user_sex_loan_df.groupby(['sex', 'date'])['real_loan_amount'].sum()}).reset_index()

	sex_date_fea_df = pd.merge(sex_date_fea_df, sex_date_loan_df, on=['sex', 'date'], how='left')
	sex_date_fea_df['sd_loan'] = sex_date_fea_df['sd_loan'].fillna(value=0)

	gc = sex_date_fea_df.groupby('sex').sd_loan

	sex_date_fea_df['sd_loan_norm_3d'] = gc.apply(lambda x: x.rolling(3).sum()).fillna(value=-1).map(lambda loan_amount: to_norm_loan(loan_amount))
	sex_date_fea_df['sd_loan_norm_7d'] = gc.apply(lambda x: x.rolling(7).sum()).fillna(value=-1).map(lambda loan_amount: to_norm_loan(loan_amount))
	sex_date_fea_df['sd_loan_norm_14d'] = gc.apply(lambda x: x.rolling(14).sum()).fillna(value=-1).map(lambda loan_amount: to_norm_loan(loan_amount))
	sex_date_fea_df['sd_loan_norm_21d'] = gc.apply(lambda x: x.rolling(21).sum()).fillna(value=-1).map(lambda loan_amount: to_norm_loan(loan_amount))
	sex_date_fea_df['sd_loan_norm_30d'] = gc.apply(lambda x: x.rolling(30).sum()).fillna(value=-1).map(lambda loan_amount: to_norm_loan(loan_amount))
	sex_date_fea_df['sd_loan_norm_60d'] = gc.apply(lambda x: x.rolling(60).sum()).fillna(value=-1).map(lambda loan_amount: to_norm_loan(loan_amount))
	sex_date_fea_df['sd_loan_norm_90d'] = gc.apply(lambda x: x.rolling(90).sum()).fillna(value=-1).map(lambda loan_amount: to_norm_loan(loan_amount))

	sex_date_fea_df['sd_loan_norm'] = sex_date_fea_df['sd_loan'].map(lambda loan_amount: to_norm_loan(loan_amount))

	del sex_date_fea_df['sd_loan']

	print(" ---- Loan features have been generated. ")

	##################################################################################################################
	#
	# 将生成的特征表写入文件中
    ##################################################################################################################

	print(" - Writing features into file ! ")

	df = pd.merge(user_date_df[['uid', 'sex', 'date']], sex_date_fea_df, on=['sex', 'date'])
	df = df[['uid', 'date', 'sd_clk_cnt', 'sd_clk_cnt_3d', 'sd_clk_cnt_7d', 'sd_clk_cnt_14d', 'sd_clk_cnt_21d', 'sd_clk_cnt_30d', 'sd_clk_cnt_60d', 
			'sd_clk_cnt_90d', 'sd_ord_cnt', 'sd_ord_cnt_3d', 'sd_ord_cnt_7d', 'sd_ord_cnt_14d', 'sd_ord_cnt_21d', 'sd_ord_cnt_30d', 'sd_ord_cnt_60d',
			'sd_ord_cnt_90d', 'sd_ctr', 'sd_ctr_3d', 'sd_ctr_7d', 'sd_ctr_14d', 'sd_ctr_21d', 'sd_ctr_30d', 'sd_ctr_60d', 'sd_ctr_90d', 'sd_loan_norm',
			'sd_loan_norm_3d', 'sd_loan_norm_7d', 'sd_loan_norm_14d', 'sd_loan_norm_21d', 'sd_loan_norm_30d', 'sd_loan_norm_60d', 'sd_loan_norm_90d']]
	df.to_csv(fea_fn, index=False)

	print("-------------------- Success ----------------------")

if __name__ == '__main__':

	st = datetime.now()

	user_fn = '../../dataset/t_user.csv'
	click_fn = '../../dataset/t_click.csv'
	order_fn = '../../dataset/t_order.csv'
	loan_fn = '../../dataset/t_loan.csv'
	fea_fn = '../../fea/fea_sex_date.csv'

	if len(sys.argv) != 6:
		print sys.argv[0] + '\t[user_fn]\t[click_fn]\t[order_fn]\t[loan_fn]\t[fea_fn]'
	else:
		user_fn = sys.argv[1]
		click_fn = sys.argv[2]
		order_fn = sys.argv[3]
		loan_fn = sys.argv[4]
		fea_fn = sys.argv[5]

    # 生成 sex * date 维度的特征, 并存放为fea_fn文件
	gen_fea(user_fn, click_fn, order_fn, loan_fn, fea_fn)

	et = datetime.now()

	print 'time cost : ' + str(et - st)
