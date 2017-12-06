#encoding=utf8

import sys
import pandas as pd
import numpy as np
from datetime import datetime
from fea_utils import *
from sklearn import preprocessing

def gen_fea(user_fn, click_fn, order_fn, loan_fn, fea_fn):

	user_df = pd.read_csv(user_fn)
	clk_df = pd.read_csv(click_fn)
	ord_df = pd.read_csv(order_fn)
	loan_df = pd.read_csv(loan_fn)
	loan_df['real_loan_amount'] = loan_df['loan_amount'].map(lambda la : to_real_loan(la))

	# 生成age表, 该表只有age一列且年龄按照从小到大的顺序排列
	ages = set(user_df.age)
	age_df = pd.DataFrame({'age': list(ages)})
	age_df = age_df.sort_values(['age'])

	# 生成date表, 该表只有date一列且日期按照从小到大的顺序排列
	date_all = loan_df['loan_time'].str.split(' ', expand=True)[0]
	start_date = pd.to_datetime(date_all.min(), format='%Y-%m-%d')
	end_date = pd.to_datetime(date_all.max(), format='%Y-%m-%d')
	date_list = map(lambda d : datetime.strftime(d, '%Y-%m-%d'), pd.date_range(start_date, end_date).tolist())
	date_df = pd.DataFrame(data={'date' : date_list})

	# 生成 age * date 的完全表
	age_df['key'] = 1
	date_df['key'] = 1

	age_date_df = pd.merge(age_df, date_df, on=['key'], how='left')
	del age_date_df['key']

	# 生成 user * date 的完全表
	user_df['key'] = 1

	user_date_df = pd.merge(user_df, date_df, on=['key'], how='left')
	del user_date_df['key']

	# 生成 点击特征数据 ---------------------------------------------------------------
	print(" -- Generating click features : ")

	clk_df['date'] = clk_df['click_time'].map(lambda ct: ct.split(' ')[0])

	user_age_clk_df = pd.merge(user_df[['uid', 'age']], clk_df, on=['uid'])

	age_date_clk_df = pd.DataFrame({'ad_clk_cnt' : user_age_clk_df.groupby(['age', 'date']).size()}).reset_index()

	age_date_fea_df = pd.merge(age_date_df, age_date_clk_df, on=['age', 'date'], how='left')
	age_date_fea_df['ad_clk_cnt'] = age_date_fea_df['ad_clk_cnt'].fillna(value=0)

	gc = age_date_fea_df.groupby('age').ad_clk_cnt

	age_date_fea_df['ad_clk_cnt_3d'] = gc.apply(lambda x: x.rolling(3).sum()).fillna(value=-1)
	age_date_fea_df['ad_clk_cnt_7d'] = gc.apply(lambda x: x.rolling(7).sum()).fillna(value=-1)
	age_date_fea_df['ad_clk_cnt_14d'] = gc.apply(lambda x: x.rolling(14).sum()).fillna(value=-1)
	age_date_fea_df['ad_clk_cnt_21d'] = gc.apply(lambda x: x.rolling(21).sum()).fillna(value=-1)
	age_date_fea_df['ad_clk_cnt_30d'] = gc.apply(lambda x: x.rolling(30).sum()).fillna(value=-1)
	age_date_fea_df['ad_clk_cnt_60d'] = gc.apply(lambda x: x.rolling(60).sum()).fillna(value=-1)
	age_date_fea_df['ad_clk_cnt_90d'] = gc.apply(lambda x: x.rolling(90).sum()).fillna(value=-1)

	print(" ---- Click features have been generated. ")

	# 生成 订单特征数据 ---------------------------------------------------------------

	print(" -- Generating order features : ")

	ord_df['date'] = ord_df['buy_time']

	user_age_ord_df = pd.merge(user_df[['uid', 'age']], ord_df, on=['uid'])

	age_date_ord_df = pd.DataFrame({'ad_ord_cnt' : user_age_ord_df.groupby(['age', 'date']).size()}).reset_index()

	age_date_fea_df = pd.merge(age_date_fea_df, age_date_ord_df, on=['age', 'date'], how='left')
	age_date_fea_df['ad_ord_cnt'] = age_date_fea_df['ad_ord_cnt'].fillna(value=0)

	gc = age_date_fea_df.groupby('age').ad_ord_cnt
	age_date_fea_df['ad_ord_cnt_3d'] = gc.apply(lambda x: x.rolling(3).sum()).fillna(value=-1)
	age_date_fea_df['ad_ord_cnt_7d'] = gc.apply(lambda x: x.rolling(7).sum()).fillna(value=-1)
	age_date_fea_df['ad_ord_cnt_14d'] = gc.apply(lambda x: x.rolling(14).sum()).fillna(value=-1)
	age_date_fea_df['ad_ord_cnt_21d'] = gc.apply(lambda x: x.rolling(21).sum()).fillna(value=-1)
	age_date_fea_df['ad_ord_cnt_30d'] = gc.apply(lambda x: x.rolling(30).sum()).fillna(value=-1)
	age_date_fea_df['ad_ord_cnt_60d'] = gc.apply(lambda x: x.rolling(60).sum()).fillna(value=-1)
	age_date_fea_df['ad_ord_cnt_90d'] = gc.apply(lambda x: x.rolling(90).sum()).fillna(value=-1)

	print(" ---- Order features have been generated. ")

	# 生成 点击率特征数据 --------------------------------------------------------------

	print(" -- Generating ctr features : ")

	age_date_fea_df['ad_ctr'] = (age_date_fea_df['ad_ord_cnt'] + 0.1) / (age_date_fea_df['ad_clk_cnt'] + 0.5)
	age_date_fea_df['ad_ctr_3d'] = (age_date_fea_df['ad_ord_cnt_3d'] + 0.1) / (age_date_fea_df['ad_clk_cnt_3d'] + 0.5)
	age_date_fea_df['ad_ctr_7d'] = (age_date_fea_df['ad_ord_cnt_7d'] + 0.1) / (age_date_fea_df['ad_clk_cnt_7d'] + 0.5)
	age_date_fea_df['ad_ctr_14d'] = (age_date_fea_df['ad_ord_cnt_14d'] + 0.1) / (age_date_fea_df['ad_clk_cnt_14d'] + 0.5)
	age_date_fea_df['ad_ctr_21d'] = (age_date_fea_df['ad_ord_cnt_21d'] + 0.1) / (age_date_fea_df['ad_clk_cnt_21d'] + 0.5)
	age_date_fea_df['ad_ctr_30d'] = (age_date_fea_df['ad_ord_cnt_30d'] + 0.1) / (age_date_fea_df['ad_clk_cnt_30d'] + 0.5)
	age_date_fea_df['ad_ctr_60d'] = (age_date_fea_df['ad_ord_cnt_60d'] + 0.1) / (age_date_fea_df['ad_clk_cnt_60d'] + 0.5)
	age_date_fea_df['ad_ctr_90d'] = (age_date_fea_df['ad_ord_cnt_90d'] + 0.1) / (age_date_fea_df['ad_clk_cnt_90d'] + 0.5)

	age_date_fea_df['ad_ctr'] = age_date_fea_df.apply(lambda x: -1 if x['ad_ord_cnt'] < 0 or x['ad_clk_cnt'] < 0 else x['ad_ctr'], axis=1)
	age_date_fea_df['ad_ctr_3d'] = age_date_fea_df.apply(lambda x: -1 if x['ad_ord_cnt_3d'] < 0 or x['ad_clk_cnt_3d'] < 0 else x['ad_ctr_3d'], axis=1)
	age_date_fea_df['ad_ctr_7d'] = age_date_fea_df.apply(lambda x: -1 if x['ad_ord_cnt_7d'] < 0 or x['ad_clk_cnt_7d'] < 0 else x['ad_ctr_7d'], axis=1)
	age_date_fea_df['ad_ctr_14d'] = age_date_fea_df.apply(lambda x: -1 if x['ad_ord_cnt_14d'] < 0 or x['ad_clk_cnt_14d'] < 0 else x['ad_ctr_14d'], axis=1)
	age_date_fea_df['ad_ctr_21d'] = age_date_fea_df.apply(lambda x: -1 if x['ad_ord_cnt_21d'] < 0 or x['ad_clk_cnt_21d'] < 0 else x['ad_ctr_21d'], axis=1)
	age_date_fea_df['ad_ctr_30d'] = age_date_fea_df.apply(lambda x: -1 if x['ad_ord_cnt_30d'] < 0 or x['ad_clk_cnt_30d'] < 0 else x['ad_ctr_30d'], axis=1)
	age_date_fea_df['ad_ctr_60d'] = age_date_fea_df.apply(lambda x: -1 if x['ad_ord_cnt_60d'] < 0 or x['ad_clk_cnt_60d'] < 0 else x['ad_ctr_60d'], axis=1)
	age_date_fea_df['ad_ctr_90d'] = age_date_fea_df.apply(lambda x: -1 if x['ad_ord_cnt_90d'] < 0 or x['ad_clk_cnt_90d'] < 0 else x['ad_ctr_90d'], axis=1)

	print(" ---- Ctr features have been generated. ")

	# 生成 贷款特征数据 ---------------------------------------------------------------

	print(" -- Generating loan features : ")
	loan_df['date'] = loan_df['loan_time'].map(lambda lt: lt.split(' ')[0])

	user_age_loan_df = pd.merge(user_df[['uid', 'age']], loan_df, on=['uid'])

	age_date_loan_df = pd.DataFrame({'ad_loan' : user_age_loan_df.groupby(['age', 'date'])['real_loan_amount'].sum()}).reset_index()

	age_date_fea_df = pd.merge(age_date_fea_df, age_date_loan_df, on=['age', 'date'], how='left')
	age_date_fea_df['ad_loan'] = age_date_fea_df['ad_loan'].fillna(value=0)

	gc = age_date_fea_df.groupby('age').ad_loan

	age_date_fea_df['ad_loan_norm_3d'] = gc.apply(lambda x: x.rolling(3).sum()).fillna(value=-1).map(lambda loan_amount: to_norm_loan(loan_amount))
	age_date_fea_df['ad_loan_norm_7d'] = gc.apply(lambda x: x.rolling(7).sum()).fillna(value=-1).map(lambda loan_amount: to_norm_loan(loan_amount))
	age_date_fea_df['ad_loan_norm_14d'] = gc.apply(lambda x: x.rolling(14).sum()).fillna(value=-1).map(lambda loan_amount: to_norm_loan(loan_amount))
	age_date_fea_df['ad_loan_norm_21d'] = gc.apply(lambda x: x.rolling(21).sum()).fillna(value=-1).map(lambda loan_amount: to_norm_loan(loan_amount))
	age_date_fea_df['ad_loan_norm_30d'] = gc.apply(lambda x: x.rolling(30).sum()).fillna(value=-1).map(lambda loan_amount: to_norm_loan(loan_amount))
	age_date_fea_df['ad_loan_norm_60d'] = gc.apply(lambda x: x.rolling(60).sum()).fillna(value=-1).map(lambda loan_amount: to_norm_loan(loan_amount))
	age_date_fea_df['ad_loan_norm_90d'] = gc.apply(lambda x: x.rolling(90).sum()).fillna(value=-1).map(lambda loan_amount: to_norm_loan(loan_amount))
	age_date_fea_df['ad_loan_norm'] = age_date_fea_df['ad_loan'].map(lambda loan_amount: to_norm_loan(loan_amount))

	del age_date_fea_df['ad_loan']

	print(" ---- Loan features have been generated. ")

	# 构建user * date的结果表

	print(" -- age * date features are being written into file ... ")

	df = pd.merge(user_date_df[['uid', 'age', 'date']], age_date_fea_df, on=['age', 'date'])
	
	df = df[['uid', 'date', 'ad_clk_cnt', 'ad_clk_cnt_3d', 'ad_clk_cnt_7d', 'ad_clk_cnt_14d', 'ad_clk_cnt_21d', 'ad_clk_cnt_30d', 'ad_clk_cnt_60d', 
			'ad_clk_cnt_90d', 'ad_ord_cnt', 'ad_ord_cnt_3d', 'ad_ord_cnt_7d', 'ad_ord_cnt_14d', 'ad_ord_cnt_21d', 'ad_ord_cnt_30d', 'ad_ord_cnt_60d',
			'ad_ord_cnt_90d', 'ad_ctr', 'ad_ctr_3d', 'ad_ctr_7d', 'ad_ctr_14d', 'ad_ctr_21d', 'ad_ctr_30d', 'ad_ctr_60d', 'ad_ctr_90d', 'ad_loan_norm',
			'ad_loan_norm_3d', 'ad_loan_norm_7d', 'ad_loan_norm_14d', 'ad_loan_norm_21d', 'ad_loan_norm_30d', 'ad_loan_norm_60d', 'ad_loan_norm_90d']]

	df.to_csv(fea_fn, index=False)

	print(" -- age * date features' file has been generated! ")

def merge(df_user_date, df_age_date):
	res = pd.merge(df_user_date, df_age_date, on=['uid', 'date'])
	return res

if __name__ == '__main__':

	st = datetime.now()

	user_fn = '../../dataset/t_user.csv'
	click_fn = '../../dataset/t_click.csv'
	order_fn = '../../dataset/t_order.csv'
	loan_fn = '../../dataset/t_loan.csv'
	fea_fn = '../../fea/fea_age_date.csv'

	if len(sys.argv) != 6:
		print sys.argv[0] + '\t[user_fn]\t[click_fn]\t[order_fn]\t[loan_fn]\t[fea_fn]'
	else:
		user_fn = sys.argv[1]
		click_fn = sys.argv[2]
		order_fn = sys.argv[3]
		loan_fn = sys.argv[4]
		fea_fn = sys.argv[5]

    # 生成 age * date 维度的特征, 并存放为fea_fn文件
	gen_fea(user_fn, click_fn, order_fn, loan_fn, fea_fn)

	et = datetime.now()

	print 'time cost : ' + str(et - st)