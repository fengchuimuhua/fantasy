#encoding=utf8

import sys
import pandas as pd
import numpy as np
from datetime import datetime
from fea_utils import *
from sklearn import preprocessing

def merge(df_user_date, df_age_sex_date):
	res = pd.merge(df_user_date, df_age_sex_date, on=['uid', 'date'])
	return res

def gen_fea(user_fn, click_fn, order_fn, loan_fn, fea_fn):

	print("------ age * sex * date features generator starts  ------")

	user_df = pd.read_csv(user_fn)
	clk_df = pd.read_csv(click_fn)
	ord_df = pd.read_csv(order_fn)
	loan_df = pd.read_csv(loan_fn)
	loan_df['real_loan_amount'] = loan_df['loan_amount'].map(lambda la : to_real_loan(la))

	ages = set(user_df.age)
	age_df = pd.DataFrame({'age': list(ages)})
	age_df = age_df.sort_values(['age'])

	sexes = set(user_df.sex)
	sex_df = pd.DataFrame({'sex':list(sexes)})
	sex_df = sex_df.sort_values(['sex'])

	date_all = loan_df['loan_time'].str.split(' ', expand=True)[0]
	start_date = pd.to_datetime(date_all.min(), format='%Y-%m-%d')
	end_date = pd.to_datetime(date_all.max(), format='%Y-%m-%d')
	date_list = map(lambda d : datetime.strftime(d, '%Y-%m-%d'), pd.date_range(start_date, end_date).tolist())
	date_df = pd.DataFrame(data={'date' : date_list})

	age_df['key'] = 1
	sex_df['key'] = 1
	date_df['key'] = 1

	age_sex_df = pd.merge(age_df, sex_df, on=['key'], how='left')
	age_sex_date_df = pd.merge(age_sex_df, date_df, on=['key'], how='left')
	del age_sex_date_df['key']

	user_df['key'] = 1
	user_date_df = pd.merge(user_df, date_df, on=['key'], how='left')
	del user_date_df['key']

	##################################################################################################################
	#
	# 生成 age * sex * date 维度的点击特征数据
    ##################################################################################################################

	print(" - Generating click features : ")

	clk_df['date'] = clk_df['click_time'].map(lambda ct: ct.split(' ')[0])
	user_age_sex_clk_df = pd.merge(user_df[['uid', 'age', 'sex']], clk_df, on=['uid'])
	age_sex_date_clk_df = pd.DataFrame({'asd_clk_cnt' : user_age_sex_clk_df.groupby(['age', 'sex', 'date']).size()}).reset_index()

	age_sex_date_fea_df = pd.merge(age_sex_date_df, age_sex_date_clk_df, on=['age', 'sex', 'date'], how='left')
	age_sex_date_fea_df['asd_clk_cnt'] = age_sex_date_fea_df['asd_clk_cnt'].fillna(value=0)

	gc = age_sex_date_fea_df.groupby(['age', 'sex']).asd_clk_cnt

	age_sex_date_fea_df['asd_clk_cnt_3d'] = gc.apply(lambda x: x.rolling(3).sum()).fillna(value=-1)
	age_sex_date_fea_df['asd_clk_cnt_7d'] = gc.apply(lambda x: x.rolling(7).sum()).fillna(value=-1)
	age_sex_date_fea_df['asd_clk_cnt_14d'] = gc.apply(lambda x: x.rolling(14).sum()).fillna(value=-1)
	age_sex_date_fea_df['asd_clk_cnt_21d'] = gc.apply(lambda x: x.rolling(21).sum()).fillna(value=-1)
	age_sex_date_fea_df['asd_clk_cnt_30d'] = gc.apply(lambda x: x.rolling(30).sum()).fillna(value=-1)
	age_sex_date_fea_df['asd_clk_cnt_60d'] = gc.apply(lambda x: x.rolling(60).sum()).fillna(value=-1)
	age_sex_date_fea_df['asd_clk_cnt_90d'] = gc.apply(lambda x: x.rolling(90).sum()).fillna(value=-1)

	print(" ---- Click features have been generated. ")

	##################################################################################################################
	#
	# 生成 age * sex * date 维度的订单特征数据
    ##################################################################################################################

	print(" - Generating order features : ")

	ord_df['date'] = ord_df['buy_time']
	user_age_sex_ord_df = pd.merge(user_df[['uid', 'age', 'sex']], ord_df, on=['uid'])

	age_sex_date_ord_df = pd.DataFrame({'asd_ord_cnt' : user_age_sex_ord_df.groupby(['age', 'sex', 'date']).size()}).reset_index()

	age_sex_date_fea_df = pd.merge(age_sex_date_fea_df, age_sex_date_ord_df, on=['age', 'sex', 'date'], how='left')
	age_sex_date_fea_df['asd_ord_cnt'] = age_sex_date_fea_df['asd_ord_cnt'].fillna(value=0)

	gc = age_sex_date_fea_df.groupby(['age', 'sex']).asd_ord_cnt

	age_sex_date_fea_df['asd_ord_cnt_3d'] = gc.apply(lambda x: x.rolling(3).sum()).fillna(value=-1)
	age_sex_date_fea_df['asd_ord_cnt_7d'] = gc.apply(lambda x: x.rolling(7).sum()).fillna(value=-1)
	age_sex_date_fea_df['asd_ord_cnt_14d'] = gc.apply(lambda x: x.rolling(14).sum()).fillna(value=-1)
	age_sex_date_fea_df['asd_ord_cnt_21d'] = gc.apply(lambda x: x.rolling(21).sum()).fillna(value=-1)
	age_sex_date_fea_df['asd_ord_cnt_30d'] = gc.apply(lambda x: x.rolling(30).sum()).fillna(value=-1)
	age_sex_date_fea_df['asd_ord_cnt_60d'] = gc.apply(lambda x: x.rolling(60).sum()).fillna(value=-1)
	age_sex_date_fea_df['asd_ord_cnt_90d'] = gc.apply(lambda x: x.rolling(90).sum()).fillna(value=-1)

	print(" ---- Order features have been generated. ")

	##################################################################################################################
	#
	# 生成 age * sex * date 维度的点击率特征数据
    ##################################################################################################################

	print(" - Generating click ratio features : ")

	age_sex_date_fea_df['asd_ctr'] = (age_sex_date_fea_df['asd_ord_cnt'] + 0.1) / (age_sex_date_fea_df['asd_clk_cnt'] + 0.5)
	age_sex_date_fea_df['asd_ctr_3d'] = (age_sex_date_fea_df['asd_ord_cnt_3d'] + 0.1) / (age_sex_date_fea_df['asd_clk_cnt_3d'] + 0.5)
	age_sex_date_fea_df['asd_ctr_7d'] = (age_sex_date_fea_df['asd_ord_cnt_7d'] + 0.1) / (age_sex_date_fea_df['asd_clk_cnt_7d'] + 0.5)
	age_sex_date_fea_df['asd_ctr_14d'] = (age_sex_date_fea_df['asd_ord_cnt_14d'] + 0.1) / (age_sex_date_fea_df['asd_clk_cnt_14d'] + 0.5)
	age_sex_date_fea_df['asd_ctr_21d'] = (age_sex_date_fea_df['asd_ord_cnt_21d'] + 0.1) / (age_sex_date_fea_df['asd_clk_cnt_21d'] + 0.5)
	age_sex_date_fea_df['asd_ctr_30d'] = (age_sex_date_fea_df['asd_ord_cnt_30d'] + 0.1) / (age_sex_date_fea_df['asd_clk_cnt_30d'] + 0.5)
	age_sex_date_fea_df['asd_ctr_60d'] = (age_sex_date_fea_df['asd_ord_cnt_60d'] + 0.1) / (age_sex_date_fea_df['asd_clk_cnt_60d'] + 0.5)
	age_sex_date_fea_df['asd_ctr_90d'] = (age_sex_date_fea_df['asd_ord_cnt_90d'] + 0.1) / (age_sex_date_fea_df['asd_clk_cnt_90d'] + 0.5)

	age_sex_date_fea_df['asd_ctr'] = age_sex_date_fea_df.apply(lambda x: -1 if x['asd_ord_cnt'] < 0 or x['asd_clk_cnt'] < 0 else x['asd_ctr'], axis=1)
	age_sex_date_fea_df['asd_ctr_3d'] = age_sex_date_fea_df.apply(lambda x: -1 if x['asd_ord_cnt_3d'] < 0 or x['asd_clk_cnt_3d'] < 0 else x['asd_ctr_3d'], axis=1)
	age_sex_date_fea_df['asd_ctr_7d'] = age_sex_date_fea_df.apply(lambda x: -1 if x['asd_ord_cnt_7d'] < 0 or x['asd_clk_cnt_7d'] < 0 else x['asd_ctr_7d'], axis=1)
	age_sex_date_fea_df['asd_ctr_14d'] = age_sex_date_fea_df.apply(lambda x: -1 if x['asd_ord_cnt_14d'] < 0 or x['asd_clk_cnt_14d'] < 0 else x['asd_ctr_14d'], axis=1)
	age_sex_date_fea_df['asd_ctr_21d'] = age_sex_date_fea_df.apply(lambda x: -1 if x['asd_ord_cnt_21d'] < 0 or x['asd_clk_cnt_21d'] < 0 else x['asd_ctr_21d'], axis=1)
	age_sex_date_fea_df['asd_ctr_30d'] = age_sex_date_fea_df.apply(lambda x: -1 if x['asd_ord_cnt_30d'] < 0 or x['asd_clk_cnt_30d'] < 0 else x['asd_ctr_30d'], axis=1)
	age_sex_date_fea_df['asd_ctr_60d'] = age_sex_date_fea_df.apply(lambda x: -1 if x['asd_ord_cnt_60d'] < 0 or x['asd_clk_cnt_60d'] < 0 else x['asd_ctr_60d'], axis=1)
	age_sex_date_fea_df['asd_ctr_90d'] = age_sex_date_fea_df.apply(lambda x: -1 if x['asd_ord_cnt_90d'] < 0 or x['asd_clk_cnt_90d'] < 0 else x['asd_ctr_90d'], axis=1)

	print(" ---- Click ratio features have been generated. ")

	##################################################################################################################
	#
	# 生成 age * sex * date 维度的贷款特征数据
    ##################################################################################################################

	print(" - Generating loan features : ")

	loan_df['date'] = loan_df['loan_time'].map(lambda lt: lt.split(' ')[0])
	user_age_sex_loan_df = pd.merge(user_df[['uid', 'age', 'sex']], loan_df, on=['uid'])
	age_sex_date_loan_df = pd.DataFrame({'asd_loan' : user_age_sex_loan_df.groupby(['age', 'sex', 'date'])['real_loan_amount'].sum()}).reset_index()

	age_sex_date_fea_df = pd.merge(age_sex_date_fea_df, age_sex_date_loan_df, on=['age', 'sex', 'date'], how='left')
	age_sex_date_fea_df['asd_loan'] = age_sex_date_fea_df['asd_loan'].fillna(value=0)

	gc = age_sex_date_fea_df.groupby(['age', 'sex']).asd_loan

	age_sex_date_fea_df['asd_loan_norm_3d'] = gc.apply(lambda x: x.rolling(3).sum()).fillna(value=-1).map(lambda loan_amount: to_norm_loan(loan_amount))
	age_sex_date_fea_df['asd_loan_norm_7d'] = gc.apply(lambda x: x.rolling(7).sum()).fillna(value=-1).map(lambda loan_amount: to_norm_loan(loan_amount))
	age_sex_date_fea_df['asd_loan_norm_14d'] = gc.apply(lambda x: x.rolling(14).sum()).fillna(value=-1).map(lambda loan_amount: to_norm_loan(loan_amount))
	age_sex_date_fea_df['asd_loan_norm_21d'] = gc.apply(lambda x: x.rolling(21).sum()).fillna(value=-1).map(lambda loan_amount: to_norm_loan(loan_amount))
	age_sex_date_fea_df['asd_loan_norm_30d'] = gc.apply(lambda x: x.rolling(30).sum()).fillna(value=-1).map(lambda loan_amount: to_norm_loan(loan_amount))
	age_sex_date_fea_df['asd_loan_norm_60d'] = gc.apply(lambda x: x.rolling(60).sum()).fillna(value=-1).map(lambda loan_amount: to_norm_loan(loan_amount))
	age_sex_date_fea_df['asd_loan_norm_90d'] = gc.apply(lambda x: x.rolling(90).sum()).fillna(value=-1).map(lambda loan_amount: to_norm_loan(loan_amount))

	age_sex_date_fea_df['asd_loan_norm'] = age_sex_date_fea_df['asd_loan'].map(lambda loan_amount: to_norm_loan(loan_amount))

	del age_sex_date_fea_df['asd_loan']

	print(" ---- Loan features have been generated. ")

	##################################################################################################################
	#
	# 将生成的特征表写入文件中
    ##################################################################################################################
	
	print(" - Writing features into file ! ")

	df = pd.merge(user_date_df[['uid', 'age', 'sex', 'date']], age_sex_date_fea_df, on=['age', 'sex', 'date'])
	df = df[['uid', 'date', 'asd_clk_cnt', 'asd_clk_cnt_3d', 'asd_clk_cnt_7d', 'asd_clk_cnt_14d', 'asd_clk_cnt_21d', 'asd_clk_cnt_30d', 'asd_clk_cnt_60d', 
			'asd_clk_cnt_90d', 'asd_ord_cnt', 'asd_ord_cnt_3d', 'asd_ord_cnt_7d', 'asd_ord_cnt_14d', 'asd_ord_cnt_21d', 'asd_ord_cnt_30d', 'asd_ord_cnt_60d',
			'asd_ord_cnt_90d', 'asd_ctr', 'asd_ctr_3d', 'asd_ctr_7d', 'asd_ctr_14d', 'asd_ctr_21d', 'asd_ctr_30d', 'asd_ctr_60d', 'asd_ctr_90d', 'asd_loan_norm',
			'asd_loan_norm_3d', 'asd_loan_norm_7d', 'asd_loan_norm_14d', 'asd_loan_norm_21d', 'asd_loan_norm_30d', 'asd_loan_norm_60d', 'asd_loan_norm_90d']]
	df.to_csv(fea_fn, index=False)

	print("----------------------- Success -------------------------")


if __name__ == '__main__':

	st = datetime.now()

	user_fn = '../../dataset/t_user.csv'
	click_fn = '../../dataset/t_click.csv'
	order_fn = '../../dataset/t_order.csv'
	loan_fn = '../../dataset/t_loan.csv'
	fea_fn = '../../fea/fea_age_sex_date.csv'

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