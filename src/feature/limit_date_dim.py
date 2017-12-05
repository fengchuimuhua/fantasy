import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from fea_utils import *
from sklearn import preprocessing

def gen_fea(user_fn, click_fn, order_fn, loan_fn, fea_fn):
	# step 0. INIT raw data
	user_df = pd.read_csv(user_fn)[['uid','limit']]
	clk_df = pd.read_csv(click_fn) # uid, click_time, pid, param
	ord_df = pd.read_csv(order_fn)
	loan_df = pd.read_csv(loan_fn)
	loan_df['real_loan_amount'] = loan_df['loan_amount'].map(lambda la : to_real_loan(la))
	# step 1 . user clasiisfication by limit
	le = preprocessing.LabelEncoder()
	limit_list = sorted(set(user_df.limit))
	le.fit(limit_list)
	user_df['limit_cat'] = le.transform(list(user_df.limit))
	limit_cate= le.transform(limit_list)
	limit_cat_df = pd.DataFrame(data={"limit_cat":limit_cate})
	print "limit_cate.size = {}".format(limit_cat_df.shape[0])
	# step 2. add date dim which represents date~date+30
	date_all = loan_df['loan_time'].str.split(' ', expand=True)[0]
	start_date = pd.to_datetime(date_all.min(), format='%Y-%m-%d')
	end_date = pd.to_datetime(date_all.max(), format='%Y-%m-%d')
	date_list = map(lambda d: datetime.strftime(d, '%Y-%m-%d'), pd.date_range(start_date, end_date).tolist())
	date_df = pd.DataFrame(data={'date': date_list})

	print set(date_df.date)
	# step 3.1. join limit_cat_df and date_df ('limit_cat', 'date')
	limit_cat_df['key'] = 1
	date_df['key'] = 1
	user_limit_cate_date_df = pd.merge(limit_cat_df, date_df, on='key')
	del user_limit_cate_date_df['key']
	print set(user_limit_cate_date_df.date)
	# setp 3.2 join user_df and [date_df  uid , limit , limit_cat, date]
	user_df['key'] = 1
	user_date_df = pd.merge(user_df , date_df , on='key')
	del user_date_df['key']

	print "step 3 done "
	#setp 4. add recent click num
	clk_df['date'] = clk_df['click_time'].map(lambda ct: ct.split(' ')[0])
	clk_user_cat_df = pd.merge(user_df[['uid','limit_cat']] , clk_df , on=['uid'])
	user_cat_date_clk_df = pd.DataFrame({'clk_cnt': clk_user_cat_df.groupby(['limit_cat', 'date']).size()}).reset_index()
	user_cat_date_df = pd.merge(user_limit_cate_date_df, user_cat_date_clk_df, on=['limit_cat', 'date'], how='left')
	user_cat_date_df['clk_cnt'] = user_cat_date_df['clk_cnt'].fillna(value=0)
	user_cat_date_df['clk_cnt_1d'] = user_cat_date_df['clk_cnt']
	gc = user_cat_date_df.groupby('limit_cat').clk_cnt
	# user_cat_date_df['clk_cnt_3d'] = gc.apply(lambda x: x.rolling(3).sum()).fillna(value=-1)
	# user_cat_date_df['clk_cnt_7d'] = gc.apply(lambda x: x.rolling(7).sum()).fillna(value=-1)
	# user_cat_date_df['clk_cnt_14d'] = gc.apply(lambda x: x.rolling(14).sum()).fillna(value=-1)
	# user_cat_date_df['clk_cnt_21d'] = gc.apply(lambda x: x.rolling(21).sum()).fillna(value=-1)
	user_cat_date_df['clk_cnt_30d'] = gc.apply(lambda x: x.rolling(30).sum()).fillna(value=-1)
	user_cat_date_df['clk_cnt_60d'] = gc.apply(lambda x: x.rolling(60).sum()).fillna(value=-1)
	user_cat_date_df['clk_cnt_90d'] = gc.apply(lambda x: x.rolling(90).sum()).fillna(value=-1)

	print set(user_cat_date_df.date)
	print "step 4 done"
	#step 5. add recent order num
	# ord_df['date'] = ord_df['buy_time']
	# ord_user_cat_df = pd.merge(user_df[['uid','limit_cat']] , ord_df , on='uid')
    #
	# user_cat_date_ord_df = pd.DataFrame({'ord_cnt': ord_user_cat_df.groupby(['limit_cat', 'date']).size()}).reset_index()
	# user_cat_date_df = pd.merge(user_cat_date_df , user_cat_date_ord_df , on=['limit_cat','date'] , how='left')
	# user_cat_date_df['ord_cnt'] = user_cat_date_df.ord_cnt
	# user_cat_date_df['ord_cnt_1d'] = user_cat_date_df.ord_cnt
	# gord = user_cat_date_df.groupby('limit_cat').ord_cnt
	# user_cat_date_df['ord_cnt_3d'] = gord.apply(lambda x: x.rolling(3).sum()).fillna(value=-1)
	# user_cat_date_df['ord_cnt_7d'] = gord.apply(lambda x: x.rolling(7).sum()).fillna(value=-1)
	# user_cat_date_df['ord_cnt_14d'] = gord.apply(lambda x: x.rolling(14).sum()).fillna(value=-1)
	# user_cat_date_df['ord_cnt_21d'] = gord.apply(lambda x: x.rolling(21).sum()).fillna(value=-1)
	# user_cat_date_df['ord_cnt_30d'] = gord.apply(lambda x: x.rolling(30).sum()).fillna(value=-1)
	# user_cat_date_df['ord_cnt_60d'] = gord.apply(lambda x: x.rolling(60).sum()).fillna(value=-1)
	# user_cat_date_df['ord_cnt_90d'] = gord.apply(lambda x: x.rolling(90).sum()).fillna(value=-1)
	# print set(user_cat_date_df.date)
	# print "step 5 done "
	#step 6. add recent ctr
	# user_cat_date_df['ctr_1d'] = (user_cat_date_df['ord_cnt_1d'] + 0.1) / (user_cat_date_df['clk_cnt_1d'] + 0.5)
	# user_cat_date_df['ctr_3d'] = (user_cat_date_df['ord_cnt_3d'] + 0.1) / (user_cat_date_df['clk_cnt_3d'] + 0.5)
	# user_cat_date_df['ctr_7d'] = (user_cat_date_df['ord_cnt_7d'] + 0.1) / (user_cat_date_df['clk_cnt_7d'] + 0.5)
	# user_cat_date_df['ctr_14d'] = (user_cat_date_df['ord_cnt_14d'] + 0.1) / (user_cat_date_df['clk_cnt_14d'] + 0.5)
	# user_cat_date_df['ctr_21d'] = (user_cat_date_df['ord_cnt_21d'] + 0.1) / (user_cat_date_df['clk_cnt_21d'] + 0.5)
	# user_cat_date_df['ctr_30d'] = (user_cat_date_df['ord_cnt_30d'] + 0.1) / (user_cat_date_df['clk_cnt_30d'] + 0.5)
	# user_cat_date_df['ctr_60d'] = (user_cat_date_df['ord_cnt_60d'] + 0.1) / (user_cat_date_df['clk_cnt_60d'] + 0.5)
	# user_cat_date_df['ctr_90d'] = (user_cat_date_df['ord_cnt_90d'] + 0.1) / (user_cat_date_df['clk_cnt_90d'] + 0.5)
	# user_cat_date_df[(user_cat_date_df['ord_cnt_1d'] < 0 ) | (user_cat_date_df['clk_cnt_1d'] < 0)] = -1
	# user_cat_date_df[(user_cat_date_df['ord_cnt_3d'] < 0)| ( user_cat_date_df['clk_cnt_3d'] < 0)] = -1
	# user_cat_date_df[(user_cat_date_df['ord_cnt_7d'] < 0 )|(user_cat_date_df['clk_cnt_7d'] < 0)] = -1
	# user_cat_date_df[(user_cat_date_df['ord_cnt_14d'] < 0)|( user_cat_date_df['clk_cnt_14d'] < 0)] = -1
	# user_cat_date_df[(user_cat_date_df['ord_cnt_21d'] < 0)|( user_cat_date_df['clk_cnt_21d'] < 0)] = -1
	# user_cat_date_df[(user_cat_date_df['ord_cnt_30d'] < 0)|( user_cat_date_df['clk_cnt_30d'] < 0)] = -1
	# user_cat_date_df[(user_cat_date_df['ord_cnt_60d'] < 0)|( user_cat_date_df['clk_cnt_60d'] < 0)] = -1
	# user_cat_date_df[(user_cat_date_df['ord_cnt_90d'] < 0)|( user_cat_date_df['clk_cnt_90d'] < 0)] = -1
	# print set(user_cat_date_df.date)
	# print "step 6 done "
	#step 7. a)|(recent loan
	loan_df['date'] = loan_df['loan_time'].map(lambda lt: lt.split(' ')[0])
	loan_user_cat_df= pd.merge(user_df[['uid','limit_cat']] , loan_df , on=['uid'])
	user_cat_date_loan_df = loan_user_cat_df.groupby(['limit_cat','date']).real_loan_amount.sum().reset_index()
	user_cat_date_df = pd.merge(user_cat_date_df , user_cat_date_loan_df , on =['limit_cat','date'] , how='left').fillna(0)
	gloan = user_cat_date_df.groupby(['limit_cat']).real_loan_amount
	user_cat_date_df['loan_1d'] = user_cat_date_df.real_loan_amount.map(lambda  rla : to_norm_loan(rla))
	user_cat_date_df['loan_3d'] = gloan.apply(lambda x: x.rolling(3).sum()).fillna(value=-1).map(lambda rla: to_norm_loan(rla))
	user_cat_date_df['loan_7d'] = gloan.apply(lambda x: x.rolling(7).sum()).fillna(value=-1).map(lambda rla: to_norm_loan(rla))
	user_cat_date_df['loan_14d'] = gloan.apply(lambda x: x.rolling(14).sum()).fillna(value=-1).map(lambda rla: to_norm_loan(rla))
	user_cat_date_df['loan_21d'] = gloan.apply(lambda x: x.rolling(21).sum()).fillna(value=-1).map(lambda rla: to_norm_loan(rla))
	user_cat_date_df['loan_30d'] = gloan.apply(lambda x: x.rolling(30).sum()).fillna(value=-1).map(lambda rla: to_norm_loan(rla))
	user_cat_date_df['loan_60d'] = gloan.apply(lambda x: x.rolling(60).sum()).fillna(value=-1).map(lambda rla: to_norm_loan(rla))
	user_cat_date_df['loan_90d'] = gloan.apply(lambda x: x.rolling(90).sum()).fillna(value=-1).map(lambda rla: to_norm_loan(rla))
	print set(user_cat_date_df.date)
	print "step 7 done "
	print len(user_date_df) , len(user_cat_date_df)
	#step 8. join user_date_df and user_cate_date_df
	print user_date_df.shape[0] , user_cat_date_df.shape[0]
	print set(user_date_df.date)
	print set(user_cat_date_df.date)
	df = pd.merge(user_date_df  , user_cat_date_df , on=['limit_cat','date'])
	print df.shape[0]
	print set(user_cat_date_df.date)
	print "step 8 done "
	#step 9. output
	df = df[['uid','date','limit_cat',
		'loan_1d', 'loan_3d', 'loan_7d', 'loan_14d', 'loan_21d', 'loan_30d', 'loan_60d', 'loan_90d']]
	
	df.columns = [['uid','date','ld_limit_cat',
		'ld_loan_1d', 'ld_loan_3d', 'ld_loan_7d', 'ld_loan_14d', 'ld_loan_21d', 'ld_loan_30d', 'ld_loan_60d', 'ld_loan_90d']]
	df.to_csv(fea_fn, index=False)
	print "step 9 done "


def merge(a , b):
	res = pd.merge(a , b , on=['uid','date'])
	return res


if __name__ == '__main__':
	st = datetime.now()
	user_fn = '../../dataset/t_user.csv'
	click_fn = '../../dataset/t_click.csv'
	order_fn = '../../dataset/t_order.csv'
	loan_fn = '../../dataset/t_loan.csv'

	fea_fn = '../../fea/fea_limit_date.csv'

	if len(sys.argv) != 6:
		print sys.argv[0] + '\t[user_fn]\t[click_fn]\t[order_fn]\t[loan_fn]\t[fea_fn]'
	else:
		user_fn = sys.argv[1]
		click_fn = sys.argv[2]
		order_fn = sys.argv[3]
		loan_fn = sys.argv[4]
		fea_fn = sys.argv[5]
	gen_fea(user_fn, click_fn, order_fn, loan_fn, fea_fn)
	et = datetime.now()
	print 'time cost : ' + str(et - st)