import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def gen_fea(user_fn, click_fn, order_fn, loan_fn, fea_fn):	
	# step 0. INIT raw data
	user_df = pd.read_csv(user_fn)[['uid', 'active_date']]
	clk_df = pd.read_csv(click_fn) # uid, click_time, pid, param
	ord_df = pd.read_csv(order_fn)
	loan_df = pd.read_csv(loan_fn)
	# for test
	#user_df = user_df[:5]
	#clk_df = clk_df[:100]
	# step 1. add date dim which represents date ~ date + 30
	date_all = loan_df['loan_time'].str.split(' ', expand=True)[0]
	start_date = pd.to_datetime(date_all.min(), format='%Y-%m-%d')
	end_date = pd.to_datetime(date_all.max(), format='%Y-%m-%d')
	date_list = map(lambda d : datetime.strftime(d, '%Y-%m-%d'), pd.date_range(start_date, end_date).tolist())
	date_df = pd.DataFrame(data={'date' : date_list})
	# step 2. join user_df and date_df ('uid', 'active_date', 'date')
	user_df['key'] = 1
	date_df['key'] = 1
	user_date_df = pd.merge(user_df, date_df, on='key')
	del user_date_df['key']
	# step 3. add activation_days ('uid', 'date', 'active_days')
	user_date_df['active_days'] = (pd.to_datetime(user_date_df['date']) - pd.to_datetime(user_date_df['active_date'])).map(lambda d : d.days)
	del user_date_df['active_date']
	# step 4. add recent click num 
	clk_df['date'] = clk_df['click_time'].map(lambda ct : ct.split(' ')[0])
	user_date_clk_df = pd.DataFrame({'clk_cnt': clk_df.groupby(['uid', 'date']).size()}).reset_index()
	user_date_df = pd.merge(user_date_df, user_date_clk_df, on=['uid','date'], how='left')
	user_date_df['clk_cnt'] = user_date_df['clk_cnt'].fillna(value=0)
	user_date_df['clk_cnt_1d'] = user_date_df['clk_cnt']
	gc = user_date_df.groupby('uid').clk_cnt
	user_date_df['clk_cnt_3d'] = gc.rolling(3).sum().reset_index()['clk_cnt'].fillna(value=-1)
	user_date_df['clk_cnt_7d'] = gc.rolling(7).sum().reset_index()['clk_cnt'].fillna(value=-1)
	user_date_df['clk_cnt_14d'] = gc.rolling(14).sum().reset_index()['clk_cnt'].fillna(value=-1)
	user_date_df['clk_cnt_21d'] = gc.rolling(21).sum().reset_index()['clk_cnt'].fillna(value=-1)
	user_date_df['clk_cnt_31d'] = gc.rolling(31).sum().reset_index()['clk_cnt'].fillna(value=-1)
	user_date_df['clk_cnt_45d'] = gc.rolling(45).sum().reset_index()['clk_cnt'].fillna(value=-1)
	# step 5. add recent order num
	ord_df['date'] = ord_df['buy_time']
	user_date_ord_df = pd.DataFrame({'ord_cnt': ord_df.groupby(['uid', 'date']).size()}).reset_index()
	user_date_df = pd.merge(user_date_df, user_date_ord_df, on=['uid','date'], how='left')
	user_date_df['ord_cnt'].fillna(value=0)
	user_date_df['ord_cnt_1d'] = user_date_df['ord_cnt']
	gord = user_date_df.groupby('uid').ord_cnt
	user_date_df['ord_cnt_3d'] = gord.rolling(3).sum().reset_index()['ord_cnt'].fillna(value=-1)
	user_date_df['ord_cnt_7d'] = gord.rolling(7).sum().reset_index()['ord_cnt'].fillna(value=-1)
	user_date_df['ord_cnt_14d'] = gord.rolling(14).sum().reset_index()['ord_cnt'].fillna(value=-1)
	user_date_df['ord_cnt_21d'] = gord.rolling(21).sum().reset_index()['ord_cnt'].fillna(value=-1)
	user_date_df['ord_cnt_31d'] = gord.rolling(31).sum().reset_index()['ord_cnt'].fillna(value=-1)
	user_date_df['ord_cnt_60d'] = gord.rolling(45).sum().reset_index()['ord_cnt'].fillna(value=-1)
	# step 6. add recent ctr
	user_date_df['ctr_1d'] = (user_date_df['ord_cnt_1d'] + 0.1) / (user_date_df['clk_cnt_1d'] + 0.5)
	user_date_df['ctr_3d'] = (user_date_df['ord_cnt_3d'] + 0.1) / (user_date_df['clk_cnt_3d'] + 0.5)
	user_date_df['ctr_7d'] = (user_date_df['ord_cnt_7d'] + 0.1) / (user_date_df['clk_cnt_7d'] + 0.5)
	user_date_df['ctr_14d'] = (user_date_df['ord_cnt_14d'] + 0.1) / (user_date_df['clk_cnt_14d'] + 0.5)
	user_date_df['ctr_21d'] = (user_date_df['ord_cnt_21d'] + 0.1) / (user_date_df['clk_cnt_21d'] + 0.5)
	user_date_df['ctr_31d'] = (user_date_df['ord_cnt_31d'] + 0.1) / (user_date_df['clk_cnt_31d'] + 0.5)
	user_date_df['ctr_60d'] = (user_date_df['ord_cnt_60d'] + 0.1) / (user_date_df['clk_cnt_60d'] + 0.5)
	user_date_df[user_date_df['ctr_1d']<0] = -1
	user_date_df[user_date_df['ctr_3d']<0] = -1
	user_date_df[user_date_df['ctr_7d']<0] = -1
	user_date_df[user_date_df['ctr_14d']<0] = -1
	user_date_df[user_date_df['ctr_21d']<0] = -1
	user_date_df[user_date_df['ctr_31d']<0] = -1
	user_date_df[user_date_df['ctr_60d']<0] = -1
	# step 7. add recent loan


if __name__ == '__main__':
	st = datetime.now()
	user_fn = '../../dataset/t_user.csv'
	click_fn = '../../dataset/t_click.csv'
	order_fn = '../../dataset/t_order.csv'
	loan_fn = '../../dataset/t_loan.csv'
	fea_fn = '../../fea/fea_user.csv'
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