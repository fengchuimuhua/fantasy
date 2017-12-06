from fea_utils import *
import sys
import pandas as pd
import numpy as np
from datetime import datetime

def ctr_norm(ord, clk, ctr):
	if ord < 0 or clk < 0:
		return -1
	else:
		return ctr

def gen_fea(user_fn, click_fn, order_fn, loan_fn, fea_fn):	
	# step 0. INIT raw data
	print "read ..."
	user_df = pd.read_csv(user_fn)[['uid', 'active_date']]
	clk_df = pd.read_csv(click_fn) # uid, click_time, pid, param
	ord_df = pd.read_csv(order_fn)
	loan_df = pd.read_csv(loan_fn)
	loan_df['real_loan_amount'] = loan_df['loan_amount'].map(lambda la : to_real_loan(la))
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
	print 'user_df , date_df'
	user_df['key'] = 1
	date_df['key'] = 1
	user_date_df = pd.merge(user_df, date_df, on='key')
	del user_date_df['key']
	# step 3. add activation_days ('uid', 'date', 'active_days')
	print 'active_days'
	user_date_df['active_days'] =  user_date_df.apply(lambda  x : ( datetime.strptime(x['date'],'%Y-%m-%d') - datetime.strptime(x['active_date'],'%Y-%m-%d') ).days , axis=1 )
	del user_date_df['active_date']
	# step 4. add recent click num
	print 'clk'
	clk_df['date'] = clk_df['click_time'].map(lambda ct : ct.split(' ')[0])
	user_date_clk_df = pd.DataFrame({'clk_cnt': clk_df.groupby(['uid', 'date']).size()}).reset_index()
	user_date_df = pd.merge(user_date_df, user_date_clk_df, on=['uid','date'], how='left')
	user_date_df['clk_cnt'] = user_date_df['clk_cnt'].fillna(value=0)
	user_date_df['clk_cnt_1d'] = user_date_df['clk_cnt']
	gc = user_date_df.groupby('uid').clk_cnt
	user_date_df['clk_cnt_3d'] = gc.apply(lambda x : x.rolling(3).sum()).fillna(value=-1)
	user_date_df['clk_cnt_7d'] = gc.apply(lambda x : x.rolling(7).sum()).fillna(value=-1)
	user_date_df['clk_cnt_14d'] = gc.apply(lambda x : x.rolling(14).sum()).fillna(value=-1)
	user_date_df['clk_cnt_21d'] = gc.apply(lambda x : x.rolling(21).sum()).fillna(value=-1)
	user_date_df['clk_cnt_30d'] = gc.apply(lambda x : x.rolling(30).sum()).fillna(value=-1)
	user_date_df['clk_cnt_60d'] = gc.apply(lambda x : x.rolling(60).sum()).fillna(value=-1)
	user_date_df['clk_cnt_90d'] = gc.apply(lambda x : x.rolling(90).sum()).fillna(value=-1)
	step 5. add recent order num
	ord_df['date'] = ord_df['buy_time']
	user_date_ord_df = pd.DataFrame({'ord_cnt': ord_df.groupby(['uid', 'date']).size()}).reset_index()
	user_date_df = pd.merge(user_date_df, user_date_ord_df, on=['uid','date'], how='left')
	user_date_df['ord_cnt'] = user_date_df['ord_cnt'].fillna(value=0)
	user_date_df['ord_cnt_1d'] = user_date_df['ord_cnt']
	gord = user_date_df.groupby('uid').ord_cnt
	user_date_df['ord_cnt_3d'] = gord.apply(lambda x : x.rolling(3).sum()).fillna(value=-1)
	user_date_df['ord_cnt_7d'] = gord.apply(lambda x : x.rolling(7).sum()).fillna(value=-1)
	user_date_df['ord_cnt_14d'] = gord.apply(lambda x : x.rolling(14).sum()).fillna(value=-1)
	user_date_df['ord_cnt_21d'] = gord.apply(lambda x : x.rolling(21).sum()).fillna(value=-1)
	user_date_df['ord_cnt_30d'] = gord.apply(lambda x : x.rolling(30).sum()).fillna(value=-1)
	user_date_df['ord_cnt_60d'] = gord.apply(lambda x : x.rolling(60).sum()).fillna(value=-1)
	user_date_df['ord_cnt_90d'] = gord.apply(lambda x : x.rolling(90).sum()).fillna(value=-1)
	step 6. add recent ctr
	user_date_df['ctr_1d'] = (user_date_df['ord_cnt_1d'] + 0.01) / (user_date_df['clk_cnt_1d'] + 0.05)
	user_date_df['ctr_3d'] = (user_date_df['ord_cnt_3d'] + 0.01) / (user_date_df['clk_cnt_3d'] + 0.05)
	user_date_df['ctr_7d'] = (user_date_df['ord_cnt_7d'] + 0.01) / (user_date_df['clk_cnt_7d'] + 0.05)
	user_date_df['ctr_14d'] = (user_date_df['ord_cnt_14d'] + 0.01) / (user_date_df['clk_cnt_14d'] + 0.05)
	user_date_df['ctr_21d'] = (user_date_df['ord_cnt_21d'] + 0.01) / (user_date_df['clk_cnt_21d'] + 0.05)
	user_date_df['ctr_30d'] = (user_date_df['ord_cnt_30d'] + 0.01) / (user_date_df['clk_cnt_30d'] + 0.05)
	user_date_df['ctr_60d'] = (user_date_df['ord_cnt_60d'] + 0.01) / (user_date_df['clk_cnt_60d'] + 0.05)
	user_date_df['ctr_90d'] = (user_date_df['ord_cnt_90d'] + 0.01) / (user_date_df['clk_cnt_90d'] + 0.05)
	user_date_df['ctr_1d'] = map(lambda ord, clk, ctr : ctr_norm(ord, clk, ctr), user_date_df['ord_cnt_1d'], user_date_df['clk_cnt_1d'], user_date_df['ctr_1d'])
	user_date_df['ctr_3d'] = map(lambda ord, clk, ctr : ctr_norm(ord, clk, ctr), user_date_df['ord_cnt_3d'], user_date_df['clk_cnt_3d'], user_date_df['ctr_3d'])
	user_date_df['ctr_7d'] = map(lambda ord, clk, ctr : ctr_norm(ord, clk, ctr), user_date_df['ord_cnt_7d'], user_date_df['clk_cnt_7d'], user_date_df['ctr_7d'])
	user_date_df['ctr_14d'] = map(lambda ord, clk, ctr : ctr_norm(ord, clk, ctr), user_date_df['ord_cnt_14d'], user_date_df['clk_cnt_14d'], user_date_df['ctr_14d'])
	user_date_df['ctr_21d'] = map(lambda ord, clk, ctr : ctr_norm(ord, clk, ctr), user_date_df['ord_cnt_21d'], user_date_df['clk_cnt_21d'], user_date_df['ctr_21d'])
	user_date_df['ctr_30d'] = map(lambda ord, clk, ctr : ctr_norm(ord, clk, ctr), user_date_df['ord_cnt_30d'], user_date_df['clk_cnt_30d'], user_date_df['ctr_30d'])
	user_date_df['ctr_60d'] = map(lambda ord, clk, ctr : ctr_norm(ord, clk, ctr), user_date_df['ord_cnt_60d'], user_date_df['clk_cnt_60d'], user_date_df['ctr_60d'])
	user_date_df['ctr_90d'] = map(lambda ord, clk, ctr : ctr_norm(ord, clk, ctr), user_date_df['ord_cnt_90d'], user_date_df['clk_cnt_90d'], user_date_df['ctr_90d'])
	#step 7. add recent loan
	print 'loan_sum'
	loan_df['date'] = loan_df['loan_time'].map(lambda lt : lt.split(' ')[0])
	uid_date_loan = loan_df.groupby(['uid', 'date']).real_loan_amount.sum().reset_index()
	user_date_df = pd.merge(user_date_df, uid_date_loan, on=['uid','date'], how='left')
	user_date_df['real_loan_amount'] = user_date_df['real_loan_amount'].fillna(value=0)
	gloan = user_date_df.groupby('uid').real_loan_amount
	user_date_df['loan_1d'] = user_date_df['real_loan_amount'].map(lambda rla : to_norm_loan(rla))
	user_date_df['loan_3d'] = gloan.apply(lambda x : x.rolling(3).sum()).fillna(value=-1).map(lambda rla : to_norm_loan(rla))
	user_date_df['loan_7d'] = gloan.apply(lambda x : x.rolling(7).sum()).fillna(value=-1).map(lambda rla : to_norm_loan(rla))
	user_date_df['loan_14d'] = gloan.apply(lambda x : x.rolling(14).sum()).fillna(value=-1).map(lambda rla : to_norm_loan(rla))
	user_date_df['loan_21d'] = gloan.apply(lambda x : x.rolling(21).sum()).fillna(value=-1).map(lambda rla : to_norm_loan(rla))
	user_date_df['loan_30d'] = gloan.apply(lambda x : x.rolling(30).sum()).fillna(value=-1).map(lambda rla : to_norm_loan(rla))
	user_date_df['loan_60d'] = gloan.apply(lambda x : x.rolling(60).sum()).fillna(value=-1).map(lambda rla : to_norm_loan(rla))
	user_date_df['loan_90d'] = gloan.apply(lambda x : x.rolling(90).sum()).fillna(value=-1).map(lambda rla : to_norm_loan(rla))
	# add loan num
	# user_date_df['loan_num_1d'] = user_date_df['real_loan_amount'].map(lambda rla : to_norm_loan(rla))
	# user_date_df['loan_num_3d'] = gloan.apply(lambda x : x.rolling(3).size()).fillna(value=-1).map(lambda rla : to_norm_loan(rla))
	# user_date_df['loan_num_7d'] = gloan.apply(lambda x : x.rolling(7).size()).fillna(value=-1).map(lambda rla : to_norm_loan(rla))
	# user_date_df['loan_num_14d'] = gloan.apply(lambda x : x.rolling(14).size()).fillna(value=-1)
	# user_date_df['loan_num_21d'] = gloan.apply(lambda x : x.rolling(21).size()).fillna(value=-1)
	# user_date_df['loan_num_30d'] = gloan.apply(lambda x : x.rolling(30).size()).fillna(value=-1)
	# user_date_df['loan_num_60d'] = gloan.apply(lambda x : x.rolling(60).size()).fillna(value=-1)
	# user_date_df['loan_num_90d'] = gloan.apply(lambda x : x.rolling(90).size()).fillna(value=-1)
	# add max loan
	print 'loan max'
	user_date_df['loan_max_3d'] = gloan.apply(lambda x: x.rolling(3).max()).fillna(value=-1).map(
		lambda rla: to_norm_loan(rla))
	user_date_df['loan_max_7d'] = gloan.apply(lambda x: x.rolling(7).max()).fillna(value=-1).map(
		lambda rla: to_norm_loan(rla))
	user_date_df['loan_max_14d'] = gloan.apply(lambda x: x.rolling(14).max()).fillna(value=-1).map(
		lambda rla: to_norm_loan(rla))
	user_date_df['loan_max_21d'] = gloan.apply(lambda x: x.rolling(21).max()).fillna(value=-1).map(
		lambda rla: to_norm_loan(rla))
	user_date_df['loan_max_30d'] = gloan.apply(lambda x: x.rolling(30).max()).fillna(value=-1).map(
		lambda rla: to_norm_loan(rla))
	user_date_df['loan_max_60d'] = gloan.apply(lambda x: x.rolling(60).max()).fillna(value=-1).map(
		lambda rla: to_norm_loan(rla))
	user_date_df['loan_max_90d'] = gloan.apply(lambda x: x.rolling(90).max()).fillna(value=-1).map(
		lambda rla: to_norm_loan(rla))
	# add mean loan
	print 'loan mean'
	user_date_df['loan_mean_3d'] = gloan.apply(lambda x: x.rolling(3).mean()).fillna(value=-1).map(
		lambda rla: to_norm_loan(rla))
	user_date_df['loan_mean_7d'] = gloan.apply(lambda x: x.rolling(7).mean()).fillna(value=-1).map(
		lambda rla: to_norm_loan(rla))
	user_date_df['loan_mean_14d'] = gloan.apply(lambda x: x.rolling(14).mean()).fillna(value=-1).map(
		lambda rla: to_norm_loan(rla))
	user_date_df['loan_mean_21d'] = gloan.apply(lambda x: x.rolling(21).mean()).fillna(value=-1).map(
		lambda rla: to_norm_loan(rla))
	user_date_df['loan_mean_30d'] = gloan.apply(lambda x: x.rolling(30).mean()).fillna(value=-1).map(
		lambda rla: to_norm_loan(rla))
	user_date_df['loan_mean_60d'] = gloan.apply(lambda x: x.rolling(60).mean()).fillna(value=-1).map(
		lambda rla: to_norm_loan(rla))
	user_date_df['loan_mean_90d'] = gloan.apply(lambda x: x.rolling(90).mean()).fillna(value=-1).map(
		lambda rla: to_norm_loan(rla))
	# add skew loan
	print 'loan skew'
	user_date_df['loan_skew_3d'] = gloan.apply(lambda x: x.rolling(3).skew()).fillna(value=-1).map(
		lambda rla: to_norm_loan(rla))
	user_date_df['loan_skew_7d'] = gloan.apply(lambda x: x.rolling(7).skew()).fillna(value=-1).map(
		lambda rla: to_norm_loan(rla))
	user_date_df['loan_skew_14d'] = gloan.apply(lambda x: x.rolling(14).skew()).fillna(value=-1).map(
		lambda rla: to_norm_loan(rla))
	user_date_df['loan_skew_21d'] = gloan.apply(lambda x: x.rolling(21).skew()).fillna(value=-1).map(
		lambda rla: to_norm_loan(rla))
	user_date_df['loan_skew_30d'] = gloan.apply(lambda x: x.rolling(30).skew()).fillna(value=-1).map(
		lambda rla: to_norm_loan(rla))
	user_date_df['loan_skew_60d'] = gloan.apply(lambda x: x.rolling(60).skew()).fillna(value=-1).map(
		lambda rla: to_norm_loan(rla))
	user_date_df['loan_skew_90d'] = gloan.apply(lambda x: x.rolling(90).skew()).fillna(value=-1).map(
		lambda rla: to_norm_loan(rla))
	# add mad loan
	print 'loan mad'
	user_date_df['loan_mad_3d'] = gloan.apply(lambda x: x.rolling(3).kurt()).fillna(value=-1).map(
		lambda rla: to_norm_loan(rla))
	user_date_df['loan_mad_7d'] = gloan.apply(lambda x: x.rolling(7).kurt()).fillna(value=-1).map(
		lambda rla: to_norm_loan(rla))
	user_date_df['loan_mad_14d'] = gloan.apply(lambda x: x.rolling(14).kurt()).fillna(value=-1).map(
		lambda rla: to_norm_loan(rla))
	user_date_df['loan_mad_21d'] = gloan.apply(lambda x: x.rolling(21).kurt()).fillna(value=-1).map(
		lambda rla: to_norm_loan(rla))
	user_date_df['loan_mad_30d'] = gloan.apply(lambda x: x.rolling(30).kurt()).fillna(value=-1).map(
		lambda rla: to_norm_loan(rla))
	user_date_df['loan_mad_60d'] = gloan.apply(lambda x: x.rolling(60).kurt()).fillna(value=-1).map(
		lambda rla: to_norm_loan(rla))
	user_date_df['loan_mad_90d'] = gloan.apply(lambda x: x.rolling(90).kurt()).fillna(value=-1).map(
		lambda rla: to_norm_loan(rla))
	# add mean loan 
	print 'loan min'
	user_date_df['loan_min_3d'] = gloan.apply(lambda x: x.rolling(3).min()).fillna(value=-1).map(
		lambda rla: to_norm_loan(rla))
	user_date_df['loan_min_7d'] = gloan.apply(lambda x: x.rolling(7).min()).fillna(value=-1).map(
		lambda rla: to_norm_loan(rla))
	user_date_df['loan_min_14d'] = gloan.apply(lambda x: x.rolling(14).min()).fillna(value=-1).map(
		lambda rla: to_norm_loan(rla))
	user_date_df['loan_min_21d'] = gloan.apply(lambda x: x.rolling(21).min()).fillna(value=-1).map(
		lambda rla: to_norm_loan(rla))
	user_date_df['loan_min_30d'] = gloan.apply(lambda x: x.rolling(30).min()).fillna(value=-1).map(
		lambda rla: to_norm_loan(rla))
	user_date_df['loan_min_60d'] = gloan.apply(lambda x: x.rolling(60).min()).fillna(value=-1).map(
		lambda rla: to_norm_loan(rla))
	user_date_df['loan_min_90d'] = gloan.apply(lambda x: x.rolling(90).min()).fillna(value=-1).map(
		lambda rla: to_norm_loan(rla))
	# add std loan 
	print 'loan std'
	user_date_df['loan_std_3d'] = gloan.apply(lambda x: x.rolling(3).std()).fillna(value=-1).map(
		lambda rla: to_norm_loan(rla))
	user_date_df['loan_std_7d'] = gloan.apply(lambda x: x.rolling(7).std()).fillna(value=-1).map(
		lambda rla: to_norm_loan(rla))
	user_date_df['loan_std_14d'] = gloan.apply(lambda x: x.rolling(14).std()).fillna(value=-1).map(
		lambda rla: to_norm_loan(rla))
	user_date_df['loan_std_21d'] = gloan.apply(lambda x: x.rolling(21).std()).fillna(value=-1).map(
		lambda rla: to_norm_loan(rla))
	user_date_df['loan_std_30d'] = gloan.apply(lambda x: x.rolling(30).std()).fillna(value=-1).map(
		lambda rla: to_norm_loan(rla))
	user_date_df['loan_std_60d'] = gloan.apply(lambda x: x.rolling(60).std()).fillna(value=-1).map(
		lambda rla: to_norm_loan(rla))
	user_date_df['loan_std_90d'] = gloan.apply(lambda x: x.rolling(90).std()).fillna(value=-1).map(
		lambda rla: to_norm_loan(rla))
	# step 8. output
	print 'save...'
	user_date_df = 	user_date_df[['uid', 'date', 'active_days',
				  #'clk_cnt_1d', 'clk_cnt_3d', 'clk_cnt_7d', 'clk_cnt_14d', 'clk_cnt_21d', 'clk_cnt_30d', 'clk_cnt_60d','clk_cnt_90d',
				  #'ord_cnt_1d', 'ord_cnt_3d', 'ord_cnt_7d', 'ord_cnt_14d', 'ord_cnt_21d', 'ord_cnt_30d', 'ord_cnt_60d', 'ord_cnt_90d',
				  #'ctr_1d', 'ctr_3d', 'ctr_7d', 'ctr_14d', 'ctr_21d', 'ctr_30d', 'ctr_60d', 'ctr_90d',
				  'loan_1d', 'loan_1d', 'loan_3d', 'loan_7d', 'loan_14d', 'loan_21d', 'loan_30d', 'loan_60d',
				  'loan_90d',
				  #'loan_num_7d', 'loan_num_14d', 'loan_num_21d', 'loan_num_30d', 'loan_num_60d', 'loan_num_90d',
				  'loan_max_7d', 'loan_max_14d', 'loan_max_21d', 'loan_max_30d', 'loan_max_60d', 'loan_max_90d',
				  'loan_min_7d', 'loan_min_14d', 'loan_min_21d', 'loan_min_30d', 'loan_min_60d', 'loan_min_90d',
				  'loan_std_7d', 'loan_std_14d', 'loan_std_21d', 'loan_std_30d', 'loan_std_60d', 'loan_std_90d',
				  'loan_mean_7d', 'loan_mean_14d', 'loan_mean_21d', 'loan_mean_30d', 'loan_mean_60d', 'loan_mean_90d',
				  'loan_skew_7d', 'loan_skew_14d', 'loan_skew_21d', 'loan_skew_30d', 'loan_skew_60d', 'loan_skew_90d',
				  'loan_mad_7d', 'loan_mad_14d', 'loan_mad_21d', 'loan_mad_30d', 'loan_mad_60d', 'loan_mad_90d'
				  ]]

	user_date_df.columns = ['uid','date','ud_active_days',
'ud_clk_cnt_1d','ud_clk_cnt_3d','ud_clk_cnt_7d','ud_clk_cnt_14d','ud_clk_cnt_21d','ud_clk_cnt_30d','ud_clk_cnt_60d','ud_clk_cnt_90d',
'ud_ord_cnt_1d','ud_ord_cnt_3d','ud_ord_cnt_7d','ud_ord_cnt_14d','ud_ord_cnt_21d','ud_ord_cnt_30d','ud_ord_cnt_60d','ud_ord_cnt_90d',
'ud_ctr_1d','ud_ctr_3d','ud_ctr_7d','ud_ctr_14d','ud_ctr_21d','ud_ctr_30d','ud_ctr_60d','ud_ctr_90d',
'loan_1d','ud_loan_1d','ud_loan_3d','ud_loan_7d','ud_loan_14d','ud_loan_21d','ud_loan_30d','ud_loan_60d','ud_loan_90d',
#'loan_num_7d','ud_loan_num_14d','ud_loan_num_21d','ud_loan_num_30d','ud_loan_num_60d','ud_loan_num_90d',
'loan_max_7d','ud_loan_max_14d','ud_loan_max_21d','ud_loan_max_30d','ud_loan_max_60d','ud_loan_max_90d',
'loan_min_7d','ud_loan_min_14d','ud_loan_min_21d','ud_loan_min_30d','ud_loan_min_60d','ud_loan_min_90d',
'loan_std_7d','ud_loan_std_14d','ud_loan_std_21d','ud_loan_std_30d','ud_loan_std_60d','ud_loan_std_90d',
 'loan_mean_7d','ud_loan_mean_14d','ud _loan_mean_21d','ud_loan_mean_30d','ud_loan_mean_60d','ud_loan_mean_90d',
 'loan_skew_7d','ud_loan_skew_14d','ud_loan_skew_21d','ud_loan_skew_30d','ud_loan_skew_60d','ud_loan_skew_90d',
'loan_mad_7d','ud_loan_mad_14d','ud_loan_mad_21d','ud_loan_mad_30d','ud_loan_mad_60d','ud_loan_mad_90d']

	user_date_df.to_csv(fea_fn, index=False)

if __name__ == '__main__':
	st = datetime.now()
	user_fn = '../../dataset/t_user.csv'
	click_fn = '../../dataset/t_click.csv'
	order_fn = '../../dataset/t_order.csv'
	loan_fn = '../../dataset/t_loan.csv'
	fea_fn = '../../fea/fea_user_date.csv'
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