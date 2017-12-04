import common
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from feature import fea_utils

def convertBoolToInt(boolVar):
	if boolVar:
		return 1
	else:
		return 0

def gen_label(user_fn, loan_fn, label_fn):
	# step 0. loan raw file
	user_df = pd.read_csv(user_fn)
	loan_df = pd.read_csv(loan_fn)
	# step 1. get user_date_df (uid, date)
	user_date_df = fea_utils.get_user_date_cross(user_df, loan_df)
	# step 2. get loan per one day and user (uid, date, real_loan_amount)
	loan_df['date'] = loan_df['loan_time'].map(lambda lt : lt.split(' ')[0])
	loan_df['real_loan_amount'] = loan_df['loan_amount'].map(lambda la : fea_utils.to_real_loan(la))
	user_date_loan_df = loan_df.groupby(['uid', 'date']).real_loan_amount.sum().reset_index()
	user_date_df = pd.merge(user_date_df, user_date_loan_df, on=['uid', 'date'], how='left')
	user_date_df['real_loan_amount'] = user_date_df['real_loan_amount'].fillna(0)
	# step 3. generate the label
	user_date_df['is_loan'] = (user_date_df['real_loan_amount'] > 0).map(convertBoolToInt)
	grp_loan = user_date_df.groupby('uid').is_loan
	user_date_df['loan_cnt_in_31d'] =  grp_loan.apply(lambda x : (x[::-1].rolling(window=31).sum())[::-1] - x).fillna(-1)
	grp_loan = user_date_df.groupby('uid').real_loan_amount
	user_date_df['real_loan_amount_in_31d'] = grp_loan.apply(lambda x : (x[::-1].rolling(window=32).sum())[::-1] - x).fillna(-1)
	user_date_df['loan_amount_in_31d'] = user_date_df['real_loan_amount_in_31d'].map(lambda x : fea_utils.to_norm_loan(x))
	# step 4. output
	user_date_df[['uid', 'date', 'real_loan_amount', 'is_loan', 'loan_cnt_in_31d', 'real_loan_amount_in_31d', 'loan_amount_in_31d']].to_csv(label_fn, index=False)

if __name__ == '__main__':
	st = datetime.now()
	user_fn = '../../dataset/t_user.csv'
	loan_fn = '../../dataset/t_loan.csv'
	label_fn = '../../label/user_date_label.csv'
	if len(sys.argv) != 4:
		print sys.argv[0] + '\t[user_fn]\t[loan_fn]\t[label_fn]'
	else:
		user_fn = sys.argv[1]
		loan_fn = sys.argv[2]
		label_fn = sys.argv[3]
	gen_label(user_fn, loan_fn, label_fn)
	et = datetime.now()
	print 'time cost : ' + str(et - st)