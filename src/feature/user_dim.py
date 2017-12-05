import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def merge(user_df, user_date_df):
	return pd.merge(user_date_df, user_df, on='uid', how='left')

def generate_fea(user_fn, fea_fn):
	# step 0. INIT
	user_df = pd.read_csv(user_fn)
	print user_df.describe()
	# step 1. gen user fea
	user_df = user_df[['uid', 'age', 'sex', 'limit']]
	user_df.columns = ['uid', 'u_age', 'u_sex', 'u_limit']
	user_df.to_csv(fea_fn, index=False)

if __name__ == '__main__':
	user_fn = '../../dataset/t_user.csv'
	fea_fn = '../../fea/fea_user.csv'
	if len(sys.argv) != 3:
		print sys.argv[0] + '\t[user_fn]\t[fea_fn]'
	else:
		user_fn = sys.argv[1]
		fea_fn = sys.argv[2]
	generate_fea(user_fn, fea_fn)