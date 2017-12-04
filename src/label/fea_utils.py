import math
import pandas as pd
import numpy as np
from datetime import datetime

def to_real_loan(norm_loan_amount):
	if norm_loan_amount < 1:
		return -1
	else:
		return   5**norm_loan_amount - 1

def to_norm_loan(real_loan_amount):
	if real_loan_amount < 0:
		return -1
	elif real_loan_amount == 0:
		return 0
	else:
		return math.log(real_loan_amount + 1, 5)

def get_user_date_cross(user_df, loan_df):
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
	return user_date_df