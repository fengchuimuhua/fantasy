
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import StratifiedKFold
from sklearn import ensemble
import pandas as pd
import numpy as np
import datetime
#coding:utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


dtfmt = '%Y-%m-%d %H:%M:%S' 
import timeit
import time
import itertools
import datetime
import math
import fea_utils as futils


# In[90]:

def gen_user_date(user_df , loan_df):
	date_all = loan_df['loan_time'].str.split(' ', expand=True)[0]
	start_date = pd.to_datetime(date_all.min(), format='%Y-%m-%d')
	end_date = pd.to_datetime(date_all.max(), format='%Y-%m-%d')
	date_list = map(lambda d : datetime.datetime.strftime(d, '%Y-%m-%d'), pd.date_range(start_date, end_date).tolist())
	date_df = pd.DataFrame(data={'date' : date_list})
	# step 2. join user_df and date_df ('uid', 'active_date', 'date')
	user_df['key'] = 1
	date_df['key'] = 1
	user_date_df = pd.merge(user_df, date_df, on='key')
	user_date_df['month'] = user_date_df.date.apply(lambda x : str(x)[5 :7 ]).astype('int')
	del user_date_df['key']
	return user_date_df[['uid','date','month']]


def gen_fea(user_fn, click_fn, order_fn, loan_fn, fea_fn):    
	#step 0. init raw data 
	user_df = pd.read_csv(user_fn)[['uid', 'active_date','limit']]
	user_df['real_limit'] = user_df.limit.map(lambda la : futils.to_real_loan(la))
	loan_df = pd.read_csv(loan_fn)
	loan_df['real_loan_amount'] = loan_df['loan_amount'].map(lambda la : futils.to_real_loan(la))
	loan_df['loan_month'] = loan_df.loan_time.apply(lambda x :x[5:7]).astype('int')


	#step 1. uid-month pair  
	date_all = loan_df['loan_time'].apply(lambda x : x[5:7]).astype('int')
	start_date = date_all.min()
	end_date = date_all.max()
	date_list = range(start_date , end_date+1)
	date_df = pd.DataFrame(data={'month' : date_list})
	date_df = pd.DataFrame(data={'month' : date_list})
	# step 2. join user_df and date_df ('uid', 'active_date', 'month')
	user_df['key'] = 1
	date_df['key'] = 1
	user_month_df = pd.merge(user_df, date_df, on='key')

	print "user_df.size = {} , date_df.size = {} , user_month_df.size = {}".format( len(user_df) , len(date_df) , len(user_month_df) )

	#step 3. left balance 

	user_date_loan_df = pd.merge(user_month_df , loan_df , on=['uid'] , how='left').fillna(0)
	user_date_loan_df['delta_month'] = user_date_loan_df.month - user_date_loan_df.loan_month
	user_date_loan_df['left_plannum'] = user_date_loan_df.apply( lambda x : -1 if x['delta_month'] <= 0 else x['plannum'] - x['delta_month'] , axis = 1)
	user_date_loan_df['left_balance'] =  user_date_loan_df.apply(lambda x : 0 if x.left_plannum < 0 else x.real_loan_amount / x.plannum * x.left_plannum , axis = 1)
	user_date_loan_df['month_need_pay'] = user_date_loan_df.apply(lambda x : 0 if x.left_plannum < 0 else x.real_loan_amount , axis=1)

	user_month_left_banalce = user_date_loan_df.groupby(['uid','month']).agg({"left_balance":np.sum , 'month_need_pay':np.sum , 'real_limit':np.max}).reset_index()

	print "user_month_left_banalce.size = {}".format(user_month_left_banalce.shape[0])

	#step 4. user month loan sum 
	user_month_loan_sum_df = loan_df.groupby(['uid','loan_month']).agg({"real_loan_amount":np.sum }).reset_index()

	#step 5. user_month_tuned_real limiit 
	agg = pd.merge(user_month_loan_sum_df , user_month_loan_sum_df , on=['uid'])
	agg = agg[agg.loan_month_x >= agg.loan_month_y]
	user_month_tuned_real_limit_df = agg.groupby(['uid', 'loan_month_x']).agg({"real_loan_amount_y":np.max }).reset_index().sort_values(['uid','loan_month_x']).rename(columns = {"loan_month_x":"month" , "real_loan_amount_y":'tuned_limit'})


    #step 6. merge left balance and user month loan sum 
	user_month_df = pd.merge(user_month_left_banalce  , user_month_loan_sum_df , left_on =['uid', 'month'] , right_on=['uid','loan_month'] ,how='left').fillna(0)
	#step 7. add user month tuned limit
	user_month_df = pd.merge(user_month_df ,user_month_tuned_real_limit_df , on=['uid','month'], how='left' ).fillna(0)
	user_month_df.tuned_limit = user_month_df.apply(lambda x: x.real_limit if x.tuned_limit < x.real_limit else x.tuned_limit, axis=1)

	print "user_month_df.size={}".format(len(user_month_df))
	del user_month_df['loan_month']    
    #step 8. add date dim which represents date-date+30
	user_month_df.to_csv("fea_user_month.csv")
	user_date_df = gen_user_date(user_df , loan_df )
	print  "user_date_df.size = {}".format(user_date_df.shape[0])
    #step9. join user_date_df , user_month_df


	user_date_df = pd.merge(user_date_df , user_month_df , on = ['uid','month'])
	print user_date_df.columns , len(user_date_df)
    #step 10. output
	user_date_df.to_csv(fea_fn , index=False)
def merge(a , b):
	return pd.merge(a , b , on =['uid','date'])

if __name__ == '__main__':
	st = datetime.datetime.now()
	user_fn = '../../dataset/t_user.csv'
	click_fn = '../../dataset/t_click.csv'
	order_fn = '../../dataset/t_order.csv'
	loan_fn = '../../dataset/t_loan.csv'
	fea_fn = '../../fea/fea_user_month_date.csv'
	if len(sys.argv) != 7:
		print sys.argv[0] + '\t[user_fn]\t[click_fn]\t[order_fn]\t[loan_fn]\t[fea_fn]'
	else:
		user_fn = sys.argv[1]
		click_fn = sys.argv[2]
		order_fn = sys.argv[3]
		loan_fn = sys.argv[4]
		fea_fn = sys.argv[6]
	gen_fea(user_fn, click_fn, order_fn, loan_fn, fea_fn)
	et = datetime.datetime.now()
	print 'time cost : ' + str(et - st)

