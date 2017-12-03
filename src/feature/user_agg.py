
# coding: utf-8

# In[80]:
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dtfmt = '%Y-%m-%d %H:%M:%S' 
import timeit
import time
import itertools
import datetime
import lightgbm as lgbm
import math
from  sklearn.metrics import mean_squared_error
import fea_utils as futils


# In[149]:

def gen_fea(user_fn, click_fn, order_fn, loan_fn , loan_sum_fn,fea_fn):
    # step 0. INIT raw data
    user_df = pd.read_csv(user_fn)
    loan_df = pd.read_csv(loan_fn)
    loan_sum_df = pd.read_csv(loan_sum_fn)

    user_df['real_limit'] = user_df.limit.map(lambda la : futils.to_real_loan(la))
    
    #step 1 . convert load data to real data space     
    loan_df = transform(loan_df , 'loan_amount')

    loan_sum_df = transform(loan_sum_df , 'loan_sum')
    
    #step 2 . add loan month , dt 
    loan_df['month'] = loan_df.loan_time.apply(lambda x : x[5:7]).astype('int')
    
    #step 3. user per month
    poi_set = list(set(user_df.uid))
    months = [8 , 9 ,10 , 11]
    poi_month = []
    import itertools
    for x in itertools.product(poi_set , months):
        poi_month.append(x)
    poi_per_month_df = pd.DataFrame(poi_month , columns=['uid','month'])
    
    #step 4. user loan agg info, mean,max,median,

    func_names = ['sum','mean','std', 'min' ,'max' ,'median','mad','skew','kurt' ]
    funcs = [np.sum , np.mean , np.std , np.min , np.max , np.median ,pd.Series.mad , pd.Series.skew , pd.Series.kurt]
    column_names = [ 'real_loan_amount_' + t for t in func_names]
    user_statistics_df = loan_df.groupby(['uid']).agg({"real_loan_amount":funcs , 'plannum':funcs})
    user_statistics_amount_df = user_statistics_df['real_loan_amount'] 
    user_statistics_amount_df.columns = column_names
    user_statistics_amount_df = user_statistics_amount_df.reset_index() # add uid to column
    user_agg_df = user_statistics_amount_df
    
#     user_statistics_plannum_df = user_statistics_df['plannum']
#     column_names = [ 'plannum' + t for t in func_names]
#     user_statistics_plannum_df.columns = column_names
#     user_statistics_plannum_df = user_statistics_plannum_df.reset_index() #add uid to column 
    
#    user_agg_df = pd.merge(user_statistics_amount_df ,user_statistics_plannum_df , on=['uid'])
    
        
    #step 5. user month loan sum 
       
    user_month_sum_df = loan_df.groupby(['uid','month']).agg({"real_loan_amount": np.sum ,"plannum":np.sum }).reset_index()
    
    #setp 6. user month loan sum agg info. mean,max,....
    column_names = [ 'real_loan_amount_month_' + t for t in func_names]
    user_month_statistics_df = user_month_sum_df.groupby(['uid']).agg({"real_loan_amount":funcs , 'plannum':funcs})
    user_month_statistics_amount_df = user_month_statistics_df[['real_loan_amount']]
    user_month_statistics_amount_df.columns = column_names
    user_month_statistics_amount_df = user_month_statistics_amount_df.reset_index()
    user_month_agg_df = user_month_statistics_amount_df
    
#     column_names = [ 'plannum_month_' + t for t in func_names]
#     user_month_statistics_plannum_df = user_month_statistics_df[['plannum']]
#     user_month_statistics_plannum_df.columns = column_names
#     user_month_statistics_plannum_df = user_month_statistics_plannum_df.reset_index()
        
#     user_month_agg_df = pd.merge(user_month_statistics_amount_df ,user_month_statistics_plannum_df , on=['uid'] )
    
    
    
    #step 7. user month loan sum 
    
    flatten_user_month_sum_df = pd.merge( poi_per_month_df , user_month_sum_df , on=['uid' , 'month'] , how='left').fillna(0)
    flatten_user_month_sum_df = flatten_user_month_sum_df.fillna(0)
    
    flatten_user_month_sum_df = flatten_user_month_sum_df.set_index(['uid','month'])['real_loan_amount'].unstack().reset_index()
    flatten_user_month_sum_df.columns = [ 'uid','aug','sep','oct','nov']
    
    
    #step 8. feature agg 
    
    user_loan_feature_df = pd.merge(flatten_user_month_sum_df , user_agg_df , on=['uid'] , how='left' ).fillna(0)
    user_loan_feature_df = pd.merge(user_loan_feature_df , user_month_agg_df , on=['uid'] , how='left').fillna(0)
    user_loan_feature_df = pd.merge(user_loan_feature_df , loan_sum_df , on=['uid'] , how='left').fillna(0)
    user_loan_feature_df = pd.merge(user_loan_feature_df , user_df[['uid','real_limit','age','sex']] , on=['uid'])
    #step 9. save file 
    user_loan_feature_df[
        ['uid', 'aug', 'sep', 'oct', 'nov', 'real_loan_amount_sum', 'real_loan_amount_mean', 'real_loan_amount_std',
         'real_loan_amount_min', 'real_loan_amount_max', 'real_loan_amount_median', 'real_loan_amount_mad',
         'real_loan_amount_skew', 'real_loan_amount_kurt', 'real_loan_amount_month_sum', 'real_loan_amount_month_mean',
         'real_loan_amount_month_std', 'real_loan_amount_month_min', 'real_loan_amount_month_max',
         'real_loan_amount_month_median', 'real_loan_amount_month_mad', 'real_loan_amount_month_skew',
         'real_loan_amount_month_kurt', 'loan_sum', 'real_loan_sum', 'real_limit', 'age', 'sex']].to_csv(fea_fn)

def transform(data , column_name):
    new_column_name = "real_" + column_name
    new_column = data[column_name].apply(lambda x :5**x - 1)
    data[new_column_name] = new_column
    return data 


if __name__ == '__main__':
    st = datetime.datetime.now()
    user_fn = '../../../raw_data/t_user.csv'
    click_fn = '../../../raw_data/t_click.csv'
    order_fn = '../../../raw_data/t_order.csv'
    loan_sum_fn = '../../../raw_data/t_loan_sum.csv'
    loan_fn = '../../../raw_data/t_loan.csv'
    fea_fn = './user_agg.csv'
    if len(sys.argv) != 7:
        print sys.argv[0] + '\t[user_fn]\t[click_fn]\t[order_fn]\t[loan_fn]\t[loan_sum_fn]\t[fea_fn]'
    else:
        user_fn = sys.argv[1]
        click_fn = sys.argv[2]
        order_fn = sys.argv[3]
        loan_fn = sys.argv[4]
        loan_sum_fn = sys.argv[5]
        fea_fn = sys.argv[6]
    gen_fea(user_fn, click_fn, order_fn, loan_fn,loan_sum_fn, fea_fn)
    et = datetime.datetime.now()
    print 'time cost : ' + str(et - st)

