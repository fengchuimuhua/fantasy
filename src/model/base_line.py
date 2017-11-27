
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# use NOV data to eval DEC performance
# A rmse: 2.277584
user_fn = '../../dataset/t_user.csv'
loansum_fn = '../../dataset/t_loan_sum.csv'
res_fn = '../../submit/1127_baseline.csv'
user_df = pd.read_csv(user_fn)
loansum_df = pd.read_csv(loansum_fn)
res_df = pd.merge(user_df, loansum_df, on='uid', how='left')
res_df['loan_sum'] = res_df['loan_sum'].fillna(value=0)
res_df.to_csv(res_fn, index=False, header=False, columns=['uid', 'loan_sum'])

loan_fn = '../../dataset/t_loan.csv'
loan_pd = pd.read_csv(loan_fn)
loan_pd['month'] = loan_pd['loan_time'].map(lambda lt : lt.split(' ')[0][ : 7])
uloan = loan_pd.groupby(['uid', 'month']).loan_amount.sum().reset_index()
res_df = pd.merge(user_df, uloan, on='uid', how='left')
res_df['loan_amount'] = res_df['loan_amount'].fillna(value=0)
res_fn = '../../submit/1127_baseline2.csv'
res_df.to_csv(res_fn, index=False, header=False, columns=['uid', 'loan_amount'])