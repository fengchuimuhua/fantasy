import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import fea_utils as futils

def gen_age_date_feature(user_fn, click_fn, order_fn, loan_fn, fea_fn):
	
	# 读取数据
	df_user = pd.read_csv(user_fn, sep=',');
	df_loan = pd.read_csv(loan_fn, sep=',');
	df_click = pd.read_csv(click_fn, sep=',');
	df_order = pd.read_csv(order_fn, sep=',');

	# 生成user * date的关于点击次数的表
	date_all = df_loan['loan_time'].str.split(' ', expand=True)[0]
	start_date = pd.to_datetime(date_all.min(), format='%Y-%m-%d')
	end_date = pd.to_datetime(date_all.max(), format='%Y-%m-%d')
	date_list = map(lambda d : datetime.strftime(d, '%Y-%m-%d'), pd.date_range(start_date, end_date).tolist())
	df_date = pd.DataFrame(data={'date' : date_list})

	df_user['key'] = 1
	df_date['key'] = 1
	df_user_date = pd.merge(df_user, df_date, on='key')
	del df_user_date['key']

	df_click['date'] = df_click['click_time'].map(lambda ct : ct.split(' ')[0])
	df_click_num = pd.DataFrame({'click_num': df_click.groupby(['uid', 'date']).size()}).reset_index()
	df_user_date_cnk = pd.merge(df_user_date, df_click_num, on=['uid', 'date'], how='left')

	df_user_date_cnk['uid_real'] = df_user_date_cnk.apply(lambda x : np.nan if pd.isnull(x.click_num) else int(x.uid), axis=1)
	df_user_date_cnk['click_num'] = df_user_date_cnk['click_num'].fillna(value=0)

	# 生成age * date的表
	ages = set(df_user.age)
	df_age = pd.DataFrame({'age': list(ages)})
	df_age = df_age.sort_values(['age'])

	df_age['key'] = 1
	df_date['key'] = 1

	df_age_date = pd.merge(df_age, df_date, on=['key'], how='left')
	del df_age_date['key']
	
	# 生成age * date * user 的大表
	df_mix = pd.merge(df_age_date, df_user_date_cnk, on=['date'], how='left')


	# 以3天为例，生成age * date的特征表
	# 其中total_cnk_3为某个年龄段用户（age）在某个日期（date）前3天（包含当前日期）的用户点击量总量
	# 其中total_cnu_3为某个年龄段用户（age）在某个日期（date）前3天（包含当前日期）有点击的user个数

	k = 3  # 统计历史前k天的结果，包含当天

	age_list = []
	date_list = []
	total_cnk_list = []
	total_cnu_list = []

	for age_index, age_row in df_age.iterrows():
	    for date_index, date_row in df_date.iterrows():
	        
	        curr_timestamp = pd.to_datetime(date_row['date'], format="%Y-%m-%d")
	        drange = pd.date_range(end=curr_timestamp, periods=k, freq='D')
	        date_list = map(lambda d : datetime.strftime(d, '%Y-%m-%d'), drange.tolist())
	        
	        if not (date_list[0] in date_list_total):
	            age_list.append(age_row['age'])
	            date_list.append(date_row['date'])
	            total_cnk_list.append(-1)
	            total_cnu_list.append(-1)
	            continue
	        
	        condition = (df_mix['date'] == date_list[0])
	        for idx in range(1, len(date_list)):
	            condition = condition | (df_mix['date'] == date_list[idx])
	        condition = condition & (df_mix['age_x'] == age_row['age'])
	        
	        df_mix_part = df_mix[condition]

	        total_click_num = df_mix_part['click_num'].sum()
	        
	        s = set(df_mix_part['uid_real'].tolist())
	        if np.nan in s:
	            s.remove(np.nan)
	        total_click_user = len(s)
	        
	        age_list.append(age_row['age'])
	        date_list.append(date_row['date'])
	        total_cnk_list.append(total_click_num)
	        total_cnu_list.append(total_click_user)

	df_res = pd.DataFrame({'age':age_list, 'date':date_list, 'total_cnk_3':total_cnk_list, 'total_cnu_3':total_cnu_list})
	df_res.to_csv(fea_fn, index=False)

if __name__ == '__main__':
	st = datetime.datetime.now()
	
	# 数据表地址
	user_fn = '../../dateset/t_user.csv'
	click_fn = '../../dataset/t_click.csv'
	order_fn = '../../dataset/t_order.csv'
	loan_fn = '../../dataset/t_loan.csv'

	# 输出结果：age * date 维度特征表
	fea_fn = '../../fea/fea_age_date.csv'

	if len(sys.argv) != 6:
		print sys.argv[0] + '\t[user_fn]\t[click_fn]\t[order_fn]\t[loan_fn]\t[fea_fn]'
	else:
		user_fn = sys.argv[1]
		click_fn = sys.argv[2]
		order_fn = sys.argv[3]
		loan_fn = sys.argv[4]
		fea_fn = sys.argv[5]

	gen_age_date_feature(user_fn, click_fn, order_fn, loan_fn, fea_fn)

	et = datetime.datetime.now()
	print 'time cost : ' + str(et - st)