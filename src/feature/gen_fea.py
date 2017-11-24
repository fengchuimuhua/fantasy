import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def generate_fea(user_fn, click_fn, order_fn, loan_fn, loan_sum_fn, fea_fn):
	# step 0. INIT
	user_df = pd.read_csv(user_fn)
	click_df = pd.read_csv(click_fn)
	order_df = pd.read_csv(order_fn)
	loan_df = pd.read_csv(loan_fn)
	loan_sum_df = pd.read_csv(loan_sum_fn)
	# step 1. 

if __name__ == '__main__':
	user_fn = '../../dataset/t_user.csv'
	click_fn = '../../dataset/t_click.csv'
	order_fn = '../../dataset/t_order.csv'
	loan_fn = '../../dataset/t_loan.csv'
	loan_sum_fn = '../../dataset/t_loan_sum.csv'
	fea_fn = 'fea.csv'
	if len(sys.argv) != 7:
		print sys.argv[0] + '\t[user_fn]\t[clk_fn]\t[ord_fn]\t[load_fn]\t[loan_sum_fn]\t[fea_fn]'
	else:
		user_fn = sys.argv[1]
		click_fn = sys.argv[2]
		order_fn = sys.argv[3]
		loan_fn = sys.argv[4]
		loan_sum_fn = sys.argv[5]
		fea_fn = sys.argv[6]
	generate_fea(user_fn, click_fn, order_fn, loan_fn, loan_sum_fn, fea_fn)