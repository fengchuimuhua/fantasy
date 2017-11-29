
import math

def to_real_loan(norm_loan_amount):
	if norm_loan_amount < 1:
		return -1
	else:
		return norm_loan_amount ** 5 - 1

def to_norm_loan(real_loan_amount):
	if real_loan_amount < 0:
		return -1
	elif real_loan_amount == 0:
		return 1
	else:
		return math.pow(real_loan_amount + 1, 0.2)