# !/usr/bin/env python2
#  -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import lib.minimization as min
import matplotlib.pyplot as plt


def upload_data(fname):
	"""Upload data stored in a file, shuffle it and convert it into a numpy matrix
	
	Args:
		fname: (string) name of the data file
	
	Returns:
		data: Numpy matrix of shuffled data loaded from the data file
		
	"""
	# upload data and create the data matrix
	data_file = pd.read_csv(fname)
	data = data_file.values
	
	for i in range(4):
		data = np.delete(data, 0, 1)
	
	# randomly shuffle the data rows
	np.random.shuffle(data)
		
	return data
	

def prepare_data(mat):
	"""Slice the data matrix into three submatrices:
	data_train, containing the 50% of the original data
	data_val, containing the 25% of the resting original data
	data_test, containing the resting 25% of the original data
		
	Args:
		mat: Numpy matrix to slice
	"""
	global data_train
	global data_val
	global data_test	
	
	data_train = mat[0 : int((len(mat)*.5))]		# 50% of data[] 
	temp = mat[int(len(mat)*.5) : int(len(mat))]
	data_val = temp[0 : int(len(temp)*.5)]		# 25% of data[]
	data_test = temp[int(len(temp)*.5) : len(temp)]		# 25% of data[]
	
	
def normalize_data(data_train, data_val, data_test):
	"""Perform the gaussian normalization of the three submatrices.
	The normalization of the validation matrix and of the test one is 
	performed by using the same mean and variance values of the train matrix
	
	Args:
		data_train: Numpy train submatrix
		data_val: Numpy validation submatrix
		data_test: Numpy test submatrix
	"""	
	global data_train_norm
	global data_val_norm
	global data_test_norm

	# initialize the normalized submatrix
	data_train_norm = np.zeros((data_train.shape), dtype=float)
	data_val_norm = np.zeros((data_val.shape), dtype=float)
	data_test_norm = np.zeros((data_test.shape), dtype=float)
	
	# gaussian normalization of column
	for i in range(data_train.shape[1]):
		mean = np.mean(data_train[:, i])
		variance = np.var(data_train[:, i])
		
		# normalize all the sumbatrix
		data_train_norm[:, i] = (data_train[:, i]-mean) / np.sqrt(variance)
		data_val_norm[:, i] = (data_val[:, i]-mean) / np.sqrt(variance)
		data_test_norm[:, i] = (data_test[:, i]-mean) / np.sqrt(variance)
	

def denormalize_data(norm_vect):
	"""Perform the gaussian denormalization of the three submatrices.
	The denormalization of the validation matrix and of the test one is performed 
	by using the same mean and variance values of the train matrix.
	
	Args:
		data_train: Numpy normalized train submatrix
		data_val: Numpy normalized validation submatrix
		data_test: Numpy normalized test submatrix
	
	Returns:
		data_train_denorm: denormalized train submatrix
		data_val_denorm: denormalized validation submatrix
		data_test_denorm: denormalized test submatrix
	"""	
	# gaussian normalization of column
	mean = np.mean(data_train[:, F0])
	variance = np.var(data_train[:, F0])
		
	# normalize all the sumbatrix
	denorm_vect = (norm_vect*np.sqrt(variance)) + mean
	
	return denorm_vect

	
def regression(F0, flag):
	"""Determine the y vector and the X matrix used in the regression formula.
	Use the methods contained into the "lab/minimization.py" file to perform
	different kinds of regression. The regression technique is passed to the 
	function in the __main__.
	
	Args:
		F0: (int) column of the data matrix where the regressand is contained
		flag: (int) indicates the regression technique used
	
	Returns:
		w: Numpy vector. Generated w
	"""
	global y_train
	global y_test
	global y_val
	global X_train
	global X_test
	global X_val
	global opt_w_grad
	global opt_w_steep
	global opt_w_stoc
	
	# determine the y vector
	y_train = data_train_norm[:, [F0]]
	y_test = data_test_norm[:, [F0]]
	y_val = data_val_norm[:, [F0]]
	
	# determine the X matrix
	X_train = np.delete(data_train_norm, F0, 1)
	X_test = np.delete(data_test_norm, F0, 1)
	X_val = np.delete(data_val_norm, F0, 1)

	if flag == 1:
		# perform LLS pseudoinverse
		lls = min.SolveLLS(y_train, X_train)
		print "\n\n>> performing LLS...\n"
		lls.run()
		w = lls.sol
	
	elif flag == 2:
		# perform Conjugate Gradient algorithm
		conj = min.Conj(y_train, X_train)
		print ">> performing Conjugate Gradient...\n"
		conj.run()
		w = conj.sol
		
	elif flag == 3:
		# perform Gradient Descent algorithm
		grad = min.SolveGrad(y_train, X_train)
		print ">> performing Gradient Descent optimization..."
		opt_nit = grad.optimization(gamma, Nit, y_val, X_val) # set a stopping condition
		print "   optimum number of iterations: " + str(opt_nit)
		
		# perform Gradient Descent algorithm
		print "   performing Gradient Descent...\n"
		grad.run(gamma, opt_nit)
		w = grad.sol
		grad.save_data(path, "Grad")
		
		dataFrame = pd.DataFrame({
			"val_err":grad.val_err[:,0],
			"train_err":grad.train_err[:,0]
			})
		dataFrame.to_csv(path+'grad_validation.dat')
				
	elif flag == 4:
		# perform Stochastic Descent algorithm
		stoc = min.StocGrad(y_train, X_train)
		print ">> performing Stochastic Gradient Descent...\n"
		stoc.run(gamma, Nit)
		w = stoc.sol
		stoc.save_data(path, "Stoc")
				
	elif flag == 5:
		# perform Steepest Descent algorithm
		steep = min.SolveSteepDesc(y_train, X_train)
		print ">> performing Steepest Descent...\n"
		steep.run(Nit)
		w = steep.sol
		steep.save_data(path, "Steep")
				
	elif flag == 6:
		# perform Ridge Regression algorithm
		ridge = min.RidgeReg(y_train, X_train)
		
		# determine the optimum lambda by using validation set
		print ">> performing lambda optimization..."
		opt_lamb = ridge.find_lamb(lamb_max, y_val, X_val)
		print "   optimum lambda: " + str(opt_lamb)
		ridge.save_data(path, "Ridge_lambda")
		
		# perform the algorithm with the optimum lambda
		print "   performing Ridge Regression...\n"
		ridge.run(opt_lamb)
		w = ridge.sol

	y_train = denormalize_data(y_train)
	y_test = denormalize_data(y_test)
	y_val = denormalize_data(y_val)
		
	return w


def process_y(y, X, w):
	"""Estimate the new y by using the model learnt from the training set and then
	determine the mean squared error.
	
	Args:
		y: y array of the regression formula
		X: X matrix of the regression formula
		w: generated w of the regression formula
	
	Returns:
		yhat: new y array estimated by using the generated w 
		delta_y: difference between the estimated array yhat and the real y array
		mserr: mean square error of the delta_y
	"""
	# determine yhat
	yhat= np.dot(X, w)
	yhat = denormalize_data(yhat)
	# determine the error between y and yhat
	delta_y = y - yhat
	mserr = (np.linalg.norm(delta_y)**2) / len(X[:, 0])
	
	return yhat, delta_y, mserr
	

# _______MAIN_______

# plot folder initialization and path definition
main_folder = os.getcwd()
if not os.path.isdir(main_folder+"/output"):
	os.mkdir ("output")
path = main_folder + "/output/"

np.random.seed(30)

gamma = 1e-6
Nit = 600
lamb_max = 600
mserr = []

# upload data
data = upload_data("data/parkinsons_updrs.data")
# prepare data	
prepare_data(data)	
# normalize data
normalize_data(data_train, data_val, data_test)
# 1st column -> total UPDRS
F0 = 1	

# Least Linear Squares (LLS)
w_LLS = regression(F0, 1)
yhat_LLS_train, delta_y_LLS_train, mserr_LLS_train = process_y(y_train, X_train, w_LLS)
yhat_LLS_test, delta_y_LLS_test, mserr_LLS_test = process_y(y_test, X_test, w_LLS)
yhat_LLS_val, delta_y_LLS_val, mserr_LLS_val = process_y(y_val, X_val, w_LLS)

# conjugate gradient
w_conj = regression(F0, 2)
yhat_conj_train, delta_y_conj_train, mserr_conj_train = process_y(y_train, X_train, w_conj)
yhat_conj_test, delta_y_conj_test, mserr_conj_test = process_y(y_test, X_test, w_conj)
yhat_conj_val, delta_y_conj_val, mserr_conj_val = process_y(y_val, X_val, w_conj)

# gradient descent
w_grad = regression(F0, 3)
yhat_grad_train, delta_y_grad_train, mserr_grad_train = process_y(y_train, X_train, w_grad)
yhat_grad_test, delta_y_grad_test, mserr_grad_test = process_y(y_test, X_test, w_grad)
yhat_grad_val, delta_y_grad_val, mserr_grad_val = process_y(y_val, X_val, w_grad)

# stochastic gradient descent (SGD)
w_stoc = regression(F0, 4)
yhat_stoc_train, delta_y_stoc_train, mserr_stoc_train = process_y(y_train, X_train, w_stoc)
yhat_stoc_test, delta_y_stoc_test, mserr_stoc_test = process_y(y_test, X_test, w_stoc)
yhat_stoc_val, delta_y_stoc_val, mserr_stoc_val = process_y(y_val, X_val, w_stoc)

# steepest descent
w_steep = regression(F0, 5)
yhat_steep_train, delta_y_steep_train, mserr_steep_train = process_y(y_train, X_train, w_steep)
yhat_steep_test, delta_y_steep_test, mserr_steep_test = process_y(y_test, X_test, w_steep)
yhat_steep_val, delta_y_steep_val, mserr_steep_val = process_y(y_val, X_val, w_steep)

# rigde regression
w_ridge = regression(F0, 6)
yhat_ridge_train, delta_y_ridge_train, mserr_ridge_train = process_y(y_train, X_train, w_ridge)
yhat_ridge_test, delta_y_ridge_test, mserr_ridge_test = process_y(y_test, X_test, w_ridge)
yhat_ridge_val, delta_y_ridge_val, mserr_ridge_val = process_y(y_val, X_val, w_ridge)

# save the optimum w vectors in a file
dataFrame = pd.DataFrame({
	'LLS_w': w_LLS[:,0], 
	'conj_w': w_conj[:,0], 
	'grad_w': w_grad[:,0], 
	'stoc_w': w_stoc[:,0], 
	'steep_w': w_steep[:,0], 
	'ridge_w': w_ridge[:,0]
	})
dataFrame.to_csv(path+"generated_w.dat")

# save the train y infos in a file
dataFrame = pd.DataFrame({
	'y_train': y_train[:,0], 
	'yhat_LLS_train': yhat_LLS_train[:,0], 
	'delta_LLS_train': delta_y_LLS_train[:,0], 
	'yhat_conj_train': yhat_conj_train[:,0], 
	'delta_conj_train': delta_y_conj_train[:,0], 
	'yhat_grad_train': yhat_grad_train[:,0], 
	'delta_grad_train': delta_y_grad_train[:,0], 
	'yhat_stoc_train': yhat_stoc_train[:,0], 
	'delta_stoc_train': delta_y_stoc_train[:,0], 
	'yhat_steep_train': yhat_steep_train[:,0], 
	'delta_steep_train': delta_y_steep_train[:,0], 
	'yhat_ridge_train': yhat_ridge_train[:,0], 
	'delta_ridge_train': delta_y_ridge_train[:,0]
	})
dataFrame.to_csv(path+"train_y.dat")

# save the test y infos in a file
dataFrame = pd.DataFrame({
	'y_test': y_test[:,0], 
	'yhat_LLS_test': yhat_LLS_test[:,0], 
	'delta_LLS_test': delta_y_LLS_test[:,0], 
	'yhat_conj_test': yhat_conj_test[:,0], 
	'delta_conj_test': delta_y_conj_test[:,0], 
	'yhat_grad_test': yhat_grad_test[:,0], 
	'delta_grad_test': delta_y_grad_test[:,0], 
	'yhat_stoc_test': yhat_stoc_test[:,0], 
	'delta_stoc_test': delta_y_stoc_test[:,0], 
	'yhat_steep_test': yhat_steep_test[:,0], 
	'delta_steep_test': delta_y_steep_test[:,0], 
	'yhat_ridge_test': yhat_ridge_test[:,0], 
	'delta_ridge_test': delta_y_ridge_test[:,0]
	})
dataFrame.to_csv(path+"test_y.dat")

# save the mean squared errors in a latex table file
dataFrame = pd.DataFrame(
	data=[
		[mserr_LLS_train, mserr_LLS_test, mserr_LLS_val], 
		[mserr_conj_train, mserr_conj_test, mserr_conj_val], 
		[mserr_grad_train, mserr_grad_test, mserr_grad_val], 
		[mserr_stoc_train, mserr_stoc_test, mserr_stoc_val], 
		[mserr_steep_train, mserr_steep_test, mserr_steep_val], 
		[mserr_ridge_train, mserr_ridge_test, mserr_ridge_val]
		], 
	columns=[
		'Train Dataset', 
		'Test Dataset', 
		'Validation Dataset'
		], 
	index=[
		'LLS Pseudoinverse', 
		'Conjugate Gradient', 
		'Gradient Descent', 
		'Stochastic Gradient', 
		'Steepest Descent', 
		'Ridge Regression'
		]
	)
dataFrame.to_latex(path+"mserr.dat")
