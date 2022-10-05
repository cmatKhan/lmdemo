import numpy as np
import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())

def linmodest(A:np.array, b:np.array) -> dict:
	"""Solve linear regression problem

	Args:
		A (np.array): a n x p matrix where n are the observations 
		and p are the predictor variables
		b (np.array): a n x 1 response vector

	Raises:
		ValueError: raised if the number of rows in A does not 
		match number of rows in 

	Returns:
		dict: A dict with keys coef (coefficients), 
		vcov (variance covariance), sigma (std), df (degrees of freedom)
	"""
	## write a function to solve simple linear regression problems

	# first check that the input is correct
	logging.debug("shape of A: %s; shape of y: %s" %(str(A.shape), str(b.shape))) #pylint: disable=W1201,C0209
	if not A.shape[0] == b.shape[0]:
		raise ValueError(f"STOP DOING THAT. number of rows {A.shape[0]} "\
			f"not equal to number of rows in response {b.shape[0]}")

	# compute the coefficients
	q,r = np.linalg.qr(A)
	coef = np.dot(np.linalg.inv(r),np.dot(q.T,b))

	# calculate the degrees of freedom. number of rows - number of columns
	df = A.shape[0] - A.shape[1]

	# compute sigma squared
	sigma2 = np.sum((b-np.matmul(A,coef))**2) / df

	# compute the variance covariance matrix
	vcov = sigma2 * np.linalg.inv(np.dot(r.T,r))

	return {k:v for k,v in zip(['coef', 'vcov', 'sigma', 'df'],
	                           [coef, vcov, np.sqrt(sigma2), df])}