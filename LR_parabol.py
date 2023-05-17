import numpy as np
import matplotlib.pyplot as plt

def main():
	# create data randomly 
	b = [2,5,7,9,11,16,19,23,22,29,29,35,37,40,46,42,39,31,30,28,20,15,10,6] # target value
	A = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25] # feature 
	plt.plot(A, b, 'ro', label='Data point')

	# transpose vector 
	A = np.array([A]).T
	b = np.array([b]).T

	#combine vectors
	x_square = np.array([A[:,0]**2]).T
	A = np.concatenate((A, x_square), axis=1)
	ones = np.ones((A.shape[0],1), dtype=np.int8)
	A = np.concatenate((ones, A), axis=1)

	# LR formula -> c: intercept; b,a: coefficient
	x = np.linalg.inv(A.transpose().dot(A)).dot(A.transpose()).dot(b) # -> c,b,a

	x_init = np.linspace(1,25,10000) # initialise first and final point of parabole 
	y0 = x[0][0] + x[1][0]*x_init + x[2][0]*x_init*x_init	# parabole equation: c + bx + ax^2
	
	# test 
	x_test = 20
	y_test = x[0][0] + x[1][0]*x_test + x[2][0]*x_test*x_test
	print("Predict value: {0}, target value is 31".format(y_test))
	plt.plot(x_test,y_test,'ro', color='black', label='Predict')

	plt.plot(x_init, y0, label='Fitting parabole')
	plt.legend(loc='upper left')
	plt.show()

main()