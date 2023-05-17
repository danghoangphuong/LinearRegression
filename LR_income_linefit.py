import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

def lr_formula(X_train, y_train):
	return np.linalg.inv(X_train.transpose().dot(X_train)).dot(X_train.transpose()).dot(y_train)

def shuffle(income, happiness):
	#shuffle data
	randIndex = np.arange(income.shape[0])
	np.random.shuffle(randIndex)
	income = income[randIndex]
	happiness = happiness[randIndex]

	#slicing
	X_train = income[:400,:]
	X_test = income[400:,:] 
	y_train = happiness[:400] 
	y_test = happiness[400:]

	return X_train, X_test, y_train, y_test
def main():
	#load data
	data_frame = pd.read_csv("income.csv")
	income = data_frame.values[: , 1]
	happiness = data_frame.values[: , 2]
	plt.plot(income, happiness, 'ro', label='Data point')

	income = np.array([income]).T
	happiness = np.array([happiness]).T

	X_train, X_test, y_train, y_test = shuffle(income, happiness)
	
	#combine vector
	ones = np.ones(X_train.shape, dtype=np.int8)
	X_train = np.concatenate((ones, X_train), axis=1)

	x = lr_formula(X_train, y_train)

	# fitting line
	x0 = np.linspace(1,8,2) 
	y0 = x[0][0] + x[1][0]*x0

	predict = X_test[0]
	print("Predict with total income ", str(predict[0]))
	actual_val = x[0][0] + x[1][0]*predict
	print("Predict value: ", str(actual_val[0]))
	print("Target value: ", str(y_test[0][0]))

	plt.plot(predict, actual_val,"ro", label="Predict value", color="black")
	plt.plot(x0, y0, label='Fitting line')
	plt.legend(loc="lower right")
	plt.show()

main()