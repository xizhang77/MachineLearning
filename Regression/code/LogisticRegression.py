# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import pandas as pd 


import matplotlib.pyplot as plt 


def inputs(m, n):
	'''
	Define the placeholder for X and Y
	:return: tensors for X and Y
	'''
	X = tf.compat.v1.placeholder(tf.float32, shape=[None, n])
	Y = tf.compat.v1.placeholder(tf.float32, shape=[None, m])

	return X, Y

def hypothesis( X, W, b ):
	'''
	Define the hypothesis
	:param X: The input tensor (data).
	:param W: The weight variable.
	:param b: The bias variable.
	:return: X*W + b.
	'''
	return tf.nn.sigmoid( tf.matmul(X, W) + b )

def loss( X, Y, W, b ):
	'''
	Create the loss/cost function
	:param X: The input tensor (data).
	:param Y: The input tensor (label).
	:param W: The weight variable.
	:param b: The bias variable.
	:return: the loss.
	'''
	y_hat = hypothesis(X, W, b) # also known as predicted Y

	# Calculating the cost 
	return tf.reduce_mean( Y*tf.log(y_hat) + (1-Y)*tf.log(1-y_hat) )
	# return tf.nn.sigmoid_cross_entropy_with_logits(logits = y_hat, labels = Y)

def train( rate, loss ):
	'''
	Optimize the loss function using gradient descent.
	:param rate: The learning_rate
	:param loss: The loss/cost
	:return: optimizing result
	'''

	opt = tf.train.GradientDescentOptimizer( learning_rate= rate )
	return opt.minimize( loss )

def accuracy( X, Y, W, b ):
	'''
	Get the accuracy of the predicted value.
	:return: accuracy
	'''
	y_hat = hypothesis(X, W, b)
	predicted = tf.equal( tf.argmax(y_hat, 1), tf.argmax(Y, 1) )

	return tf.reduce_mean( tf.cast(predicted, dtype=tf.float32) )

if __name__ == '__main__':
	# Import the dataset
	data = pd.read_csv('data/iris.csv', header = None)
	x = data.iloc[:, :-1].values 
	y = data.iloc[:, -1:].values

	n, m = len(x), len( set(y) )

	## Initialize the training variables
	# Two different ways to initialize: random or 0
	W = tf.Variable(tf.random_normal([n, m]), name='weight')
	b = tf.Variable(tf.random_normal([m]), name='bias')

	# W = tf.Variable(0.0, name='weight')
	# b = tf.Variable(0.0, name='bias')

	# Launch the graph in a session.
	with tf.compat.v1.Session() as sess:
		# Initialize the variables W and b.
		sess.run( tf.global_variables_initializer() )
		alpha, epochs = 0.0035, 500

		# Get the input tensors
		X, Y = inputs( m, n )

		cost = loss( X, Y, W, b )
		optimizer = train( alpha, cost )
		acc = accuracy( X, Y, W, b )

		for epoch in range( epochs ):
			sess.run( optimizer, feed_dict = {X : x, Y : y} )
			if epoch % 50 == 0:
				c = sess.run( cost, feed_dict = {X : x, Y : y} ) 
				a = sess.run( acc, feed_dict = {X : x, Y : y} ) 
				print "Epoch : ", epoch, ", cost =", c, "accuracy =", a
		'''
		# Get the optimized result
		training_cost = sess.run( cost, feed_dict ={X: x, Y: y} ) 
		weight = sess.run(W) 
		bias = sess.run(b) 
		print "Training cost =", training_cost, "Weight =", weight, "bias =", bias, '\n'

		# Calculating the predictions 
		predictions = weight * x + bias 

		# Plotting the Results 
		plt.plot(x, y, 'ro', label ='Original data') 
		plt.plot(x, predictions, label ='Fitted line') 
		plt.title('Linear Regression Result') 
		plt.legend() 
		plt.show() 
		'''