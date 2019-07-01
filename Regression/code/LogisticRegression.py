# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import pandas as pd 


import matplotlib.pyplot as plt 


def inputs(m):
	'''
	Define the placeholder for X and Y
	:return: tensors for X and Y
	'''
	X = tf.compat.v1.placeholder(tf.float32, shape=[None, m])
	Y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

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

def loss( y_hat, Y ):
	'''
	Create the loss/cost function
	:param y_hat: predicted Y
	:param Y: The input tensor (label).
	:return: the loss.
	'''

	return - tf.reduce_mean( Y*tf.math.log(y_hat) + (1-Y)*tf.math.log(1-y_hat) )
	# return tf.nn.sigmoid_cross_entropy_with_logits(logits = y_hat, labels = Y)

def train( rate, loss ):
	'''
	Optimize the loss function using gradient descent.
	:param rate: The learning_rate
	:param loss: The loss/cost
	:return: optimizing result
	'''

	opt = tf.compat.v1.train.GradientDescentOptimizer( learning_rate= rate )
	return opt.minimize( loss )

def accuracy( y_hat, Y ):
	'''
	Get the accuracy of the predicted value.
	:param y_hat: predicted Y
	:param Y: The input tensor (label).
	:return: accuracy
	'''

	predicted = tf.cast( y_hat > 0.5, dtype=tf.float32 )
	return tf.reduce_mean( tf.cast( tf.equal(predicted, Y), dtype=tf.float32) )

if __name__ == '__main__':
	# Import the dataset
	data = pd.read_csv('data/iris.csv', header = None)
	x = data.iloc[:, :-1].values 
	y = data.iloc[:, -1:].values

	n, m = len(x), len(x[0])

	
	## Initialize the training variables
	# Two different ways to initialize: random or 0
	W = tf.Variable(tf.random.normal([m, 1]), name='weight')
	b = tf.Variable(tf.random.normal([1]), name='bias')

	# Launch the graph in a session.
	with tf.compat.v1.Session() as sess:
		# Initialize the variables W and b.
		sess.run( tf.compat.v1.global_variables_initializer() )
		alpha, epochs = 0.0035, 500

		# Get the input tensors
		X, Y = inputs( m )

		y_hat = hypothesis( X, W, b )
		cost = loss( y_hat, Y )
		opt = train( alpha, cost )
		acc = accuracy( y_hat, Y )

		for epoch in range( epochs ):
			sess.run( opt, feed_dict = {X : x, Y : y} )
			if epoch % 50 == 0:
				c = sess.run( cost, feed_dict = {X : x, Y : y} ) 
				a = sess.run( acc, feed_dict = {X : x, Y : y} ) 
				print "Epoch : ", epoch, ", cost =", c, ", accuracy =", a
		'''
		# Get the optimized result
		training_cost = sess.run( cost, feed_dict ={X: x, Y: y} ) 
		weight = sess.run(W) 
		bias = sess.run(b) 
		
		print "Training cost =", training_cost, "Accuracy= ", traning_acc, "Weight =", weight, "bias =", bias, '\n'

		
		# Calculating the predictions 
		predictions = weight * x + bias 

		# Plotting the Results 
		plt.plot(x, y, 'ro', label ='Original data') 
		plt.plot(x, predictions, label ='Fitted line') 
		plt.title('Linear Regression Result') 
		plt.legend() 
		plt.show() 
		'''