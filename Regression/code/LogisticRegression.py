# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import pandas as pd 


import matplotlib.pyplot as plt 


def inputs(m):
	'''
	Define the placeholder for X and Y
	:return type X: tensor
	:return type Y: tensor
	'''
	X = tf.compat.v1.placeholder(tf.float32, shape=[None, m])
	Y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

	return X, Y

def hypothesis( X, W, b ):
	'''
	Define the hypothesis
	:input X: The input tensor (data).
	:input W: The weight variable.
	:input b: The bias variable.
	:return: X*W + b.
	'''
	return tf.nn.sigmoid( tf.matmul(X, W) + b )

def loss( Y_hat, Y ):
	'''
	Create the loss/cost function
	:input Y_hat: The predicted tensor.
	:input Y: The label tensor.
	:return: the loss.
	'''

	return - tf.reduce_mean( Y*tf.math.log(Y_hat) + (1-Y)*tf.math.log(1-Y_hat) )
	# return tf.nn.sigmoid_cross_entropy_with_logits(logits = y_hat, labels = Y)

def train( rate, loss ):
	'''
	Optimize the loss function using gradient descent.
	:input rate: The learning_rate (tensor)
	:input loss: The loss/cost (tensor)
	:return: optimizing result
	'''

	opt = tf.compat.v1.train.GradientDescentOptimizer( learning_rate= rate )
	return opt.minimize( loss )

def accuracy( Y_hat, Y ):
	'''
	Get the accuracy of the predicted value.
	:input Y_hat: The predicted tensor.
	:input Y: The label tensor.
	:return: accuracy
	'''

	predicted = tf.cast( Y_hat > 0.5, dtype=tf.float32 )
	return tf.reduce_mean( tf.cast( tf.equal(predicted, Y), dtype=tf.float32) )


def plotData( x, y ):
	'''
	Visualize the given data points (when m = 2)
	:input x: The input data (narray).
	:input y: The input label (narray).
	'''
	x_pos = np.array([ x[i] for i in range(len(x)) if y[i] == 1 ])
	x_neg = np.array([ x[i] for i in range(len(x)) if y[i] == 0 ])

	plt.scatter(x_pos[:, 0], x_pos[:, 1], color = 'blue', label = 'Positive') 

	# Plotting the Negative Data Points 
	plt.scatter(x_neg[:, 0], x_neg[:, 1], color = 'red', label = 'Negative') 

	plt.title('Plot of given data and decision boundary') 
	plt.legend() 
	plt.show() 

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
				# print "Temp results:", sess.run( tf.cast( y_hat > 0.5, dtype=tf.float32 ), feed_dict = {X : x, Y : y} )
		
		# Get the optimized result
		weight = sess.run(W) 
		bias = sess.run(b) 

		# Plotting the Decision Boundary
		decision_boundary_x = np.array([min(x[:, 0]), max(x[:, 0])])
		decision_boundary_y = (- 1.0 / weight[0]) *(decision_boundary_x * weight + bias) 
		decision_boundary_y = [sum(decision_boundary_y[:, 0]), sum(decision_boundary_y[:, 1])] 
		plt.plot(decision_boundary_x, decision_boundary_y) 
		plotData( x, y )