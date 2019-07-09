# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

from sklearn import datasets

import matplotlib.pyplot as plt 


def inputs( m ):
	'''
	Define the placeholder for X and Y
	:return type X: tensor
	:return type Y: tensor
	'''
	X = tf.compat.v1.placeholder(tf.float32, shape=[None, m])
	Y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

	return X, Y

def hyperplane( X, W, b ):
	'''
	Define the hyperplane
	:input X: The input tensor (data).
	:input W: The weight variable.
	:input b: The bias variable.
	:return: X*W + b.
	'''
	return tf.add(tf.matmul(X, W), b)

def loss( X, Y, W, b, beta ):
	'''
	Create the loss/cost function with l2 regularization
	:input X: The data tensor.
	:input Y: The label tensor.
	:input W: The weight variable (tensor).
	:input b: The bias variable (tensor).
	:input beta: The regularization parameter for weight variable
	:return: the loss.
	'''
	l2_norm = tf.reduce_sum( tf.square(W) )

	y_hat = hyperplane(X, W, b)
	classification_loss = tf.reduce_mean(tf.maximum(0., tf.subtract(1., tf.multiply(y_hat, y))))

	return tf.add(classification_loss, tf.multiply(beta, l2_norm))


def train( rate, loss ):
	'''
	Optimize the loss function using gradient descent.
	:input rate: The learning rate 
	:input loss: The loss/cost (tensor)
	:return: optimizing result
	'''

	opt = tf.compat.v1.train.GradientDescentOptimizer( learning_rate = rate )
	return opt.minimize( loss )

def accuracy( Y_hat, Y ):
	'''
	Get the accuracy of the predicted value.
	:input Y_hat: The predicted tensor.
	:input Y: The label tensor.
	:return: accuracy
	'''
	predicted = tf.sign( Y_hat )
	return tf.reduce_mean( tf.cast( tf.equal(predicted, Y), dtype=tf.float32) )

def plotData( x, y ):
	'''
	Visualize the given data points (when m = 2)
	:input x: The input data (narray).
	:input y: The input label (narray).
	'''
	x_pos = np.array([ x[i] for i in range(len(x)) if y[i] == 1 ])
	x_neg = np.array([ x[i] for i in range(len(x)) if y[i] == -1 ])

	plt.scatter(x_pos[:,0], x_pos[:,1], color = 'blue', label = 'Positive') 
	plt.scatter(x_neg[:,0], x_neg[:,1], color = 'red', label = 'Negative') 

	plt.title('Linear SVM') 
	plt.legend() 
	plt.show() 

if __name__ == '__main__':

	## Get dataset
	iris = datasets.load_iris()
	x = np.array([[x[0], x[3]] for x in iris.data])
	y = np.array([1 if label == 0 else -1 for label in iris.target]).reshape((x.shape[0],1))
	
	
	## Initialize the training variables
	W = tf.Variable(tf.random_normal([x.shape[1], 1]), name='weight')
	b = tf.Variable(tf.random_normal([1]), name='bias')

	# Launch the graph in a session.
	with tf.compat.v1.Session() as sess:
		# Initialize the variables W and b.
		sess.run( tf.global_variables_initializer() )
		alpha, beta, epochs = 0.01, 0.1, 2000

		# Get the input tensors
		X, Y = inputs( x.shape[1] )

		# cost = loss( X, Y, W, b )
		y_hat = hyperplane( X, W, b )
		cost = loss( X, Y, W, b, beta )
		opt = train( alpha, cost )
		acc = accuracy( y_hat, Y )

		for epoch in range( epochs ):
			sess.run( opt, feed_dict = {X : x, Y : y} )
			if epoch % 50 == 0:
				c, a = sess.run( [cost, acc], feed_dict = {X : x, Y : y} )  
				print "Epoch : ", epoch, ", cost =", c, ", accuracy =", a

		# Get the optimized result
		weight, bias = sess.run([W, b]) 
		x_line = [point[0] for point in x]

		# Find the separator line.
		y_line = [-weight[0]/weight[1]*i-bias/weight[1] for i in x_line]

		# Plotting the Results 
		plt.plot(x_line, y_line, 'r-', label='Separator' )
		plotData( x, y )


