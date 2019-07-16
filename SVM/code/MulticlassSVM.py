# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

from sklearn import datasets

import matplotlib.pyplot as plt 

def GaussKernel( X1, X2, gamma ):
	'''
	Generate the RBF kernel for X1 and X2.
	When X1 = X2 = Input Data, it generates the RBF(x, x) for training
	When X1 = Input Data and X2 = Predicted Data, it generates the RBF(x, z) for prediction
	:param X1: The data tensor.
	:param X2: The data tensor.
	:param gamma: Hyperparameter (Negative Number).
	:return type Kernel: tensor
	'''
	
	sq_X1 = tf.reshape(tf.reduce_sum(tf.square(X1), 1), [-1,1])
	sq_X2 = tf.reshape(tf.reduce_sum(tf.square(X2), 1), [-1,1])
	sq_dist = tf.add(tf.subtract(sq_X1, tf.multiply(2., tf.matmul(X1,tf.transpose(X2)))), tf.transpose(sq_X2))

	#sq_dist = ||x_i - x_j||^2 = 2x_ix_j
	# sq_dist = tf.multiply(2., tf.matmul(X, tf.transpose(X)))

	return tf.exp( tf.multiply(gamma, tf.abs(sq_dist)) )

def CrossLabel( Y, numClass, numSample ):
	'''
	Generate the cross class label matrix for Y
	:input Y: The label tensor in shape (# class, # samples)
	:input numClass:
	:input numSample:
	:return type Loss: tensor in shape (# class, # samples, # samples)
	'''
	m1 = tf.expand_dims( Y, 1 )
	# m1 = tf.reshape( Y, [numClass, 1, numSample] )
	m2 = tf.reshape( m1, [numClass, numSample, 1] )
	return tf.matmul( m2, m1 )

def GaussLoss( Y, Kernel, alpha, numClass, numSample ):
	'''
	Create the SVM loss using dual op
	:input Y: The label tensor.
	:input Kernel: The kernel tensor.
	:input alpha: The variable for dual optimization.
	:input numClass: Y.shape[0] (Used in function CrossLabel)
	:input numSample:: Y.shape[1] (Used in function CrossLabel)
	:return type Loss: tensor
	'''
	term1 = tf.reduce_sum(alpha)
	
	cross_alpha = tf.matmul(tf.transpose(alpha), alpha)
	cross_y = CrossLabel( Y, numClass, numSample )
	term2 = tf.reduce_sum(tf.multiply(Kernel, tf.multiply(cross_alpha, cross_y)), [1,2])

	return tf.reduce_sum( tf.subtract(term2, term1) )


def train( rate, loss ):
	'''
	Optimize the loss function using gradient descent.
	:input rate: The learning rate 
	:input loss: The loss/cost (tensor)
	:return type Optimizer: tensor
	'''
	opt = tf.compat.v1.train.GradientDescentOptimizer( learning_rate = rate )
	return opt.minimize( loss )

def GaussPrediction( Y, Kernel, alpha ):
	'''
	Get the predicted value.
	:input Y: The label tensor.
	:input Kernel: The predicted kernel tensor.
	:input alpha: The variable for dual optimization.
	'''
	Y_hat = tf.matmul(tf.multiply(Y , alpha), Kernel)
	predicted = tf.argmax(Y_hat - tf.expand_dims(tf.reduce_mean(Y_hat, 1), 1), 0)
	return predicted

def PlotData( x, iris ):
	'''
	Visualize the given data points (when m = 2)
	:input x: The input data (narray)
	:input iris: The original dataset Iris (narray).
	'''

	class0 = np.array([ x[i] for i in range(len(x)) if iris.target[i] == 0 ])
	class1 = np.array([ x[i] for i in range(len(x)) if iris.target[i] == 1 ])
	class2 = np.array([ x[i] for i in range(len(x)) if iris.target[i] == 2 ])

	plt.scatter(class0[:,0], class0[:,1], color = 'blue', label = 'First Class') 
	plt.scatter(class1[:,0], class1[:,1], color = 'black', label = 'Second Class') 
	plt.scatter(class2[:,0], class2[:,1], color = 'green', label = 'Third Class') 

	plt.title('Multi-Class SVM') 
	plt.legend() 
	plt.show() 


if __name__ == '__main__':

	## Get dataset
	iris = datasets.load_iris()
	x = np.array([[x[0], x[3]] for x in iris.data])
	y_c1 = np.array([1 if y == 0 else -1 for y in iris.target])
	y_c2 = np.array([1 if y == 1 else -1 for y in iris.target])
	y_c3 = np.array([1 if y == 2 else -1 for y in iris.target])
	y = np.array([y_c1, y_c2, y_c3]) 
	
	# PlotData( x_vals, iris )

	## Initialize the training variables
	alpha = tf.Variable(tf.random_normal([y.shape[0], x.shape[0]]))

	# Launch the graph in a session.
	with tf.compat.v1.Session() as sess:
		# Initialize the variables W and b.
		sess.run( tf.global_variables_initializer() )
		gamma, beta, epochs = -10., 0.01, 500

		# Create the input tensors
		X = tf.compat.v1.placeholder(tf.float32, shape=[None, x.shape[1]])
		Y = tf.compat.v1.placeholder(tf.float32, shape=[y.shape[0], None])
		Pred = tf.compat.v1.placeholder(tf.float32, shape=[None, x.shape[1]])

		# Training
		rbf_kernel = GaussKernel( X, Pred, gamma )
		cost = GaussLoss( Y, rbf_kernel, alpha, y.shape[0], y.shape[1] )
		opt = train( beta, cost )

		# Prediction and accuracy
		predict = GaussPrediction( Y, rbf_kernel, alpha )
		acc = tf.reduce_mean( tf.cast( tf.equal(predict, tf.argmax(Y, 0)), dtype=tf.float32) )

		for epoch in range( epochs ):
			sess.run( opt, feed_dict = {X : x, Y : y, Pred: x} )
			if epoch % 50 == 0:
				c, a = sess.run( [cost, acc], feed_dict = {X : x, Y : y, Pred: x} )  
				print "Epoch : ", epoch, ", cost =", c, ", accuracy =", a
		
		# Plotting the Results
		min_x, max_x = min( x[:, 0] ) - 1, max( x[:, 0] ) + 1
		min_y, max_y = min( x[:, 1] ) - 1, max( x[:, 1] ) + 1
		grid_xx, grid_yy = np.meshgrid( np.arange(min_x, max_x, 0.02), np.arange(min_y, max_y, 0.02) )
		grid_vals = np.c_[ grid_xx.reshape( -1 ), grid_yy.reshape( -1 ) ] #Shape N*2
		predict_grid = sess.run( predict, feed_dict = {X : x, Y : y, Pred: grid_vals} )
		predict_grid = predict_grid.reshape( grid_xx.shape )

		plt.contourf(grid_xx, grid_yy, predict_grid, cmap=plt.cm.Paired, alpha=0.8)
		PlotData( x, iris )