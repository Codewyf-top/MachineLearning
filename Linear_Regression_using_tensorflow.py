# -*- coding: utf-8 -*-
"""
@Time ： 2020/10/24 12:03 上午
@Auth ： Codewyf
@File ：Linear_Regression_using_tensorflow.py
@IDE ：PyCharm
@Motto：Go Ahead Instead of Heasitating

"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

np.random.seed(101)
tf.set_random_seed(101)

#Gererating random linear data
#There will be 50 data points ranging from 0 to 50
x = np.linspace(0, 50, 50)
y = 2*x + 3

#Adding noise to the random linear data
x += np.random.uniform(-1, 1, 50)
y += np.random.uniform(-1, 1, 50)

n = len(x)  #Number of data points
#Let us visualize the training data
#Plot of Training Data
plt.scatter(x, y)
plt.xlabel('x')
plt.xlabel('y')
plt.title("Training Data")
plt.show()

X = tf.placeholder("float")
Y = tf.placeholder("float")

W = tf.Variable(np.random.randn(), name = 'W') #weights
b = tf.Variable(np.random.randn(), name = 'b') #Bias

learning_rate = 0.01
training_epochs = 1000

#Hypothesis
y_pred = tf.add(tf.multiply(X, W), b)

#Mean Squared Error Cost Function
cost = tf.reduce_sum(tf.pow(y_pred-Y, 2)) / (2*n)

#Gradient Descent Optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

#Golbal Variables Initializer
init = tf.global_variables_initializer()

#Starting the Tensorflow Session
with tf.Session() as sess:

    #Initializing the Variables
    sess.run(init)

    #Interating through all the epochs
    for epoch in range(training_epochs):

        #Feeding each data point into the optimizer using Feed Dictionary
        for (_x, _y) in zip(x, y):
            sess.run(optimizer, feed_dict={X:_x, Y:_y})

        #Displaying the result after every 50 epoches
        if (epoch + 1) % 50 == 0:
            #Cauculating the cost at every epoch
            c = sess.run(cost, feed_dict={X:_x, Y:_y})
            print("Epoch", (epoch + 1), ": cost = ", c, "Weights = ", sess.run(W), "Bias = ", sess.run(b))

            #Storing necessary values to be used outside the Session
            training_cost = sess.run(cost, feed_dict={X:_x, Y:_y})
            weight = sess.run(W)
            bias = sess.run(b)

#Calculating the predictions
precictions = weight * x + bias
print("Training cost:", training_cost, "Weights = ", weight, "bias = ", bias, '\n')

plt.plot(x, y, 'ro', label = 'Original data')
plt.plot(x, precictions, label = 'Fitted line')
plt.title('Linear Regression Result')
plt.legend()
plt.show()