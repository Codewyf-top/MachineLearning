import numpy as np
import tensorflow as tf
import math
import logging
# logging.basicConfig(level=logging.DEBUG)
import matplotlib.pyplot as plt

#Network parameters
n_hidden1 = 10
n_hidden2 = 10
n_input = 6
n_output = 4

#Learning parameters
learning_constant = 10
number_epochs = 1000
batch_size = 1000

#Defining the input and the output
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_output])

#DEFINING WEIGHTS AND BIASES

#Biases first hidden layer
b1 = tf.Variable(tf.random_normal([n_hidden1]))

#Biases second hidden layer
b2 = tf.Variable(tf.random_normal([n_hidden2]))

#Biases output layer
b3 = tf.Variable(tf.random_normal([n_output]))

#Weights connecting input layer with first hidden layer
w1 = tf.Variable(tf.random_normal([n_input, n_hidden1]))

#Weights connecting first hidden layer with second hidden layer
w2 = tf.Variable(tf.random_normal([n_hidden1, n_hidden2]))

#Weights connecting second layer with output layer
w3 = tf.Variable(tf.random_normal([n_hidden2, n_output]))

def multilayer_perceptron(input_d):
    #Task of neurons of first hidden layer
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(input_d, w1), b1))
    #Task of neurons of second hidden layer
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, w2), b2))
    #Task of neurons of output layer
    out_layer = tf.nn.softmax(tf.add(tf.matmul(layer_2, w3), b3))

    return out_layer
    
#Create model 
neural_network = multilayer_perceptron(X)

#Define loss an d optimizer
loss_op = tf.reduce_mean(tf.math.squared_difference(neural_network, Y))

# loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=neural_network,labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_constant).minimize(loss_op)

#Initializing the variables
init = tf.global_variables_initializer()

x_data=np.loadtxt('car_data.txt', encoding='UTF-16')
y_data=np.loadtxt('car_label.txt', encoding='UTF-16')

x = x_data[0:1000, :]
y = y_data[0:1000, 0:-1]

x_test = x_data[1001:, :]
y_test = y_data[1001:, 0:-1]

label_train = y_data[0:1000, -1]
label_test = y_data[1001:, -1]

with tf.Session() as sess:
    sess.run(init)
    #Training epoch 
    for epoch in range(number_epochs):
        sess.run(optimizer, feed_dict={X: x, Y: y})
        #Display the epoch
        if epoch % 200 == 0:
            print("Epoch:", '%d' % (epoch))
            pred = (neural_network) # Apply softmax to logits
            output = pred.eval({X: x})
            print("Prediction:", output)

            estimated_class = tf.argmax(pred, 1) #+1e-50-1e-50
            correct_prediction = tf.equal(tf.argmax(pred, 1), label_train)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            print(accuracy.eval({X: x}))
        
        
        
    #Test model
    pred = (neural_network) # Apply softmax to logits
    output = pred.eval({X: x_test})
    print("Prediction:", output)

    estimated_class = tf.argmax(pred.eval({X: x_test}), 1) #+1e-50-1e-50
    correct_prediction = tf.equal(tf.argmax(pred.eval({X: x_test}), 1), label_test)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(accuracy.eval({X: x}))

    

