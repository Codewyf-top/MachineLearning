import numpy as np
import tensorflow as tf
import math
import logging

logging.basicConfig(level=logging.DEBUG)
import matplotlib.pyplot as plt

# Network parameters
n_hidden1 = 10
n_input = 1
n_output = 1

# Learning parameters
learning_constant = 0.00001
number_epochs = 1000

# Defining the input and the output
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_output])

# DEFINING WEIGHTS AND BIASES

# Biases first hidden layer
b1 = tf.Variable(tf.random_normal([n_hidden1]))

# Biases output layer
b2 = tf.Variable(tf.random_normal([n_output]))

# Weights connecting input layer with first hidden layer
w1 = tf.Variable(tf.random_normal([n_input, n_hidden1]))

# Weights connecting first hidden layer with output layer
w2 = tf.Variable(tf.random_normal([n_hidden1, n_output]))


def multilayer_perceptron(input_d):
    # Task of neurons of first hidden layer
    layer_1 = tf.nn.softplus(tf.add(tf.matmul(input_d, w1), b1))
    # Task of neurons of output layer
    out_layer = tf.add(tf.matmul(layer_1, w2), b2)

    return out_layer


# Create model
neural_network = multilayer_perceptron(X)
# Define loss and optimizer
loss_op = tf.reduce_mean(tf.math.squared_difference(neural_network, Y))
optimizer = tf.train.GradientDescentOptimizer(learning_constant).minimize(loss_op)
# Initializing the variables
init = tf.global_variables_initializer()

# Generating random linear data
# There will be 50 data points ranging from 0 to 50
x = np.linspace(0, 50, 50)
y = 2 * x + 3

# Adding noise to the random linear data
x += np.random.uniform(-1, 1, 50)
y += np.random.uniform(-1, 1, 50)

# reshape x and y
x = x[:, np.newaxis]
y = y[:, np.newaxis]

with tf.Session() as sess:
    sess.run(init)
    # Training epoch
    for epoch in range(number_epochs):
        sess.run(optimizer, feed_dict={X: x, Y: y})
        if (epoch + 1) % 100 == 0:
            # Calculating the cost at every epoch
            pred = (neural_network)  # Apply softmax to logits
            accuracy = tf.keras.losses.MSE(pred, Y)
            print("Accuracy:", accuracy.eval({X: x, Y: y}))
            output = pred.eval({X: x})
            # tf.keras.evaluate(pred,batch_x)
            print("Prediction:", pred.eval({X: x}))

            # Plotting the Results
            plt.plot(x, y, 'ro', label='Original data')
            plt.plot(x, output, label='Fitted line')
            plt.title('Linear Regression Result')
            plt.legend()
            plt.show()




