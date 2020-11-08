import numpy as np
import tensorflow as tf
import math
import logging
import os
#for MacOS system
os.environ['KMP_DUPLICATE_LIB_OK']='True'
# logging.basicConfig(level=logging.DEBUG)
import matplotlib.pyplot as plt

def get_features_name(index):
    '''
    Getting features label text form the airfoil data.
    '''
    features_text = ['Frequency', 'Angle of attack', 'Chord length',
                     'Free-stream velocity', 'Suction side displacement thickness']
    return features_text[index]


def add_layer(input_layer, in_nodes, out_nodes, activation_function = None):
    '''
    Adding new layer to neural network.
    '''
    Weight = tf.Variable(tf.random_normal([in_nodes, out_nodes]))
    biases = tf.Variable(tf.random_normal([out_nodes]))
    Wx_plus_b = tf.add(tf.matmul(input_layer, Weight), biases)

    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


def multilayer_perceptron(input_d):
    layer_1 = add_layer(input_d, n_input, n_hidden1, tf.nn.sigmoid)
    layer_2 = add_layer(layer_1, n_hidden1, n_hidden2, tf.nn.sigmoid)
    out_layer = add_layer(layer_2, n_hidden2, n_output)

    return out_layer


def standardize(ndarray):
    '''
    ### Z-score normalizartion \n
    return a list of normalized new ndarray also the mean and standard deviation of the original ndarray axis 0 (row). \n
    rescaling the data =>  x = (x - x.mean) / x.std  
    '''
    mean = np.mean(ndarray, axis=0)
    std = np.std(ndarray, axis=0)
    result = (ndarray - mean) / std
    return [result, mean, std]


def norm(nd):
    '''
    ### Min - Max Normalization \n
    return a list of normalized new ndarray also the maximum and minimum of the original ndarray axis 0 (row). \n
    rescaling the data => x = (x - min) / (max - min)
    '''
    nd_min = np.min(nd, axis=0)
    nd_max = np.max(nd, axis=0)
    result = (nd - nd_min) / (nd_max - nd_min)
    return  [result, nd_max, nd_min]



# load data file
data=np.loadtxt('airfoil.txt', encoding='UTF-16')
x_data = data[:, 0:5]
y_data = data[:, 5]


# standardize the data
x_data, data_mean, data_std = standardize(x_data)


# setting training dataset
x = x_data[0:1000, :]
# reshape y to fit the placeholder
y = y_data[0:1000, np.newaxis]


# setting test dataset
x_test = x_data[1001:, :]
# reshape y to fit the placeholder
y_test = y_data[1001:, np.newaxis]


#Network parameters
n_hidden1 = 10
n_hidden2 = 10

n_input = 5
n_output = 1

#Learning parameters
learning_constant = 0.01
number_epochs = 10000

#Defining the input and the output
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_output])

#Create model 
neural_network = multilayer_perceptron(X)
#Define loss and optimizer
loss_op = tf.reduce_mean(tf.math.squared_difference(neural_network, Y))
optimizer = tf.train.GradientDescentOptimizer(learning_constant).minimize(loss_op)
#Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    #Training epoch 
    for epoch in range(number_epochs):
        sess.run(optimizer, feed_dict={X: x, Y: y})
        if (epoch + 1) % 2000 == 0:
            print("Epoch:", '%d' % (epoch + 1))
            # Calculating the cost at every epoch
            pred = (neural_network) 
            accuracy = tf.reduce_mean(tf.keras.losses.MSE(pred, Y))
            print("Accuracy:", accuracy.eval({X: x, Y: y}))
            print(sess.run(loss_op, feed_dict={X: x, Y: y}))

        # Print Training result
        if (epoch + 1) % 10000 == 0:
            output = pred.eval({X: x})
            for i in range(x.shape[1]):
                plt.subplot(230+i+1)
                plt.plot(x[:,i:i+1], y, 'ro', markersize=6, label = 'Original value')
                plt.plot(x[:,i:i+1], output, 'bo', markersize=4, label = 'Predicted value')
                plt.legend()
                plt.title(get_features_name(i))
            plt.show()    

            rng = [np.min(y), np.max(y)]
            plt.plot(rng, rng)
            plt.scatter(y, output)
            plt.show()    

    # Test result
    pred = (neural_network) # Apply softmax to logits
    accuracy = tf.reduce_mean(tf.keras.losses.MSE(pred, Y))
    print("Accuracy:", accuracy.eval({X: x_test, Y: y_test}))
    output = pred.eval({X: x_test})
    for i in range(x_test.shape[1]):
        plt.subplot(230+i+1)
        plt.plot(x_test[:,i:i+1], y_test, 'ro', markersize=6, label = 'Original value')
        plt.plot(x_test[:,i:i+1], output, 'bo', markersize=4, label = 'Predicted value')
        plt.legend()
        plt.title(get_features_name(i))
    plt.show()    

    # Inspect how good the predictions match the labels
    rng = [np.min(y), np.max(y)]
    plt.plot(rng, rng)
    plt.scatter(y_test, output)
    plt.show()    



