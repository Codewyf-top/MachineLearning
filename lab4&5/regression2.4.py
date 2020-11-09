# -*- coding: utf-8 -*-
"""
@Time ： 2020/11/5 7:52 下午
@Auth ： Codewyf
@File ：build_a_neural_network.py
@IDE ：PyCharm
@Motto：Go Ahead Instead of Heasitating

"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# from sklearn.model_selection import cross_val_score
import os
#for MacOS system
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import tensorboard
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')
# from sklearn import linear_model
#hyper parameters
learning_rate = 0.01
EPOCH = 10000

#Network parameters
n_hidden1 = 10
n_hidden2 = 10
# n_hidden3 = 10
# n_hidden4 = 10
n_input = 5
n_output = 1

def get_features_name(index):
    '''
    Getting features label text form the airfoil data.
    '''
    features_text = ['Frequency', 'Angle of attack', 'Chord length',
                     'Free-stream velocity', 'Suction side displacement thickness']
    return features_text[index]

def add_layer(inputs, in_size, out_size, n_layer, activation_function=None):
    #add one more layer and return the output of this layer
    layer_name = 'layer%s' % n_layer
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
            tf.summary.histogram(layer_name+'weights', Weights)
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.random_normal([out_size]), name='b') #因为biases初始值不为0所以初始值随便加上一个0.1
            tf.summary.histogram(layer_name + 'biases', biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs, Weights) + biases #Wx_plus_b = Weights * x + b iases
        if activation_function is None:
            outputs = Wx_plus_b
            tf.summary.histogram(layer_name + 'outputs', outputs)
        else:
            outputs = activation_function(Wx_plus_b)
        return outputs

def normalization(ndarry):
    return (ndarry - np.mean(ndarry, axis=0)) / np.std(ndarry, axis=0)

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

def multilayer_perceptron(input_d):
    l1 = add_layer(xs, n_input, n_hidden1, n_layer=1, activation_function=tf.nn.sigmoid)
    l2 = add_layer(l1, n_hidden1, n_hidden2, n_layer=2, activation_function=tf.nn.sigmoid)
    # l3 = add_layer(l2, n_hidden2, n_hidden3, n_layer=3, activation_function=tf.nn.relu)
    # l4 = add_layer(l3, n_hidden3, n_hidden4, n_layer=4, activation_function=tf.nn.relu)
    #add output layer
    output_layer = add_layer(l2, n_hidden2, n_output, n_layer=5, activation_function=None) #in_size=10是l1的outsize 1是y_data的size 如果activation function=None, 就为线性函数
    return output_layer

def get_accuracy(neural_network, labels):
    # Tensorflow variable for predicted one-hot value
    pred = tf.nn.softmax((neural_network))
    accuracy = tf.reduce_mean(tf.keras.losses.MSE(pred, labels))
    return accuracy

#define placeholder for inputs to network
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, n_input], name='x_input')#None表示无论你给多少例子都ok，1代表一个属性; 行数不限制，列只能有一列
    ys = tf.placeholder(tf.float32, [None, n_output], name='y_input')

#create model
neural_network = multilayer_perceptron(xs)
#the error between prediction and real data
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.math.squared_difference(neural_network, ys),name='loss') #reduce_sum是对每一个例子进行求和 reduce_mean对所有和求一个平均值
    tf.summary.scalar('loss', loss)
with tf.name_scope('optimizer'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss) #learning rate 这里的train_step相当于optimizer

# load data file
data=np.loadtxt('airfoil.txt', encoding='UTF-16')
x_data = data[:, 0:5]
y_data = data[:, 5]

# standardize the data
x_data, data_mean, data_std = standardize(x_data)

# setting training dataset
x = x_data[0:1000, :]
# reshape y to fit the placeholder
y = y_data[:, np.newaxis]

# # setting test dataset
# x_test = x_data[1001:, :]
# # reshape y to fit the placeholder
# y_test = y_data[1001:, np.newaxis]

iter_saver = tf.train.Saver(max_to_keep=3)  # keep 3 last iterations
best_saver = tf.train.Saver(max_to_keep=5)  # keep 5 last best models

#important step
def run(x,y,x_test,y_test):
    global cross_val_loss
    init = tf.global_variables_initializer()#初始所有变量
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter("logs/", sess.graph)

        #training eopch
        for epoch in range(EPOCH):
            sess.run(optimizer, feed_dict={xs:x, ys:y})
            if (epoch + 1) % 2000 == 0:
                print("Epoch:", '%d' % (epoch + 1))
                pred = (neural_network)
                accuracy = tf.reduce_mean(tf.keras.losses.MSE(pred, ys))
                print("Train loss:", accuracy.eval({xs:x, ys:y}))
                RMESE = tf.sqrt(loss)
                print("Average Root Mean Square Error:",RMESE.eval({xs:x, ys:y}))

                result = sess.run(merged, feed_dict={xs:x, ys:y})
                writer.add_summary(result, epoch)#i->步数
            if (epoch + 1) % 10000 == 0:
                cross_val_loss = cross_val_loss + accuracy.eval({xs:x, ys:y})
                save_path = saver.save(sess, f"checkpoint/{fold-1}.ckpt")
                output = pred.eval({xs:x})
                for i in range(x.shape[1]):
                    plt.subplot(230 + i + 1)
                    plt.plot(x[:, i:i + 1], y, 'ro', markersize=6, label='Original value')
                    plt.plot(x[:, i:i + 1], output, 'bo', markersize=4, label='Predicted value')
                    plt.legend()
                    plt.title(get_features_name(i))
                plt.show()

                rng = [np.min(y), np.max(y)]
                plt.plot(rng, rng)
                plt.scatter(y, output)
                plt.show()

        # Test result
        pred = (neural_network)  # Apply softmax to logits
        accuracy = tf.reduce_mean(tf.keras.losses.MSE(pred, ys))
        print("Test loss:", accuracy.eval({xs: x_test, ys: y_test}))
        output = pred.eval({xs: x_test})

        for i in range(x.shape[1]):
            plt.subplot(230 + i + 1)
            plt.plot(x_test[:, i:i + 1], y_test, 'ro', markersize=6, label='Original value')
            plt.plot(x_test[:, i:i + 1], output, 'bo', markersize=4, label='Predicted value')
            plt.legend()
            plt.title(get_features_name(i))
        plt.show()

        #Inspect how good the predictions match the labels
        rng = [np.min(y), np.max(y)]
        plt.plot(rng, rng)
        plt.scatter(y_test, output)
        plt.show()

def main():
    global fold,cross_val_loss

    new_x_train=[]
    new_y_train=[]
    new_x_test =[]
    new_y_test =[]

    kf = KFold(10, False)
    for train_index, test_index in kf.split(x_data):

        index_train=list(train_index)
        index_test =list(test_index)
        for i in range(len(index_train)):
            new_x_train.append(x_data[index_train[i]])
            new_y_train.append(y[index_train[i]])
        for j in range(len((index_test))):
            new_x_test.append(x_data[index_test[j]])
            new_y_test.append(y[index_test[j]])
            new_x_train1=np.array(new_x_train)
            new_y_train1=np.array(new_y_train)
            new_x_test1=np.array(new_x_test)
            new_y_test1 =np.array(new_y_test)
        print(f"fold:{fold}")
        fold = fold + 1
        run(new_x_train1,new_y_train1,new_x_test1,new_y_test1)

    cross_val_loss = cross_val_loss % 10
    print(f"cross_val_loss:{cross_val_loss}")
cross_val_loss =0
fold=1
main()

