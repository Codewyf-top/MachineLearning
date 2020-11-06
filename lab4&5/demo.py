# -*- coding: utf-8 -*-
"""
@Time ： 2020/10/30 5:45 下午
@Auth ： Codewyf
@File ：demo.py
@IDE ：PyCharm
@Motto：Go Ahead Instead of Heasitating

"""
import numpy as np
import tensorflow as tf
import math
import logging
logging.basicConfig(level=logging.DEBUG)
import matplotlib.pyplot as plt
import os
#for MacOS system
os.environ['KMP_DUPLICATE_LIB_OK']='True'


#Network parameters
n_hidden1 = 10
n_hidden2 = 10
n_input = 5
n_output = 1
#Learning parameters
learning_constant = 0.05
number_epochs = 1000
batch_size = 1000
training_display = 10
data_path = "data/model"


#Defining the input and the output
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_output])
#DEFINING WEIGHTS AND BIASES
#Biases first hidden layer
b1 = tf.Variable(tf.random_normal([n_hidden1])) #Biases second hidden layer
tf.summary.histogram('hidden_layer/bias',b1)
b2 = tf.Variable(tf.random_normal([n_hidden2])) #Biases output layer
tf.summary.histogram('hidden_layer/bias',b2)
b3 = tf.Variable(tf.random_normal([n_output]))
tf.summary.histogram('output_layer/bias',b3)
#Weights connecting input layer with first hidden layer
w1 = tf.Variable(tf.random_normal([n_input, n_hidden1])) #Weights connecting first hidden layer with second hidden layer
tf.summary.histogram('hidden_layer/weight',w1)
w2 = tf.Variable(tf.random_normal([n_hidden1, n_hidden2])) #Weights connecting second hidden layer with output layer
tf.summary.histogram('hidden_layer/weight',w2)
w3 = tf.Variable(tf.random_normal([n_hidden2, n_output]))
tf.summary.histogram('output_layer/weight',w3)

def multilayer_perceptron(input_d):
#Task of neurons of first hidden layer
#tf.matmul -> Multiplies matrix `a` by matrix `b`, producing `a` * `b`.
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(input_d, w1), b1)) #Task of neurons of second hidden layer
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, w2), b2)) #Task of neurons of output layer
    out_layer = tf.add(tf.matmul(layer_2, w3),b3)
    return out_layer

#Create model
neural_network = multilayer_perceptron(X)

#Define loss and optimizer
loss_op = tf.reduce_mean(tf.math.squared_difference(neural_network,Y))
tf.summary.scalar("loss",loss_op)
#loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=neural_netw ork,labels=Y))
#GD

optimizer = tf.train.GradientDescentOptimizer(learning_constant).minimize(loss_op)

merged = tf.summary.merge_all()

#设置绘图器
# plt.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
# plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
# fig = plt.figure()
# ax = fig.add_subplot(1,1,1)
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('Process of Training')
# ax.scatter(x_data, y_data,s=5,label ="Training Data")
# plt.plot(x,y,c = "red",label = "Original Curve")
# plt.show(block = False)
#Initializing the variables
init = tf.global_variables_initializer()
#
# batch_x1=np.loadtxt('x1.txt')
# batch_x2=np.loadtxt('x2.txt')
# batch_x3=np.loadtxt('x3.txt')
# batch_x4=np.loadtxt('x4.txt')
# batch_x5=np.loadtxt('x5.txt')
# batch_y1=np.loadtxt('y1.txt')


# label=batch_y1#+1e-50-1e-50
#
# batch_x=np.column_stack((batch_x1, batch_x2, batch_x3, batch_x4, batch_x5))
# batch_y=np.column_stack(batch_y1)


#batch_x_train=batch_x[:,0:599]
#batch_y_train=batch_y[:,0:599]

# batch_x_test=batch_x[:,600:1000]
# batch_y_test=batch_y[:,600:1000]

x_data=np.loadtxt('data/air_foil_x.txt', encoding='UTF-16')
y_data=np.loadtxt('data/air_foil_y.txt', encoding='UTF-16')

# setting training dataset
x = tf.nn.local_response_normalization((x_data[0:1000, :]), depth_radius=None, bias=None,alpha=None, name=None)
# reshape y to fit the placeholder
y = tf.nn.local_response_normalization((y_data[0:1000, np.newaxis]), depth_radius=None, bias=None,alpha=None, name=None)

# setting test dataset
x_test = x_data[1001:, :]
# reshape y to fit the placeholder
y_test = y_data[1001:, np.newaxis]

#
# label_train=label[0:599]
# label_test=label[600:1000]


with tf.Session() as sess:
    #initlization
    sess.run(init)
    # load exists models to continue to train
    # saver = tf.train.Saver()
    # saver.restore(sess, "data/model")

    #write training data to logs
    writer = tf.summary.FileWriter("logs", sess.graph)
    #Training epoch
    for epoch in range(number_epochs):
        sess.run(optimizer, feed_dict={X: x, Y: y})
        if (epoch + 1) % 2000 == 0:
            print("Epoch:", '%d' % (epoch + 1))
            # Calculating the cost at every epoch
            pred = (neural_network) # Apply softmax to logits
            accuracy = tf.keras.losses.MSE(pred, Y)
            print("Accuracy:", np.mean(accuracy.eval({X: x, Y: y}), axis=0))
        #Display the epoch if epoch % 100 == 0:
        print("Epoch:", '%d' % (epoch))
        if epoch % training_display == 0:
            result = sess.run(merged,feed_dict={X: x, Y: y})
            writer.add_summary(result, epoch)
            print(epoch, sess.run(loss_op, feed_dict={X: x, Y: y}))
            plt.savefig("picture\\" + str(epoch) + ".png")  # save training picture
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        saver = tf.train.Saver()
        saver.save(sess, data_path)  # save model
    # Test model
    # pred = (neural_network) # Apply softmax to logits
    # accuracy=tf.keras.losses.MSE(pred,Y)
    # print("Accuracy:", accuracy.eval({X: x, Y: y}))
    # tf.keras.evaluate(pred,batch_x)
    # print("Prediction:", pred.eval({X: x}))
    # output=neural_network.eval({X: x})
    # plt.plot(batch_y_train[0:10], 'ro', output[0:10], 'bo')
    # plt.ylabel('some numbers')
    # plt.show()
    #
    # estimated_class=tf.argmax(pred, 1)#+1e-50-1e-50
    # correct_prediction1 = tf.equal(tf.argmax(pred, 1),label)
    # accuracy1 = tf.reduce_mean(tf.cast(correct_prediction1, tf.float32))
    #
    # print(accuracy1.eval({X: batch_x}))
    