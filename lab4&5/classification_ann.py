import numpy as np
# np.set_printoptions(threshold=np.inf)  
import tensorflow as tf
import matplotlib.pyplot as plt
import csv

def get_features_name(index):
    features_test = ["Gender", "Age", "Height", "Weight", "family_history_with_overweight",
                 "FAVC", "FCVC", "NCP", "CAEC", "SMOKE", "CH2O", "SCC", "FAF", "TUE", 
                 "CALC", "MTRANS", "NObeyesdad"]
    return features_test[index]


def discrete_handle(index ,category):
    feature_name = get_features_name(index)
    category_set = {
        "Gender": ["Female", "Male"],
        "family_history_with_overweight": ["no", "yes"],
        "FAVC": ["no", "yes"],
        "CAEC": ["no", "Sometimes", "Frequently", "Always"],
        "SMOKE": ["no", "yes"],
        "SCC": ["no", "yes"],
        "CALC": ["no", "Sometimes", "Frequently", "Always"],
        "MTRANS": ["Public_Transportation", "Automobile", "Motorbike", "Walking", "Bike"],
        "NObeyesdad": ["Insufficient_Weight" ,"Normal_Weight" , "Overweight_Level_I", "Overweight_Level_II", "Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III"]
    }
    digit = category_set[feature_name].index(category)
    size = len(category_set[feature_name])
    return digit, size


def one_hot_encode(digit, size):
    '''
    One-hot encode the labels \n
    return a one-hot key (type: list) \n
    e.g: \n
    input => digit: 0, size: 3  
    means one-hot encode label 1 into (1, 0, 0)  \n
    input => digit: 3, size: 5  
    means one-hot encode label 1 into (0, 0, 0, 1, 0)  
    '''
    return [1. if i == digit else 0. for i in range(size)]


def standardize(ndarray,selected_column=None):
    '''
    ### Z-score normalizartion \n
    return a list of normalized new ndarray also the mean and standard deviation of the original ndarray axis 0 (row). \n
    rescaling the data =>  x = (x - x.mean) / x.std  
    '''
    mean = np.mean(ndarray, axis=0)
    std = np.std(ndarray, axis=0)
    if selected_column is not None:
        unselected_column = [i for i in range(ndarray.shape[1]) if i not in selected_column]
        mean[unselected_column] = 0
        std[unselected_column] = 1
    result = (ndarray - mean) / std
    return [result, mean, std]


dataset_path = tf.keras.utils.get_file("ObesityDataSet_raw_and_data_sinthetic.zip", "https://archive.ics.uci.edu/ml/machine-learning-databases/00544/ObesityDataSet_raw_and_data_sinthetic%20(2).zip", extract=True)[:-4] + ".csv"

processed_data = []
features_test = []

with open(dataset_path, newline='') as datafile:
    reader = csv.reader(datafile)
    numerical_cols = [1, 2, 3, 6, 7, 10, 12, 13]
    
    for row in reader:
        if reader.line_num == 1 :
            # get the fist row of data (column titles)
            features_test = row
        else : 
            tmp_row = []
            for index in range(len(row)):
                if index not in numerical_cols:
                    label_digit, category_size = discrete_handle(index ,row[index])
                    one_hot_key = one_hot_encode(label_digit, category_size)
                    if index == (len(row) - 1):
                        tmp_row.extend(one_hot_key)
                        tmp_row.append(label_digit)
                    else:
                        tmp_row.extend(one_hot_key)
                else:
                    tmp_row.append(float(row[index]))
            processed_data.append(tmp_row)

data = np.array(processed_data)

x_data = data[:, :-8]
y_data = data[:, -8:]
numerical_cols = [2, 3, 4, 9, 10, 17, 20, 21]
x_data, mean, std = standardize(x_data, selected_column=numerical_cols)


# setting training dataset
x = x_data[0:1000, :]
y = y_data[0:1000, 0:-1]

# setting test dataset
x_test = x_data[1001:, :]
y_test = y_data[1001:, 0:-1]

# load label corresponding to the y_data, e.g. [1,0,0,0] (y) => 0 (label)
label_train = y_data[0:1000, -1]
label_test = y_data[1001:, -1]

#Network parameters
n_hidden1 = 10
n_hidden2 = 10
n_hidden3 = 10
n_hidden4 = 10
n_hidden5 = 10
n_input = 31
n_output = 7

#Learning parameters
learning_constant = 1
number_epochs = 10000
batch_size = 1000

#Defining the input and the output
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_output])

def add_layer(input_layer, in_size, out_size, activation_function = None):
    '''
    Adding new layer to neural network.
    '''
    Weight = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.random_normal([out_size]))
    Wx_plus_b = tf.add(tf.matmul(input_layer, Weight), biases)

    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

def multilayer_perceptron(input_d):
    layer_1 = add_layer(input_d, n_input, n_hidden1, tf.nn.sigmoid)
    layer_2 = add_layer(layer_1, n_hidden1, n_hidden2, tf.nn.sigmoid)
    layer_3 = add_layer(layer_2, n_hidden2, n_hidden3, tf.nn.sigmoid)
    layer_4 = add_layer(layer_3, n_hidden3, n_hidden4, tf.nn.sigmoid)
    layer_5 = add_layer(layer_4, n_hidden4, n_hidden5, tf.nn.sigmoid)
    out_layer = add_layer(layer_5, n_hidden5, n_output)


    return out_layer
    
#Create model 
neural_network = multilayer_perceptron(X)

#Define loss an d optimizer
# loss_op = tf.reduce_mean(tf.math.squared_difference(neural_network, Y))
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=neural_network,labels=Y))
# loss_op = tf.reduce_mean()
optimizer = tf.train.GradientDescentOptimizer(learning_constant).minimize(loss_op)

#Initializing the variables
init = tf.global_variables_initializer()


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
            print("Accuracy:" , accuracy.eval({X: x}))
        

    #Test model
    pred = (neural_network) # Apply softmax to logits
    print("Prediction:", pred.eval({X: x_test}))

    estimated_class = tf.argmax(pred, 1) #+1e-50-1e-50
    correct_prediction = tf.equal(tf.argmax(pred, 1), label_test)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval({X: x_test}))

print("The data was downloaded in path:", dataset_path)

