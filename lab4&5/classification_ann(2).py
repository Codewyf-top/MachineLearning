import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import csv
import os
#for MacOS system
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def get_weight_type(index):
    Nobeyesdad = ["Insufficient_Weight" ,"Normal_Weight" , "Overweight_Level_I",
                  "Overweight_Level_II", "Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III"]
    return Nobeyesdad[index]


def get_features_name(index):
    features_test = ["Gender", "Age", "Height", "Weight", "family_history_with_overweight",
                 "FAVC", "FCVC", "NCP", "CAEC", "SMOKE", "CH2O", "SCC", "FAF", "TUE", 
                 "CALC", "MTRANS", "NObeyesdad"]
    return features_test[index]


def discrete_handle(feature_name ,category):
    '''
    Handling the discrete values in data \n
    Input (feature_name, category) \n
    index is the   
    '''
    category_set = {
        "Gender": ["Female", "Male"],
        "family_history_with_overweight": ["no", "yes"],
        "FAVC": ["no", "yes"],
        "CAEC": ["no", "Sometimes", "Frequently", "Always"],
        "SMOKE": ["no", "yes"],
        "SCC": ["no", "yes"],
        "CALC": ["no", "Sometimes", "Frequently", "Always"],
        "MTRANS": ["Public_Transportation", "Automobile", "Motorbike", "Walking", "Bike"],
        "NObeyesdad": ["Insufficient_Weight" ,"Normal_Weight" , "Overweight_Level_I",
                       "Overweight_Level_II", "Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III"]
    }
    class_digit = category_set[feature_name].index(category)
    size = len(category_set[feature_name])
    return class_digit, size


def one_hot_encode(class_digit, size):
    '''
    One-hot encode the labels \n
    return a one-hot key (type: list) \n
    e.g: \n
    input => class_digit: 0, size: 3  
    means one-hot encode label 1 into (1, 0, 0)  \n
    input => class_digit: 3, size: 5  
    means one-hot encode label 1 into (0, 0, 0, 1, 0)  
    '''
    return [1. if i == class_digit else 0. for i in range(size)]


def data_row_handle(row, numerical_cols):
    '''
    return a list of numerical values \n
    Data-handling for one single row of the data \n
    Tranform all the data value in a row to numerical value \n  
    Encoding the discrete value as one-hot key
    '''
    tmp_row = []
    for index in range(len(row)):
        if index not in numerical_cols:
            feature_name = get_features_name(index)
            class_digit, category_size = discrete_handle(feature_name, row[index])
            one_hot_key = one_hot_encode(class_digit, category_size)
            if index == (len(row) - 1):
                tmp_row.extend(one_hot_key)
                tmp_row.append(class_digit)
            else:
                tmp_row.extend(one_hot_key)
        else:
            tmp_row.append(float(row[index]))

    return tmp_row


def standardize(ndarray,selected_column=None):
    '''
    ### Z-score normalizartion \n
    return a list of normalized new ndarray also the mean and standard deviation of
    the original ndarray axis 0 (row). \n
    Ingnoring the unselected columns (columns of con-numerical data) in the data \n 
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


def norm(nd, selected_column=None):
    '''
    ### Min - Max Normalization \n
    return a list of normalized new ndarray also the maximum and minimum of
    the original ndarray axis 0 (row). \n
    Ingnoring the unselected columns (columns of con-numerical data) in the data \n 
    rescaling the data => x = (x - min) / (max - min)
    '''
    nd_min = np.min(nd, axis=0)
    nd_max = np.max(nd, axis=0)

    if selected_column is not None:
        unselected_column = [i for i in range(nd.shape[1]) if i not in selected_column]
        nd_max[unselected_column] = 1
        nd_min[unselected_column] = 0

    result = (nd - nd_min) / (nd_max - nd_min)
    return  [result, nd_max, nd_min]


def data_seperate(data):
    '''
    Seperate the data into x, y and labels
    '''
    x = data[:, :-8]
    y = data[:, -8:-1]
    # load label corresponding to the y_data, e.g. [1,0,0,0] (y) => 0 (label)
    labels = data[:, -1]
    return x, y, labels


def data_iter(batch_size, data):
    '''
    Iterate the data by batch size and sperate them as x, y and labels
    '''
    num_examples = len(data)
    for i in range(0, num_examples, batch_size):
        batch_data = data[i: min(i + batch_size, num_examples)]
        x, y, labels = data_seperate(batch_data) 
        yield x, y, labels


def get_k_fold_data(data, i, k):
    '''
    Saperate data as k-fold and assign the i fold as valid dataset
    '''
    fold_size = data.shape[0] // k
    data_train, data_valid = None, None

    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        data_part = data[idx, :]
        if j == i:
            data_valid = data_part
        elif data_train is None:
            data_train = data_part
        else:
            data_train = np.vstack((data_train, data_part))

    return data_train, data_valid


# Download the data file
dataset_path = tf.keras.utils.get_file("ObesityDataSet_raw_and_data_sinthetic.zip",
                                       "https://archive.ics.uci.edu/ml/machine-learning-databases/00544/ObesityDataSet_raw_and_data_sinthetic%20(2).zip",
                                        extract=True)[:-4] + ".csv"

# Read the data from the file
with open(dataset_path, newline='') as datafile:
    reader = csv.reader(datafile)
    numerical_cols = [1, 2, 3, 6, 7, 10, 12, 13]
    processed_data = []

    for row in reader:
        if reader.line_num > 1 : 
            tmp_row = data_row_handle(row, numerical_cols)
            processed_data.append(tmp_row) 

data = np.array(processed_data)

# Z-score normalize the data
numerical_cols = [2, 3, 4, 9, 10, 17, 20, 21]
data, mean, std = standardize(data, selected_column=numerical_cols)

# Random shuffle the data
np.random.seed(3)
np.random.shuffle(data)
# print(*data.tolist(), sep='\n')

# setting train and valid dataset, data_ann is for 10-fold validation
data_ann = data[:1000, :]
data_valid = data_ann[:100, :]
data_train = data_ann[100:1000, :]

# setting test dataset
data_test = data[1000:, :]

#Network parameters
n_hidden1 = 10
n_hidden2 = 10
n_input = 31
n_output = 7

#Learning parameters
learning_constant = 0.94
number_epochs = 500
drop_probability = 0.5
batch_size = 256

#Defining the input and the output
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_output])


def dropout_layer(X, drop_prob):
    assert 0 <= drop_prob <= 1
    # If drop_probability equals 1, all elements are dropped out
    if drop_prob == 1:
        return tf.zeros_like(X)
    # If drop_probability equals 0, all elements are kept
    if drop_prob == 0:
        return X

    # If drop_probaillity between 0 and 1, randomly choose elements to keep based on keep probability
    keep_prob = 1 - drop_prob 
    mask = tf.random.uniform(shape=tf.shape(X), minval=0, maxval=1) < keep_prob
    return tf.cast(mask, dtype=tf.float32) * X / keep_prob


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
    out_layer = add_layer(layer_2, n_hidden2, n_output)
    # out_layer = add_layer(layer_1, n_hidden1, n_output)

    
    return out_layer
    
#Create model 
neural_network = multilayer_perceptron(X)

#Define loss an d optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=neural_network,labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_constant).minimize(loss_op)

#Initializing the variables
init = tf.global_variables_initializer()


def get_accuracy(neural_network, labels):
    # Tensorflow variable for predicted one-hot value
    pred = tf.nn.softmax((neural_network))
    # Tensorflow variable for label of predicted one-hot values, e.g. argmax(0,1,0) => 1 
    estimated_class = tf.argmax(pred, 1)
    correct_prediction = tf.equal(estimated_class, labels)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy


def train(sess ,init, number_epochs, optimizer, data_train,
          data_valid, neural_network, k_fold_flag=False):

    sess.run(init)

    # List for storing accuracy value every 10 epochs
    train_acc_list = []
    valid_acc_list = []

    x_valid, y_valid, labels_valid = data_seperate(data_valid)
    x_train, y_train, labels_train = data_seperate(data_train)

    #Training epoch 
    for epoch in range(number_epochs): 
        # Train the network batch by batch
        for batch_x, batch_y, batch_labels in data_iter(batch_size, data_train):
            sess.run(optimizer, feed_dict={X: batch_x, Y: batch_y})
         
        if epoch % 10 == 0 or epoch % (number_epochs // 5) == 0:
            accuracy_train = get_accuracy(neural_network, labels_train).eval({X: x_train})
            accuracy_valid = get_accuracy(neural_network, labels_valid).eval({X: x_valid})
            train_acc_list.append(accuracy_train)
            valid_acc_list.append(accuracy_valid)
            # If the function is not in a k-fold validation, then print accuracy  
            if epoch % (number_epochs // 5) == 0 and k_fold_flag == False:
                print("Epoch:", '%d' % (epoch + 1), "train acc:" , accuracy_train,
                      "valid acc:", accuracy_valid)

    # Final accuracy
    accuracy_train = get_accuracy(neural_network, labels_train)
    accuracy_valid = get_accuracy(neural_network, labels_valid)
    final_train_acc = accuracy_train.eval({X: x_train})
    final_valid_acc = accuracy_valid.eval({X: x_valid})

    return final_train_acc, final_valid_acc, train_acc_list, valid_acc_list


# k-fold validation
def k_fold(sess ,init, number_epochs, optimizer, data_train, data_valid, neural_network, k):
    assert k > 1

    # List for storing accuracy value every 10 epochs
    ktrain_acc_list = []
    kvalid_acc_list = []

    # for calculating the mean accuracy
    mean_train_acc, mean_valid_acc = 0, 0
    for i in range(k):
        data_train, data_valid = get_k_fold_data(data, i, k)
        train_acc, valid_acc, train_acc_list, valid_acc_list = train(sess ,init, number_epochs, optimizer, data_train,
                                                                     data_valid, neural_network, k_fold_flag=True)
        print("Fold", i+1, "train acc:" , train_acc, "valid acc:", valid_acc)
        mean_train_acc += train_acc
        mean_valid_acc += valid_acc
        ktrain_acc_list.append(train_acc_list)
        kvalid_acc_list.append(valid_acc_list)

    mean_train_acc /= k
    mean_valid_acc /= k
    
    return mean_train_acc, mean_valid_acc, ktrain_acc_list, kvalid_acc_list


def plot_two_curves(x1, y1, x2, y2,label1="Train accuracy", label2="Valid accuracy",
                        xlabel="Epochs", ylabel="Accuracy", title=""):
    '''
    Plot Train accuracy curve and  Valid accuracy curve
    '''
    plt.plot(x1, y1, label=label1)
    plt.plot(x2, y2, label=label2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()

    import tensorflow as tf


with tf.Session() as sess:
    print("# First run of the training set and validating set")
    train_acc, valid_acc, train_acc_list, valid_acc_list = train(sess ,init, number_epochs,
                                                                 optimizer, data_train, data_valid, neural_network)

    # generate every epoch coressponding to the data pionts in training accarucy set and valid accuracy set
    epochs = [i for i in range(0, number_epochs, 10)]

    # Draw the accuracy curve
    plot_two_curves(epochs, train_acc_list, epochs, valid_acc_list, title="The first training")
    plt.show()
    print("*" * 50) # seperate the output on terminal

    k = 10
    print("#  {}-fold validation".format(k))
    train_acc, valid_acc, train_acc_lists, valid_acc_lists = k_fold(sess ,init, number_epochs, optimizer,
                                                                    data_train, data_valid, neural_network, k)

    print("{}-fold average".format(k), "train acc:" , train_acc, "valid acc:", valid_acc)
    print("*" * 50) # seperate the output on terminal

    # ndarrays for storing the accuracy information every 10 epochs
    ntrain_acc_list = np.zeros(number_epochs // 10)
    nvalid_acc_list = np.zeros(number_epochs // 10)

    # Draw 10 accuracy curves
    for i in range(k):
        ntrain_acc_list = ntrain_acc_list + np.array(train_acc_lists[i])
        nvalid_acc_list = nvalid_acc_list + np.array(valid_acc_lists[i])

        plt.subplot(2, 5, (i+1))
        plot_two_curves(epochs, train_acc_lists[i], epochs, valid_acc_lists[i],
                            title="Fold {} accuracy".format(i+1))
    plt.show()

    # Darw an average accuracy curve
    ntrain_acc_list = ntrain_acc_list / k
    nvalid_acc_list = nvalid_acc_list / k           
    plot_two_curves(epochs, ntrain_acc_list.tolist(), epochs, nvalid_acc_list.tolist(),
                            title="10 -fold average accuracy")
    plt.show()

    print("# Test model")
    x_test, y_test, labels_test = data_seperate(data_test)
    # Print the last 5 predictions
    pred = tf.nn.softmax((neural_network))
    output = pred.eval({X: x_test})[-5:,:]
    predicted_class_label = [get_weight_type(labels) for labels in np.argmax(output, axis=1)]
    print("Last 5 predictions:", *[item for item in zip(output.tolist(), predicted_class_label)], sep='\n')

    # Print the accuracy of test data
    correct_prediction = tf.equal(tf.argmax(pred, 1), labels_test)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Test accuracy:", accuracy.eval({X: x_test}))

print("The data was downloaded in path:", dataset_path)

