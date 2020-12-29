from os import times
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import csv

def get_weight_type(index):
    Nobeyesdad = ["No_Obesity", "Obesity"]
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

    if feature_name == "NObeyesdad":
        category = label_binary_trans(category)

    category_set = {
        "Gender": ["Female", "Male"],
        "family_history_with_overweight": ["no", "yes"],
        "FAVC": ["no", "yes"],
        "CAEC": ["no", "Sometimes", "Frequently", "Always"],
        "SMOKE": ["no", "yes"],
        "SCC": ["no", "yes"],
        "CALC": ["no", "Sometimes", "Frequently", "Always"],
        "MTRANS": ["Public_Transportation", "Automobile", "Motorbike", "Walking", "Bike"],
        "NObeyesdad": ["No_Obesity", "Obesity"]
    }
    class_digit = category_set[feature_name].index(category)
    size = len(category_set[feature_name])
    return class_digit, size


def label_binary_trans(label_name):
    if label_name == "Insufficient_Weight":
        return "No_Obesity"
    elif label_name == "Normal_Weight":
        return "No_Obesity" 
    elif label_name == "Overweight_Level_I":
        return "No_Obesity" 
    elif label_name == "Overweight_Level_II":
        return "No_Obesity" 
    elif label_name == "Obesity_Type_I":
        return "Obesity" 
    elif label_name == "Obesity_Type_II":
        return "Obesity" 
    elif label_name == "Obesity_Type_III":
        return "Obesity" 


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
    x = data[:, :-3]
    y = data[:, -3:-1]
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
    Split the data as k-fold and assign the i fold as a valid dataset
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

# setting train and valid dataset, data_ann is for 10-fold validation
data_ann = data[:1000, :]
data_valid = data_ann[:100, :]
data_train = data_ann[100:1000, :]

# setting test dataset
data_test = data[1000:, :]


# Network parameters
n_hidden1 = 10
n_hidden2 = 10
n_input = 31
n_output = 2

# Note of training without dropout
# lr = 0.93, eps = 500, b_s = 256, final test acc = 95% ~ 99%
# Lr experiment note
# number_epochs = 250 batch_size = 512
# Learning parameters
learning_constant = 0.93
number_epochs = 500
batch_size = 256

# set keep probablity for dropout layer
# set keep_prob = 1 to disable dropout
keep_prob = 1


# Defining the input and the output
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_output])

# Defining the keep probalility of dropout layer
KP = tf.placeholder("float")

def add_layer(input_layer, in_size, out_size, activation_function=None,
              dropout=False, keep_prob=tf.constant(1.0, dtype=tf.float32)):
    '''
    Adding new layer to neural network.
    The dropout option default to be disable
    '''
    w = tf.Variable(tf.random_normal([in_size, out_size], seed=3))
    b = tf.Variable(tf.random_normal([out_size], seed=3))
    wx_b = tf.add(tf.matmul(input_layer, w), b)

    if activation_function is None:
        outputs = wx_b
    else:
        outputs = activation_function(wx_b)

    if dropout == True:
        outputs= tf.nn.dropout(outputs, keep_prob)
    return outputs


def multilayer_perceptron(input_d, keep_p):
    '''
    keep_prob is keep probability for dropout layer
    '''
    layer_1 = add_layer(input_d, n_input, n_hidden1, tf.nn.sigmoid, dropout=True, keep_prob=keep_p)
    layer_2 = add_layer(layer_1, n_hidden1, n_hidden2, tf.nn.sigmoid, dropout=True, keep_prob=keep_p)
    out_layer = add_layer(layer_2, n_hidden2, n_output)

    return out_layer


# Create model 
neural_network = multilayer_perceptron(X, KP)

# Define loss an d optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=neural_network,labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_constant).minimize(loss_op)

# Initializing the variables
init = tf.global_variables_initializer()

# Set saver for the well trained model
saver = tf.train.Saver()


def get_accuracy(neural_network, features, labels):
    '''
    return accuracy comparing the predicted values and labels
    '''
    # Tensorflow variable for predicted one-hot value
    pred = tf.nn.softmax(neural_network)
    # Tensorflow variable for label of predicted one-hot values, e.g. argmax(0,1,0) => 1 
    estimated_class = tf.argmax(pred, 1)
    correct_prediction = tf.equal(estimated_class, labels)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    acc = accuracy.eval({X: features, KP: 1})
    return acc


def train(sess ,init, number_epochs, keep_prob, optimizer, data_train,
          data_valid, neural_network, print_acc=False, record_acc=False):

    sess.run(init)

    # List for storing accuracy value every 10 epochs
    train_acc_list = []
    valid_acc_list = []

    x_valid, y_valid, labels_valid = data_seperate(data_valid)
    x_train, y_train, labels_train = data_seperate(data_train)

    print_interval = (number_epochs // 5)

    # Training epoch 
    for epoch in range(number_epochs): 
        # Train the network batch by batch
        for batch_x, batch_y, batch_labels in data_iter(batch_size, data_train):
            sess.run(optimizer, feed_dict={X: batch_x, Y: batch_y, KP: keep_prob})

        if print_acc == True or record_acc == True:
            if record_acc == True and epoch % 10 == 0:
                accuracy_train = get_accuracy(neural_network, x_train, labels_train)
                accuracy_valid = get_accuracy(neural_network, x_valid, labels_valid)
                train_acc_list.append(accuracy_train)
                valid_acc_list.append(accuracy_valid)
            if print_acc == True and epoch % print_interval == 0:
                accuracy_train = get_accuracy(neural_network, x_train, labels_train)
                accuracy_valid = get_accuracy(neural_network, x_valid, labels_valid)
                print("Epoch:", '%d' % (epoch + 1), "train acc:" , accuracy_train,
                    "valid acc:", accuracy_valid)

    # Final accuracy
    final_train_acc = get_accuracy(neural_network, x_train, labels_train)
    final_valid_acc = get_accuracy(neural_network, x_valid, labels_valid)

    if record_acc == True:
        return final_train_acc, final_valid_acc, train_acc_list, valid_acc_list

    return final_train_acc, final_valid_acc


def k_fold(sess ,init, number_epochs, keep_prob, optimizer, data_train, data_valid, neural_network, k):
    '''
    K-fold validation function
    '''
    assert k > 1

    # for calculating the mean accuracy
    for i in range(k):
        data_train, data_valid = get_k_fold_data(data, i, k)
        train_acc, valid_acc, train_acc_list, valid_acc_list = train(sess ,init, number_epochs, keep_prob, optimizer,
                                                                     data_train, data_valid, neural_network, record_acc=True)
        print("Fold", i+1, "train acc:" , train_acc, "valid acc:", valid_acc)
        yield  i, train_acc, valid_acc, train_acc_list, valid_acc_list


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


def lr_experiment(step, start_learning_rate, experiment_times, sess, init, number_epochs, keep_prob,
                  loss_op, data_train, data_valid, neural_network):
    '''
    change the learning rate 10 times in a range
    '''
    print("# Changing the learning rate form {:.4f} to {}, step is {}"
          .format(start_learning_rate, start_learning_rate+(experiment_times * step), step))
    learning_rate_list, train_acc_list, valid_acc_list = [], [], []
    for i in range(experiment_times):
        avg_train_acc, avg_valid_acc = 0, 0
        learning_rate = start_learning_rate+(step * i)
        learning_rate_list.append(learning_rate)
        test_times = 10
        for j in range(test_times):
            optimizer_tmp = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_op)
            train_acc, valid_acc = train(sess, init, number_epochs, keep_prob, optimizer_tmp,
                                         data_train, data_valid, neural_network)
            avg_train_acc += train_acc
            avg_valid_acc += valid_acc
        avg_train_acc /= test_times
        avg_valid_acc /= test_times
        train_acc_list.append((avg_train_acc))
        valid_acc_list.append((avg_valid_acc))
        print("Learning rate: {:.2f} avg train acc: {:.3f} avg valid acc: {:.3f}".format(learning_rate, avg_train_acc, avg_valid_acc))
    # Draw two curves of accuracy when learing rate is changin
    plt.plot([start_learning_rate, start_learning_rate + step * experiment_times], [0.95, 0.95], '--',label ="accuracy = 95%")
    plt.legend()
    plot_two_curves(learning_rate_list, train_acc_list, learning_rate_list, valid_acc_list,
                    xlabel="Learning_rate", ylabel="Accuracy", title="Learning rate changing experiment")
    plt.text(0,0,"Accuracy changing follow learning rate changing bewteen {:.2f} to {:.2f}, and step is {:.2f}. Numbers of epochs:{}"
            .format(start_learning_rate, start_learning_rate+(experiment_times * step), step, number_epochs))                
    plt.show()
    print("*" * 50) # seperate the output on terminal


# def ep_experiment(sess, init, number_epochs, keep_prob, optimizer, data_train,
#                   data_valid, neural_network):
    



with tf.Session() as sess:
    

    print("# First run of the training set and validating set")
    train_acc, valid_acc, train_acc_list, valid_acc_list = train(sess, init, number_epochs, keep_prob,
                                                                 optimizer, data_train, data_valid,
                                                                 neural_network, print_acc=True, record_acc=True)

    # generate every epoch coressponding to the data pionts in training accarucy set and valid accuracy set
    epochs = [i for i in range(0, number_epochs, 10)]

    # Draw the accuracy curve
    plot_two_curves(epochs, train_acc_list, epochs, valid_acc_list, title="The first training")
    plt.show()
    print("*" * 50) # seperate the output on terminal



    # change the learning rate in a range
    # for each learning rate,  run 10 times of training and get the average accuracy
    # step = 0.1
    # start_learning_rate = 0
    # experiment_times = 2
    # lr_experiment(step, start_learning_rate, experiment_times, sess, init, number_epochs,
    #               keep_prob, loss_op, data_train, data_valid, neural_network)


    # set k-fold validation
    # 2 <= k 
    k = 10
    print("#  {}-fold validation".format(k))
    avg_train_acc, avg_valid_acc = 0, 0
    # ndarrays for storing the accuracy information every 10 epochs
    ntrain_acc_list = np.zeros(number_epochs // 10)
    nvalid_acc_list = np.zeros(number_epochs // 10)
    # Get the round number, accuracy and accuracy recording list of every round in k-fold validation
    for round, train_acc, valid_acc, train_acc_list, valid_acc_list in k_fold(sess, init, number_epochs, keep_prob, optimizer,
                                                                              data_train, data_valid, neural_network, k):
        
        ntrain_acc_list = ntrain_acc_list + np.array(train_acc_list)
        nvalid_acc_list = nvalid_acc_list + np.array(valid_acc_list)
        avg_train_acc = avg_train_acc + train_acc
        avg_valid_acc = avg_valid_acc + valid_acc
        # save the trained network
        # save_path = saver.save(sess, "my_ann/trained_net_{}.ckpt".format(round))

    # Darw an average accuracy curve
    ntrain_acc_list = ntrain_acc_list / k
    nvalid_acc_list = nvalid_acc_list / k 
    avg_train_acc = avg_train_acc / k
    avg_valid_acc = avg_valid_acc / k
    print("{}-fold average train acc: {} valid acc: {}".format(k, avg_train_acc, avg_valid_acc))
    plot_two_curves(epochs, ntrain_acc_list.tolist(), epochs, nvalid_acc_list.tolist(),
                            title="{}-fold average accuracy".format(k))
    plt.show()
    print("*" * 50) # seperate the output on terminal
    

    print("# Test model")
    x_test, y_test, labels_test = data_seperate(data_test)
    # Print the last 5 predictions
    pred = tf.nn.softmax((neural_network))
    output = pred.eval({X: x_test, KP: 1})[-5:,:]
    predicted_class_label = [get_weight_type(labels) for labels in np.argmax(output, axis=1)]
    print("Last 5 rows of y data and label:")
    print(*data[-5:,-3:].tolist(), sep='\n')
    print("Last 5 predictions:",
          *[item for item in zip(output.round(decimals=1).tolist(), predicted_class_label)], sep='\n')

    # Print the accuracy of test data
    correct_prediction = tf.equal(tf.argmax(pred, 1), labels_test)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Test accuracy:", accuracy.eval({X: x_test, KP: 1}))

print("The data was downloaded in path:", dataset_path)


