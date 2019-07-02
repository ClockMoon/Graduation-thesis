import tensorflow as tf
from tensorflow.contrib import rnn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    return numerator / (denominator + 1e-7)

def ReverseMinMaxScaler(org_x, x):
    org_x_np = np.asarray(org_x)
    x_np = np.asarray(x)
    return (x_np * (org_x_np.max() - org_x_np.min() + 1e-7)) + org_x_np.min()


def FindMinMax(data):
    Mindata = np.min(data,0)
    Maxdata = np.max(data,0)
    return Mindata, Maxdata


def MinMaxReturn(val, Min, Max):
    ''' Return normalized data '''
    return val * (Max - Min + 1e-7) + Min

def CSVread(file, variables):
    ''' CSV Format 파일 읽기 '''
    data = pd.read_csv(file, delimiter=',',
                       na_values=['NAN'], header=0,
                       engine='python',
                       usecols=variables)
    return data

def _draw(r1, r2):
    """ Draw RNN Train Result """
    r1= ReverseMinMaxScaler(soil_humi,r1)
    r2= ReverseMinMaxScaler(soil_humi,r2)
    plt.plot(r1)
    plt.plot(r2)
    plt.xlabel("Time Period")
    plt.ylabel("Soil Moisture")
    line1, = plt.plot(r1, label='Test Y')
    line2, = plt.plot(r2, label='Predicted Y')
    plt.legend([line1, line2], ['Test Y', 'Predicted Y'])
    plt.show()

def Model(x, y, summerLength, seq_length, iterations):
    ''' Main model '''
    dataX, dataY = [], []
    for i in range(0, summerLength - seq_length - 1):
        size = i + seq_length
        _x = x[i: size]
        _y = y[size: size + predict_length]
        dataX.append(_x)
        dataY.append(_y)
    dataX = np.array(dataX)
    dataY = np.array(dataY)
    # train/test split
    train_size = int(len(dataY) * 0.8)
    trainX, testX = np.array(dataX[: train_size]), np.array(dataX[train_size:])
    trainY, testY = np.array(dataY[: train_size]), np.array(dataY[train_size:])

    # input place holders
    X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
    Y = tf.placeholder(tf.float32, [None, 1])

    cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh)

    def lstm_cell():
        ''' Use LSTM model '''
        cell = rnn.BasicLSTMCell(hidden_dim, state_is_tuple=True)
        return cell

    # Build a Multi RNN LSTM network
    # multi_cells = rnn.MultiRNNCell([lstm_cell() for _ in range(5)], state_is_tuple=True)
    # utputs, _states = tf.nn.dynamic_rnn(multi_cells, X, dtype = tf.float32)

    # Build a LSTM network
    cell = tf.contrib.rnn.BasicLSTMCell(
        num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh)
    outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
    Y_pred = tf.contrib.layers.fully_connected(outputs[:, -1], output_dim,
                                               activation_fn=None)  # We use the last cell's output

    # cost/loss
    loss = tf.reduce_sum(tf.square(Y_pred - Y))  # sum of the squares
    # optimizer
    optimizer = tf.train.RMSPropOptimizer(learning_rate)
    train = optimizer.minimize(loss)

    # RMSE
    targets = tf.placeholder(tf.float32, [None, 1])
    predictions = tf.placeholder(tf.float32, [None, 1])
    rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        # Training step
        for i in range(iterations):
            _, cost = sess.run([train, loss], feed_dict={X: trainX, Y: trainY})
            if (i + 1) % (iterations / 20) == 0:
                print("[step: {}] loss: {}".format(i + 1, cost))
        # train_predict = sess.run(Y_pred, feed_dict={X: trainX})
        test_predict = sess.run(Y_pred, feed_dict={X: testX})
        rmse_val = sess.run(rmse, feed_dict={
            targets: testY, predictions: test_predict})
        print("RMSE: {}".format(rmse_val))
    return testY, test_predict


if __name__ == "__main__":
    # Settings
    file_path = 'D:/Moon/Download/SURFACE_ASOS_90_HR_2017_2017_2018/'
    file = 'SURFACE_ASOS_90_HR_2017_2017_2018_3.csv'
    variables = ["10cm","상대습도","일조시간","증발산량","일시"]

    # Define Hyperparameter
    seq_length = 4
    iterations = 5000
    hidden_dim = 2
    predict_length = 1
    learning_rate = 0.0001
    data_dim = 4
    output_dim = 1

    # Read data & Get variables
    my_data = CSVread(file_path + file, variables)
    soil_humi = my_data[variables[0]]
    huminity = my_data[variables[1]]
    sunny = my_data[variables[2]]
    ETo = my_data[variables[3]]
    time = my_data[variables[4]]
    Mintemp, Maxtemp = FindMinMax(soil_humi)

    soil_humi = MinMaxScaler(soil_humi)
    huminity = MinMaxScaler(huminity)
    sunny = MinMaxScaler(sunny)
    ETo = MinMaxScaler(ETo)
    # Get Summer Season
    start = 0
    end = 2920
    day_soil_humi = np.array(soil_humi[start: end])
    day_huminity = np.array(huminity[start: end])
    day_sunny = np.array(sunny[start: end])
    day_ETo = np.array(ETo[start: end])
    day_Length = end - start + 1

    x = np.array([day_soil_humi,day_huminity,day_sunny,day_ETo])
    x = x.transpose()
    y = day_soil_humi

    testY, test_predict = Model(x, y, day_Length, seq_length, iterations)
    _draw(MinMaxReturn(testY, Mintemp, Maxtemp), MinMaxReturn(test_predict, Mintemp, Maxtemp))

