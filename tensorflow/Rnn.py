import tensorflow as tf
from tensorflow.contrib import rnn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def MinMaxScaler(data):
    '''
    Min Max Normalization
    http://sebastianraschka.com/Articles/2014_about_feature_scaling.html
    '''
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # 0으로 나눠지는 것을 방지하기 위해 아주작은 값을 분모에 더합니다.
    return numerator / (denominator + 1e-7)

def ReverseMinMaxScaler(org_x, x):
    org_x_np = np.asarray(org_x)
    x_np = np.asarray(x)
    return (x_np * (org_x_np.max() - org_x_np.min() + 1e-7)) + org_x_np.min()

def FindMinMax(data):
    Mindata = min(data)
    Maxdata = np.max(data, 0)
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

def getStartEnd(startDate, endDate, time):
    ''' 특정 계절에 대한 시작/끝 인덱스'''
    start, end = 1, 1
    for idx in range(len(time)):
        if (time[idx] == startDate):
            start = idx
        if (time[idx] == endDate):
            end = idx
            break
    return [start, end]

def _draw(r1, r2):
    r1= ReverseMinMaxScaler(soil_mois,r1)
    r2= ReverseMinMaxScaler(soil_mois,r2)
    """ Draw RNN Train Result """
    plt.plot(r1)
    plt.plot(r2)
    plt.xlabel("Time Period")
    plt.ylabel("10cm")
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
    #train_size = int(len(dataY) * 0.8)
    #trainX, testX = np.array(dataX[: train_size]), np.array(dataX[train_size:])
    #trainY, testY = np.array(dataY[: train_size]), np.array(dataY[train_size:])

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
            _, cost = sess.run([train, loss], feed_dict={X: dataX, Y: dataY})
            if (i + 1) % (iterations / 20) == 0:
                print("[step: {}] loss: {}".format(i + 1, cost))
                print(y)
        # train_predict = sess.run(Y_pred, feed_dict={X: trainX})
        test_predict = sess.run(Y_pred, feed_dict={X: dataX})
        rmse_val = sess.run(rmse, feed_dict={
            targets: dataY, predictions: test_predict})
        print("RMSE: {}".format(rmse_val))
    return dataY, test_predict

if __name__ == "__main__":
    # Settings
    file_path = 'D:/Moon/Download/SURFACE_ASOS_90_HR_2017_2017_2018/'
    file = 'SURFACE_ASOS_90_HR_2017_2017_2018_3.csv'
    variables = ["10cm", "상대습도", "일조시간","증발산량", "일시"]

    # Define Hyperparameter
    seq_length = 2
    predict_length = 1
    learning_rate = 0.01
    iterations = 3000
    data_dim = 4
    hidden_dim = 3
    output_dim = 1

    # Read data & Get variables
    my_data = CSVread(file_path + file, variables)
    soil_mois = my_data[variables[0]]
    huminity = my_data[variables[1]]
    sunny = my_data[variables[2]]
    ETo = my_data[variables[3]]
    time = my_data[variables[4]]


    Minsoil, Maxsoil = FindMinMax(soil_mois)

    soil_mois = MinMaxScaler(soil_mois)
    huminity = MinMaxScaler(huminity)
    sunny = MinMaxScaler(sunny)
    ETo = MinMaxScaler(ETo)

    start = 0
    endnum = 24
    # Get Summer Season
    #start, end = getStartEnd("1", "60", time)
    soil_mois = np.array(soil_mois[start: endnum])
    huminity = np.array(huminity[start: endnum])
    sunny = np.array(sunny[start: endnum])
    ETo = np.array(ETo[start: endnum])
    DataLength = endnum

    x = np.array([soil_mois, huminity, sunny, ETo])
    x = x.transpose()
    y = soil_mois
    print(x)
    dataY, test_predict = Model(x, y, DataLength, seq_length, iterations)
    _draw(MinMaxReturn(dataY, Minsoil, Maxsoil), MinMaxReturn(test_predict, Minsoil, Maxsoil))
