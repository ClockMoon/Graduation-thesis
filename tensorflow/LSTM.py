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
    # 0으로 나눠지는 것을 방지하기 위해 아주작은  값을 분모에 더합니다.
    return numerator / (denominator + 1e-7)

def FindMinMax(data):
    print(data)
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

'''def getStartEnd(startDate, endDate, time):
     특정 계절에 대한 시작/끝 인덱스
    start, end = 1, 1
    for idx in range(len(time)):
        #print(time[idx])
        if time[idx] == startDate:
            start = idx
        if time[idx] == endDate:
            end = idx
            break
    return [start, end]'''
def _draw(r1, r2):
    """ Draw RNN Train Result """
    plt.plot(r1)
    plt.plot(r2)
    plt.xlabel("Time Period")
    plt.ylabel("Temperature")
    line1, = plt.plot(r1, label='Test Y')
    line2, = plt.plot(r2, label='Predicted Y')
    plt.legend([line1, line2], ['Test Y', 'Predicted Y'])
    plt.show()

def Model(x, y, summerLength, seq_length, iterations):
    ''' Main model '''
    print("why?!")
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
    file = '엑셀루.csv'
    #variables = ["증발산량", "상대습도", "풍속", "강우량", "일조시간", "최저온도", "최고온도"]
    variables = ["증발산량","평균온도"]
    my_data = CSVread(file_path + file, variables)

    # Define Hyperparameter
    seq_length = 3
    predict_length = 1
    learning_rate = 0.001
    iterations = 3000
    data_dim = 2
    hidden_dim = 6
    output_dim = 1

    # Read data & Get variables
    ETo = my_data[variables[0]]
    Heat = my_data[variables[1]]
    #huminity = my_data[variables[1]]
    #windV = my_data[variables[2]]
    #rain = my_data[variables[3]]
    #sunny = my_data[variables[4]]
    #lowTemp = my_data[variables[5]]
    #highTemp = my_data[variables[6]]

    MinETo, MaxETo = FindMinMax(ETo)

    ETo = MinMaxScaler(ETo)
    Heat = MinMaxScaler(Heat)
    #huminity = MinMaxScaler(huminity)
    #windV = MinMaxScaler(windV)
    #rain = MinMaxScaler(rain)
    #sunny = MinMaxScaler(sunny)
    #lowTemp = MinMaxScaler(lowTemp)
    #highTemp = MinMaxScaler(highTemp)
    # Get Summer Season
    start = 0
    end = 2918
    day_ETo= np.array(ETo[start: end])
    day_heat = np.array(Heat[start:end])
    #day_huminity = np.array(huminity[start: end])
    #day_windV = np.array(windV[start: end])
    #day_rain = np.array(rain[start: end])
    #day_sunny = np.array(sunny[start: end])
    #day_lowTemp = np.array(lowTemp[start: end])
    #day_highTemp = np.array(highTemp[start: end])
    day_Length = end - start + 1

    #x = np.array([day_ETo,day_huminity,day_windV,day_sunny,day_sunny,day_lowTemp,day_highTemp])
    x = np.array([day_ETo,day_heat])
    x = x.transpose()
    y = day_ETo
    print(x,y)

    testY, test_predict = Model(x, y, day_Length, seq_length, iterations)
    _draw(MinMaxReturn(testY, MinETo, MaxETo), MinMaxReturn(test_predict, MinETo, MaxETo))