import tensorflow as tf
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def CSVread(file, variables):
    ''' CSV Format 파일 읽기 '''
    data = pd.read_csv(file, delimiter=',',
                       na_values=['NAN'], header=0,
                       engine='python',
                       usecols=variables)
    return data

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

def _draw(r1, r2):
    r1= ReverseMinMaxScaler(soil_mois_ori,r1)
    r2= ReverseMinMaxScaler(soil_mois_ori,r2)
    """ Draw RNN Train Result """
    plt.plot(r1)
    plt.plot(r2)
    plt.xlabel("Time Period")
    plt.ylabel("10cm")
    line1, = plt.plot(r1, label='Test Y')
    line2, = plt.plot(r2, label='Predicted Y')
    plt.legend([line1, line2], ['Test Y', 'Predicted Y'])
    print(r2)
    plt.show()

file_path = 'D:/Moon/Download/SURFACE_ASOS_90_HR_2017_2017_2018/'
file = 'SURFACE_ASOS_90_HR_2017_2017_2018_2.csv'
variables = ["10cm", "상대습도", "일조시간","증발산량"]
my_data = CSVread(file_path + file, variables)

soil_mois_ori = my_data[variables[0]]
huminity = my_data[variables[1]]
sunny = my_data[variables[2]]
ETo = my_data[variables[3]]

soil_mois = MinMaxScaler(soil_mois_ori)
huminity = MinMaxScaler(huminity)
sunny = MinMaxScaler(sunny)
ETo = MinMaxScaler(ETo)

start = 0
end = 24
soil_mois = np.array(soil_mois[start: end])
huminity = np.array(huminity[start: end])
sunny = np.array(sunny[start: end])
ETo = np.array(ETo[start: end])

x__data = np.array([huminity, sunny, ETo])
x___data = x__data.transpose()
y = np.array(soil_mois,ndmin=2)
y___data = y.T
dataX = np.array(x___data)
dataY = np.array(y___data)

W = tf.Variable(tf.random_uniform([3,1]),tf.float32)
b = tf.Variable(tf.random_uniform([1]),tf.float32)

X = tf.placeholder(tf.float32, shape=[None,3])
Y = tf.placeholder(tf.float32, shape=[None,1])

hypothesis = tf.matmul(X,W)+b
cost = tf.reduce_mean(tf.square(hypothesis - dataY))
rate = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(rate)
train = optimizer.minimize(cost)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for step in range(3000):
    cost_val, predicted,_ = sess.run([cost,hypothesis,train], feed_dict={X:dataX ,Y:dataY})
    if step%1000 == 0:
        print(step, "cost: ", cost_val, "\nPrediction:\n", predicted, sess.run(W), sess.run(b) )
sess.close()

_draw(dataY, predicted)
