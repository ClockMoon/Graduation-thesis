import tensorflow as tf
import numpy as np

#data
x_data = [[0.,2.,0.,4.,0.],
          [1.,0.,3.,0.,5.]] #x_data를 matrix로 변경
y_data = [1,2,3,4,5]

print(x_data)

#Weight
W = tf.Variable(tf.random_uniform([1,2],-1.0,1.0)) #가중치도 1x2로 변경
b = tf.Variable(tf.random_uniform([1],-1.0,1.0))

#hypothesis
hypothesis = tf.matmul(W, x_data) + b #W*X 의 행렬곱

cost = tf.reduce_mean(tf.square(hypothesis - y_data))

#minimize
a = tf.Variable(0.1) #alpha, learning rate
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

# befor starting, initialize variables
init = tf.global_variables_initializer()

#launch
sess = tf.Session()
sess.run(init)

# fit the line
for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
        print (step, sess.run(cost), sess.run(W), sess.run(b) )