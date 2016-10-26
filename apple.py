#coding:utf8
import numpy as np
#import matplotlib.pyplot as plt
import tensorflow as tf

#変数を入れる箱
#x:リンゴみかんを何個ずつ買うか
#y:合計で何円か
x = tf.placeholder(tf.float32, shape=(None, 2), name="x")
y_ = tf.placeholder(tf.float32, shape=(None, 1), name="y")

#モデル式の係数をいれる箱
#a:リンゴみかんがそれぞれ1個何円か
a = tf.Variable(tf.zeros((2,1)),name="a")

#モデル式そのもの y=[a1,a2]*[x1,x2].T
y = tf.matmul(x,a)

#誤差は誤差二乗平均
loss = tf.reduce_mean(tf.square(y_-y))
#最適化は最急降下法
train_step = tf.train.GradientDescentOptimizer(0.02).minimize(loss)

#学習データ
train_x = np.array([[1.,3.],[3.,1.],[5.,7.]])
train_y = np.array([190.,330.,660.]).reshape(3,1)

print 'x',train_x
print 'y',train_y

#おまじない
sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

for i in range(100):
    #引数1に置いた変数を返すためにsessionをまわす
    #train_stepにはlossが，lossにはy_，,yが，yにはx,aがひもづいているし，
    #aが変数と登録してあるからaを更新しにいく
    sess.run(train_step,feed_dict={x:train_x,y_:train_y})
    if (i+1) % 10 == 0:
        print i+1

est_a = sess.run(a,feed_dict={x:train_x,y_:train_y})

print est_a[0],est_a[1]


new_x=np.array([2.,4.]).reshape(1,2)

new_y = sess.run(y, feed_dict={x:new_x})
print new_y

sess.close()
