#coding:utf8
import numpy as np
#import matplotlib.pyplot as plt
import tensorflow as tf



#---------------------------------------------------------
#tfにモデルの概形を伝える
#---------------------------------------------------------

#変数を入れる箱
#x:リンゴみかんを何個ずつ買うか
#y:買える1,買えない0
x = tf.placeholder(tf.float32, shape=(None, 2), name="x")
y_ = tf.placeholder(tf.float32, shape=(None, 1), name="y")

#モデル式の係数=最適化する値をいれる箱
#a:リンゴみかんの単価
#b:財布の中身
a = tf.Variable(tf.ones((2,1)),name="a")
b = tf.Variable(200.,name="b")


#モデル式そのもの y=sigmoid([a1,a2]*[x1,x2].T+b)
u = tf.matmul(x,a)+b
y = tf.sigmoid(u)

#誤差はクロスエントロピー
# loss = tf.reduce_mean(tf.square(y_-y))
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(u,y_))

#最適化は最急降下法
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

#---------------------------------------------------------
#tfにモデルの概形を伝え終わりました
#---------------------------------------------------------

#---------------------------------------------------------
#対象データ読み込み
#---------------------------------------------------------
#学習データ
train_x = np.array([[2.,3.],[0.,16.],[3.,1.],[2.,8.]])
train_y = np.array([1.,1.,0.,0.]).reshape(4,1)

print 'x',train_x
print 'y',train_y


#---------------------------------------------------------
#ループまわす準備，初期化
#---------------------------------------------------------

#おまじない
sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

#---------------------------------------------------------
#実際のがくしゅうるーぷ
#---------------------------------------------------------
for i in range(1000):
    #引数1に置いた変数を返すためにsessionをまわす
    #train_stepにはlossが，lossにはy_，,yが，yにはx,aがひもづいているし，
    #aが変数と登録してあるからaを更新しにいく
    _,a_,b_,l=sess.run([train_step,a,b,loss],feed_dict={x:train_x,y_:train_y})
    if (i+1) % 100 == 0:
        print i+1,a_,b_,l

est_a = sess.run(a,feed_dict={x:train_x,y_:train_y})

print est_a[0],est_a[1]

#---------------------------------------------------------
#tfに渡したもでるで予測してみる
#---------------------------------------------------------
new_x=np.array([1.,11.]).reshape(1,2)

new_y = sess.run(y, feed_dict={x:new_x})
print new_y

sess.close()
