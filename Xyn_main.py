# 引入需要的库
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


############################################## 数据集 #################################################
# 下载数据
IRIS_TRAIN_URL = "iris_training.csv"
IRIS_TEST_URL = "iris_test.csv"

names = ['sepal-length', 'sepal-width', 'petal-1ength', 'petal-width', 'species']

train = pd.read_csv(IRIS_TRAIN_URL, names=names, skiprows=1)  # 这里需要跳过的第一行数据
test = pd.read_csv(IRIS_TEST_URL, names=names, skiprows=1)

# 训练并测试输入数据
Xtrain = train.drop("species", axis=1)  # 删除最后一列
Xtest = test.drop("species", axis=1)

# 将结果值编码成独热二进制
ytrain = pd.get_dummies(train.species)
ytest = pd.get_dummies(test.species)
for target in ytest.values:
    print(target, target.argmax(axis=0))



############################################## 模型参数 #################################################
# 训练3种神经网络构架；（4-5-3）（4-10-3）（4-20-3）
# 作损失函数对迭代次数图
num_hidden_nodes = [5, 10, 20]
loss_plot = {5: [], 10: [], 20: []}
weights1 = {5: None, 10: None, 20: None}
weights2 = {5: None, 10: None, 20: None}
num_iters = 2000


def create_train_model(hidden_nodes, num_iters):

    # 重置图形
    tf.reset_default_graph()

    # 定义输入输出的占位符
    X = tf.placeholder(shape=(120, 4), dtype=tf.float64, name='X')
    y = tf.placeholder(shape=(120, 3), dtype=tf.float64, name='y')

    # 定义三层神经网络间的两组权重
    W1 = tf.Variable(np.random.rand(4, hidden_nodes), dtype=tf.float64)
    W2 = tf.Variable(np.random.rand(hidden_nodes, 3), dtype=tf.float64)

    # 创建神经网络图
    A1 = tf.sigmoid(tf.matmul(X, W1))
    y_est = tf.sigmoid(tf.matmul(A1, W2))

    # 定义损失函数
    deltas = tf.square(y_est - y)
    loss = tf.reduce_sum(deltas)

    # 定义一个最小化损失函数的训练操作
    optimizer = tf.train.GradientDescentOptimizer(0.005)
    train = optimizer.minimize(loss)

    # 初始化变量并运行会话
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # 迭代
    for i in range(num_iters):
        sess.run(train, feed_dict={X: Xtrain, y: ytrain})
        loss_plot[hidden_nodes].append(sess.run(loss, feed_dict={X: Xtrain.values, y: ytrain.values}))
        weights1 = sess.run(W1) # 更新权重
        weights2 = sess.run(W2)

    print("loss (hidden nodes: %d, iterations: %d): %.2f" % (hidden_nodes, num_iters, loss_plot[hidden_nodes][-1]))
    sess.close()
    return weights1, weights2


plt.figure(figsize=(12,8))
for hidden_nodes in num_hidden_nodes:
    weights1[hidden_nodes], weights2[hidden_nodes] = create_train_model(hidden_nodes, num_iters)
    plt.plot(range(num_iters), loss_plot[hidden_nodes], label="nn: 4-%d-3" % hidden_nodes)

plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(fontsize=12)
plt.show()


############################################## 模型评估 #################################################
# 在测试集上评估模型
X = tf.placeholder(shape=(30, 4), dtype=tf.float64, name='X')
y = tf.placeholder(shape=(30, 3), dtype=tf.float64, name='y')

for hidden_nodes in num_hidden_nodes:

    # Forward propagation
    W1 = tf.Variable(weights1[hidden_nodes])
    W2 = tf.Variable(weights2[hidden_nodes])
    A1 = tf.sigmoid(tf.matmul(X, W1))
    y_est = tf.sigmoid(tf.matmul(A1, W2))

    # 计算预测的输出值
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        y_est_np = sess.run(y_est, feed_dict={X: Xtest, y: ytest})

    # 计算预测的准确度   这里是列方向的比较
    correct = [estimate.argmax(axis=0) == target.argmax(axis=0) for estimate, target in zip(y_est_np, ytest.values)]

    accuracy = 100 * sum(correct) / len(correct)
    print('Network architecture 4-%d-3, accuracy: %.2f%%' % (hidden_nodes, accuracy))

