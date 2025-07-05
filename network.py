"""
network.py
~~~~~~~~~~

实现一个前馈神经网络的随机梯度下降学习算法的模块。梯度使用反向传播计算。
请注意，我专注于使代码简单、易于阅读和易于修改。它没有进行优化，
并且省略了许多理想的特性。
"""

# 标准库
import random  # 导入随机数生成模块，用于打乱训练数据

# 第三方库
import numpy as np  # 导入NumPy库，用于数值计算


class Network(object):

    def __init__(self, sizes):
        """参数列表``sizes``包含神经网络各层中神经元的数量。
        例如，如果列表是[2, 3, 1]，那么它将是一个三层网络，
        第一层包含2个神经元，第二层3个神经元，第三层1个神经元。
        网络的偏置和权重使用均值为0、方差为1的高斯分布随机初始化。
        注意，第一层被假定为输入层，按照惯例，我们不会为这些神经元设置任何偏置，
        因为偏置仅在计算后续层的输出时使用。"""
        self.num_layers = len(sizes)  # 记录神经网络的层数
        self.sizes = sizes  # 存储每层的神经元数量
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]  # 除输入层外每层的偏置初始化为随机值
        self.weights = [np.random.randn(y, x)  # 每两层之间的权重初始化为随机值
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """如果``a``是输入，返回神经网络的输出。"""
        for b, w in zip(self.biases, self.weights):  # 遍历每一层的偏置和权重
            a = sigmoid(np.dot(w, a) + b)  # 计算当前层的激活值：sigmoid(w·a + b)
        return a  # 返回最终的输出

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """使用小批量随机梯度下降训练神经网络。
        ``training_data``是一个元组``(x, y)``的列表，表示训练输入和期望输出。
        其他非可选参数不言自明。如果提供了``test_data``，
        则网络将在每个训练周期后针对测试数据进行评估，并打印部分进度。
        这对于跟踪进度很有用，但会大大减慢速度。"""
        if test_data:
            n_test = len(test_data)  # 如果提供测试数据，获取测试数据长度
        n = len(training_data)  # 训练数据的长度
        for j in range(epochs):  # 遍历每个训练周期
            random.shuffle(training_data)  # 随机打乱训练数据
            mini_batches = [  # 创建小批量数据
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]
            for mini_batch in mini_batches:  # 遍历每个小批量
                self.update_mini_batch(mini_batch, eta)  # 使用当前小批量更新权重和偏置
            if test_data:  # 如果提供了测试数据
                print("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test))  # 打印当前周期的测试准确度
            else:
                print("Epoch {0} complete".format(j))  # 打印周期完成信息

    def update_mini_batch(self, mini_batch, eta):
        """通过使用反向传播应用梯度下降，更新网络的权重和偏置。
        ``mini_batch``是一个元组``(x, y)``的列表，``eta``是学习率。"""
        nabla_b = [np.zeros(b.shape) for b in self.biases]  # 初始化偏置的梯度为全0
        nabla_w = [np.zeros(w.shape) for w in self.weights]  # 初始化权重的梯度为全0
        for x, y in mini_batch:  # 遍历小批量中的每个样本
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)  # 计算当前样本的梯度
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]  # 累加偏置梯度
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]  # 累加权重梯度
        self.weights = [w - (eta / len(mini_batch)) * nw  # 更新权重
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb  # 更新偏置
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """返回一个元组``(nabla_b, nabla_w)``，表示代价函数C_x的梯度。
        ``nabla_b``和``nabla_w``是层次化的numpy数组列表，
        类似于``self.biases``和``self.weights``。"""
        nabla_b = [np.zeros(b.shape) for b in self.biases]  # 初始化偏置梯度为全0
        nabla_w = [np.zeros(w.shape) for w in self.weights]  # 初始化权重梯度为全0
        # 前向传播
        activation = x  # 初始激活值为输入x
        activations = [x]  # 存储所有激活值的列表，逐层
        zs = []  # 存储所有z向量的列表，逐层
        for b, w in zip(self.biases, self.weights):  # 遍历每一层
            z = np.dot(w, activation) + b  # 计算加权输入z
            zs.append(z)  # 保存z
            activation = sigmoid(z)  # 计算激活值
            activations.append(activation)  # 保存激活值
        # 反向传播
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])  # 计算输出层的误差
        nabla_b[-1] = delta  # 输出层偏置的梯度
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())  # 输出层权重的梯度
        # 注意，下面循环中的变量l与书中第2章的表示法略有不同。
        # 这里，l=1表示最后一层神经元，l=2是倒数第二层，依此类推。
        # 这是书中方案的重新编号，在这里用来利用Python可以在列表中使用
        # 负索引这一事实。
        for l in range(2, self.num_layers):  # 从倒数第二层开始向前计算
            z = zs[-l]  # 获取当前层的z值
            sp = sigmoid_prime(z)  # 计算sigmoid的导数
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp  # 计算当前层的误差
            nabla_b[-l] = delta  # 当前层偏置的梯度
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())  # 当前层权重的梯度
        return (nabla_b, nabla_w)  # 返回所有梯度

    def evaluate(self, test_data):
        """返回神经网络输出正确结果的测试输入数量。
        注意，假定神经网络的输出是最终层中激活值最大的神经元的索引。
        
        计算过程：
        1. 对每个测试样本，通过feedforward方法获取网络输出
        2. 使用np.argmax找出输出层中激活值最大的神经元索引（预测的数字）
        3. 将预测结果与真实标签y进行比较
        4. 统计预测正确的样本数量并返回
        """
        test_results = [(np.argmax(self.feedforward(x)), y)  # 计算网络输出与期望输出
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)  # 返回正确预测的样本数量

    def cost_derivative(self, output_activations, y):
        r"""返回向量 \partial C_x / \partial a，表示输出激活值的偏导数。"""
        return (output_activations - y)  # 返回输出激活值与期望输出的差


# 辅助函数
def sigmoid(z):
    """sigmoid激活函数。"""
    return 1.0 / (1.0 + np.exp(-z))  # 返回sigmoid函数值


def sigmoid_prime(z):
    """sigmoid函数的导数。"""
    return sigmoid(z) * (1 - sigmoid(z))  # 返回sigmoid函数的导数
