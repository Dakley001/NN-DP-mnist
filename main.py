"""
main.py
~~~~~~~

主程序文件，用于加载MNIST数据集、创建神经网络和训练模型。
这是神经网络示例项目的入口点，演示了如何使用反向传播算法
训练一个简单的前馈神经网络来识别手写数字。
"""

import mnist_loader  # 导入MNIST数据加载模块
import network  # 导入神经网络模块

# 加载MNIST数据集，分成训练、验证和测试数据
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# 创建一个具有三层的神经网络：784个输入神经元(对应28x28像素图像)，30个隐藏层神经元和10个输出神经元(对应0-9数字)
net = network.Network([784, 30, 10])

# 使用随机梯度下降法训练模型：30个训练周期，每个小批量10个样本，学习率为3.0，使用测试数据评估模型
net.SGD(training_data, epochs=30, mini_batch_size=10, eta=3.0, test_data=test_data)