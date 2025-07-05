"""
mnist_loader
~~~~~~~~~~~~

用于加载MNIST图像数据的库。有关返回的数据结构的详细信息，请参阅
``load_data``和``load_data_wrapper``的文档字符串。在实践中，
``load_data_wrapper``是我们神经网络代码通常调用的函数。
"""

# 标准库
# import cPickle  # 适用于python2，序列化和反序列化对象
import pickle  # 适用于python3，序列化和反序列化对象
import gzip  # 用于读取gzip压缩文件

# 第三方库
import numpy as np  # 用于数值计算的库


def load_data():
    """返回包含训练数据、验证数据和测试数据的MNIST数据元组。

    ``training_data``作为包含两个条目的元组返回。
    第一个条目包含实际的训练图像。这是一个包含50,000个条目的numpy ndarray。
    每个条目又是一个包含784个值的numpy ndarray，表示单个MNIST图像中的
    28 * 28 = 784个像素。

    ``training_data``元组中的第二个条目是一个包含50,000个条目的numpy ndarray。
    这些条目只是对应于元组的第一个条目中包含的图像的数字值(0...9)。

    ``validation_data``和``test_data``类似，只是每个只包含10,000张图像。

    这是一种很好的数据格式，但对于在神经网络中使用，
    对``training_data``的格式进行一些修改是有帮助的。
    这在下面的包装函数``load_data_wrapper()``中完成。
    """
    f = gzip.open('data/mnist.pkl.gz', 'rb')  # 以二进制读取模式打开gzip文件
    # 因为 Python 2 的 pickle 文件可能包含二进制数据，而latin1可以正确处理这些数据
    training_data, validation_data, test_data = pickle.load(f, encoding='latin1')  # 加载pickle数据，使用latin1编码
    f.close()  # 关闭文件
    return (training_data, validation_data, test_data)  # 返回加载的数据


def load_data_wrapper():
    """返回包含``(training_data, validation_data, test_data)``的元组。
    基于``load_data``，但格式更适合在我们的神经网络实现中使用。

    特别地，``training_data``是一个包含50,000个
    2元组``(x, y)``的列表。``x``是一个784维numpy.ndarray，
    包含输入图像。``y``是一个10维numpy.ndarray，
    表示对应于``x``的正确数字的单位向量。

    ``validation_data``和``test_data``是包含10,000个
    2元组``(x, y)``的列表。在每种情况下，``x``是一个
    784维numpy.ndarry，包含输入图像，``y``是相应的
    分类，即对应于``x``的数字值(整数)。

    显然，这意味着我们对训练数据和验证/测试数据使用略有不同的格式。
    这些格式被证明是在我们的神经网络代码中使用的最方便的格式。"""
    tr_d, va_d, te_d = load_data()  # 加载原始MNIST数据
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]  # 将训练输入重塑为784x1列向量
    training_results = [vectorized_result(y) for y in tr_d[1]]  # 将训练标签转换为10维单位向量
    training_data = list(zip(training_inputs, training_results))  # 组合输入和期望输出为训练数据列表
    
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]  # 将验证输入重塑为784x1列向量
    validation_data = list(zip(validation_inputs, va_d[1]))  # 组合验证输入和标签
    
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]  # 将测试输入重塑为784x1列向量
    test_data = list(zip(test_inputs, te_d[1]))  # 组合测试输入和标签
    
    return (training_data, validation_data, test_data)  # 返回处理后的数据


def vectorized_result(j):
    """返回一个10维单位向量，在第j个位置为1.0，
    其他位置为0。用于将数字(0...9)转换为神经网络的期望输出。"""
    e = np.zeros((10, 1))  # 创建一个10x1的全零向量
    e[j] = 1.0  # 将第j个位置设置为1.0
    return e  # 返回结果向量
