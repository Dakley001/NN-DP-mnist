"""
mnist_average_darkness
~~~~~~~~~~~~~~~~~~~~~~

一个用于识别MNIST数据集中手写数字的朴素分类器。
该程序基于数字的暗度（像素值之和）进行分类——
思想是像"1"这样的数字往往比"8"这样的数字暗度低，
因为后者形状更复杂。当展示一个图像时，分类器返回训练数据中
平均暗度最接近的数字。

该程序分两步工作：首先训练分类器，然后将分类器
应用于MNIST测试数据，以查看有多少数字被正确分类。

不用说，这不是一个很好的识别手写数字的方法！
不过，它有助于展示朴素想法可以得到什么样的性能。"""

# 标准库
from collections import defaultdict  # 导入默认字典，用于统计各数字的暗度

# 自定义库
import mnist_loader  # 导入MNIST数据加载模块


def main():
    training_data, validation_data, test_data = mnist_loader.load_data()  # 加载MNIST数据集
    # 训练阶段：根据训练数据计算每个数字的平均暗度
    avgs = avg_darknesses(training_data)  # 计算每个数字的平均暗度
    # 测试阶段：查看有多少测试图像被正确分类
    num_correct = sum(int(guess_digit(image, avgs) == digit)  # 对每个测试样本进行预测并与实际值比较
                      for image, digit in zip(test_data[0], test_data[1]))
    print("Baseline classifier using average darkness of image.")  # 打印分类器信息
    print("%s of %s values correct." % (num_correct, len(test_data[1])))  # 打印正确率


def avg_darknesses(training_data):
    """返回一个defaultdict，其键是0到9的数字。
    对于每个数字，我们计算包含该数字的训练图像的平均暗度值。
    任何特定图像的暗度只是每个像素的暗度之和。"""
    digit_counts = defaultdict(int)  # 创建一个默认字典，用于记录每个数字出现的次数
    darknesses = defaultdict(float)  # 创建一个默认字典，用于记录每个数字的总暗度
    for image, digit in zip(training_data[0], training_data[1]):  # 遍历训练数据
        digit_counts[digit] += 1  # 增加当前数字的计数
        darknesses[digit] += sum(image)  # 累加当前数字图像的总暗度
    avgs = defaultdict(float)  # 创建一个默认字典，用于存储每个数字的平均暗度
    for digit, n in digit_counts.items():  # 遍历每个数字及其出现次数（使用items而不是iteritems，适配Python 3）
        avgs[digit] = darknesses[digit] / n  # 计算平均暗度
    return avgs  # 返回平均暗度字典


def guess_digit(image, avgs):
    """返回训练数据中平均暗度与'image'的暗度最接近的数字。
    注意，'avgs'被假定为一个defaultdict，其键是0...9，
    其值是训练数据中相应的平均暗度。"""
    darkness = sum(image)  # 计算当前图像的总暗度
    distances = {k: abs(v - darkness) for k, v in avgs.items()}  # 计算当前图像与每个数字平均暗度的距离（使用items而不是iteritems）
    return min(distances, key=lambda x: distances[x])  # 返回距离最小的数字（使用lambda函数明确指定key参数类型）


if __name__ == "__main__":
    main()  # 如果直接运行此脚本，则执行主函数
