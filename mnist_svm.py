"""
mnist_svm
~~~~~~~~~

一个使用SVM分类器识别MNIST数据集中手写数字的分类器程序。"""

# 自定义库
import mnist_loader  # 导入MNIST数据加载模块

# 第三方库
from sklearn import svm  # 导入sklearn的支持向量机模块


def svm_baseline():
    training_data, validation_data, test_data = mnist_loader.load_data()  # 加载MNIST数据集
    # 训练
    clf = svm.SVC()  # 创建SVM分类器实例
    clf.fit(training_data[0], training_data[1])  # 使用训练数据训练SVM分类器
    # 测试
    predictions = [int(a) for a in clf.predict(test_data[0])]  # 对测试数据进行预测并转换为整数
    num_correct = sum(int(a == y) for a, y in zip(predictions, test_data[1]))  # 计算正确预测的数量
    print("Baseline classifier using an SVM.")  # 打印分类器信息
    print("%s of %s values correct." % (num_correct, len(test_data[1])))  # 打印正确率


if __name__ == "__main__":
    svm_baseline()  # 如果直接运行此脚本，则执行SVM基准测试

