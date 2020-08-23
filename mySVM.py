import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score
from sklearn.metrics import precision_recall_fscore_support, classification_report


# 加载手势训练数据时，将类别标签转换为整数
def label_type(s):
    it = {'finger': 0, 'palm': 1, 'other': 2}
    return it[str(s, encoding="utf-8")]


# 计算SVM性能指标
def R_P():
    y_true = np.array([1, 1, 1, 1, 0, 0])
    y_hat = np.array([1, 0, 1, 1, 1, 1])

    print('Accuracy:\t', accuracy_score(y_true, y_hat))

    precision = precision_score(y_true, y_hat)
    print('Precision:\t', precision)

    recall = recall_score(y_true, y_hat)
    print('Recall:\t', recall)

    print('f1 score:\t', f1_score(y_true, y_hat))
    # print(2*(precision*recall)/(precision + recall))

    print('F-beta:\n')
    for beta in np.logspace(-3, 3, num=7, base=10):
        fbeta = fbeta_score(y_true, y_hat, beta=beta)
        print('\tbeta=%9.3f\tF-beta=%.3f' % (beta, fbeta))

    print(precision_recall_fscore_support(y_true, y_hat))
    print(classification_report(y_true, y_hat))


# 显示准确率
def show_accuracy(a, b):
    # 计算预测值和真实值一样的正确率
    acc = a.ravel() == b.ravel()
    print('precision:%.2f%%' % ((100 * float(acc.sum())) / a.size))


# 显示召回率
def show_recall(y, y_hat):
    # 提取出那个小样本集中的预测和真实一样的正确率
    print('Recall"%.2f%%' % (100 * float(np.sum(y_hat[y == 1] == 1)) / np.extract(y == 1, y).size))


# 训练手形识别svm
# path = 'handdata.txt' 手势训练数据文件路径
# startFeatureCol=0，属性数据起始列号（0起编号），起始列之前可能是一些序号之类的对训练无用的列
# endFeatureCol=2,属性数据结束列，属性数据取到第endFeatureCol-1列
# labelCol=2类别标签所在列号（0起编号），一般是做后一列
# modelpath="hand_svm_model.m"训练完的模型路径
def svm_train(datapath='handdata.txt', startFeatureCol=0, endFeatureCol=2, labelCol=2, modelpath="hand_svm_model.m"):
    print('加载SVM训练数据...', end='')
    data = np.loadtxt(datapath, dtype=float, delimiter=' ', converters={labelCol: label_type})
    x, y = np.split(data, (labelCol,), axis=1)  # split(数据，分割位置(此处设置为labelCol，即类别标签之前所有列)，轴=1（水平分割） or 0（垂直分割）)
    x = x[:, startFeatureCol:endFeatureCol]  # 只保留属性数据起始列到结束列之前的特征数据
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.95)
    print('OK')
    print('训练SVM模型...', end='')
    # kernel = 'linear'时，为线性核，C越大分类效果越好，但有可能会过拟合（defaul C = 1）。
    # kernel = 'rbf'时（default），为高斯核，gamma值越小，分类界面越连续；gamma值越大，分类界面越“散”，分类效果越好，但有可能会过拟合。
    # decision_function_shape = 'ovr'时，为one v rest，即一个类别与其他类别进行划分，
    # decision_function_shape = 'ovo'时，为one v one，即将类别两两之间进行划分，用二分类的方法模拟多分类的结果。
    clf = svm.SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovr')  # 模型设置
    clf.fit(x_train, y_train.ravel())  # 训练
    print('OK')
    print('训练精度', clf.score(x_train, y_train))  # 精度
    y_hat = clf.predict(x_train)
    # show_accuracy(y_hat, y_train)
    print('测试精度', clf.score(x_test, y_test))
    y_hat = clf.predict(x_test)
    # show_accuracy(y_hat, y_test)
    joblib.dump(clf, modelpath)
    return clf


# svm预测，x：特征数据,modelpath="hand_svm_model.m"svm模型路径
def svm_predict(x, modelpath="hand_svm_model.m"):
    clf = joblib.load(modelpath)
    y = clf.predit(x)
    return y

if __name__ == '__main__':  # 程序从这儿开始执行
    svm_train(datapath = 'handdata_2features.txt',startFeatureCol=0,endFeatureCol=2,labelCol=2,modelpath="hand_svm_model2.m")#训练手形识别svm
