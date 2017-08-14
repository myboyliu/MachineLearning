import numpy as np
from sklearn import svm
import matplotlib.colors
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from time import time

classifier_type = 'RF'

print('载入训练数据...')
t = time()
data = pd.read_csv('../Total_Data/data/MNIST.train.csv', header=0, dtype=np.int)
print('载入完成，耗时%f秒' % (time() - t))
y = data['label'].values
x = data.values[:, 1:]
print('图片个数：%d，图片像素数目：%d' % x.shape)
images = x.reshape(-1, 28, 28)
y = y.ravel()

print('载入测试数据...')
t = time()
data_test = pd.read_csv('data/MNIST.test.csv', header=0, dtype=np.int)
data_test = data_test.values
images_test_result = data_test.reshape(-1, 28, 28)
print('载入完成，耗时%f秒' % (time() - t))

np.random.seed(0)
x, x_test, y, y_test = train_test_split(x, y, train_size=0.8, random_state=1)
images = x.reshape(-1, 28, 28)
images_test = x_test.reshape(-1, 28, 28)
print(x.shape, x_test.shape)

matplotlib.rcParams['font.sans-serif'] = [u'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(15, 9), facecolor='w')
for index, image in enumerate(images[:16]):
    plt.subplot(4, 8, index + 1)
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title(u'训练图片: %i' % y[index])
for index, image in enumerate(images_test_result[:16]):
    plt.subplot(4, 8, index + 17)
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title(u'测试图片')
plt.tight_layout()
plt.show()

# 05SVM
if classifier_type == '05SVM':
    # params = {'C':np.logspace(1, 4, 4, base=10), 'gamma':np.logspace(-10, -2, 9, base=10)}
    # clf = svm.SVC(kernel='rbf')
    # model = GridSearchCV(clf, param_grid=params, cv=3)
    model = svm.SVC(C=1000, kernel='rbf', gamma=1e-10)
    print('SVM开始训练...')
    t = time()
    model.fit(x, y)
    t = time() - t
    print('SVM训练结束，耗时%d分钟%.3f秒' % (int(t/60), t - 60*int(t/60)))
    # print '最优分类器：', model.best_estimator_
    # print '最优参数：\t', model.best_params_
    # print 'model.cv_results_ ='
    # pprint(model.cv_results_)
    t = time()
    y_hat = model.predict(x)
    t = time() - t
    print('SVM训练集准确率：%.3f%%，耗时%d分钟%.3f秒' % (accuracy_score(y, y_hat)*100, int(t/60), t - 60*int(t/60)))
    t = time()
    y_test_hat = model.predict(x_test)
    t = time() - t
    print('SVM测试集准确率：%.3f%%，耗时%d分钟%.3f秒' % (accuracy_score(y_test, y_test_hat)*100, int(t/60), t - 60*int(t/60)))
    #save_result(model)
elif classifier_type == 'RF':
    rfc = RandomForestClassifier(100, criterion='gini', min_samples_split=2,
                                 min_impurity_split=1e-10, bootstrap=True, oob_score=True)
    print('随机森林开始训练...')
    t = time()
    rfc.fit(x, y)
    t = time() - t
    print('随机森林训练结束，耗时%d分钟%.3f秒' % (int(t/60), t - 60*int(t/60)))
    print('OOB准确率：%.3f%%' % (rfc.oob_score_*100))
    t = time()
    y_hat = rfc.predict(x)
    t = time() - t
    print('随机森林训练集准确率：%.3f%%，预测耗时：%d秒' % (accuracy_score(y, y_hat)*100, t))
    t = time()
    y_test_hat = rfc.predict(x_test)
    t = time() - t
    print('随机森林测试集准确率：%.3f%%，预测耗时：%d秒' % (accuracy_score(y_test, y_test_hat)*100, t))
    #save_result(rfc)

err = (y_test != y_test_hat)
err_images = images_test[err]
err_y_hat = y_test_hat[err]
err_y = y_test[err]
print(err_y_hat)
print(err_y)
plt.figure(figsize=(10, 8), facecolor='w')
for index, image in enumerate(err_images):
    if index >= 12:
        break
    plt.subplot(3, 4, index + 1)
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title(u'错分为：%i，真实值：%i' % (err_y_hat[index], err_y[index]))
plt.suptitle(u'数字图片手写体识别：分类器%s' % classifier_type, fontsize=18)
plt.tight_layout(rect=(0, 0, 1, 0.95))
plt.show()