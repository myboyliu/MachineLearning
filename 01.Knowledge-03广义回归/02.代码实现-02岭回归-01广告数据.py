import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

# pandas读入
data = pd.read_csv('../Total_Data/data/Advertising.csv')    # TV、Radio、Newspaper、Sales
x = data[['TV','Radio','Newspaper']]
y = data['Sales']

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
# model = Lasso()
model = Ridge()
alpha_can = np.logspace(-3, 2, 10)
np.set_printoptions(suppress=True)
# print('alpha_can = ', alpha_can)
ridge_model = GridSearchCV(model, param_grid={'alpha': alpha_can}, cv=10) # 超参数优化
ridge_model.fit(x_train, y_train)
# print('超参数:', ridge_model.best_params_)

order = y_test.argsort(axis=0)
y_test = y_test.values[order]
x_test = x_test.values[order, :]
y_hat = ridge_model.predict(x_test)
print('GridSearchCV-得分:', ridge_model.score(x_test, y_test))
mse = np.average((y_hat - np.array(y_test)) ** 2)  # Mean Squared Error
rmse = np.sqrt(mse)  # Root Mean Squared Error
print('GridSearchCV-MSE:', mse)
print('GridSearchCV-RMSE:', rmse)

model = RidgeCV(alphas=np.logspace(-3, 2, 50), fit_intercept=False)
model.fit(x_train, y_train)
y_model_hat = model.predict(x_test)
print('RidgeCV-得分:', model.score(x_test, y_test))
mse = np.average((y_model_hat - np.array(y_test)) ** 2)  # Mean Squared Error
rmse = np.sqrt(mse)  # Root Mean Squared Error
print('RidgeCV-MSE:', mse)
print('RidgeCV-RMSE:', rmse)

# 如果使用pipeline，加上PolynomialFeatures,图像会更加平缓，没有这么震荡
pipemodel = Pipeline([('poly', PolynomialFeatures()),
                      ('linear', RidgeCV(alphas=np.logspace(-3, 2, 50), fit_intercept=False))])
pipemodel.fit(x_train, y_train)
y_pipemodel_hat = pipemodel.predict(x_test)
print('PolynomialFeatures-得分:', pipemodel.score(x_test, y_test))
mse = np.average((y_pipemodel_hat - np.array(y_test)) ** 2)  # Mean Squared Error
rmse = np.sqrt(mse)  # Root Mean Squared Error
print('PolynomialFeatures-MSE:', mse)
print('PolynomialFeatures-RMSE:', rmse)

t = np.arange(len(x_test))
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False
plt.figure(facecolor='w')
plt.plot(t, y_test, 'r-', linewidth=2, label=u'真实数据')
plt.plot(t, y_hat, 'g-', linewidth=2, label=u'GridSearchCV预测数据')
plt.plot(t, y_model_hat, 'b-', linewidth=2, label=u'RidgeCV预测数据')
plt.plot(t, y_pipemodel_hat, 'y-', linewidth=2, label=u'PolynomialFeatures预测数据')
plt.title(u'线性回归预测销量', fontsize=18)
plt.legend(loc='lower right')
plt.grid()
plt.show()