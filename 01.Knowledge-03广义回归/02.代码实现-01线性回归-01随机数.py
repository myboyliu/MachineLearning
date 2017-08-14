import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False
N = 10
N1 = 6
x = np.linspace(-3, 3, N)
noise = np.random.normal(0, 0.02, x.shape)
y = np.sin(4 * x) + x + noise

X = x[:, np.newaxis]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                    random_state=42)
regressor = LinearRegression()
regressor.fit(X_train, y_train)
print('Weight coefficients:', regressor.coef_) # 最小二乘法的系数\n",
print('y-axis intercept:', regressor.intercept_) # 最小二乘法的截距\n",

plt.plot(X_train, y_train, 'o')

y_pred_train = regressor.predict(X_train)
print('Score:', regressor.score(X_test, y_test))

model = Pipeline([('poly', PolynomialFeatures(degree=N1)),
                  ('linear', LinearRegression(fit_intercept=False))])
# model.set_params(poly__degree=N1)
model.fit(X_train, y_train)
x_hat = np.linspace(X_train.min(), X_train.max(), num=N)
x_hat.shape = -1, 1
y_hat = model.predict(x_hat)

plt.plot(X_train, y_train, 'o', label="data")
plt.plot(X_train, y_pred_train, 'g-', linewidth=2, label="1阶")
plt.plot(x_hat, y_hat, 'r-', linewidth=2, label=u"%d阶" % N1)

# plt.plot(x_hat, y_hat, color=clrs[i], lw=line_width[i], alpha=0.75, label=label, zorder=z)

plt.legend(loc='best')
plt.show()