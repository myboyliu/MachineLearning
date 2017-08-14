import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

np.random.seed(0)
np.set_printoptions(linewidth=1000)
N = 9
x = np.linspace(0, 8, N) + np.random.randn(N)
x = np.sort(x)
y = x**2 - 4*x - 3 + np.random.randn(N)
x.shape = -1, 1
y.shape = -1, 1

model = KNeighborsRegressor()
model.fit(x, y)

x_hat = np.linspace(x.min(), x.max(), num=10)
x_hat.shape = -1, 1
y_hat = model.predict(x_hat)
plt.figure(figsize=(18, 12), facecolor='w')
plt.subplot(2, 2, 1)
plt.plot(x, y, 'ro', ms=10, zorder=N)
s = model.score(x, y)
label = u'$R^2$=%.3f' % (s)
plt.plot(x_hat, y_hat, color='r', lw=2, label=label, alpha=0.75)
plt.legend(loc='upper left')
plt.grid(True)
plt.title('KNN回归', fontsize=18)
plt.xlabel('X', fontsize=16)
plt.ylabel('Y', fontsize=16)

plt.tight_layout(1, rect=(0, 0, 1, 0.95))
plt.show()