import pandas as pd

path = 'iris.data'  # 数据文件路径
data = pd.read_csv(path, header=None)
x_prime = data.values[:, :4]
# print(x_prime)

y =data.values[:,-1:]
YY = []
for index, vec in enumerate(y):
    YY.append(vec[0])

y = pd.Categorical(YY).codes
print(y)