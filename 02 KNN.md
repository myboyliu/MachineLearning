KNN算法-K Nearest Neighbor(分类算法)
=
#1 原理
数据映射到高维空间中的点、找出K个最近的样本、投票结果
![images](images/01.png)
距离与近似度

#2 如何衡量距离
数学中距离满足三个要求：必须是正数、必须对称、满足三角不等式

#3 常用的距离
##3.1 闵可夫斯基距离 Minkowski
<img src="http://latex.codecogs.com/svg.latex?d_{ij}{(q)}=[\sum_{k=1}^p(x_{ik}-x_{jk})^q]^{\frac{1}{q}},%20q%20%3E%200" style="border:none;">
q越大，差异越大的维度对最终距离影响越大
- 曼哈顿距离:q = 1，城市距离
<img src="http://latex.codecogs.com/svg.latex?d_{ij}=|X_1-X_2|%20+%20|Y_1-Y_2|" style="border:none;">
- 欧氏距离：q = 2，直线距离
<img src="http://latex.codecogs.com/svg.latex?d_{ij}=\sqrt{(x_1-x_2)^2+(y_1-y_2)^2}" style="border:none;">
- 切比雪夫距离:q趋近于无穷大，棋盘距离
<img src="http://latex.codecogs.com/svg.latex?d_{ij}=\lim_{p\to\infty}(\sum_{i=1}^n|x_i-y_i|^p)^{\frac{1}{p}}=max|x_i-y_i|" style="border:none;">

##3.2 马氏距离
考虑数据分布
![images](images/02.png)

#4 具体做法
计算出测试点与样本中的每个点的距离，找到一个导致最小距离的点，它的类别就是测试点的类别，这个距离的计算，一般使用欧氏距离