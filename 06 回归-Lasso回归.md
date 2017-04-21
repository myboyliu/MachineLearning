Lasso回归
=
岭回归如果特征很多，那么每个特征都会涉及到一个系数，那么就太复杂了，因为每个系数都会影响最后的结果
lasso可以保证有些特征的系数为0，有些不为0，这样就起到了一个筛选参数的作用

<img src="http://latex.codecogs.com/svg.latex?L(\Theta)=argmin_{\Theta}\{\sum_{i=1}^m(y_i-\sum_{j=1}^px_{ij}\Theta_j)^2+K\sum_{j=1}^p|\Theta_j|^d\}" style="border>
一般形式，如果d=1，就是lasso回归，d=2就是岭回归

其中的约束是|\Theta_j|^d < t

