from sklearn.linear_model import Perceptron

clf =Perceptron(n_iter=100, eta0=0.05, random_state=None)
clf.fit([[0,0],[0,1],[1,0],[1,1]],[0,0,0,1])
print(clf.predict([1,1]))