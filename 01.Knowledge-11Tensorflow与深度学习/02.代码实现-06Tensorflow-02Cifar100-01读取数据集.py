'''
Cifar10的读取是通过Cifar10_input读取的，但是它只能读取cifar10的数据，不能读取cifar100的数据，而且它读出来的数据
都是24*24大小的图片，我们希望找到一个办法可以读取cifar10与cifar100，并且图片大小为32*32的
cifar100有100个类，每个类有600个图像，其中包含训练集500以及测试集100.这100个类又被分组为20个大类
'''