###数据结构总结：
###提取每一个类型数据的办法：





###编辑ETL





###范数的函数主要看ord的选项，数学建模

import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as la
import sympy as sp

a = np.arange(8)-4
la.norm(a,2)
class linearalgebra():
    def pca(self,x,g_c):
        result=la.norm((x-g_c),2)
        print(type(result))

        result = np.min(result)
        return result
    ###通过数学推导这个只能得到l=1的主成分，并且d是X.T*X的最大特征值的特征向量（奇异值分解就是特征值分解)
    def pca2(self,x):
        ###这个函数返回特征值和特征向量Ax=lamda*x,只需给出x就可以得出A,lamda
        a,lamda=np.linalg.eig(x)
        ###通过求diag去掉矩阵成向量，就可以得到最大的特征值。
        m=max(np.diag(lamda))
        print(m)
        ###求的最大特征值，再求他的对应的特征向量np.linalg.solve,d,np.gradient
        for i in range:
            if(m in lamda[:,i]):
                print(i)
                print(lamda[:,i])
                d = lamda[:,i]
        d=np.linalg.solve(a*x*x.T,m)
        print(d)
        if d.T*d == 1:
            result = max(np.trace(d.T*x*x.T*d))
            print(result)
            return result


class probability():
    ###mean 和 cov是数组,类的第一个函数是构造函数用self
    def multinorm(self,mean,cov):
        ###多变量的正态分布
        x,y = np.random.multivariate_normal(mean, cov, (3, 3)).T
        plt.plot(x, y, 'x')
        plt.axis('equal')
        plt.show()
        for i in x:
            print(i)
        for j in y:
            print(y)
        return x,y
    def entropy(x):
        ###假设p是正态的
        y = np.random.standard_normal((x))
        for i in range(1,len(y)):
            try:
                I_x = -np.log(y[i-1])
            except:
                break
            print(I_x)
            return I_x


class computing():
    def computation(self,x):
        for i in range(0,len(x)):
            np.exp(x[i])
            if i+1 :
                break
            sum=np.exp(x[i])+np.exp(x[i+1])
            return sum
    ###病态指数
    def sickcondi(x):
        a,lamda_x=(np.linalg.eig(x))
        result=np.max(lamda_x)/np.min(lamda_x)
        return result

    ##梯度
    def directionalderivative(u,x,f_x):
        y=u.T*np.norm(f_x)*np.gradient(x,2)
        result=np.min(y)
        return result
    ###hessian and Jacobian,.可能需要使用符号运算～没网先跳过
    def differentialmatrix(x,rules):
        return 0



    ###lagrangian 约束优化
    def lagran(x,lamda,alpha):
        return 0


###enumerate
 ###   def test(self,x,y):

###可以使用assert,如果出现assert error那么就是正确的了


from scipy import *

###OLS  x.transpose() = x.T
###问题出inv（）函数需要多维数组，并且矩阵不可以使用数组，所以出现问题
###解决办法就是看如何转换数组的关系。scipy.linalg.det计算的是矩阵的行列式，numpy.linagle.det计算的是数组的行列式


###简化一下输入必须为3*3 的矩阵或者是N * N的矩阵： ，numpy.pad 可以为矩阵补充0！！！
    def ols(x,y):
        ##assert type(x) == np.matrix
        ##assert type(y) == np.matrix
        ###x=np.asmatrix(x).reshape(3,3)
        ###y=np.asmatrix(y).reshape(3,1)
        ####n行 n列行列式!=0才有逆矩阵  matrix.reshape 配合scipy.linalg.det后者是array.reshape配合np.linalg.det
        ###x=data[:,:-1]
        #assert np.linalg.det(x)!= 0
        ###n行 1列
        ###y=np.array(data[:,-1])
        ###要求满rank ,linalg.matrix_rank()
        xtest = np.dot(x.T,x)
        print(xtest)
        ###if np.linalg.det(xtest)!=0:
        p=np.linalg.inv(xtest)
        result=np.dot(np.dot(p,x.T),y)
        print('结果是',result)
        return result






### MLE
    def mle(x,y):

        return 0



###Possiblity:
    def probability(x):
        return 0








###test
### x =
####jacobian, Hessian， second derivative




###x = [1,2,3,4,5,6,7,7,8,9] testing it


###失败了，因为总是等于0！，用另外一种办法使用范数最小值

### 算法总结




if __name__ == "__main__":
    a = np.arange(0, 2500) * 8 - 450
    a1 = a.reshape(50,50)
    b = np.arange(50, 55) * 8 - 50
    ###np.matrix竟然不被建议，只能用回ndarray
    x = np.matrix(np.arange(81).reshape((9, 9)))
    y = np.matrix(np.arange(9).reshape((3, 3)))

    x1 = np.matrix(np.arange(25),5,5)
    y = np.arange(0,100,5)
    I_x = probability.entropy(8000)
    c=linearalgebra().pca(a, b)
    print(c)
    y = np.reshape(np.arange(2 * 3 * 4), (2, 3, 4))
    offset = sum(y.strides * np.array((1, 1, 1)))