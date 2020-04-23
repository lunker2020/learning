import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt

import kerascourse as ke
import pandas

###数据处理专题

###1.如何把一个数组分割
a = np.arange(0,10000,5)
b = len(a)/2
b=int(b)
afore=a[:b]
alast=a[b:]
import matplotlib as plot
plot(afore)

x = ke.probability.entropy(88)
print(x)

