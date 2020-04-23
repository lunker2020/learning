from __future__ import division
from sympy import *

import sympy as py

###符号运算用于推导数学公式

x = py.Symbol('x')
y = py.Symbol('y')
z = py.symbol('z')
f1 = 2 * x - y + z - 10
f2 = 3 * x + 2 * y - z - 16
f3 = x + 6 * y - z - 28
print(py.solve([f1, f2, f3]))
result = py.solve([f1,f2,f3])
print(float(result[x]))


