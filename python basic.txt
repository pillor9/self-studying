# python 相关

## containers

 ## 列表 ：

 ### 循环：
 ```
 animals = ['cat', 'dog', 'monkey']

 for idx, animal in enumerate(animals): 
 * 注意枚举函数
    print('#{}: {}'.format(idx + 1, animal))
```
 ### 构造：
 ```
 1. nums = [0, 1, 2, 3, 4]
 squares = []
 for x in nums:
    squares.append(x ** 2)
 print(squares)
 2. nums = [0, 1, 2, 3, 4]
 even_squares = [x ** 2 for x in nums if x % 2 == 0]
 * 可包含条件
 print(even_squares)
```
 ## 字典 ：
 * 强调对应关系
 ### 字典范例：
 ```
  d = {'cat': 'cute', 'dog': 'furry'}  # Create a new dictionary with some data
 print(d['cat'])       # Get an entry from a dictionary; prints "cute"
 print('cat' in d)     # Check if a dictionary has a given key; prints "True"
 d['fish'] = 'wet'    # Set an entry in a dictionary
 print(d['fish'])      # Prints "wet"
 print(d.get('monkey', 'N/A'))  # Get an element with a default; prints "N/A"
 print(d.get('fish', 'N/A'))    # Get an element with a default; prints "wet"
```
  ### 字典推导式：
  ```
 nums = [0, 1, 2, 3, 4]
 even_num_to_square = {x: x ** 2 for x in nums if x % 2 == 0}
 print(even_num_to_square)
```
 ## 集合 ：
 * 类似列表，但是具有无序性。

 ### 循环：
 ```
 animals = {'cat', 'dog', 'fish'}
 for idx, animal in enumerate(animals):
    print('#{}: {}'.format(idx + 1, animal))
```
 * 这里具有无序特点
 ### 构造：
 ```
 from math import sqrt
 pi={int(sqrt(x)) for x in range(30)}
 print(pi)
```
 ## 元组：

 ### 概念：
  元组是一个（不可变的）有序列表。元组在很多方面类似于列表;最重要的区别之一是元组可以用作字典中的键和集合的元素，而列表不能。这是一个微不足道的例子：
 ```
 d = {(x, x + 1): x for x in range(10)}  # Create a dictionary with tuple keys
 t = (5, 6)       # Create a tuple
 print(type(t))
 print(d[t])       
 print(d[(1, 2)])
 ```
 * 注意：不能修改。

## 函数：
 利用def构建
 ```
 def sign(x):
    if x > 0:
        return 'positive'
    elif x < 0:
        return 'negative'
    else:
        return 'zero'

 for x in [-1, 0, 1]:
    print(sign(x))
```
## Class:
 * 面向对象编程核心概念
```
 class Greeter:

    # Constructor
    def __init__(self, name):
        self.name = name  # Create an instance variable

    # Instance method
    def greet(self, loud=False):
        if loud:
          print('HELLO, {}'.format(self.name.upper()))
        else:
          print('Hello, {}!'.format(self.name))

 g = Greeter('Fred')  # Construct an instance of the Greeter class
 g.greet()            # Call an instance method; prints "Hello, Fred"
 g.greet(loud=True)   # Call an instance method; prints "HELLO, FRED!"
```
## Numpy:
 ### 概念：
  是python中科学计算的核心库。它提供了一个高性能的多维数组对象，以及用于处理这些数组的工具。
   import numpy as np

 ### Arrays
 numpy数组

 ### 数组索引

  切片：与 Python 列表类似，可以对 numpy 数组进行切片。由于数组可能是多维的，因此必须为数组的每个维度指定一个切片：

  * 数组的切片是对相同数据的视图，因此修改它将修改原始数组。

  * 将整数索引与切片混合会产生较低秩的数组，而仅使用切片会产生与原始数组具有相同秩的数组：（维数降低）

  * 整数数组索引：当您使用切片索引到 numpy 数组时，生成的数组视图将始终是原始数组的子数组。相比之下，整数数组索引允许您使用另一个数组中的数据构建任意数组。
  ### 计算
   np.dot()函数用来计算内积，或者矩阵相乘。

 ### Broadcasting 
 demo:
 #### 一般方法：
```
 # We will add the vector v to each row of the matrix x,
 # storing the result in the matrix y
 x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
 v = np.array([1, 0, 1])
 y = np.empty_like(x)   # Create an empty matrix with the same shape as x

 # Add the vector v to each row of the matrix x with an explicit loop
 for i in range(4):
    y[i, :] = x[i, :] + v

 print(y)
```
 #### Broadcasting 方法： 
```
 import numpy as np

 # We will add the vector v to each row of the matrix x,
 # storing the result in the matrix y
 x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
 v = np.array([1, 0, 1])
 y = x + v  # Add v to each row of x using broadcasting
 print(y)
```
 #### Broadcasting 遵循的规则：
  1. If the arrays do not have the same rank, prepend the shape of the lower rank array with 1s until both shapes have the same length.
  2. The two arrays are said to be compatible in a dimension if they have the same size in the dimension, or if one of the arrays has size 1 in that dimension.
  3. The arrays can be broadcast together if they are compatible in all dimensions.
  4. After broadcasting, each array behaves as if it had shape equal to the elementwise maximum of shapes of the two input arrays.
  5. In any dimension where one array had size 1 and the other array had size greater than 1, the first array behaves as if it were copied along that dimension

 ## Matliotlib
 
 ### 概念：
 Matplotlib 是一个绘图库。在本节中，简要介绍 matplotlib.pyplot 模块，该模块提供了一个类似于 MATLAB 的绘图系统。
```
 import matplotlib.pyplot as plt

 * 通过运行这个特殊的 iPython 命令，我们将内联显示绘图：
 ```
 %matplotlib inline

 ### plot绘图：

 #### demo:

 ``` 
 y_sin = np.sin(x)
 y_cos = np.cos(x)

 # Plot the points using matplotlib
 plt.plot(x, y_sin)
 plt.plot(x, y_cos)
 plt.xlabel('x axis label')
 plt.ylabel('y axis label')
 lt.title('Sine and Cosine')
 plt.legend(['Sine', 'Cosine'])
 ```
 ### subplot:
```
# Compute the x and y coordinates for points on sine and cosine curves
x = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

# Set up a subplot grid that has height 2 and width 1,
# and set the first such subplot as active.
plt.subplot(2, 1, 1)

# Make the first plot
plt.plot(x, y_sin)
plt.title('Sine')

# Set the second subplot as active, and make the second plot.
plt.subplot(2, 1, 2)
plt.plot(x, y_cos)
plt.title('Cosine')

# Show the figure.
plt.show()
```