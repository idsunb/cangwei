#画出 z =x**2 + y**2 的等高线图像
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 20, 100)
y = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x, y)
Z = X**2 + Y**2

#在等高线上标注数值

plt.figure()
CS = plt.contour(X, Y, Z)
plt.clabel(CS, inline=True, fontsize=10)

plt.show()






# plt.contour(X, Y, Z)
# plt.show()
