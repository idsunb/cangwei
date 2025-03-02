#测试矩阵的乘法

# import numpy as np
# import matplotlib.pyplot as plt

# # 定义参数
# #{p11: 0.6, p12: 0.4}
# #{p21: 0.3, p22: 0.6,p23:0.1}
# #{p31: 0.1, p32: 0.1,p33:0.8}

# #相乘的矩阵
# matrixp1 = np.array([0.6, 0.4])
# matrixp2 = np.array([0.3, 0.6,0.1])
# matrixp3 = np.array([0.1, 0.1,0.8])


# #三个矩阵相乘
# matrixp = np.dot(matrixp1,matrixp2)
# # matrixp = np.dot(matrixp,matrixp3)
# print('matrixp:',matrixp)
import numpy as np


# 定义矩阵
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# 矩阵乘法
C = np.matmul(A, B)
print(C)







# # 定义矩阵
# matrixp1 = np.array([[0.6, 0.4]])
# matrixp2 = np.array([[0.3, 0.6, 0.1], [0.2, 0.5, 0.3]])
# matrixp3 = np.array([[0.1, 0.1, 0.8], [0.2, 0.3, 0.5], [0.4, 0.4, 0.2]])

# # 使用 numpy.dot
# result_dot = np.dot(matrixp1, matrixp2)
# print('Result using np.dot:', result_dot)

# # 使用 numpy.matmul
# result_matmul = np.matmul(result_dot, matrixp3)
# print('Result using np.matmul:', result_matmul)

# # 使用 @ 运算符
# result_at = result_dot @ matrixp3
# print('Result using @ operator:', result_at)



# # 定义两组概率
# probabilities1 = np.array([0.6, 0.4])
# probabilities2 = np.array([0.3, 0.6, 0.1])

# # 计算外积（Kronecker 积）
# combinations = np.outer(probabilities1, probabilities2)

# print('Combinations matrix:')
# print(combinations)