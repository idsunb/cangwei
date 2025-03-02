#根据概率和赔率计算仓位

#有一个标的的收益为 {p0｜r0，p1|r1, p2|r2, ..., pn|rn}
#pi: 概率
#ri: 收益率  ri =Ri-1
#fi: 不同证券的仓位
#如何下注使得增长最快，
#H=logRg=∑pi*log（∑fi*Ri）




import sympy as sp

# 定义符号变量
f, r1, p, r2 = sp.symbols('f r1 p r2')

# 定义对数期望值函数
L = p * sp.log(1 + f * r1) + (1 - p) * sp.log(1 +f * r2)

# 对函数 L(f) 求导
dL_df = sp.diff(L, f)

# 输出导数
print(dL_df)

# 求解方程 dL_df = 0
f_star = sp.solve(dL_df, f)[0]

# 输出最优解
print(f_star)

# # 将表达式合并为一个分数形式
# expr_together = sp.together(f_star)

# # 输出分数形式
# print(expr_together.evalf())

# p=0.3 r1=0.3 r2=0.05 时 f等于
print(f_star.evalf(subs={p: 0.6, r1: 0.2, r2: -0.05}))


#用蒙特卡洛模拟增长，并做出图形
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# 定义参数


# 定义模拟次数
n = 100

# 定义初始资金
initial_money = 100.0

# 定义资金列表
money_list = [initial_money]

#不进行仓位管理，直接模拟资金的增长

# 进行模拟


f = float(f_star.evalf(subs={p: 0.6, r1: 0.2, r2: -0.15}))
print('f:', f)

r1 = 0.2
r2 = -0.15
p = 0.6

count = 0

for i in range(n):
    # 生成一个随机数
    random_number = np.random.rand()
    # 根据随机数的大小，决定增长率
    if random_number < p:
        #对ranmdom_number小于p的情况，进行计数
        count += 1

        money = money_list[-1] * (1 + r1*f)
        money_list.append(money)
    else:
        money = money_list[-1] * (1 + r2*f)
        money_list.append(money)

print(count)


# 绘制图形
plt.plot(money_list)
plt.xlabel('Number of trades')
plt.ylabel('Money')
# plt.show()

# 显示plot图上的最后一个点的值
plt.text(n, money_list[-1], money_list[-1], ha='center', va='bottom', fontsize=10)
plt.show()









# r1 = 1
# r2 = -0.5
# p = 0.5





# for i in range(n):
#     # 生成一个随机数
#     random_number = np.random.rand()
#     # 根据随机数的大小，决定增长率
#     if random_number < p:
#         money = money_list[-1] * (1 + f * r1)
#         money_list.append(money)
#     else:
#         money = money_list[-1] * (1 + f * r2)
#         money_list.append(money)
    
    


# # 绘制图形
# plt.plot(money_list)
# plt.xlabel('Number of trades')
# plt.ylabel('Money')
# plt.show()
# # 从图中可以看出，资金的增长是不稳定的，有时会增长，有时会下降。这是因为我们在每次模拟中都使用了随机数，而随机数的分布是不确定的。如果我们增加模拟次数，资金的增长会更加平稳。

