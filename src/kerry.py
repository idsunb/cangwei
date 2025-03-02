#根据概率和赔率计算仓位,推导凯利公式




import sympy as sp

# 定义符号变量
f, w, l, p,b = sp.symbols('f w l p b')

# 定义对数期望值函数
L = p * sp.log(1 + f * w) + (1 - p) * sp.log(1 - f*l)

# 对函数 L(f) 求导
dL_df = sp.diff(L, f)

# 输出导数
print(dL_df)

# 求解方程 dL_df = 0
f_star = sp.solve(dL_df, f)[0]


# 输出最优解
print(f_star)

# 假定 b= w/l
# 假定 l=1
f_star = f_star.subs(w, b*l)
f_star = f_star.subs(l, 1)


print(f_star)

# 输出分数形式
print(f_star.evalf())


