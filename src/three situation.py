#根据概率和赔率计算仓位

#有一个标的的收益为 {p0｜r0，p1|r1, p2|r2, ..., pn|rn}
#pi: 概率
#ri: 收益率  ri =Ri-1
#fi: 不同证券的仓位
#如何下注使得增长最快，
#H=logRg=∑pi*log（∑fi*Ri）




import sympy as sp

# 定义符号变量
f, p0,p1,p2,r0,r1,r2= sp.symbols('f p0 p1 p2 r0 r1 r2')

# 定义对数期望值函数
L = p0 * sp.log(1 + f * r0) + p1 * sp.log(1 +f * r1)   + p2 * sp.log(1 +f * r2)

# 对函数 L(f) 求导
dL_df = sp.diff(L, f)

psum = sp.Eq(p0+p1+p2,1)




# 代入参数
dL_df =dL_df.evalf(subs={p0: 0.6, r0: 0.2, p1: 0.39, r1: -0.15, p2: 0.01, r2: -0.5})


# 输出导数
print(dL_df)

# 求解方程 dL_df = 0
f_star = sp.solve(dL_df, f)[0]
# f_star = sp.solve((dL_df,psum), f)


# 输出最优解
print(f_star)

# # 将表达式合并为一个分数形式
# expr_together = sp.together(f_star)

# # 输出分数形式
# print(expr_together.evalf())

# p=0.3 r1=0.3 r2=0.05 时 f等于
# print(f_star.evalf(subs={p: 0.5, r1: 1, r2: -0.5}))