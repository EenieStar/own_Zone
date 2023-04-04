import pandas as pd

df = pd.DataFrame([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]],
                  columns=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'])
list1 = ['H', 'I', 'J', 'K', 'L', 'M', 'N']

print(df)
# ~ 符号对于布尔值来说是指 取反
# df是一个两行n列的列表
# 我们希望删除df中在list1中存在的列名的列
# 但是我们发现df和list1中相同的只有'H', 'I', 'J'
print(df.columns.isin(list1))
print(~df.columns.isin(list1))
# 即我们需要在操作前首先排除list1中的'K', 'L', 'M', 'N'
print(df.columns[~df.columns.isin(list1)])# 此操作只显示bool值为TRUE的列
new_df = df[df.columns[~df.columns.isin(list1)]]
print(new_df)
