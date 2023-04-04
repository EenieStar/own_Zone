# Case Study - Pedestrian Activity____案例研究--行人活动
import pandas as pd
from fbprophet import Prophet
from matplotlib import pyplot as plt

# 使用墨尔本行人数据
df = pd.read_csv('D:/Prophet/official_eg/Handling Shocks/example_pedestrians_covid.csv')
# check file
print(df.shape, '\n', df.head())
# 作图分析
df.set_index('ds').plot()
plt.show()
# 可见数据在2020-1急速下降并保持新的趋势
'''以默认状态训练模型'''
m = Prophet(growth='')
m = m.fit(df)
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)

m.plot(forecast)
plt.axhline(y=0, color='red')
plt.title('默认 Prophet')

# python matpltlib 显示中文
plt.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体（解决中文无法显示的问题）
plt.rcParams['axes.unicode_minus'] = False # 解决保存图像时负号“-”显示方块的问题

#-*-coding:gb2312-*-

m.plot_components(forecast)

plt.show()
"""
在默认的Prophet中，认为像这些大的峰值在未来是有可能出现的
尽管在我们的预测范围内（在这种情况下是1年），我们现实中不会看到同样幅度的东西。
"""
