"""Treating COVID-19 lockdowns as a one-off holidays"""
# 了防止大的跌幅和峰值被趋势成分捕获，我们可以把受COVID-19影响的日子当作假期，在未来不会再重复。
# 添加自定义假日在之前的《04_季节性、假日效应和回归因子》有更详细的解释。

import pandas as pd
from fbprophet import Prophet
from matplotlib import pyplot as plt
from prophet.plot import add_changepoints_to_plot

# 使用墨尔本行人数据
df = pd.read_csv('D:/Prophet/official_eg/Handling Shocks/example_pedestrians_covid.csv')
# check file
# print(df.shape, '\n', df.head())
# 作图分析
df.set_index('ds').plot()

# 可见数据在2020-1急速下降并保持新的趋势

lockdowns = pd.DataFrame([
    {'holiday': 'lockdown_1', 'ds': '2020-03-21', 'lower_window': 0, 'ds_upper': '2020-06-06'},
    {'holiday': 'lockdown_2', 'ds': '2020-07-09', 'lower_window': 0, 'ds_upper': '2020-10-27'},
    {'holiday': 'lockdown_3', 'ds': '2021-02-13', 'lower_window': 0, 'ds_upper': '2021-02-17'},
    {'holiday': 'lockdown_4', 'ds': '2021-05-28', 'lower_window': 0, 'ds_upper': '2021-06-10'},
])
# 整理封控日期，计算影响范围
for t_col in ['ds', 'ds_upper']:
    lockdowns[t_col] = pd.to_datetime(lockdowns[t_col])
lockdowns['upper_window'] = (lockdowns['ds_upper'] - lockdowns['ds']).dt.days
print(lockdowns)

"""
我们为每个封锁期都有一个条目，
ds指定了封锁期的开始。
ds_upper不被先知使用，
但它是我们计算upper_window的一个方便方法
"""
"""
upper_window告诉Prophet封锁的时间跨度为封锁开始后的x天。请注意，假期回归是包括上限的。
"""

# 开始预测
m = Prophet(holidays=lockdowns)
m = m.fit(df)
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)

# 可视化

# python matpltlib 显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']    # 指定默认字体（解决中文无法显示的问题）
plt.rcParams['axes.unicode_minus'] = False      # 解决保存图像时负号“-”显示方块的问题

#-*-coding:gb2312-*-

m.plot(forecast)
plt.title('将疫情封控视为只有一次假期')
m.plot_components(forecast)
plt.axhline(y=0, color='red')       # 显示人数为 0 的底线，超越底线则为错误预测
plt.tight_layout()
plt.legend(loc='upper right')

# 显示节假日图像
fig = m.plot(forecast)
a = add_changepoints_to_plot(fig.gca(), m, forecast)

plt.show()
