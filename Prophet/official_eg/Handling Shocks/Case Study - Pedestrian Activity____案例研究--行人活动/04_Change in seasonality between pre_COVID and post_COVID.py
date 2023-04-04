# Changes in seasonality between pre- and post-COVID
# 季节性成分图中显示，
# 在一周中，星期五相对于其他日子，活动达到了高点。
# 如果我们不能确定这样的季节性是否在封控后还会出现，
# 我们就需要在模型中加入季节性。

import pandas as pd
from fbprophet import Prophet
import matplotlib.pyplot as plt

# 使用墨尔本行人数据
df = pd.read_csv('D:/Prophet/official_eg/Handling Shocks/example_pedestrians_covid.csv')
# 定义季节选项——封锁日期
lockdowns = pd.DataFrame([
    {'holiday': 'lockdown_1', 'ds': '2020-03-21', 'lower_window': 0, 'ds_upper': '2020-06-06'},
    {'holiday': 'lockdown_2', 'ds': '2020-07-09', 'lower_window': 0, 'ds_upper': '2020-10-27'},
    {'holiday': 'lockdown_3', 'ds': '2021-02-13', 'lower_window': 0, 'ds_upper': '2021-02-17'},
    {'holiday': 'lockdown_4', 'ds': '2021-05-28', 'lower_window': 0, 'ds_upper': '2021-06-10'},
])
# 整理封控日期，计算影响范围
# 形成最终的季节选项
for t_col in ['ds', 'ds_upper']:
    lockdowns[t_col] = pd.to_datetime(lockdowns[t_col])
lockdowns['upper_window'] = (lockdowns['ds_upper'] - lockdowns['ds']).dt.days
print(lockdowns)


# 首先，我们在历史数据框架中定义了布尔列，以标记 "pre_covid "和 "post_covid "时期：

df['pre_covid'] = pd.to_datetime(df['ds']) < pd.to_datetime('2020-03-21')
df['post_covid'] = ~df['pre_covid']
# check ordered file
print(df.head())

# 关闭默认季节性
m = Prophet(weekly_seasonality=False, holidays=lockdowns)
# 添加pre_COVID和post_COVID周季节性
m.add_seasonality(name='weekly_pre_covid', period=7, fourier_order=3, condition_name='pre_covid')
m.add_seasonality(name='weekly_post_covid', period=7, fourier_order=3, condition_name='post_covid')

m = m.fit(df)

future = m.make_future_dataframe(periods=366)
future['pre_covid'] = pd.to_datetime(future['ds']) < pd.to_datetime('2020-03-21')
future['post_covid'] = ~future['pre_covid']

forecast = m.predict(future)
# 画图
m.plot(forecast)
plt.axhline(y=0, color='red')
plt.title('将封控作为节假日 & 加入新的周季节性')
m.plot_components(forecast)

# python matpltlib 显示中文
plt.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体（解决中文无法显示的问题）
plt.rcParams['axes.unicode_minus'] = False # 解决保存图像时负号“-”显示方块的问题
plt.show()
