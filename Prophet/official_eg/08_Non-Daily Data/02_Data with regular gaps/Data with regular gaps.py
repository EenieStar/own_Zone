from fbprophet import Prophet
import pandas as pd
from prophet.plot import plot_plotly, plot_components_plotly

# 对于一些比 日 小的时间单位
# prophet依旧可以处理
# 这里使用 Yosemite 每日温度（数据时间单位为 5 min）
df = pd.read_csv('/official_eg/08_Non-Daily Data//example_yosemite_temps.csv')
print(df.head())
# 假设数据集只有 12AM--6AM 的观测值
df2 = df.copy()                         # 创建 df2 并复制 df
df2['ds'] = pd.to_datetime(df2['ds'])   # 将 df2 中的 ds列 转化为datetime格式

# 更新 df2 使得 df2 只有 12AM--6AM 的观测值
# 将df切片并赋值给 df2
df2 = df2[df2['ds'].dt.hour < 6]

# 预测
m = Prophet()
m.fit(df2)

future = m.make_future_dataframe(periods=300, freq='H')
forecast = m.predict(future)

fig = plot_plotly(m, forecast)
fig01 = plot_components_plotly(m, forecast)

fig.show()
fig01.show()

# 这样的预测结果会很差
# 在这里，我们需要先在使用 m.make_future_dataframe() 方法将 future 限制在12AM-6AM
future2 = future.copy()
future2 = future2[future2['ds'].dt.hour < 6]
forecast2 = m.predict(future2)
fig02 = plot_plotly(m, forecast2)
fig03 = plot_components_plotly(m, forecast2)

fig02.show()
fig03.show()
