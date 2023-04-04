from fbprophet import Prophet
import pandas as pd
from prophet.plot import plot_plotly, plot_components_plotly

# 对于一些比 日 小的时间单位
# prophet依旧可以处理
# 这里使用 Yosemite 每日温度（数据时间单位为 5 min）
df = pd.read_csv('/official_eg/08_Non-Daily Data/example_yosemite_temps.csv')
print(df.head())
m = Prophet(changepoint_prior_scale=0.01)
m.fit(df)

# freq 参数表示预测的时间粒度为小时
# 我们预测未来300小时的数据
future = m.make_future_dataframe(periods=300, freq='H')
forecast = m.predict(future)
fig = plot_plotly(m, forecast)
fig1 = plot_components_plotly(m, forecast)
fig.show()
fig1.show()
