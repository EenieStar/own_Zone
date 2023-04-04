from fbprophet import Prophet
import pandas as pd
from prophet.plot import add_changepoints_to_plot, plot_plotly
import matplotlib.pyplot as plt

# 读取文件数据
df = pd.read_csv('D://Prophet//official_eg/02_Saturating Forecasts//example_wp_log_R.csv')
# 检查数据
# print(df.columns)
# print(df.values)

# 训练拟合
m = Prophet()
m.fit(df)

# 预测
# 建立空白dataframe
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)

# 可变点 changepoints 位置可视化
fig = plot_plotly(m, forecast)
fig_changepoints = m.plot(forecast)
a = add_changepoints_to_plot(fig_changepoints.gca(), m, forecast)
plt.show()










