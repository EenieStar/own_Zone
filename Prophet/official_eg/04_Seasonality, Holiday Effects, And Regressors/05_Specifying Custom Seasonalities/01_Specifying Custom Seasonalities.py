import pandas as pd
from fbprophet import Prophet


# 读取数据
df = pd.read_csv("D://Prophet//official_eg/01_Quick_start//example_wp_log_peyton_manning.csv")
# 取消默认的周季节性
m = Prophet(weekly_seasonality=False)
# 使用Prophet.add_seasonality() 自定义季节性
m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
# 训练 fit()
m.fit(df)
# 新建 dataframe 并预测
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)

fig = m.plot_components(forecast)
fig.savefig("自定义 monthly 季节性后的组成部分图.png")

