# 如果趋势变化是过拟合(灵活性太大)或欠拟合(灵活性不够)，
# 则可以使用输入参数changepoint_prior_scale调整稀疏先验的强度。
# 默认情况下，这个参数为0.05
# changepoint_prior_scale 越大会使得趋势变化更加灵活
# 以下是更改 changepoint_prior_scale 后的预测代码
import pandas as pd
from fbprophet import Prophet
from prophet.plot import plot_plotly

# 读取文件数据
df = pd.read_csv('D://Prophet//official_eg/02_Saturating Forecasts//example_wp_log_R.csv')

m = Prophet(changepoint_prior_scale=0.5)
m.fit(df)
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)
fig = plot_plotly(m, fcst=forecast)
fig.show()
