# 我们还可以指定 changepoints 的位置
# 使用 changepoints 参数手动指定潜在变化点的位置
from fbprophet import Prophet
import pandas as pd
from prophet.plot import add_changepoints_to_plot,plot_plotly

# 读取文件数据
df = pd.read_csv('D://Prophet//official_eg/02_Saturating Forecasts//example_wp_log_R.csv')

m = Prophet(changepoints=['2014-01_Modeling Holidays and Special Events-01_Modeling Holidays and Special Events'])
m.fit(df)

future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)
fig = m.plot(forecast)
fig.savefig('指定changepoints.png')
