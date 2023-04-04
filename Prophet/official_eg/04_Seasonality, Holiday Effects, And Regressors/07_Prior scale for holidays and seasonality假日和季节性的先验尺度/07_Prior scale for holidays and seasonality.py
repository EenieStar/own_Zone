# 假日和季节性的先验尺度
import pandas as pd
from fbprophet import Prophet
from matplotlib import pyplot as plt
from fbprophet.plot import plot_forecast_component

# 读取数据
df = pd.read_csv("D://Prophet//official_eg/01_Quick_start//example_wp_log_peyton_manning.csv")


playoffs = pd.DataFrame({
    'holiday': 'playoff',
    'ds': pd.to_datetime(['2008-01-13', '2009-01-03', '2010-01-16',
                          '2010-01-24', '2010-02-07', '2011-01-08',
                          '2013-01-12', '2014-01-12', '2014-01-19',
                          '2014-02-02', '2015-01-11', '2016-01-17',
                          '2016-01-24', '2016-02-07'
                          ]),
    'lower_window': 0,
    'upper_window': 1,
})
superbowls = pd.DataFrame({
    'holiday': 'superbowl',
    'ds': pd.to_datetime(['2010-02-07', '2014-02-02', '2016-02-07']),
    'lower_window': 0,
    'upper_window': 1,
})
holidays = pd.concat([playoffs, superbowls])
# 使用holidays_prior_scale参数来规定假日的先验尺度
# 以调整训练的拟合度
# 使得其更加平滑。
##m = Prophet(holidays=holidays, holidays_prior_scale=0.05).fit(df)
##future = m.make_future_dataframe(365)
##forecast = m.predict(future)
##screen = forecast[(forecast['playoff'] + forecast['superbowl']).abs() > 0][
##        ['ds', 'playoff', 'superbowl']][-10:]
##print(screen)

# 过在节假日数据框架中包含一个列 prior_scale ，可以为单个节假日单独设置优先比例。
# 单独季节性的先验量表（ prior_scale ）可以作为参数传递给add_seasonality。
m = Prophet(holidays=holidays)
m.add_seasonality(name='weekly', period=7, fourier_order=3, prior_scale=0.001)
m.fit(df)
future = m.make_future_dataframe(365)
forecast = m.predict(future)

fig = m.plot_components(forecast)
fig.savefig('改变单一季节先验改变使得平滑.png')
plot_forecast_component(m, forecast, 'weekly')
plot_forecast_component(m, forecast, 'superbowl')
plot_forecast_component(m, forecast, 'playoff')

plt.show()
