from fbprophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt


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

m = Prophet(holidays=holidays)

# 使用 Prophet.add_country_holidays() 将某国家节日加入到 holidays 参数中
# 每个国家的假日由 Python 中的假日包提供。
m.add_country_holidays(country_name='CN')
m.fit(df)
# 你可以通过模型的 train_holiday_names
# 输出训练中的 holidays 目录
print(m.train_holiday_names)

future = m.make_future_dataframe(365)
forecast = m.predict(future)
fig = m.plot_components(forecast)
plt.show()
