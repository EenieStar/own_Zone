# 模拟节假日和特殊事件
# Modeling Holidays and Special Events
import pandas as pd
from fbprophet import Prophet
import matplotlib.pyplot as plt

df = pd.read_csv("D://Prophet//official_eg/01_Quick_start//example_wp_log_peyton_manning.csv")
# 创建数据dataframe
# 包括佩顿·曼宁所有季后赛出场的日期:

playoffs = pd.DataFrame({

    'holiday': 'playoff',
    'ds': pd.to_datetime(['2008-01-13', '2009-01-03', '2010-01-16',
                          '2010-01-24', '2010-02-07', '2011-01-08',
                          '2013-01-12', '2014-01-12', '2014-01-19',
                          '2014-02-02', '2015-01-11', '2016-01-17',
                          '2016-01-24', '2016-02-07']),

    'lower_window': 0,
    'upper_window': 1,
})

superbowls = pd.DataFrame({
                    'holiday': 'superbowl',
                    'ds': pd.to_datetime(['2010-02-07', '2014-02-02', '2016-02-07']),
                    'lower_window': 0,
                    'upper_window': 1,
})
holidays = pd.concat((playoffs, superbowls))

# 在NFL赛季的周日添加一个额外的效果


def nfl_sunday(ds):
    """在NFL赛季的周日添加一个额外的效果"""
    data = pd.to_datetime(ds)
    if data.weekday() == 6 and (data.month > 8 or data.month < 2):      # 若为八月到第二年的二月之间的周六
        return 1
    else:
        return 0


df['nfl_sunday'] = df['ds'].apply(nfl_sunday)

m = Prophet()
# 加入额外的回归量
m.add_regressor('nfl_sunday')
m.fit(df)
future = m.make_future_dataframe(365)
future['nfl_sunday'] = future['ds'].apply(nfl_sunday)

forecast = m.predict(future)
fig = m.plot_components(forecast)
plt.show()
