# 模拟节假日和特殊事件
# Modeling Holidays and Special Events
import pandas as pd
from fbprophet import Prophet
from fbprophet.plot import plot_forecast_component
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

# 检查创建的dataframe
#print(playoffs)
#print('\n')
#print(superbowls)
# 将两种事件归为节假日
holidays = pd.concat((playoffs, superbowls))
# 一旦创建了dataframe，假日影响就会通过假日参数 (holidays) 将创建好的dataframe传入预测中。
m = Prophet(holidays=holidays)
m.fit(df)
future = m.make_future_dataframe(periods=365)

forecast = m.predict(future)
# forecast（dataframe）中观察节假日影响 [仅仅观看最后十行数据]
holiday_effect = forecast[(forecast['playoff'] + forecast['superbowl']).abs() > 0][
                         ['ds', 'playoff', 'superbowl']][-10:]
# print(holiday_effect)


# 画出单个组件图
# 可以使用plot_forecast_component函数(从prophet导入)绘制单个假日。
plot_forecast_component(m, forecast, 'superbowl')
plot_forecast_component(m, forecast, 'playoff')
plt.show()





