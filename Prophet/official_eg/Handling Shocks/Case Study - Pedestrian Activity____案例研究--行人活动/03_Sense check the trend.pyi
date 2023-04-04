import pandas as pd
from fbprophet import Prophet
from matplotlib import pyplot as plt
from prophet.plot import add_changepoints_to_plot

df = pd.read_csv('D:/Prophet/official_eg/Handling Shocks/example_pedestrians_covid.csv')

m = Prophet()
m = m.fit(df)
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)

fig = m.plot(forecast)
a = add_changepoints_to_plot(fig.gca(), m, forecast)
# 自定义节假日 lockdowns
lockdowns = pd.DataFrame([
    {'holiday': 'lockdown_1', 'ds': '2020-03-21', 'lower_window': 0, 'ds_upper': '2020-06-06'},
    {'holiday': 'lockdown_2', 'ds': '2020-07-09', 'lower_window': 0, 'ds_upper': '2020-10-27'},
    {'holiday': 'lockdown_3', 'ds': '2021-02-13', 'lower_window': 0, 'ds_upper': '2021-02-17'},
    {'holiday': 'lockdown_4', 'ds': '2021-05-28', 'lower_window': 0, 'ds_upper': '2021-06-10'},
])
# 整理封控日期，计算影响范围
for t_col in ['ds', 'ds_upper']:
    lockdowns[t_col] = pd.to_datetime(lockdowns[t_col])
lockdowns['upper_window'] = (lockdowns['ds_upper'] - lockdowns['ds']).dt.days
print(lockdowns)

m1_changepoints = (
    # 10 potential changepoints in 2.5 years
    pd.date_range('2017-06-02', '2020-01-01', periods=10).date.tolist() +
    # 15 potential changepoints in 1 year 2 months
    pd.date_range('2020-02-01', '2021-04-01', periods=15).date.tolist()
 )
# 默认changepoint_prior_scale 是0.05， 相比较而言，1.0更加灵活
m1 = Prophet(holidays=lockdowns, changepoints=m1_changepoints, changepoint_prior_scale=1.0)
m1.fit(df)
forecast1 = m1.predict(future)
fig1 = m1.plot(forecast1)
# 显示可变点
a1 = add_changepoints_to_plot(fig1.gca(), m1, forecast1)
plt.show()
