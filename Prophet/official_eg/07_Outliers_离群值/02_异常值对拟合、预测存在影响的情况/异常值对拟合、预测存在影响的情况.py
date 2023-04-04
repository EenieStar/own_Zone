import pandas as pd
from fbprophet import Prophet

df = pd.read_csv('example_wp_log_R_outliers2.csv')

m = Prophet()
m.fit(df)
future = m.make_future_dataframe(periods=1096)
forecast = m.predict(future)
fig = m.plot(forecast)
fig.savefig('异常值对拟合、预测存在影响')
# 在这里，2015年6月的一组极端异常值打乱了季节性估计，因此它们的影响将永远影响到未来。正确的做法是删除它们
df.loc[(df['ds'] > '2015-06-01') & (df['ds'] < '2015-06-30'), 'y'] = None
model = Prophet().fit(df)
fig = model.plot(model.predict(future))
fig.savefig('删除异常值后的拟合、预测')
