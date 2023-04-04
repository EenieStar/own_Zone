# Outliers/ 离群值
import pandas as pd
from fbprophet import Prophet
from fbprophet.plot import plot_plotly

# 使用之前的维基百科访问量作为数据集
df = pd.read_csv('example_wp_log_R_outliers1.csv')
print(df)
m = Prophet()
m.fit(df)

future = m.make_future_dataframe(1096)
forecast = m.predict(future)

# 制图发现，存在错误数据
fig = m.plot(forecast)
fig.savefig('维基百科R语言页面访问量预测')

# 处理异常值最好的方法就是删除异常值
# Prophet在你删除异常值后会在训练过程补齐
#
# 删除异常值，设置异常值为NA
# loc[]将行索引'ds'在 2010-01-01 到 2011-01-01 的列'y'的值改为None
df.loc[(df['ds'] > '2010-01-01') & (df['ds'] < '2011-01-01'), 'y'] = None
model = Prophet().fit(df)
fig = model.plot(model.predict(future))
fig.savefig('删除异常值后的预测')


