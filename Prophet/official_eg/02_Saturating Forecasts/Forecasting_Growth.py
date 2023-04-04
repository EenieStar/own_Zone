from fbprophet import Prophet
import pandas as pd

# 读取文件数据
df = pd.read_csv('D://Prophet//official_eg/02_Saturating Forecasts//example_wp_log_R.csv')
# 检查数据
# print(df.columns)
print(df.values)
# 为每一行指定一个cap
# 需要注意的是，cap必须为了dataframe中的每一行指定，而且它不一定是数值型常量。
df['cap'] = 8.5
# 然后像之前一样拟合模型，只是传入了一个指定 logistic growth 的额外参数
m = Prophet(growth='logistic')
# fit方法拟合
m.fit(df)
# 我们将 cap 保持在历史值不变，并预测未来5年的情况
#future = m.make_future_dataframe(periods=1826)
#future['cap'] = 8.5
#fcst = m.predict(future)
#fig = m.plot(fcst)
#fig.savefig('预测结果')
#fig.show()

# 要使用具有饱和最小值的logistic预测增长趋势，在指定最小饱和度时(使用关键词 ‘floor’ )还必须指定最大容量(使用关键词 ‘cap’ )。
df['y'] = 10-df['y']
df['cap'] = 6
df['floor'] = 1.5
# 建立预测的dataframe
future = m.make_future_dataframe(periods=1826)

future['cap'] = 6
future['floor'] = 1.5
m = Prophet(growth='logistic')
m.fit(df)
fcst = m.predict(future)
fig = m.plot(fcst)
fig.savefig('预测结果(最小饱和)')

