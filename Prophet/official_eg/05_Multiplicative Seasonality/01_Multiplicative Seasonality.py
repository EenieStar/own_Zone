# Multiplicative Seasonality / 乘法季节性
import pandas as pd
from fbprophet import Prophet
import matplotlib.pyplot as plt
# 一个航空客流量例子
df = pd.read_csv('D://Prophet//official_eg/05_Multiplicative Seasonality/example_air_passengers.csv')

m = Prophet()
m.fit(df)

future = m.make_future_dataframe(50, freq='MS')
forecast = m.predict(future)
fig = m.plot(forecast)
fig.savefig('可加性季节性在此情况并不适用.png')

# 从图中可知，这个时间序列周期明显，但预测值的拟合程度在刚开始的几个周期内过大。最后又变得很小。
# 在这个时间序列中，季节性并不是Prophet所假设的一个常数可加性因子，而是随着趋势而增长的。
# 这就是乘性季节性
# 通过在输入参数中设置seasonality_mode='multiplicative'， Prophet可以对乘性季节性进行建模
m = Prophet(seasonality_mode='multiplicative')
m.fit(df)

future = m.make_future_dataframe(50, freq='MS')
forecast = m.predict(future)
fig = m.plot(forecast)
fig.savefig('可乘性季节性.png')
# 绘图
# 展示季节性趋势占总趋势的百分比
fig = m.plot(forecast)
fig.savefig('季节性趋势占总趋势的百分比.png')

# 建立 Prophet() 对象时，
# 可以对季节性模式进行优先设置值 seasonality_mode = ‘additive’ / ‘multiplicative’
# 这种情况下，接下来的所有的季节性因素和额外的回归因素都会被改写为相同模式。
# 不过，在后续自定义添加季节性[ (Prophet().add_seasonality() ]和额外回归[ (Prophet().add_regressor() ]时也可以使用mode参数进行自定义。
# 即mode = 'additive'/ 'multiplicative'

# 示例如下
# 首先设置总的季节性模式
m = Prophet(seasonality_mode='multiplicative')

# 自定义额外季节性 'quarterly'
m.add_seasonality('quarterly', period=91.25, fourier_order=8, mode='additive')
# 自定义额外回归 'regressor'
m.add_regressor('regressor', mode='additive')
m.fit(df)
future = m.make_future_dataframe(365)
future['regressor'] = future['ds']
forecast = m.predict(future)

fig = m.plot_components(forecast)
fig.savefig("总季节性与回归模式设置与细分季节性与回归模式覆盖.png")
