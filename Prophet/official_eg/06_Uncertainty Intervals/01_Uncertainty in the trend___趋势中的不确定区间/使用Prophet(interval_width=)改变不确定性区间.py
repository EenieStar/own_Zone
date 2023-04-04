# 使用 Prophet(interval_width改变预测的不确定性区间的宽度)
# interval_width的值越小，不确定性区间的宽度越小

import pandas as pd
from fbprophet import Prophet
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_csv("D://Prophet//official_eg/01_Quick_start//example_wp_log_peyton_manning.csv")

# fit方法
m = Prophet(interval_width=0.1)
m.fit(df)

# 使用Prophet.make_future_dataframe建立新的空dataframe帮助预测
future = m.make_future_dataframe(periods=365)
# print(future.tail())

# 使用 predict 预测
forecast = m.predict(future)
print('预测结果的后五行：')
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

# 画图方法1
# 调用Prophet.plot方法画出predict预测
fig1 = m.plot(forecast)
# Prophet.plot_components()方法会画出一些细节图
# 一般包含序列的趋势、年季节性、周季节性
fig2 = m.plot_components(forecast)
# 保存 #
fig1.savefig('改变不确定性区间——预测结果（interval_width=0.1）.png')
fig2.savefig('改变不确定性区间——结果细节.png')

# 画出的图中 #
# 由Prophet.plot()画出的图fig1中的点代表原始数据值 #
# 带状图表示拟合后的范围 #
# 由Prophet.plot_components()画出的图fig2中的三幅图分别是 #
# 拟合后的整体趋势，以年为时间节点的年季节性，以周为时间单位的周季节性 #

# 画图方法2
# 使用 plot_plotly 绘图方法可以先预览，后保存，比上述的绘图方法简单
# from prophet.plot import plot_plotly, plot_components_plotly

# 预测结果图
# fig1 = plot_plotly(m, forecast)
# 预测结果细节组件图
# fig2 = plot_components_plotly(m, forecast)

# fig2.show()
# plt.show()
