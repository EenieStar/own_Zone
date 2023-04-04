# 并行化的交叉验证
import pandas as pd
import matplotlib.pyplot as plt
from fbprophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric
from dask.distributed import Client

# 读取数据
df = pd.read_csv("D://Prophet//official_eg/01_Quick_start//example_wp_log_peyton_manning.csv")
print(df.iloc[730, :])
m = Prophet()
# manning文档的数据包含了2007-12~2016-01

m.fit(df)
# future = m.make_future_dataframe(periods=365)
# forecast = m.predict(future)
client = Client()       # 连接到集群
df_cv = cross_validation(m, initial='730 days', period='180 days', horizon='365 days', parallel='dask')
# 在设置cutoff时可以使用日期格式将cutoff时间点传递给cross_validation()函数的cutoffs参数

print(df_cv.head())

# 使用performance_metrics()函数计算预测性能的一些统计数据
df_p = performance_metrics(df_cv)
print(df_p.head())
# 使用 plot_cross_validation_metrics 对交叉验证指标进行可视化
fig_validation = plot_cross_validation_metric(df_cv, 'mape', color='red', point_color='black')
plt.show()

