import pandas as pd
from fbprophet import Prophet
import matplotlib.pyplot as plt
from prophet.plot import plot_yearly
# 傅里叶级数和季节性
# 它决定了季节性变化的快慢。
# 读取数据
df = pd.read_csv("D://Prophet//official_eg/01_Quick_start//example_wp_log_peyton_manning.csv")

m = Prophet()
m.fit(df)
# 显示默认的变化率
plot_yearly(m)

# 默认值通常是合适的，但当季节性需要适应更高频率的变化时，可以增加默认值，使得最终得到的曲线陡峭。
# 数值越小越平滑，越大越陡峭
# 下面我们将值增加到20
"""N个傅里叶项对应于用于建模周期的2N个变量"""
m = Prophet(yearly_seasonality=20).fit(df)
plot_yearly(m)
plt.show()
