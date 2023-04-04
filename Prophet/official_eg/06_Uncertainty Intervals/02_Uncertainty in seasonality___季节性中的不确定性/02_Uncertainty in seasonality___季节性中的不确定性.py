# 季节不确定性
# 使用 mcmc.samples 参数(默认为0)进行完整的贝叶斯抽样，以获取季节性的不确定性
# 使用 Quick_start 中Peyton'Manning的前六个月的数据作为例子
import pandas as pd
from fbprophet import Prophet
# 读取数据
df = pd.read_csv("D://Prophet//official_eg/01_Quick_start//example_wp_log_peyton_manning.csv")
# 建立 Prophet 对象，使用mcmc.samples参数
m = Prophet(mcmc_samples=300)
m.fit(df)
# 在fit()过程中可以将show_progress设置为False(默认为True),以此使其不显示训练过程。
future = m.make_future_dataframe(180)
forecast = m.predict(future)

fig = m.plot_components(forecast)



