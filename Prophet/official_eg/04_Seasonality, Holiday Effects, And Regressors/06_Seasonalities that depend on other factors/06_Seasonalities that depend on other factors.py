# 使用条件季节性来构建单独的季节和淡季周季节性

import pandas as pd
from fbprophet import Prophet
from matplotlib import pyplot as plt

# 读取数据
df = pd.read_csv("D://Prophet//official_eg/01_Quick_start//example_wp_log_peyton_manning.csv")


# 定义区分函数
def is_nfl_season(ds):
    """添加一个bool值列"""
    """区分每个日期是 NFL 赛季内还是赛季外"""
    """NFL赛季是每年的 8 月到第二年的 2 月"""
    date = pd.to_datetime(ds)
    return date.month > 8 or date.month < 2


df['on_season'] = df['ds'].apply(is_nfl_season)     # 旺季
df['off_season'] = ~df['ds'].apply(is_nfl_season)   # 淡季

print(df['on_season'])

# 然后，我们禁用内置的每周季节性，并将其替换为两个指定这些列作为条件的每周季节性。
# 我们禁用内置的每周季节性
m = Prophet(weekly_seasonality=False)
# 并将其替换为两个指定这些列作为条件的每周季节性。
m.add_seasonality(name='weekly_on_season', period=7, fourier_order=3, condition_name='on_season')
m.add_seasonality(name='weekly_off_season', period=7, fourier_order=3, condition_name='off_season')

m.fit(df)
future = m.make_future_dataframe(365)
# 我们还必须将该列添加到我们正在进行预测的未来数据框中。
future['on_season'] = future['ds'].apply(is_nfl_season)
future['off_season'] = ~future['ds'].apply(is_nfl_season)


forecast = m.predict(future)

fig = m.plot_components(forecast)
plt.show()
