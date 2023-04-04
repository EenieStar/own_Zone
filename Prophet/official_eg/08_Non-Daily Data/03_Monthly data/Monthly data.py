import pandas as pd
from fbprophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly

# 这里我们预测美国未来10年的零售量。
df = pd.read_csv('example_retail_sales.csv')
m = Prophet(seasonality_mode='multiplicative')
m.fit(df)
future = m.make_future_dataframe(periods=3652)
forecast = m.predict(future)

fig = plot_plotly(m, forecast)
fig.show()




