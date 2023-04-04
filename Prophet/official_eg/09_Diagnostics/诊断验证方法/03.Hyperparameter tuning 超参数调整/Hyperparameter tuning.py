# Hyperparameter tuning / 超参数的调整

import itertools
import pandas as pd
from fbprophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric

# 读取数据
df = pd.read_csv("D://Prophet//official_eg/01_Quick_start//example_wp_log_peyton_manning.csv")

param_grid = {
    'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
    'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
}
# 生成所有参数组合.
# itertools.product() 函数生成笛卡尔积，即两两相互配对
all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
# 以上代码为简便写法：列表内涵
# 使用列表内含可以简便写法，提高效率
'''
以下为原写法

all_params = []
for v in itertools.product(*param_grid.values()):
    param = dict(zip(param_grid.keys(), v))
    all_params.append(param)
print(all_params)
'''


rmses = []   # rmses 列表存储每个参数的RMSEs

cutoffs = pd.to_datetime(['2013-02-15', '2013-08-15', '2014-02-15'])

# 使用交叉验证评价所有参数
for params in all_params:
    m = Prophet(**params).fit(df)       # 使用给出的参数们训练模型
    df_cv = cross_validation(m, cutoffs=cutoffs, horizon='30 days')
    df_p = performance_metrics(df_cv, rolling_window=1)
    rmses.append(df_p['rmse'].values[0])

# 找出最优
tuning_results = pd.DataFrame(all_params)
tuning_results['rmse'] = rmses
print(tuning_results)

# 选择最优参数
best_param = tuning_results[tuning_results['rmse'] == min(tuning_results['rmse'])]
print(best_param)
