# Hyperparameter tuning / 超参数的调整
交叉验证的目的就是调整模型的超参数，可以调整的参数有且不限于changepoint_prior_scale和seasonality_prior_scale

示例中网格搜索几个不同参数，在 **cutoffs** 上使用并行化

这里的参数是根据30天范围（horizon）内平均的 **RMSE** 来评估的，但是不同的性能评价指标指标可能适合不同的问题。