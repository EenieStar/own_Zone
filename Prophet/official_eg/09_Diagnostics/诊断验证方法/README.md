#  交叉验证
**时间序列交叉验证功能，用历史数据衡量预测误差**

可以使用Cross validation函数正对一系列历史日期进行交叉验证
指定指定预测范围（horizon），然后可以选择初始训练期（initial）的大小和截止日期的间隔（period）
Cross validation 函数的输出是一个dataframe, 包含ds, yhat, yhat_lower, yhat_upper, y, cutoff
