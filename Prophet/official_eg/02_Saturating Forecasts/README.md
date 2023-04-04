By default, Prophet uses a linear model for its forecast.
默认情况下，Prophet使用线性模型(linear)对增长进行预测。
When forecasting growth, there is usually some maximum achievable point: total market size, total population size, etc.
在预测增长时，（使用logistic ）通常是预测一些最大可实现点:总市场规模，总人口规模等。
This is called the carrying capacity, and the forecast should saturate at this point.
这种可实现点被称为承载力，预测结果在这一点上应该达到了最大限度（饱和）。

预测趋势时，分为预测最大趋势和最小趋势，即数据的预测可能会走高或走低
预测最大趋势时使用 cap 关键词进行指定最大饱和值怕(非常量)
要使用具有饱和最小值的logistic预测增长趋势，在指定最小饱和度时(使用关键词 ‘floor’ )还必须指定最大容量(使用关键词 ‘cap’ )。
logistic函数有一个隐式的最小值0，并且在0处达到饱和




## 例子使用维基百科上“R语言”页面的访问量日志作为数据进行实验
