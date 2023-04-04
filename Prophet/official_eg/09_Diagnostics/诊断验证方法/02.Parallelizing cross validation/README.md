# 并行化的交叉验证     
### 需要首先配置并行环境
通过设置指定 **parallel** 关键字，交叉验证也可以在Python中以并行模式运行。支持四种模式

* parallel=None (Default, no parallelization)
* parallel="processes"
* parallel="threads"
* parallel="dask"

对于规模不大的问题，建议使用 **parallel="processes"** 这种情况适用于在单机上进行交叉验证，且可以达到最高性能。
对于大型问题，使用 **dask** 集群可以在多机器上进行交叉验证。

dask不包含在 Prophet 中需要单独安装。

