04_Fourier Order for Seasonalities季节性的傅里叶级数

Seasonalities are estimated using a partial Fourier sum. See the paper for complete details
估计季节性使用了部分的傅里叶加法，详细信息请参阅论文：https://peerj.com/preprints/3190/

The number of terms in the partial sum (the order) is a parameter that determines how quickly the seasonality can change. 
部分和中的项数(级数)是一个参数，它决定了季节性变化的快慢。 
# 默认值通常是合适的，但当季节性需要适应更高频率的变化时，可以增加默认值，而且通常不太平滑。#
# 增加傅里叶项的数量可以使季节性更适合变化更快的周期，但也可能导致过拟合:
# N个傅里叶项对应于用于建模周期的2N个变量