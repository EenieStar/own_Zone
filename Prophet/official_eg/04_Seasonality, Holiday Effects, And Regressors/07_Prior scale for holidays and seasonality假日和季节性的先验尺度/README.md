# 07_Prior scale for holidays and seasonality
# 假日和季节性的先验尺度

有时候，节假日在训练中会出现过拟合的情况
使用holidays_prior_scale参数来规定假日的先验尺度，以调整训练的拟合度，使得其更加平滑。

### **默认情况下，此参数为10，正则化很少。**

减少 holidays_prior_scale 参数可以减弱节日效应: 
