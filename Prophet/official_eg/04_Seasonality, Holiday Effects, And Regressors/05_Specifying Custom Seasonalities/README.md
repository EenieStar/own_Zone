# 定制指定季节性

## 01
默认情况下，当数据中的时间序列超过两个周期，Prophet将会在每个周期内自动训练（通过Prophet.fit()方法）周季节性、年季节性。
当数据只以日为单位时，它还将训练（通过Prophet.fit()方法）每日季节性

也可以通过函数 add_seasonality()方法自定义添加你想要的季节性（例如：每月、每季度、每小时等）

add_seasonality()的输入包含三个方面，自定义的季节性名称、以天为单位的季节周期(小时周期换算为x/24天)

例如，这里我们拟合了来自Quickstart的Peyton Manning数据，但将每周的季节性替换为每月的季节性。然后每月的季节性将出现在组成部分的图中:

## 02
建立 Prophet() 对象时，
可以对季节性模式进行优先设置值 seasonality_mode = ‘additive’ / ‘multiplicative’
这种情况下，接下来的所有的季节性因素和额外的回归因素都会被改写为相同模式。
不过，在后续自定义添加季节性[ (Prophet().add_seasonality() ]和额外回归[ (Prophet().add_regressor() ]时也可以使用mode参数进行自定义。
即mode = 'additive'/ 'multiplicative'