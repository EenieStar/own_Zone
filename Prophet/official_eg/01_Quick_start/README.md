创建prophet类，之后调用其 fit 和 predict 方法
prophet的输入永远是一个dataframe,且包含两列'ds'、'y'
其中:
    ds 必须是一个datetime(应为同一个格式，YYYY-MM-DD 或者 YYYY-MM-DD HH:mm:ss[时间戳])
    y 必须是数值型，表示我们即将进行预测的数据

官方示例使用 Peyton Manning 的相关数据
https://github.com/facebook/prophet/blob/main/examples/example_wp_log_peyton_manning.csv

