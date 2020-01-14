1.在启动程序之前首先应该安装两个python第三方库

	1). pystruct ---- Link: http://pystruct.github.io/
	
	2). nltk ---- Link: http://www.nltk.org/
		以及相关数据包: corpora; taggers; tokenizers;

2.模块介绍
	
	1). model文件夹: 用于保存已训练好的CRF模型, 用于在线预测
	
	2). ResultData文件夹: 用于保存运行结果数据, 不同参数有分别对应的子文件夹
	
	3). SourceData文件夹: 用于存放源数据集的, 了解评论数据格式, 可对其文件进行参考
	
	4). FraudGuard.py: FraudGuard主程序
	
	5). k_fold_crossvalidation_multithread_CRF.py: 用于统计"CRF"模型 K折交叉效验的结果
	
	6). reviewContent_process.py 评论文本处理辅助程序
		
3.首先启动FraudGuard.py

	1). 首先设置好:δ - K - α; 三个关键参数,其他默认即可
	
	2). OUT_CAMPAIGN_FEATURE： 输出群组特征; 如果没有输出群组特征将无法运行模块5).
			
	3). OUT_CAMPAIGN:  输出群组信息
		
	4). LOAD_EFGCRF_PREDICT： 载入已训练好的模型, 用于在线预测
		将训练好的模型保存到model文件夹下, 接下来即可进行在线预测

4.当FraudGuard.py运行完毕后, 按照设置好的相应参数, 可运行以下模块:
	
	1). k_fold_crossvalidation_multithread_CRF.py;

5.实验结果 
	通过交叉效验模块4.1)运行完后, 将统计结果按照如ResultData文件夹下的"例子文件"(NYC_d0.5_K100_A2.0: 表示使用YelpNYC数据集, δ=0.5; K=100; α=2.0)模式保存观察即可
	