1、利用txt_annotation.py文件可以获得基于Bayes-ResNet50恶意软件分类模型训练用的cls_test.txt和cls_train.txt文件
2、运行train.py对恶意软件数据进行训练
3、运行eval.py文件可获得评估结果，结果保存在生成的metrics_out文件夹中
4、classification.py文件使用训练好的模型进行预测
5、metrics.py是贝叶斯卷积层和贝叶斯全连接层计算相关指标的脚本
6、server.py文件是运行FastAPI的主文件，使用uvicorn server.py:app --reload运行起来
7、platform文件夹存放我们的云平台相关代码