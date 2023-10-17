> assertion
  third party

> quadratic_para 拟合数据

> results_opti 

> benchmark
- data (data, label)



> cal_fairness.py
 
   cal_fairness函数:
 - 通过spec.json文件输入模型等信息，模型信息对应到model.txt文件，也是修改参数时需要修改的文件。
 - 返回公平性，这里把前五十个文件看成一类，后五十个看成一类。原始网络公平性为0.06
 - 其他函数可忽略，为完成功能所需的函数。

> benchmark文件夹
> 
data/jigsaw文件夹：

 - data0-99.txt为输入文件，内容为一个列表，且列表长度不一样，因为每个文件代表一条评论，评论长度不一样，每个词语用50维向量表示。
 - labels.txt为标签文件。
 - 其他文件为删除某个词的输入文件，可忽略

nnet/jigsaw_lstm文件夹

 - bias、h_c、weights文件夹表示model.txt中提取出来的网络的参数信息
 - original/model.txt文件为模型文件，parse.py为处理model.txt的文件，可以得到四层（三层lstm一层linear）的参数信息，输入一般是（x,50），x表示多少词汇，每个data.txt文件不同。
 - spec.json为框架信息，定义网络结构及参数信息，作为输入文件用于计算公平性
 

> 其他文件夹
> 
用于构建框架，可忽略


> TODO

 - 修改参数信息，改变weights,bias,h_c文件夹的参数信息，重新输入到cal_fairness，计算公平性
 - 根据参数和公平性拟合函数
 - 采取无约束优化方法优化公平性
 - 试着添加约束得到更好的结果

