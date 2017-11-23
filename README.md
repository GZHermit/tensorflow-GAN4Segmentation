[使用说明]
主程序入口:main.py
控制参数：
1.g_name str(可选项): 
- vgg32
- vgg16
- vgg8
- resnet50

2.d_name str(可选项):
- null
- disc_add_vgg
- disc_add_res50

3.learning_rate float
- 1e-4:训练vgg
- 1e-3:训练res50
- 3e-5:训练并且加d

4.is_multitask bool 
- True:d的image feature是从g的某一层中导出来的
- False:d有独立的model用于提取image feature

5.is_val bool
- True:开始模型验证
- False:开始模型训练

6.lambd float(d_loss的控制参数)
- 1e-2

7.data_dir list(list里面放置各种训练数据集的路径)

8.restore_from str(存放已经训练好的权重，可用于val或者断点重训)

9.baseweight_from dict(存放预训练权重，用于初始化模型)

百度网盘:
resnet50预训练权重:  链接: https://pan.baidu.com/s/1qYwVxAs 密码: vfs8
vgg16预训练权重:     链接: https://pan.baidu.com/s/1jHFilO6 密码: 8bed
