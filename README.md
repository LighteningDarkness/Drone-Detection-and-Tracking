# 前排提示
此仓库为某高校直研考核代码，可能仅针对此任务有效，请谨慎clone

# 说明
代码基于YOLOv7以及ByteTrack代码改动，将ByteTrack检测器由YOLOx切换为YOLOv7.
# 数据准备
请将数据集解压后放置于dataset下，并将解压后文件夹改名为Drone-Detection&Tracking，输入以下命令以生成yolo格式所需的文件：
``` shell
python dataset/generate_yolo.py
```
# 训练
给定150个带标签视频，选取其中约80%作为训练集，剩余作为验证集，train.py代码进行了改动，仅支持此次任务需求，计算准确率时采取micro-acc策略，因此计算时仅仅将置信度最高的bbox作为目标候选，您可以通过以下命令复现：
``` shell
./train.sh
```

# 验证
验证提供了两种方式，如果您想知道YOLOv7 real-time object detection的效果，您可以通过以下命令实现：
``` shell
python cal_acc.py --method yolov7
```
如果您想利用ByteTrack+YOLOv7验证，可以采用以下命令：
``` shell
python cal_acc.py --method ByteTrack
```

# 推理
如果您想利用YOLOv7得出测试集的结果，可以输入以下命令，结果储存在results：
``` shell
./detect.sh
```
如果您想利用ByteTrack+YOLOv7得出测试集的结果，可以输入以下命令，结果储存在result_bt：
``` shell
./track.sh
```