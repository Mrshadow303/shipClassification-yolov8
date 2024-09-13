本项目使用yolov8模型进行训练，目的是识别船类红外模拟图像



数据集采用格式为

```
dataset
    ├──images         存放训练图片原图
    │    ├──test         
    │    ├──train        
    │    └──val
    ├──lables         存放训练标签
    │    ├──test         
    │    ├──train        
    │    ├──val        
    │    ├──train.cache        
    │    └──val.cache
    └──yolov8.yaml    配置文件，需要有数据集路径、类别数、类别名称
```



yolov8.yaml示例

```
# 数据集路径
train: .../datasets/images/train
val: .../datasets/images/val
test: .../datasets/images/test

# 类别数
nc: 10

# 类别名称
names:
  - A
  - BB
  - CCCC
  - DD
  - E
  - F
  - G
  - H
  - I
  - J


```



训练

```
yolo detect train data=...\datasets\yolov8.yaml model=...\ultralytics\models\yolo\detect\yolov8n.pt epochs=10 imgsz=640 batch=16 device=0
```



分类预测(这里是训练后的模型)

```
yolo predict model=...\runs\detect\train\weights\best.pt source=path\image.png
```



yolo需要配置环境变量，也可以直接使用绝对路径

```
E:\Conda\Conda\Scripts\yolo.exe predict model=...\runs\detect\train\weights\best.pt source=path\image.png
```




