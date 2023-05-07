# DUT CV Project 2023: Image Classification

## 目录

1. [简介](#简介)
2. [代码结构](#代码结构)
3. [环境配置](#环境配置)
4. [快速复现](#快速复现)
5. [从零开始](#从零开始)
6. [使用说明](#使用说明)
7. [改进思路](#可能的改进思路)

## 简介

本项目是我的本硕贯通课程《计算机视觉》的期末大作业，作业内容是参加[图像分类挑战赛](https://www.kaggle.com/competitions/dlut-cv-project-2023-image-classification)并获得分数。本项目的模型获得了竞赛的第4名，测试集Public精度为68.758%，Private精度为69.540%。
本项目训练了3个标准模型和4个轻量级模型，挑战赛只能提交轻量级模型的结果。下表列出了各个模型的测试集精度以及模型的参数量和计算量。权重文件可在[百度网盘](https://pan.baidu.com/s/1qYJeqftS_35cd3iy7kf7Fg)下载，提取码为学校校庆日（4位数字，如2月14日为0214）。

| Model               | Public score | Private score | Parameters |   FLOPs   |
|:--------------------|:------------:|:-------------:|:----------:|:---------:|
| ResNet-50           |   75.017%    |    75.333%    |   26.31M   | 1349.23M  |
| ConvNeXt-Tiny       |   77.293%    |    77.218%    |   27.85M   | 1457.47M  |
| SwinTransformerV2-T | **77.844%**  |  **77.551%**  |   18.98M   |  990.55M  |
| ResNet-26           | **67.827%**  |  **68.183%**  |   7.95M    |  746.26M  |
| RepVGG-11           |   63.568%    |    64.172%    |   7.62M    |  713.09M  |
| ConvNeXt-Nano       |   63.793%    |    64.264%    |   7.43M    |  576.71M  |
| SwinTransformerV2-N |   56.931%    |    57.275%    |   7.42M    |  377.7M   |

如果你想了解我的技术细节，请查看我的[技术报告](docs/report.pdf)。

## 代码结构

```
DutImageClassification
├── data               # 数据集相关代码
    ├── __init__.py    # 数据处理、数据增强、数据集构建等代码
    └── convert.py     # 风格转换脚本
├── docs               # 存放相关文档
    └── report.pdf     # 技术报告
├── models             # 存放模型结构代码和权重
    ├── weights        # 存放权重文件
        ├── students   # 存放轻量级模型的权重
        └── teachers   # 存放标准模型的权重
    ├── __init__.py    # 模型API
    ├── convnext.py    # ConvNeXt模型代码
    ├── repvgg.py      # RepVGG模型代码
    ├── resnet.py      # ResNet模型代码
    └── swintfv2.py    # SwinTransformer模型代码
├── results            # 存放csv推理结果文件
    ├── students       # 存放轻量级模型的预测结果
    └── teachers       # 存放标准模型的预测结果
├── utils              # 存放其他代码
    ├── __init__.py    # 相关API
    ├── loss.py        # 损失函数
    └── progress.py    # 简易进度条
├── inference.py       # 推理脚本
├── LICENSE            # LICENSE文件
├── main.sh            # 主程序
├── README.md          # 说明文件
├── requirements.txt   # 依赖库列表
├── train.py           # 训练脚本
└── val.py             # 验证脚本
```

## 环境配置

先通过以下命令克隆本项目：

```shell
git clone https://github.com/Yue-0/DutImageClassification.git
cd ./DutImageClassification
mkdir -p models/weights/students
mkdir models/weights/teachers
mkdir -p results/students
mkdir results/teachers
```

本项目的解释器要求Python>=3.8，因为部分代码使用了海象运算符，所以3.8以前的Python版本不能运行本项目。

本项目的依赖库列表在[requirements](requirements.txt)中列出，包括：

* Thop >= 0.1.1
* NumPy >= 1.23.5
* OpenCV >= 4.5.5
* PyTorch >= 1.13.0
* PaddleHub >= 2.3.1
* TorchVision >= 0.14.0

项目对各个依赖库的版本没有太严格的要求，不要太旧即可，上面列出的是我使用的版本。
使用以下命令一键安装依赖库：

```shell
pip install -r requirements.txt
```

PaddleHub可能需要依赖PaddlePaddle，如果安装失败，请先[安装对应版本的PaddlePaddle](https://paddlepaddle.org.cn)。**如果你不需要从零开始复现我的结果，则无需安装PaddleHub。**

## 快速复现

* 本项目只提供在Linux系统下的复现方案，不保证在Windows系统下能正常运行；
* 本项目在PyCharm和VsCode等编辑器中运行时可能存在进度条异常的情况，建议在终端运行；
* 以下所有命令都在DutImageClassification目录下运行。

### 1.准备测试数据集

将竞赛数据集的测试集文件夹（test文件夹）拷贝到data文件夹下，保证data文件夹具有如下结构：

```
data
├── test
    ├── xxx.jpg
    ├── ...
    └── xxx.jpg
├── __init__.py
└── convert.py
```

### 2.下载训练好的权重

训练好的模型权重在[百度网盘](https://pan.baidu.com/s/1qYJeqftS_35cd3iy7kf7Fg)中，提取码为学校校庆日（4位数字，如2月14日为0214）。下载好压缩包后，解压到models文件夹中，覆盖weights文件夹。解压后的weights文件夹应具有如下结构：

```
weights
├── students
    ├── ConvNextNano
        └── best.pt
    ├── RepVGG
        └── best.pt
    ├── ResNet26
        └── best.pt
    └── SwinTransformerN
        └── best.pt
└── teachers
    ├── ConvNextTiny
        └── best.pt
    ├── ResNet50
        └── best.pt
    └── SwinTransformerT
        └── best.pt
```

### 3.生成csv预测结果文件

使用以下指令生成用于提交到竞赛服务器的结果文件，生成的文件在results/students/result.csv：

```shell
python inference.py --data data --models ConvNextNano ResNet26 RepVgg --weights \
models/weights/students/ConvNextNano/best.pt \
models/weights/students/ResNet26/best.pt \
models/weights/students/RepVgg/best.pt \
--output results/students/result.csv
```

使用以下指令生成用于提交作业的得分结果文件，生成的文件在results/students/score.csv：

```shell
python inference.py -s --data data --models ConvNextNano ResNet26 RepVgg --weights \
models/weights/students/ConvNextNano/best.pt \
models/weights/students/ResNet26/best.pt \
models/weights/students/RepVgg/best.pt \
--output results/students/score.csv
```

使用以下指令生成Public精度为80.413%的结果文件，生成的文件在results/teachers/ensemble.csv，**该结果不能用于提交**：

```shell
python inference.py --data data --models ConvNextTiny ResNet50 SwinTransformerT --weights \
models/weights/teachers/ConvNextTiny/best.pt \
models/weights/teachers/ResNet50/best.pt \
models/weights/teachers/SwinTransformerT/best.pt \
--output results/teachers/emsenble.csv
```

## 从零开始

本项目提供了完整的从零开始复现的脚本， 首先请将竞赛数据集拷贝到data文件夹下，保证data文件夹具有如下结构：

```
data
├── test
    ├── 00000.jpg
    ├── 00001.jpg
    ├── ...
    └── 14499.jpg
├── train
    ├── 00
        ├── 000.jpg
        ├── 001.jpg
        ├── ...
        └── 499.jpg
    ├── 01
    ├── ...
    └── 49
├── val
    ├── 00
        ├── 00.jpg
        ├── 01.jpg
        ├── ...
        └── 49.jpg
    ├── ...
    └── 49
├── __init__.py
└── convert.py
```

然后在DutImageClassification目录下使用以下命令即可一键从零开始复现我的结果：

```shell
bash main.sh
```

## 使用说明

本项目提供了一些脚本，用于实现一些特定的功能。

### 1.查看轻量级模型的参数量和计算量

本项目一共定义了四个轻量级模型，它们分别具有ResNet、RepVGG、ConvNeXt和SwinTransformer的结构，使用以下命令查看各个轻量级模型的参数量和计算量：

```
python models/resnet.py
              repvgg.py
              convnext.py
              swintfv2.py
```

参数量和计算量在类别数为50且输入图像大小为3x128x128的情况下统计。
你也可以通过val.py查看各种模型对指定输入图像大小的计算量，具体方法在[val.py的使用方法](#4.验证)中介绍。

### 2.风格转换

本项目单独实现了用于风格转换的脚本，需要把数据构建成如下形式：

```
path_to_images
├── train
    ├── xxx
        ├── xxx.jpg
        ├── ...
        └── xxx.jpg
    ├── ...
    └── xxx
├── val
    ├── xxx
        ├── xxx.jpg
        ├── ...
        └── xxx.jpg
    ├── ...
    └── xxx
```

然后，使用以下命令进行风格转换：

```shell
python data/convert.py path/to/images
```

转换完毕后，会在图像目录下生成三个文件夹，分别存放动漫风格、素描风格和梵高油画风格的图像，每个文件夹的结构都与原图像文件夹的结构相同。

### 3.训练模型

[train.py](train.py)是模型训练脚本，通过终端命令可以训练指定的模型。例如，下面的命令用于训练一个ResNet-50，数据集路径为data，权重文件保存在models/weights/teachers/ResNet50下，会保存best.pt和last.pt两个权重文件，分别表示验证集精度最高的权重和最后一次训练后保存的权重：

```shell
python train.py --save=models/weights/teachers/ResNet50 --data=data --model=ResNet50 \
--epochs=120 --lr=1e-1 --batch_size=256 --weight_decay=1e-4 --warmup=0
```

下面的命令用于从某权重文件中加载参数继续训练：

```shell
python train.py --save=models/weights/teachers/ResNet50 --data=data --model=ResNet50 \
--epochs=120 --lr=1e-1 --batch_size=256 --weight_decay=1e-4 --warmup=0 \
--pretrained=models/weights/teachers/ResNet50/last.pt
```

下面的命令用于使用训练好的ResNet-50蒸馏RepVGG：

```shell
python train.py --save=models/weights/students/RepVgg --data=data --model=RepVgg \
--epochs=180 --lr=7.5e-2 --batch_size=192 --weight_decay=5e-5 --warmup=1 \
--method=distillation --teacher=ResNet50 --teacher_weights=models/weights/teachers/ResNet50/best.pt
```

下表列出了train.py的所有命令行参数：

| 参数名               |  类型   |  默认   | 说明                                                       |
|:------------------|:-----:|:-----:|:---------------------------------------------------------|
| --save            | Path  |   -   | 权重保存的路径                                                  |
| --data            | Path  |   -   | 数据集路径                                                    |
| --model           | Model |   -   | 模型名称                                                     |
| --device          |  str  | None  | 计算设备，如"cpu"、"cuda"，若为None，则自动选择可用的设备                     |
| --pretrained      | Path  | None  | 预训练权重的路径，若为None，则标准模型将自动加载torchvision提供的权重，轻量级模型将随机初始化权重 |
| --lr              | float | 1e-3  | 基础学习率                                                    |
| --batch_size      |  int  |  64   | mini-batch的大小                                            |
| --epochs          |  int  |  100  | 训练轮数                                                     |
| --warmup          |  int  |   0   | 线性学习率预热的轮数，若为0，则不使用线性预热                                  |
| --weight_decay    | float | 5e-5  | L2正则化系数，若为0，则不使用L2正则化                                    |
| --num_workers     |  int  |   4   | 加载数据的线程数                                                 |
| --label_smooth    | float |  0.1  | 标签平滑系数，若为0，则不使用标签平滑                                      |
| --method          |  str  | train | 训练模式，目前仅支持"train"或"distillation"，分别表示普通训练和蒸馏训练           |
| --test_size       |  int  |  160  | 训练图像resize的短边尺寸                                          |
| --train_size      |  int  |  128  | 神经网络接受的输入图像尺寸                                            |
| --teacher         | Model | None  | 教师模型的名称，**在蒸馏训练时必须指定此参数**，仅在蒸馏训练时有效                      |
| --teacher_weights | Path  | None  | 教师模型的权重文件路径，**在蒸馏训练时必须指定此参数**，仅在蒸馏训练时有效                  |
| --alpha           | float |  0.7  | 蒸馏权重系数，仅在蒸馏训练时有效                                         |
| --temperature     | float |  1.0  | 蒸馏温度，仅在蒸馏训练时有效                                           |

其中Model类型包括：
* ResNet50
* ResNet26
* RepVgg
* ConvNextTiny
* ConvNextNano
* SwinTransformerV2T
* SwinTransformerV2N

以上名称区别大小写，目前仅支持以上7个模型名称。

### 4.验证

[val.py](val.py)是模型验证脚本，通过终端命令可以验证指定的模型和权重。

如果你没有[从零开始复现](#从零开始)过，你需要先运行以下命令来进行风格转换。风格转换需要的时间比较久，请耐心等待。风格转换的命令只需要执行一次，运行成功后，后续无需再次运行。

```shell
python data/convert.py path/to/data
```

执行完风格转换后，可以进行模型验证。验证集是经过数据增强后的验证集，具体细节在我的[技术报告](docs/report.pdf)中说明。
下面的命令用于验证训练好的ResNet-50，数据集路径为data，权重文件为在models/weights/teachers/ResNet50/best.pt：

```shell
python val.py --data=data --model=ResNet50 --weights=models/weights/teachers/ResNet50/best.pt
```

程序执行结束后，会输出模型的验证集精度，以及模型的参数量和计算量。

下表列出了val.py的所有命令行参数：

| 参数名          |  类型   |  默认值  | 说明                                   |
|:-------------|:-----:|:-----:|:-------------------------------------|
| --data       | Path  |   -   | 数据集路径                                |
| --model      | Model |   -   | 模型名称                                 |
| --weights    | Path  |   -   | 模型权重路径                               |
| --device     |  str  | None  | 计算设备，如"cpu"、"cuda"，若为None，则自动选择可用的设备 |
| --test_size  |  int  |  160  | 图像resize的短边尺寸                        |
| --train_size |  int  |  128  | 神经网络接受的输入图像尺寸                        |

### 5.预测

[inference.py](inference.py)是模型预测和集成的脚本，通过终端命令可以集成指令模型的预测结果。
[快速复现](#3.生成csv预测结果文件)中给出了该脚本的使用示例，脚本的所有命令行参数如下：

| 参数名          |  类型   |  默认值  | 说明                                   |
|:-------------|:-----:|:-----:|:-------------------------------------|
| --data       | Path  |   -   | 数据集路径                                |
| --models     | Model |   -   | 模型名称，多个模型用空格隔开                       |
| --weights    | Path  |   -   | 模型权重路径，多个模型权重用空格隔开，顺序与模型名称一一对应       |
| --output     | Path  |   -   | 输出的csv文件路径                           |
| --device     |  str  | None  | 计算设备，如"cpu"、"cuda"，若为None，则自动选择可用的设备 |
| --score      | bool  | False | 是否输出各个类别的置信度得分csv文件，如需设为True，直接-s即可  |
| --test_size  |  int  |  160  | 图像resize的短边尺寸                        |
| --train_size |  int  |  128  | 神经网络接受的输入图像尺寸                        |

## 可能的改进思路

我的方法只能获得2023年挑战赛的第4名，如果想要改进我的方法，可以考虑从以下方面入手：
1. 更有效的数据增强。我使用的随机风格变换代码在[convert.py](data/convert.py)中，可以换用生成图像风格与测试集风格更接近的生成方法来进行风格转换。
2. 更先进的模型。我的所有模型代码都定义在[models](models)文件夹中，可以换用更先进的模型取得更好的效果，不过需要注意模型的参数量和计算量不能超过挑战赛的限定值。
3. 更强大的知识蒸馏方法。我的知识蒸馏代码在[train.py](train.py)的class Distiller中，可以换用更强大的知识蒸馏手段提高学生模型的精度。
4. 更适合的超参数。我使用的所有超参数都没有经过细调，精细地调整各种超参数可能也可以提高模型的精度，如改变batch_size和学习率等。
