# TensorFlow 教程 - 新手入门笔记
## 介绍

TensorFlow™ 是一个采用数据流图（data flow graphs），用于数值计算的开源软件库。TensorFlow 最初由Google大脑小组（隶属于Google机器智能研究机构）的研究员和工程师们开发出来，用于机器学习和深度神经网络方面的研究，但这个系统的通用性使其也可广泛用于其他计算领域。它是谷歌基于DistBelief进行研发的第二代人工智能学习系统。2015年11月9日，Google发布人工智能系统TensorFlow并宣布开源。

## TensorFlow名字的来源

其命名来源于本身的原理，Tensor（张量）意味着N维数组，Flow（流）意味着基于数据流图的计算。Tensorflow运行过程就是张量从图的一端流动到另一端的计算过程。张量从图中流过的直观图像是这个工具取名为“TensorFlow”的原因。

## 什么是数据流图（Data Flow Graph）？

数据流图用“节点”(nodes)和“线”（edges）的有向图来描述数学计算。“节点”一般用来表示施加的数学操作，但也可以表示数据输入（feed in）的起点/输出（push out）的终点，或者是读取/写入持久变量（persistent variable）的终点。“线”表示“节点”之间的输入/输出关系。这些数据“线”可以运输“size可动态调整”的多维数组，即“张量”（tensor）。一旦输入端所有张量准备好，节点将被分配到各种计算设备完成异步并行地执行计算。

![](../asserts/newbie-first-01.gif)

## Tensorflow的特性

高度的灵活性： TensorFlow不是一个严格的“神经网络”库。只要你可以将你的计算表示为一个数据流图，你就可以使用TensorFlow。

可移植性（Portability）：Tensorflow可以运行在台式机、服务器、手机移动等等设备上。而且它可以充分使用计算资源，在多CPU和多GPU上运行。

多语言支持：Tensorflow提供了一套易用的Python使用接口来构建和执行graphs，也同样提供了一套易于C++使用的接口（目前训练神经网络只支持python，C++接口只能使用已经训练好的模型）。未来还会支持Go、Java、Lua、JavaScript、R等等。

性能最优化：TensorFlow给予了线程、队列、异步操作等最佳的支持，TensorFlow可以把你手边硬件的计算潜能全部发挥出来，它可以充分利用多CPU和多GPU。

## 下载及安装

既可以直接使用二进制程序包也可以从github源码库克隆源码编译安装。

要求
TensorFlow 提供的Python API支持Python2.7和Python3.3+

GPU版本的二进制程序包只能使用Cuda Toolkit8.0 和 cuDNN v5。如果你使用的是其他版本（Cuda toolkit >= 7.0 and cuDNN >= v3），那你就必须使用源码重新编译安装。

推荐几种Linux平台的安装方式：

* Pip install： 可能会升级你之前安装过的Python包，对你机器上的Python程序造成影响。
* Anaconda install：把TensorFlow安装在Anaconda提供的环境中，不会影响其他Python程序。
* Installing from sources：把TensorFlow源码构建成一个pip wheel 文件，使用pip工具安装它。
Pip installation
Pip是一个用来安装和管理Python软件包的包管理系统。

安装pip（如果已经安装，可以跳过）
Ubuntu/Linux 64-bit
```
$ sudo apt-get install python-pip python-dev
```

直接使用pip安装TensorFlow
```
$ pip install tensorflow
```
如果提示找不到对应的包，使用
```
pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.12.0rc1-cp27-none-linux_x86_64.whl
```

安装GPU支持的版本：
Requires CUDA toolkit 8.0 and CuDNN v5. 其他版本，参考下面的 “Installing from sources”
```
$ pip install tensorflow-gpu
```
如果提示找不到对应的包，使用
```
pip install --ignore-installed --upgrade TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-0.12.0rc1-cp27-none-linux_x86_64.whl
```

注意：如果是从一个较老的版本（TensorFlow<0.7.1）进行升级，需要首先卸载之前的TensorFlow和protobuf使用：pip uninstall

#### Anaconda installation
Anaconda是一个Python发行版，包括大量的数字和科学计算包。使用“conda”来管理软件包，并且拥有自己的环境系统。安装步骤
安装Anaconda
创建conda环境
激活conda环境，在其中安装TensorFlow
每次使用TensorFlow时，激活conda环境

Anaconda具体的安装和使用可以参考：https://www.continuum.io/downloads

#### Installing from sources

从源码构建TensorFlow，具体的步骤参考：http://blog.csdn.net/toormi/article/details/52904551#t8

