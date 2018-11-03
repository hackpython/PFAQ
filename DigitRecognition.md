# 识别数字

## 背景介绍

机器学习（或深度学习）入门的"Hello World"，即识别MNIST数据集（手写数据集）。手写识别属于典型的图像分类问题，比较简单，同时MNIST数据集也很完备。MNIST数据集作为一个简单的计算机视觉数据集，包含一系列如图1所示的手写数字图片和对应的标签。图片是28x28的像素矩阵，标签则对应着0~9的10个数字。每张图片都经过了大小归一化和居中处理。有很多算法在MNIST上进行实验

## 数字识别PaddlePaddle-fluid版代码：

https://github.com/PaddlePaddle/book/tree/develop/02.recognize_digits