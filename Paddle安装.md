# Paddle安装

## `待审核`1.问题：paddlepaddle是否支持使用conda安装和支持MKL？

+ 版本号：`1.1.0`

+ 标签：`conda安装` `MKL`

+ 问题描述：以前在安装tensorflow的时候，发现tensorflow支持使用conda安装，超级方便。无论是windows还是Linux，只需要一个命令：
conda install tensorflow-gpu

paddlepaddle什么时候才能提供这样的安装方式？

+ 问题解答：

目前还没有支持conda，但是我们会尽快列入计划

你可以尝试使用下面命令来一键安装

```
pip install paddlepaddle
```

目前Fluid release1.3版本支持windows MKL功能。而Fluid Linux版本一直支持MKL功能。