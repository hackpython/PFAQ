# PaddlePaddle GPU模式

## 1.问题：PaddlePaddle使用GPU训练时候报错 

+ 关键字：`GPU`

+ 问题描述：使用PaddlePaddle 0.9.0版本在KVM虚拟机的ubunut环境下使用GPU运行代码报错，使用的是CUDA 8.0

+ 报错输出：

```python
root@gputest:~/demo/mnist# sh train.sh 
Using debug command gdb --args
GNU gdb (Ubuntu 7.7.1-0ubuntu5~14.04.2) 7.7.1
Copyright (C) 2014 Free Software Foundation, Inc.
License GPLv3+: GNU GPL version 3 or later <http://gnu.org/licenses/gpl.html>
This is free software: you are free to change and redistribute it.
There is NO WARRANTY, to the extent permitted by law.  Type "show copying"
and "show warranty" for details.
This GDB was configured as "x86_64-linux-gnu".
Type "show configuration" for configuration details.
For bug reporting instructions, please see:
<http://www.gnu.org/software/gdb/bugs/>.
Find the GDB manual and other documentation resources online at:
<http://www.gnu.org/software/gdb/documentation/>.
For help, type "help".
Type "apropos word" to search for commands related to "word"...
Reading symbols from /usr/bin/../opt/paddle/bin/paddle_trainer...done.
(gdb) r
Starting program: /usr/opt/paddle/bin/paddle_trainer --config=vgg_16_mnist.py --dot_period=10 --log_period=100 --test_all_data_in_one_period=1 --use_gpu=1 --trainer_count=4 --num_passes=10 --save_dir=./mnist_vgg_model
[Thread debugging using libthread_db enabled]
Using host libthread_db library "/lib/x86_64-linux-gnu/libthread_db.so.1".
I0221 18:48:34.436648 29562 Util.cpp:155] commandline: /usr/opt/paddle/bin/paddle_trainer --config=vgg_16_mnist.py --dot_period=10 --log_period=100 --test_all_data_in_one_period=1 --use_gpu=1 --trainer_count=4 --num_passes=10 --save_dir=./mnist_vgg_model 
[New Thread 0x7ffff3686700 (LWP 29567)]
[New Thread 0x7ffff2e85700 (LWP 29568)]
```

+ 问题解答：
目前PaddlePaddle多在CUDA 7.5环境下运行，PaddlePaddle在GPU环境下大致的三种情况：

	+ 通常建议使用CUDA 7.5，paddle在这个环境上用的多，出问题的概率小
	+ CUDA 8.0环境源码编译安装PaddlePaddle + 8.0环境运行，这种环境，PaddlePaddle使用GPU通常也没有问题，经过测试。
	+ CUDA 7.5环境源码编译 + 8.0环境运行，一般也是没有问题的，不过没有测试过提问者当前使用的ubuntu环境。

+ 解决方法：
使用CUDA 7.5环境，同时建议安装最新的PaddlePaddle。

+ 问题拓展：
不止PaddlePaddle，大多数深度学习框架对GPU的版本支持都有相应的要求，如TensorFlow 1.5版本只支持cuda 9.0，所以选择对应版本的PaddlePaddle框架以及该版本Paddle要求的CUDA版本是让其正常运行的关键点。

## 2.问题：PaddlePaddle如何指定对应的GPU设备进行模型的训练

+ 关键字：`GPU` `指定设备`

+ 问题描述：当一台设备上有多个GPU设备时，PaddlePaddle怎么指定具体使用某个GPU设备

+ 问题解答：

例如机器上有4块GPU，编号从0开始，指定使用2、3号GPU：

+ 方式1：通过 `CUDA_VISIBLE_DEVICES` 环境变量来指定特定的GPU。

```python
env CUDA_VISIBLE_DEVICES=2,3 paddle train --use_gpu=true --trainer_count=2
```

+ 方式2：通过命令行参数 `--gpu_id` 指定。

```python
paddle train --use_gpu=true --trainer_count=2 --gpu_id=2
```

## 3.问题：DNN可以同时运行在GPU和CPU上吗？

+ 关键字：`DNN` `GPU` `CPU`

+ 问题描述：在PaddlePaddle的github issue讨论中看到整体网络使用GPU的时候，可以部分layer用CPU，但我没找到相应的设置方式，想请教下怎么设置？

+ 问题解答：
目前PaddlePaddle分为Fluid版、v2版以及比较旧的v1版本，让神经网络的部分layer使用CPU，其他层使用GPU训练在PaddlePaddle的v1版中是支持的，而v2版本没有支持这一功能，目前PaddlePaddle的v2版主要以维持其稳定性为主，后期估计不会增加这以支持。

	最新的fluid版本是支持这一功能的，会在run的时候，自动进行CPU/GPU的转换，您可以试下fluid版本。在fluid版本中，如果的有的op没有GPU的kernel 而且该op的输入是在GPU的， 在Op里面会将数据拷贝到CPU端，这个过程不需要用户参与的。

+ 解决方法：
如果实现的神经网络部分网络结构需要使用GPU而另一部分需要使用CPU，可以尝试Fluid版的PaddlePaddle，它会自动的对不支持GPU的操作自动调配到CPU上训练执行，不需要用户进行额外的操作，较为易用。
