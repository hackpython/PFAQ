# Paddle多卡模式

## `待审核`1.问题：fluid 1.0, 1.3 多卡训练出core

+ 版本号：`1.1.0` `1.3.0`

+ 标签：`多卡`

+ 问题描述：以前1.0可以跑的代码，现在多卡都不行了，直接出core了，然后下载fluid 1.3运行，同样出core

+ 报错输出：

![](https://raw.githubusercontent.com/ayuLiao/images/master/paddle%E5%A4%9A%E5%8D%A11.png)

+ 相关代码：

```
parallel_exe = exe
if args.parallel:
	print('parallel exec')
	parallel_exe = fluid.ParallelExecutor(use_cuda=True, loss_name=total_loss.name)
fetch_vars = [class_loss, offset_loss, lr]
var_names = [var.name for var in fetch_vars]
```

+ 问题分析：

从报错图片来看，报错应该与NCCL的版本有关，请核对一下当前设备使用的NCCL版本

+ 解决方法：

NCCL版本的问题。切换到`nccl2.1.*`就可以了