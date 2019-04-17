# Paddle GPU模式

## `已审核`1.问题：在使用Fluid进行模型训练时，使用了GPU进行训练，但发现GPU的利用率几乎为0，这是为何？

+ 版本号：`1.1.0`

+ 标签：`GPU利用率`

+ 问题描述：在使用Fluid进行模型训练时，使用了GPU进行训练，但发现GPU的利用率几乎为0，这是为何？

+ 问题分析：在使用Fluid进行训练时，训练的设备是GPU，此时Fluid会将要训练的数据拷贝到GPU中，在进行数据拷贝是，GPU的利用率是几乎为0的。通常如果训练数据比较大，而模型计算量有比较小，这就会导致GPU大部分时间都拷贝数据，造成GPU利用率为0的现象。


+ 解决方法：

如果训练的模型比较简单，可以尝试直接使用CPU进行训练，或者使用多几张GPU卡来训练，降低IO占用的时间。

## `已审核`2.问题：使用Fluid训练模型时，使用多卡训练报错

+ 版本号：`1.1.0`

+ 标签：`多卡训练`

+ 问题描述：使用Fluid训练模型时，使用多卡训练报错

+ 报错输出：

```
EnforceNotMet: Failed to find dynamic library: libnccl.so ( libnccl.so: cannot open shared object file: No such file or directory ) 
 Please specify its path correctly using following ways: 
 Method. set environment variable LD_LIBRARY_PATH on Linux or DYLD_LIBRARY_PATH on Mac OS. 
 For instance, issue command: export LD_LIBRARY_PATH=... 
 Note: After Mac OS 10.11, using the DYLD_LIBRARY_PATH is impossible unless System Integrity Protection (SIP) is disabled. at [/paddle/paddle/fluid/platform/dynload/dynamic_loader.cc:157]
PaddlePaddle Call Stacks: 
0       0x7f19fce44e96p paddle::platform::EnforceNotMet::EnforceNotMet(std::__exception_ptr::exception_ptr, char const*, int) + 486
1       0x7f19fe6ea71ep paddle::platform::dynload::GetNCCLDsoHandle() + 1822
2       0x7f19fcf3d0f9p void std::__once_call_impl<std::_Bind_simple<decltype (ncclCommInitAll({parm#1}...)) paddle::platform::dynload::DynLoad__ncclCommInitAll::operator()<ncclComm**, int, int*>(ncclComm**, int, int*)::{lambda()#1} ()> >() + 9
3       0x7f1a6dfc2a80p pthread_once + 80
4       0x7f19fcf40651p paddle::platform::NCCLContextMap::NCCLContextMap(std::vector<boost::variant<paddle::platform::CUDAPlace, paddle::platform::CPUPlace
```

+ 问题分析：
从报错信息中发现`EnforceNotMet: Failed to find dynamic library: libnccl.so ( libnccl.so: cannot open shared object file: No such file or directory )`语句，即表示没有找到libnccl.so文件，这可能是由于nccl安装不正确导致的。

+ 解决方法：

报错信息提示没有找到libnccl.so文件，可以尝试全局搜索一下该文件。

```bash
find / -name "libnccl.so*"
```

然后将找到的路径添加到 LD_LIBRARY_PATH 环境变量中则可。

+ 问题拓展：

NCCL是Nvidia Collective multi-GPU Communication Library的简称，它是一个实现多GPU的collective communication通信（all-gather, reduce, broadcast）库，Nvidia做了很多优化，以在PCIe、Nvlink、InfiniBand上实现较高的通信速度。

在深度学习使用多GPU并行训练时，通常会使用NCCL进行通信。

## `已审核`3.问题：Fluid1.2支持使用cuDNN6吗？

+ 版本号：`1.2.0`

+ 标签：`cuDNN6`

+ 问题描述：Fluid1.2支持使用cuDNN6吗？

+ 问题分析：
paddle目前发布的版本对cuda、cudnn的支持情况如下：cuda8.0+cudnn5, cuda8+cudnn7, cuda9+cudnn7。如果要使用cuDNN6，需要自己从源码编译一下进行使用。通过编译安装后，使用 use_gpu=True 后，可以正常运行，说明编译安装成功

+ 解决方法：

要实现Fluid1.2对cuDNN6的支持，需要从源码编译安装Paddle，具体的步骤请参考官方文档：
http://paddlepaddle.org/documentation/docs/zh/1.2/beginners_guide/install/compile/compile_Ubuntu.html

## `已审核`4.GPU多卡情况下加载模型评估结果有误

+ 版本号：`1.2.0`

+ 标签：`GPU`

+ 问题描述：我在GPU多卡下的模型预测结果有问题。训练的时候训2个batch我就把模型保存下来，但是重新加载这个模型来评估的时候模型的预测结果跟我当时保存它的时候不一致。
因为我保存模型参数的时候和我重新加载模型参数进行评估的时候，我都用同一份测试数据进行预测，按道理模型参数一致，测试数据一致，预测结果也应该一致。

+ 问题详情：

![](https://raw.githubusercontent.com/ayuLiao/images/master/paddlegpu1.png)

这是我测一个batch的数据，预测结果有两维，左边的是加载模型之后的预测结果，右边的是保存模型时的预测结果，如图所示前17个分数是一致的，从第18个结果开始就不一致了。
用的是Python预测的，用fluid.io.load_params加载模型，用fluid.io.save_params保存模型。

+ 问题分析：
使用的Paddle接口可能存在问题，对于预测任务，建议使用Paddle中的预测接口

+ 解决方法：

python预测接口是load_inference_model和save_inference_model. 需要把program desc即model保存下来，建议使用预测接口报错，具体参考：
http://paddlepaddle.org/documentation/docs/zh/1.3/api_guides/low_level/inference.html























