# 图像分类

## 背景介绍
图像相比文字能够提供更加生动、容易理解及更具艺术感的信息，是人们转递与交换信息的重要来源。在本教程中，我们专注于图像识别领域的一个重要问题，即图像分类。

图像分类是根据图像的语义信息将不同类别图像区分开来，是计算机视觉中重要的基本问题，也是图像检测、图像分割、物体跟踪、行为分析等其他高层视觉任务的基础。图像分类在很多领域有广泛应用，包括安防领域的人脸识别和智能视频分析等，交通领域的交通场景识别，互联网领域基于内容的图像检索和相册自动归类，医学领域的图像识别等。

基于深度学习的图像分类方法，可以通过有监督或无监督的方式学习层次化的特征描述，从而取代了手工设计或选择图像特征的工作。深度学习模型中的卷积神经网络(Convolution Neural Network, CNN)近年来在图像领域取得了惊人的成绩，CNN直接利用图像像素信息作为输入，最大程度上保留了输入图像的所有信息，通过卷积操作进行特征的提取和高层抽象，模型输出直接是图像识别的结果。这种基于"输入-输出"直接端到端的学习方法取得了非常好的效果，得到了广泛的应用。

## 图像分类PaddlePaddle-fluid版代码：
https://github.com/PaddlePaddle/book/tree/develop/03.image_classification

## `已审阅` 1.问题：在使用VGG网络的时候提示vgg包不存在

 + 问题描述：

在使用VGG卷积神经网络训练CIFAR-10数据集的时候，在导入VGG网络包的时候出错，错误提示没有vgg库。

 + 报错信息：

```
ImportError                               Traceback (most recent call last)
<ipython-input-1-600eb39503dc> in <module>
     16     from paddle.fluid.inferencer import *
     17 
---> 18 from vgg import vgg_bn_drop
     19 from resnet import resnet_cifar10

ImportError: No module named 'vgg'
```

 + 问题复现：在项目的开头就使用导包的方式导入vgg卷积神经网络，结果就会报错。错误代码如下：

```python
from vgg import vgg_bn_drop
```

 + 解决问题：PaddlePaddle的Fluid版本没有直接提供VGG卷积神经网络的接口，所以使用VGG神经完了还需要自己去定义这个VGG网络。

 + 问题分析：在PaddlePaddle的V2版本中，PaddlePaddle提供了`paddle.v2.networks.vgg_16_network`接口，这个就是可以直接使用的VGG16卷积神经网络接口。用过V2版本的用户会错误理解Fluid版本也有相同的接口，所以导致错误的出现。

 + 问题拓展：VGGNet是牛津大学计算机视觉组（Visual Geometry Group）和Google DeepMind公司的研究员一起研发的深度卷积神经网络。其探索了神经网络深度与其性能之间的关系，目前主要有VGG-16与VGG-19两种结构，其中VGG-16网络猴子那个包含参数的层数有16个，总共包含1.38亿个参数，其优点是简化了卷积神经网络的结构，而缺点就是训练的特征数量非常大。其结构如下图：

 ![](https://img-blog.csdn.net/20180117142931666?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMjU3MzcxNjk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)



## `已审阅` 2.问题：使用彩色图训练的是出现尺寸不一致的错误

 + 问题描述：使用CIFAR-10彩色图像数据集进行训练，按照定义图片输入数据的方式来定义输入层。根据图片的大小，输入层的形状设置成[1, 32, 32]，结果在训练的时候报错。


 + 报错信息：

```
<ipython-input-5-fb9e47c67b84> in train(use_cuda, train_program, params_dirname)
     37         num_epochs=EPOCH_NUM,
     38         event_handler=event_handler,
---> 39         feed_order=['pixel', 'label'])

/usr/local/lib/python3.5/dist-packages/paddle/fluid/contrib/trainer.py in train(self, num_epochs, event_handler, reader, feed_order)
    403         else:
    404             self._train_by_executor(num_epochs, event_handler, reader,
--> 405                                     feed_order)
    406 
    407     def test(self, reader, feed_order):

/usr/local/lib/python3.5/dist-packages/paddle/fluid/contrib/trainer.py in _train_by_executor(self, num_epochs, event_handler, reader, feed_order)
    481             exe = executor.Executor(self.place)
    482             reader = feeder.decorate_reader(reader, multi_devices=False)
--> 483             self._train_by_any_executor(event_handler, exe, num_epochs, reader)
    484 
    485     def _train_by_any_executor(self, event_handler, exe, num_epochs, reader):

/usr/local/lib/python3.5/dist-packages/paddle/fluid/contrib/trainer.py in _train_by_any_executor(self, event_handler, exe, num_epochs, reader)
    510                                       fetch_list=[
    511                                           var.name
--> 512                                           for var in self.train_func_outputs
    513                                       ])
    514                 else:

/usr/local/lib/python3.5/dist-packages/paddle/fluid/executor.py in run(self, program, feed, fetch_list, feed_var_name, fetch_var_name, scope, return_numpy, use_program_cache)
    468 
    469         self._feed_data(program, feed, feed_var_name, scope)
--> 470         self.executor.run(program.desc, scope, 0, True, True)
    471         outs = self._fetch_data(fetch_list, fetch_var_name, scope)
    472         if return_numpy:

EnforceNotMet: Enforce failed. Expected framework::slice_ddim(x_dims, 0, rank + 1) == framework::slice_ddim(label_dims, 0, rank + 1), but received framework::slice_ddim(x_dims, 0, rank + 1):384 != framework::slice_ddim(label_dims, 0, rank + 1):128.
Input(X) and Input(Label) shall have the same shape except the last dimension. at [/paddle/paddle/fluid/operators/cross_entropy_op.cc:37]
PaddlePaddle Call Stacks: 
```

 + 问题复现：在定义输入层的时候，使用输入层接口`fluid.layers.data`定义图像输入层，参数`shape`设置成`[1, 32, 32]`，结果就会报错。错误代码如下：

```python
def inference_network():
    images = fluid.layers.data(name='pixel', shape=[1, 32, 32], dtype='float32')
    predict = vgg_bn_drop(images)
    return predict
```


 + 解决问题：使用输入层接口`fluid.layers.data`定义图像输入层时，`shape`参数要满足输入格式是`[通道数, 宽, 高]`，出现上面的错误是因为通道数写错了。彩色图是有三个通道的，所以因为是3而不是1。

```python
def inference_network():
    images = fluid.layers.data(name='pixel', shape=[3, 32, 32], dtype='float32')
    predict = vgg_bn_drop(images)
    return predict
```

 + 问题拓展：图片有单通道的灰度图，还要三通道的彩色图，所以在定义输入层的形状的时候要根据图片是否是彩色图片来设置图片的通道数。彩色图的三个通道分别是RGB，分别表示红色、绿色、蓝色。

 + 问题分析：神经网络在处理图像数据时，通常将图像数字看成相应的多维矩阵，此时不同维度的含义就很重要的了，维度弄错了，神经网络处理的数据就完成不一样了，此时训练出来的模型就完全与自身预期不相符。


## `已审阅` 3.问题：使用CIFAR-10彩色图片训练出现输出数据维度错误

 + 问题描述：在使用CIFAR-10彩色图片训练，其中定义输入层的形状为`[3072]`的时候，出现卷积层输入的形状不为4维或者5维的错误。


 + 报错信息：

```
/usr/local/lib/python3.5/dist-packages/paddle/fluid/nets.py in img_conv_group(input, conv_num_filter, pool_size, conv_padding, conv_filter_size, conv_act, param_attr, conv_with_batchnorm, conv_batchnorm_drop_rate, pool_stride, pool_type, use_cudnn)
    229             param_attr=param_attr[i],
    230             act=local_conv_act,
--> 231             use_cudnn=use_cudnn)
    232 
    233         if conv_with_batchnorm[i]:

/usr/local/lib/python3.5/dist-packages/paddle/fluid/layers/nn.py in conv2d(input, num_filters, filter_size, stride, padding, dilation, groups, param_attr, bias_attr, use_cudnn, act, name)
   1639             'groups': groups,
   1640             'use_cudnn': use_cudnn,
-> 1641             'use_mkldnn': False
   1642         })
   1643 

/usr/local/lib/python3.5/dist-packages/paddle/fluid/layer_helper.py in append_op(self, *args, **kwargs)
     48 
     49     def append_op(self, *args, **kwargs):
---> 50         return self.main_program.current_block().append_op(*args, **kwargs)
     51 
     52     def multiple_input(self, input_param_name='input'):

/usr/local/lib/python3.5/dist-packages/paddle/fluid/framework.py in append_op(self, *args, **kwargs)
   1205         """
   1206         op_desc = self.desc.append_op()
-> 1207         op = Operator(block=self, desc=op_desc, *args, **kwargs)
   1208         self.ops.append(op)
   1209         return op

/usr/local/lib/python3.5/dist-packages/paddle/fluid/framework.py in __init__(***failed resolving arguments***)
    654         if self._has_kernel(type):
    655             self.desc.infer_var_type(self.block.desc)
--> 656             self.desc.infer_shape(self.block.desc)
    657 
    658     def _has_kernel(self, op_type):

EnforceNotMet: Conv intput should be 4-D or 5-D tensor. at [/paddle/paddle/fluid/operators/conv_op.cc:47]
PaddlePaddle Call Stacks: 
0       0x7f8683d586b6p paddle::platform::EnforceNotMet::EnforceNotMet(std::__exception_ptr::exception_ptr, char const*, int) + 486
1       0x7f86845cf940p paddle::operators::ConvOp::InferShape(paddle::framework::InferShapeContext*) const + 3440
2       0x7f8683e00f86p paddle::framework::OpDesc::InferShape(paddle::framework::BlockDesc const&) const + 902
```

 + 问题复现：在定义网络的输入层的时候，使用`fluid.layers.data`接口定义输入数据的时，设置`shape`的值为`[3072]`，在启动训练的时候出错。错误代码如下：

```python
def inference_network():
    images = fluid.layers.data(name='pixel', shape=[3072], dtype='float32')
    predict = vgg_bn_drop(images)
    return predict
```

 + 解决问题：在使用`fluid.layers.data`接口定义图片输入层时，设置`shape`应该的是`[通道数, 宽, 高]`，所以设置为`[3, 32, 32]`。正确代码如下：

```python
def inference_network():
    images = fluid.layers.data(name='pixel', shape=[3, 32, 32], dtype='float32')
    predict = vgg_bn_drop(images)
    return predict
```

 + 问题拓展：在定义输出层的，其中参数`shape`的值应该是输出数据的形状，而不是大小。在V2版本的接口是设置大小，所以有些用户会误以为Fluid也是设置输入的大小，所以会导致错误。

 + 问题分析：PaddlePaddle团队对PaddlePaddle框架的优化速度较快，很多接口也采用更加容易理解的方法进行了重新，此时就有可能会造成与此前版本不一致的情况，请移步阅读PaddlePaddle Fluid版本的文档：

 http://www.paddlepaddle.org/documentation/docs/zh/1.1/beginners_guide/index.html


## `已审阅` 4.问题：图像预测部分预测的没有输出类别的名称

 + 问题描述：在使用预测模型预测图片的时候，输出的是一个整数，而不是图片的类别名称。


 + 报错信息：

```
infer results:  5
```

 + 问题复现：使用训练保存的预测模型预测经过预处理的图片，使用预测的结果经过`np.argmax(results[0])`进行处理，获取概率最大的值，最后输出的是一个整数值。复现代码如下：

```python
img = load_image('dog.png') 
results = inferencer.infer({'pixel': img})
print("infer results: ", np.argmax(results[0]))
```

 + 解决问题：在训练的时候，使用的是类别的标签，也就是一个整数，所以在预测的时候输出的也是类别的标签。需要输出类别的名称，还要根据标签对应的类别名称进行输出。

```python
img = load_image('dog.png')
results = inferencer.infer({'pixel': img})

label_list = [
    "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse",
    "ship", "truck"]
print("infer results: ", label_list[np.argmax(results[0])])
```

 + 问题拓展：PaddlePaddle的分类标签都是整数值，其中`int64`类型的标签非常常用。PaddlePaddle的分类标签不支持使用字符串的方式输入类别名称，所以都是整数作为类别的标签，而已必须要从0开始标记。


## `已审阅` 5.问题：在使用预测模型预测图片的时候出现in_dims[1]:32 != filter_dims[1] * groups:3错误

 + 问题描述：在使用预测模型预测图片的时候，图片也经过预处理，但是在执行预测的时候就保存，错误提示in_dims[1]:32 != filter_dims[1] * groups:3。


 + 报错信息：

```
<ipython-input-17-246b35b3c3dc> in infer(use_cuda, inference_program, params_dirname)
     27 
     28     ## inference
---> 29     results = inferencer.infer({'pixel': img})
     30 
     31     label_list = [

/usr/local/lib/python3.5/dist-packages/paddle/fluid/contrib/inferencer.py in infer(self, inputs, return_numpy)
    102             results = self.exe.run(feed=inputs,
    103                                    fetch_list=[self.predict_var.name],
--> 104                                    return_numpy=return_numpy)
    105 
    106         return results

/usr/local/lib/python3.5/dist-packages/paddle/fluid/executor.py in run(self, program, feed, fetch_list, feed_var_name, fetch_var_name, scope, return_numpy, use_program_cache)
    468 
    469         self._feed_data(program, feed, feed_var_name, scope)
--> 470         self.executor.run(program.desc, scope, 0, True, True)
    471         outs = self._fetch_data(fetch_list, fetch_var_name, scope)
    472         if return_numpy:

EnforceNotMet: Enforce failed. Expected in_dims[1] == filter_dims[1] * groups, but received in_dims[1]:32 != filter_dims[1] * groups:3.
The number of input channels should be equal to filter channels * groups. at [/paddle/paddle/fluid/operators/conv_op.cc:60]
PaddlePaddle Call Stacks: 
0       0x7ff682f386b6p paddle::platform::EnforceNotMet::EnforceNotMet(std::__exception_ptr::exception_ptr, char const*, int) + 486
1       0x7ff6837b01c6p paddle::operators::ConvOp::InferShape(paddle::framework::InferShapeContext*) const + 5622
```

 + 问题复现：使用训练CIFAR-10数据的模型来预测图片，图片进下面的代码进行预处理，先是对图片进行压缩统一大小，然后再对图片转换成向量，接着进行归一化，最后改变图片的维度。但是在执行预测的时候就会出现。错误代码如下：

```python
def load_image(file):
    im = Image.open(file)
    im = im.resize((32, 32), Image.ANTIALIAS)
    im = np.array(im).astype(np.float32)
    im = im / 255.0
    im = numpy.expand_dims(im, axis=0)
    return im
```

 + 解决问题：使用预测模型进行预测图片的时候，图片的通道顺序需要的是`(通道数, 宽, 高)`，但PIL打开方式是`(宽, 高, 通道数)`，所以才会导致出现的出现，需要加一行代码`im.transpose((2, 0, 1))`改名通道顺序。正确代码如下：

```python
def load_image(file):
    im = Image.open(file)
    im = im.resize((32, 32), Image.ANTIALIAS)
    im = np.array(im).astype(np.float32)
    im = im.transpose((2, 0, 1))  ## CHW
    im = im / 255.0
    im = numpy.expand_dims(im, axis=0)
    return im
```

 + 问题拓展：图片转化成向量，它的维度分别由通道数、图片的宽、图片的高。PaddlePaddle读取的方式是`(通道数, 宽, 高)`，所以无论是用什么工具对图片进行预处理，都要转换成这个顺序。PIL工具的`im.transpose((2, 0, 1))`函数，0表示图片的宽、1表示图片的高、2表示图片的通道数。


## `已审阅` 6.问题：使用预测模型预测图片时出现输出数据维度错误

 + 问题描述：在使用预测模型进行预测图片的时候，图片也经过预处理，但是在执行预测的时出现卷积层输入的形状不为4维或者5维的错误。


 + 报错信息：

```
/usr/local/lib/python3.5/dist-packages/paddle/fluid/contrib/inferencer.py in infer(self, inputs, return_numpy)
    102             results = self.exe.run(feed=inputs,
    103                                    fetch_list=[self.predict_var.name],
--> 104                                    return_numpy=return_numpy)
    105 
    106         return results

/usr/local/lib/python3.5/dist-packages/paddle/fluid/executor.py in run(self, program, feed, fetch_list, feed_var_name, fetch_var_name, scope, return_numpy, use_program_cache)
    468 
    469         self._feed_data(program, feed, feed_var_name, scope)
--> 470         self.executor.run(program.desc, scope, 0, True, True)
    471         outs = self._fetch_data(fetch_list, fetch_var_name, scope)
    472         if return_numpy:

EnforceNotMet: Conv intput should be 4-D or 5-D tensor. at [/paddle/paddle/fluid/operators/conv_op.cc:47]
PaddlePaddle Call Stacks: 
```

 + 问题复现：使用预测模型来预测图片，图片进下面的代码进行预处理，先是对图片进行压缩统一大小，然后再对图片转换成向量，接着改变图片的顺序，最后进行归一化。但是在执行预测的时候就会出现。错误代码如下：

```python
def load_image(file):
    im = Image.open(file)
    im = im.resize((32, 32), Image.ANTIALIAS)
    im = np.array(im).astype(np.float32)
    im = im.transpose((2, 0, 1))  ## CHW
    im = im / 255.0
    return im
```

 + 解决问题：PaddlePaddle最后读取预测的数据应该是有4个维度的，分别是Batch大小、图片的通道数、图片的宽、图片的高。错误的代码是因为没有对图片数据加一个Batch的维度。所以最后需要使用`numpy.expand_dims(im, axis=0)`需要进行修改图片的维度。

```python
def load_image(file):
    im = Image.open(file)
    im = im.resize((32, 32), Image.ANTIALIAS)
    im = np.array(im).astype(np.float32)
    im = im.transpose((2, 0, 1))  ## CHW
    im = im / 255.0
    im = numpy.expand_dims(im, axis=0)
    return im
```

 + 问题拓展：不仅在预测的是需要设置数据的Batch大小，在训练的是也需要设置图片的Batch的大小，在使用接口`fluid.layers.data`定义输入层时，`shape`参数设置也应该要有数据的Batch大小的，但是不用我们手动设置，PaddlePaddle可以自动设置。


## `已审阅` 7.问题：Fluid版的PaddlePaddle加载图像数据报错

+ 关键字：`加载图像数据` `Fluid版`

+ 问题描述：使用Fluid版的PaddlePaddle搭建图像分类模型，运行时报错，错误为`Aborted at 1520823806 (unix time) try "date -d @1520823806" if you are using GNU date ***
PC: @ 0x0 (unknown)`，自己观察报错代码段是图像数据处理那块，所以感觉应该是PaddlePaddle在加载图像时出现了错误。

+ 报错代码段：
```python
# 读入数据
img = Image.open(image_file)
img = img.resize((32, 32), Image.ANTIALIAS)
test_data = np.array(img).astype("float32")
test_data = test_data[np.newaxis, :] / 255
```

+ 报错输出：
```bash
*** Aborted at 1520823806 (unix time) try "date -d @1520823806" if you are using GNU date ***
PC: @                0x0 (unknown)
*** SIGSEGV (@0x50) received by PID 35 (TID 0x7ffc9cc40700) from PID 80; stack trace: ***
    @     0x7ffc9c7f1390 (unknown)
    @     0x7ffc9ca0c73c (unknown)
    @     0x7ffc9ca15851 (unknown)
    @     0x7ffc9ca10564 (unknown)
    @     0x7ffc9ca14da9 (unknown)
    @     0x7ffc9c55356d (unknown)
    @     0x7ffc9ca10564 (unknown)
    @     0x7ffc9c553624 __libc_dlopen_mode
    @     0x7ffc9c525a45 (unknown)
    @     0x7ffc9c7eea99 __pthread_once_slow
    @     0x7ffc9c525b64 backtrace
    @     0x7ffc92fb8519 paddle::platform::EnforceNotMet::EnforceNotMet()
    @     0x7ffc9354bb48 paddle::operators::ConvOp::InferShape()
    @     0x7ffc936adc63 paddle::framework::OperatorWithKernel::RunImpl()
    @     0x7ffc936ab4d8 paddle::framework::OperatorBase::Run()
    @     0x7ffc930567e2 paddle::framework::Executor::Run()
    @     0x7ffc92fd52d3 _ZZN8pybind1112cpp_function10initializeIZNS0_C4IvN6paddle9framework8ExecutorEIRKNS4_11ProgramDescEPNS4_5ScopeEibbEINS_4nameENS_9is_methodENS_7siblingEEEEMT0_FT_DpT1_EDpRKT2_EUlPS5_S8_SA_ibbE_vISO_S8_SA_ibbEISB_SC_SD_EEEvOSF_PFSE_SH_ESN_ENUlRNS_6detail13function_callEE1_4_FUNESV_
    @     0x7ffc92fd1fa4 pybind11::cpp_function::dispatcher()
    @           0x4cad00 PyEval_EvalFrameEx
    @           0x4c2705 PyEval_EvalCodeEx
    @           0x4ca088 PyEval_EvalFrameEx
    @           0x4c9d7f PyEval_EvalFrameEx
    @           0x4c2705 PyEval_EvalCodeEx
    @           0x4c24a9 PyEval_EvalCode
    @           0x4f19ef (unknown)
    @           0x4ec372 PyRun_FileExFlags
    @           0x4eaaf1 PyRun_SimpleFileExFlags
    @           0x49e208 Py_Main
    @     0x7ffc9c430830 __libc_start_main
    @           0x49da59 _start
    @                0x0 (unknown)
Segmentation fault (core dumped)
```

+ 复现方式：
读入自定义数据时，使用上面报错代码段代码，使得读入数据的格式为
```
[[[[0.654902   0.7764706  1.        ]
   [0.6666667  0.78431374 0.99607843]
   [0.6784314  0.79607844 1.        ]
   ...
   [0.6901961  0.8156863  1.        ]
   [0.6862745  0.80784315 1.        ]
   [0.6784314  0.8        1.        ]]
   ...
   [[0.63529414 0.75686276 0.9843137 ]
   [0.6431373  0.7647059  0.98039216]
   [0.65882355 0.7764706  0.9882353 ]
   ...
   [0.6745098  0.8        0.9843137 ]
   [0.6666667  0.7921569  0.9882353 ]
   [0.6627451  0.78039217 0.9882353 ]]]]
```
随后就会出现如上报错

+ 解决方案：
报错的原因是读入数据的格式有问题，从复现方式输出的数据格式可以知道，图像数据的宽为3，但通道channel却有多个，这是因为图像数据的读入使用了PIL中的Image模块，PIL打开图片存储顺序为H(高度)，W(宽度)，C(通道)，这与Fluid版PaddlePaddle要求的格式有差异，Fluid版PaddlePaddle要求输入数据里的channel在最前面，PaddlePaddle要求数据顺序为CHW，所以需要转换顺序，


    读入数据的代码段修改成如下形式：
    ```python
    img = Image.open(image_file)
    img = img.resize((32, 32), Image.ANTIALIAS)
    test_data = np.array(img).astype("float32")
    #transpose矩阵转置，高维数组需要使用一个由轴编号组成的元组
    test_data = np.transpose(test_data, (2, 0, 1))
    test_data = test_data[np.newaxis, :] / 255
```

+ 问题分析：
图像数据读入后其实就是一个三维矩阵，不同的读入方式会造成该矩阵的不同维度表示不同含义，此时如果没有理解矩阵中不同维度所代表的含义而直接将这些数据交由PaddlePaddle进行训练建模，就难以获得好的模型或者直接因使用错误而导致报错，这里的报错时因为使用了PIL的Image模块来读入图像，却没有注意PIL读入图像后不同维度所代表的含义，所以导致报错，使用numpy的transpose()方法对矩阵进行转置变换后，获得预期的矩阵则可。

+ 问题拓展：
在深度学习建立图像模型的过程中，通常都不可避免的要处理图像数据，为了避免类似问题，这里简单讨论一下常见的读入图像数据的方法以.

PIL读入图像数据，代码如下：

```python
from PIL import Image
img  = Image.open(imgpath) #读入
img.show() #展示
```

读入后，图像矩阵对应维度的意义为H(高度)、W(宽度)、C(通道)，且PIL.Image 数据是 uinit8 型的，范围是0-255

OpenCV读入图像数据，代码如下：

```python
import cv2
img = cv2.imread(imgpath)
```

通过OpenCV读入RGB图像后，其颜色通道顺序是B,G,R。

采用 matplotlib.image 读入图片数据，代码如下：

```python
import matplotlib.image as mpimg
lena = mpimg.imread('lena.png')
```

通过matplotlib的`mpimg`方法读入图像数据，其中的数据是 float32 型的，范围是0-1。


+ 问题研究：
对自己要训练的数据进行相应的预处理操作是建立深度学习模型常见的步骤，如果对自己要处理的数据结构不是特别熟悉，可以先尝试将数据预处理的代码单独写出来，然后用少量数据来验证这部分代码，除此之外，还可以使用numpy、cPickle等工具，单独的写数据处理的代码，将处理后的数据通过numpy、cPickle等工具持久化，即保存为二进制的文件，在使用PaddlePaddle训练模型时，直接以相应的方式读入这些二进制文件，此时读入的数据就是要处理数据相应的矩阵了，不必再关心数据预处理的方面的逻辑，而且也方便他人再次使用该数据进行模型的复现。



## `已审阅` 8.问题：'NoneType' object has no attribute 'imread'

+ 关键字： `NoneType` `no attribute`

+ 问题描述：通过PaddlePaddle构建了训练模型，使用了自定义的数据集对模型进行训练，出现`'NoneType' object has no attribute 'imread'`报错

+ 报错代码段：

```python
import paddle.v2 as paddle

img, label = sample
img = paddle.image.load_image(img) #报错
img = paddle.image.simple_transform(img, 70, self.imageSize, True)
return img.flatten().astype('float32'), label
```

+ 报错图片：

![](https://user-images.githubusercontent.com/37174503/37147359-a3f7fdda-2301-11e8-8b78-9bfd4b955345.png)

+ 报错输出：
```bash
File "/work/code/MyReader.py", line 14, in train_mapper
img = paddle.image.load_image(img) 
File "/usr/local/lib/python2.7/dist-packages/paddle/v2/image.py", line 159, in load_image im = cv2.imread(file, flag) 
AttributeError: 'NoneType' object has no attribute 'imread'
```

+ 复现方式：
旧版的PaddlePaddle使用paddle.v2的load_image方法读入图像数据时就会报出`AttributeError: 'NoneType' object has no attribute 'imread'`错误

+ 解决方案：
该问题是由opencv引起的，安装最新的opencv解决上述问题，具体操作为：

```bash
sudo apt-get install -y libopencv-dev #安装opencv

sudo pip install -U opencv-python #安装opencv的python库
```

+ 问题分析：
这个报错是由`img = paddle.image.load_image(img)`引起的，即paddle本身的方法引发了该错误，如果PaddlePaddle使用方式没错，而且又是新版的，那么最有可能的问题就是环境依赖问题。每一款PaddlePaddle在发布之前都会经专业的测试团队内测，确保框架本身的稳定性，虽然无法保证PaddlePaddle绝对稳定，但可以保证PaddlePaddle在大多数常规环境是可以正常运行的，所以遇到PaddlePaddle本身的错误，可以先考虑环境依赖问题，如果环境依赖没有问题，再把眼光发在PaddlePaddle本身。

PaddlePaddle的load_image方法用于加载图像，读入图像数据时使用了opencv，即该方法依赖opencv，而报错`'NoneType' object has no attribute 'imread'`则表明opencv缺少imread方法，尝试安装最新的opencv即可解决该问题。

+ 问题拓展：
OpenCV的全称是Open Source Computer Vision Library，是一个跨平台的计算机视觉库。OpenCV是由英特尔公司发起并参与开发，以BSD许可证授权发行，可以在商业和研究领域中免费使用。OpenCV可用于开发实时的图像处理、计算机视觉以及模式识别程序。该程序库也可以使用英特尔公司的IPP进行加速处理。

OpenCV主要分为2版与3版，3版的OpenCV相对2版有较大的修改，且不向后兼容，所以在使用时要注意区分。

OpenCV经过多年的发展在图像处理方面已经非常成熟，PaddlePaddle部分图像处理相关的方法抽象于OpenCV，在保证易用的同时保证了稳定。

目前PaddlePaddle的docker镜像中并没有安装OpenCV，在使用PaddlePaddle的Docker镜像进行图像方面的处理时需要先安装OpenCV。

+ 问题研究：
我们常见的RGB图像可以看成对应的三维矩阵，处理时可以使用数据处理方面的库来进行处理，比如numpy，这样对图像就可以进行很多操作了，比如常见的平滑处理就是将每个像素点的值等比压缩到0~1之间，而将图像读入成三维矩阵的工具就是OpenCV，当然不止OpenCV，但OpenCV算是最强大的图像处理工具，它可以实现对图像的各种变化，OpenCV底层使用了C++来编写，这保证了它在图像处理方面的速度。



## `已审阅` 9.问题：Fluid版本的PaddlePaddle如何在训练前加载此前训练好的模型？

+ 关键字：`Fluid版本` `预训练`

+ 问题描述：Fluid版本的PaddlePaddle在模型保存上与旧版的PaddlePaddle差异较大，文档中也没有对这方面的详细描述，所以存有疑惑，可以简单解释一下？

+ 背景知识：
预训练模型加载时深度学习中很常见的需求，特别是在图像识别、图像分割领域，这是因为图像处理方面，模型结构通常都比较深，如VGG、GoogleNet等都有比较深的结构，而且图像模型的训练数据集量往往也比较大，这就导致训练成本会比较高，如果每次都要使用ImageNet数据集来训练一个图像模型，是非常费力的，所以加载已有预训练模型就是一个不错的选择，因为图像数据集底层分布结构是相似的，所以对于其他图像网络而言，预训练的图像模式是可以借来使用的，即所谓迁移学习，将预训练模型的底层特征拿过来使用，接在新的图像模型结构上，对新的数据进行训练，这样就大大减少了训练数据量。

+ 问题讨论：
在Fluid中，模型由一个或多个program来表示，program包含了block，block中包含了op和variable。

在保存模型时，program--block--{op, variable}这一系列的拓扑结构会被保存成一个文件，variable具体的值会被保存在其他文件里。

下面来讨论一下如何保存模型。

保存一个训练中的模型，你需要做两步：

1.保存program：

```python
with open(filename, "wb") as f:
        f.write(program.desc.serialize_to_string())
```

2.保存program中各个persistable的varibale的值：
```python
fluid.io.save_persistables(executor, dirname, main_program)
```

其中main_program就是刚刚被保存的那个program。

接着来看如何恢复模型：

同样，从保存的文件中恢复，也需要两步：

1.恢复program:

```python
with open(filename, "rb") as f:
        program_desc_str = f.read()
program = Program.parse_from_string(program_desc_str)
```

2.恢复persistable的variable：

```
load_persistables(executor, dirname, main_program)
```

同样，main_program应当是刚刚恢复出来的那个program。

然后，program就可以放到exe.run()里面去做正常的训练了。

+ 问题分析：
要使用加载预训练模型，就要弄明白Fluid版本的PaddlePaddle如何加载以及保存模型，通过问题讨论已经知道了，目前Fluid版本的PaddlePaddle只能将加载以及保存分开来执行。

但通过这种方法加载的模型通常用来预测，即训练好了一个模型后，然后将训练好的模型加载进来，并用来进行执行相关的预测任务，同样，这种方式也可以用来加载模型，然后继续训练，目前Fluid版本的PaddlePaddle还没有提供相应的接口来直接实现加载模型然后继续训练的功能，只能通过上面的方式将两种操作分开来执行。

+ 问题拓展：
在官方示例代码中，保存模型具体操作为：

```python
fluid.io.save_inference_model(model_path, ['image'], [out], exe)
```

即使用了`save_inference_model`方法来对模型进行保存操作，那与之前讨论的模型加载和模型保存操作有什么关系呢？

save_inference_model内部也是调用了问题讨论中提及的那两个接口。另外，在调用这两个接口之前，save_inference_model还会做一些模型裁剪的工作，裁减掉backward的部分。backward部分对于inference来说不需要，但对模型的继续训练是需要的。

但save_inference_model这个封装并不适用与你的需求，即“保存模型，下次恢复出来继续训练”这个需求，目前Fluid版本的PaddlePaddle还没有高层的API能够实现，这个接口在未来应该会很快实现，相关的讨论如下：

https://github.com/PaddlePaddle/Paddle/issues/10248


## `已审阅` 10.问题：ValueError: var img not in this block

+ 关键字：`ValueError` `var`

+ 问题描述：
1、将caffe VGG16模型转化为fluid要求的类型，其中模型的加载和存储使用的用法来自：https://github.com/PaddlePaddle/Paddle/issues/8973 <br>
2、输入数据类型：images = fluid.layers.data(name='img', shape=data_shape, dtype='float32')，data_shape = [3, 300, 300]；
加载模型：predict = vgg16_raw(images)，然后run：
loss = exe.run( #fluid.default_main_program(), main_program, feed={"img": img_data, "label": y_data}, fetch_list=[avg_cost])

+ 报错输出：
```bash
/home/yczhao/anaconda2/lib/python2.7/site-packages/paddle/fluid/average.py:42: Warning: The WeightedAverage is deprecated, please use fluid.metrics.Accuracy instead.
(self.class.name), Warning)
Traceback (most recent call last):
File "vgg16.py", line 301, in 
main()
File "vgg16.py", line 263, in main
fetch_list=[avg_cost])
File "/home/yczhao/anaconda2/lib/python2.7/site-packages/paddle/fluid/executor.py", line 335, in run
fetch_var_name=fetch_var_name)
File "/home/yczhao/anaconda2/lib/python2.7/site-packages/paddle/fluid/executor.py", line 234, in _add_feed_fetch_ops
out = global_block.var(name)
File "/home/yczhao/anaconda2/lib/python2.7/site-packages/paddle/fluid/framework.py", line 724, in var
raise ValueError("var %s not in this block" % name)
ValueError: var img not in this block
```

+ 相关代码段：

```python
#读入已有模型
#restore from trained model
with open(vgg16_filename, "rb") as f:
    program_desc_str = f.read()
main_program = Program.parse_from_string(program_desc_str)
load_persistables(exe, vgg16_dirname, main_program)

#构建模型
images = fluid.layers.data(name='img', shape=data_shape, dtype='float32')
label = fluid.layers.data(name='label', shape=[300*300], dtype='float32')

# Train program
predict = vgg16_raw(images)

cost = crossentropy_seg(predict, label)
avg_cost = fluid.layers.mean(x=cost)

# Optimization
optimizer = fluid.optimizer.Adam(learning_rate=args.learning_rate)
opts = optimizer.minimize(avg_cost)

loss = exe.run(
#fluid.default_main_program(),
main_program,
feed={"img": img_data,
      "label": y_data},
fetch_list=[avg_cost])
```

+ 问题分析：
从报错`ValueError: var img not in this block`可知，报错的原因是因为模块中没有img这个变量，即代码中调用了模块中的img变量，而模型中本身是不存在的，所以造成了这个错误，是用法问题，看到报错的相关代码，有两大块，一块是载入模型，另一块是写了一个新的模型结构，然后通过exe.run()方法来执行，这里就存在一个逻辑问题，即你载入了以存在的模型，那么就是使用训练该模型了，新构建的模型结构没有被使用，此时你又使用新模型的变量`images`，它的name为img，就可能会造成`var img not in this block`，因为导入使用的旧模型不一定有名为img的变量。

+ 解决方法：
理清思路，确定自己的具体任务，造成这个问题的原因是使用了加载的模型，而该模型中没有名为img的变量，但看到相关代码段中构建新模型的代码，其中创建了名为img的变量，即使用者以为自己可以使用新构建的模型，但又加载了已存在的模型，理清楚思路，如果要使用img变量，就放弃加载模型，如果可以不使用，那就修改训练模型的代码则可。




