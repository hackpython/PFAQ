# 图像分类

## 背景介绍
图像相比文字能够提供更加生动、容易理解及更具艺术感的信息，是人们转递与交换信息的重要来源。在本教程中，我们专注于图像识别领域的一个重要问题，即图像分类。

图像分类是根据图像的语义信息将不同类别图像区分开来，是计算机视觉中重要的基本问题，也是图像检测、图像分割、物体跟踪、行为分析等其他高层视觉任务的基础。图像分类在很多领域有广泛应用，包括安防领域的人脸识别和智能视频分析等，交通领域的交通场景识别，互联网领域基于内容的图像检索和相册自动归类，医学领域的图像识别等。

基于深度学习的图像分类方法，可以通过有监督或无监督的方式学习层次化的特征描述，从而取代了手工设计或选择图像特征的工作。深度学习模型中的卷积神经网络(Convolution Neural Network, CNN)近年来在图像领域取得了惊人的成绩，CNN直接利用图像像素信息作为输入，最大程度上保留了输入图像的所有信息，通过卷积操作进行特征的提取和高层抽象，模型输出直接是图像识别的结果。这种基于"输入-输出"直接端到端的学习方法取得了非常好的效果，得到了广泛的应用。

## 图像分类PaddlePaddle-fluid版代码：
https://github.com/PaddlePaddle/book/tree/develop/03.image_classification

## 1.问题：'NoneType' object has no attribute 'imread'

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


## 2.问题：from __future__ imports must occur at the beginning of the file

+ 关键字： `__future__`

+ 问题描述：

+ 报错代码段：
```python
import paddle
import paddle.fluid as fluid
import sys
from __future__ import print_function #报错
```

+ 报错输出：
```bash
7de5cf6102fb:python -u /opt/project/3_image_classification.py
  File "/opt/project/3_image_classification.py", line 10
    from __future__ import print_function
SyntaxError: from __future__ imports must occur at the beginning of the file
```

+ 复现方式：
PaddlePaddle实现VGG模型，用于图像分类任务，在代码一开头导入需要的第三方库时，使用报错代码段中的导入方式，出现如下错误。

+ 解决方式：
更改依赖库导入顺序，将`__future__`的导入放在一开头，修改后如下：
```python
from __future__ import print_function
import paddle
import paddle.fluid as fluid
import sys
```

+ 问题分析：
`__future__`的声明必须出现在模块顶部附近。在`__future__`声明之前可以出现的唯一行是模块docstring（如果有的话）、评论、空白行和其他未来的陈述。

这是python文档中的要求，在python代码中使用`__future__`要注意这个要求，该要求具体可参考：
https://docs.python.org/2/reference/simple_stmts.html#future

+ 问题拓展：
`__future__`模块的作用主要是将python新版本中的部分功能在当前版本中使用，这是一种比较稳妥的做法，比如想在python2.7中使用python3新字符串的特性，就可以使用`__future__`，具体导入`from __future__ import unicode_literals`



## 3.问题：Fluid版的PaddlePaddle加载图像数据报错

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


## 4.问题：ZeroDivisionError: float division by zero

+ 关键字：`ZeroDivisionError` 

+ 问题描述：使用PaddlePaddle编写图像分类模型，使用了paddle.trainer.SGD的test方法，传入了相应的数据，报`ZeroDivisionError: float division by zero`错误。

+ 报错代码段：

```python
import os
import sys
import paddle.v2 as paddle
from MyReader import MyReader
from vgg import vgg_bn_drop

def start_trainer(self, trainer, num_passes, save_parameters_name, trainer_reader, test_reader):
    # 获得数据
    reader = paddle.batch(reader=paddle.reader.shuffle(reader=trainer_reader,buf_size=50000),batch_size=128)
    .
    .
    .
    # 定义训练事件
    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id % 100 == 0:
                print "\nPass %d, Batch %d, Cost %f, Error %s" % (
                    event.pass_id, event.batch_id, event.cost, event.metrics['classification_error_evaluator'])
            else:
                sys.stdout.write('.')
                sys.stdout.flush()

        # 每一轮训练完成之后
        if isinstance(event, paddle.event.EndPass):
            # 保存训练好的参数
            with open(save_parameters_name, 'w') as f:
                trainer.save_parameter_to_tar(f)

            # 测试准确率  报错
            result = trainer.test(reader=paddle.batch(reader=test_reader,
                                                      batch_size=128),feeding=feeding)
            print "\nTest with Pass %d, Classification_Error %s" % (
            event.pass_id, result.metrics['classification_error_evaluator'])

# 报错
paddleUtil.start_trainer(trainer=trainer, num_passes=100, save_parameters_name=parameters_path,trainer_reader=trainer_reader, test_reader=test_reader)

```

+ 报错输出：
```bash
Pass 0, Batch 0, Cost 1.256652, Error 0.6953125 Traceback (most recent call last): 
File "train.py", line 150, in trainer_reader=trainer_reader, test_reader=test_reader) 
File "train.py", line 126, in start_trainer feeding=feeding) 
File "/usr/local/lib/python2.7/dist-packages/paddle/v2/trainer.py", line 214, in train gm=self.__gradient_machine__)) 
File "train.py", line 112, in event_handler feeding=feeding) 
File "/usr/local/lib/python2.7/dist-packages/paddle/v2/trainer.py", line 247, in test evaluator=evaluator, cost=total_cost / num_samples) ZeroDivisionError: float division by zero   
```

+ 复现方法：
使用v2版的paddlepaddle实现上述代码，出现`ZeroDivisionError: float division by zero `错误

+ 解决方法：
旧版的PaddlePaddle在某些的优化并没有特别完善，ZeroDivisionError错误出现的原因就是浮点数除以了0，即分数分母为0了，要解决这个问题，最简单的方式就是使用新版的PaddlePaddle，它对这个问题进行了相应的处理，不会报出相应的错误，当然自己加上相应的逻辑代码对传入的数据进行判断，如果为0，则不再将数据传入也可，建议使用新版的PaddlePaddle。

+ 问题分析：
数学上，分数的分母是不能为0的，实践在编程中也一样，这里报出的错误为`ZeroDivisionError: float division by zero`，即浮点除以了0，提升的很明显。出现这个错误的原因是传入给PaddlePaddle训练的数据中存在0，当PaddlePaddle进行自动求导计算梯度等参数时，除以了该值。在新版的PaddlePaddle对该问题做了相应的优化，避免的这种情况报错导致训练进程崩溃的情况，当然如果自己能对传入的数据进行简单的清洗，如对0值加一个极小的值，避免分母为0的情况，这样就可以避免`ZeroDivisionError`报错。

+ 问题拓展：
在使用PaddlePaddle编写模型时，最好每训练一定轮数后就保存此时训练好的模型，这样就可以减少遇到问题时训练进程崩溃所造成的损失，当解决了当前的问题后，可以直接载入此前训练好的模型继续训练，而不必再重头训练，这种保险的做法适用于任何模型的训练。

+ 问题研究：
在深度学习的一些常见算法中，对分母为0的情况也做了简单处理，如给任意分数的分母都加一个极小的值，确保其分母不为0，一个具体的例子就是Batch Normalization，从它的论文中可以看出，在具体的normalize操作时，分母加了一个极小值，如下图
![](https://raw.githubusercontent.com/ayuLiao/images/master/%E5%9B%BE%E5%83%8F%E5%88%86%E7%B1%BB_BN%E6%9E%81%E5%B0%8F%E5%80%BC.png)

我们在编写具体的算法时，也可以使用类似的方式来操作，避免分母为0的错误发生。


## 5.问题：在图片分类中,使用自定义数据在vgg模型训练,如果加上了BN层,反而没办法收敛 

+ 关键词：`网络无法收敛`

+ 问题描述：在自定义数据图片中,如果使用安装book中的第三章图片分类,如果按照这里网络,使用后果BN层,错误率一直不变的；trainer_cost是在收敛的,但是test_cost不但没降,反而上升了,如图:
trainer_cost一直在收敛
![](https://user-images.githubusercontent.com/26297768/34917234-94dc9e20-f97e-11e7-9a76-e64f02f9b037.png)

test_cost却一直在上升
![](https://user-images.githubusercontent.com/26297768/34917236-9a3e35fe-f97e-11e7-8422-ec5e1b484a57.png)

如果使用的是没加上BN层，错误率是会一直下降的

+ 背景知识：
批标准化（Batch Normalization ）简称BN算法，是为了克服神经网络层数加深导致难以训练而诞生的一个算法。在神经网络中，每一层的输入在经过层内操作之后必然会导致与原来对应的输入信号分布不同,,并且前层神经网络的增加会被后面的神经网络不对的累积放大，BN算法就看要解决这个问题，它可以用来规范化某些层或者所有层的输入，从而固定每层输入信号的均值与方差。

Batch Normalization直观如下图：
![](https://pic4.zhimg.com/80/v2-cd6a1087598ee5a2ae24d1a86a6b2190_hd.jpg)
（图片来源：YJango）

具体算法为:
![](https://img-blog.csdn.net/20171221183555258?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvSk5pbmdXZWk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

+ 相关代码段：
```python
'''
VGG模型
'''
def conv_block(ipt, num_filter, groups, dropouts, num_channels=None):
        return paddle.networks.img_conv_group(
            input=ipt,
            num_channels=num_channels,
            pool_size=2,
            pool_stride=2,
            conv_num_filter=[num_filter] * groups,
            conv_filter_size=3,
            conv_act=paddle.activation.Relu(),
            conv_with_batchnorm=True,
            conv_batchnorm_drop_rate=dropouts,
            pool_type=paddle.pooling.Max())
    # 定义一个VGG16的卷积组
    conv1 = conv_block(input, 64, 2, [0.3, 0], 3)
    conv2 = conv_block(conv1, 128, 2, [0.4, 0])
    conv3 = conv_block(conv2, 256, 3, [0.4, 0.4, 0])
    conv4 = conv_block(conv3, 512, 3, [0.4, 0.4, 0])
    conv5 = conv_block(conv4, 512, 3, [0.4, 0.4, 0])
    # 定义第一个drop层
    drop = paddle.layer.dropout(input=conv5, dropout_rate=0.5)
    # 定义第一层全连接层
    fc1 = paddle.layer.fc(input=drop, size=512, act=paddle.activation.Linear())
    # 定义BN层
    bn = paddle.layer.batch_norm(input=fc1,
                                 act=paddle.activation.Relu(),
                                 layer_attr=paddle.attr.Extra(drop_rate=0.5))
    # 定义第二层全连接层
    fc2 = paddle.layer.fc(input=bn, size=512, act=paddle.activation.Linear())
    # 获取全连接输出，获得分类器
    predict = paddle.layer.fc(input=fc2,
                          size=class_dim,
                          act=paddle.activation.Softmax())
    return predict
```

+ 复现方式：
通过旧版的PaddlePaddle构建模型，然后使用比较小的自定数据集来训练模型，在通过PaddlePaddle使用BN层的情况下，通过训练集数据训练出了模型，该模型用于测试集数据进行测试，出现损失无法下降的现象

+ 解决方法：
1.增加数据量，不使用小型数据进行训练
2.如果无法增大数据量，则关闭Dropout，不使用Droput

+ 问题分析：
网络无法收敛的原因有很多种可能，如训练数据集太小，导致网络模型难以学习到数据的特征分布，导致在使用测试数据进行测试，难以收敛，或者是训练次数不足，同样会导致网络模型难以学习到数据的特征分布，导致模型再测试时效果不好，为了判断是否是这种原因，常见的做法就是使用一个比较常见的数据集来训练，保证足够的次数，比如使用cifar-10来训练数据，暂时放弃自定义的图像先，看看在使用了BN层的情况下，是否可以收敛，如果可以正常收敛，那么可能就是自定数据集或训练次数的问题。

    但描述中有一句很重要，即`如果使用的是没加上BN层，使用这个错误率是会一直下降的`，也就是没有使用BN层，自定义数据集在测试时，其损失是正常下降的，也就是说，没有BN时，训练没有问题，造成这个问题也有多种可能，如网络结构的一些具体参数的问题或训练数据的概率分布与测试数据的概率分布差异较大，导致BN层计算出的方差与均值差异都比较大，导致模型在测试数据上难以收敛。

    还有一种可能就是，因为训练时，数据量较小，同时又开启了Dropout，这就会使得特征丢失更多，从而无法收敛，当数据量比较小时，神经网络学习到的数据特征信息本来就不多，此时又使用了Dropout随机丢弃特征信息，从而导致模型其实没学到什么，造成在测试时，损失无法下降。针对本问题，就是这个原因

+ 问题拓展：
Dropout是指在深度学习网络的训练过程中，对于神经网络单元，按照一定的概率将其暂时从网络中丢弃。注意是暂时，对于随机梯度下降来说，由于是随机丢弃，故而每一个mini-batch都在训练不同的网络，这种方法可以有效的防止网络过拟合，不过对于小的训练数据而已，往往是没有必要的。

+ 问题研究：
该问题其实有很多中原因，每种原因都会造成训练数据时，损失可以正常降低，而测试数据时损失却无法降低，关键的一个描述在于`如果使用的是没加上BN层，错误率是会一直下降的`，即没有BN时是正常的，看到该问题的相关代码中，我们封装了conv_block方法来实现不同的层，在该方法中，使用了paddle.layer.batch_norm方法，该方法不止会使用使用了BatchNormalization还会开启使用dropout，当我们关闭BN层时，即不使用paddle.layer.batch_norm方法时，其实也就关闭了dropout，这样训练出的模型使用在测试数据上一样可以正常收敛，如果想开启BatchNormalization，却要关闭dropout，就将`drop_rate`设置为0，不使用dropout。

## 6.问题：Fluid版本的PaddlePaddle如何在训练前加载此前训练好的模型？

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

## 7.问题：paddlepaddle在进行图像训练的时候出现内存不足进程被杀死的问题

+ 关键字：`图像模型` `内存不足` `训练图像`

+ 问题描述：使用的train.py的源码基本和官方示例一致，网络是官方教程中图像分类的vgg网络，但是使用了自己的图像数据集，每张大约200k大小，batch_size=50，learning_rate大约比官方使用的cifar数据集时设定的低一个数量级，为什么还是会把内存跑完？占用内存是跟哪些参数有关系的呢？

+ 报错截图：

![](https://user-images.githubusercontent.com/35390572/34934616-129edf8e-fa16-11e7-8ca5-e0b75077d091.PNG)

+ 相关代码：
```python
datadim = 3*224*224 #图像大小
classdim = 5
```

+ 复现方式：
使用PaddlePaddle构建了图像分类模型，将图像大小设置为一个较大的值，开始训练后，运行一段时间后，程序崩溃。

+ 解决方法：
将图像大小设置的小一些，图像分类模型没有必要使用特别大的图像，如果要训练比较大的图像数据，就需要使用比较好的硬件，因为随着训练的进行，模型中的数据会一直写入到内存中，此时又使用了比较大图像，那么就会消耗更多的内存。具体解决方法为：

```python
datadim = 3*32*32
classdim = 5
```
展开而言，对于内存不足导致训练失败的情况，常见的解决方法有下面几个方面
1.减少输入图像的尺寸
2.减少batch，减少每次的输入图像数量
4.转换数据格式、渐近的加载数据
5.使用关系型数据库
6.购买显存更大的显卡

具体而言，可参考如下链接：https://machinelearningmastery.com/large-data-files-machine-learning/
中译文：https://www.leiphone.com/news/201705/sghfB2wSub6W01Jy.html


+ 问题分析：
模型在训练时需要暂时相应的内存，占用内存大小取决于喂养给模型的数据大小，想要提供模型的准备率盲目的增大训练数据量并不可取，增大训练数据量虽可以提高模型的准确率，但与模型本身的结构也是有很大的关系，即模型本身学习能力不是特别强，给再多数据量也不会有特别大的提高。

    使用大的训练数据量就需要注意内存开销问题，过大的内存开销可能会使模型在训练过程中崩溃，PaddlePaddle目前还有没做这方面的处理，所以需要用户自己注意，你需要在模型训练一定轮数后保存模型，如每100轮保存一次图像，避免模型崩溃导致前功尽弃。PaddlePaddle后期会添加相应的功能限制模型最大可使用内存，动态调整，避免模型因内存不足而崩溃。

+ 问题拓展：
一张RGB图像，其长宽为50\*500，数据类型为单精度浮点型，假设单精度会占用4B的显存，即32bit，那么这张图像所占的显存大小就为500*500*3*4B = 3M，一张图像会500*500的RGB图像会占用大约3M的显存空间，似乎不大，但训练神经网络模型都需要比较大的数据量，那么此时显存的占用率就会比较高。

    喂养给模型训练的数据虽然会占用一定的内存，但跟重要的原因是神经网络模型的大量中间变量以及中间参数，这些参数会占用大量的显存，所以模型的复杂度也是造成内存崩溃的一个主要原因，占用显存的层一般为卷积层、全连接层、BN层等，可以适当简化模型达到节省内存的目的。


+ 问题研究：
训练模型时，出现不可测意外的可能性较高，训练模型时做好相应的保存工作是很有必要的，当发生意外中断时，下次运行就可以接着此前保存的模型继续运行。

    内存优化是所有深度学习框架都面临的问题，PaddlePaddle也在往更节省内存努力。PaddlePaddle内存方面的内容可以参看 http://www.paddlepaddle.org/documentation/docs/zh/0.12.0/faq/local/index_cn.html#id11

## 8.问题：ValueError: var img not in this block

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


## 9.问题：ValueError: insecure string pickle

+ 关键字：`ValueError` `pickle`

+ 问题描述：PaddlePaddle构建模型训练自定义的图像分类数据集，出现ValueError: insecure string pickle

+ 报错输出：
```bash
Traceback (most recent call last):
File "/home/showme/python/two.py", line 97, in 
feeding=feeding)
File "/usr/local/lib/python2.7/dist-packages/paddle/v2/trainer.py", line 146, in train
for batch_id, data_batch in enumerate(reader()):
File "/usr/local/lib/python2.7/dist-packages/paddle/v2/minibatch.py", line 33, in batch_reader
for instance in r:
File "/usr/local/lib/python2.7/dist-packages/paddle/v2/reader/decorator.py", line 67, in data_reader
for e in reader():
File "/usr/local/lib/python2.7/dist-packages/paddle/v2/dataset/cifar.py", line 60, in reader
batch = cPickle.load(f.extractfile(name))
ValueError: insecure string pickle
```

+ 复现方式：
安装cPickle，然后使用cPickle打包数据，在不关闭的情况下，解压打包的数据，出现`ValueError: insecure string pickle`错误，具体代码如下：

```python
>>> out = open('xxx.dmp', 'w')
>>> cPickle.dump(d, out)
>>> k = cPickle.load(open('xxx.dmp', 'r'))
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ValueError: insecure string pickle
```

+ 解决方法：
通过cPickle读入数据后，要记得将读入数据的通道关闭再通过cPickle去读取打包后的数据，具体代码如下：

```python
>>> out = open('xxx.dmp', 'w')
>>> cPickle.dump(d, out)
>>> out.close() # 保存后，先关闭，再加载
>>> k = cPickle.load(open('xxx.dmp', 'r'))
````

+ 问题分析：
`ValueError: insecure string pickle`问题的解决方法比较简单，造成的原因就是cPickler存储数据后没有关闭存储的线程，而是直接读取数据，因为此时数据文件还被cPickler的存储线程所占用，此时使用cPickler读入数据就会导致`insecure string pickle`，避免该问题的方式就是读入数据前关闭存储数据的线程。

+ 问题拓展：
Python的序列化是指把变量从内存中变为可以储存/传输的数据/文件的过程，通常我们会将序列化后的内容写入磁盘或则通过网络传输到其他设备上，反过来，将序列化后的内容从新加载，读入内存，称为反序列化，Python中有两个模块用来做序列化与反序列化的事情，分布是cPickle与pickle，前者使用c来实现，速度回快一些，所以大多数情况下都使用cPickle。


# 图像分类

## 背景介绍
图像相比文字能够提供更加生动、容易理解及更具艺术感的信息，是人们转递与交换信息的重要来源。在本教程中，我们专注于图像识别领域的一个重要问题，即图像分类。

图像分类是根据图像的语义信息将不同类别图像区分开来，是计算机视觉中重要的基本问题，也是图像检测、图像分割、物体跟踪、行为分析等其他高层视觉任务的基础。图像分类在很多领域有广泛应用，包括安防领域的人脸识别和智能视频分析等，交通领域的交通场景识别，互联网领域基于内容的图像检索和相册自动归类，医学领域的图像识别等。

基于深度学习的图像分类方法，可以通过有监督或无监督的方式学习层次化的特征描述，从而取代了手工设计或选择图像特征的工作。深度学习模型中的卷积神经网络(Convolution Neural Network, CNN)近年来在图像领域取得了惊人的成绩，CNN直接利用图像像素信息作为输入，最大程度上保留了输入图像的所有信息，通过卷积操作进行特征的提取和高层抽象，模型输出直接是图像识别的结果。这种基于"输入-输出"直接端到端的学习方法取得了非常好的效果，得到了广泛的应用。

## 图像分类PaddlePaddle-fluid版代码：
https://github.com/PaddlePaddle/book/tree/develop/03.image_classification
