# 识别数字

## 背景介绍

机器学习（或深度学习）入门的"Hello World"，即识别MNIST数据集（手写数据集）。手写识别属于典型的图像分类问题，比较简单，同时MNIST数据集也很完备。MNIST数据集作为一个简单的计算机视觉数据集，包含一系列如图1所示的手写数字图片和对应的标签。图片是28x28的像素矩阵，标签则对应着0~9的10个数字。每张图片都经过了大小归一化和居中处理。有很多算法在MNIST上进行实验

## 数字识别PaddlePaddle-fluid版代码：

https://github.com/PaddlePaddle/book/tree/develop/02.recognize_digits

PaddlePaddle文档中的内容目前依旧是PaddlePaddle-v2版本，建议使用Fluid版本来编写数字识别模型

## 1.问题：libmkldnn.so.0: cannot open shared object file: No such file or directory

+ 问题描述：安装PaddlePaddle，使用PaddlePaddle编写识别MNIST数据集模型后，运行报`ImportError: libmkldnn.so.0: cannot open shared object file: No such file or directory`。

+ 报错代码段：

```python
import paddle.fluid as fluid
```

+ 报错输出：

```python
~/paddle/benchmark-master/fluid# python mnist.py --device GPU
Traceback (most recent call last):
File ""mnist.py"", line 10, in 
import paddle.fluid as fluid
File ""/usr/local/lib/python2.7/dist-packages/paddle/fluid/init.py"", line 17, in 
import framework
File ""/usr/local/lib/python2.7/dist-packages/paddle/fluid/framework.py"", line 22, in 
from . import core
ImportError: libmkldnn.so.0: cannot open shared object file: No such file or directory
```

+ 解决方法1：
出现这种问题的原因是在导入 paddle.fluid 时需要加载 `libmkldnn.so` 和 `libmklml_intel.so`， 但是系统没有找到该文件。一般通过pip安装PaddlePaddle时会将 libmkldnn.so 和 libmklml_intel.so 拷贝到 /usr/local/lib 路径下，所以解决办法是将该路径加到 LD_LIBRARY_PATH 环境变量下， 即： export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH 。

注意：如果是在虚拟环境中安装PaddlePaddle， libmkldnn.so 和 libmklml_intel.so 可能不在 /usr/local/lib 路径下。

+ 解决方法2：
重新安装Docker版本的PaddlePaddle，可以避免这种问题，命令如下：`docker pull hub.baidubce.com/paddlepaddle/paddle:latest`

+ 问题分析：
从报错输出就可以判断出，报错的原因与libmkldnn.so有关，知道了问题原因，接着就要确定该问题与PaddlePaddle的关系，具体而言就是libmkldnn.so与PaddlePaddle有什么关系，弄明白这个问题就知道报错的具体原因。通过解决方法1的分析可以知道pip安装PaddlePaddle时会将 libmkldnn.so 和 libmklml_intel.so 拷贝到 /usr/local/lib 路径下，这就是libmkldnn.so与PaddlePaddle的关系，那么出现`No such file or directory`的原因很有可能就是/usr/local/lib 路径下没有相应的文件，那么解决方案就是讲相应文件的路径添加到环境变量中，即： export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH，让PaddlePaddle可以在/usr/local/lib 路径下找到相应的文件

+ 问题拓展：
PaddlePaddle支持 MKL 和 OpenBlAS 两种BLAS库。默认使用MKL。如果使用MKL并且机器含有AVX2指令集， 还会下载MKL-DNN数学库，如果关闭MKL，则会使用OpenBLAS作为BLAS库。
	MKL-DNN是Intel 最近发布了开源的深度学习软件包 ，来替换之前的 MKLML。MKL-DNN 专门优化了一系列深度学习里的操作符。

	MKL-DNN 优化的操作符多用于 CNN 模型，其中包括 Convolution, Inner Product, Pooling, Batch Normalization, Activation。Intel 团队在不久之后会加入 RNN cell 和 LSTM cell 来提升 RNN 模型在 CPU 上的性能。为了得到更好的性能，MKL-DNN 使用了定制的数据格式，这也使得与 MXNet 的集成变得复杂，因为 MXNet 里自带的操作符不能够理解 MKL-DNN 的定制的数据格式。

	PaddlePaddle在使用MKL-DNN时，需要安装MKL-DNN的必要组件，这些组件是开发面向英特尔 MKL-DNN 的应用所必需的，如下：

	共享库 (/usr/local/lib)：

	libiomp5.so
	libmkldnn.so
	libmklml_intel.so
	标头文件 (/usr/local/include)：

	mkldnn.h
	mkldnn.hpp
	mkldnn_types.h
	文档 (/usr/local/share/doc/mkldnn)：

	英特尔许可和版权声明
	构成 HTML 文档的各种文件（在 /reference/html之下）


+ 问题研究：
xxx.so.0: cannot open shared object file: No such file or directory原因一般有两个, 一个是操作系统里确实没有包含该共享库(lib*.so.*文件)或者共享库版本不对, 遇到这种情况那就去网上下载并安装上即可，另外一个原因就是已经安装了该共享库, 但执行需要调用该共享库的程序的时候, 程序按照默认共享库路径找不到该共享库文件。

	当然，最简单的解决方法，就会使用封装好的Docker镜像。

## 2.问题：docker镜像无法联网下载数据文件

+ 问题描述：win10安装PaddlePaddle的docker镜像之后，运行手写数字识别的模型，无法联网下载数据文件该如何解决？

+ 报错输出：

```python
Cache file /root/.cache/paddle/dataset/mnist/train-labels-idx1-ubyte.gz not found, downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
Traceback (most recent call last):
File "train_with_paddle.py", line 117, in <module>
main()
File "train_with_paddle.py", line 87, in main
paddle.reader.shuffle(paddle.dataset.mnist.train(), buf_size=8192),
File "/usr/lib64/python2.7/site-packages/paddle/v2/dataset/mnist.py", line 91, in train
TRAIN_LABEL_MD5), 100)
File "/usr/lib64/python2.7/site-packages/paddle/v2/dataset/common.py", line 85, in download
r = requests.get(url, stream=True)
File "/usr/lib/python2.7/site-packages/requests/api.py", line 71, in get
return request('get', url, params=params, **kwargs)
File "/usr/lib/python2.7/site-packages/requests/api.py", line 57, in request
return session.request(method=method, url=url, **kwargs)
File "/usr/lib/python2.7/site-packages/requests/sessions.py", line 475, in request
resp = self.send(prep, **send_kwargs)
File "/usr/lib/python2.7/site-packages/requests/sessions.py", line 585, in send
r = adapter.send(request, **kwargs)
File "/usr/lib/python2.7/site-packages/requests/adapters.py", line 442, in send
raise ConnectionError(e, request=request)
requests.exceptions.ConnectionError: HTTPConnectionPool(host='yann.lecun.com', port=80): Max retries exceeded with url: /exdb/mnist/train-labels-idx1-ubyte.gz (Caused by NewConnectionError('<requests.packages.urllib3.connection.HTTPConnection object at 0x7f5d86ae1550>: Failed to establish a new connection: [Errno -2] Name or service not known',))
```

+ 解决方案1：如果本机无法联网，请从一台能联网的机器上下载数据集，然后拷贝到docker镜像的/root/.cache/paddle/dataset/mnist/ 路径中去。

+ 解决方案2：电脑本身是可以连网，docker镜像里没法连网，请尝试在docker镜像命令行环境尝试`wget www.baidu.com`，如果输出为下：

```bash
root@c47b61dfeb66 /]# wget www.baidu.com
--2018-09-04 11:21:14-- http://www.baidu.com/
Resolving www.baidu.com (www.baidu.com)... failed: Name or service not known.
wget: unable to resolve host address 'www.baidu.com'
```

则说明docker内的DNS解析有问题，您可以执行一下命令 echo "nameserver 8.8.8.8" >>  /etc/resolv.conf && echo "nameserver 8.8.4.4" >>  /etc/resolv.conf，修改Docker的DNS，将其改成8.8.4.4

+ 问题分析：
分析问题的第一步就会回看"现场"，即报错的具体输出，通常报错时，会将这个错误栈都输出，方便使用者定位原始报错位置，但报错的关键原因依旧是最后几句，就上面的问题而已，就是`requests.exceptions.ConnectionError: HTTPConnectionPool(host='yann.lecun.com', port=80): Max retries exceeded with url: /exdb/mnist/train-labels-idx1-ubyte.gz (Caused by NewConnectionError('<requests.packages.urllib3.connection.HTTPConnection object at 0x7f5d86ae1550>: Failed to establish a new connection: [Errno -2] Name or service not known',))`，简单看一下，其实就知道是网络问题了，关键点在于`Max retries exceeded with url`，即该url超出最大的重试次数了，`Failed to establish a new connection`，即无法建立新连接，从而可以判断是网络出了问题，那么要解决这个问题其实就要先解决网络问题，因为这个错误是从docker报出的，那么首先就要判断是docker无法连接网络还是本地无法连接网络，再一步步解决

+ 问题拓展：
如果从报错信息中看到了HTTPConnectionPool、XXXConnectionError之类的字眼，大概率就是网络出了问题，才会导致这类报错。网络报错是有很明显标志的，如无法连接、连接次数过多等，确定了是网络问题后，就好解决了，你只需要确认是什么导致你网络无法连接，这里就有多种原因了，有软件层面的，有硬件层的，简单讨论一下，软件层面通常有两种情况，一种是你使用了全局代理或全局抓包软件，这些软件如果开了全局模式就会将所有的网络请求都重定向到指定的地址，这可能会导致网络断开，另一种可能就是你的host文件有问题，host文件可以理解为本地的DNS，检查一下DNS，看看是否有相应的配置，将请求地址重定向了。硬件层可能性就很多了，网卡设备坏了，路由有问题等等

+ 问题研究：
docker内的网络问题除了可能是本地网络有问题外，还有可能就是docker镜像无法连接网络，判断是本地无法连接网络还是单独docekr镜像无法连网，方法是在本地ping一下公网，如果只是单纯docker无法连接网络，可以尝试下面的方法：
	1.使用--net:host选项sudo docker run --net:host --name ubuntu_bash -i -t ubuntu:latest /bin/bash<br>
	2.使用--dns选项sudo docker run --dns 8.8.8.8 --dns 8.8.4.4 --name ubuntu_bash -i -t ubuntu:latest /bin/bash<br>
	3.改dns servervi /etc/default/docker去掉“docker_OPTS="--dns 8.8.8.8 --dns 8.8.4.4"”前的#号<br>
	4.不用dnsmasqvi /etc/NetworkManager/NetworkManager.conf 在dns=dnsmasq前加个#号注释掉sudo restart network-managersudo restart docker <br>
	5.重建docker0网络pkill dockeriptables -t nat -Fifconfig docker0 downbrctl delbr docker0docker -d<br>
	6.直接在docker内修改/etc/hosts


## 3.问题：'SGD' object has no attribute 'save_parameter_to_tar'

+ 关键字：`SGD` `attribute` `save_parameter_to_tar`

+ 问题描述：使用PaddlePaddle编写识别MNIST数据集代码时，编写相应的模型保存逻辑，但执行到保存方法`trainer.save_parameter_to_tar(f=f)`时，报'SGD' object has no attribute 'save_parameter_to_tar'

+ 报错代码段：
```python
if isinstance(event, paddle.event.EndPass):
    # 保存训练好的参数
    model_path = '../model'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    with open(model_path + "/model.tar", 'w') as f:
        trainer.save_parameter_to_tar(f=f)
```

+ 报错输出：

```python
'SGD' object has no attribute 'save_parameter_to_tar'
```

+ 复现方式：
使用旧版本的PaddlePaddle运行上述MNIST数据集识别代码，出现'SGD' object has no attribute 'save_parameter_to_tar'

+ 解决方法：
使用fluid版PaddlePaddle以及相应的模型保存方法，即`save_params`，通过Docker镜像安装最新的PaddlePaddle

+ 问题分析：
在旧版的PaddlePaddle中，为paddle.trainer.SGD对象提供了save_parameter_to_tar方法，但在新版的PaddlePaddle中，将该方法移除了，而使用`save_params`方法将其代替，让使用更加方便，常见的使用方式如下，展示部分代码片段：

```python
trainer = fluid.Trainer(
    train_func= train_program,
    optimizer_func=optimizer_program,
    place=place
)

def event_handler(event):

    if isinstance(event, fluid.EndEpochEvent):
        avg_cost, accuracy = trainer.test(
            reader=test_reader, feed_order=['pixel','label']
        )
        print('\nTest with Pass {0}, loss {1:2.2}, Acc {2:2.2}'.format(
            event.epoch,avg_cost,accuracy))

        if params_dirname is not None:
        	#保存模型
            trainer.save_params(params_dirname)
```

+ 问题拓展：
在深度学习训练模型的过程中，保存模型的逻辑是非常重要的，因为很多模型训练时间比较长，如果中途遇到不可抗因素导致训练终止，而你又没有做报错模型的逻辑，那么此前的训练就浪费了，再次训练又要重头开始，非常耗费时间以及精力，所谓训练好的模型其实就是模型结构上的节点中已经有了有意义的参数，这些参数是在训练通常中通过梯度下降算法更新获得的，训练模型的目的其实就是为了获得这些参数，将这些参数保存起来，下次训练时可以直接在此前的训练成果上进行，方便也快速，而且将模型保存起来也方便使用，毕竟训练出模型的目的就是使用，所以保存模型是非常必要的。

+ 问题研究：
因为PaddlePaddle框架目前在高速发展中，很多方法都会逐步优化成更加友好的方法，方便用户使用，所以一些旧的使用方法就容易出错，通常使用新版PaddlePaddle中的新方法来代替旧方法则可，当然，你也可以降级PaddlePaddle以适应自己的代码，但通常不建议这样做，因为新版的框架除了API方面变得更简单外，内部也做了很多调整，使得它比旧版的框架更加快速稳定，所以依旧建议修改自己的代码以适应新版的PaddlePaddle框架。


## 4.问题： No such file or directory: '/work/image/infer_3.png'

+ 关键字：`file` `directory`

+ 问题描述：PaddlePaddle使用Docker镜像进行安装，使用的是最新版的，即Fluid版的PaddlePaddle，运行Github上手写数字识别的代码，MNIST数据集正常下载，但代码运行到最后报`No such file or directory: '/work/image/infer_3.png'`

+ 报错截图：
![](https://raw.githubusercontent.com/ayuLiao/images/master/%E8%AF%86%E5%88%AB%E6%95%B0%E5%AD%971.png)

+ 复现方式：
安装PaddlePaddle最新Docker镜像，下拉`https://github.com/PaddlePaddle/book/blob/f5e6bfa8346a96d013bc04d14570a27ab2f7f613/02.recognize_digits/train.py`中的代码，使用docker运行，出现`No such file or directory: '/work/image/infer_3.png'`

+ 解决方法：
需要将MNIST数据集中的一张图像存放到你指定的目录下，具体的目录得看使用docker运行python代码时的命令，当然报错信息也会给出相应的路径，这里的路径为`'/work/image/infer_3.png'`，你只需将要识别的手写数字图像命名为infer_3.png，并放在docker运行环境的/work/image目录下则可，当然，这些都可以通过python代码来完成，官方github中提供的代码并秘钥执行这些逻辑，你可以自行添加。

+ 问题分析：
`No such file or directory：XXX`这类问题是非常常见的，不止出现在深度学习相关的任务中，报错输出也将报错原因表明的很清楚，即文件或目录不存在，那么创建需要的文件或目录则可，另一种方法就是修过代码，不从这个目录中读取文件。

+ 问题拓展：
深度学习训练出来的模型通常都是要使用的，所谓使用其实就是输入数据获得输出，如果代码读入数据所在的路径不存在，那么就会报`No such file or directory：XXX`，使用时注意一下路径问题则可。

+ 问题研究：
略

## 5.问题：TypeError: __init__() got an unexpected keyword argument 'optimizer_func'

+ 关键字：`optimizer_func` `MNIST`

+ 描述问题：按照官方文档中的识别手写数字的例子运行识别手写数字的代码，出现`TypeError: __init__() got an unexpected keyword argument 'optimizer_func'`错误

+ 报错代码段：

```python
trainer = fluid.Trainer(train_func=train_program, place=place, optimizer_func=optimizer_program)
```

+ 报错截图：
![](https://ai.bdstatic.com/file/31D2124901644389B7453D1C96A6AACF)

+ 报错输出：
```python
TypeError: __init__() got an unexpected keyword argument 'optimizer_func'
```

+ 复现方式：
使用0.13.0版本的PaddlePaddle运行该代码，就会出现`TypeError: __init__() got an unexpected keyword argument 'optimizer_func'`

+ 解决方法：
使用正确版本的PaddlePaddle进行模型的训练。不同版本的PaddlePaddle之间方法有可能不相同，当前该问题就是因为PaddlePaddle版本不对，`optimizer_func`出现在0.14.0版本之后，在0.13.0版本中，优化器函数应该为optimizer

+ 问题分析：
当我们构建好模型后，就需要通过相应的优化算法来训练模型，比如常见的SGD算法、Adam算法等，PaddlePaddle已经为我们封装好了这些算法，使用起来非常方便。因为PaddlePaddle在高速的发展，某些方法在新版的PaddlePaddle中已经被更新的，此时再使用旧版的方法就会报错，在fluid版本的PaddlePaddle中，优化器的使用方法为`fluid.optimizer.Adam(learning_rate=0.001)`，下面展示相关的代码片段，如下：

```python

def optimizer_program():
	#新版PaddlePaddle优化器使用
    return fluid.optimizer.Adam(learning_rate=0.001)

use_cuda = False
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
# Trainer 需要接受训练程序 train_program, place 和优化器 optimizer_func
trainer = fluid.Trainer(
    train_func= train_program,
    optimizer_func=optimizer_program,
    place=place
)
```

+ 问题拓展：
目前训练神经网络时都使用梯度下降法作为优化算法，梯度下降法有多种，如随机梯度下降法（Stochastic Gradient Descent，即SGD）、小批量梯度下降算法（Mini-Batch Gradient Descent，即MBGD）、SGD with Monentum、AdaDelta算法、Adam算法等等，Adam算法是比较常用的算法，因为它算是集大成的优化算法了，使用了一阶动量、二阶动量等，但在很多论文中依旧使用SGD随机梯度下降算法作为优化算法，SGD是一个比较朴素的算法，它的训练速度比较慢，而Adam算法会在一开始使用较大的学习速率更新网络参数，然后随着训练次数的加多而动态减小学习速率，但也因为Adam算法方便使用，即封装的太好，让研究人员难以进行细粒度的控制，有时使用Adam算法反而让模型难以收敛，而SGD虽然训练速度比较慢，但如果你对自己的使用的数据集非常了解，你也可以手动的调整SGD的学习效率。

+ 问题研究：
因为PaddlePaddle高速发展的原因，网络上很多教程或代码没有来得及更新，导致很多用户通过阅读旧的教程或代码来编写相应的结构，从而造成类似的错误。在编写模型时，尽量从一个更高的角度去看待模型，即只用先了解模型编写的流程以及具体意义则可，具体的实现代码要以`https://github.com/PaddlePaddle/book`上的代码为准，上面的代码通常都是最新的代码，便于学习与理解。

## 6.问题：ImportError: No module named XXX

+ 关键字：`ImportError` `module`

+ 问题描述：在使用PaddlePaddle进行手写字母识别时，出现`ImportError: No module named XXX`，缺失某些文件

+ 报错输出：

```python
ImportError: No module named XXX(具体缺失的模块名称)
```

+ 复现方式：
PaddlePaddle运行训练某些代码时，缺少必要的模块

+ 解决方案：
凡事出现`ImportError: No module named XXX`，就说明缺少了某个库，该库在代码中被使用或者是其他库的依赖，如ImportError: No module named PIL 或 ImportError: No module named 'urllib3' ，通常只需通过pip将缺失的库安装上则可，命令为：`pip install XXX(缺失模块名)`

+ 问题分析：
`ImportError: No module named XXX`在前面已经详细的讨论过了，即缺少了某个库，缺少什么就安装什么则可，值得一提的是，如果你使用了PaddlePaddle的Docker镜像进行训练，那么进入Docker镜像的命令行环境后，使用pip安装了缺失的库后，不可以直接退出，如果直接退出，那么安装的数据就会丧失，下次进入该docker镜像依旧缺失相应的库，如果你多docker进行做了改动，就需要提交这个改动，使用commit命令则可。

+ 问题拓展：
PaddlePaddle的docker镜像的python环境只提供了基本的依赖包，一些你可能需要使用的包在这里并不会提供，比如在处理图像数据时，你可能需要使用opencv包，但docker镜像中并没有提供该包，此时就需要你自己手动安装了，对于这样的需求有两种方式，第一种，使用Dockerfile，即自己编写Dockerfile文件来构建一个新的Docker镜像，该Dockerfile文件最好基于PaddlePaddle提供的Dockerfile来编写，这样Docker镜像中就安装PaddlePaddle需要的基本库，同时也安装了你需要的库，具体命令如下：

	```bash
	FROM paddlepaddle/paddle:latest
	MAINTAINER ayuliao <ayuliao@163.com>

	RUN apt-get update && apt-get install -y xxx #你需要的库
	RUN pip install xxx #你需要的库
	```

	第二种方式就是进入Docker镜像中，直接通过相关的命令来安装，如下：

	```bash
	docker run -t -i 你的镜像 /bin/bash
	pip install xxx #安装命令
	docker commit -m="提交信息" -a="作者" 记录的ID 镜像名称 #保存变动
	```

+ 问题研究：
因为python提供丰富的第三方库，这可以使开发效率变得极高，比如使用numpy、pandas库来分析数据，引入它俩就可以快速对数据进行分析了，但这样就了一些麻烦，就是他人在编写代码时使用了很多第三方库加快开发速度，而你获得他的代码后，却无法运行，一大原因就是缺少某些库，其报错表现就是`ImportError: No module named XXX`，解决方法，就是安装上相应的库，主要库与库直接的版本冲突则可。

## *7.问题：TypeError: sequence item 1: expected str instance, bytes found

+ 关键字：`TypeError` `API`

+ 问题描述：使用百度人工智能API进行数字识别，secretkey正常填写，报出`TypeError: sequence item 1: expected str instance, bytes found`

+ 报错代码段：

```python
import requests, base64, urllib, time
from datetime import datetime
print datetime.utcnow()
start = time.time()
import sign_sample
import xlrd
import xlwt
import glob
import os
from xlutils.copy import copy
accesskey = ''
secretkey = '20ab82349d694ffe957d4921c5f6be7e'
signer = sign_sample.BceSigner(accesskey, secretkey)

headers={
'Host': 'aip.baidubce.com',
'content-type': 'application/x-www-form-urlencoded'
}
request = {
'method': 'POST',
'uri': '/rest/2.0/ocr/v1/numbers',
'headers': headers
}
headers['Authorization'] = signer.gen_authorization(request)
url = 'http://' + str(headers['Host']) + str(request['uri'])
def get_file_content(filePath):
	with open(filePath, 'rb') as fp:
		return fp.read()
		oldWb = xlrd.open_workbook(r'e:\data\test4.xls', formatting_info=True)
		sheet_rows = oldWb.sheet_by_index(0)
		ws=copy(oldWb)
		wb=ws.get_sheet(0)
		""" 读取图片 """
		WSI_MASK_PATH = 'C:/Users/55556769/Desktop/test1'#存放图片的文件夹路径
		wsi_mask_paths = glob.glob(os.path.join(WSI_MASK_PATH, '*.jpg'))
		for i in wsi_mask_paths:
			""" 调用行驶证识别 """
			body = {
			'image': base64.b64encode(get_file_content(i)),
			# 'id_card_side': 'front'
			}
			response = requests.request(request['method'], url, headers=headers, data=urllib.urlencode(body))
			result = eval((response.text).encode('UTF-8'))
			print(result)
			if result.has_key("words_result"):
				result_words=result["words_result"]
				if result["words_result"][0].has_key('words'):
					wb.write(wsi_mask_paths.index(i)+sheet_rows.nrows,0, result["words_result"][0]["words"]) # 写入位置，及文本
				else:
					wb.write(wsi_mask_paths.index(i) + sheet_rows.nrows, 0, unicode(i, "gbk"))
				else:
					wb.write(wsi_mask_paths.index(i) + sheet_rows.nrows, 0, unicode(i, "gbk"))
					try:
						os.getcwd()
						os.chdir('e:\\data')
						ws.save("test4.xls") # 保存文件到制定路径
					except Exception as err:
						fillte = '导出失败:' + str(err)
						print(fillte)
					else:
						succefull = '导出成功'
						print(succefull)
```

+ 解决方法：
不使用secretkey，而使用Token的方式来请求百度的数字识别API，代码如下：

```python
'''
Created on 2018-8-17
数字识别接口-Python3 -API示例代码
@author: 小帅丶
'''
import urllib3,base64
from urllib.parse import urlencode
access_token='自己的token'
http=urllib3.PoolManager()
url='https://aip.baidubce.com/rest/2.0/ocr/v1/numbers?access_token='+access_token
f = open('图片本地路径地址','rb')
#参数image：图像base64编码
img = base64.b64encode(f.read())
params={'image':img}
#对base64数据进行urlencode处理
params=urlencode(params)
request=http.request('POST', 
                      url,
                      body=params,
                      headers={'Content-Type':'application/x-www-form-urlencoded'})
#对返回的byte字节进行处理。Python3输出位串，而不是可读的字符串，需要进行转换
result = str(request.data,'utf-8')
print(result)
```

+ 问题分析：

+ 问题拓展：

+ 问题研究：

## 8.问题：error writing to output: Broken pipe

+ 关键字：`Broken pipe` `uncompress failed` `数字识别`

+ 问题描述：在mac上使用PaddlePaddle运行官网数字识别的例子，报出`gzcat: error writing to output: Broken pipe` 和 `
gzcat: /Users/wanghongtao/.cache/paddle/dataset/mnist/train-images-idx3-ubyte.gz: uncompress failed` 

+ 报错输出：

```python
I1224 13:05:25.112965 2439074624 GradientMachine.cpp:94] Initing parameters..
I1224 13:05:25.113909 2439074624 GradientMachine.cpp:101] Init parameters done.
*** Aborted at 1514091925 (unix time) try "date -d @1514091925" if you are using GNU date ***
PC: @ 0x0 (unknown)
*** SIGFPE (@0x7fff30c35fbb) received by PID 18537 (TID 0x7fff91615340) stack trace: ***
@ 0x7fff5877ff5a _sigtramp
@ 0x7fff30c35fbc CFNumberCreate
@ 0x7fff32d7d3a8 -[NSPlaceholderNumber initWithDouble:]
@ 0x7fff3bd87bb1 +[CALayer defaultValueForKey:]
@ 0x7fff3bd86e11 classDescription_locked()
@ 0x7fff3bd8629a classDescription_locked()
@ 0x7fff3bd8629a classDescription_locked()
@ 0x7fff3bd85bc7 classDescription()
@ 0x7fff3bd857f2 CA::Layer::class_state()
@ 0x7fff3bd856b1 -[CALayer init]
@ 0x7fff2ee15f0b -[_NSBackingLayer init]
@ 0x7fff2e29a297 -[NSView makeBackingLayer]
@ 0x7fff2e29a100 -[NSView(NSInternal) _createLayerAndInitialize]
@ 0x7fff2eb397ff -[NSView _updateLayerBackedness]
@ 0x7fff2eb38237 -[NSView didChangeValueForKey:]
@ 0x7fff2e2a612a __49-[NSThemeFrame _floatTitlebarAndToolbarFromInit:]_block_invoke
@ 0x7fff2ec1a571 +[NSAnimationContext runAnimationGroup:]
@ 0x7fff2e2a5d7c -[NSThemeFrame _floatTitlebarAndToolbarFromInit:]
@ 0x7fff2e2a3965 -[NSThemeFrame initWithFrame:styleMask:owner:]
@ 0x7fff2e2a240a -[NSWindow _commonInitFrame:styleMask:backing:defer:]
@ 0x7fff2e2a0c3d -[NSWindow _initContent:styleMask:backing:defer:contentView:]
@ 0x7fff2e2a06f6 -[NSWindow initWithContentRect:styleMask:backing:defer:]
gzcat: error writing to output: Broken pipe
gzcat: /Users/wanghongtao/.cache/paddle/dataset/mnist/train-images-idx3-ubyte.gz: uncompress failed
```

+ 解决方法：
从报错信息可以看出train-images-idx3-ubyte.gz文件uncompress failed，即解压失败，这可能是因为网络原因，到时train-images-idx3-ubyte.gz文件没有下载全，请删除uncompress failed的train-images-idx3-ubyte.gz文件，重新下载。

+ 问题分析：
从报错的输出来看，关键的报错信息为`gzcat: error writing to output: Broken pipe`、`uncompress failed`，该命令报错，很有可能就是数据解压时发生了错误。当我在使用PaddlePaddle构建数字识别模型时，要读入压缩的MNIST数据集，此时就会涉及到数据文件的解压缩，从uncompress failed可以看出，数据文件train-images-idx3-ubyte.gz解压缩失败，那么就有两种情况，一种是gzcat命令使用不正常，到时对正常的压缩文件解压失败，另一种就是该文件是一个不正常的压缩文件，解码命令无法对其进行解压，判断方法也很简单，不在代码层面上解压，直接通过命令行的方式对该数据集进行解压，看看是否可以正常解压。如果判断出是数据文件有问题，你可以手动重新下载过MNIST数据集文件，或删除该路径已有的MNIST数据集文件，再次使用PaddlePaddle下载MNIST数据集。

+ 问题拓展：

gzcat命令是一个与压缩解压有关的命令，可以通过`man gzcat`来查看一下它的信息，如下：

```bash
NAME
     gzip -- compression/decompression tool using Lempel-Ziv coding (LZ77)

SYNOPSIS
     gzip [-cdfhkLlNnqrtVv] [-S suffix] file [file [...]]
     gunzip [-cfhkLNqrtVv] [-S suffix] file [file [...]]
     zcat [-fhV] file [file [...]]

DESCRIPTION
     The gzip program compresses and decompresses files using Lempel-Ziv cod-
     ing (LZ77).  If no files are specified, gzip will compress from standard
     input, or decompress to standard output.  When in compression mode, each
     file will be replaced with another file with the suffix, set by the -S
     suffix option, added, if possible.
```

在不同的系统中，gzcat命令可能不同，在某些系统上，zcat可以作为gzcat安装。如果是解压缩时出现了问题且确定自己的数据集没有问题，可以尝试使用其他的解压缩命令，如unzip等，如果是数据集本身存在问题，就需要重新下载了

+ 问题研究：
我们在训练模型需要一定量的数据，而这些数据通常在互联网中以压缩文件的形式提供，压缩的数据可以节省使用的宽带，所以训练时，对数据进行解压缩是一种常见的操作，当然，你可以先解了压缩，在使用代码直接读取解压后的数据源进行模型的训练，在上述问题中报错很有可能就是数据文件是损失的，导致解压失败了，从而造成程序无法进行，重新下载数据文件就可以解决当前的问题了。


## 9.问题：PaddlePaddle预测MNIST数据问题

+ 问题描述：在训练好模型之后，自己写了一个paddle版的预测脚本。
训练模型评估结果如下:

```python
Best pass is 00087,  error is 0.180342, which means predict get error as 0.212333464155
The classification accuracy is  94.68%
evaluating from pass output/pass-00087
```

但每次执行脚本`python predict.py -c mlp_mnist.py -d ./raw_data/train -m mlp_mnist_model/pass-00087`的时候，输入图片的序号不同，predict结果都是相同的，如下:


```python
3  <- 输入图片序号
[[ 0.08217829  0.08400964  0.14461318  0.11315996  0.05480111  0.12913492
   0.07871926  0.11914014  0.11503445  0.07920909]]  predict输出向量
2 1  <- predict结果和标签

13
[[ 0.08217829  0.08400964  0.14461318  0.11315996  0.05480111  0.12913492
   0.07871926  0.11914014  0.11503445  0.07920909]]   <-这个结果相同
2 6

908
[[ 0.08217829  0.08400964  0.14461318  0.11315996  0.05480111  0.12913492
   0.07871926  0.11914014  0.11503445  0.07920909]]
2 7
```

+ 报错代码段：

```python
# predict.py文件
class Prediction():
     def __init__(self, train_conf, data_dir, model_dir):

         conf = parse_config(
             train_conf,
             'is_predict=1')
         self.network = swig_paddle.GradientMachine.createFromConfigProto(
             conf.model_config)
         self.network.loadParameters(model_dir)

         self.images, self.labels = read_data("./data/raw_data/", "train")

         slots = [ dense_vector( 28 * 28 ) ]
         self.converter = DataProviderConverter(slots)

     def predict(self, index):
         print self.images[index].tolist()
         input = self.converter([self.images[index].tolist() ])
         output = self.network.forwardTest(input)
         prob = output[0]["value"]
         predict = np.argsort(-prob)
         print prob
         print predict[0][0], self.labels[index]

     def plot(self, index):
         plt.imshow(self.images[index], cmap="Greys_r")
         plt.show()

def main():
	arguments = docopt(__doc__)
	train_conf = arguments['CONF']
	data_dir = arguments['DATA']
	model_dir = arguments['MODEL']
	swig_paddle.initPaddle("--use_gpu=0")
	predict = Prediction(train_conf, data_dir, model_dir)
	line = raw_input()
	index = int(line.strip().split()[0])
	predict.predict(index)

# 配置文件
#######################Network Configuration #############

data_size = 1 * 28 * 28
label_size = 10
img = data_layer(name='pixel', size=data_size)

# The first fully-connected layer
hidden1 = fc_layer(input=img, size=128, act=TanhActivation())
# The second fully-connected layer and the according activation function
hidden2 = fc_layer(input=hidden1, size=64, act=TanhActivation())
# The thrid fully-connected layer, note that the hidden size should be 10,
# which is the number of unique digits
predict = fc_layer(input=hidden2, size=10, act=SoftmaxActivation())
```

+ 解决方法：
将保存代码中的`input = self.converter([self.images[index].tolist() ])`修改为`input = self.converter([[self.images[index].flatten().tolist() ]])`则可。

+ 问题分析：

+ 问题拓展：

+ 问题研究：

## 10.问题：gzip: stdout: Broken pipe ; Segmentation fault

+ 关键字： `Broken pipe` `Segmentation fault`

+ 问题描述：我在centos7.3上install paddlepaddle,然后希望训练模型mnist_v2.py，执行后报错，错误为`gzip: stdout: Broken pipe
Segmentation fault`

+ 报错输出：

```python
[root@localhost dense]# python mnist_v2.py
I0509 05:58:53.570152 8785 Util.cpp:166] commandline: --use_gpu=False --trainer_count=1
I0509 05:58:53.576031 8785 GradientMachine.cpp:94] Initing parameters..
I0509 05:58:53.582324 8785 GradientMachine.cpp:101] Init parameters done.
Cache file /root/.cache/paddle/dataset/mnist/train-images-idx3-ubyte.gz not found, downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
[==================================================]Cache file /root/.cache/paddle/dataset/mnist/train-labels-idx1-ubyte.gz not found, downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
[==================================================]*** Aborted at 1525845546 (unix time) try "date -d @1525845546" if you are using GNU date ***
PC: @ 0x0 (unknown)
*** SIGSEGV (@0x50) received by PID 8785 (TID 0x7f715aa41740) from PID 80; stack trace: ***
@ 0x7f715a2535e0 (unknown)
@ 0x7f715a837aa6 _dl_relocate_object
@ 0x7f715a83fc5c dl_open_worker
@ 0x7f715a83b2d4 _dl_catch_error
@ 0x7f715a83f2cb _dl_open
@ 0x7f71598aba72 do_dlopen
@ 0x7f715a83b2d4 _dl_catch_error
@ 0x7f71598abb32 __GI___libc_dlopen_mode
@ 0x7f71598842a5 init
@ 0x7f715a250e20 __GI___pthread_once
@ 0x7f71598843bc __GI___backtrace
@ 0x7f71502e1643 (unknown)
@ 0x7f71502e1b80 (unknown)
@ 0x7f71502c883b (unknown)
@ 0x7f715a4a789c (unknown)
@ 0x7f715a4a9cba PyNumber_Multiply
@ 0x7f715a5412f0 PyEval_EvalFrameEx
@ 0x7f715a4c61f8 (unknown)
@ 0x7f715a53fb01 PyEval_EvalFrameEx
@ 0x7f715a4c61f8 (unknown)
@ 0x7f715a53fb01 PyEval_EvalFrameEx
@ 0x7f715a4c61f8 (unknown)
@ 0x7f715a4c1c6f (unknown)
@ 0x7f715a53fb01 PyEval_EvalFrameEx
@ 0x7f715a546efd PyEval_EvalCodeEx
@ 0x7f715a5443fc PyEval_EvalFrameEx
@ 0x7f715a546efd PyEval_EvalCodeEx
@ 0x7f715a5443fc PyEval_EvalFrameEx
@ 0x7f715a546efd PyEval_EvalCodeEx
@ 0x7f715a547002 PyEval_EvalCode
@ 0x7f715a56043f (unknown)
@ 0x7f715a5615fe PyRun_FileExFlags

gzip: stdout: Broken pipe
Segmentation fault
```

+ 解决方法：
从错误信息来看，Python在读取数据/root/.cache/paddle/dataset/mnist/train-images-idx3-ubyte.gz 和 /root/.cache/paddle/dataset/mnist/train-labels-idx1-ubyte.gz时出现了问题，先判断这两个文件本身是否有问题，即直接在命令行使用gzip对这两个文件进行解压，如果无法解压，说明这两个文件本身存在问题，请删除错误的文件，重新下载，如果解压成功，则读取这两个文件的代码可能存在问题，请尝试单独写一段压缩文件读取的代码进行验证，如下：

```python
import gzip
with gzip.open('somefile.gz', 'rt') as f:
    text = f.read()
```

+ 问题分析：
该文件的关键报错为`gzip: stdout: Broken pipe`、`Segmentation fault`，看到报错信息中的其他信息，可以发现train-images-idx3-ubyte.gz文件以及train-labels-idx1-ubyte.gz可能存在问题，但依旧不知道具体的问题是什么，对于多种情况的问题，使用排除法，从简单的情况开始，首先检查这两个文件是否有问题，从关键报错信息可以看到，gzip命令报错，那么就在命令行中直接使用gzip命令去解压这两个压缩文件，如果可以正常解压，则python代码有问题，单独写一段简单的python代码尝试读取压缩文件中内容，如果无法解压，则说明这两个文件本身存在问题，可能是下载时网络不稳定或其他原因，导致文件损坏了，删除损失文件，重新下载则可。

+ 问题拓展：
gzip命令是linux中常见的压缩解压命令，pythong中提供了gzip包方便我们直接使用它，gzip包的背后其实就是系统级的gzip命令，通常看见与gzip命令相关的报错，很有可能就是解压缩的时候出现了问题，出现问题的情况通常就是两种，要么解压缩的文件本身有问题，如文件是损坏的，无法解压，要么是gzip使用方式有问题，根据不同的情况做相应的操作就可以解决该问题了。

	文件压缩其实是一个很有趣的算法问题，如果让文件尽可能的小同时又保存其中的数据是一个值得研究的问题，当下常见的压缩算法有：

	1.字段算法
	2.Fixed Bit Length Packing 固定位长算法
	3.Run Length Encoding RLE
	4.霍夫曼编码（Huffman Encoding）
	5.Lempel-Ziv (LZ77)
	等等


+ 问题研究：
当看见与解压压缩有关的报错时，就要有上面谈论的想法，即判断是什么情况，是文件有问题还是代码有问题，利用排除法的思维找到具体的报错原因，解压与压缩的报错并不难解决，因为底层的解压或压缩的逻辑并不需要我们实现，所以解决起来比较轻松。



















