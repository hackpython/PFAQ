# 线性回归

## 背景介绍
给定一个大小为n的数据集$${y_i,x_{i1},...,x_{id}}^n_{i=1}$$，其中$$x_{i1},…,x_{id}$$是第i个样本d个属性上的取值，yi是该样本待预测的目标。线性回归模型假设目标yi可以被属性间的线性组合描述，即

$$y_i=ω_1x_{i1}+ω_2x_{i2}+…+ω_dx_{id}+b,i=1,…,n$$

初看起来，这个假设实在过于简单了，变量间的真实关系很难是线性的。但由于线性回归模型有形式简单和易于建模分析的优点，它在实际问题中得到了大量的应用。很多经典的统计学习、机器学习书籍[2,3,4]也选择对线性模型独立成章重点讲解。

## 房价预测PaddlePaddle-fluid版代码：
https://github.com/PaddlePaddle/book/blob/develop/01.fit_a_line/train.py

PaddlePaddle文档中的内容目前依旧是PaddlePaddle-v2版本，建议使用Fluid版本来编写房价预测模型

## `已审阅` 1.问题：module对象没有model属性

+ 版本号：`1.0.1`

+ 标签：`module` `model属性`

+ 问题描述：利用波士顿房价数据集进行模型的训练和预测时，使用了uci_housing.model()，uci_housing模块封装了该数据集的相关操作，但却报出`'module' object has no attribute 'model'`错误，module对象没有model属性。

+ 报错代码段：

```python
paddle.infer(
    output_layer = y_predict,
    parameters = paddle.dataset.uci_housing.model(), #报错
    input = [item for item in paddle.dataset.uci_housing.test()()]
)
```

+ 报错截图：

    ![](https://raw.githubusercontent.com/jizhi/images/master/%E7%BA%BF%E6%80%A7%E5%9B%9E%E5%BD%921.png)

+ 报错输出：

```python
I0116 13:53:48.957136 15297 Util.cpp:166] commandline:  --use_gpu=False --trainer_count=1
Traceback (most recent call last):
  File "housing.py", line 13, in <module>
    parameters=paddle.dataset.uci_housing.model(),
AttributeError: 'module' object has no attribute 'model'
```

+ 复现方式：
使用旧版本的paddlepaddle编写代码，训练波士顿房价数据，使用了paddle.dataset.uci_housing.model()，出现`'module' object has no attribute 'model'`错误

+ 解决方案：
PaddlePaddle通过uci_housing模块引入了数据集合UCI Housing Data Set，并且在该模块中封装了数据的下载以及预处理，其下载数据会保存在下载数据保存在~/.cache/paddle/dataset/uci_housing/housing.data，而预处理方法会因版本不同而略有不同，旧版本的PaddlePaddle可能会出现`'module' object has no attribute 'model'`报错，请尝试安装最新版本的PaddlePaddle。PaddlePaddle 0.15.0 版本不会遇到module对象没有model属性的问题。

+ 问题分析：
在开发过程中，经常会遇到`'module' object has no attribute 'xxx'`的问题，其原因报错信息说的很清楚，就是某个对象没有某个属性，这个错误出现的通常原因就是没有正确使用某个对象，造成错误写法的原因很有可能是这种写法是该对象旧的用法，而最新的对象已经删除了这个属性了，此时再用久的方法来使用该对象就会出现`'module' object has no attribute 'xxx'`的问题，最简单的解决方式就是看一下官方文档了解该对象最新的用法。

+ 问题拓展：
`'module' object has no attribute 'xxx'`这类问题其实就是使用方式不当的问题，以后遇到第三库报错，遇到缺少什么什么通常都是使用问题，这时就要回顾一下自己使用时该库的代码，注意使用第三库的版本以及python字符编码等问题，通常阅读一下最新的官方文本或自己百度一下都可以找到相应的解决方案。

+ 问题研究：
这类问题算是经验型问题，只要遇到过，有印象，解决起来都类似的方式，因为某对象缺失某属性是硬性问题，如果是自己编写的对象，那么就编写上相应的属性，如果是第三库，那么通常就是该库的用法发生了变动，你使用的方式不是最新的方式，通过最新的方式使用则可，或者降级自己的第三库来配合自己的代码，至于这两种方法选择使用哪一种，主要看自己的“代价”，即该代码方便还是还旧的库使用方便，建议修改代码，使用新的第三方库，让代码更加优雅点。


## `已审阅`  2.问题：“非法指令”或“illegal instruction”

+ 版本号：`1.0.1`

+ 标签：`非法指令` `illegal instruction` `avx指令集`

+ 问题描述：通过源码编译安装的方式成功安装了PaddlePaddle当前最新版本，执行波士顿房价预测代码时，出现illegalinstruction错误。

+ 报错代码段：
```python
import paddle.v2 as paddle
paddle.init(use_gpu=False, trainer_count=1) #报错
```

+ 报错输出：
```
非法指令 (核心已转储)
```

+ 复现方式：
ubuntu以源码安装的方式安装旧版的PaddlePaddle，代码可以安装编译流程都较正常，但使用时会出现`illegalinstruction`

+ 解决方案：
PaddlePaddle使用avx SIMD指令提高cpu执行效率，因此错误的使用二进制发行版可能会导致这种错误，请先判断你的电脑是否支持AVX指令集，再选择性的安装支持AVX指令集的PaddlePaddle还是不支持AVX指令集的PaddlePaddle，或者使用Docker镜像来安装最新版本的PaddlePaddle，Docker镜像中的PaddlePaddle默认支持是支持AVX指令集的，可以提高cpu的执行效率。

+ 问题分析：
与问题3类似的是该问题依旧是`paddle.init`初始化PaddlePaddle就报illegal instruction，原因与问题3类似，检查系统是否支持PaddlePaddle初始化时所需要的资源，从AVX指令集开始检查。

+ 问题拓展：
遇到框架初始化问题，可以先尝试换个电脑或系统再次安装一下该框架，判断是否是电脑或系统的问题，如果是，通常就是该电脑或系统不支持该版本框架所需要的资源，可以参考问题3处理一下这类问题。

+ 问题研究：
因为PaddlePaddle以源码安装的方式比较繁杂，要注意比较多的细节，而且要对自己使用的系统有一定的了解，不然很有可能会导致虽然安装上了，但无法使用PaddlePaddle的问题，一个原因是PaddlePaddle对系统依赖比较大，另一个原因就是安装时做错了某些步骤。这种问题其实很常见，从运维角度来看，这是非常繁杂的工程，我们需要将开放环境一个个的部署到线上服务器上，很有可能就出现类似的问题，Docker就是一个很好的解决方案，它将开放所需要的环境都封装在镜像中了，方便使用。




## `已审阅` 3.问题：下载housing.data失败

+ 版本号：`1.0.1`

+ 标签：`数据` `housing`

+ 问题描述：PaddlePaddle环境安装好了，尝试运行波士顿房价预测的实验， 但是遇到问题下载housing.data 失败，尝试用VPN也没成功，`RuntimeError: Cannot download https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data within retry limit 3`，尝试从浏览器直接访问housing.data文件，报403Forbidden 错误。

+ 报错输出：
```
RuntimeError: Cannot download https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data within retry limit 3
```

+ 解决方案：官方的housing.data数据由于放在国外服务器，网络常常不稳定，导致下载失败。我们目前已经在develop上把数据源移到了国内服务器上，修复了这个问题，但在之前放发布的版本上，例如0.15.0，由于非bug级别代码不能轻易修复，因此这里未改变。您可以试试手动访问我们的国内服务器下载这个数据集：
`wget http://paddlemodels.bj.bcebos.com/uci_housing/housing.data`或者使用迅雷直接下载，地址为`http://paddlemodels.bj.bcebos.com/uci_housing/housing.data`

+ 问题分析：
使用PaddlePaddle构建波士顿房价预测模型时是需要相应的训练数据的，将这些数据喂养给你构建的神经网络，通过多轮训练才能获得一个合理的房价预测模型，这一切的前提都是要有数据，PaddlePaddle为了方便波士顿房价预测模型的构建，将波士顿房价数据的获取封装在框架内了，不需要用户自己下载，但封装的越好，越方便，对用户而言就越不透明，因为波士顿房价数据在国外服务器上，所以有时下载会出现相应的问题，导致在使用时，数据格式与使用的代码不匹配出现错误，或者没有下载完全导致代码无法解压缩等。对于类似这样的问题，你都可以自己手动从去下载，或者将此前下载失败的数据删除，再次运行一遍代码，让PaddlePaddle重新下载，注意要删错误数据，因为PaddlePaddle下载前会检查相应的文件夹中是否存在该数据。

+ 问题拓展：
波士顿房价数据美国人口普查局收集的美国马萨诸塞州波士顿住房价格的有关信息, 数据集很小，只有506个案例，有14个属性，如下：
    CRIM--城镇人均犯罪率
    ZN - 占地面积超过25,000平方英尺的住宅用地比例。
    INDUS - 每个城镇非零售业务的比例。
    CHAS - Charles River虚拟变量（如果是河道，则为1;否则为0）
    NOX - 一氧化氮浓度（每千万份）
    RM - 每间住宅的平均房间数
    AGE - 1940年以前建造的自住单位比例
    DIS加权距离波士顿的五个就业中心
    RAD - 径向高速公路的可达性指数
    TAX - 每10,000美元的全额物业税率
    PTRATIO - 城镇的学生与教师比例
    B - 1000（Bk - 0.63）^ 2其中Bk是城镇黑人的比例
    LSTAT - 人口状况下降％
    MEDV - 自有住房的中位数报价, 单位1000美元

    熟悉了数据集的属性结构后，你就可以利用这份数据做其他的事情，不一定只按照PaddlePaddle提供的代码来使用这份数据。

    PaddlePaddle为了方便用户使用，将一些常用的小型数据集的下载逻辑都封装在了PaddlePaddle中，对用户透明，如果遇到类似的问题，即模型要使用的数据集报错，如果确认自己的使用逻辑没有问题，就尝试重新下载该数据。

+ 问题研究：
数据是任何模型的根基，所谓深度学习，其实就是通过非常多的参数构建一个函数，用该函数来描述训练数据的分布，所以在测试模型时，通常要求使用同分布的数据来进行测试，波士顿房间预测模型也这样，每当遇到数据问题时，从两个方面考虑，一方面考虑代码使用错误，即使用的数据属性是数据集中不存在的，这种情况你就需要修改你使用的方式，另一方面考虑是否数据集下载时遇到了问题，比如网络不稳定，或磁盘满了等情况，这种情况就需要清理一下环境重新下载一下数据。


## `已审阅` 4.问题：No modul named Ipython

+ 关键词：`modul` `Ipython`

+ 问题描述:
PaddlePaddle安装成功，运行了PaddlePaddle官网首页的程序是正常的，但运行其他程序报错，错误为`No modul named Ipython`


+ 报错截图：
![](https://ai.bdstatic.com/file/8C972E2F16124A98BED75994390E3C5F)


+ 解决方案：
这个是因为您看的教程不是Paddle最新的Fluid教程, 是v2版本的教程, paddle v2版本依赖python的ipython包, 你可以使用 `pip install ipython`安装, 或者参考ipython的官网: https://ipython.org/install.html 进行安装

+ 问题分析：
通常而言，使用pip安装某个第三库时，pip会检查当前的python环境中已经存在的第三方库，判断安装的库所需要的其他依赖库是否已经存在于本地，如果已经存在，就不在下载，如果不存在，pip就会自动将这些依赖库也一同下载了，通常不会出现库依赖问题。当然，在后期你可以删了某些必要的库，就会出现`No modul named xxx`，此时根据提示，安装相应的库则可。

+ 问题拓展：
一个python环境使用久了，很可能对某些库进行了一些删改，这些删改可能就会影响到其他库的使用，这类问题其实都很好解决，更具提示，通过pip将需要的库安装则可。

+ 问题研究：
环境依赖问题，根据系统给出的报错提示，进行相应的操作则可，展开来说，这类问题的解决方法就在报错信息中，更加报错信息，增加改动当前的开发环境则可。


## `已审阅` 5.问题：The kernel appears to have died. It will restart automatically

+ 关键词：`jupyter notebook` `paddle.init` `kernel崩溃`

+ 问题描述：在window 10上新建了一个python2.7的paddlepadddle环境。
按照教程入门教程，在jupyter notebook上面执行第二步paddle.init时，`jupyter提示kernel崩溃`

+ 报错代码段：
```python
import paddle.v2 as paddle
# Initialize PaddlePaddle.
paddle.init(use_gpu=False, trainer_count=1) #kernel崩溃
```

+ 报错输出：
```python
[I 05:42:34.183 NotebookApp] 302 GET / (172.17.0.1) 0.66ms
[I 05:42:43.970 NotebookApp] Writing notebook-signing key to /root/.local/share/jupyter/notebook_secret
[W 05:42:43.973 NotebookApp] Notebook 01.fit_a_line/README.cn.ipynb is not trusted
[I 05:42:45.933 NotebookApp] Kernel started: cd1df3d8-56ba-4acf-8c6b-bdd51915ce1e
[I 05:42:45.947 NotebookApp] 302 GET /notebooks/01.fit_a_line/image/train_and_test.png (172.17.0.1) 4.33ms
[I 05:42:45.954 NotebookApp] 302 GET /notebooks/01.fit_a_line/image/ranges.png (172.17.0.1) 6.63ms
[I 05:42:45.963 NotebookApp] 302 GET /notebooks/01.fit_a_line/image/predictions.png (172.17.0.1) 1.31ms
[I 05:42:47.446 NotebookApp] Adapting to protocol v5.1 for kernel cd1df3d8-56ba-4acf-8c6b-bdd51915ce1e
[I 05:43:27.926 NotebookApp] KernelRestarter: restarting kernel (1/5), keep random ports
WARNING:root:kernel cd1df3d8-56ba-4acf-8c6b-bdd51915ce1e restarted
```

+ 复现方式：
在Windows 10下，使用Jupyter Notebook运行波士顿房价预测模型，出现kernel崩溃

+ 解决方案：
首先尝试不使用Jupyter Notebook而直接使用命令行来运行模型预测代码，如果不使用Jupyter NoteBook代码运行正常，则说明PaddlePaddle与模型代码本身没有什么问题，可能是Jupyter NoteBook本身的一些问题，如依赖环境冲突等，推荐使用anaconda来安装Jupyter NoteBook，如果使用命令行依旧出现问题，则先判断系统是否支持AVX命令集，使用下面命令查看
```bash
if cat /proc/cpuinfo | grep -i avx; then echo Yes; else echo No; fi
```
输出Yes表示电脑支持AVX命令集，输出No表示电脑不支持，判断自己安装的PaddlePaddle是否是相应的版本，电脑不支持AVX命令集的请按照no-avx版本的paddlepaddle

+ 问题分析：
Jupyter Notebook内核崩溃的原因有很多种可能，这是个很泛化的问题，要解决这种问题，要关注造成内核崩溃的那句代码，再从大的方向去排查错误，两个大的方向，一个是你的Jupyter Notebook本身可能就有问题，比如安装时其实就没有完全成功，另一个就是你的代码使用时有问题，区分是哪一种简单而言就是将这段代码拿出来直接使用python命令行来运行，不借助Jupyter Notebook，如果可以成功，那么你就应该尝试从新安装Jupyter Notebook了，如果失败，则关注你的使用代码，因为此时不论是不是在Jupyter Notebook上运行，代码都会报错。

+ 问题拓展：
在开发时，我们会遇到这种常见的问题，这种问题从报错信息看，都是笼统了，因为有很多种原因都可以造成这一的错误，此时报错信息中的内容也不会给出具体的解决方式，不像某些错误，缺少了什么modul，你安装一下相应的modul则可，遇到这种造成原因有多种的情况，第一步通常是判断造成当前这个问题最有可能的原因，回忆一下报错前所作的操作，或者通过排除法，将这些情况一个个分离出来，试验一下看代码是否可以运行，从而判断出是什么原因导致了这种报错，知道了原因后，才方便自己针对性的去修改。

+ 问题研究：
Jupyter Notebook内核崩溃很多时候是资源崩溃，几个常见原因，没有做任何处理的大量读入数据进内存、向系统索要不存在的资源，当然不排除Jupyter Notebook本身就存在问题，解决这里问题，保持清晰的思路，找到多种可能的原因，利用排除法的方式逐一将可能原因排除，找到真正的原因，知道了报错的这种原因，才好进一步修改。针对本问题而言，就是`paddle.init`报错，那很有可能就是向系统索取相应资源时，无法获得该资源，导致Jupyter Notebook内核崩溃。


## `已审阅` 6.问题：Fatal Python error: PyThreadState_Get: no current thread

+ 关键词：`brew` `anaconda` `no current thread`

+ 问题描述：
成功安装PaddlePaddle后，运行波士顿房价预测的代码，报错`Fatal Python error: PyThreadState_Get: no current `

+ 报错输出：
```python
Fatal Python error: PyThreadState_Get: no current thread
```

+ 复现方式：
在Mac中同时存在brew安装的python2.7与anaconda版本python2.7，在anaconda版本的python下安装PaddlePaddle，安装成功，使用安装成功的PaddlePaddle执行房间预测模型报`Fatal Python error: PyThreadState_Get: no current thread`

+ 解决方法：
该问题是由于brew的python和anaconda的python冲突造成的，解决方法如下：

1. 执行 `otool -L /anaconda2/lib/python2.7/site-packages/py_paddle/_swig_paddle.so`

会得到如下输出：

```python
/anaconda2/lib/python2.7/site-packages/py_paddle/_swig_paddle.so 
/System/Library/Frameworks/CoreFoundation.framework/Versions/A/CoreFoundation (compatibility version 150.0.0, current version 1445.12.0) 
/System/Library/Frameworks/Security.framework/Versions/A/Security (compatibility version 1.0.0, current version 58286.20.16) 
/usr/local/opt/python/Frameworks/Python.framework/Versions/2.7/Python (compatibility version 2.7.0, current version 2.7.0) 
/usr/lib/libc++.1.dylib (compatibility version 1.0.0, current version 400.9.0) 
/usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 1252.0.0)
```

可以发现，并没有/usr/local/opt/python/Frameworks/Python.framework/Versions/2.7/Python 这个路径。

2. 执行`install_name_tool -change /usr/local/opt/python/Frameworks/Python.framework/Versions/2.7/Python /anaconda/lib/libpython2.7.dylib /anaconda/lib/python2.7/site-packages/py_paddle/_swig_paddle.so`

此时再通过PaddlePaddle运行波士顿房价预测的代码，不会再出现上述问题。

+ 问题分析：
PyThreadState_GET()方法是python内核中的方法，该方法主要用于python线程方面的操作，而线程其实涉及到了对系统资源的调用，当系统本地中有多种不同的python且没有做环境隔离，就可能会发生python版本冲突的问题，冲突问题的表现形式可能就是`Fatal Python error: PyThreadState_Get: no current thread`，因为是内核级的代码，我们通常无需去修改，也难以修改，成本太高，所以更建议的方法就是修改系统中的环境，如解决方法中所使用的方法，通过相应的配置，修改python的开发环境，避免python版本冲突的发生。

+ 问题拓展：
通常而言，内核级的问题都是比较严重的问题，所以这种级别的问题是会被快速修复的，如果你使用的python稳定版本中，出现了这种级别的问题，通常都是环境问题，如版本冲突或系统资源限制等，要解决这个问题最好就是对python的版本进行控制，通常可以使用pyenv、virtualenv等工具，pyenv只支持linux与mac，使用这些工具为不同版本的python创建独立的虚拟开发环境，这些开发环境不会影响到本地环境，做了很好的隔离，当然对于具体的问题，如`Fatal Python error: PyThreadState_Get: no current thread`也可以使用具体的解决方法。

+ 问题研究：
PyThreadState_GET是python内核中的一个方法，其部分相关内核代码如下：

    ```python
    void PyErr_Restore(PyObject *type, PyObject *value, PyObject *traceback)
    {
        PyThreadState *tstate = PyThreadState_GET();
        PyObject *oldtype, *oldvalue, *oldtraceback;

        if (traceback != NULL && !PyTraceBack_Check(traceback)) {
            /* XXX Should never happen -- fatal error instead? */
            /* Well, it could be None. */
            Py_DECREF(traceback);
            traceback = NULL;
        }

        // 保存以前的异常信息
        oldtype = tstate->curexc_type;
        oldvalue = tstate->curexc_value;
        oldtraceback = tstate->curexc_traceback;
        // 设置当前的异常信息
        tstate->curexc_type = type;
        tstate->curexc_value = value;
        tstate->curexc_traceback = traceback;
        // 抛弃以前的异常信息
        Py_XDECREF(oldtype);
        Py_XDECREF(oldvalue);
        Py_XDECREF(oldtraceback);
    }
    ```

    python通过PyThreadState_GET()可以获得当前获得线程，并将异常信息存放到了线程状态对象中。

    python内核级的代码通常是不会有什么报错的，但如果遇到了这个级别的错误，第一个要考虑的依旧是开发环境问题，针对`Fatal Python error: PyThreadState_Get: no current thread`而言，它通常出现在mac系统中，常见的原因就是mac中存在多个python环境，一个优雅的方式就是在mac上使用pyenv，这样就可以通过pyenv来隔绝系统原本代码的brew安装的python与其他自己后面安装的python相互隔离了。


## `已审阅` 7.问题：报错张量类型不正确

+ 版本号：`1.0.1`

+ 标签：`张量类型`

+ 问题描述：使用PaddlePaddle编写好网络运行时，报张量类型不正确的错误，我反复检查了自己的网络结构中张量类型相关的定义，并打印了传入数据的类型，类型都相匹配，但运行时就是报张量类似错误，逐步排查后，发现是训练代码段会抛出该错误，不知道如果修改？

+ 报错信息：

```bash
EnforceNotMet: Tensor holds the wrong type, it holds l at [/paddle/paddle/fluid/framework/tensor_impl.h:29]
PaddlePaddle Call Stacks: 
0       0x7fe0624486b6p paddle::platform::EnforceNotMet::EnforceNotMet(std::__exception_ptr::exception_ptr, char const*, int) + 486
1       0x7fe0624501c0p float const* paddle::framework::Tensor::data<float>() const + 192
2       0x7fe06278f813p void paddle::operators::ElementwiseComputeEx<paddle::operators::SubFunctor<float>, paddle::platform::CPUDeviceContext, float, float>(paddle::framework::ExecutionContext const&, paddle::framework::Tensor const*, paddle::framework::Tensor const*, int, paddle::operators::SubFunctor<float>, paddle::framework::Tensor*) + 67
3       0x7fe062936e53p paddle::operators::ElementwiseSubKernel<paddle::platform::CPUDeviceContext, float>::Compute(paddle::framework::ExecutionContext const&) const + 323
4       0x7fe062936ed3p std::_Function_handler<void (paddle::framework::ExecutionContext const&), paddle::framework::OpKernelRegistrarFunctor<paddle::platform::CPUPlace, false, 0ul, paddle::operators::ElementwiseSubKernel<paddle::platform::CPUDeviceContext, float>, paddle::operators::ElementwiseSubKernel<paddle::platform::CPUDeviceContext, double>, paddle::operators::ElementwiseSubKernel<paddle::platform::CPUDeviceContext, int>, paddle::operators::ElementwiseSubKernel<paddle::platform::CPUDeviceContext, long> >::operator()(char const*, char const*) const::{lambda(paddle::framework::ExecutionContext const&)#1}>::_M_invoke(std::_Any_data const&, paddle::framework::ExecutionContext const&) + 35
5       0x7fe062fc52ecp paddle::framework::OperatorWithKernel::RunImpl(paddle::framework::Scope const&, boost::variant<paddle::platform::CUDAPlace, paddle::platform::CPUPlace, paddle::platform::CUDAPinnedPlace, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_> const&) const + 492
6       0x7fe062fc171fp paddle::framework::OperatorBase::Run(paddle::framework::Scope const&, boost::variant<paddle::platform::CUDAPlace, paddle::platform::CPUPlace, paddle::platform::CUDAPinnedPlace, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_> const&) + 255
7       0x7fe0625094eap paddle::framework::Executor::RunPreparedContext(paddle::framework::ExecutorPrepareContext*, paddle::framework::Scope*, bool, bool, bool) + 298
8       0x7fe062509ee0p paddle::framework::Executor::Run(paddle::framework::ProgramDesc const&, paddle::framework::Scope*, int, bool, bool) + 128
9       0x7fe06242ff5dp
10      0x7fe06247ade4p pybind11::cpp_function::dispatcher(_object*, _object*, _object*) + 2596
11            0x4e9ba7p PyCFunction_Call + 119
12            0x53c6d5p PyEval_EvalFrameEx + 23029

```

+ 问题复现

```
def train_program():
    # feature vector of length 13
    x = fluid.layers.data(name='x', shape=[11], dtype='float32')
    y_predict = fluid.layers.fc(input=x, size=1, act=None)

    y = fluid.layers.data(name='y', shape=[1], dtype='int64')
    loss = fluid.layers.square_error_cost(input=y_predict, label=y)
    avg_loss = fluid.layers.mean(loss)

    return avg_loss
```

+ 问题分析：张量类型错误有多种可能，但就提供的代码而言，可能是损失函数定义的类型不满足当前的网络结构，出错代码中使用了square_error_cost，即平方差损失，导致代码报错。

+ 问题解决：
使用交叉熵损失函数代替平方差损失函数

```
paddle.fluid.layers.cross_entropy(input, label, soft_label=False, ignore_index=-100)
```

+ 问题拓展：
损失函数（loss function）是用来估量你模型的预测值f(x)与真实值Y的不一致程度，它是一个非负实值函数,通常使用L(Y, f(x))来表示，损失函数越小，模型的鲁棒性就越好。损失函数是经验风险函数的核心部分，也是结构风险函数重要组成部分。损失函数有多种，选择适合的损失函数对模型的训练有很大的帮助。

## `已审阅` 8.问题：训练时，输出的损失值为nan

+ 版本号：`1.0.1`

+ 标签：`损失值` `nan`

+ 问题描述：使用PaddlePaddle训练神经网络时，没有报错，但输出结果一直为nan

+ 报错信息：

```
Train cost, Step 0, Cost nan
Train cost, Step 100, Cost nan
Train cost, Step 200, Cost nan
Train cost, Step 300, Cost nan
Train cost, Step 400, Cost nan
Train cost, Step 500, Cost nan
Train cost, Step 600, Cost nan
Train cost, Step 700, Cost nan
```

+ 问题复现

```
def train_program():
    # feature vector of length 13
    x = fluid.layers.data(name='x', shape=[11], dtype='float32')
    y_predict = fluid.layers.fc(input=x, size=6, act=None)

    y = fluid.layers.data(name='y', shape=[1], dtype='int64')
    loss = fluid.layers.cross_entropy(input=y_predict, label=y)
    avg_loss = fluid.layers.mean(loss)

    return avg_loss
```

+ 问题分析：神经网络的输出结果与神经网络的最后一层有很大的关系，因为整个神经网络的输出结果其实就是最后一层结构的输出结构，根据描述，网络在训练运行时并没有报错，但输出与预期不符合，很有可能就是最后一层没有使用正确的激活函数。

+ 问题解决：
从报错代码端中可以看出，最后一层y_predict的act参数为None，即没有使用激活函数，这很可能导致神经网络输出错误结果，在最后一层使用Softmax激活函数

```
def train_program():
    # feature vector of length 13
    x = fluid.layers.data(name='x', shape=[11], dtype='float32')
    y_predict = fluid.layers.fc(input=x, size=6, act='softmax')

    y = fluid.layers.data(name='y', shape=[1], dtype='int64')
    loss = fluid.layers.cross_entropy(input=y_predict, label=y)
    avg_loss = fluid.layers.mean(loss)

    return avg_loss
```

+ 问题拓展：softmax用于多分类过程中，它将多个神经元的输出，映射到（0,1）区间内，可以看成概率来理解，从而来进行多分类，直观解释如下图：

![](https://pic1.zhimg.com/80/v2-87b232ab0e292a536e94b73952caadd0_hd.jpg)

# `已审阅` 9.问题：`EnforceNotMet: Enforce failed`
+ 报错信息

```
EnforceNotMet: Enforce failed. Expected lbl < class_num, but received lbl:6 >= class_num:6.
 at [/paddle/paddle/fluid/operators/math/cross_entropy.cc:52]
PaddlePaddle Call Stacks: 
0       0x7f39995286b6p paddle::platform::EnforceNotMet::EnforceNotMet(std::__exception_ptr::exception_ptr, char const*, int) + 486
1       0x7f3999fd4d0ep paddle::operators::math::CrossEntropyFunctor<paddle::platform::CPUDeviceContext, float>::operator()(paddle::platform::CPUDeviceContext const&, paddle::framework::Tensor*, paddle::framework::Tensor const*, paddle::framework::Tensor const*, bool, int) + 6190
2       0x7f3999eda038p paddle::operators::CrossEntropyOpKernel<paddle::platform::CPUDeviceContext, float>::Compute(paddle::framework::ExecutionContext const&) const + 472
3       0x7f3999eda1a3p std::_Function_handler<void (paddle::framework::ExecutionContext const&), paddle::framework::OpKernelRegistrarFunctor<paddle::platform::CPUPlace, false, 0ul, paddle::operators::CrossEntropyOpKernel<paddle::platform::CPUDeviceContext, float>, paddle::operators::CrossEntropyOpKernel<paddle::platform::CPUDeviceContext, double> >::operator()(char const*, char const*) const::{lambda(paddle::framework::ExecutionContext const&)#1}>::_M_invoke(std::_Any_data const&, paddle::framework::ExecutionContext const&) + 35
4       0x7f399a0a52ecp paddle::framework::OperatorWithKernel::RunImpl(paddle::framework::Scope const&, boost::variant<paddle::platform::CUDAPlace, paddle::platform::CPUPlace, paddle::platform::CUDAPinnedPlace, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_> const&) const + 492
5       0x7f399a0a171fp paddle::framework::OperatorBase::Run(paddle::framework::Scope const&, boost::variant<paddle::platform::CUDAPlace, paddle::platform::CPUPlace, paddle::platform::CUDAPinnedPlace, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_> const&) + 255
6       0x7f39995e94eap paddle::framework::Executor::RunPreparedContext(paddle::framework::ExecutorPrepareContext*, paddle::framework::Scope*, bool, bool, bool) + 298
7       0x7f39995e9ee0p paddle::framework::Executor::Run(paddle::framework::ProgramDesc const&, paddle::framework::Scope*, int, bool, bool) + 128
8       0x7f399950ff5dp
```

+ 问题复现

```
def getdata():
    def reader():
        for i in range(len(data_y)):
            yield np.array(data_X[i]).astype('float32'), np.array(data_y[i]).astype('int64')
    return reader
```


+ 问题解决
PaddlePaddle的label要从0开始递增。

```
def getdata():
    def reader():
        for i in range(len(data_y)):
            yield np.array(data_X[i]).astype('float32'), np.array(data_y[i]- 3).astype('int64')
    return reader
```