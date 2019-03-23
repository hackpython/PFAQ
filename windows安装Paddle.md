# Windows安装Paddle

## `已审核`1.问题：在Windows下安装1.3版本的paddle后训练报错

+ 版本号：`1.3.0`

+ 标签：`Windows10` `1.3版本`

+ 问题描述：在windows 10下安装1.3版本的Paddle，安装成功，但使用1.3版的Paddle训练时报错，此前使用1.2版本的Paddle可以正常使用。

+ 报错输出：

```
Traceback (most recent call last):
  File "C:\Python35\lib\site-packages\paddle\fluid\framework.py", line 38, in <module>
    from . import core
ImportError: DLL load failed: 动态链接库(DLL)初始化例程失败。

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "D:/ubuntu/paddlepaddle/pycharm/PycharmProjects/LearnPaddle2/note1/test_paddle.py", line 3, in <module>
    import paddle.fluid as fluid
  File "C:\Python35\lib\site-packages\paddle\fluid\__init__.py", line 18, in <module>
    from . import framework
  File "C:\Python35\lib\site-packages\paddle\fluid\framework.py", line 47, in <module>
    (executable_path, executable_path, cpt.get_exception_message(e)))
ImportError: NOTE: You may need to run "set PATH=C:\Python35;%PATH%"
        if you encounters "DLL load failed" errors. If you have python
        installed in other directory, replace "C:\Python35" with your own
        directory. The original error is: 
 DLL load failed: 动态链接库(DLL)初始化例程失败。
 ```

+ 问题分析：

问题报错中，可以看出，是缺少了相应的DLL，可以使用Dependency Walker工具来查看系统的DLL文件，该工具地址为：http://www.dependencywalker.com/

+ 解决方法：

通过Dependency Walker工具查看系统的DLL文件，如下图

![](https://raw.githubusercontent.com/ayuliao/images/master/PaddleDLL%E7%BC%BA%E5%B0%91.png)

从图中看出，系统中缺mkldnn.dll和libiomp5md.dll，而Paddle考虑到这方面的因素，在site-packages\paddle\libs路径下，已经准备好了mkldnn.dll和libiomp5md.dll，你只需要将其拷贝到系统路径比如C:/windows/system32下则可。

## `待审核`2.问题：Windows10 家庭版可以安装吗？

+ 版本号：`1.3.0`

+ 标签：`Windows10` `1.3.0`

+ 问题描述：Windows10 家庭版可以安装吗？

+ 问题分析：目前Paddle对window7、window8、windows10都已经有比较好的支持，windows10家庭版也可以进行安装，但不同人安装的不同系统中，可能存在差异，即缺少一下需要的DLL文件，需要自行下载或直接将site-packages\paddle\libs路径下的DLL复制到系统目录

+ 解决方法：可以安装，参考http://www.paddlepaddle.org/documentation/docs/zh/1.3/beginners_guide/index.html ，与	Windows 10 专业版/企业版 安装方式类似。

## `待审核`3.问题：paddlepaddle是否支持使用conda安装和支持MKL？

+ 版本号：`1.1.0`

+ 标签：`conda安装` `MKL`

+ 问题描述：以前在安装tensorflow的时候，发现tensorflow支持使用conda安装，很方便。无论是windows还是Linux，只需要一个命令：
conda install tensorflow-gpu

paddlepaddle什么时候才能提供这样的安装方式？

+ 问题解答：

目前还没有支持conda，但是我们会尽快列入计划

你可以尝试使用下面命令来一键安装

```
pip install paddlepaddle
```

目前Fluid release1.3版本支持windows MKL功能。而Fluid Linux版本一直支持MKL功能。


## `待审核`4.问题：Windows下pip安装成功但import paddle.fluid验证安装失败

+ 版本号：`1.2.1`

+ 标签：`windows` `import失败`

+ 问题描述：

版本、环境信息：
   1）PaddlePaddle版本：1.2.1
   2）CPU：Intel(R) Core(TM) i5-7500
   3）操作系统：Windows 7 专业版 + Python 3.5.1

安装方式信息：
pip安装：pip3 install paddlepaddle

+ 报错输出：

```
Traceback (most recent call last):
  File "C:\Users\Administrator\AppData\Local\Programs\Python\Python35\lib\site-packages\paddle\fluid\framework.py", line 27, in <module>
    from . import core
ImportError: DLL load failed: 找不到指定的程序。

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "E:/Output/Python_output/PARL/PARL-kog/main.py", line 1, in <module>
    import paddle.fluid
  File "C:\Users\Administrator\AppData\Local\Programs\Python\Python35\lib\site-packages\paddle\fluid\__init__.py", line 18, in <module>
    from . import framework
  File "C:\Users\Administrator\AppData\Local\Programs\Python\Python35\lib\site-packages\paddle\fluid\framework.py", line 33, in <module>
    directory. The original error is: \n""" + cpt.get_exception_message(e))
ImportError: NOTE: You may need to run "export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH"
    if you encounters "libmkldnn.so not found" errors. If you have python
    installed in other directory, replace "/usr/local/lib" with your own
    directory. The original error is: 
DLL load failed: 找不到指定的程序。
```

+ 问题分析：

从报错输出可以看出，缺少量必要DLL，从具体的报错细节来看，一般情况下是因为python35.dll不在path环境变量里面，可以尝试手工加到path环境变量中再试一下

+ 解决方法：

1. 用Administrator账号权限运行一下。
2. 如果这样还不行，可以试验一下用virtualenv，具体步骤是
a. pip install virtualenv
	b. virtualenv test
	c. cd test
	d. scripts\activate
	e. pip install paddlepaddle

或者尝试使用python3.6来安装PaddlePaddle，主要安装python3.6时，勾选 Add Python 3.6 to PATH

![](https://raw.githubusercontent.com/ayuliao/images/master/Paddlepython3.6.png)


## `待审核`5.问题：windows中是否可以同时安装CPU与GPU版本的Paddle？

+ 版本号：`1.1.0`

+ 标签：`Windows` `CPU` `GPU`

+ 问题描述：windows中是否可以同时安装CPU与GPU版本的Paddle？

+ 问题分析：

可以同时安装，两种并没有冲突，但实际上并没有必要同时安装CPU与GPU版本的Paddle

建议安装GPU版本，一方面GPU的计算性能更好，另一方面它也可以运行CPU的训练任务


## `待审核`6.问题：Windows环境模型预测 (c++)教程无法走通 生成失败

+ 版本号：`1.3.0`

+ 标签：`windows` `模型预测`

+ 问题描述：PaddlePaddle版本为1.3 win10 cpu ，在走Windows环境模型预测这个教程的时候遇到了错误，在vs2015中生成解决方案失败。

实践上与官网唯一有改动的地方在cmake命令上
cmake .. -G "Visual Studio 14 2015 Win64" -DWITH_GPU=OFF -DWITH_MKL=OFF -DWITH_STATIC_LIB=ON -DCMAKE_BUILD_TYPE=Release -DDEMO_NAME=simple_on_word2vec -DPADDLE_LIB=C:\Paddle-develop\fluid_inference
最后的路径指向了解压后的预测库根目录。

http://paddlepaddle.org/documentation/docs/zh/1.3/advanced_usage/deploy/inference/build_and_install_lib_cn.html

+ 报错输出：

```
1>------ 已启动生成: 项目: simple_on_word2vec, 配置: Release x64 ------
1> simple_on_word2vec.cc
1>C:\Paddle-develop\fluid_inference\third_party\install\gflags\include\gflags/gflags.h(288): error C2065: “unused”: 未声明的标识符
1>C:\Paddle-develop\fluid_inference\third_party\install\glog\include\glog/logging.h(47): fatal error C1083: 无法打开包括文件: “unistd.h”: No such file or directory
========== 生成: 成功 0 个，失败 1 个，最新 1 个，跳过 0 个 ==========
```

+ 问题分析：

通过问题描述与报错输出可知，应该时glog库有问题

+ 解决方法：

官网上的inference包默认是linux下的，所以win下的需要自己去编译，依照这个文档，http://paddlepaddle.org/documentation/docs/zh/1.3/beginners_guide/install/compile/compile_Windows.html 编译， 完成后在build的目录下就有fluid_inference_install_dir目录

## `待审核`7.问题：关于安装Paddle时pip3和pip的区别

+ 版本号：`1.1.0`

+ 标签：`pip` `windows`

+ 问题描述：看到官网说在Windows 下的安装命令是pip install paddlepaddle或者是pip3 install paddlepaddle.
我想问的是，使用pip install paddlepaddle安装的paddlepaddle是不是只支持python 2.7而使用
pip3 install paddlepaddle命令安装的paddlepaddle是不是只支持python 3.x？
还是说两个命令安装的paddlepaddle没有区别，是同时支持python 2.7和python 3.x的？

+ 解决方法：

如果您是使用Python3.x做开发，需要用pip3安装

paddlepaddle预计1.3.0版本支持在Windows下进行GPU训练

## `待审核`8.问题：PaddlePaddle在安装过程中，采用什么类型的开发环境会好点

+ 版本号：`1.1.0`

+ 标签：`开发环境`

+ 问题描述：PaddlePaddle在安装过程中，采用什么类型的开发环境会好点，例如linux，windows，或者unix等，开发环境对运行性能有差异吗？

+ 解决方法：
Paddle支持的环境包括：Ubuntu, CentOS, MacOS, Windows7/8/10，详细参考 http://www.paddlepaddle.org/documentation/docs/zh/1.2/beginners_guide/install/index_cn.html 。 建议Linux开发环境，目前发型版本只有Linux支持GPU。 性能差异取决于硬件。


## `待审核`9.问题：安装错误win10专业版fluid1.2，通过pip安装，报错“image找不到”

+ 版本号：`1.2.0`

+ 标签：`windows` `pillow`

+ 问题描述：，我的笔记本电脑是 win10专业版，anaconda2，python 3.6，pip version 18.1
命令： pip install paddlepaddle

+ 报错输出：

![](https://raw.githubusercontent.com/ayuliao/images/master/paddlepillowerror.png)

+ 问题分析：

从报错可以看出是pillow的问题，pillow中，如果_imaging 对应的C模块不存在，pillow加载使用它时就会报错。

+ 解决方法：

尝试重新安装

```
pip install Pillow

Collecting Pillow
  Downloading Pillow-5.1.0-cp36-cp36m-manylinux1_x86_64.whl (2.0MB)
    100% |████████████████████████████████| 2.0MB 753kB/s
Installing collected packages: Pillow
Successfully installed Pillow-5.1.0
```

然后使用 `from PIL import Image` 来代替

更多细节参考：https://stackoverflow.com/questions/9558562/the-imaging-c-module-is-not-installed-on-windows


## `待审核`10.问题：Windows下pip安装失败

+ 版本号：`1.2.0`

+ 标签：`windows` `pip`

+ 问题描述：

系统环境

PaddlePaddle版本1.2
CPU型号Intel(R) Core(TM) i5-5257U
操作系统 Windows10 专业版
安装方式

下载安装python
https://www.python.org/ftp/python/3.7.2/python-3.7.2-amd64.exe
安装PaddlePaddle
pip3 install paddlepaddle

+ 报错输出：

```
import paddle.fluid
Traceback (most recent call last):
File "", line 1, in 
File "D:\Python-3.7.2\lib\site-packages\paddle_init_.py", line 25, in 
import paddle.dataset
File "D:\Python-3.7.2\lib\site-packages\paddle\dataset_init_.py", line 29, in 
import paddle.dataset.flowers
File "D:\Python-3.7.2\lib\site-packages\paddle\dataset\flowers.py", line 38, in 
import scipy.io as scio
File "D:\Python-3.7.2\lib\site-packages\scipy\io_init_.py", line 97, in 
from .matlab import loadmat, savemat, whosmat, byteordercodes
File "D:\Python-3.7.2\lib\site-packages\scipy\io\matlab_init_.py", line 13, in 
from .mio import loadmat, savemat, whosmat
File "D:\Python-3.7.2\lib\site-packages\scipy\io\matlab\mio.py", line 10, in 
from .miobase import get_matfile_version, docfiller
File "D:\Python-3.7.2\lib\site-packages\scipy\io\matlab\miobase.py", line 22, in 
from scipy.misc import doccer
File "D:\Python-3.7.2\lib\site-packages\scipy\misc_init_.py", line 68, in 
from scipy.interpolate.pade import pade as pade
File "D:\Python-3.7.2\lib\site-packages\scipy\interpolate_init.py", line 175, in 
from .interpolate import *
File "D:\Python-3.7.2\lib\site-packages\scipy\interpolate\interpolate.py", line 21, in 
import scipy.special as spec
File "D:\Python-3.7.2\lib\site-packages\scipy\special_init.py", line 641, in 
from ._ufuncs import *
File "_ufuncs.pyx", line 1, in init scipy.special._ufuncs
ImportError: DLL load failed: 找不到指定的模块。
```
+ 解决方法：

请尝试利用pip重装或更新scipy。

建议使用anaconda的python环境，安装万anaconda python后，利用conda更新所有的包，然后再尝试通过pip安装paddlepaddle

```
conda update --all
```

如果依旧出现`File "_ufuncs.pyx", line 1, in init scipy.special._ufuncs`，尝试安装mkl

```
conda install mkl
```
