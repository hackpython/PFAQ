# MacOS安装PaddlePaddle

## `已审阅` 1.问题：基于Docker编译Paddle与MacOS本机编译Paddle的疑问

+ 关键字：`Docker编译安装` `MacOS本机编译安装`

+ 问题描述：PaddlePaddle官方文档中，关于MacOS下安装PaddlePaddle只提及了MacOS中使用Docker环境安装PaddlePaddle的内容，没有Mac本机安装的内容

+ 问题讨论：
基于Docker容器编译PaddlePaddle与本机上直接编译PaddlePaddle，所使用的编译执行命令是不一样的，但是官网仅仅给出了基于Docker容器编译PaddlePaddle所执行的命令。

+ 问题解答：

	1.基于Docker容器编译PaddlePaddle，需要执行：

	```bash
	# 1. 获取源码

	git clone https://github.com/PaddlePaddle/Paddle.git

	cd Paddle

	# 2. 可选步骤：源码中构建用于编译PaddlePaddle的Docker镜像

	docker build -t paddle:dev .

	# 3. 执行下面的命令编译CPU-Only的二进制

	docker run -it -v $PWD:/paddle -e "WITH_GPU=OFF" -e "WITH_TESTING=OFF" paddlepaddle/paddle_manylinux_devel:cuda8.0_cudnn5 bash -x /paddle/paddle/scripts/paddle_build.sh build

	# 4. 或者也可以使用为上述可选步骤构建的镜像（必须先执行第2步）

	docker run -it -v $PWD:/paddle -e "WITH_GPU=OFF" -e "WITH_TESTING=OFF" paddle:dev
	```

	2.直接在本机上编译PaddlePaddle，需要执行：

	```bash
	# 1. 使用virtualenvwrapper创建python虚环境并将工作空间切换到虚环境

	mkvirtualenv paddle-venv

	workon paddle-venv

	# 2. 获取源码

	git clone https://github.com/PaddlePaddle/Paddle.git

	cd Paddle

	# 3. 执行下面的命令编译CPU-Only的二进制

	mkdir build && cd build

	cmake .. -DWITH_GPU=OFF -DWITH_TESTING=OFF

	make -j$(nproc)
	```

    更详细的内容，请参考[官方文档](http://www.paddlepaddle.org/documentation/docs/zh/1.0/beginners_guide/install/install_MacOS.html#docker)

## `已审阅` 2.问题：Configuring incomplete, errors occured!

+ 关键字：`Configuring incomplete`

+ 问题描述：以源码方式在MacOS上安装时，出现`Configuring incomplete, errors occured!`

+ 报错截图：
![](https://user-images.githubusercontent.com/17102274/42515239-e24be824-848d-11e8-9f3d-3baf156dcea8.png)

	![](https://user-images.githubusercontent.com/17102274/42515246-e6f7c2d0-848d-11e8-853a-7d7401e4650f.png)

+ 解决方法：

	安装PaddlePaddle编译时需要的各种依赖则可，如下：

	```bash
	pip install wheel
	brew install protobuf@3.1
	pip install protobuf==3.1.0
	```

	如果执行pip install protobuf==3.1.0时报错，输出下图内容：

	![](https://user-images.githubusercontent.com/17102274/42515286-fb7a7b76-848d-11e8-931a-a7f61bd6374b.png)

	从图中可以获得报错的关键为`Cannot uninstall 'six'`，那么解决方法就是先安装好`six`，再尝试安装`protobuf 3.1.0`如下：

	```bash
	easy_install -U six 
	pip install protobuf==3.1.0
	```


## `已审阅`3.问题：python import fluid error

+ 关键字：

+ 问题描述：mac pip安装paddle，import paddle.v2 没有问题，调用init也没有问题，
	但是import paddle.v2.fluid 就会出现下面的错误：
	
	```bash
	import paddle.v2.fluid as fluid
	Fatal Python error: PyThreadState_Get: no current thread
	Abort trap: 6
	```

+ 解决方法：
需要使用brew install python作为python环境。Paddle发布的mac版本的包是基于brew python编译的。

## `已审阅` 4.问题：/bin/sh: wget: command not found

+ 关键字：`wget`

+ 问题描述：MacOS 10.12下编译PaddlePaddle出现`/bin/sh: wget: command not found`

+ 报错截图：
![](https://user-images.githubusercontent.com/17102274/42515304-0bd7012e-848e-11e8-966f-946361ac7a56.png)

+ 解决方法：
报错的原因从报错输出的信息中可以发现，即没有有找到wget命令，安装wget则可，安装命令如下：

```bash
brew install wget
```

## `已审阅`  5.问题：No rule to make target

+ 关键字：`CMake`

+ 问题描述：官网中只介绍了Mac下使用Docker安装编译PaddlePaddle的方式，因为我对Docker不怎么熟悉，想直接安装到本地的Mac系统中，MacOS版本为10.13，是符合要求的，但尝试了多次后，已经出现`No rule to make target`错误

+ 报错截图：
![](https://user-images.githubusercontent.com/17102274/42515324-1bd9c020-848e-11e8-8934-d7da5fc1f090.png)

+ 解决方法：
该问题是有CMake引擎的，修改CMake编译命令，打开WITH_FLUID_ONLY编译选项，修改后编译命令如下：

	```bash
	cmake .. -DWITH_FLUID_ONLY=ON -DWITH_GPU=OFF -DWITH_TESTING=OFF
	```

## `已审阅` 6.问题：[paddle/fluid/platform/CMakeFiles/profiler_py_proto.dir/all] Error 2

+ 关键字：`fluid` `CMakeFiles`

+ 问题描述：MacOS本机直接通过源码编译的方式安装PaddlePaddle出现`[paddle/fluid/platform/CMakeFiles/profiler_py_proto.dir/all] Error 2`

+ 报错截图：
![](https://user-images.githubusercontent.com/17102274/42515350-28c055ce-848e-11e8-9b90-c294b375d8a4.png)

+ 解决方法：
    使用cmake版本为3.4则可


## `已审阅` 7.问题：No such file or directory

+ 关键字：`develop分支`

+ 问题描述：
MacOS本地编译PaddlePaddle github上develop分支的代码出现，出现上面的错误

+ 报错截图：
![](https://user-images.githubusercontent.com/17102274/42515402-453cc0d4-848e-11e8-9a03-a579ea8e4d2d.png)

+ 解决方法：
因为此时develop分支上Generating build/.timestamp这一步涉及的代码还在进行修改，所以并不能保证稳定，建议切换回稳定分支进行编译安装。

	可以通过执行如下命令将分支切换到0.14.0进行编译:

	```bash
	cd Paddle
	git checkout -b release/1.1
	cd build &&  rm -rf *
	cmake .. -DWITH_FLUID_ONLY=ON -DWITH_GPU=OFF -DWITH_TESTING=OFF
	make -j4
	```

	编译成功后的结果如图：

	![](https://user-images.githubusercontent.com/17102274/42515418-4fb71e56-848e-11e8-81c6-da2a5553a27a.png)



## `已审阅` 8.问题：paddle源码编译（osx）报各种module找不到的问题

+ 关键字：`源码编译` `缺失module`

+ 问题描述：
从源码编译，最后`cmake ..`时
`Could NOT find PY_google.protobuf (missing: PY_GOOGLE.PROTOBUF)
CMake Error at cmake/FindPythonModule.cmake:27 (message):
python module google.protobuf is not found`
若通过-D设置路径后，又会有其他的如`Could not find PY_wheel`等其他找不到的情况

+ 解决方法：
![](https://cloud.githubusercontent.com/assets/728699/19915727/51f7cb68-a0ef-11e6-86cc-febf82a07602.png)
如上，当cmake找到python解释器和python库时，如果安装了许多pythons，它总会找到不同版本的Python。在这种情况下，您应该明确选择应该使用哪个python。

	通过cmake显式设置python包。只要确保python libs和python解释器是相同的python可以解决所有这些问题。当这个python包有一些原生扩展时，例如numpy，显式set python包可能会失败。

## `已审阅` 9.问题：ld terminated with signal 9 [Killed] 

+ 关键字：`编译安装`

+ 问题描述：
在MacOS下，本地直接编译安装PaddlePaddle遇到`collect2: ld terminated with signal 9 [Killed] `

+ 解决方法：
该问题是由磁盘空间不足造成的，你的硬盘要有30G+的空余空间，请尝试清理出足够的磁盘空间，重新安装。


## `已审阅` 10.问题：在Mac上无法安装numpy等Python包，权限错误

+ 关键字：`权限错误`

+ 问题描述：
因为需要安装numpy等包，但在Mac自带的Python上无法安装，导致难以将PaddlePaddle正常安装到Mac本地

+ 问题解答：
Mac上对自带的Python和包有严格的权限保护，最好不要在自带的Python上安装。建议用virtualenv建立一个新的Python环境来操作。

	virtualenv的基本原理是将机器上的Python运行所需的运行环境完整地拷贝一份。我们可以在一台机器上制造多份拷贝，并在这多个拷贝之间自由切换，这样就相当于在一台机器上拥有了多个相互隔离、互不干扰的Python环境。

+ 解决方法：
下面使用virtualenv为Paddle生成一个专用的Python环境。

	安装virtualenv，virtualenv本身也是Python的一个包，可以用pip进行安装：

	```
	 sudo -H pip install virtualenv
	```

	由于virtualenv需要安装给系统自带的Python，因此需要使用sudo权限。接着使用安装好的virtualenv创建一个新的Python运行环境：

	```
	virtualenv --no-site-packages paddle
	```

	--no-site-packages 参数表示不拷贝已有的任何第三方包，创造一个完全干净的新Python环境。后面的paddle是我们为这个新创建的环境取的名字。执行完这一步后，当前目录下应该会出现一个名为paddle（或者你取的其他名字）的目录。这个目录里保存了运行一个Python环境所需要的各种文件。

	启动运行环境：

	```
	source paddle/bin/activate
	```

	执行后会发现命令提示符前面增加了(paddle)字样，说明已经成功启动了名为‘paddle’的Python环境。执行which python，可以发现使用的已经是刚刚创建的paddle目录下的Python。

	在这个环境中，我们可以自由地进行Paddle的安装、使用和开发工作，无需担心对系统自带Python的影响。

	如果我们经常使用Paddle，我们每次打开终端后都需要执行一下source paddle/bin/activate来启动环境，比较繁琐。为了简便，可以修改终端的配置文件，来让终端每次启动后自动启动特定的Python环境。

	执行:

	```
	vi ~/.bash_profile
	```

	打开终端配置文件，并在文件的最后添加一行：

	```
	source paddle/bin/activate
	```

	这样，每次打开终端时就会自动启动名为‘paddle’的Python环境了。

