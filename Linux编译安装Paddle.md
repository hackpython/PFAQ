# Linux编译安装PaddlePaddle

## 1.问题：生成Docker镜像时，无法下载需要的golang，导致`tar: Error is not recoverable: exiting now`

+ 关键字：`golang` `Docker镜像`

+ 问题描述：
根据官方文档中提供的步骤安装Docker，出现上述问题

+ 报错截图：
![](https://user-images.githubusercontent.com/17102274/42516245-314346be-8490-11e8-85cc-eb95e9f0e02c.png)

+ 问题分析：
由上图可知，生成docker镜像时需要下载golang，访问的网址为`https://storage.googleapis.com/golang/go1.8.1.linux-amd64.tar.gz`（谷歌有关），故而使用者需要保证电脑可以科学上网。

+ 解决方法：
选择下载并使用docker.paddlepaddlehub.com/paddle:latest-devdocker镜像，执行命令如下：

```
git clone https://github.com/PaddlePaddle/Paddle.git

cd Paddle

git checkout -b 0.14.0 origin/release/0.14.0


sudo docker run --name paddle-test -v $PWD:/paddle --network=host -it docker.paddlepaddlehub.com/paddle:latest-dev /bin/bash
```

进入docker编译GPU版本的PaddlePaddle，执行命令如下：

```
mkdir build && cd build
# 编译GPU版本的PaddlePaddle
cmake .. -DWITH_FLUID_ONLY=ON -DWITH_GPU=ON -DWITH_TESTING=ON
make -j$(nproc)
```

通过上面的方式操作后：

![](https://user-images.githubusercontent.com/17102274/42516287-46ccae8a-8490-11e8-9186-985efff3629c.png)

接着安装PaddlePaddle并运行线性回归test_fit_a_line.py程序测试一下PaddlePaddle是安装成功则可

```bash
pip install build/python/dist/*.whl
python python/paddle/fluid/tests/book/test_fit_a_line.py
```

## 2.问题：GPU版本的PaddlePaddle运行结果报错

+ 关键字：`GPU` `运行报错`

+ 问题描述：
在Docker镜像上，成功安装PaddlePaddle，但一运行就报错

+ 报错截图：
![](https://user-images.githubusercontent.com/17102274/42516300-50f04f8e-8490-11e8-95f1-613d3d3f6ca6.png)

![](https://user-images.githubusercontent.com/17102274/42516303-5594bd22-8490-11e8-8c01-55741484f126.png)

+ 问题分析：
使用`sudo docker run --name paddle-test -v $PWD:/paddle --network=host -it docker.paddlepaddlehub.com/paddle:latest-dev /bin/bash`命令创建的docker容器仅能支持运行CPU版本的PaddlePaddle。

+ 解决方法：
使用如下命令重新开启支持GPU运行的docker容器：

```
export CUDA_SO="$(\ls /usr/lib64/libcuda* | xargs -I{} echo '-v {}:{}') $(\ls /usr/lib64/libnvidia* | xargs -I{} echo '-v {}:{}')"

export DEVICES=$(\ls /dev/nvidia* | xargs -I{} echo '--device {}:{}')

sudo docker run ${CUDA_SO} ${DEVICES} --rm --name paddle-test-gpu -v /usr/bin/nvidia-smi:/usr/bin/nvidia-smi -v $PWD:/paddle --network=host -it docker.paddlepaddlehub.com/paddle:latest-dev /bin/bash
```

进入docker之后执行如下命令进行PaddlePaddle的安装及测试运行：

```
export LD_LIBRARY_PATH=/usr/lib64:/usr/local/lib:$LD_LIBRARY_PATH
pip install build/python/dist/*.whl
python python/paddle/fluid/tests/book/test_fit_a_line.py
```

## 3.问题：CMake源码编译，Paddle版本号为0.0.0

+ 关键字：`CMake` `版本号0.0.0`

+ 问题描述：在Liunx环境上，通过编译源码的方式安装PaddlePaddle，当安装成功后，运行 `paddle version`, 出现 `PaddlePaddle 0.0.0`

+ 问题解答：

如果运行 `paddle version`, 出现`PaddlePaddle 0.0.0`；或者运行 `cmake ..`，出现

```bash
CMake Warning at cmake/version.cmake:20 (message):
Cannot add paddle version from git tag
```

+ 解决方法：
在dev分支下这个情况是正常的，在release分支下通过export PADDLE_VERSION=对应版本号 来解决

## 4.问题：paddlepaddle\*.whl is not a supported wheel on this platform

+ 关键字：`wheel` `platform`

+ 问题描述：安装PaddlePaddle过程中，出现`paddlepaddle\*.whl is not a supported wheel on this platform`

+ 问题解答：
`paddlepaddle\*.whl is not a supported wheel on this platform`表示你当前使用的PaddlePaddle不支持你当前使用的系统平台，即没有找到和当前系统匹配的paddlepaddle安装包。最新的paddlepaddle python安装包支持Linux x86_64和MacOS 10.12操作系统，并安装了python 2.7和pip 9.0.1。

+ 解决方法：

请先尝试安装最新的pip，方法如下：

```bash
pip install --upgrade pip
```

如果还不行，可以执行 `python -c "import pip; print(pip.pep425tags.get_supported())"` 获取当前系统支持的python包的后缀，
并对比是否和正在安装的后缀一致。

如果系统支持的是 `linux_x86_64` 而安装包是 `manylinux1_x86_64` ，需要升级pip版本到最新；
如果系统支持 `manylinux1_x86_64` 而安装包（本地）是 `linux_x86_64` ，可以重命名这个whl包为 `manylinux1_x86_64` 再安装。












