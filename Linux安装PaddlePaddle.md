# Linux安装PaddlePaddle

## `已审阅` 1.问题：cuda9.0需要安装哪一个版本的paddle，安装包在哪

+ 版本号：`0.14.0`

+ 标签：`cuda 9.0`

+ 问题描述：cuda9.0需要安装哪一个版本的paddle，安装包在哪，希望安装Fluid版本的Paddle，而不是旧版的Paddle

+ 问题解答：
`paddlepaddle-gpu==1.1`使用CUDA 9.0和cuDNN 7编译的0.14.0版本

因此，`pip install paddlepaddle-gpu==1.1`即可。

可以参考安装说明文档：
http://paddlepaddle.org/documentation/docs/zh/0.14.0/new_docs/beginners_guide/install/install_doc.html#linuxpaddlepaddle

## `已审阅` 2.问题：pip install paddlepaddle-gpu==0.14.0.post87 安装 fluid 版本报错

+ 版本号：`0.14.0`

+ 标签：`GPU`

+ 问题描述：
使用  `pip install paddlepaddle-gpu==0.14.0.post87`命令在公司内部开发GPU机器上安装PaddlePaddle，安装信息如下：

![](https://user-images.githubusercontent.com/12878507/45028894-606ba980-b079-11e8-98e7-6e80f1c3f386.png)

机器的CUDA信息如下：
![](https://user-images.githubusercontent.com/12878507/45028950-8c872a80-b079-11e8-82f2-ca6591203eb1.png)

按照官网安装：pip install paddlepaddle-gpu==0.14.0.post87
执行 import paddle.fluid as fluid 失败
![](https://user-images.githubusercontent.com/12878507/45028976-a0329100-b079-11e8-84a7-07253eafb3cb.png)

奇怪的是，同样的环境下，上周运行成功，这周确运行失败，求解答

+ 解决方法：
这通常是GPU显存不足导致的，请检查一下机器的显存，确保显存足够后再尝试import paddle.fluid


## `已审阅` 3.问题：CUDA driver version is insufficient

+ 版本号：`1.1.0`

+ 标签：`CUDA` `insufficient`

+ 问题描述：在使用PaddlePaddle GPU的Docker镜像的时候，出现 `Cuda Error: CUDA driver version is insufficient for CUDA runtime version`

+ 问题解答：
通常出现 `Cuda Error: CUDA driver version is insufficient for CUDA runtime version`, 原因在于没有把机器上CUDA相关的驱动和库映射到容器内部。

+ 解决方法：
使用nvidia-docker, 命令只需要将docker换为nvidia-docker即可。
更多请参考：https://github.com/NVIDIA/nvidia-docker


## `已审阅` 4.问题：安装CPU版本后训练主动abort，gdb显示Illegal instruction

+ 版本号：`1.1.0`

+ 标签：`CPU版本` `Illegal instruction`

+ 问题描述：成功安装了PaddlePaddle CPU版本后，使用Paddle训练模型，训练过程中，Paddle会自动退出，gdb显示Illegal instruction

+ 报错输出：

```bash
*** Aborted at 1539697466 (unix time) try "date -d @1539697466" if you are using GNU date ***
PC: @                0x0 (unknown)
*** SIGILL (@0x7fe3a27b7912) received by PID 13005 (TID 0x7fe4059d8700) from PID 18446744072140585234; stack trace: ***
    @       0x318b20f500 (unknown)
    @     0x7fe3a27b7912 paddle::framework::VisitDataType<>()
    @     0x7fe3a279f84f paddle::operators::math::set_constant_with_place<>()
    @     0x7fe3a1e50c21 paddle::operators::FillConstantOp::RunImpl()
    @     0x7fe3a27526bf paddle::framework::OperatorBase::Run()
    @     0x7fe3a1ca31ea paddle::framework::Executor::RunPreparedContext()
    @     0x7fe3a1ca3be0 paddle::framework::Executor::Run()
    @     0x7fe3a1bc9e7d _ZZN8pybind1112cpp_function10initializeIZN6paddle6pybindL13pybind11_initEvEUlRNS2_9framework8ExecutorERKNS4_11ProgramDescEPNS4_5ScopeEibbE63_vIS6_S9_SB_ibbEINS_4nameENS_9is_methodENS_7siblingEEEEvOT_PFT0_DpT1_EDpRKT2_ENUlRNS_6detail13function_callEE1_4_FUNEST_
    @     0x7fe3a1c14c24 pybind11::cpp_function::dispatcher()
    @     0x7fe405acf3e4 PyEval_EvalFrameEx
    @     0x7fe405ad0130 PyEval_EvalCodeEx
    @     0x7fe405ace4a1 PyEval_EvalFrameEx
    @     0x7fe405ad0130 PyEval_EvalCodeEx
    @     0x7fe405ace4a1 PyEval_EvalFrameEx
    @     0x7fe405ad0130 PyEval_EvalCodeEx
    @     0x7fe405a5c181 function_call
    @     0x7fe405a340f3 PyObject_Call
    @     0x7fe405accde7 PyEval_EvalFrameEx
    @     0x7fe405acec56 PyEval_EvalFrameEx
    @     0x7fe405ad0130 PyEval_EvalCodeEx
    @     0x7fe405a5c27d function_call
    @     0x7fe405a340f3 PyObject_Call
    @     0x7fe405accde7 PyEval_EvalFrameEx
    @     0x7fe405ad0130 PyEval_EvalCodeEx
    @     0x7fe405a5c181 function_call
    @     0x7fe405a340f3 PyObject_Call
    @     0x7fe405a46f7f instancemethod_call
    @     0x7fe405a340f3 PyObject_Call
    @     0x7fe405a8abd4 slot_tp_call
    @     0x7fe405a340f3 PyObject_Call
    @     0x7fe405acd887 PyEval_EvalFrameEx
    @     0x7fe405acec56 PyEval_EvalFrameEx
```

+ 问题解答：
CPU版本PaddlePaddle自动退出的原因通常是因为所在机器不支持AVX2指令集而主动abort。简单的判断方法：
用gdb-7.9以上版本（因编译C++文件用的工具集是gcc-4.8.2，目前只知道gdb-7.9这个版本可以debug gcc4编译出来的目标文件）：

```bash
$ /path/to/gdb -iex "set auto-load safe-path /" -iex "set solib-search-path /path/to/gcc-4/lib" /path/to/python -c core.xxx
```

在gdb界面：

```bash
(gdb) disas
```

找到箭头所指的指令，例如：

```bash
   0x00007f381ae4b90d <+3101>:  test   %r8,%r8
=> 0x00007f381ae4b912 <+3106>:  vbroadcastss %xmm0,%ymm1
   0x00007f381ae4b917 <+3111>:  lea    (%r12,%rdx,4),%rdi
```

然后google一下这个指令需要的指令集。上面例子中的带xmm和ymm操作数的vbroadcastss指令只在AVX2中支持

然后看下自己的CPU是否支持该指令集

```bash
cat /proc/cpuinfo | grep flags | uniq | grep avx --color
```

如果没有AVX2，就表示确实是指令集不支持引起的主动abort

+ 解决方法：
如果没有AVX2指令集，就需要要安装不支持AVX2指令集版本的PaddlePaddle，默认安装的PaddlePaddle是支持AVX2指令集的，因为AVX2可以加速模型训练的过程，更多细节可以参考安装文档

http://www.paddlepaddle.org/documentation/docs/zh/1.0/beginners_guide/install/Start.html#paddlepaddle


## `已审阅` 5.问题：nvidia-docker运行镜像latest-gpu-cuda8.0-cudnn7: SIGILL

+ 版本号：`1.1.0`

+ 标签：`nvidia-docker` `cuda8.0` `cudnn7`

+ 问题描述：使用`sudo nvidia-docker run --name Paddle -it -v $PWD:/work hub.baidubce.com/paddlepaddle/paddle:latest-gpu-cuda8.0-cudnn7 /bin/bash`，安装成功后，出现如下问题

```bash
import paddle.fluid
*** Aborted at 1539682149 (unix time) try "date -d @1539682149" if you are using GNU date ***
PC: @ 0x0 (unknown)
*** SIGILL (@0x7f6ac6ea9436) received by PID 16 (TID 0x7f6b07bc7700) from PID 18446744072751846454; stack trace: ***
```

+ 解决方法：
请先确定一下机器是否支持AVX2指令集，如果不支持，请按照相应的不支持AVX2指令集的PaddlePaddle，可以解决该问题。


## `已审阅` 6.问题：安装paddlepaddle fluid版本后import paddle.fluid error

+ 版本号：`1.1.0`

+ 标签：`import error`

+ 问题描述：使用的系统是Ubuntu 16.04，GPU相关环境：cuda8.0, cudnn 6.0, 安装最新版的paddlepaddle fluid 后，import paddle时问题如下：

在命令行下

```
import paddle
import paddle.v2
```

都没问题，唯独

```
import paddle.fluid
```

报错，为何只有fluid版本import时会有问题？

+ 报错输出：

```bash
Traceback (most recent call last):
File "", line 1, in 
File "/usr/local/lib/python2.7/dist-packages/paddle/fluid/init.py", line 132, in 
bootstrap()
File "/usr/local/lib/python2.7/dist-packages/paddle/fluid/init.py", line 126, in bootstrap
core.init_devices(not in_test)
paddle.fluid.core.EnforceNotMet: CUBLAS: not initialized, at [/paddle/paddle/fluid/platform/device_context.cc:153]
PaddlePaddle Call Stacks:
0 0x7f0238da06f6p paddle::platform::EnforceNotMet::EnforceNotMet(std::exception_ptr::exception_ptr, char const*, int) + 486
1 0x7f0239b1ee54p paddle::platform::CUDADeviceContext::CUDADeviceContext(paddle::platform::CUDAPlace) + 1684
2 0x7f0239b1feb0p paddle::platform::DeviceContextPool::DeviceContextPool(std::vector<boost::variant<paddle::platform::CUDAPlace, paddle::platform::CPUPlace, paddle::platform::CUDAPinnedPlace, boost::detail::variant::void, boost::detail::variant::void, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_>, std::allocator<boost::variant<paddle::platform::CUDAPlace, paddle::platform::CPUPlace, paddle::platform::CUDAPinnedPlace, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_> > > const&) + 752
3 0x7f0238e368bcp paddle::framework::InitDevices(bool, std::vector<int, std::allocator >) + 588
4 0x7f0238e36addp paddle::framework::InitDevices(bool) + 285
5 0x7f0238d865bap
6 0x7f0238db1804p pybind11::cpp_function::dispatcher(_object*, _object*, _object*) + 2596
7 0x4bc3fap PyEval_EvalFrameEx + 1482
8 0x4b9ab6p PyEval_EvalCodeEx + 774
9 0x4c1e6fp PyEval_EvalFrameEx + 24639
10 0x4b9ab6p PyEval_EvalCodeEx + 774
11 0x4b97a6p PyEval_EvalCode + 22
12 0x4b96dfp PyImport_ExecCodeModuleEx + 191
13 0x4b2b06p
14 0x4b402cp
15 0x4a4ae1p
16 0x4a4513p PyImport_ImportModuleLevel + 2259
17 0x4a59e4p
18 0x4a577ep PyObject_Call + 62
19 0x4c5e10p PyEval_CallObjectWithKeywords + 48
20 0x4be6d7p PyEval_EvalFrameEx + 10407
21 0x4b9ab6p PyEval_EvalCodeEx + 774
22 0x4eb30fp
23 0x44a7a2p PyRun_InteractiveOneFlags + 400
24 0x44a56dp PyRun_InteractiveLoopFlags + 186
25 0x43092ep
26 0x493ae2p Py_Main + 1554
27 0x7f026bfa1830p __libc_start_main + 240
28 0x4933e9p _start + 41
```

+ 解决方法：
请先查看您系统GPU幻觉的适配关系，应该选择和您的系统已经安装的CUDA版本相同的whl包，您的系统是cuda 8.0, cudnn 6 应该使用cuda8.0_cudnn7_avx_mkl才可以适配。

然后尝试如下命令看看是否报错

```
>>> import paddle.fluid
```



如果报错，则可能是GPU 和CUDA环境没有正确配置

如果没有报错，请判断是否有给所有相关文件`sudo权限`



## `已审阅` 7.问题：在Fluid版本训练的时报以下错误，是不是显卡的问题？

+ 版本号：`0.14.0`

+ 标签：`GPU` `Fluid版本`

+ 问题描述：
我安装的是cuda9.0和cudnn7.0，我看文档以为默认安装的是0.14.0.post97的，谁知道是post87的，那么以下的错误是这个问题吗？我在使用V2版本训练一个手写数据那个例子的时候meiyou错，GPU能够正常使用，Fluid版本就不行了。

+ 报错输出：

```bash
Traceback (most recent call last):
  File "train.py", line 240, in <module>
    main()
  File "train.py", line 236, in main
    train(args)
  File "train.py", line 147, in train
    exe.run(fluid.default_startup_program())
  File "/usr/local/lib/python2.7/dist-packages/paddle/fluid/executor.py", line 443, in run
    self.executor.run(program.desc, scope, 0, True, True)
paddle.fluid.core.EnforceNotMet: enforce allocating <= available failed, 1827927622 > 1359806208
 at [/paddle/paddle/fluid/platform/gpu_info.cc:119]
PaddlePaddle Call Stacks: 
0       0x7f1bac5312f6p paddle::platform::EnforceNotMet::EnforceNotMet(std::__exception_ptr::exception_ptr, char const*, int) + 486
1       0x7f1bad3a95bep paddle::platform::GpuMaxChunkSize() + 766
2       0x7f1bad2d92ddp paddle::memory::GetGPUBuddyAllocator(int) + 141
3       0x7f1bad2d94ecp void* paddle::memory::Alloc<paddle::platform::CUDAPlace>(paddle::platform::CUDAPlace, unsigned long) + 28
4       0x7f1bad2ced42p paddle::framework::Tensor::mutable_data(boost::variant<paddle::platform::CUDAPlace, paddle::platform::CPUPlace, paddle::platform::CUDAPinnedPlace, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_>, std::type_index) + 866
5       0x7f1bac7a0cbfp paddle::operators::FillConstantOp::RunImpl(paddle::framework::Scope const&, boost::variant<paddle::platform::CUDAPlace, paddle::platform::CPUPlace, paddle::platform::CUDAPinnedPlace, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_> const&) const + 1007
6       0x7f1bad261ebdp paddle::framework::OperatorBase::Run(paddle::framework::Scope const&, boost::variant<paddle::platform::CUDAPlace, paddle::platform::CPUPlace, paddle::platform::CUDAPinnedPlace, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_> const&) + 205
7       0x7f1bac5cd06fp paddle::framework::Executor::RunPreparedContext(paddle::framework::ExecutorPrepareContext*, paddle::framework::Scope*, bool, bool, bool) + 255
8       0x7f1bac5ce0c0p paddle::framework::Executor::Run(paddle::framework::ProgramDesc const&, paddle::framework::Scope*, int, bool, bool) + 128
9       0x7f1bac548cbbp void pybind11::cpp_function::initialize<pybind11::cpp_function::initialize<void, paddle::framework::Executor, paddle::framework::ProgramDesc const&, paddle::framework::Scope*, int, bool, bool, pybind11::name, pybind11::is_method, pybind11::sibling>(void (paddle::framework::Executor::*)(paddle::framework::ProgramDesc const&, paddle::framework::Scope*, int, bool, bool), pybind11::name const&, pybind11::is_method const&, pybind11::sibling const&)::{lambda(paddle::framework::Executor*, paddle::framework::ProgramDesc const&, paddle::framework::Scope*, int, bool, bool)#1}, void, paddle::framework::Executor*, paddle::framework::ProgramDesc const&, paddle::framework::Scope*, int, bool, bool, pybind11::name, pybind11::is_method, pybind11::sibling>(pybind11::cpp_function::initialize<void, paddle::framework::Executor, paddle::framework::ProgramDesc const&, paddle::framework::Scope*, int, bool, bool, pybind11::name, pybind11::is_method, pybind11::sibling>(void (paddle::framework::Executor::*)(paddle::framework::ProgramDesc const&, paddle::framework::Scope*, int, bool, bool), pybind11::name const&, pybind11::is_method const&, pybind11::sibling const&)::{lambda(paddle::framework::Executor*, paddle::framework::ProgramDesc const&, paddle::framework::Scope*, int, bool, bool)#1}&&, void (*)(paddle::framework::Executor*, paddle::framework::ProgramDesc const&, paddle::framework::Scope*, int, bool, bool), pybind11::name const&, pybind11::is_method const&, pybind11::sibling const&)::{lambda(pybind11::detail::function_call&)#3}::_FUN(pybind11::detail::function_call) + 555
10      0x7f1bac5411c4p pybind11::cpp_function::dispatcher(_object*, _object*, _object*) + 2596
11            0x4c37edp PyEval_EvalFrameEx + 31165
12            0x4b9ab6p PyEval_EvalCodeEx + 774
13            0x4c16e7p PyEval_EvalFrameEx + 22711
14            0x4b9ab6p PyEval_EvalCodeEx + 774
15            0x4c1e6fp PyEval_EvalFrameEx + 24639
16            0x4c136fp PyEval_EvalFrameEx + 21823
17            0x4b9ab6p PyEval_EvalCodeEx + 774
18            0x4eb30fp
19            0x4e5422p PyRun_FileExFlags + 130
20            0x4e3cd6p PyRun_SimpleFileExFlags + 390
21            0x493ae2p Py_Main + 1554
22      0x7f1bd6ae9830p __libc_start_main + 240
23            0x4933e9p _start + 41
```

+ 解决方法：
该问题通常是GPU显存不足造成的，请在显存充足的GPU服务器上再次尝试则可。可以检查一下机器的显存使用情况。

方法如下：

```bash
test@test:~$ nvidia-smi
Tue Jul 24 08:24:22 2018       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 384.130                Driver Version: 384.130                   |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX 960     Off  | 00000000:01:00.0  On |                  N/A |
| 22%   52C    P2   100W / 120W |   1757MiB /  1994MiB |     98%      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|    0      1071      G   /usr/lib/xorg/Xorg                           314MiB |
|    0      1622      G   compiz                                       149MiB |
|    0      2201      G   fcitx-qimpanel                                 7MiB |
|    0     15304      G   ...-token=58D78B2D4A63DAE7ED838021B2136723    74MiB |
|    0     15598      C   python                                      1197MiB |
+-----------------------------------------------------------------------------+
```


## `已审阅` 8.问题：使用新版的pip安装了GPU版的PaddlePaddle0.14.0，跑一个简单的测试程序，出现Segmentation fault

+ 版本号：`0.14.0`

+ 标签：`GPU` `Segmentation fault`

+ 问题描述：版本为paddlepaddle_gpu-0.14.0.post87-cp27-cp27mu-manylinux1_x86_64.whl
测试程序如下
其中 如果place为cpu，可以正常输出，改成gpu则core。老版本0.13 可以正常跑

```bash
def testpaddle014():
    place = fluid.CUDAPlace(0)
    #place = fluid.CPUPlace()
    print 'version', paddle.__version__, place
    input = fluid.layers.data(name='input', shape=[3,50,50], dtype='float32')
    
    output = fluid.layers.conv2d(input=input,num_filters=1,filter_size=3,stride=1,padding=1,groups=1,act=None)
    #output = fluid.layers.fc(input=input,size=2)
    
    fetch_list = [output.name]
    data = np.zeros((2,3,50,50), np.float32)
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    outputlist = exe.run(
                fluid.default_main_program(),
                feed={'input': data},
                fetch_list=fetch_list
            )
    print 'output', outputlist[0].shape
```

+ 解决方法：
安装版本为`paddlepaddle_gpu-0.14.0.post87-cp27-cp27mu-manylinux1_x86_64.whl`，其中post87是指在CUDA8.0、cudnn7.0编译的，请确定您机器上是否安装了对应版本的cudnn。造成问题描述中现象的情况通常可能是幻觉不匹配导致的

## `已审阅` 9.问题：AI studio中， fluid GPU训练报错，CPU下可以运行, 报错显示resize_bilinear处错误 
 
+ 版本号：`0.14.0`

+ 标签：`AI studio` `GPU` `resize_bilinear`

+ 问题描述：最近用baidu ai studio 平台训练模型，在cpu模式下可以正常运行，但是换到gpu下就直接报如下错误：

```bash
*** Aborted at 1535591447 (unix time) try "date -d @1535591447" if you are using GNU date ***
PC: @                0x0 (unknown)
*** SIGSEGV (@0x10213006e40) received by PID 3682 (TID 0x7f26368fb700) from PID 318795328; stack trace: ***
    @     0x7f26360b0390 (unknown)
    @     0x7f2635712c76 (unknown)
    @     0x7f25f555253d paddle::operators::BilinearInterpOpCUDAKernel<>::Compute()
    @     0x7f25f5a3d44e paddle::framework::OperatorWithKernel::RunImpl()
    @     0x7f25f5a3a6ed paddle::framework::OperatorBase::Run()
    @     0x7f25f4ca14af paddle::framework::Executor::RunPreparedContext()
    @     0x7f25f4ca2500 paddle::framework::Executor::Run()
    @     0x7f25f4c1d0fb _ZZN8pybind1112cpp_function10initializeIZNS0_C1IvN6paddle9framework8ExecutorEJRKNS4_11ProgramDescEPNS4_5ScopeEibbEJNS_4nameENS_9is_methodENS_7siblingEEEEMT0_FT_DpT1_EDpRKT2_EUlPS5_S8_SA_ibbE_vJSO_S8_SA_ibbEJSB_SC_SD_EEEvOSF_PFSE_SH_ESN_ENUlRNS_6detail13function_callEE1_4_FUNESV_
    @     0x7f25f4c15604 pybind11::cpp_function::dispatcher()
    @     0x7f26363c99d2 PyEval_EvalFrameEx
    @     0x7f26363cbced PyEval_EvalCodeEx
    @     0x7f26363c8e72 PyEval_EvalFrameEx
    @     0x7f26363cbced PyEval_EvalCodeEx
    @     0x7f26363c8e72 PyEval_EvalFrameEx
    @     0x7f26363cbced PyEval_EvalCodeEx
    @     0x7f26363cbe22 PyEval_EvalCode
    @     0x7f26363f7d72 PyRun_FileExFlags
    @     0x7f26363f90e9 PyRun_SimpleFileExFlags
    @     0x7f263640f00d Py_Main
    @     0x7f26355e5830 __libc_start_main
    @           0x4007b1 (unknown)
Segmentation fault (core dumped)
```

我的环境是baidu AI studio提供的kernel, paddlepaddle版本是0.12.0, paddlepaddle-gpu 是0.14.0, cuda是8.0, cudnn是5.0，提供的gpu是TITAN X (Pascal)，驱动版本是375.26.
请问这是什么原因导致的？看我的错误中好像和fluid.layers.resize_bilinear有关系？但是为何cpu下没有错误呢？

+ 补充描述：
我将paddlepaddle-gpu版本改成了0.14.0.post87(虽然cudnn版本是5.0)，得到如下输出：

```bash
*** Aborted at 1535718570 (unix time) try "date -d @1535718570" if you are using GNU date ***
PC: @                0x0 (unknown)
*** SIGSEGV (@0x0) received by PID 1195 (TID 0x7f165a24e700) from PID 0; stack trace: ***
    @     0x7f1659a03390 (unknown)
    @                0x0 (unknown)
Segmentation fault (core dumped)
```

整体的环境是：baidu AI studio提供的kernel, paddlepaddle版本是0.12.0, paddlepaddle-gpu 是0.14.0, cuda是8.0, cudnn是5.0，提供的gpu是TITAN X (Pascal)，驱动版本是375.26.

+ 解决方法：
问题的原因出现在resize_bilinear上，resize_bilinear的scale参数不能设置为1(这个在cpu下是可以的)，虽然设置为１的时候没什么意义，但是我的代码中有个地方就设置为了１，所以才会报错


## `已审阅` 10.问题：安装完了PaddlePaddle后，发现python相关的单元测试都过不了

+ 版本号：`1.1.0`

+ 标签：`单元测试`

+ 问题描述：

如果出现以下python相关的单元测试都过不了的情况：

```
    24 - test_PyDataProvider (Failed)
    26 - test_RecurrentGradientMachine (Failed)
    27 - test_NetworkCompare (Failed)
    28 - test_PyDataProvider2 (Failed)
    32 - test_Prediction (Failed)
    33 - test_Compare (Failed)
    34 - test_Trainer (Failed)
    35 - test_TrainerOnePass (Failed)
    36 - test_CompareTwoNets (Failed)
    37 - test_CompareTwoOpts (Failed)
    38 - test_CompareSparse (Failed)
    39 - test_recurrent_machine_generation (Failed)
    40 - test_PyDataProviderWrapper (Failed)
    41 - test_config_parser (Failed)
    42 - test_swig_api (Failed)
    43 - layers_test (Failed)
```

并且查询PaddlePaddle单元测试的日志，提示：

```
    paddle package is already in your PYTHONPATH. But unittest need a clean environment.
    Please uninstall paddle package before start unittest. Try to 'pip uninstall paddle'.
```

+ 解决方法：
卸载PaddlePaddle包 `pip uninstall paddle`, 清理掉老旧的PaddlePaddle安装包，使得单元测试有一个干净的环境。如果PaddlePaddle包已经在python的site-packages里面，单元测试会引用site-packages里面的python包，而不是源码目录里 `/python` 目录下的python包。同时，即便设置 `PYTHONPATH` 到 `/python` 也没用，因为python的搜索路径是优先已经安装的python包。



## `已审阅` 11.问题：生成Docker镜像时，无法下载需要的golang，导致`tar: Error is not recoverable: exiting now`

+ 版本号：`0.14.0`

+ 标签：`golang` `Docker镜像`

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

## `已审阅`  12.问题：GPU版本的PaddlePaddle运行结果报错

+ 版本号：`1.1.0`

+ 标签：`GPU` `运行报错`

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

## `已审阅` 13.问题：CMake源码编译，Paddle版本号为0.0.0

+ 版本号：`1.1.0`

+ 标签：`CMake` `版本号0.0.0`

+ 问题描述：在Liunx环境上，通过编译源码的方式安装PaddlePaddle，当安装成功后，运行 `paddle version`, 出现 `PaddlePaddle 0.0.0`

+ 问题解答：

如果运行 `paddle version`, 出现`PaddlePaddle 0.0.0`；或者运行 `cmake ..`，出现

```bash
CMake Warning at cmake/version.cmake:20 (message):
Cannot add paddle version from git tag
```

+ 解决方法：
在dev分支下这个情况是正常的，在release分支下通过export PADDLE_VERSION=对应版本号 来解决

## `已审阅` 14.问题：paddlepaddle\*.whl is not a supported wheel on this platform

+ 版本号：`1.1.0`

+ 标签：`wheel` `platform`

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







