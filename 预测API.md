# 预测API

## `待审核`1.问题：c++ inference CreatePaddlePredictor segmentation fault

+ 版本号：`1.1.0`

+ 标签：``

+ 问题描述：


版本、环境信息：
   1）PaddlePaddle版本：1.3，cpu_avx_openblas， http://www.paddlepaddle.org/documentation/docs/zh/1.2/advanced_usage/deploy/inference/build_and_install_lib_cn.html
   2）CPU：Intel® Core™ i7-7700 CPU @ 3.60GHz × 8
   3）GPU：未开启
   4）系统环境：Ubuntu 16.04
5) gcc版本: gcc (Ubuntu 5.4.0-6ubuntu1~16.04.10) 5.4.0 20160609
-预测信息
   1）C++预测：1.3，cpu_avx_openblas

	```
	version.txt：
	GIT COMMIT ID: 4b3f9e5
	WITH_MKL: OFF
	WITH_MKLDNN: OFF
	WITH_GPU: OFF
	```

   2）CMake包含路径的完整命令

	```
	cmake_minimum_required(VERSION 3.0)
	project(test CXX C)

	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
	set(CMAKE_STATIC_LIBRARY_PREFIX "")

	#PADDLE_LIB = "/home/apollo/baidu/vp/PaddlePaddle/fluid_inference"

	include_directories("${PADDLE_LIB}/third_party/install/protobuf/include")
	include_directories("${PADDLE_LIB}/third_party/install/glog/include")
	include_directories("${PADDLE_LIB}/third_party/install/gflags/include")
	include_directories("${PADDLE_LIB}/third_party/install/xxhash/include")
	include_directories("${PADDLE_LIB}/third_party/install/snappy/include")
	include_directories("${PADDLE_LIB}/third_party/install/snappystream/include")
	include_directories("${PADDLE_LIB}/third_party/install/zlib/include")
	include_directories("${PADDLE_LIB}/third_party/boost")
	include_directories("${PADDLE_LIB}/third_party/eigen3")
	include_directories("${PADDLE_LIB}/paddle/include")

	link_directories("${PADDLE_LIB}/third_party/install/snappy/lib")
	link_directories("${PADDLE_LIB}/third_party/install/snappystream/lib")
	link_directories("${PADDLE_LIB}/third_party/install/zlib/lib")
	link_directories("${PADDLE_LIB}/third_party/install/protobuf/lib")
	link_directories("${PADDLE_LIB}/third_party/install/glog/lib")
	link_directories("${PADDLE_LIB}/third_party/install/gflags/lib")
	link_directories("${PADDLE_LIB}/third_party/install/xxhash/lib")
	link_directories("${PADDLE_LIB}/paddle/lib")

	set(MATH_LIB ${PADDLE_LIB}/third_party/install/openblas/lib/libopenblas${CMAKE_STATIC_LIBRARY_SUFFIX})
	set(EXTERNAL_LIB "-lrt -ldl -lpthread")
	set(DEPS ${PADDLE_LIB}/paddle/lib/libpaddle_fluid${CMAKE_SHARED_LIBRARY_SUFFIX})

	add_executable(test_fluid test_fluid.cpp)
	target_link_libraries(test_fluid glog gflags protobuf snappystream snappy z xxhash ${MATH_LIB} ${DEPS} ${EXTERNAL_LIB} )
	   3）API信息（如调用请提供）

	#include "paddle_inference_api.h"
	#include <string>
	#include <iostream>

	int main()
	{
	    paddle::NativeConfig config;
	    //config.model_dir = "/home/apollo/baidu/vp/output_x86/rfcn_0315";
	    //config.prog_file = "model";
	    //config.param_file = "params";
	    config.prog_file = "/home/apollo/baidu/vp/PaddlePaddle/output_x86/rfcn_0315/model";
	    config.param_file = "/home/apollo/baidu/vp/PaddlePaddle/output_x86/rfcn_0315/params";
	    config.use_gpu = false;
	    // config.fraction_of_gpu_memory = 0.15;
	    // config.device = 0;

	    auto _predictor = paddle::CreatePaddlePredictor<paddle::NativeConfig>(config);
	    std::cout<<"create predictor after."<<std::endl;
	}
	```

   4）预测库来源：官网下载

复现信息：如为报错，请给出复现环境、复现步骤
问题描述：请详细描述您的问题，同步贴出报错信息、日志/代码关键片段
CreatePaddlePredictor内部报错，gdb信息如下

	```
	(gdb) r
	Starting program: /home/apollo/baidu/vp/PaddlePaddle/fluid_inference/build/test_fluid 
	[Thread debugging using libthread_db enabled]
	Using host libthread_db library "/lib/x86_64-linux-gnu/libthread_db.so.1".
	[New Thread 0x7ffff3c34700 (LWP 7858)]
	[New Thread 0x7ffff3433700 (LWP 7859)]
	[New Thread 0x7ffff0c32700 (LWP 7860)]

	Thread 1 "test_fluid" received signal SIGSEGV, Segmentation fault.
	0x00007ffff480b37a in std::basic_string<char, std::char_traits<char>, std::allocator<char> >::basic_string(std::string const&) () from /usr/lib/x86_64-linux-gnu/libstdc++.so.6
	(gdb) bt
	#0  0x00007ffff480b37a in std::basic_string<char, std::char_traits<char>, std::allocator<char> >::basic_string(std::string const&) () from /usr/lib/x86_64-linux-gnu/libstdc++.so.6
	#1  0x00007ffff53a42fe in paddle::NativePaddlePredictor::NativePaddlePredictor(paddle::NativeConfig const&) () from /home/apollo/baidu/vp/PaddlePaddle/fluid_inference/paddle/lib/libpaddle_fluid.so
	#2  0x00007ffff539ee42 in std::unique_ptr<paddle::PaddlePredictor, std::default_delete<paddle::PaddlePredictor> > paddle::CreatePaddlePredictor<paddle::NativeConfig, (paddle::PaddleEngineKind)0>(paddle::NativeConfig const&) () from /home/apollo/baidu/vp/PaddlePaddle/fluid_inference/paddle/lib/libpaddle_fluid.so
	#3  0x00007ffff539f7c1 in std::unique_ptr<paddle::PaddlePredictor, std::default_delete<paddle::PaddlePredictor> > paddle::CreatePaddlePredictor<paddle::NativeConfig>(paddle::NativeConfig const&) ()
	   from /home/apollo/baidu/vp/PaddlePaddle/fluid_inference/paddle/lib/libpaddle_fluid.so
	#4  0x0000000000401c3d in main ()
	```

备注：目前该问题已知是由gcc版本引起，本机gcc版本5.4,自己源码编译后的库可以正常加载模型。


+ 解决方法：

由于官网下载的预测库都是gcc48编译的，而用户在gcc54环境（paddle的latest-dev镜像）下运行，因此直接core。
建议：官网文档标清楚，开发镜像和发布版本的gcc差别要尽早解决。


