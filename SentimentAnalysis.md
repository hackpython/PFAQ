# 情感分析

## 背景介绍
在自然语言处理中，情感分析一般是指判断一段文本所表达的情绪状态。其中，一段文本可以是一个句子，一个段落或一个文档。情绪状态可以是两类，如（正面，负面），（高兴，悲伤）；也可以是三类，如（积极，消极，中性）等等。

在自然语言处理中，情感分析属于典型的文本分类问题，即把需要进行情感分析的文本划分为其所属类别。文本分类涉及文本表示和分类方法两个问题。

## 1.问题：情感分析的dome，怎么用自定义的训练集和测试集？

+ 关键字：`自定义训练集` `自定义测试集`

+ 问题描述：
情感分析demo的例子是用的 imdb 的数据集，我想用自己的数据集。请问怎么把训练集和测试集运用到demo里面？

+ 解决方法：
首先要熟悉你想使用的数据集的结构，熟悉了结构后，才能编写对应的处理代码，整体的逻辑其实简单，通常将数据2-8分成测试集与训练集，然后定义一个方法，在每一轮都返回一batch的数据，让trainer去处理则可。

    因为实例中的情感分析Demo其实是文本分类问题，所以在使用自定义数据时，处理好输入的文本以及该文本对应的标签的关系则可，具体细节可以参考PaddlePaddle Model中文本分类的代码，https://github.com/PaddlePaddle/models/blob/59adc0d6f38cd2351e16608d6c9d4e72dd5e7fea/fluid/text_classification/train.py

+ 问题拓展：
因为PaddlePaddle中的各种示例代码以演示PaddlePaddle用法为主，所以通常使用的都是简单的训练数据，而且封装好了处理这些数据的接口，让使用者不必关系数据是如何被有效组织的，只需关注此时构建神经网络的结构。但这也就导致了很多人想替换自己的数据集遇到点困难。

    其实只要你熟悉自己要使用数据集的结构，要让PaddlePaddle使用这些数据来训练模型还是很简单的，你完全可以模型PaddlePaddle示例代码中处理数据方法的内部实现，其实实现逻辑都不复杂，以情感分析使用的imdb数据集处理逻辑为例，其处理的方法为在`movielens.py`文件中，其实都是对python内置结构的使用。

+ 问题研究：
最简单的情感分析其实就是一个文本二分类问题，将一句话划分为正面情绪还是负面情绪，而复杂点的其实就是文本的定义分类问题，研究情感分析时，处理要理解其自然语言处理方面的内容，还有就是文本多分类的内容。


## 2.问题：在预处理我的中文数据集以后，运行train.sh脚本进行训练，出现错误。请问如何解决？

+ 关键字：`自定义中文数据集`

+ 问题描述：我模仿PaddlePaddle处理数据的方式预处了自己的中文数据集，然后使用官方提供的train.sh脚本进行训练，出现下面错误。

+ 报错输出：

```bash
root@caffe:~/paddlepaddle/paddle/demo/sentiment# sh train.sh
I0428 08:17:43.295579   763 Util.cpp:155] commandline: /usr/bin/../opt/paddle/bin/paddle_trainer --config=trainer_config.py --save_dir=./model_output --job=train --use_gpu=false --trainer_count=4 --num_passes=10 --log_period=10 --dot_period=20 --show_parameter_stats_period=100 --test_all_data_in_one_period=1 
I0428 08:17:43.295797   763 Util.cpp:130] Calling runInitFunctions
I0428 08:17:43.296051   763 Util.cpp:143] Call runInitFunctions done.
[INFO 2017-04-28 08:17:43,550 networks.py:1466] The input order is [word, label]
[INFO 2017-04-28 08:17:43,550 networks.py:1472] The output order is [__cost_0__]
I0428 08:17:43.552901   763 Trainer.cpp:170] trainer mode: Normal
*** Aborted at 1493367463 (unix time) try "date -d @1493367463" if you are using GNU date ***
PC: @           0x76e252 paddle::Weight::Weight()
*** SIGSEGV (@0xa8) received by PID 763 (TID 0x7f1c4d6be780) from PID 168; stack trace: ***
    @     0x7f1c4cfb3330 (unknown)
    @           0x76e252 paddle::Weight::Weight()
    @           0x5a3ec9 paddle::TableProjection::TableProjection()
    @           0x5a44b3 _ZNSt17_Function_handlerIFPN6paddle10ProjectionENS0_16ProjectionConfigESt10shared_ptrINS0_9ParameterEEbEZNS0_14ClassRegistrarIS1_JS3_S6_bEE13registerClassINS0_15TableProjectionEEEvRKSsEUlS3_S6_bE_E9_M_invokeERKSt9_Any_dataS3_S6_b
    @           0x5865e7 paddle::Projection::create()
    @           0x5d841a paddle::MixedLayer::init()
    @           0x62ecaf paddle::NeuralNetwork::init()
    @           0x64d982 paddle::MultiGradientMachine::MultiGradientMachine()
    @           0x6513ee paddle::GradientMachine::create()
    @           0x67c3e8 paddle::TrainerInternal::init()
    @           0x678b3e paddle::Trainer::init()
    @           0x5132a9 main
    @     0x7f1c4b3a4f45 (unknown)
    @           0x51f2a5 (unknown)
    @                0x0 (unknown)
/usr/bin/paddle: line 109:   763 Segmentation fault      (core dumped) ${DEBUGGER} $MYDIR/../opt/paddle/bin/paddle_trainer ${@:2}
```

+ 解决方法：
从报错信息看，可以发现，网络结构大多都创建与初始成功了，此时你需要确定一下你预处理后数据是否有问题，特别是返回的词ID是不是不等于训练字典的大小，通过print方法打印一下自己预处理后的训练集结构，确定自己的数据集结构没有问题，训练应该就是正常了。

+ 问题拓展：
训练英文数据与训练中文数据没有本质的区别，对计算机而言，它只理解0-1，无法理解我们使用的任何语言，训练文本数据的底层思想其实就是统计学，统计一个词出现的位置以及出现的频率，可以从一定程度上代表这个词，对于英文或中文都符合这种统计，其实处理语言外，还有很多具有类似规律的数据集，如用户浏览商品的轨迹数据集，这种数据集都可以使用自然语言处理的方式训练。


## 3.问题：在Ubuntu 14.04 LTS下执行./get_imdb.sh获取数据脚本问题

+ 关键字：`get_imdb.sh` `获取数据`

+ 问题描述：
在进行数据准备时，如果系统为新系统，且没有安装zip解压工具时，执行./get_imdb.sh 下载的原始数据中有.zip的压缩包，脚本执行最后会报错，避免用户执行后才发现问题。

+ 问题讨论：
在linux类的系统中，unzip是最基本的命令之一，所以PaddlePaddle没有在每个文档或代码中进行unzip安装的提示与判断，因为这会导致文档与代码变得冗余，而且缺少unzip时，系统会给出`unzip: command not found`提示，该提示已经较清楚的提示了当前环境缺少需要的unzip工具，自己安装则可。

+ 解决方案：
对于缺少某些比较基础的工具，需要使用者自行安装，如果要在代码中实现这些工具的自动安装会导致代码变得非常冗余。在不同的linux发行版本中，安装方式并不一定相同，使用相应的代码处理，就导致增加了很多代码都在处理这些内容，让PaddlePaddle变得冗余。


## 4.单机执行情感分析的demo出错，之前用MPI运行没有问题

+ 关键字：`单机执行` `pserver` `分布式`

+ 问题描述：使用PaddlePaddle编写了情感分析的demo，该demo使用MPI运行没有问题，但使用简单的单机运行却报错

+ 报错输出：

```bash
$ python query_dnn_single.py
I1222 11:32:54.626684 23753 Util.cpp:166] commandline: --use_gpu=False
load dictionary...
I1222 11:32:54.637953 23753 GradientMachine.cpp:94] Initing parameters..
I1222 11:32:54.647127 23753 GradientMachine.cpp:101] Init parameters done.
I1222 11:32:54.647704 23753 ParameterClient2.cpp:113] pserver 0 127.0.0.1:20134
W1222 11:32:54.648000 23753 LightNetwork.cpp:397] connection refused by pserver, try again!
W1222 11:32:55.648346 23753 LightNetwork.cpp:397] connection refused by pserver, try again!
W1222 11:32:56.648589 23753 LightNetwork.cpp:397] connection refused by pserver, try again!
W1222 11:32:57.648828 23753 LightNetwork.cpp:397] connection refused by pserver, try again!
W1222 11:32:58.649035 23753 LightNetwork.cpp:397] connection refused by pserver, try again!
W1222 11:32:59.649250 23753 LightNetwork.cpp:397] connection refused by pserver, try again!
W1222 11:33:00.649497 23753 LightNetwork.cpp:397] connection refused by pserver, try again!
W1222 11:33:01.649745 23753 LightNetwork.cpp:397] connection refused by pserver, try again!
F1222 11:33:01.649802 23753 LightNetwork.cpp:399] connection refused by pserver, maybe pserver failed!
*** Check failure stack trace: ***
@ 0x7ff3756c1d2d google::LogMessage::Fail()
@ 0x7ff3756c57dc google::LogMessage::SendToLog()
@ 0x7ff3756c1853 google::LogMessage::Flush()
@ 0x7ff3756c6cee google::LogMessageFatal::~LogMessageFatal()
@ 0x7ff37552d37f paddle::SocketClient::TcpClient()
@ 0x7ff37552d521 paddle::SocketClient::SocketClient()
@ 0x7ff3772492e7 paddle::ParameterClient2::init()
@ 0x7ff376da8c4d paddle::RemoteParameterUpdater::init()
@ 0x7ff3756a1cba ParameterUpdater::init()
@ 0x7ff37531ca3b _wrap_ParameterUpdater_init
@ 0x7ff39b73cbad PyEval_EvalFrameEx
@ 0x7ff39b73dc3e PyEval_EvalCodeEx
@ 0x7ff39b73d1f7 PyEval_EvalFrameEx
@ 0x7ff39b73dc3e PyEval_EvalCodeEx
@ 0x7ff39b73d1f7 PyEval_EvalFrameEx
@ 0x7ff39b73dc3e PyEval_EvalCodeEx
@ 0x7ff39b73d1f7 PyEval_EvalFrameEx
@ 0x7ff39b73dc3e PyEval_EvalCodeEx
@ 0x7ff39b73dd52 PyEval_EvalCode
@ 0x7ff39b75e450 PyRun_FileExFlags
@ 0x7ff39b75e62f PyRun_SimpleFileExFlags
@ 0x7ff39b773fd4 Py_Main
@ 0x318ae1ecdd (unknown)
@ 0x400729 (unknown)
Aborted
```

+ 问题分析：
从报错输出的信息可以看出，程序在重复尝试连接pserver，最终因为连接不上，给出`connection refused by pserver, maybe pserver failed!`，pserver是Parameter server参数服务器的一种简称，参数服务器是PaddlePaddle在集群环境下分布式是训练所需要的，而在单机上，pserver却是不被需要的，如果单机中使用了集群环境的配置，却没有启动pserver，那么就会出现`connection refused by pserver, maybe pserver failed!`，导致程序运行失败。

+ 解决方法：
将代码中集群相关的代码删除、修改，更多PaddlePaddle分布式训练的内容，参考相关的官方文档：

http://www.paddlepaddle.org/documentation/docs/zh/1.0.0/user_guides/howto/training/cluster_quick_start.html




## 5.问题：关于CNN，LSTM文本训练结果不收敛的问题请教

+ 关键字：`CNN` `LSTM` `不收敛`

+ 问题描述：
我用相同的训练集和测试集，分别通过cnn和lstm进行了训练，网络结构参考book教程中的情感分析模型。
dataprovider中我把所有数据加载到内存中：
@Provider(init_hook=hook, cache=CacheType.CACHE_PASS_IN_MEM)

    以cnn模型为例，我的参数为：batch_size=512, trainer_count=7
    其他参数是：

    ```python
    settings(
        batch_size=512,
        learning_rate=2e-3,
        learning_method=AdamOptimizer(),
        average_window=0.5,
        regularization=L2Regularization(8e-4),
        gradient_clipping_threshold=25)
    ```

    但是看训练结果，第一轮训练的classification_error_evaluator为0.114，而且随着训练的进行，有增高的趋势：（grep 训练日志关键字之后的结果）

    ![](https://user-images.githubusercontent.com/22980558/28861337-fd8672c0-7792-11e7-8350-cceb53a5a405.png)

    请问出现这种情况正常么？
    我的原始训练样本顺序，所有的正例‘1’都排在负例‘0’前面，这个会有影响么？

+ 问题解答：
原始样本顺序会有比较大的影响，在学习神经网络的过程中，你可以发现，dropout、随机梯度下载、参数初始随机化等等技术点都是让模型在训练的过程中变得更加随机，随机性是让深度神经网络具有优秀泛化能力的一个关键，随机性可以让神经网络训练后，可以让模型在单输入的情况产生多种输出，随机性也可以限制神经网络中信息的流向，迫使网络学习到更有意义的信息等。

+ 解决方法：
尝试让模型的训练数据随机化，通常可以使用python的shuffle方法让数据随机。

+ 问题研究：
输入数据的随机性会让系统产生额外的噪音，这些随机噪音对神经网络获得泛化能力是有益处的。

    + 随机噪音使得神经网络在单输入的情况下产生多输出的结果；
    + 随机噪音限制了信息在网络中的流向，促使网络学习数据中有意义的主要信息；
    + 随机噪声在梯度下降的时候提供了“探索能量”，使得网络可以寻找到更好的优化结果。


## 6.问题：新版的paddle网络配置文件怎么生成二进制网络配置供老版本sdk调用？

+ 关键字：`旧版SDK调用` `二进制网络配置`

+ 问题描述：使用PaddlePaddle开发项目时，遇到一个问题，希望可以通过新版的PaddlePaddle生成旧版SDK可以调用的二进制配置文件，请问这怎么实现？

+ 问题解答：
首先建议你不要使用旧版的SDK，因为旧版的SDK是不开源以及要被弃用的。你可以尝试下面的方式为就的预测SDK生成二进制配置文件，你可以使用`generate_config.py`来实现这个目的，使用方式如下：

    ```python
    import paddle.trainer.config_parser
    import struct
    bin_str = paddle.trainer.config_parser.parse_config_and_serialize("test_fc.py", "cost=0")
    print struct.pack("i", len(bin_str))
    print bin_str
    ```

+ 问题讨论：
按照问题解答中提供的使用代码，依旧遇到问题，报错输出如下：

    ```
    vocab_path = ../data/vocab.txt ModelPredict read_vocab begin
    ModelPredict read_vocab end!
    [INFO][PredictorInternal.cpp][init][370] model: [../data/model]
    [INFO][PredictorInternal.cpp][init][371] conf:  [../data/binary_conf]
    [INFO][PredictorInternal.cpp][readBinaryConf][40] content length of binary conf is [1346].
    [FATAL][PredictorInternal.cpp][readBinaryConf][62] [1350] vs. [1352] binary_conf header size != actual file size.
    [FATAL][PredictorInternal.cpp][readBinaryConf][63] Please check your binary_conf file.
    [FATAL][PredictorInternal.cpp][init][388] failed to read binary conf file into string.
    [FATAL][Predictor.cpp][init][75] failed to call imp_->init.
    paddle Predictor init failed,   model_path= ../data
    !!!!!!!!!!!!!!init error !!!!!!!!!!!!!!!!
    Out[3]: False
    ```

    在生成二进制文件时，自己校验了文件长度，如下：

    ```bash
    [WARNING 2016-11-02 16:35:27,196 default_decorators.py:40] please use keyword arguments in paddle config.
    [INFO 2016-11-02 16:35:27,199 networks.py:1122] The input order is [word]
    [INFO 2016-11-02 16:35:27,199 networks.py:1129] The output order is [__fc_layer_0__]
    binary heard length：4
    binary network length：1346
    ```

    从报错输出的信息来看，应该是python的print方法写入了额外的`\n`到stdout中，可以尝试使用`sys.stdout.write`来代替print方法




## 7.问题：lstmemory的hidden size怎么指定，官网教程为什么除以4

+ 关键字：`lstmemory`

+ 问题描述：关于官网情感分析文档中对应RNN与LSTM的讲解有几个疑惑，如下：

    1.hidden_dim = 512，LSTM hidden vector = 128，为什么要除以4？如果要指定lstm hidden vector长度为80，那是不是要定义hidden_dim = 320？有点怪异。

    2.如果return_seq，那这个seq的长度是每个句子的实际长度还是这个batch中最大的那个长度？

    3.如果是每个句子的实际长度，那paddle是怎么做到batch计算的；如果是batch中的最长句子长度，那如果对返回的seq操作时，怎样知道句子的实际长度？

+ 问题解答：

    >1.hidden_dim = 512，LSTM hidden vector = 128，为什么要除以4？

    这个并不怪异,LSTM 的hidden size 根据输入向量自动推断出来，需要理解 hidden vector的含义。
    ![](https://user-images.githubusercontent.com/5842774/32474519-72fc9cba-c332-11e7-9c26-2a5fbca0d007.png)

    下面fc层的size是hidden size * 4，而fc层可学习参数就对应了上图公式中圈出来的红色部分。将四个乘法运算合并一次计算完毕，这一点并不怪异。

    ```python
    fc = paddle.layer.fc(
        input=inputs,
        size=hidden_size * 4,
        act=linear,
        param_attr=para_attr,
        bias_attr=bias_attr)
    lstm = paddle.layer.lstmemory(input=fc, reverse 0, act=tanh)
    ```

    >对于第二、第三个问题：

    + RNN每个时间步都有输出，return_seq=True时，会将RNN所有时刻的输出全部返回，就等于句子的实际长度。
    + 一个序列在内存中是连续的，如果一个batch中含有多个不等长的序列，在PaddlePaddle中有专门的信息标示每个序列在batch中的起始位置。这个信息在data layer中接受序列输入时，已经计算好，不需要用户特别的关心。


## 8.问题：lstmemory这个layer的输出是什么样子？

+ 关键字：`lstmemory`

+ 问题描述：看了代码没有看的太懂，所以想问问lstmemory这个layer的输出是什么？我理解的是，根据公式，在某个时间t，lstm的输出就是一个浮点数，其维度为1没错吧？然后输入一个时间序列x的话，输出的就是一个1维的时间序列y，通过maxpooling之后得到的就是得到的就是max(y)，也只有1维，就是一个浮点数，请问有理解错吗？

+ 问题解答：您的理解是不对的，公式里的输入、输出都是向量表示，并不是单个数，此类写法在deep learning相关论文中是比较普遍的。相应的，lstmemory在每个时间步t的输出是一个向量，而非一个数。

    公式里的$x_t$是向量，不是一个数，$W_*$都是矩阵。这种向量化的符号表示是比较常见的，可以参考
    http://ufldl.stanford.edu/wiki/index.php/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C


