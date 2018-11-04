# 词向量

## 背景介绍
词的向量表征，也称为word embedding，是自然语言处理中常见的一个操作，是搜索引擎、广告系统、推荐系统等互联网服务背后常见的基础技术。

在这些互联网服务里，我们经常要比较两个词或者两段文本之间的相关性。为了做这样的比较，我们往往先要把词表示成计算机适合处理的方式。最自然的方式恐怕莫过于向量空间模型(vector space model)。 在这种方式里，每个词被表示成一个实数向量（one-hot vector），其长度为字典大小，每个维度对应一个字典里的每个词，除了这个词对应维度上的值是1，其他元素都是0。

在机器学习领域里，各种“知识”被各种模型表示，词向量模型(word embedding model)就是其中的一类。词向量模型可以是概率模型、共生矩阵(co-occurrence matrix)模型或神经元网络模型。基于神经网络的模型不需要计算存储一个在全语料上统计的大表，而是通过学习语义信息得到词向量。


1.model_zoo词向量训练数据规模
https://github.com/PaddlePaddle/Paddle/issues/1008

2.embedding_layer层产出的词向量的两个问题
https://github.com/PaddlePaddle/Paddle/issues/2770

## 词向量PaddlePaddle-fluid版代码：
https://github.com/PaddlePaddle/book/tree/c66605770b1ea4d04f290c21d8b72ef4e4d6f7e6/04.word2vec


## 1.问题：怎么加载预训练的embedding层？

+ 关键字：`预加载` `embedding`

+ 问题描述：现在需要将paddlepaddle框架里面的embedding层替换成我在网络上找到的word2vec字典，然后用这个字典来embedding, 这个embedding层怎么外接？

+ 问题分析：
embedding层主要的作用是词嵌套，即将词转换为对应的向量，在PaddlePaddle中词嵌套的做法是类似的，简单而言，对于任意语料数据，在PaddlePaddle中都可以通过类似的操作将其embedding，这里可以参考词向量的文档 http://www.paddlepaddle.org/documentation/docs/zh/1.0/beginners_guide/basics/word2vec/index.html

其代码如下：

    ```
        embed_first = fluid.layers.embedding(
            input=first_word,
            size=[dict_size,EMBED_SIZE],
            dtype='float32',
            is_sparse=is_sparse, #是否使用稀疏更新的标志。
            param_attr='shared_w' #该图层参数
        )

    	.
    	.
    	.

        # 此函数沿着提到的轴连接输入并将其作为输出返回，即沿axis方向连接
        # 将输入的词连接成一层
        concat_embed = fluid.layers.concat(
            input=[embed_first,
                   embed_second,
                   embed_third,
                   embed_fourth],axis=1)
    ```

    问题中描述了word2vec字典，但没有指明是已经做完词嵌套的字典还是普通的字典，如果是普通的字典，处理方式就类似上面的代码，该代码使用N-gram做词嵌套，如果是已经做了词嵌套，那么问的真正的问题是怎么加载已经使用word2vec训练好的词向量模型，即怎么加载预训练好的embedding层，加载预训练的embedding层的代码为：

    ```
    # 通过is_static=True设置embedding层的param_attr为固定
    emb_para = paddle.attr.Param(name='emb', initial_std=0., is_static=True)
    paddle.layer.embedding(size=word_dim, input=x, param_attr=emb_para)

    # 创建parameters后将embedding层参数赋值为预训练的词向量
    parameters = paddle.parameters.create(crf_cost)
    parameters.set('emb', load_parameter(conll05.get_embedding(), 44068, 32))
    ```

    预加载embedding的代码可以参考 语义角色标注的代码 https://github.com/PaddlePaddle/book/blob/develop/07.label_semantic_roles/train.py

+ 问题拓展：
词嵌套技术其实是自然语言处理的基础，因为要进行自然语言处理，第一件事就是怎么让计算机明白我们的语言，这就需要使用词嵌套，词嵌套的方式有很多，但最本质的理论其实就是统计概率，简单而言，当前主流观点认为词的含义可以通过其周围的词来描述，无论是N-gram还是word2vec都是这个思想，背后就是统计学，一个词周围如果经常出现某些词，就可以通过这些词来描述这个词，通过这种方法就解决了使用向量表示词后，已经可以保留词中的内在信息。

+ 问题研究：
PaddlePaddle做词嵌套训练的方式是类似的，都是通过固定的几个方法接受相应的传入数据，所以使用其他语料数据进行训练其实没有什么特别要操作的，需要的注意就是喂养语料数据相应的格式。





## 2.问题：加载预训练的embedding层参数出错误 

+ 关键字： `预训练` `embedding`

+ 问题描述：因为需要外加一个embedding层，所以做了如下处理：

```
emb = embedding_layer(input=data, size=emb_dim, param_attr=ParamAttr(name='_source_language_embedding'))
```

    这个_source_language_embedding文件是本地的embedding文件除去第一列词之后的向量做了二进制转换之后的文件。

    dataprovider中的word_dict换成了原始embedding文件的词和对应每一行的id的映射。

    ```
    settings.word_dict
    def stacked_lstm_net(input_dim,
    class_dim=2,
    emb_dim=50,
    hid_dim=512,
    stacked_num=3,
    is_predict=False):
    ```

    网络中的emb_dim改成了embedding文件中向量的维度，50维。
    然后跑的时候出现了上面的报错。

+ 报错输出：

```
F0711 15:20:57.064468 15230 Parameter.cpp:383] Check failed: s.read(reinterpret_cast<char*>(vec.getData()), header.size * sizeof(real)) 
*** Check failure stack trace: ***
    @          0x13b4438  google::LogMessage::Fail()
    @          0x13b4390  google::LogMessage::SendToLog()
    @          0x13b3e25  google::LogMessage::Flush()
    @          0x13b6be6  google::LogMessageFatal::~LogMessageFatal()
    @           0x82e17f  paddle::Parameter::load()
    @           0x82e86d  paddle::Parameter::load()
    @           0x6d90ba  paddle::GradientMachine::loadParameters()
    @           0x712efc  paddle::ParameterUtil::loadParametersWithPath()
    @           0x70202f  paddle::Trainer::init()
    @           0x591ee9  main
    @     0x7f4433a81bd5  __libc_start_main
    @           0x59dbf5  (unknown)
/home/huhaibing01/tools/paddlell/bin/paddle_local: line 109: 15230 Aborted                 ${DEBUGGER} $MYDIR/../opt/paddle/bin/paddle_trainer ${@:2}
 @QiJune
```

+ 问题分析：
加载预训练embedding层出现错误，可能性有多种，下面尝试一步步来解决这个具体问题，从问题描述中可知其对向量了做了二进制转换，那么先要确认向量做二进制转换是使用PaddlePaddle通过保存参数来实现的还是使用其他方式实现的？如果是使用PaddlePaddle实现的，那么一种可能的原因就是参数转过过程没有正确的转化为可以被PaddlePaddle加载的参数，从报错也可以看出错误出自paddle_trainer，该模块主要是做转转化的，如果使用时传入的是错误的参数，PaddlePaddle运行到此处就会报错。

+ 问题讨论：
通过分析，推断问题可以出现在二进制转换时，虽然使用PaddlePaddle进行转换，但没有转换出可以被PaddlePaddle加载的参数，即错乱了，将转化方式修改一下，修改代码如下：

    ```python
    def get_para_count(input):
        """
        Compute the total number of embedding parameters in input text file.
        input: the name of input text file
        """
        numRows = 1
        paraDim = 0
        with open(input) as f:
            line = f.readline()
            line = line.strip()
            paraDim = len(line.split(" "))
            for line in f:
                numRows += 1
        print(paraDim)
        return numRows * (paraDim-1)

    def text2binary(input, output, paddle_head=True):
        """
        Convert a text parameter file of embedding model to be a binary file.
        input: the name of input text parameter file, for example:
               -0.7845433,1.1937413,-0.1704215,...
               0.0000909,0.0009465,-0.0008813,...
               ...
               the format is:
               1) it doesn't have filehead
               2) each line stores the same dimension of parameters,
                  the separator is commas ','
        output: the name of output binary parameter file, the format is:
               1) the first 16 bytes is filehead:
                 version(4 bytes), floatSize(4 bytes), paraCount(8 bytes)
               2) the next (paraCount * 4) bytes is parameters, each has 4 bytes
        """
        #fi = open(input, "r")
        #fo = open(output, "wb")

        newHead = struct.pack("iil", 0, 4, get_para_count(input))
        np.random.seed()
        #header = struct.pack("iil", 0, 4, height * width)
        param = np.float32(np.random.rand(5590593, 50))
        fi = open(input, 'r')
        count_i = 0
        for line in fi:
            line = line.split(" ")
            count_j = 0
            for x in range(1, len(line)):
                    param[count_i][count_j] = float(line[x])
                    count_j += 1
            count_i += 1
        with open(output, "wb") as fparam:
            fparam.write(newHead + param.tostring())
    ```

    依旧出现报错，报错如下：

    ```
    I0712 14:19:19.295415  5174 PyDataProvider2.cpp:257] loading dataprovider dataprovider::process
    [INFO 2017-07-12 14:19:23,378 dataprovider.py:26] dict len : 5590593
    I0712 14:19:23.396491  5174 GradientMachine.cpp:123] Loading parameters from ./opinion_data/pre-car
    I0712 14:19:24.078368  5174 Parameter.cpp:344] missing parameters [./opinion_data/pre-car/___fc_layer_0__.w0] while loading model.
    I0712 14:19:24.078407  5174 Parameter.cpp:354] ___fc_layer_0__.w0 missing, set to random.
    I0712 14:19:24.093816  5174 Parameter.cpp:344] missing parameters [./opinion_data/pre-car/___fc_layer_0__.wbias] while loading model.
    I0712 14:19:24.093829  5174 Parameter.cpp:354] ___fc_layer_0__.wbias missing, set to random.
    I0712 14:19:24.093876  5174 Parameter.cpp:344] missing parameters [./opinion_data/pre-car/___lstmemory_0__.w0] while loading model.
    I0712 14:19:24.093881  5174 Parameter.cpp:354] ___lstmemory_0__.w0 missing, set to random.
    I0712 14:19:24.098644  5174 Parameter.cpp:344] missing parameters [./opinion_data/pre-car/___lstmemory_0__.wbias] while loading model.
    I0712 14:19:24.098651  5174 Parameter.cpp:354] ___lstmemory_0__.wbias missing, set to random.
    I0712 14:19:24.098726  5174 Parameter.cpp:344] missing parameters [./opinion_data/pre-car/___fc_layer_1__.w0] while loading model.
    I0712 14:19:24.098731  5174 Parameter.cpp:354] ___fc_layer_1__.w0 missing, set to random.
    I0712 14:19:24.117671  5174 Parameter.cpp:344] missing parameters [./opinion_data/pre-car/___fc_layer_1__.w1] while loading model.
    I0712 14:19:24.117678  5174 Parameter.cpp:354] ___fc_layer_1__.w1 missing, set to random.
    ```

    此时报出信息已经变了，此时的信息其实不是报错信息，而是正常的信息，`___fc_layer_1__.w1 missing, set to random.`表示w1这个参数没有提供初始化，将采用随机初始化。此时一开始的问题已经解决了，变成了另一个正常现象，虽然报出信息不是报错，但模型依旧没有正常训练。

+ 问题分析：
观察修改代码后的输出信息，可以判断，此时退出的原因已经不是转化问题了，即此时已经和加载预训练embedding层的参数没有关系了，至少从输出的日志信息来看已经没有问题了，但出现这种输出，通常这种情况，即训练模型时，没有做什么训练，代码没有报错就直接退出了，可能是`data provider`没有提供任何训练数据，训练任务没有数据要处理，其他环节也没有什么异常就会直接退出。建议依然是请优先检测读数据过程。

+ 解决方法：
对于一开始出现的问题，需要修改二进制转化的相关代码，参考代码为 https://github.com/WoNiuHu/Paddle/blob/develop/v1_api_demo/model_zoo/embedding/paraconvert.py ，在参考代码中将随机矩阵转换成二进制，你只需要把随机矩阵替换成你需要的即可，此时就可以解决加载预训练的embedding层参数出错误的问题了，如果出现什么都没做就报出信息的情况，请先考虑数据提供者`data provider`是否真正的提供了数据，如果没有提供数据且其它环节并无异常，训练代码就会直接退出。



## 3.问题：PaddlePaddle存储的二进制模型参数文件的格式是什么样，如何转为明文？

+ 关键字：`提取数据` `二进制模型参数文件`

+ 问题描述：由于模型文件太大，没有办法在本地加载，需要取出embedding层的参数存到Key-Value的存储系统中，即从二进制文件中提取出明文的参数，有没有什么接口可以获取某一层的参数，以及其对应的结构是什么样的？

+ 解决方法：
使用PaddlePaddle读取明文的参数比较简单，下面是示例代码：

    1.将 PaddlePaddle 保存的二进制参数还原回明文txt：

    ```
    def read_parameter(fname, width):
        s = open(fname).read()
        # 跳过16 位 头信息
        vec = np.fromstring(s[16:], dtype=np.float32)
        # 需要指定 width 还原回原始矩阵的形状，width 是配置中layer的size
        np.savetxt(fname + ".csv", vec.reshape(width, -1),
                fmt="%.6f", delimiter=",")
    ```

    2.将明文的参数转化为可以被 PaddlePaddle 加载的模型参数，主要用于预训练：
    下面的代码生成一个随机矩阵，保存为可以被 PaddlePaddle 加载的模型参数。

    ```
    def gen_rand_param(param_file, width, height, need_trans):
        np.random.seed()
        # 头信息的前两位第一位固定写0，第二位固定写4，只有在使用 double  精度时，需要改为8
        # 第三位信息记录一共有多少个数值
        header = struct.pack("iil", 0, 4, height * width)
        param = np.float32(np.random.rand(height, width))
        with open(param_file, "w") as fparam:
            fparam.write(header + param.tostring())
    ```

    代码中的width 就是网络配置中layer的size

+ 问题拓展：
不同的深度学习框架保存模型时具体的保存格式可能不同，但通常都会提供保存模型参数的方法与相应的读入方法，方便使用者预加载保存好的模型，这部分在实际项目开发中是很重要的，通常而言，学习训练一个模型就是为了处理相应的任务，如果不将进过训练所学到的参数持久化存储起来，就没有什么意义了。


## 4.问题：PaddlePaddle中embedding的作用是什么？

+ 关键字： `embedding作用`

+ 问题描述：看了PaddlePaddle词向量相关的代码，大量使用了embedding_layer+fc_layer的结构，请问这里常用的paddle.layer.embedding()方法的具有有什么作用？在PaddlePaddle中是如何实现的？

+ 相关代码：

```python
embed_first = fluid.layers.embedding(
        input=first_word,
        size=[dict_size,EMBED_SIZE],
        dtype='float32',
        is_sparse=is_sparse, #是否使用稀疏更新的标志。
        param_attr='shared_w' #该图层参数
    )
```

+ 问题讨论：
问题描述中其实涉及了两个主要的问题，即paddlepaddle中embedding()方法的作用是什么？以及PaddlePaddle如何实现embedding方法？下面就这两个问题简单解答一下

+ 问题解答：
关于 embedding 的作用：

	+ 1.为了让计算机处理输入，首先需要对数据的规范化表示方法。one-hot，BOW，n-gram 等等，都是人类设计出的不同表示输入数据方法。
	+ 2.embedding 是一种distributed representation，它的出现是相对于 one-hot 的表示法。
		+ 在 one-hot 表示方法中，一个编码单元表示一个个体，除了某一个维度上的值是 1，其余维度都是 0。
		+ distributed representation 用几个编码单元而不是一个编码单元来表示一个个体，是一类表示学习方法，用一个更低维度的实向量表示“一个概念”（可以理解为机器学习任务输入观察的输入特征），向量的每个维度在实数域 $R$ 取值
	+ 3.embedding 在自然语言处理任务中获得了很大的成功，所以也常被翻译为“词向量”。但是，作为一类表示学方法，我们可以为所有离散的输入学习对应的 embedding 表达，并不不局限于自然语言处理任务中的词语。
	+ 4.简单而言，引入 embedding 的动机有：
		+ 将高度稀疏的离散输入嵌入到一个新的实向量空间，对抗维数灾难，使用更少的维度，编码更丰富的信息。
		+ 我们观测的不同离散变量都可以嵌入同一个实向量空间，得到统一的表达形式。定义于实向量空间上的各种数学运算，可以作为描述语义的数学工具。

	embeding层可以理解为从一个矩阵中选择一行，一行对应着一个离散的新的特征表达，是一种取词操作

	关于PaddlePaddle中embedding方法是如何实现的？可以参考 https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/trainer_config_helpers/layers.py#L938，PaddlePaddle的embedding是用table_projection实现的，关键代码片段为：

	```python
	with mixed_layer(
            name=name,
            size=size,
            act=LinearActivation(),
            bias_attr=False,
            layer_attr=layer_attr) as mix:
        mix += table_projection(input=input, size=size, param_attr=param_attr)
    return mix
    ```

    table_projection的计算公式是`out.row[i] += table.row[ids[i]]`，在上述代码连接中也可以找到

+ 问题拓展：
embedding，即词向量，这个概念并不是PaddlePaddle独有的，在自然语言处理方面，embedding操作也是非常常见的，其主要目的将词转为相应的向量，这样一个词就可以映射到高维空间，一个常见的做法就是计算词与词之间的距离，因为词被表示成了向量，计算距离就是简单的计算两个向量间的距离，通过这种向量计算获得的值通常都可以看作是两个词之间的相似度。




## 5.问题：在PaddlePaddle中embedding层和fc层的区别在哪里？

+ 关键字：`embedding` `全连接层` 

+ 问题描述：在使用PaddlePaddle实现词向量相关任务时，参考了官方词向量相关的代码，里面使用了很多embedding层与fc层链接使用的结构，请问embedding层与fc层具有有什么区别？

+ 相关代码：

```python
 embed_fourth = fluid.layers.embedding(
        input=fourth_word,
        size=[dict_size, EMBED_SIZE],
        dtype='float32',
        is_sparse=is_sparse,
        param_attr='shared_w')

    # 此函数沿着提到的轴连接输入并将其作为输出返回，即沿axis方向连接
    # 将输入的词连接成一层
    concat_embed = fluid.layers.concat(
        input=[embed_first,
               embed_second,
               embed_third,
               embed_fourth],axis=1)
    #隐藏层，全连接
    hidden1 = fluid.layers.fc(input=concat_embed,
                              size=HIDDEN_SIZE,
                              act='sigmoid')
```

+ 问题解答：
简单而言，embeding层可以理解为从一个矩阵中选择一行，一行对应着一个离散的新的特征表达，是一种取词操作，而fc层，即全连接层，它的实质是矩阵乘法。一些用户会将embeding看成一种训练的结果，这是不对的，embedding 层仅仅就只是完成“取矩阵的一行”这样的操作，而获得的结果是构建词向量的神经网络在训练时更新模型参数的结果。


+ 问题研究：
embeding层与全连接层虽然在作用上有明显的不同，但两层训练的方式是完全相同的，都是通过反向传播算法获得梯度，通过梯度下降算法进行相应参数的更新，只是embeding的目的就是这些更新后的参数，这些参数组成的矩阵就是训练后要获得的词向量，而全连接层的参数就只是参数，某种程度反应了训练数据的特征。




## 6.问题：如何加载的外部预训练的embedding也作为模型参数参加训练

+ 关键字：`embedding` `预训练`

+ 问题描述：
训练的时候，如何加载的外部预训练的embedding也作为模型参数参加训练？

+ 相关代码：

```python
emb = embedding_layer(input=data, size=emb_dim, param_attr=ParamAttr(name='_source_language_embedding'))`

emb_para = ParameterAttribute(name='emb', initial_std=0., learning_rate=0.)
fc1 = fc_layer(input=emb, size=hid_dim, act=linear, bias_attr=bias_attr, param_attr=emb_para)
```
    
是否像上面这样写，就可以实现加载外部预训练的embedding了？

+ 问题解答：
请把 parameters.set调用放在 paddle.parameters.create调用之后。
下面的代码把embedding参数的学习率设置为0，这和 is_static=True 参数是等价的，embedding这个参数将不再更新，请确定这是您需要。

```python
emb_para = ParameterAttribute(name='emb', initial_std=0., learning_rate=0.)
```


## 7.问题：如何选择concat拼接的axis？

+ 关键字： `concat` `axis`

+ 问题描述：在参考官方文档中关于词向量的写法时，有个疑惑，在官方文档中对词进行了concat拼接（4x(32x1)->1x128x1），想问下如何拼接得到（4x32x1)来复现word2vec源码进行sum pooling？

+ 相关代码：

    ```python
    paddle.init(use_gpu=False, trainer_count=3)
    word_dict = paddle.dataset.imikolov.build_dict()
    dict_size = len(word_dict)
    # Every layer takes integer value of range [0, dict_size)
    firstword = paddle.layer.data(
        name="firstw", type=paddle.data_type.integer_value(dict_size))
    secondword = paddle.layer.data(
        name="secondw", type=paddle.data_type.integer_value(dict_size))
    thirdword = paddle.layer.data(
        name="thirdw", type=paddle.data_type.integer_value(dict_size))
    fourthword = paddle.layer.data(
        name="fourthw", type=paddle.data_type.integer_value(dict_size))
    nextword = paddle.layer.data(
        name="fifthw", type=paddle.data_type.integer_value(dict_size))

    Efirst = wordemb(firstword)
    Esecond = wordemb(secondword)
    Ethird = wordemb(thirdword)
    Efourth = wordemb(fourthword)

    contextemb = paddle.layer.concat(input=[Efirst, Esecond, Ethird, Efourth])
    ```

+ 问题解答：
要实现词向量的sum pooling，可以参考如下代码https://github.com/reyoung/ChineseWordVectors/blob/master/cbow.py#L36

    在PaddlePaddle中，词向量的sum pooling其实就是一次输入一个数组，使用`paddle.pooling.Sum`完成

+ 问题讨论：
要实现类似下图中的video watches or video search tokens 的数量会存在小于固定值（50）的情况,而不是像word2vec的数量是固定的window size这部分应该怎么处理?

![](https://raw.githubusercontent.com/PaddlePaddle/book/develop/05.recommender_system/image/Deep_candidate_generation_model_architecture.en.png)

使用PaddlePaddle的pooling则可，类似如下代码：

```python
mov_title_id = paddle.layer.data(
    name='movie_title',
    type=paddle.data_type.integer_value_sequence(len(movie_title_dict)))
mov_title_emb = paddle.layer.embedding(input=mov_title_id, size=32)
paddle.layer.pooling(input= mov_title_emb, pooling_type=paddle.pooling.Sum())
```


## 8.问题： 在PaddlePaddle中嵌入实现究竟是什么？

+ 关键字： `embedding` `嵌入`

+ 问题描述：
在使用PaddlePaddle时我很好奇PaddlePaddle的嵌入层是如何实现的，我浏览的源代码，但并没有找到相关的内容，我希望你们可以帮助我找到这一块内容

+ 问题解答：
PaddlePaddle中的嵌入层实现代码：https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/gserver/layers/TableProjection.cpp

Emebedding层在Paddle中是一个投影，投影是一种轻量级的层，本身不能独立存在，只能作为paddle.layer.mixed_layer的输入。

+ 问题讨论：
在官方的相关脚本中，有如下代码：

```python
data = paddle.layer.data（“ word ”，paddle.data_type.integer_value_sequence（input_dim））
emb = paddle.layer.embedding（input = data，size = emb_dim）; 
conv_3 = paddle.networks.sequence_conv_pool（input = emb，context_len = 3， 
 hidden_​​size = hid_dim）
```

    我们其中，我们想自己替换嵌入这一步，数据转EMB，数据怎么解析，转成我们的向量后怎么封装成EMB，查看嵌入也就是相对其内部格式有一个了解，这样才能对接

    要替换前，需要先理解PaddlePaddle存储的二进制参数文件的格式，这个可以参考问题3的内容，接着可以继续参考PaddleBook中SRL一节中的实例代码，使用了加载预训练参数，你使用方法也可以参考这个例子，https://github.com/PaddlePaddle/book/blob/develop/07.label_semantic_roles/train.py#L142
















