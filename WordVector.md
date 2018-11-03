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
