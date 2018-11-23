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

## `待审阅` 1.问题：使用PTB数据集训练词向量模型出现张量类型错误

 + 关键字：`数据类型`，`dtype`

 + 问题描述：使用PTB数据集训练词向量模型，设置输入层的`dtype`参数值为`float32`，在启动训练的时候出现张量类型错误。

 + 报错信息：

```
<ipython-input-6-daf8837e1db3> in train(use_cuda, train_program, params_dirname)
     37         num_epochs=1,
     38         event_handler=event_handler,
---> 39         feed_order=['firstw', 'secondw', 'thirdw', 'fourthw', 'nextw'])

/usr/local/lib/python3.5/dist-packages/paddle/fluid/contrib/trainer.py in train(self, num_epochs, event_handler, reader, feed_order)
    403         else:
    404             self._train_by_executor(num_epochs, event_handler, reader,
--> 405                                     feed_order)
    406 
    407     def test(self, reader, feed_order):

/usr/local/lib/python3.5/dist-packages/paddle/fluid/contrib/trainer.py in _train_by_executor(self, num_epochs, event_handler, reader, feed_order)
    481             exe = executor.Executor(self.place)
    482             reader = feeder.decorate_reader(reader, multi_devices=False)
--> 483             self._train_by_any_executor(event_handler, exe, num_epochs, reader)
    484 
    485     def _train_by_any_executor(self, event_handler, exe, num_epochs, reader):

/usr/local/lib/python3.5/dist-packages/paddle/fluid/contrib/trainer.py in _train_by_any_executor(self, event_handler, exe, num_epochs, reader)
    510                                       fetch_list=[
    511                                           var.name
--> 512                                           for var in self.train_func_outputs
    513                                       ])
    514                 else:

/usr/local/lib/python3.5/dist-packages/paddle/fluid/executor.py in run(self, program, feed, fetch_list, feed_var_name, fetch_var_name, scope, return_numpy, use_program_cache)
    468 
    469         self._feed_data(program, feed, feed_var_name, scope)
--> 470         self.executor.run(program.desc, scope, 0, True, True)
    471         outs = self._fetch_data(fetch_list, fetch_var_name, scope)
    472         if return_numpy:

EnforceNotMet: Tensor holds the wrong type, it holds f at [/paddle/paddle/fluid/framework/tensor_impl.h:29]
PaddlePaddle Call Stacks: 
```

 + 问题复现：在使用`fluid.layers.data`接口定义网络的输出层，设置每个输入层的`name`为单独的名称，`shape`的值为`[1]`且设置`dtype`的值为`float32`，启动训练的时候就会出现该错误。错误代码如下：

```python
first_word = fluid.layers.data(name='firstw', shape=[1], dtype='float32')
second_word = fluid.layers.data(name='secondw', shape=[1], dtype='float32')
third_word = fluid.layers.data(name='thirdw', shape=[1], dtype='float32')
fourth_word = fluid.layers.data(name='fourthw', shape=[1], dtype='float32')
```

 + 解决问题：因为PTB数据集下训练的时候，已经把单词转换成整数，所以输入的数据应该是整数而不是浮点数字，出现的错误也是因为这个原因。正确代码如下：

```python
first_word = fluid.layers.data(name='firstw', shape=[1], dtype='int64')
second_word = fluid.layers.data(name='secondw', shape=[1], dtype='int64')
third_word = fluid.layers.data(name='thirdw', shape=[1], dtype='int64')
fourth_word = fluid.layers.data(name='fourthw', shape=[1], dtype='int64')
```

 + 问题拓展：PaddlePaddle的输入层数据类型有`float`、`int`、`uint`、`bool`，但是就没有字符串类型，所以训练数据都会转换成相应数据类型，所以在PTB数据集数据中也是把字符串的单词转换成整型。

 + 问题分析：编写神经网络时，获得编写任何程序时，细节都是重要的，细节不正确就会导致程序运行不起来，而深度学习的编程中，类型不正确是常出现的错误，要避免这类错误，你需要熟悉你使用的训练数据的数据类型，如果不熟悉，此时最好的方法就是在使用时打印一下数据的类型与shape，方便编写出正确的fluid.layers.data




## `待审阅` 2.问题：设置向量表征类型为整型时训练报错

 + 关键字：`数据类型`，`词向量`

 + 问题描述：定义N-gram神经网络训练PTB数据集时，使用PaddlePaddle内置的`fluid.layers.embedding`接口计算词向量，当设置该数据类型为`int64`时报错。

 + 报错信息：

```
<ipython-input-6-daf8837e1db3> in train(use_cuda, train_program, params_dirname)
     31         # optimizer=fluid.optimizer.SGD(learning_rate=0.001),
     32         optimizer_func=optimizer_func,
---> 33         place=place)
     34 
     35     trainer.train(

/usr/local/lib/python3.5/dist-packages/paddle/fluid/contrib/trainer.py in __init__(self, train_func, optimizer_func, param_path, place, parallel, checkpoint_config)
    280         with self._prog_and_scope_guard():
    281             exe = executor.Executor(place)
--> 282             exe.run(self.startup_program)
    283 
    284         if self.checkpoint_cfg and self.checkpoint_cfg.load_serial is not None:

/usr/local/lib/python3.5/dist-packages/paddle/fluid/executor.py in run(self, program, feed, fetch_list, feed_var_name, fetch_var_name, scope, return_numpy, use_program_cache)
    468 
    469         self._feed_data(program, feed, feed_var_name, scope)
--> 470         self.executor.run(program.desc, scope, 0, True, True)
    471         outs = self._fetch_data(fetch_list, fetch_var_name, scope)
    472         if return_numpy:

EnforceNotMet: op uniform_random does not have kernel for data_type[int64_t]:data_layout[ANY_LAYOUT]:place[CPUPlace]:library_type[PLAIN] at [/paddle/paddle/fluid/framework/operator.cc:733]
PaddlePaddle Call Stacks: 
```

 + 问题复现：使用`fluid.layers.embedding`接口定义词向量时，设置参数`dtype`的值为`int64`，`size`为`[数据的单词数量, 词向量维度]`，在训练的时候就会报这个错误。错误代码如下：

```python
embed_first = fluid.layers.embedding(
    input=first_word,
    size=[dict_size, EMBED_SIZE],
    dtype='int64',
    is_sparse=is_sparse,
    param_attr='shared_w')
embed_second = fluid.layers.embedding(
    input=second_word,
    size=[dict_size, EMBED_SIZE],
    dtype='int64',
    is_sparse=is_sparse,
    param_attr='shared_w')
embed_third = fluid.layers.embedding(
    input=third_word,
    size=[dict_size, EMBED_SIZE],
    dtype='int64',
    is_sparse=is_sparse,
    param_attr='shared_w')
embed_fourth = fluid.layers.embedding(
    input=fourth_word,
    size=[dict_size, EMBED_SIZE],
    dtype='int64',
    is_sparse=is_sparse,
    param_attr='shared_w')
```

 + 解决问题：输入层的数据类型虽然是`int64`，但是词向量的数据类型是`float32`。用户可能是理解误以为词向量的数据类型也许是`int64`，所以才会导致错误。正确代码如下：

```python
embed_first = fluid.layers.embedding(
    input=first_word,
    size=[dict_size, EMBED_SIZE],
    dtype='float32',
    is_sparse=is_sparse,
    param_attr='shared_w')
embed_second = fluid.layers.embedding(
    input=second_word,
    size=[dict_size, EMBED_SIZE],
    dtype='float32',
    is_sparse=is_sparse,
    param_attr='shared_w')
embed_third = fluid.layers.embedding(
    input=third_word,
    size=[dict_size, EMBED_SIZE],
    dtype='float32',
    is_sparse=is_sparse,
    param_attr='shared_w')
embed_fourth = fluid.layers.embedding(
    input=fourth_word,
    size=[dict_size, EMBED_SIZE],
    dtype='float32',
    is_sparse=is_sparse,
    param_attr='shared_w')
```

 + 问题拓展：词向量模型可将一个 one-hot vector映射到一个维度更低的实数向量（embedding vector），如`embedding(母亲节)=[0.3,4.2,−1.5,...]`，`embedding(康乃馨)=[0.2,5.6,−2.3,...]`。在这个映射到的实数向量表示中，希望两个语义（或用法）上相似的词对应的词向量“更像”。

 + 问题分析：NLP中，词向量技术是比较底层的计算，是很多上层技术的支撑，如RNN、LSTM等，输入都是经过词向量嵌入后的向量，将词编码成相应要保持其语义信息是更好的，即将词编程稠密向量，one-hot独热向量虽然简单，但会编码维度灾难与语义鸿沟的问题。


## `待审阅` 3.问题：在使用PTB数据集训练词向量模型出现输入(X)和输入(label)的形状不一致

 + 关键字：`数据维度`，`concat`

 + 问题描述：在使用N-gram神经网络训练PTB数据集时，使用接口`fluid.layers.concat`把四个词向量连接起来，最后经过全连接层输出。但是把`fluid.layers.concat`的`axis`参数设置为0时就报错。

 + 报错信息：

```
<ipython-input-6-daf8837e1db3> in train(use_cuda, train_program, params_dirname)
     37         num_epochs=1,
     38         event_handler=event_handler,
---> 39         feed_order=['firstw', 'secondw', 'thirdw', 'fourthw', 'nextw'])

/usr/local/lib/python3.5/dist-packages/paddle/fluid/contrib/trainer.py in train(self, num_epochs, event_handler, reader, feed_order)
    403         else:
    404             self._train_by_executor(num_epochs, event_handler, reader,
--> 405                                     feed_order)
    406 
    407     def test(self, reader, feed_order):

/usr/local/lib/python3.5/dist-packages/paddle/fluid/contrib/trainer.py in _train_by_executor(self, num_epochs, event_handler, reader, feed_order)
    481             exe = executor.Executor(self.place)
    482             reader = feeder.decorate_reader(reader, multi_devices=False)
--> 483             self._train_by_any_executor(event_handler, exe, num_epochs, reader)
    484 
    485     def _train_by_any_executor(self, event_handler, exe, num_epochs, reader):

/usr/local/lib/python3.5/dist-packages/paddle/fluid/contrib/trainer.py in _train_by_any_executor(self, event_handler, exe, num_epochs, reader)
    510                                       fetch_list=[
    511                                           var.name
--> 512                                           for var in self.train_func_outputs
    513                                       ])
    514                 else:

/usr/local/lib/python3.5/dist-packages/paddle/fluid/executor.py in run(self, program, feed, fetch_list, feed_var_name, fetch_var_name, scope, return_numpy, use_program_cache)
    468 
    469         self._feed_data(program, feed, feed_var_name, scope)
--> 470         self.executor.run(program.desc, scope, 0, True, True)
    471         outs = self._fetch_data(fetch_list, fetch_var_name, scope)
    472         if return_numpy:

EnforceNotMet: Enforce failed. Expected framework::slice_ddim(x_dims, 0, rank - 1) == framework::slice_ddim(label_dims, 0, rank - 1), but received framework::slice_ddim(x_dims, 0, rank - 1):400 != framework::slice_ddim(label_dims, 0, rank - 1):100.
Input(X) and Input(Label) shall have the same shape except the last dimension. at [/paddle/paddle/fluid/operators/cross_entropy_op.cc:37]
PaddlePaddle Call Stacks: 
```

 + 问题复现：使用四个`fluid.layers.embedding`接口建立四个词向量，接着把这四个词向量通过`fluid.layers.concat`接口连接在一起，但是在设置`axis`参数的值为0的时候就会报错。错误代码如下：

```python
embed_first = fluid.layers.embedding(
    input=first_word,
    size=[dict_size, EMBED_SIZE],
    dtype='float32',
    is_sparse=is_sparse,
    param_attr='shared_w')
embed_second = fluid.layers.embedding(
    input=second_word,
    size=[dict_size, EMBED_SIZE],
    dtype='float32',
    is_sparse=is_sparse,
    param_attr='shared_w')
embed_third = fluid.layers.embedding(
    input=third_word,
    size=[dict_size, EMBED_SIZE],
    dtype='float32',
    is_sparse=is_sparse,
    param_attr='shared_w')
embed_fourth = fluid.layers.embedding(
    input=fourth_word,
    size=[dict_size, EMBED_SIZE],
    dtype='float32',
    is_sparse=is_sparse,
    param_attr='shared_w')

concat_embed = fluid.layers.concat(
    input=[embed_first, embed_second, embed_third, embed_fourth], axis=0)
```

 + 解决问题：`fluid.layers.concat`接口中的`axis`参数是指把上面的四个将张量连接在一起的整数轴，因为输出数据都是一维的，所以这个参数的应该是1。经过拼接后，该层输出的形状应该是`(Batch大小, 四个词向量维度的和)`。

```python
embed_first = fluid.layers.embedding(
    input=first_word,
    size=[dict_size, EMBED_SIZE],
    dtype='float32',
    is_sparse=is_sparse,
    param_attr='shared_w')
embed_second = fluid.layers.embedding(
    input=second_word,
    size=[dict_size, EMBED_SIZE],
    dtype='float32',
    is_sparse=is_sparse,
    param_attr='shared_w')
embed_third = fluid.layers.embedding(
    input=third_word,
    size=[dict_size, EMBED_SIZE],
    dtype='float32',
    is_sparse=is_sparse,
    param_attr='shared_w')
embed_fourth = fluid.layers.embedding(
    input=fourth_word,
    size=[dict_size, EMBED_SIZE],
    dtype='float32',
    is_sparse=is_sparse,
    param_attr='shared_w')

concat_embed = fluid.layers.concat(
    input=[embed_first, embed_second, embed_third, embed_fourth], axis=1)
```

 + 问题拓展：`fluid.layers.embedding`接口是将高度稀疏的离散输入嵌入到一个新的实向量空间，对抗维数灾难，使用更少的维度，编码更丰富的信息。之后使用`fluid.layers.concat`把多个这样的向量并在一起，最后把数据送入到神经网络中。




## `待审阅` 4.问题：在使用N-gram神经网络训练时出现ids[i]>row_number的错误

 + 关键字：`词向量`，`N-gram神经网络`

 + 问题描述：在使用N-gram神经网络训练PTB数据集时，手动设置了字典大小，最后在启动训练的时候出现错误，错误提示ids[i]>row_number。

 + 报错信息：

```
<ipython-input-6-daf8837e1db3> in train(use_cuda, train_program, params_dirname)
     37         num_epochs=1,
     38         event_handler=event_handler,
---> 39         feed_order=['firstw', 'secondw', 'thirdw', 'fourthw', 'nextw'])

/usr/local/lib/python3.5/dist-packages/paddle/fluid/contrib/trainer.py in train(self, num_epochs, event_handler, reader, feed_order)
    403         else:
    404             self._train_by_executor(num_epochs, event_handler, reader,
--> 405                                     feed_order)
    406 
    407     def test(self, reader, feed_order):

/usr/local/lib/python3.5/dist-packages/paddle/fluid/contrib/trainer.py in _train_by_executor(self, num_epochs, event_handler, reader, feed_order)
    481             exe = executor.Executor(self.place)
    482             reader = feeder.decorate_reader(reader, multi_devices=False)
--> 483             self._train_by_any_executor(event_handler, exe, num_epochs, reader)
    484 
    485     def _train_by_any_executor(self, event_handler, exe, num_epochs, reader):

/usr/local/lib/python3.5/dist-packages/paddle/fluid/contrib/trainer.py in _train_by_any_executor(self, event_handler, exe, num_epochs, reader)
    510                                       fetch_list=[
    511                                           var.name
--> 512                                           for var in self.train_func_outputs
    513                                       ])
    514                 else:

/usr/local/lib/python3.5/dist-packages/paddle/fluid/executor.py in run(self, program, feed, fetch_list, feed_var_name, fetch_var_name, scope, return_numpy, use_program_cache)
    468 
    469         self._feed_data(program, feed, feed_var_name, scope)
--> 470         self.executor.run(program.desc, scope, 0, True, True)
    471         outs = self._fetch_data(fetch_list, fetch_var_name, scope)
    472         if return_numpy:

EnforceNotMet: Enforce failed. Expected ids[i] < row_number, but received ids[i]:2073 >= row_number:32.
 at [/paddle/paddle/fluid/operators/lookup_table_op.h:59]
PaddlePaddle Call Stacks: 
```

 + 问题复现：在是使用`fluid.layers.embedding`接口构造N-gram神经网络，其中设置`size`参数的值为`[32, 32]`，在执行训练的时候就会报该错误。错误代码如下：

```python
dict_size = 32
embed_first = fluid.layers.embedding(
    input=first_word,
    size=[dict_size, EMBED_SIZE],
    dtype='float32',
    is_sparse=is_sparse,
    param_attr='shared_w')
embed_second = fluid.layers.embedding(
    input=second_word,
    size=[dict_size, EMBED_SIZE],
    dtype='float32',
    is_sparse=is_sparse,
    param_attr='shared_w')
embed_third = fluid.layers.embedding(
    input=third_word,
    size=[dict_size, EMBED_SIZE],
    dtype='float32',
    is_sparse=is_sparse,
    param_attr='shared_w')
embed_fourth = fluid.layers.embedding(
    input=fourth_word,
    size=[dict_size, EMBED_SIZE],
    dtype='float32',
    is_sparse=is_sparse,
    param_attr='shared_w')
```

 + 解决问题：`fluid.layers.embedding`接口的`size`参数的值应该是`[数据集的字典大小, 词向量大小]`。所以第一个参数的值应该是字典的大小，所以要根据字典的实际大小来进行赋值。正确代码如下：

```python
word_dict = paddle.dataset.imikolov.build_dict()
dict_size = len(word_dict)

embed_first = fluid.layers.embedding(
    input=first_word,
    size=[dict_size, EMBED_SIZE],
    dtype='float32',
    is_sparse=is_sparse,
    param_attr='shared_w')
embed_second = fluid.layers.embedding(
    input=second_word,
    size=[dict_size, EMBED_SIZE],
    dtype='float32',
    is_sparse=is_sparse,
    param_attr='shared_w')
embed_third = fluid.layers.embedding(
    input=third_word,
    size=[dict_size, EMBED_SIZE],
    dtype='float32',
    is_sparse=is_sparse,
    param_attr='shared_w')
embed_fourth = fluid.layers.embedding(
    input=fourth_word,
    size=[dict_size, EMBED_SIZE],
    dtype='float32',
    is_sparse=is_sparse,
    param_attr='shared_w')
```

 + 问题分析：关于fluid.layers.embedding()方法的更多内容，可以参考API文档相关部分：
 http://www.paddlepaddle.org/documentation/docs/zh/1.1/api/layers.html#embedding


## `待审阅` 5.问题：在创建预测数据时报错data and recursive_seq_lens do not match

 + 关键字：`张量`，`数据维度`

 + 问题描述：在使用`fluid.create_lod_tensor`创建一个词向量预测输出，在执行创建的时候报错，数据维度和参数recursive_seq_lens不匹配。

 + 报错信息：

```
<ipython-input-7-422a1a374a70> in infer(use_cuda, inference_program, params_dirname)
     17     lod = [[2]]
     18 
---> 19     first_word = fluid.create_lod_tensor(data1, lod, place)
     20     second_word = fluid.create_lod_tensor(data2, lod, place)
     21     third_word = fluid.create_lod_tensor(data3, lod, place)

/usr/local/lib/python3.5/dist-packages/paddle/fluid/lod_tensor.py in create_lod_tensor(data, recursive_seq_lens, place)
     74         assert [
     75             new_recursive_seq_lens
---> 76         ] == recursive_seq_lens, "data and recursive_seq_lens do not match"
     77         flattened_data = np.concatenate(data, axis=0).astype("int64")
     78         flattened_data = flattened_data.reshape([len(flattened_data), 1])

AssertionError: data and recursive_seq_lens do not match
```

 + 问题复现：定义一个形状为`(1, 1)`的整型数据，然后再定义一个`(1, 1)`的表示列表的长度，这个设置为2，最后在执行` fluid.create_lod_tensor`接口创建预测数据的时候报错。错误代码如下：

```python
data1 = [[211]]  # 'among'
data2 = [[6]]  # 'a'
data3 = [[96]]  # 'group'
data4 = [[4]]  # 'of'
lod = [[2]]

first_word = fluid.create_lod_tensor(data1, lod, place)
second_word = fluid.create_lod_tensor(data2, lod, place)
third_word = fluid.create_lod_tensor(data3, lod, place)
fourth_word = fluid.create_lod_tensor(data4, lod, place)
```

 + 解决问题：`recursive_seq_lens`这个参数是指输入的数据列表的长度信息，我们输入的数据的长度是1，所以参数`recursive_seq_lens`也应该是的值也应该是`[[1]]`，而不是输入数据的维度数量。正确代码如下：

```python
data1 = [[211]]  # 'among'
data2 = [[6]]  # 'a'
data3 = [[96]]  # 'group'
data4 = [[4]]  # 'of'
lod = [[1]]

first_word = fluid.create_lod_tensor(data1, lod, place)
second_word = fluid.create_lod_tensor(data2, lod, place)
third_word = fluid.create_lod_tensor(data3, lod, place)
fourth_word = fluid.create_lod_tensor(data4, lod, place)
```

 + 问题拓展：`fluid.create_lod_tensor`接口也支持多维不同长度的数据，如：创建一个张量来表示两个句子，一个是2个单词，一个是3个单词。那就需要设置`recursive_seq_lens`参数的值为`[[2,3]]`。




## `待审阅` 6.问题：在执行词向量预测的是出现预测数据不能完全转换为Python ndarray的错误

 + 关键字：`预测`，`numpy`

 + 问题描述：通过使用`fluid.create_lod_tensor`创建一个预测数据，然后使用预测器对数据进行预测，在执行预测的时候，报预测的张量数据中有一些包含LoD信息，它们不能完全转换为Python ndarray的错误。

 + 报错信息：

```
<ipython-input-9-196da4d402b0> in infer(use_cuda, inference_program, params_dirname)
     27             'secondw': second_word,
     28             'thirdw': third_word,
---> 29             'fourthw': fourth_word
     30         })
     31 

······
/usr/local/lib/python3.5/dist-packages/paddle/fluid/contrib/inferencer.py in infer(self, inputs, return_numpy)
    102             results = self.exe.run(feed=inputs,
    103                                    fetch_list=[self.predict_var.name],
--> 104                                    return_numpy=return_numpy)
    105 
    106         return results

/usr/local/lib/python3.5/dist-packages/paddle/fluid/executor.py in run(self, program, feed, fetch_list, feed_var_name, fetch_var_name, scope, return_numpy, use_program_cache)
    471         outs = self._fetch_data(fetch_list, fetch_var_name, scope)
    472         if return_numpy:
--> 473             outs = as_numpy(outs)
    474         return outs

/usr/local/lib/python3.5/dist-packages/paddle/fluid/executor.py in as_numpy(tensor)
     83         return [as_numpy(t) for t in tensor]
     84     if isinstance(tensor, list):
---> 85         return [as_numpy(t) for t in tensor]
     86     assert isinstance(tensor, core.LoDTensor)
     87     lod = tensor.lod()

/usr/local/lib/python3.5/dist-packages/paddle/fluid/executor.py in <listcomp>(.0)
     83         return [as_numpy(t) for t in tensor]
     84     if isinstance(tensor, list):
---> 85         return [as_numpy(t) for t in tensor]
     86     assert isinstance(tensor, core.LoDTensor)
     87     lod = tensor.lod()

/usr/local/lib/python3.5/dist-packages/paddle/fluid/executor.py in as_numpy(tensor)
     90             They can not be completely cast to Python ndarray. \
     91             Please set the parameter 'return_numpy' as 'False' to \
---> 92             return LoDTensor itself directly.")
     93     return np.array(tensor)
     94 

RuntimeError: Some of your fetched tensors hold LoD information.             They can not be completely cast to Python ndarray.             Please set the parameter 'return_numpy' as 'False' to             return LoDTensor itself directly.

```

 + 问题复现：使用定义的网络和训练过程中保存的模型参数创建一个预测器，然后再使用` fluid.create_lod_tensor`创建4个预测数据，最后使用预测器对4个预测数据进行预测，没有设置参数`return_numpy`，在执行的时候就报错。错误代码如下：

```python
first_word = fluid.create_lod_tensor(data1, lod, place)
second_word = fluid.create_lod_tensor(data2, lod, place)
third_word = fluid.create_lod_tensor(data3, lod, place)
fourth_word = fluid.create_lod_tensor(data4, lod, place)

result = inferencer.infer(
    {
        'firstw': first_word,
        'secondw': second_word,
        'thirdw': third_word,
        'fourthw': fourth_word
    })
```

 + 解决问题：在对词向量进行预测时，返回的结果并不是一个numpy值，而接口`paddle.fluid.contrib.inferencer.infer`默认的返回值的是一个numpy类型的，所以就报错。需要设置参数`return_numpy`的值为`False`。正确代码如下：

```python
first_word = fluid.create_lod_tensor(data1, lod, place)
second_word = fluid.create_lod_tensor(data2, lod, place)
third_word = fluid.create_lod_tensor(data3, lod, place)
fourth_word = fluid.create_lod_tensor(data4, lod, place)

result = inferencer.infer(
    {
        'firstw': first_word,
        'secondw': second_word,
        'thirdw': third_word,
        'fourthw': fourth_word
    },
    return_numpy=False)
```

 + 问题拓展：接口`paddle.fluid.contrib.inferencer.infer`是属于高层接口，与低层接口不同的是，高层接口不需要用户操作执行器。高层接口使用更简单，虽然没有低层接口灵活，但是使用方便，适合初学者使用。




## `待审阅` 7.问题：在使用词向量模型预测字符串数据时出现数据类型错误

 + 关键字：`张量`，`预测数据`

 + 问题描述：在使用`fluid.create_lod_tensor`接口定义词向量预测数据的时候，预测数据时一个二维的字符串列表数据，出现数据类型的错误。

 + 报错信息：

```
<ipython-input-7-746fcd239131> in infer(use_cuda, inference_program, params_dirname)
     17     lod = [[1]]
     18 
---> 19     first_word = fluid.create_lod_tensor(data1, lod, place)
     20     second_word = fluid.create_lod_tensor(data2, lod, place)
     21     third_word = fluid.create_lod_tensor(data3, lod, place)

/opt/conda/envs/py35-paddle1.0.0/lib/python3.5/site-packages/paddle/fluid/lod_tensor.py in create_lod_tensor(data, recursive_seq_lens, place)
     75             new_recursive_seq_lens
     76         ] == recursive_seq_lens, "data and recursive_seq_lens do not match"
---> 77         flattened_data = np.concatenate(data, axis=0).astype("int64")
     78         flattened_data = flattened_data.reshape([len(flattened_data), 1])
     79         return create_lod_tensor(flattened_data, recursive_seq_lens, place)

ValueError: invalid literal for int() with base 10: 'among'
```

 + 问题复现：定义一个`data1`的二维字符串列表变量，然后使用这变量通过`fluid.create_lod_tensor`接口创建一个张量数据，用于之后预测的数据，在执行创建的时候就会报错。错误代码如下：

```python
data1 = [['among']]
data2 = [['a']]
data3 = [['group']]
data4 = [['of']]
lod = [[1]]

first_word = fluid.create_lod_tensor(data1, lod, place)
second_word = fluid.create_lod_tensor(data2, lod, place)
third_word = fluid.create_lod_tensor(data3, lod, place)
fourth_word = fluid.create_lod_tensor(data4, lod, place)
```

 + 解决问题：`fluid.create_lod_tensor`接口的参数`data`的类型是一个numpy、列表胡或者是张量，上面之所以会报错，是因为使用了字符串，这些需要的是一个整型的单词标签。正确代码如下：

```python
data1 = [[211]]  # 'among'
data2 = [[6]]  # 'a'
data3 = [[96]]  # 'group'
data4 = [[4]]  # 'of'
lod = [[1]]

first_word = fluid.create_lod_tensor(data1, lod, place)
second_word = fluid.create_lod_tensor(data2, lod, place)
third_word = fluid.create_lod_tensor(data3, lod, place)
fourth_word = fluid.create_lod_tensor(data4, lod, place)
```

 + 问题分析：在PaddlePaddle的训练数据中，都是以数字的形式表示的，并不会使用字符串的方式进行训练。所以这种NLP数据都会有一个数据字典，把字符串的文字转换成整型数据，再进行训练。




## `待审阅` 8.问题：在调用PaddlePaddle提供的词向量数据集接口用于训练时出现数据长度错误

 + 关键字：`PTB数据集`，`数据维度`

 + 问题描述：使用PaddlePaddle提供的词向量PTB数据集接口`paddle.dataset.imikolov.train`创建训练数据，然后使用这个数据进行训练时，出现错误，错误提示数据的长度不正确。

 + 报错信息：

```
<ipython-input-6-daf8837e1db3> in train(use_cuda, train_program, params_dirname)
     37         num_epochs=1,
     38         event_handler=event_handler,
---> 39         feed_order=['firstw', 'secondw', 'thirdw', 'fourthw', 'nextw'])

/opt/conda/envs/py35-paddle1.0.0/lib/python3.5/site-packages/paddle/fluid/contrib/trainer.py in train(self, num_epochs, event_handler, reader, feed_order)
    403         else:
    404             self._train_by_executor(num_epochs, event_handler, reader,
--> 405                                     feed_order)
    406 
    407     def test(self, reader, feed_order):

/opt/conda/envs/py35-paddle1.0.0/lib/python3.5/site-packages/paddle/fluid/contrib/trainer.py in _train_by_executor(self, num_epochs, event_handler, reader, feed_order)
    481             exe = executor.Executor(self.place)
    482             reader = feeder.decorate_reader(reader, multi_devices=False)
--> 483             self._train_by_any_executor(event_handler, exe, num_epochs, reader)
    484 
    485     def _train_by_any_executor(self, event_handler, exe, num_epochs, reader):

/opt/conda/envs/py35-paddle1.0.0/lib/python3.5/site-packages/paddle/fluid/contrib/trainer.py in _train_by_any_executor(self, event_handler, exe, num_epochs, reader)
    494         for epoch_id in epochs:
    495             event_handler(BeginEpochEvent(epoch_id))
--> 496             for step_id, data in enumerate(reader()):
    497                 if self.__stop:
    498                     if self.checkpoint_cfg:

/opt/conda/envs/py35-paddle1.0.0/lib/python3.5/site-packages/paddle/fluid/data_feeder.py in __reader_creator__()
    275             if not multi_devices:
    276                 for item in reader():
--> 277                     yield self.feed(item)
    278             else:
    279                 num = self._get_number_of_places_(num_places)

/opt/conda/envs/py35-paddle1.0.0/lib/python3.5/site-packages/paddle/fluid/data_feeder.py in feed(self, iterable)
    189             assert len(each_sample) == len(converter), (
    190                 "The number of fields in data (%s) does not match " +
--> 191                 "len(feed_list) (%s)") % (len(each_sample), len(converter))
    192             for each_converter, each_slot in six.moves.zip(converter,
    193                                                            each_sample):

AssertionError: The number of fields in data (7) does not match len(feed_list) (5)
```

 + 问题复现：使用PTB数据集`paddle.dataset.imikolov.build_dict`创建一个数据集字典，然后使用这个字典通过调用`paddle.dataset.imikolov.train`接口创建一个训练数据，参数`n`设置为7，启动训练的时候就会上面的错误。错误代码如下：

```python
word_dict = paddle.dataset.imikolov.build_dict()
train_reader = paddle.batch(paddle.dataset.imikolov.train(word_dict, 7), 64)
trainer.train(
    reader=train_reader,
    num_epochs=1,
    event_handler=event_handler,
    feed_order=['firstw', 'secondw', 'thirdw', 'fourthw', 'nextw'])
```

 + 解决问题：我们在训练时定义的`feed_order`只有5个输入数据，包括一个label的数据，而在定义训练数据的长度是7，所以导致输入数据的长度不同。`paddle.dataset.imikolov.train`接口的参数应该设置为5。正确代码如下：

```python
word_dict = paddle.dataset.imikolov.build_dict()
train_reader = paddle.batch(paddle.dataset.imikolov.train(word_dict, 5), 64)
trainer.train(
    reader=train_reader,
    num_epochs=1,
    event_handler=event_handler,
    feed_order=['firstw', 'secondw', 'thirdw', 'fourthw', 'nextw'])
```

 + 问题拓展：PaddlePaddle提供的`paddle.dataset.imikolov.train`接口可以动态设置输出一条数据的单词数量，如果要修改这个数量，需要修改网络的词向量数量和训练接口的`feed_order`参数值。




## `已审阅` 9.问题：在使用词向量模型预测是得不到预测的单词

 + 关键字：`预测结果`，`数据字典`

 + 问题描述：使用训练好的模型参数和定义的网络创建一个预测器，使用这样预测器预测数据，得到一个预测结果，但是这个预测结果不是一个单词，而是一个数字。


 + 问题复现：通过使用`paddle.fluid.contrib.inferencer.infer`预测接口预测四个单词数据，得到下一个单词。但是把最大概率的结果值输出，得到的是一个数字，而不是想要的一个字符串的单词。

```python
result = inferencer.infer(inputs={'firstw': first_word,
                                  'secondw': second_word,
                                  'thirdw': third_word,
                                  'fourthw': fourth_word},
                          return_numpy=False)
most_possible_word_index = numpy.argmax(result[0])
print("预测结果是：", most_possible_word_index)
```

 + 解决问题：我们训练的时候传入的是一个张量数据，所以预测输出的也应该是一个张量类型的数据，输出的也就是一个张量数据。如果需要输出预测的当初，还有根据数据集的字典获取对应的单词。处理代码如下：

```python
result = inferencer.infer(inputs={'firstw': first_word,
                                  'secondw': second_word,
                                  'thirdw': third_word,
                                  'fourthw': fourth_word},
                          return_numpy=False)
most_possible_word_index = numpy.argmax(result[0])
print("预测结果是：", [key for key, value in six.iteritems(word_dict) if value == most_possible_word_index][0])
```

 + 问题拓展：在图像分类任务上，预测得接时每个类别得概率，从这些结果中获取最概率的也是其数字标签。如果想得到类别的名称，还要根据数字标签对应的类别的名称才能获取预测结果的名称。


## `已审阅` 10.问题：怎么加载预训练的embedding层？

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

## `已审阅` 11.问题：PaddlePaddle中embedding的作用是什么？

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




## `已审阅` 12.问题：在PaddlePaddle中embedding层和fc层的区别在哪里？

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

