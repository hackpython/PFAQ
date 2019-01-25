# 机器翻译

## 背景介绍
机器翻译（machine translation, MT）是用计算机来实现不同语言之间翻译的技术。被翻译的语言通常称为源语言（source language），翻译成的结果语言称为目标语言（target language）。机器翻译即实现从源语言到目标语言转换的过程，是自然语言处理的重要研究领域之一。

近年来，深度学习技术的发展为解决上述挑战提供了新的思路。将深度学习应用于机器翻译任务的方法大致分为两类：1）仍以统计机器翻译系统为框架，只是利用神经网络来改进其中的关键模块，如语言模型、调序模型等（见图1的左半部分）；2）不再以统计机器翻译系统为框架，而是直接用神经网络将源语言映射到目标语言，即端到端的神经网络机器翻译（End-to-End Neural Machine Translation, End-to-End NMT）（见图1的右半部分），简称为NMT模型。


## `已审核`1.问题：'map' object is not subscriptable

+ 问题描述：我按照PaddlePaddle官方文档编写机器翻译模型，出现这个错误，对照了文档中的代码，也没有编写错误。

+ 报错信息：

```
Original sentence:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-27-e241afef7936> in <module>()
     20 
     21     print("Original sentence:")
---> 22     print(" ".join([src_dict[w] for w in feed_data[0][0][1:-1]]))
     23 
     24     print("Translated score and sentence:")

TypeError: 'map' object is not subscriptable
```

+ 问题复现：

```
exe = Executor(place)
exe.run(framework.default_startup_program())

for data in test_data():
    feed_data = map(lambda x: [x[0]], data)
    feed_dict = feeder.feed(feed_data)
    feed_dict['init_ids'] = init_ids
    feed_dict['init_scores'] = init_scores

    results = exe.run(
        framework.default_main_program(),
        feed=feed_dict,
        fetch_list=[translation_ids, translation_scores],
        return_numpy=False)
```

+ 问题分析：
在Python3中，map返回的会是一个map类型的可迭代对象，该对象不同直接通过下标获取，在Python2中是没有问题的，出现该问题，只需要将代码修改成python3兼容的模式则可


+ 问题解决：

如果想通过下标获取map对象，可以先将map对象转为list对象，这样就可以直接通过下标获取了

```
exe = Executor(place)
exe.run(framework.default_startup_program())

for data in test_data():
    feed_data = list(map(lambda x: [x[0]], data))
    feed_dict = feeder.feed(feed_data)
    feed_dict['init_ids'] = init_ids
    feed_dict['init_scores'] = init_scores

    results = exe.run(
        framework.default_main_program(),
        feed=feed_dict,
        fetch_list=[translation_ids, translation_scores],
        return_numpy=False)
```

+ 问题拓展：
map()方法是python内置方法，python2与python3中map()方法是有不同的，python3中考虑到一切性将数据全部返回会比较消耗内存，就就修改成生成对象的形式，即取的时候才会获得，而且只生效一次。

## `已审核`2.问题：按照文档编写，出现name 'result_ids_lod' is not defined

+ 问题描述：我使用Fluid1.1按照文档编写相应的结果，出现name 'result_ids_lod' is not defined错误

+ 报错信息：

```
Original sentence:
<unk> , l&apos; indice de la bourse technologique a perdu 0,8 % , et a fini avec <unk> points du marché .
Translated score and sentence:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<ipython-input-14-9adba1019ebd> in <module>()
     23     print("Translated score and sentence:")
     24     for i in range(beam_size):
---> 25         start_pos = result_ids_lod[1][i] + 1
     26         end_pos = result_ids_lod[1][i+1]
     27         print("%d\t%.4f\t%s\n" % (i+1, result_scores[end_pos-1],

NameError: name 'result_ids_lod' is not defined
```

+ 问题复现：

```
for data in test_data():
    feed_data = list(map(lambda x: [x[0]], data))
    feed_dict = feeder.feed(feed_data)
    feed_dict['init_ids'] = init_ids
    feed_dict['init_scores'] = init_scores

    results = exe.run(
        framework.default_main_program(),
        feed=feed_dict,
        fetch_list=[translation_ids, translation_scores],
        return_numpy=False)

    result_ids = np.array(results[0])
    result_scores = np.array(results[1])
```

+ 问题解决：

文档中的代码缺少了`result_ids_lod = results[0].lod()`，导致出现了上述问题。

```
for data in test_data():
    feed_data = list(map(lambda x: [x[0]], data))
    feed_dict = feeder.feed(feed_data)
    feed_dict['init_ids'] = init_ids
    feed_dict['init_scores'] = init_scores

    results = exe.run(
        framework.default_main_program(),
        feed=feed_dict,
        fetch_list=[translation_ids, translation_scores],
        return_numpy=False)

    result_ids = np.array(results[0])
    result_ids_lod = results[0].lod()
    result_scores = np.array(results[1])
```

最新的代码请参考https://github.com/PaddlePaddle/book/blob/fa35415f2b3f5a5d3e045ff0564d5df0a5a2b0d5/08.machine_translation/infer.py


## `已审核`3.问题：The number of fields in data (3) does not match len(feed_list)

+ 问题描述：使用PaddlePaddle构建机器翻译模型，出现`The number of fields in data (3) does not match len(feed_list)`

```
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
<ipython-input-17-49ae4ab0d47c> in <module>()
      5         num_epochs=EPOCH_NUM,
      6         event_handler=event_handler,
----> 7         feed_order=feed_order)

~/anaconda3/envs/paddle/lib/python3.5/site-packages/paddle/fluid/contrib/trainer.py in train(self, num_epochs, event_handler, reader, feed_order)
    403         else:
    404             self._train_by_executor(num_epochs, event_handler, reader,
--> 405                                     feed_order)
    406 
    407     def test(self, reader, feed_order):

~/anaconda3/envs/paddle/lib/python3.5/site-packages/paddle/fluid/contrib/trainer.py in _train_by_executor(self, num_epochs, event_handler, reader, feed_order)
    481             exe = executor.Executor(self.place)
    482             reader = feeder.decorate_reader(reader, multi_devices=False)
--> 483             self._train_by_any_executor(event_handler, exe, num_epochs, reader)
    484 
    485     def _train_by_any_executor(self, event_handler, exe, num_epochs, reader):

~/anaconda3/envs/paddle/lib/python3.5/site-packages/paddle/fluid/contrib/trainer.py in _train_by_any_executor(self, event_handler, exe, num_epochs, reader)
    494         for epoch_id in epochs:
    495             event_handler(BeginEpochEvent(epoch_id))
--> 496             for step_id, data in enumerate(reader()):
    497                 if self.__stop:
    498                     if self.checkpoint_cfg:

~/anaconda3/envs/paddle/lib/python3.5/site-packages/paddle/fluid/data_feeder.py in __reader_creator__()
    275             if not multi_devices:
    276                 for item in reader():
--> 277                     yield self.feed(item)
    278             else:
    279                 num = self._get_number_of_places_(num_places)

~/anaconda3/envs/paddle/lib/python3.5/site-packages/paddle/fluid/data_feeder.py in feed(self, iterable)
    189             assert len(each_sample) == len(converter), (
    190                 "The number of fields in data (%s) does not match " +
--> 191                 "len(feed_list) (%s)") % (len(each_sample), len(converter))
    192             for each_converter, each_slot in six.moves.zip(converter,
    193                                                            each_sample):

AssertionError: The number of fields in data (3) does not match len(feed_list) (1)
```

+ 问题复现：

```
def decode(context, is_sparse):
    init_state = context
    array_len = pd.fill_constant(shape=[1], dtype='int64', value=max_length)
    counter = pd.zeros(shape=[1], dtype='int64', force_cpu=True)

    # fill the first element with init_state
    state_array = pd.create_array('float32')
    pd.array_write(init_state, array=state_array, i=counter)

    # ids, scores as memory
    ids_array = pd.create_array('int64')
    scores_array = pd.create_array('float32')

    init_ids = pd.data(name="init_ids", shape=[1], dtype="int64", lod_level=1)
    init_scores = pd.data(
        name="init_scores", shape=[1], dtype="float32", lod_level=1)
```

+ 问题分析：从报错信息与复现代码来看，可能对PaddlePaddle中lod_level的概念没有理解清楚，load_level表示LoDTensor的等级，不传是默认为0，表示输入的数据不是序列数据，这方面更具体的内容可以参考http://www.paddlepaddle.org/documentation/docs/en/1.0/design/concepts/lod_tensor.html


+ 问题解决：

观察输入的训练数据，将lod_level改成对应的等级

```
def decode(context, is_sparse):
    init_state = context
    array_len = pd.fill_constant(shape=[1], dtype='int64', value=max_length)
    counter = pd.zeros(shape=[1], dtype='int64', force_cpu=True)

    # fill the first element with init_state
    state_array = pd.create_array('float32')
    pd.array_write(init_state, array=state_array, i=counter)

    # ids, scores as memory
    ids_array = pd.create_array('int64')
    scores_array = pd.create_array('float32')

    init_ids = pd.data(name="init_ids", shape=[1], dtype="int64", lod_level=2)
    init_scores = pd.data(
        name="init_scores", shape=[1], dtype="float32", lod_level=2)
```


## `已审核`4.问题：Tensor holds the wrong type

+ 问题描述：我根据文档编写机器翻译模型，出现了`Tensor holds the wrong type`

+ 报错信息：

```
---------------------------------------------------------------------------
EnforceNotMet                             Traceback (most recent call last)
<ipython-input-11-49ae4ab0d47c> in <module>()
      5         num_epochs=EPOCH_NUM,
      6         event_handler=event_handler,
----> 7         feed_order=feed_order)

~/anaconda3/envs/paddle/lib/python3.5/site-packages/paddle/fluid/contrib/trainer.py in train(self, num_epochs, event_handler, reader, feed_order)
    403         else:
    404             self._train_by_executor(num_epochs, event_handler, reader,
--> 405                                     feed_order)
    406 
    407     def test(self, reader, feed_order):

~/anaconda3/envs/paddle/lib/python3.5/site-packages/paddle/fluid/contrib/trainer.py in _train_by_executor(self, num_epochs, event_handler, reader, feed_order)
    481             exe = executor.Executor(self.place)
    482             reader = feeder.decorate_reader(reader, multi_devices=False)
--> 483             self._train_by_any_executor(event_handler, exe, num_epochs, reader)
    484 
    485     def _train_by_any_executor(self, event_handler, exe, num_epochs, reader):

~/anaconda3/envs/paddle/lib/python3.5/site-packages/paddle/fluid/contrib/trainer.py in _train_by_any_executor(self, event_handler, exe, num_epochs, reader)
    510                                       fetch_list=[
    511                                           var.name
--> 512                                           for var in self.train_func_outputs
    513                                       ])
    514                 else:

~/anaconda3/envs/paddle/lib/python3.5/site-packages/paddle/fluid/executor.py in run(self, program, feed, fetch_list, feed_var_name, fetch_var_name, scope, return_numpy, use_program_cache)
    468 
    469         self._feed_data(program, feed, feed_var_name, scope)
--> 470         self.executor.run(program.desc, scope, 0, True, True)
    471         outs = self._fetch_data(fetch_list, fetch_var_name, scope)
    472         if return_numpy:

EnforceNotMet: Tensor holds the wrong type, it holds i at [/Users/paddle/minqiyang/Paddle/paddle/fluid/framework/tensor_impl.h:29]
PaddlePaddle Call Stacks: 
0          0x11f436a68p paddle::platform::EnforceNotMet::EnforceNotMet(std::exception_ptr, char const*, int) + 760
1          0x11f792f42p long long const* paddle::framework::Tensor::data<long long>() const + 258
2          0x11fc36fabp paddle::operators::LookupTableKernel<float>::Compute(paddle::framework::ExecutionContext const&) const + 331
3          0x11fc36e20p std::__1::__function::__func<paddle::framework::OpKernelRegistrarFunctor<paddle::platform::CPUPlace, false, 0ul, paddle::operators::LookupTableKernel<float>, paddle::operators::LookupTableKernel<double> >::operator()(char const*, char const*) const::'lambda'(paddle::framework::ExecutionContext const&), std::__1::allocator<paddle::framework::OpKernelRegistrarFunctor<paddle::platform::CPUPlace, false, 0ul, paddle::operators::LookupTableKernel<float>, paddle::operators::LookupTableKernel<double> >::operator()(char const*, char const*) const::'lambda'(paddle::framework::ExecutionContext const&)>, void (paddle::framework::ExecutionContext const&)>::operator()(paddle::framework::ExecutionContext const&) + 32
4          0x12025f223p paddle::framework::OperatorWithKernel::RunImpl(paddle::framework::Scope const&, boost::variant<paddle::platform::CUDAPlace, paddle::platform::CPUPlace, paddle::platform::CUDAPinnedPlace, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_> const&) const + 659
5          0x12025b141p paddle::framework::OperatorBase::Run(paddle::framework::Scope const&, boost::variant<paddle::platform::CUDAPlace, paddle::platform::CPUPlace, paddle::platform::CUDAPinnedPlace, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_> const&) + 577
6          0x11f5043a6p paddle::framework::Executor::RunPreparedContext(paddle::framework::ExecutorPrepareContext*, paddle::framework::Scope*, bool, bool, bool) + 390
7          0x11f503dd3p paddle::framework::Executor::Run(paddle::framework::ProgramDesc const&, paddle::framework::Scope*, int, bool, bool) + 163
8          0x11f46a837p void pybind11::cpp_function::initialize<paddle::pybind::pybind11_init()::$_64, void, paddle::framework::Executor&, paddle::framework::ProgramDesc const&, paddle::framework::Scope*, int, bool, bool, pybind11::name, pybind11::is_method, pybind11::sibling>(paddle::pybind::pybind11_init()::$_64&&, void (*)(paddle::framework::Executor&, paddle::framework::ProgramDesc const&, paddle::framework::Scope*, int, bool, bool), pybind11::name const&, pybind11::is_method const&, pybind11::sibling const&)::'lambda'(pybind11::detail::function_call&)::__invoke(pybind11::detail::function_call&) + 135
9          0x11f4413aap pybind11::cpp_function::dispatcher(_object*, _object*, _object*) + 5786
10         0x10f4b559fp PyCFunction_Call + 127
11         0x10f5807e7p PyEval_EvalFrameEx + 33207
12         0x10f576fafp _PyEval_EvalCodeWithName + 335
13         0x10f57d2a7p PyEval_EvalFrameEx + 19575
14         0x10f576fafp _PyEval_EvalCodeWithName + 335
15         0x10f57d2a7p PyEval_EvalFrameEx + 19575
16         0x10f57cfb8p PyEval_EvalFrameEx + 18824
17         0x10f576fafp _PyEval_EvalCodeWithName + 335
18         0x10f57d2a7p PyEval_EvalFrameEx + 19575
19         0x10f576fafp _PyEval_EvalCodeWithName + 335
20         0x10f5713b5p builtin_exec + 341
21         0x10f4b555ep PyCFunction_Call + 62
22         0x10f57cec7p PyEval_EvalFrameEx + 18583
23         0x10f576fafp _PyEval_EvalCodeWithName + 335
24         0x10f57d2a7p PyEval_EvalFrameEx + 19575
25         0x10f576fafp _PyEval_EvalCodeWithName + 335
26         0x10f57d2a7p PyEval_EvalFrameEx + 19575
27         0x10f576fafp _PyEval_EvalCodeWithName + 335
28         0x10f57d2a7p PyEval_EvalFrameEx + 19575
29         0x10f576fafp _PyEval_EvalCodeWithName + 335
30         0x10f4826aap function_call + 106
31         0x10f43eb35p PyObject_Call + 69
32         0x10f57dc9bp PyEval_EvalFrameEx + 22123
33         0x10f576fafp _PyEval_EvalCodeWithName + 335
34         0x10f57d2a7p PyEval_EvalFrameEx + 19575
35         0x10f576fafp _PyEval_EvalCodeWithName + 335
36         0x10f57d2a7p PyEval_EvalFrameEx + 19575
37         0x10f57cfb8p PyEval_EvalFrameEx + 18824
38         0x10f57cfb8p PyEval_EvalFrameEx + 18824
39         0x10f576fafp _PyEval_EvalCodeWithName + 335
40         0x10f4826aap function_call + 106
41         0x10f43eb35p PyObject_Call + 69
42         0x10f57dc9bp PyEval_EvalFrameEx + 22123
43         0x10f576fafp _PyEval_EvalCodeWithName + 335
44         0x10f4826aap function_call + 106
45         0x10f43eb35p PyObject_Call + 69
46         0x10f57dc9bp PyEval_EvalFrameEx + 22123
47         0x10f576fafp _PyEval_EvalCodeWithName + 335
48         0x10f57d2a7p PyEval_EvalFrameEx + 19575
49         0x10f57cfb8p PyEval_EvalFrameEx + 18824
50         0x10f576fafp _PyEval_EvalCodeWithName + 335
51         0x10f4826aap function_call + 106
52         0x10f43eb35p PyObject_Call + 69
53         0x10f57dc9bp PyEval_EvalFrameEx + 22123
54         0x10f576fafp _PyEval_EvalCodeWithName + 335
55         0x10f57d2a7p PyEval_EvalFrameEx + 19575
56         0x10f576fafp _PyEval_EvalCodeWithName + 335
57         0x10f4826aap function_call + 106
58         0x10f43eb35p PyObject_Call + 69
59         0x10f57dc9bp PyEval_EvalFrameEx + 22123
60         0x10f57cfb8p PyEval_EvalFrameEx + 18824
61         0x10f57cfb8p PyEval_EvalFrameEx + 18824
62         0x10f57cfb8p PyEval_EvalFrameEx + 18824
63         0x10f57cfb8p PyEval_EvalFrameEx + 18824
64         0x10f57cfb8p PyEval_EvalFrameEx + 18824
65         0x10f576fafp _PyEval_EvalCodeWithName + 335
66         0x10f57d2a7p PyEval_EvalFrameEx + 19575
67         0x10f576fafp _PyEval_EvalCodeWithName + 335
68         0x10f5713b5p builtin_exec + 341
69         0x10f4b555ep PyCFunction_Call + 62
70         0x10f57cec7p PyEval_EvalFrameEx + 18583
71         0x10f576fafp _PyEval_EvalCodeWithName + 335
72         0x10f57d2a7p PyEval_EvalFrameEx + 19575
73         0x10f576fafp _PyEval_EvalCodeWithName + 335
74         0x10f4826aap function_call + 106
75         0x10f43eb35p PyObject_Call + 69
76         0x10f5ee906p RunModule + 182
77         0x10f5edb03p Py_Main + 2979
78         0x10f42f861p main + 497
79      0x7fff5dffe015p start + 1
80                 0x5p

```

+ 问题复现：

```
def encoder(is_sparse):
    src_word_id = pd.data(
     name="src_word_id", shape=[1], dtype='int32', lod_level=1)
    src_embedding = pd.embedding(
     input=src_word_id,
     size=[dict_size, word_dim],
     dtype='float32',
     is_sparse=is_sparse,
     param_attr=fluid.ParamAttr(name='vemb'))

    fc1 = pd.fc(input=src_embedding, size=hidden_dim * 4, act='tanh')
    lstm_hidden0, lstm_0 = pd.dynamic_lstm(input=fc1, size=hidden_dim * 4)
    encoder_out = pd.sequence_last_step(input=lstm_hidden0)
    return encoder_out
```

+ 问题分析：
该问题很有可能是`paddle.fluid.layers.data`定义的输入类型不正确，Fluid中需要注意的是对于float而言，float32是主要实数类型，而对int而言，int64是主要标签类型。在使用时，最好对应上。


+ 问题解决：

修改成正确的数据类型。

```
def encoder(is_sparse):
    src_word_id = pd.data(
     name="src_word_id", shape=[1], dtype='int64', lod_level=1)
    src_embedding = pd.embedding(
     input=src_word_id,
     size=[dict_size, word_dim],
     dtype='float32',
     is_sparse=is_sparse,
     param_attr=fluid.ParamAttr(name='vemb'))

    fc1 = pd.fc(input=src_embedding, size=hidden_dim * 4, act='tanh')
    lstm_hidden0, lstm_0 = pd.dynamic_lstm(input=fc1, size=hidden_dim * 4)
    encoder_out = pd.sequence_last_step(input=lstm_hidden0)
    return encoder_out
```

## `已审核`5.为什么trainer.train中没有feeding参数也可以保持数据输入与模型中input的关系

+ 关键字：`机器翻译` `feeding参数`

+ 问题描述：
根据paddle文档的介绍，“ Reader返回的数据可以包括多列，我们需要一个Python dict把列 序号映射到网络里的数据层。”
参考：https://github.com/PaddlePaddle/book/blob/develop/01.fit_a_line/train.py

但是我发现在机器翻译例子中，并没有使用feeding方式，没有指定数据与模型input的关系，
参考：https://github.com/PaddlePaddle/book/blob/develop/08.machine_translation/train.py
请问程序是如何保证数据与模型input对应一致的呢

+ 问题解答：
Paddle会对网络配置进行解析，解析出来的 数据层 的顺序和定义顺序一致，而Paddle默认的input顺序和解析出的数据层的顺序一致。所以在不使用feeding方式时，input顺序须和数据层定义顺序保持一致。

## `已审核`6.demo machine translate "beam_search() got multiple values for keyword argument 'end_id'"

+ 关键字：`机器翻译` `beam_search`

+ 问题描述：使用Fluid 0.14 CPU版的PaddlePaddle运行机器翻译的例子时，出现`beam_search() got multiple values for keyword argument 'end_id'`

+ 报错输出：

```bash
Traceback (most recent call last):
  File "infer.py", line 196, in <module>
    main(use_cuda)
  File "infer.py", line 191, in main
    decode_main(False)  # Beam Search does not support CUDA
  File "infer.py", line 133, in decode_main
    translation_ids, translation_scores = decode(context)
  File "infer.py", line 98, in decode
    beam_size, end_id=10, level=0)
TypeError: beam_search() got multiple values for keyword argument 'end_id'
```

+ 问题解答：
这个问题很有可能是PaddlePaddle版本问题，因为`beamsearch`相关的内容在Fluid 0.14版后依旧有相应的修改，所以造成该问题的原因可能是你使用机器学习示例中的用法与Paddle版本中提供的api不匹配。

+ 解决方法：
将PaddlePaddle更新成最新的版本，参考官方最新的机器翻译的实例，如下

    http://www.paddlepaddle.org/documentation/docs/zh/1.0/beginners_guide/basics/machine_translation/index.html

    或者直接参考PaddlePaddle提供的book中关于机器翻译的代码，如下

    https://github.com/PaddlePaddle/book/tree/develop/08.machine_translation


## `待审核`6.问题：使用Fluid编写机器翻译模型，报错

+ 问题描述：使用Fluid编写机器翻译模型，报错

+ 报错输出：

```
Traceback (most recent call last):
  File "train.py", line 173, in <module>
    train()
  File "train.py", line 63, in train
    max_length=args.max_length)
  File "/Users/jizhi/Desktop/Paddle/models/fluid/PaddleNLP/neural_machine_translation/rnn_search/attention_model.py", line 81, in seq_to_seq_net
    input_seq=src_embedding, gate_size=encoder_size)
  File "/Users/jizhi/Desktop/Paddle/models/fluid/PaddleNLP/neural_machine_translation/rnn_search/attention_model.py", line 59, in bi_lstm_encoder
    size=gate_size * 4, use_peepholes=False)
TypeError: dynamic_lstm() missing 1 required positional argument: 'input'
```

+ 相关代码：

```
forward, _ = fluid.layers.dynamic_lstm(
            size=gate_size * 4, use_peepholes=False)
        input_reversed_proj = fluid.layers.fc(input=input_seq,
                                              size=gate_size * 4,
                                              act='tanh',
                                              bias_attr=False)
reversed, _ = fluid.layers.dynamic_lstm(
    input=input_reversed_proj,
    size=gate_size * 4,
    is_reverse=True,
    use_peepholes=False)
```

+ 解决方法：

```
forward, _ = fluid.layers.dynamic_lstm(
        input=input_forward_proj,size=gate_size * 4, use_peepholes=False)
        input_reversed_proj = fluid.layers.fc(input=input_seq,
                                              size=gate_size * 4,
                                              act='tanh',
                                              bias_attr=False)
reversed, _ = fluid.layers.dynamic_lstm(
    input=input_reversed_proj,
    size=gate_size * 4,
    is_reverse=True,
    use_peepholes=False)
```


## `待审核`7.问题：使用Fluid运行官方models中的rnn_search的模型，报错

+ 问题描述：使用Fluid运行官方models中的rnn_search的模型，报错

+ 报错输出：

```
Traceback (most recent call last):
  File "train.py", line 173, in <module>
    train()
  File "train.py", line 103, in train
    exe.run(framework.default_startup_program())
  File "/Users/jizhi/anaconda3/envs/paddle/lib/python3.5/site-packages/paddle/fluid/executor.py", line 470, in run
    self.executor.run(program.desc, scope, 0, True, True)
paddle.fluid.core.EnforceNotMet: Cannot run operator on place CUDAPlace(0) at [/Users/paddle/minqiyang/Paddle/paddle/fluid/framework/operator.cc:146]
PaddlePaddle Call Stacks:
0          0x10d0eaa68p paddle::platform::EnforceNotMet::EnforceNotMet(std::exception_ptr, char const*, int) + 760
1          0x10df0f1f9p paddle::framework::OperatorBase::Run(paddle::framework::Scope const&, boost::variant<paddle::platform::CUDAPlace, paddle::platform::CPUPlace, paddle::platform::CUDAPinnedPlace, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_> const&) + 761
2          0x10d1b83a6p paddle::framework::Executor::RunPreparedContext(paddle::framework::ExecutorPrepareContext*, paddle::framework::Scope*, bool, bool, bool) + 390
3          0x10d1b7dd3p paddle::framework::Executor::Run(paddle::framework::ProgramDesc const&, paddle::framework::Scope*, int, bool, bool) + 163
4          0x10d11e837p void pybind11::cpp_function::initialize<paddle::pybind::pybind11_init()::$_64, void, paddle::framework::Executor&, paddle::framework::ProgramDesc const&, paddle::framework::Scope*, int, bool, bool, pybind11::name, pybind11::is_method, pybind11::sibling>(paddle::pybind::pybind11_init()::$_64&&, void (*)(paddle::framework::Executor&, paddle::framework::ProgramDesc const&, paddle::framework::Scope*, int, bool, bool), pybind11::name const&, pybind11::is_method const&, pybind11::sibling const&)::'lambda'(pybind11::detail::function_call&)::__invoke(pybind11::detail::function_call&) + 135
5          0x10d0f53aap pybind11::cpp_function::dispatcher(_object*, _object*, _object*) + 5786
6          0x10038859fp PyCFunction_Call + 127
7          0x1004537e7p PyEval_EvalFrameEx + 33207
8          0x100449fafp _PyEval_EvalCodeWithName + 335
9          0x1004502a7p PyEval_EvalFrameEx + 19575
10         0x100449fafp _PyEval_EvalCodeWithName + 335
11         0x1004502a7p PyEval_EvalFrameEx + 19575
12         0x100449fafp _PyEval_EvalCodeWithName + 335
13         0x10049c758p PyRun_FileExFlags + 248
14         0x10049beeep PyRun_SimpleFileExFlags + 382
15         0x1004c0d86p Py_Main + 3622
16         0x100302861p main + 497
17      0x7fff5f4da015p start + 1
18                 0x2p
```

+ 问题分析：

报错输出中有`Cannot run operator on place CUDAPlace(0)`，通常是运行设备中没有GPU，使用CPU运作则可。

+ 解决方法：

```
place  = core.CPUPlace()
```


## `待审核`8.问题：通过Fluid实现机器翻译模型报错

+ 问题描述：通过Fluid实现机器翻译模型报错

+ 报错输出：

```
TTraceback (most recent call last):
  File "train.py", line 173, in <module>
    train()
  File "train.py", line 108, in train
    feeder = fluid.DataFeeder()
TypeError: __init__() missing 2 required positional arguments: 'feed_list' and 'place'
```

+ 解决方法：

DataFeeder()方法需要相关的参数，将相关参数传入则可。

```
place  = core.CPUPlace()
exe = Executor(place)
exe.run(framework.default_startup_program())

feed_list = [
    main_program.global_block().var(var_name) for var_name in feed_order
]
feeder = fluid.DataFeeder(feed_list, place)

```


##  `待审核`9.问题：Enforce failed. Expected w_dims[0] == frame_size

+ + 问题描述：运行Fluid编写的模型时，输出Enforce failed. Expected w_dims[0] == frame_size

+ 报错输出：

```
Traceback (most recent call last):
  File "train.py", line 173, in <module>
    train()
  File "train.py", line 63, in train
    max_length=args.max_length)
  File "/Users/jizhi/Desktop/Paddle/models/fluid/PaddleNLP/neural_machine_translation/rnn_search/attention_model.py", line 81, in seq_to_seq_net
    input_seq=src_embedding, gate_size=encoder_size)
  File "/Users/jizhi/Desktop/Paddle/models/fluid/PaddleNLP/neural_machine_translation/rnn_search/attention_model.py", line 59, in bi_lstm_encoder
    size=gate_size, use_peepholes=False)
  File "/Users/jizhi/anaconda3/envs/paddle/lib/python3.5/site-packages/paddle/fluid/layers/nn.py", line 452, in dynamic_lstm
    'candidate_activation': candidate_activation
  File "/Users/jizhi/anaconda3/envs/paddle/lib/python3.5/site-packages/paddle/fluid/layer_helper.py", line 50, in append_op
    return self.main_program.current_block().append_op(*args, **kwargs)
  File "/Users/jizhi/anaconda3/envs/paddle/lib/python3.5/site-packages/paddle/fluid/framework.py", line 1207, in append_op
    op = Operator(block=self, desc=op_desc, *args, **kwargs)
  File "/Users/jizhi/anaconda3/envs/paddle/lib/python3.5/site-packages/paddle/fluid/framework.py", line 656, in __init__
    self.desc.infer_shape(self.block.desc)
paddle.fluid.core.EnforceNotMet: Enforce failed. Expected w_dims[0] == frame_size, but received w_dims[0]:128 != frame_size:512.
The first dimension of Input(Weight) should be 512. at [/Users/paddle/minqiyang/Paddle/paddle/fluid/operators/lstm_op.cc:63]
PaddlePaddle Call Stacks:
0          0x10cf18a68p paddle::platform::EnforceNotMet::EnforceNotMet(std::exception_ptr, char const*, int) + 760
1          0x10da71c6cp paddle::operators::LSTMOp::InferShape(paddle::framework::InferShapeContext*) const + 6892
2          0x10cfd5e68p paddle::framework::OpDesc::InferShape(paddle::framework::BlockDesc const&) const + 1496
3          0x10cf9f479p _ZZN8pybind1112cpp_function10initializeIZNS0_C1IvN6paddle9framework6OpDescEJRKNS4_9BlockDescEEJNS_4nameENS_9is_methodENS_7siblingEEEEMT0_KFT_DpT1_EDpRKT2_EUlPKS5_S8_E_vJSN_S8_EJS9_SA_SB_EEEvOSD_PFSC_SF_ESL_ENKUlRNS_6detail13function_callEE_clESU_ + 185
4          0x10cf233aap pybind11::cpp_function::dispatcher(_object*, _object*, _object*) + 5786
5          0x1001a959fp PyCFunction_Call + 127
6          0x1002747e7p PyEval_EvalFrameEx + 33207
7          0x10026afafp _PyEval_EvalCodeWithName + 335
8          0x1001766aap function_call + 106
9          0x100132b35p PyObject_Call + 69
10         0x100155694p method_call + 148
11         0x100132b35p PyObject_Call + 69
12         0x1001cf415p slot_tp_init + 117
13         0x1001d3ac1p type_call + 209
14         0x100132b35p PyObject_Call + 69
15         0x100271c9bp PyEval_EvalFrameEx + 22123
16         0x10026afafp _PyEval_EvalCodeWithName + 335
17         0x1001766aap function_call + 106
18         0x100132b35p PyObject_Call + 69
19         0x100271c9bp PyEval_EvalFrameEx + 22123
20         0x10026afafp _PyEval_EvalCodeWithName + 335
21         0x1002712a7p PyEval_EvalFrameEx + 19575
22         0x10026afafp _PyEval_EvalCodeWithName + 335
23         0x1002712a7p PyEval_EvalFrameEx + 19575
24         0x10026afafp _PyEval_EvalCodeWithName + 335
25         0x1002712a7p PyEval_EvalFrameEx + 19575
26         0x10026afafp _PyEval_EvalCodeWithName + 335
27         0x1002712a7p PyEval_EvalFrameEx + 19575
28         0x10026afafp _PyEval_EvalCodeWithName + 335
29         0x1002712a7p PyEval_EvalFrameEx + 19575
30         0x10026afafp _PyEval_EvalCodeWithName + 335
31         0x1002bd758p PyRun_FileExFlags + 248
32         0x1002bceeep PyRun_SimpleFileExFlags + 382
33         0x1002e1d86p Py_Main + 3622
34         0x100123861p main + 497
35      0x7fff5f4da015p start + 1
36                 0x2p
```

+ 相关代码：

```
forward, _ = fluid.layers.dynamic_lstm(
    input=input_forward_proj,
    size=gate_size * 4, use_peepholes=False)
input_reversed_proj = fluid.layers.fc(input=input_seq,
                                      size=gate_size * 4,
                                      act='tanh',
                                      bias_attr=False)
reversed, _ = fluid.layers.dynamic_lstm(
    input=input_reversed_proj,
    size=gate_size,
    is_reverse=True,
    use_peepholes=False)
return forward, reversed
```

+ 解决方法：

```
forward, _ = fluid.layers.dynamic_lstm(
            input=input_forward_proj,
            size=gate_size * 4, use_peepholes=False)
input_reversed_proj = fluid.layers.fc(input=input_seq,
                                      size=gate_size * 4,
                                      act='tanh',
                                      bias_attr=False)
reversed, _ = fluid.layers.dynamic_lstm(
    input=input_reversed_proj,
    size=gate_size * 4,
    is_reverse=True,
    use_peepholes=False)
return forward, reversed
```


##  `待审核`10.问题：op uniform_random does not have kernel for data_type

+ 问题描述：使用Fluid实现机器翻译模型时，报op uniform_random does not have kernel for data_type

+ 报错输出：

```
Traceback (most recent call last):
  File "train.py", line 173, in <module>
    train()
  File "train.py", line 103, in train
    exe.run(framework.default_startup_program())
  File "/Users/jizhi/anaconda3/envs/paddle/lib/python3.5/site-packages/paddle/fluid/executor.py", line 470, in run
    self.executor.run(program.desc, scope, 0, True, True)
paddle.fluid.core.EnforceNotMet: op uniform_random does not have kernel for data_type[int64_t]:data_layout[ANY_LAYOUT]:place[CPUPlace]:library_type[PLAIN] at [/Users/paddle/minqiyang/Paddle/paddle/fluid/framework/operator.cc:726]
PaddlePaddle Call Stacks:
0          0x1197e4a68p paddle::platform::EnforceNotMet::EnforceNotMet(std::exception_ptr, char const*, int) + 760
1          0x11a60d558p paddle::framework::OperatorWithKernel::RunImpl(paddle::framework::Scope const&, boost::variant<paddle::platform::CUDAPlace, paddle::platform::CPUPlace, paddle::platform::CUDAPinnedPlace, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_> const&) const + 1480
2          0x11a609141p paddle::framework::OperatorBase::Run(paddle::framework::Scope const&, boost::variant<paddle::platform::CUDAPlace, paddle::platform::CPUPlace, paddle::platform::CUDAPinnedPlace, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_> const&) + 577
3          0x1198b23a6p paddle::framework::Executor::RunPreparedContext(paddle::framework::ExecutorPrepareContext*, paddle::framework::Scope*, bool, bool, bool) + 390
4          0x1198b1dd3p paddle::framework::Executor::Run(paddle::framework::ProgramDesc const&, paddle::framework::Scope*, int, bool, bool) + 163
5          0x119818837p void pybind11::cpp_function::initialize<paddle::pybind::pybind11_init()::$_64, void, paddle::framework::Executor&, paddle::framework::ProgramDesc const&, paddle::framework::Scope*, int, bool, bool, pybind11::name, pybind11::is_method, pybind11::sibling>(paddle::pybind::pybind11_init()::$_64&&, void (*)(paddle::framework::Executor&, paddle::framework::ProgramDesc const&, paddle::framework::Scope*, int, bool, bool), pybind11::name const&, pybind11::is_method const&, pybind11::sibling const&)::'lambda'(pybind11::detail::function_call&)::__invoke(pybind11::detail::function_call&) + 135
6          0x1197ef3aap pybind11::cpp_function::dispatcher(_object*, _object*, _object*) + 5786
7          0x10ca8859fp PyCFunction_Call + 127
8          0x10cb537e7p PyEval_EvalFrameEx + 33207
9          0x10cb49fafp _PyEval_EvalCodeWithName + 335
10         0x10cb502a7p PyEval_EvalFrameEx + 19575
11         0x10cb49fafp _PyEval_EvalCodeWithName + 335
12         0x10cb502a7p PyEval_EvalFrameEx + 19575
13         0x10cb49fafp _PyEval_EvalCodeWithName + 335
14         0x10cb9c758p PyRun_FileExFlags + 248
15         0x10cb9beeep PyRun_SimpleFileExFlags + 382
16         0x10cbc0d86p Py_Main + 3622
17         0x10ca02861p main + 497
18      0x7fff5f4da015p start + 1
19                 0x2p
```


+ 相关代码：

```
src_word_idx = fluid.layers.data(
        name='source_sequence', shape=[1], dtype='int64', lod_level=1)
        
src_embedding = fluid.layers.embedding(
input=src_word_idx,
size=[source_dict_dim, embedding_dim],
dtype='int64')
```

+ 解决方法：

```
src_word_idx = fluid.layers.data(
        name='source_sequence', shape=[1], dtype='int64', lod_level=1)

src_embedding = fluid.layers.embedding(
    input=src_word_idx,
    size=[source_dict_dim, embedding_dim],
    dtype='float32')
```








