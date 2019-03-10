# 语义角色标注

## 背景介绍

自然语言分析技术大致分为三个层面：词法分析、句法分析和语义分析。语义角色标注（Semantic Role Labeling，SRL）以句子的谓词为中心，不对句子所包含的语义信息进行深入分析，只分析句子中各成分与谓词之间的关系，即句子的谓词（Predicate）- 论元（Argument）结构，并用语义角色来描述这些结构关系，是许多自然语言理解任务（如信息抽取，篇章分析，深度问答等）的一个重要中间步骤。

## `已审核`1.问题：加载PaddlePaddle保存的二进制模型报错

+ 版本号：`1.0.1`

+ 标签：`二进制模型`

+ 问题描述：使用PaddlePaddle构建深度双向LSTM模型时，读入PaddlePaddle中保存的二进制模型报错，模型结构如下：

![](https://github.com/PaddlePaddle/book/blob/develop/07.label_semantic_roles/image/db_lstm_network.png?raw=true)

+ 报错信息：

```
---------------------------------------------------------------------------
EnforceNotMet                             Traceback (most recent call last)
<ipython-input-12-72068bc55db5> in <module>()
----> 1 main(use_cuda=False)

<ipython-input-11-07e1807c26ad> in main(use_cuda, is_local)
      6     save_dirname = "label_semantic_roles.inference.model"
      7 
----> 8     train(use_cuda, save_dirname, is_local)
      9     infer(use_cuda, save_dirname)

<ipython-input-9-d78ec2340a66> in train(use_cuda, save_dirname, is_local)
     36             staircase=True))
     37 
---> 38     sgd_optimizer.minimize(avg_cost)
     39 
     40     crf_decode = fluid.layers.crf_decoding(

~/anaconda3/envs/paddle/lib/python3.5/site-packages/paddle/fluid/optimizer.py in minimize(self, loss, startup_program, parameter_list, no_grad_set)
    257         """
    258         params_grads = append_backward(loss, parameter_list, no_grad_set,
--> 259                                        [error_clip_callback])
    260 
    261         params_grads = sorted(params_grads, key=lambda x: x[0].name)

~/anaconda3/envs/paddle/lib/python3.5/site-packages/paddle/fluid/backward.py in append_backward(loss, parameter_list, no_grad_set, callbacks)
    581 
    582     _append_backward_ops_(root_block, op_path, root_block, no_grad_dict,
--> 583                           grad_to_var, callbacks)
    584 
    585     # Because calc_gradient may be called multiple times,

~/anaconda3/envs/paddle/lib/python3.5/site-packages/paddle/fluid/backward.py in _append_backward_ops_(block, ops, target_block, no_grad_dict, grad_to_var, callbacks)
    367         # Getting op's corresponding grad_op
    368         grad_op_desc, op_grad_to_var = core.get_grad_op_desc(
--> 369             op.desc, cpt.to_text(no_grad_dict[block.idx]), grad_sub_block_list)
    370 
    371         grad_op_descs.extend(grad_op_desc)

EnforceNotMet: grad_op_maker_ should not be null
Operator GradOpMaker has not been registered. at [/Users/paddle/minqiyang/Paddle/paddle/fluid/framework/op_info.h:61]
PaddlePaddle Call Stacks: 
0          0x119048a68p paddle::platform::EnforceNotMet::EnforceNotMet(std::exception_ptr, char const*, int) + 760
1          0x119073a63p paddle::framework::OpInfo::GradOpMaker() const + 195
2          0x11906fc6fp void pybind11::cpp_function::initialize<paddle::pybind::pybind11_init()::$_42, std::__1::pair<std::__1::vector<paddle::framework::OpDesc*, std::__1::allocator<paddle::framework::OpDesc*> >, std::__1::unordered_map<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >, std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >, std::__1::hash<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> > >, std::__1::equal_to<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> > >, std::__1::allocator<std::__1::pair<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> > const, std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> > > > > >, paddle::framework::OpDesc const&, std::__1::unordered_set<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >, std::__1::hash<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> > >, std::__1::equal_to<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> > >, std::__1::allocator<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> > > > const&, std::__1::vector<paddle::framework::BlockDesc*, std::__1::allocator<paddle::framework::BlockDesc*> > const&, pybind11::name, pybind11::scope, pybind11::sibling>(paddle::pybind::pybind11_init()::$_42&&, std::__1::pair<std::__1::vector<paddle::framework::OpDesc*, std::__1::allocator<paddle::framework::OpDesc*> >, std::__1::unordered_map<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >, std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >, std::__1::hash<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> > >, std::__1::equal_to<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> > >, std::__1::allocator<std::__1::pair<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> > const, std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> > > > > > (*)(paddle::framework::OpDesc const&, std::__1::unordered_set<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >, std::__1::hash<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> > >, std::__1::equal_to<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> > >, std::__1::allocator<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> > > > const&, std::__1::vector<paddle::framework::BlockDesc*, std::__1::allocator<paddle::framework::BlockDesc*> > const&), pybind11::name const&, pybind11::scope const&, pybind11::sibling const&)::'lambda'(pybind11::detail::function_call&)::__invoke(pybind11::detail::function_call&) + 335
3          0x1190533aap pybind11::cpp_function::dispatcher(_object*, _object*, _object*) + 5786
4          0x10b34a59fp PyCFunction_Call + 127
5          0x10b411ec7p PyEval_EvalFrameEx + 18583
6          0x10b40bfafp _PyEval_EvalCodeWithName + 335
7          0x10b4122a7p PyEval_EvalFrameEx + 19575
8          0x10b40bfafp _PyEval_EvalCodeWithName + 335
9          0x10b4122a7p PyEval_EvalFrameEx + 19575
10         0x10b40bfafp _PyEval_EvalCodeWithName + 335
11         0x10b4122a7p PyEval_EvalFrameEx + 19575
12         0x10b40bfafp _PyEval_EvalCodeWithName + 335
13         0x10b4122a7p PyEval_EvalFrameEx + 19575
14         0x10b40bfafp _PyEval_EvalCodeWithName + 335
15         0x10b4122a7p PyEval_EvalFrameEx + 19575
16         0x10b40bfafp _PyEval_EvalCodeWithName + 335
17         0x10b4063b5p builtin_exec + 341
18         0x10b34a55ep PyCFunction_Call + 62
19         0x10b411ec7p PyEval_EvalFrameEx + 18583
20         0x10b40bfafp _PyEval_EvalCodeWithName + 335
21         0x10b4122a7p PyEval_EvalFrameEx + 19575
22         0x10b40bfafp _PyEval_EvalCodeWithName + 335
23         0x10b4122a7p PyEval_EvalFrameEx + 19575
24         0x10b40bfafp _PyEval_EvalCodeWithName + 335
25         0x10b4122a7p PyEval_EvalFrameEx + 19575
26         0x10b40bfafp _PyEval_EvalCodeWithName + 335
27         0x10b3176aap function_call + 106
28         0x10b2d3b35p PyObject_Call + 69
29         0x10b412c9bp PyEval_EvalFrameEx + 22123
30         0x10b40bfafp _PyEval_EvalCodeWithName + 335
31         0x10b4122a7p PyEval_EvalFrameEx + 19575
32         0x10b40bfafp _PyEval_EvalCodeWithName + 335
33         0x10b4122a7p PyEval_EvalFrameEx + 19575
34         0x10b411fb8p PyEval_EvalFrameEx + 18824
35         0x10b411fb8p PyEval_EvalFrameEx + 18824
36         0x10b40bfafp _PyEval_EvalCodeWithName + 335
37         0x10b3176aap function_call + 106
38         0x10b2d3b35p PyObject_Call + 69
39         0x10b412c9bp PyEval_EvalFrameEx + 22123
40         0x10b40bfafp _PyEval_EvalCodeWithName + 335
41         0x10b3176aap function_call + 106
42         0x10b2d3b35p PyObject_Call + 69
43         0x10b412c9bp PyEval_EvalFrameEx + 22123
44         0x10b40bfafp _PyEval_EvalCodeWithName + 335
45         0x10b4122a7p PyEval_EvalFrameEx + 19575
46         0x10b411fb8p PyEval_EvalFrameEx + 18824
47         0x10b40bfafp _PyEval_EvalCodeWithName + 335
48         0x10b3176aap function_call + 106
49         0x10b2d3b35p PyObject_Call + 69
50         0x10b412c9bp PyEval_EvalFrameEx + 22123
51         0x10b40bfafp _PyEval_EvalCodeWithName + 335
52         0x10b4122a7p PyEval_EvalFrameEx + 19575
53         0x10b40bfafp _PyEval_EvalCodeWithName + 335
54         0x10b3176aap function_call + 106
55         0x10b2d3b35p PyObject_Call + 69
56         0x10b412c9bp PyEval_EvalFrameEx + 22123
57         0x10b411fb8p PyEval_EvalFrameEx + 18824
58         0x10b411fb8p PyEval_EvalFrameEx + 18824
59         0x10b411fb8p PyEval_EvalFrameEx + 18824
60         0x10b411fb8p PyEval_EvalFrameEx + 18824
61         0x10b411fb8p PyEval_EvalFrameEx + 18824
62         0x10b40bfafp _PyEval_EvalCodeWithName + 335
63         0x10b4122a7p PyEval_EvalFrameEx + 19575
64         0x10b40bfafp _PyEval_EvalCodeWithName + 335
65         0x10b4063b5p builtin_exec + 341
66         0x10b34a55ep PyCFunction_Call + 62
67         0x10b411ec7p PyEval_EvalFrameEx + 18583
68         0x10b40bfafp _PyEval_EvalCodeWithName + 335
69         0x10b4122a7p PyEval_EvalFrameEx + 19575
70         0x10b40bfafp _PyEval_EvalCodeWithName + 335
71         0x10b3176aap function_call + 106
72         0x10b2d3b35p PyObject_Call + 69
73         0x10b483906p RunModule + 182
74         0x10b482b03p Py_Main + 2979
75         0x10b2c4861p main + 497
76      0x7fff5dffe015p start + 1
77                 0x5p
```

+ 问题复现：

```
def load_parameter(file_name, h, w):
    with open(file_name, 'rb') as f:
        f.read()
        return np.fromfile(f, dtype=np.float32).reshape(h,w)
```

+ 问题解决：

不能直接以二进制的形式读入PaddlePaddle二进制文件的所有问题，因为PaddlePaddle二进制内容中有着头部信息，这些信息在加载模型时是不需要的，如果将头部信息也一同读入，那么训练模型时就会因为读入的数据信息有问题而导致报错。

解决该问题非常简单，将头部信息跳过则可，如下：

```
def load_parameter(file_name, h, w):
    with open(file_name, 'rb') as f:
        f.read(16)
        return np.fromfile(f, dtype=np.float32).reshape(h,w)
```

+ 问题拓展：

很多框架都提供导入预训练模型的特性，这种特性在大型模型训练上异常常见，其实不用想的过于神秘，所谓读入二进制的模型其实就是模型模型中的节点名与对应的参数读入，这个并不难理解，因为所谓的训练其实就是通过一系列线性与非线性的计算，让节点拥有不同的值，从而让整个模型的参数可以构成一个复杂的函数，用来实现我们的目标，而预加载二进制模型就是将此前训练好的函数参数读入，直接写到这次要训练的模型中，此时如果读入的数据的结构与当前训练的模型结构不一致就会出现各种问题。

我们很难确保此前存入二进制文件的结构，所以为了确保可以正常读入并使用这些二进制文件的参数数据，使用同一版本的框架，训练同一类似的模型则可，当然在迁移训练等具体任务，要使用不同的策略。


## `已审核`2.问题：Data Type mismatch

+ 版本号：`1.0.1`

+ 标签：`Data Type mismatch`

+ 问题描述：使用PaddlePaddle训练语义模型时，出现Data Type mismatch: 5 to 6

+ 报错信息：

```
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
<ipython-input-8-72068bc55db5> in <module>()
----> 1 main(use_cuda=False)

<ipython-input-7-07e1807c26ad> in main(use_cuda, is_local)
      6     save_dirname = "label_semantic_roles.inference.model"
      7 
----> 8     train(use_cuda, save_dirname, is_local)
      9     infer(use_cuda, save_dirname)

<ipython-input-5-1640010d9832> in train(use_cuda, save_dirname, is_local)
     19 
     20     # define network topology
---> 21     feature_out = db_lstm(**locals())
     22     target = fluid.layers.data(
     23         name='target', shape=[1], dtype='int64', lod_level=1)

<ipython-input-4-32d8dbfd3d35> in db_lstm(word, predicate, ctx_n2, ctx_n1, ctx_0, ctx_p1, ctx_p2, mark, **ignored)
     34     ]
     35 
---> 36     hidden_0 = fluid.layers.sums(input=hidden_0_layers)
     37 
     38     lstm_0 = fluid.layers.dynamic_lstm(

~/anaconda3/envs/paddle/lib/python3.5/site-packages/paddle/fluid/layers/tensor.py in sums(input, out)
    223     if out is None:
    224         out = helper.create_variable_for_type_inference(
--> 225             dtype=helper.input_dtype())
    226     helper.append_op(
    227         type='sum',

~/anaconda3/envs/paddle/lib/python3.5/site-packages/paddle/fluid/layer_helper.py in input_dtype(self, input_param_name)
    107             elif dtype != each.dtype:
    108                 raise ValueError("Data Type mismatch: %d to %d" %
--> 109                                  (dtype, each.dtype))
    110         return dtype
    111 

ValueError: Data Type mismatch: 5 to 6
```

+ 问题复现：

```
    predicate_embedding = fluid.layers.embedding(
        input=predicate,
        size=[pred_dict_len, word_dim],
        dtype='float64',
        is_sparse=IS_SPARSE,
        param_attr = 'vemb'
    )
    
    mark_embedding = fluid.layers.embedding(
        input=mark,
        size=[mark_dict_len, mark_dim],
        dtype='float64',
        is_sparse=IS_SPARSE)
```

+ 问题分析：从报错输出就可以很容易的知道问题出在数据类型不匹配上，通常的解决方法就是打印一下当下使用的训练数据的类型以及形状，再使用PaddlePaddle中与之相匹配的数据类型

+ 问题解决：

修改数据输入层的数据类型

```
    predicate_embedding = fluid.layers.embedding(
        input=predicate,
        size=[pred_dict_len, word_dim],
        dtype='float32',
        is_sparse=IS_SPARSE,
        param_attr = 'vemb'
    )
    
    mark_embedding = fluid.layers.embedding(
        input=mark,
        size=[mark_dict_len, mark_dim],
        dtype='float32',
        is_sparse=IS_SPARSE)
```

+ 问题拓展：

PaddlePaddle Fluid目前支持的数据类型包括:

float16： 部分操作支持
float32: 主要实数类型
float64: 次要实数类型，支持大部分操作
int32: 次要标签类型
int64: 主要标签类型
uint64: 次要标签类型
bool: 控制流数据类型
int16: 次要标签类型
uint8: 输入数据类型，可用于图像像素

## `已审核`3.问题：使用PaddlePaddle训练模型时，程序没有报错也没有任何输出

+ 版本号：`1.0.1`

+ 标签：`模型训练`

+ 问题藐视：使用PaddlePaddle训练模型时，程序没有报错也没有任何输出

+ 问题复现：

```
main(use_cuda=True)
```

+ 问题分析：通常程序没有报错也没有输出说明模型的整个结构是没有问题的，但无法运行，通常的原因就是机器上没有对应的设备运行程序的代码，比如GPU没有安装对应的驱动，或者在PaddlePaddle中指定使用GPU但却安装的事CPU版本的PaddlePaddle都会出现这种现象

+ 问题解决：

不使用GPU训练则可。

```
main(use_cuda=False)
```

+ 问题拓展：

CPU和GPU在训练模型速度方面有较大差距，本质原因就是CPU并不擅长高精度的浮点运算，而GPU却擅长，所有可以使用GPU的情况下尽量使用GPU来训练模型。要通过PaddlePaddle正常使用设备中的GPU，需要检查GPU是否安装正确的驱动以及是否安装GPU版本的PaddlePaddle，这部分的内容请参考文档安装部分：http://www.paddlepaddle.org/documentation/docs/zh/1.1/beginners_guide/index.html

## `已审核`4.问题：使用PaddlePaddle训练模型时，运行到一半异常抛出

+ 版本号：`1.0.1`

+ 标签：`模型训练`

+ 问题描述：使用PaddlePaddle构建双向循环神经网络做语义角色标注时，程序一开始正常运行，一会后就抛出`var verb_data not in this block`，程序一开始都正常运行了，为何还会抛出错误？

+ 报错输出：

```
avg_cost:[137.97392]
avg_cost:[67.44944]
second per batch: 0.27469959259033205
avg_cost:[84.60524]
second per batch: 0.22026275396347045
avg_cost:[71.06846]
second per batch: 0.20097293059031168
avg_cost:[77.449486]
second per batch: 0.18832539916038513
avg_cost:[72.48873]
second per batch: 0.18232287883758544
avg_cost:[72.0446]
second per batch: 0.17702320019404094
avg_cost:[91.9725]
second per batch: 0.17438508442470005
avg_cost:[58.212975]
second per batch: 0.17232521176338195
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
<ipython-input-8-72068bc55db5> in <module>()
----> 1 main(use_cuda=False)

<ipython-input-7-07e1807c26ad> in main(use_cuda, is_local)
      6     save_dirname = "label_semantic_roles.inference.model"
      7 
----> 8     train(use_cuda, save_dirname, is_local)
      9     infer(use_cuda, save_dirname)

<ipython-input-5-36c4af1cda8c> in train(use_cuda, save_dirname, is_local)
     89                 batch_id = batch_id + 1
     90 
---> 91     train_loop(fluid.default_main_program())

<ipython-input-5-36c4af1cda8c> in train_loop(main_program)
     84                                 'ctx_n1_data', 'ctx_0_data', 'ctx_p1_data',
     85                                 'ctx_p2_data', 'mark_data'
---> 86                             ], [feature_out], exe)
     87                         return
     88 

~/anaconda3/envs/paddle/lib/python3.5/site-packages/paddle/fluid/io.py in save_inference_model(dirname, feeded_var_names, target_vars, executor, main_program, model_filename, params_filename, export_for_deployment)
    651         fetch_var_names = [v.name for v in target_vars]
    652 
--> 653         prepend_feed_ops(main_program, feeded_var_names)
    654         append_fetch_ops(main_program, fetch_var_names)
    655 

~/anaconda3/envs/paddle/lib/python3.5/site-packages/paddle/fluid/io.py in prepend_feed_ops(inference_program, feed_target_names, feed_holder_name)
    517 
    518     for i, name in enumerate(feed_target_names):
--> 519         out = global_block.var(name)
    520         global_block._prepend_op(
    521             type='feed',

~/anaconda3/envs/paddle/lib/python3.5/site-packages/paddle/fluid/framework.py in var(self, name)
   1038         v = self.vars.get(name, None)
   1039         if v is None:
-> 1040             raise ValueError("var %s not in this block" % name)
   1041         return v
   1042 

ValueError: var verb_data not in this block
```

+ 问题复现：

```
word = fluid.layers.data(
    name='word_data', shape=[1], dtype='int64', lod_level=1)
predicate = fluid.layers.data(
    name='word_data', shape=[1], dtype='int64', lod_level=1)
ctx_n2 = fluid.layers.data(
    name='ctx_n2_data', shape=[1], dtype='int64', lod_level=1)
ctx_n1 = fluid.layers.data(
    name='ctx_n1_data', shape=[1], dtype='int64', lod_level=1)
ctx_0 = fluid.layers.data(
    name='ctx_0_data', shape=[1], dtype='int64', lod_level=1)
ctx_p1 = fluid.layers.data(
    name='ctx_p1_data', shape=[1], dtype='int64', lod_level=1)
ctx_p2 = fluid.layers.data(
    name='ctx_p2_data', shape=[1], dtype='int64', lod_level=1)
mark = fluid.layers.data(
    name='mark_data', shape=[1], dtype='int64', lod_level=1)
```

+ 问题分析：

在PaddlePaddle中，编写的模型并不是传统的Python程序，即可以执行运行的，PaddlePaddle与其他一些知名框架一样，使用Python语言来描述整个模型的结构，但此时并没有数据传入，即此时只是一个骨骼，只有当数据传入了，节点之间的相互关系才会被调用，此时如果模型中一开始的节点是正常的，但后半部分的某个节点有问题，就会出现运行到一半就报错的情况，从报错信息	`var verb_data not in this block`看，就是没有名为verb_data的节点，但却在逻辑中使用了它，所以报错了

+ 问题解决：

将重复的word_data修改为verb_data。

```
word = fluid.layers.data(
        name='word_data', shape=[1], dtype='int64', lod_level=1)
predicate = fluid.layers.data(
    name='verb_data', shape=[1], dtype='int64', lod_level=1)
ctx_n2 = fluid.layers.data(
    name='ctx_n2_data', shape=[1], dtype='int64', lod_level=1)
ctx_n1 = fluid.layers.data(
    name='ctx_n1_data', shape=[1], dtype='int64', lod_level=1)
ctx_0 = fluid.layers.data(
    name='ctx_0_data', shape=[1], dtype='int64', lod_level=1)
ctx_p1 = fluid.layers.data(
    name='ctx_p1_data', shape=[1], dtype='int64', lod_level=1)
ctx_p2 = fluid.layers.data(
    name='ctx_p2_data', shape=[1], dtype='int64', lod_level=1)
mark = fluid.layers.data(
    name='mark_data', shape=[1], dtype='int64', lod_level=1)
```

+ 问题拓展：
在参考他人代码，模仿着写时，很容易出现这样的问题，因为代码并不是自己重头写的，而PaddlePaddle的代码又不想传统的python代码那样可以断点调试，此时想要在运行前直接看出自己代码中的细节错误是比较困难的，此时先让程序运行，通过报错输出来定位代码中问题所在却是一个好方法。


## `已审核`5.问题： grad_op_maker_ should not be null

+ 版本号：`1.0.1`

+ 标签：`null` `grad_op_maker`

+ 问题描述：使用PaddlePaddle构建语义角色标注模型时，出现`grad_op_maker_ should not be null`

+ 报错信息：

```
---------------------------------------------------------------------------
EnforceNotMet                             Traceback (most recent call last)
<ipython-input-11-72068bc55db5> in <module>()
----> 1 main(use_cuda=False)

<ipython-input-10-07e1807c26ad> in main(use_cuda, is_local)
      6     save_dirname = "label_semantic_roles.inference.model"
      7 
----> 8     train(use_cuda, save_dirname, is_local)
      9     infer(use_cuda, save_dirname)

<ipython-input-5-f1a3bdbedafc> in train(use_cuda, save_dirname, is_local)
     36             staircase=True))
     37 
---> 38     sgd_optimizer.minimize(avg_cost)
     39 
     40     crf_decode = fluid.layers.crf_decoding(

~/anaconda3/envs/paddle/lib/python3.5/site-packages/paddle/fluid/optimizer.py in minimize(self, loss, startup_program, parameter_list, no_grad_set)
    257         """
    258         params_grads = append_backward(loss, parameter_list, no_grad_set,
--> 259                                        [error_clip_callback])
    260 
    261         params_grads = sorted(params_grads, key=lambda x: x[0].name)

~/anaconda3/envs/paddle/lib/python3.5/site-packages/paddle/fluid/backward.py in append_backward(loss, parameter_list, no_grad_set, callbacks)
    581 
    582     _append_backward_ops_(root_block, op_path, root_block, no_grad_dict,
--> 583                           grad_to_var, callbacks)
    584 
    585     # Because calc_gradient may be called multiple times,

~/anaconda3/envs/paddle/lib/python3.5/site-packages/paddle/fluid/backward.py in _append_backward_ops_(block, ops, target_block, no_grad_dict, grad_to_var, callbacks)
    367         # Getting op's corresponding grad_op
    368         grad_op_desc, op_grad_to_var = core.get_grad_op_desc(
--> 369             op.desc, cpt.to_text(no_grad_dict[block.idx]), grad_sub_block_list)
    370 
    371         grad_op_descs.extend(grad_op_desc)

EnforceNotMet: grad_op_maker_ should not be null
Operator GradOpMaker has not been registered. at [/Users/paddle/minqiyang/Paddle/paddle/fluid/framework/op_info.h:61]
PaddlePaddle Call Stacks: 
0          0x114bc8a68p paddle::platform::EnforceNotMet::EnforceNotMet(std::exception_ptr, char const*, int) + 760
1          0x114bf3a63p paddle::framework::OpInfo::GradOpMaker() const + 195
2          0x114befc6fp void pybind11::cpp_function::initialize<paddle::pybind::pybind11_init()::$_42, std::__1::pair<std::__1::vector<paddle::framework::OpDesc*, std::__1::allocator<paddle::framework::OpDesc*> >, std::__1::unordered_map<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >, std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >, std::__1::hash<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> > >, std::__1::equal_to<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> > >, std::__1::allocator<std::__1::pair<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> > const, std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> > > > > >, paddle::framework::OpDesc const&, std::__1::unordered_set<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >, std::__1::hash<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> > >, std::__1::equal_to<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> > >, std::__1::allocator<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> > > > const&, std::__1::vector<paddle::framework::BlockDesc*, std::__1::allocator<paddle::framework::BlockDesc*> > const&, pybind11::name, pybind11::scope, pybind11::sibling>(paddle::pybind::pybind11_init()::$_42&&, std::__1::pair<std::__1::vector<paddle::framework::OpDesc*, std::__1::allocator<paddle::framework::OpDesc*> >, std::__1::unordered_map<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >, std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >, std::__1::hash<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> > >, std::__1::equal_to<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> > >, std::__1::allocator<std::__1::pair<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> > const, std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> > > > > > (*)(paddle::framework::OpDesc const&, std::__1::unordered_set<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >, std::__1::hash<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> > >, std::__1::equal_to<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> > >, std::__1::allocator<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> > > > const&, std::__1::vector<paddle::framework::BlockDesc*, std::__1::allocator<paddle::framework::BlockDesc*> > const&), pybind11::name const&, pybind11::scope const&, pybind11::sibling const&)::'lambda'(pybind11::detail::function_call&)::__invoke(pybind11::detail::function_call&) + 335
3          0x114bd33aap pybind11::cpp_function::dispatcher(_object*, _object*, _object*) + 5786
4          0x106ebf59fp PyCFunction_Call + 127
5          0x106f86ec7p PyEval_EvalFrameEx + 18583
6          0x106f80fafp _PyEval_EvalCodeWithName + 335
7          0x106f872a7p PyEval_EvalFrameEx + 19575
8          0x106f80fafp _PyEval_EvalCodeWithName + 335
9          0x106f872a7p PyEval_EvalFrameEx + 19575
10         0x106f80fafp _PyEval_EvalCodeWithName + 335
11         0x106f872a7p PyEval_EvalFrameEx + 19575
12         0x106f80fafp _PyEval_EvalCodeWithName + 335
13         0x106f872a7p PyEval_EvalFrameEx + 19575
14         0x106f80fafp _PyEval_EvalCodeWithName + 335
15         0x106f872a7p PyEval_EvalFrameEx + 19575
16         0x106f80fafp _PyEval_EvalCodeWithName + 335
17         0x106f7b3b5p builtin_exec + 341
18         0x106ebf55ep PyCFunction_Call + 62
19         0x106f86ec7p PyEval_EvalFrameEx + 18583
20         0x106f80fafp _PyEval_EvalCodeWithName + 335
21         0x106f872a7p PyEval_EvalFrameEx + 19575
22         0x106f80fafp _PyEval_EvalCodeWithName + 335
23         0x106f872a7p PyEval_EvalFrameEx + 19575
24         0x106f80fafp _PyEval_EvalCodeWithName + 335
25         0x106f872a7p PyEval_EvalFrameEx + 19575
26         0x106f80fafp _PyEval_EvalCodeWithName + 335
27         0x106e8c6aap function_call + 106
28         0x106e48b35p PyObject_Call + 69
29         0x106f87c9bp PyEval_EvalFrameEx + 22123
30         0x106f80fafp _PyEval_EvalCodeWithName + 335
31         0x106f872a7p PyEval_EvalFrameEx + 19575
32         0x106f80fafp _PyEval_EvalCodeWithName + 335
33         0x106f872a7p PyEval_EvalFrameEx + 19575
34         0x106f86fb8p PyEval_EvalFrameEx + 18824
35         0x106f86fb8p PyEval_EvalFrameEx + 18824
36         0x106f80fafp _PyEval_EvalCodeWithName + 335
37         0x106e8c6aap function_call + 106
38         0x106e48b35p PyObject_Call + 69
39         0x106f87c9bp PyEval_EvalFrameEx + 22123
40         0x106f80fafp _PyEval_EvalCodeWithName + 335
41         0x106e8c6aap function_call + 106
42         0x106e48b35p PyObject_Call + 69
43         0x106f87c9bp PyEval_EvalFrameEx + 22123
44         0x106f80fafp _PyEval_EvalCodeWithName + 335
45         0x106f872a7p PyEval_EvalFrameEx + 19575
46         0x106f86fb8p PyEval_EvalFrameEx + 18824
47         0x106f80fafp _PyEval_EvalCodeWithName + 335
48         0x106e8c6aap function_call + 106
49         0x106e48b35p PyObject_Call + 69
50         0x106f87c9bp PyEval_EvalFrameEx + 22123
51         0x106f80fafp _PyEval_EvalCodeWithName + 335
52         0x106f872a7p PyEval_EvalFrameEx + 19575
53         0x106f80fafp _PyEval_EvalCodeWithName + 335
54         0x106e8c6aap function_call + 106
55         0x106e48b35p PyObject_Call + 69
56         0x106f87c9bp PyEval_EvalFrameEx + 22123
57         0x106f86fb8p PyEval_EvalFrameEx + 18824
58         0x106f86fb8p PyEval_EvalFrameEx + 18824
59         0x106f86fb8p PyEval_EvalFrameEx + 18824
60         0x106f86fb8p PyEval_EvalFrameEx + 18824
61         0x106f86fb8p PyEval_EvalFrameEx + 18824
62         0x106f80fafp _PyEval_EvalCodeWithName + 335
63         0x106f872a7p PyEval_EvalFrameEx + 19575
64         0x106f80fafp _PyEval_EvalCodeWithName + 335
65         0x106f7b3b5p builtin_exec + 341
66         0x106ebf55ep PyCFunction_Call + 62
67         0x106f86ec7p PyEval_EvalFrameEx + 18583
68         0x106f80fafp _PyEval_EvalCodeWithName + 335
69         0x106f872a7p PyEval_EvalFrameEx + 19575
70         0x106f80fafp _PyEval_EvalCodeWithName + 335
71         0x106e8c6aap function_call + 106
72         0x106e48b35p PyObject_Call + 69
73         0x106ff8906p RunModule + 182
74         0x106ff7b03p Py_Main + 2979
75         0x106e39861p main + 497
76      0x7fff5dffe015p start + 1
77                 0x5p
```

+ 问题复现：

```
with fluid.scope_guard(inference_scope):
		[inference_program, feed_target_names,
         fetch_targets] = fluid.io.load_inference_model(save_dirname, exe)
        lod = [[24]]
        base_shape = [1]
        word = fluid.create_random_int_lodtensor(
            lod, base_shape, place, low=0, high=word_dict_len - 1)
```

+ 问题分析：

从报错信息以及复现代码来看，该错误的出现的主要原因就是没有理解PaddlePaddle中LoDTensor的概率，LoDTensor是PaddlePaddle支持的特殊数据类型，主要用于处理序列数据，如句子或文章段落数据等，有了LoDTensor后，处理序列性数据就非常简单。它需要用户: 1. 传入一个mini-batch需要被训练的所有数据; 2.每个序列的长度信息。 用户可以使用 fluid.create_lod_tensor 来创建 LoDTensor。

传入序列信息的时候，需要设置序列嵌套深度，lod_level。 例如训练数据是词汇组成的句子，lod_level=1；训练数据是 词汇先组成了句子， 句子再组成了段落，那么 lod_level=2。实例代码如下：

```
sentence = fluid.layers.data(name="sentence", dtype="int64", shape=[1], lod_level=1)

...

exe.run(feed={
  "sentence": create_lod_tensor(
    data=numpy.array([1, 3, 4, 5, 3, 6, 8], dtype='int64').reshape(-1, 1),
    lod=[4, 1, 2],
    place=fluid.CPUPlace()
  )
})
```

+ 问题解决：

理解了LoDTensor，修改该问题就就比较明了了，如下：

```
with fluid.scope_guard(inference_scope):
		[inference_program, feed_target_names,
         fetch_targets] = fluid.io.load_inference_model(save_dirname, exe)
        lod = [[3, 4, 2]]
        base_shape = [1]
        word = fluid.create_random_int_lodtensor(
            lod, base_shape, place, low=0, high=word_dict_len - 1)
```



## `已审核`6.问题：运行报错，出现`'NoneType' object has no attribute 'get_tensor'`

+ 版本号：`1.0.1`

+ 标签：`NoneType`

+ 问题描述：通过PaddlePaddle构建LSTM模型，训练时报出`'NoneType' object has no attribute 'get_tensor'`，代码中是有get_tensor了，就算将get_tensor删除，也会出现相似的错误

+ 报错信息：

```
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
<ipython-input-8-72068bc55db5> in <module>()
----> 1 main(use_cuda=False)

<ipython-input-7-07e1807c26ad> in main(use_cuda, is_local)
      6     save_dirname = "label_semantic_roles.inference.model"
      7 
----> 8     train(use_cuda, save_dirname, is_local)
      9     infer(use_cuda, save_dirname)

<ipython-input-5-d2090fc0f279> in train(use_cuda, save_dirname, is_local)
     89                 batch_id = batch_id + 1
     90 
---> 91     train_loop(fluid.default_main_program())

<ipython-input-5-d2090fc0f279> in train_loop(main_program)
     58 #         exe.run(fluid.default_startup_program())
     59         embedding_param = fluid.global_scope().find_var(
---> 60             embedding_name).get_tensor()
     61         embedding_param.set(
     62             load_parameter(conll05.get_embedding(), word_dict_len, word_dim),

AttributeError: 'NoneType' object has no attribute 'get_tensor'
```

+ 问题复现：

```
def train_loop(main_program):
        embedding_param = fluid.global_scope().find_var(
            embedding_name).get_tensor()
        embedding_param.set(
            load_parameter(conll05.get_embedding(), word_dict_len, word_dim),
            place)

        start_time = time.time()
        batch_id = 0
```

+ 问题分析：通过问题描述可知，神经网络的结构中是编写了相应的节点的，但是在运行训练模型时出现节点不存在，通常导致这种报错的原因就是没有运行fluid.default_startup_program()方法对定义的模型结构以及参数进行初始化，PaddlePaddle是设计与运行分离的一种框架，即Python只是用于设计整个结构，但运行为了保证速度依旧交由C++，而PaddlePaddle中必须使用fluid.default_startup_program()方法初始化模型中的结构与参数

+ 问题解决：

添加运行fluid.default_startup_program()方法的逻辑则可。

```
exe = fluid.Executor(place)
def train_loop(main_program):
        exe.run(fluid.default_startup_program())
        embedding_param = fluid.global_scope().find_var(
            embedding_name).get_tensor()
        embedding_param.set(
            load_parameter(conll05.get_embedding(), word_dict_len, word_dim),
            place)

        start_time = time.time()
        batch_id = 0
```


## `已审核`7.问题:程序训练完后，执行预测逻辑报错

+ 版本号：`1.0.1`

+ 标签：`预测网络`

+ 问题描述：PaddlePaddle编写完程序后，执行预测逻辑报错

+ 报错信息

```
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
<ipython-input-8-05889ab707b5> in <module>()
----> 1 infer(False, 'label_semantic_roles2.inference.model')

<ipython-input-6-355fb331d494> in infer(use_cuda, save_dirname)
     13         # we want to obtain data from using fetch operators).
     14         [inference_program, feed_target_names,
---> 15          fetch_targets] = fluid.io.load_inference_model(save_dirname, exe)
     16 
     17         # Setup inputs by creating LoDTensors to represent sequences of words.

~/anaconda3/envs/paddle/lib/python3.5/site-packages/paddle/fluid/io.py in load_inference_model(dirname, executor, model_filename, params_filename, pserver_endpoints)
    728     """
    729     if not os.path.isdir(dirname):
--> 730         raise ValueError("There is no directory named '%s'", dirname)
    731 
    732     if model_filename is not None:

ValueError: ("There is no directory named '%s'", 'label_semantic_roles2.inference.model')
```

+ 问题分析：从问题描述与报错信息可以看出，在执行预测时，找不到二进制的参数文件导致报错，出现这个问题可能是对PaddlePaddle中预测的理解有偏差，当我们编写好模型并训练后，通常会将模型训练的参数结果以二进制的形式保存，而所谓的预测其实就是将训练好的模型对应的二进制参数文件读入到原有的结构中再次输入新的数据运行获得模型的输出结构，如果此时二进制文件不存在或者二进制文件被损坏，预测就会出现错误。

+ 问题复现：

```
infer(False, 'label_semantic_roles.inference.model')
```

+ 问题解决：

进行训练时，记录好保存文件的路径，在进行预测时，将该二进制文件路径传入

```
# 模型文件路径
save_dirname = "label_semantic_roles.inference.model"
# 训练
train(use_cuda, save_dirname, is_local)

# 预测
infer(False, 'label_semantic_roles.inference.model')
```

## `已审核`8.问题：var ctx_p1_data not in this block

+ 版本号：`1.0.1`

+ 标签：`not in this block`

+ 问题描述：使用PaddlePaddle编写语义标注模型时出现`var ctx_p1_data not in this block`

+ 报错信息：

```
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
<ipython-input-8-72068bc55db5> in <module>()
----> 1 main(use_cuda=False)

<ipython-input-7-07e1807c26ad> in main(use_cuda, is_local)
      6     save_dirname = "label_semantic_roles.inference.model"
      7 
----> 8     train(use_cuda, save_dirname, is_local)
      9     infer(use_cuda, save_dirname)

<ipython-input-5-b0e2b3d92a50> in train(use_cuda, save_dirname, is_local)
     89                 batch_id = batch_id + 1
     90 
---> 91     train_loop(fluid.default_startup_program())

<ipython-input-5-b0e2b3d92a50> in train_loop(main_program)
     68             for data in train_data():
     69                 cost = exe.run(
---> 70                     main_program, feed=feeder.feed(data), fetch_list=[avg_cost])
     71                 cost = cost[0]
     72 

~/anaconda3/envs/paddle/lib/python3.5/site-packages/paddle/fluid/executor.py in run(self, program, feed, fetch_list, feed_var_name, fetch_var_name, scope, return_numpy, use_program_cache)
    465                 fetch_list=fetch_list,
    466                 feed_var_name=feed_var_name,
--> 467                 fetch_var_name=fetch_var_name)
    468 
    469         self._feed_data(program, feed, feed_var_name, scope)

~/anaconda3/envs/paddle/lib/python3.5/site-packages/paddle/fluid/executor.py in _add_feed_fetch_ops(self, program, feed, fetch_list, feed_var_name, fetch_var_name)
    313         if not has_feed_operators(global_block, feed, feed_var_name):
    314             for i, name in enumerate(feed):
--> 315                 out = global_block.var(name)
    316                 global_block._prepend_op(
    317                     type='feed',

~/anaconda3/envs/paddle/lib/python3.5/site-packages/paddle/fluid/framework.py in var(self, name)
   1038         v = self.vars.get(name, None)
   1039         if v is None:
-> 1040             raise ValueError("var %s not in this block" % name)
   1041         return v
   1042 

ValueError: var ctx_p1_data not in this block
```


+ 问题复现：

```
 def train_loop(main_program):
        exe.run(fluid.default_startup_program())
        embedding_param = fluid.global_scope().find_var(
            embedding_name).get_tensor()
        embedding_param.set(
            load_parameter(conll05.get_embedding(), word_dict_len, word_dim),
            place)
        ...

train_loop(fluid.default_startup_program())
```

+ 问题分析：从报错信息来看，程序中使用了ctx_p1_data节点，但模型中却没有这个节点，造成这个问题的可能性很多，可能是你构建的模型中存在该节点，但你读入的二进制模型中没有该节点，此时就会出现这样的错误，但从提供的代码看，该错误是因为错误的出现是因为错误的使用了fluid.default_startup_program()

+ 解决方法：

将传入train_loop()方法的fluid.default_startup_program()改为fluid.default_main_program()

```
 def train_loop(main_program):
        exe.run(fluid.default_startup_program())
        embedding_param = fluid.global_scope().find_var(
            embedding_name).get_tensor()
        embedding_param.set(
            load_parameter(conll05.get_embedding(), word_dict_len, word_dim),
            place)
        ...

train_loop(fluid.default_main_program())
```


+ 问题拓展：

用户完成网络定义后，一段 Fluid 程序中通常存在 2 段 Program：

1.fluid.default_startup_program：定义了创建模型参数，输入输出，以及模型中可学习参数的初始化等各种操作default_startup_program 可以由框架自动生成，使用时无需显示地创建如果调用修改了参数的默认初始化方式，框架会自动的将相关的修改加入default_startup_program

2.fluid.default_main_program ：定义了神经网络模型，前向反向计算，以及优化算法对网络中可学习参数的更新使用Fluid的核心就是构建起 default_main_program

