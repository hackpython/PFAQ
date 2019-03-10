# 个性化推荐

## 背景介绍

在网络技术不断发展和电子商务规模不断扩大的背景下，商品数量和种类快速增长，用户需要花费大量时间才能找到自己想买的商品，这就是信息超载问题。为了解决这个难题，推荐系统（Recommender System）应运而生。

个性化推荐系统是信息过滤系统（Information Filtering System）的子集，它可以用在很多领域，如电影、音乐、电商和 Feed 流推荐等。推荐系统通过分析、挖掘用户行为，发现用户的个性化需求与兴趣特点，将用户可能感兴趣的信息或商品推荐给用户。与搜索引擎不同，推荐系统不需要用户准确地描述出自己的需求，而是根据分析历史行为建模，主动提供满足用户兴趣和需求的信息


深度学习具有优秀的自动提取特征的能力，能够学习多层次的抽象特征表示，并对异质或跨域的内容信息进行学习，可以一定程度上处理推荐系统冷启动问题。

## 个性化推进系统PaddlePaddle-fluid版代码：
https://github.com/PaddlePaddle/book/tree/develop/05.recommender_system


## `已审阅` 1.问题：使用ml-1m数据集数据在训练时出现非法指令

 + 版本号：`1.1.0`

+ 标签：`非法指令`，`embedding`

 + 问题描述：使用ml-1m数据集训练个性化推荐模型，使用ml-1m数据集中的数据转换成词向量，然后训练时出现非法指令的错误。

 + 报错信息：

```
<ipython-input-8-71a7f986f7ba> in train(use_cuda, train_program, params_dirname)
     39         event_handler=event_handler,
     40         reader=train_reader,
---> 41         feed_order=feed_order)

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
    510                                       fetch_list=[
    511                                           var.name
--> 512                                           for var in self.train_func_outputs
    513                                       ])
    514                 else:

/opt/conda/envs/py35-paddle1.0.0/lib/python3.5/site-packages/paddle/fluid/executor.py in run(self, program, feed, fetch_list, feed_var_name, fetch_var_name, scope, return_numpy, use_program_cache)
    468 
    469         self._feed_data(program, feed, feed_var_name, scope)
--> 470         self.executor.run(program.desc, scope, 0, True, True)
    471         outs = self._fetch_data(fetch_list, fetch_var_name, scope)
    472         if return_numpy:

EnforceNotMet: an illegal instruction was encountered at [/paddle/paddle/fluid/platform/device_context.cc:230]
PaddlePaddle Call Stacks: 
```

 + 问题复现：通过`paddle.dataset.movielens.max_job_id`接口获取最大的工作类型ID，然后把获得的值作为`fluid.layers.embedding`接口参数`size`的值传进去，最好使用这些词向量构建成一个网络进行训练，在训练的时候就出现非法指令错误。错误代码如下：

```python
USR_JOB_DICT_SIZE = paddle.dataset.movielens.max_job_id()
usr_job_id = layers.data(name='job_id', shape=[1], dtype="int64")
usr_job_emb = layers.embedding(
    input=usr_job_id,
    size=[USR_JOB_DICT_SIZE, 16],
    param_attr='job_table',
    is_sparse=IS_SPARSE)
usr_job_fc = layers.fc(input=usr_job_emb, size=16)
```

 + 解决问题：通过`paddle.dataset.movielens.max_job_id`的接口获得的是最大的工作类型ID，工作类型的ID是从0开始标号的，所以最大的工作类型ID再加上1才是工作类型的数量，上面的错误就是工作类别数量少了1个。正确代码如下：

```python
USR_JOB_DICT_SIZE = paddle.dataset.movielens.max_job_id() + 1
usr_job_id = layers.data(name='job_id', shape=[1], dtype="int64")
usr_job_emb = layers.embedding(
    input=usr_job_id,
    size=[USR_JOB_DICT_SIZE, 16],
    param_attr='job_table',
    is_sparse=IS_SPARSE)
usr_job_fc = layers.fc(input=usr_job_emb, size=16)
```

 + 问题拓展：不仅是以上的接口，还获取最大用户ID的接口`paddle.dataset.movielens.max_user_id`和获取最大电影ID的接口`paddle.dataset.movielens.max_movie_id`也是需要同样的操作，否则也会出现类型的问题。




## `已审阅` 2.问题：在使用ml-1m数据集训练个性化推荐模型的时候出现输入数据维度不相同错误

 + 版本号：`1.1.0`

+ 标签：`数据维度`，`余弦相似度`

 + 问题描述：通过定义一个用户提取用户特征综合模型输出大小为100和一个电影特征模型输出大小为200，来训练一个个性化推荐模型，在训练的时候，出现输入数据维度不相同错误。

 + 报错信息：

```
<ipython-input-5-7d5f9bd60bf7> in inference_program()
      3     mov_combined_features = get_mov_combined_features()
      4 
----> 5     inference = layers.cos_sim(X=usr_combined_features, Y=mov_combined_features)
      6     scale_infer = layers.scale(x=inference, scale=5.0)
      7 

/opt/conda/envs/py35-paddle1.0.0/lib/python3.5/site-packages/paddle/fluid/layers/nn.py in cos_sim(X, Y)
    947         outputs={'Out': [out],
    948                  'XNorm': [xnorm],
--> 949                  'YNorm': [ynorm]})
    950     return out
    951 

/opt/conda/envs/py35-paddle1.0.0/lib/python3.5/site-packages/paddle/fluid/layer_helper.py in append_op(self, *args, **kwargs)
     48 
     49     def append_op(self, *args, **kwargs):
---> 50         return self.main_program.current_block().append_op(*args, **kwargs)
     51 
     52     def multiple_input(self, input_param_name='input'):

/opt/conda/envs/py35-paddle1.0.0/lib/python3.5/site-packages/paddle/fluid/framework.py in append_op(self, *args, **kwargs)
   1205         """
   1206         op_desc = self.desc.append_op()
-> 1207         op = Operator(block=self, desc=op_desc, *args, **kwargs)
   1208         self.ops.append(op)
   1209         return op

/opt/conda/envs/py35-paddle1.0.0/lib/python3.5/site-packages/paddle/fluid/framework.py in __init__(***failed resolving arguments***)
    654         if self._has_kernel(type):
    655             self.desc.infer_var_type(self.block.desc)
--> 656             self.desc.infer_shape(self.block.desc)
    657 
    658     def _has_kernel(self, op_type):

EnforceNotMet: Enforce failed. Expected framework::slice_ddim(x_dims, 1, x_dims.size()) == framework::slice_ddim(y_dims, 1, y_dims.size()), but received framework::slice_ddim(x_dims, 1, x_dims.size()):100 != framework::slice_ddim(y_dims, 1, y_dims.size()):200.
All dimensions except the 1st of Input(X) and Input(Y) must be equal. at [/paddle/paddle/fluid/operators/cos_sim_op.cc:50]
PaddlePaddle Call Stacks: 
```

 + 问题复现：定义一个用户特征模型，在最后的全连接层`fluid.layers.fc`的接口中设置大小为100。在定义一个电影特征模型，在最后的全连接层`fluid.layers.fc`的接口中设置大小为200。最后对这两个模型的全连接层通过`fluid.layers.cos_sim`计算它们的余弦相似度，在训练的时候就会出现以上的问题。错误代码如下：

```python
concat_embed = layers.concat(input=[usr_fc, usr_gender_fc, usr_age_fc, usr_job_fc], axis=1)
usr_combined_features = layers.fc(input=concat_embed, size=100, act="tanh")
······
concat_embed = layers.concat(input=[mov_fc, mov_categories_hidden, mov_title_conv], axis=1)
mov_combined_features = layers.fc(input=concat_embed, size=200, act="tanh")

inference = layers.cos_sim(X=usr_combined_features, Y=mov_combined_features)
```

 + 解决问题：`fluid.layers.cos_sim`接口的参数`X`和`Y`要求必须有相同的形状，除了输入Y的第一维可以是1(与输入X不同)，而上面的错误时因为两个全连接层的大小设置不一样，导致的输出`X`和`Y`形状不相同，所以才导致错误。正确代码如下：

```python
concat_embed = layers.concat(input=[usr_fc, usr_gender_fc, usr_age_fc, usr_job_fc], axis=1)
usr_combined_features = layers.fc(input=concat_embed, size=200, act="tanh")
······
concat_embed = layers.concat(input=[mov_fc, mov_categories_hidden, mov_title_conv], axis=1)
mov_combined_features = layers.fc(input=concat_embed, size=200, act="tanh")

inference = layers.cos_sim(X=usr_combined_features, Y=mov_combined_features)
```



## `已审阅` 3.问题：在训练用的特征和电影特征之间的分数是出现张量类型错误

 + 版本号：`1.1.0`

+ 标签：`张量`，`数据类型`

 + 问题描述：在训练用的特征和电影特征之间的分数，定义的`fluid.layers.data`的数量类型为`int64`，最后在训练的是就出现张量类型错误。

 + 报错信息：

```
<ipython-input-8-71a7f986f7ba> in train(use_cuda, train_program, params_dirname)
     39         event_handler=event_handler,
     40         reader=train_reader,
---> 41         feed_order=feed_order)

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
    510                                       fetch_list=[
    511                                           var.name
--> 512                                           for var in self.train_func_outputs
    513                                       ])
    514                 else:

/opt/conda/envs/py35-paddle1.0.0/lib/python3.5/site-packages/paddle/fluid/executor.py in run(self, program, feed, fetch_list, feed_var_name, fetch_var_name, scope, return_numpy, use_program_cache)
    468 
    469         self._feed_data(program, feed, feed_var_name, scope)
--> 470         self.executor.run(program.desc, scope, 0, True, True)
    471         outs = self._fetch_data(fetch_list, fetch_var_name, scope)
    472         if return_numpy:

EnforceNotMet: Tensor holds the wrong type, it holds l at [/paddle/paddle/fluid/framework/tensor_impl.h:29]
PaddlePaddle Call Stacks: 
```

 + 问题复现：获取到预测程序之后，再通过`fluid.layers.data`接口定义一个label输入，`dtype`参数的值设置为`int64`，作为用户与电影之间的得分，然后使用这个label和预测程序创建一个损失和函数，在最后的训练时出现以上的错误。错误代码如下：

```python
def train_program():
    scale_infer = inference_program()
    label = layers.data(name='score', shape=[1], dtype='int64')
    square_cost = layers.square_error_cost(input=scale_infer, label=label)
    avg_cost = layers.mean(square_cost)
    return [avg_cost, scale_infer]
```

 + 解决问题：在数据集中，用户与电影之间的分数是整数，但是使用的是平方误差损失函数，所以输出的结果应该是浮点类型的。在定义label的时候，`fluid.layers.data`设置的类型应该是`float32`。正确代码如下：

```python
def train_program():
    scale_infer = inference_program()
    label = layers.data(name='score', shape=[1], dtype='float32')
    square_cost = layers.square_error_cost(input=scale_infer, label=label)
    avg_cost = layers.mean(square_cost)
    return [avg_cost, scale_infer]
```



## `已审阅` 4.问题：在使用ml-1m数据集训练模型是出现序列数字元素的错误

 + 版本号：`1.1.0`

+ 标签：`序列数据`

 + 问题描述：在使用ml-1m数据集训练个性化推荐模型，在执行训练的时候出现值错误，错误提示使用序列设置数组元素。

 + 报错信息：

```
<ipython-input-8-71a7f986f7ba> in train(use_cuda, train_program, params_dirname)
     39         event_handler=event_handler,
     40         reader=train_reader,
---> 41         feed_order=feed_order)

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
    196         for each_name, each_converter in six.moves.zip(self.feed_names,
    197                                                        converter):
--> 198             ret_dict[each_name] = each_converter.done()
    199         return ret_dict
    200 

/opt/conda/envs/py35-paddle1.0.0/lib/python3.5/site-packages/paddle/fluid/data_feeder.py in done(self)
     71 
     72     def done(self):
---> 73         arr = numpy.array(self.data, dtype=self.dtype)
     74         if self.shape and len(arr.shape) != len(self.shape):
     75             arr = arr.reshape(self.shape)

ValueError: setting an array element with a sequence.
```

 + 问题复现：在使用`fluid.layers.data`接口定义电影名称数据输入，`lod_level`使用默认值，然后作为词向量的输入参数。最后在训练的时候，就会出现以上的错误。错误代码如下：

```python
CATEGORY_DICT_SIZE = len(paddle.dataset.movielens.movie_categories())
category_id = layers.data(name='category_id', shape=[1], dtype='int64')
mov_categories_emb = layers.embedding(input=category_id, size=[CATEGORY_DICT_SIZE, 32], is_sparse=IS_SPARSE)
mov_categories_hidden = layers.sequence_pool(input=mov_categories_emb, pool_type="sum")

MOV_TITLE_DICT_SIZE = len(paddle.dataset.movielens.get_movie_title_dict())
mov_title_id = layers.data(name='movie_title', shape=[1], dtype='int64')
mov_title_emb = layers.embedding(input=mov_title_id, size=[MOV_TITLE_DICT_SIZE, 32], is_sparse=IS_SPARSE)
mov_title_conv = nets.sequence_conv_pool(
    input=mov_title_emb,
    num_filters=32,
    filter_size=3,
    act="tanh",
    pool_type="sum")
```

 + 解决问题：电影的标题和电影的类型都是名称类型的字符串数据，所以数据应该是一个序列数据，`fluid.layers.data`接口的`lod_level`参数应该是1，定义这个数据是一个序列数据。正确代码如下：

```python
CATEGORY_DICT_SIZE = len(paddle.dataset.movielens.movie_categories())
category_id = layers.data(name='category_id', shape=[1], dtype='int64', lod_level=1)
mov_categories_emb = layers.embedding(input=category_id, size=[CATEGORY_DICT_SIZE, 32], is_sparse=IS_SPARSE)
mov_categories_hidden = layers.sequence_pool(input=mov_categories_emb, pool_type="sum")

MOV_TITLE_DICT_SIZE = len(paddle.dataset.movielens.get_movie_title_dict())
mov_title_id = layers.data(name='movie_title', shape=[1], dtype='int64', lod_level=1)
mov_title_emb = layers.embedding(input=mov_title_id, size=[MOV_TITLE_DICT_SIZE, 32], is_sparse=IS_SPARSE)
mov_title_conv = nets.sequence_conv_pool(
    input=mov_title_emb,
    num_filters=32,
    filter_size=3,
    act="tanh",
    pool_type="sum")
```

 + 问题拓展：不仅仅是电影的名称，这个数据集中电影的类别也是字符串数据`paddle.dataset.movielens.movie_categories()`，也需要使用序列数据方式定义。




## `已审阅` 5.问题：在执行创建电影名称张量数据时出错

+ 版本号：`1.1.0`

+ 标签：`张量数据`

 + 问题描述：在创建数据用于预测时，定义一个定义的名称的张量数据时出现错误，错误提示真实的数据长度和设置参数`recursive_seq_lens`的值不相等。 

 + 报错信息：

```
<ipython-input-9-bc164656c591> in infer(use_cuda, inference_program, params_dirname)
     19     job_id = fluid.create_lod_tensor([[10]], [[1]], place)
     20     movie_id = fluid.create_lod_tensor([[783]], [[1]], place)
---> 21     category_id = fluid.create_lod_tensor([[10, 8, 9]], [[1]], place)
     22     movie_title = fluid.create_lod_tensor([[1069, 4140, 2923, 710, 988]], [[5]], place)
     23 

/opt/conda/envs/py35-paddle1.0.0/lib/python3.5/site-packages/paddle/fluid/lod_tensor.py in create_lod_tensor(data, recursive_seq_lens, place)
     74         assert [
     75             new_recursive_seq_lens
---> 76         ] == recursive_seq_lens, "data and recursive_seq_lens do not match"
     77         flattened_data = np.concatenate(data, axis=0).astype("int64")
     78         flattened_data = flattened_data.reshape([len(flattened_data), 1])

AssertionError: data and recursive_seq_lens do not match
```

 + 问题复现：根据数据字典定义一个电影名称的列表，然后使用这个类别通过`fluid.create_lod_tensor`创建一个电影名称的张量数据，在执行创建时机出现以上的错误。错误代码如下：

```python
movie_id = fluid.create_lod_tensor([[783]], [[1]], place)
category_id = fluid.create_lod_tensor([[10, 8, 9]], [[1]], place)
movie_title = fluid.create_lod_tensor([[1069, 4140, 2923, 710, 988]], [[1]], place)
```

 + 解决问题：这个定义名称有五个单词，所以`recursive_seq_lens`参数的值应该是5，而不是1。正确代码如下：

```python
movie_id = fluid.create_lod_tensor([[783]], [[1]], place)
category_id = fluid.create_lod_tensor([[10, 8, 9]], [[3]], place)
movie_title = fluid.create_lod_tensor([[1069, 4140, 2923, 710, 988]], [[5]], place)
```

 + 问题拓展：对于创建张量的PaddlePaddle还提供了`paddle.fluid.layers.create_tensor`这个接口，这个这个接口跟`fluid.create_lod_tensor`不一样的是，这个接口在创建时没有赋值，只有当执行器执行`run`函数时，通过`feed`参数执行赋值。




## `已审阅` 6.问题：在使用ml-1m数据集做分类是出现非法指令错误
　
 + 版本号：`1.1.0`

+ 标签：`非法指令`，`分类`

 + 问题描述：通过使用用户的特征和电影的特征得到余弦相似度，然后想做一个分类的任务，结果在训练的是出现非法指令错误。

 + 报错信息：

```
<ipython-input-8-476df06ac06a> in train(use_cuda, train_program, params_dirname)
     39         event_handler=event_handler,
     40         reader=train_reader,
---> 41         feed_order=feed_order)

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
    510                                       fetch_list=[
    511                                           var.name
--> 512                                           for var in self.train_func_outputs
    513                                       ])
    514                 else:

/opt/conda/envs/py35-paddle1.0.0/lib/python3.5/site-packages/paddle/fluid/executor.py in run(self, program, feed, fetch_list, feed_var_name, fetch_var_name, scope, return_numpy, use_program_cache)
    468 
    469         self._feed_data(program, feed, feed_var_name, scope)
--> 470         self.executor.run(program.desc, scope, 0, True, True)
    471         outs = self._fetch_data(fetch_list, fetch_var_name, scope)
    472         if return_numpy:

EnforceNotMet: an illegal instruction was encountered at [/paddle/paddle/fluid/platform/device_context.cc:230]
PaddlePaddle Call Stacks: 
```

 + 问题复现：通过使用交叉熵损失函数`fluid.layers.cross_entropy`来计划对于ml-1m数据集做分类，设置输入的数据类型设置为`int64`，在执行训练时，就会出现以上问题，错误如下：

```python
def train_program():
    scale_infer = inference_program()
    label = layers.data(name='score', shape=[1], dtype='int64')
    square_cost = layers.cross_entropy(input=scale_infer, label=label)
    avg_cost = layers.mean(square_cost)
    return [avg_cost, scale_infer]
```

 + 解决问题：ml-1m数据集的分类范围是1到5，而分类的label必须要从0开始标记。而个性化推荐模型计算的分数可以使用回归的方式处理，所以可以`fluid.layers.square_error_cost`接口做一个回归预测。正确代码如下：

```python
def train_program():
    scale_infer = inference_program()
    label = layers.data(name='score', shape=[1], dtype='float32')
    square_cost = layers.square_error_cost(input=scale_infer, label=label)
    avg_cost = layers.mean(square_cost)
    return [avg_cost, scale_infer]
```

 + 问题拓展：平方误差损失（squared error loss）使用预测值和真实值之间误差的平方作为样本损失，是回归问题中最为基本的损失函数。交叉熵（cross entropy） 是分类问题中使用最为广泛的损失函数。


## `已审阅` 7.问题：关于个性化推荐网络结构的一些疑问

+ 版本号：`1.1.0`

+ 标签：`个性化推荐网络`

+ 问题描述：
在推荐系统中，embedding层与fc层经常联合使用，我对embedding不是很熟悉，可以理解成embedding层的作用，基本都是把一个维度很大离散的输入，映射程固定长度的词向量，且词向量之间的距离可以表示原始输入的相似度？还有就是，每一个embedding_layer都会接一个同等长度的fc_layer，这么做的好处是什么呢？

+ 相关代码：

    ```python
    usr_emb= layers.embedding(
            input=uid,
            dtype='float32',
            size=[USE_DICT_SIZE, 32],
            param_attr='user_table',
            is_sparse=IS_SPARSE
        )

    usr_fc = layers.fc(input=usr_emb, size=32)
    ```

+ 问题解答：
问题描述中，提及了多个问题，这里分别对这几个问题进行简单的解答。

    1.embedding的理解？
    embedding概念其实在词向量中的章节有提及，在paddle 中的 embedding_layer 会学习一个实数矩阵，这个矩阵是一个大的词表，矩阵的每一列是为字典中每一个词语学习到的一个 dense 表示，通常大家会提到 distribute representation 这个概念

    distribute representation 核心大体上有两个作用：

    （1）对抗维数灾难，假如我们只使用 0，1 这样简单的离散变量，一个离散的二维向量只能编码4种信息，但是如果我们使用二维连续的实向量，却能够编码理论上无穷种信息 ，也就是说实向量空间的表达能力是远远大于离散向量空间，使得我们可以用更低的维度，编码更多的信息，来缓解计算的压力；<br>
    （2）学习好的词向量，是高维实向量空间中的一个点，于是我们可以通过实向量空间的数学特性，或者其实定义的一些数学操作，来刻画语言在语义层面的一些复杂性。

    2.为什么推荐系统中embedding层与fc层经常配合使用，即embedding层的输出通常作为fc层的输入？

    (1)fc + embedding 也不是什么非常特殊的设计，需要的是找到复杂度恰好的那个模型。
    (2)对官方文档中使用的推荐系统模型而言，cnn 处理电影名那里算是一个常规的序列模型，其它部分和接多个fc 没啥区别，都可以简单理解成得到原始数据更好的表示。

+ 问题研究：

	+ 在推荐系统这个例子中，性别，以及其它一些离散特征，永远都只会取到其中一个值，是一个one-hot 表示，这时候过embedding layer 和 one-hot 直接接一个fc 是没啥区别的，计算上完全一样。
	+ 如果你觉得不是非常好理解， 这和接了两个 fc ，第一个 fc 没有引入非线性是一样的。
	+ embedding + fc 这种配置，本身也不算是“固定的”，或者说“通用的” 就一定需要这样配的配置方法。机器学习模型设计，通常还是首先考虑尽可能地拟合住数据，在这个过程中，控制模型的复杂度。先从简单的模型开始尝试，果拟合准确率已经非常高了，够用的话，在数据没有扩大的情况下，通常可以不再考虑更复杂的模型。提高复杂度简单来说就是一个layer，一个layer 往上叠加。
	+ 推荐系统这个例子中的模型已经是复杂度较低的一个模型，你可以试试去掉fc层，训练集的拟合精度会不会降；然后再加一个fc 层，看见训练集的拟合精度会不会提升，拟合速率会不会有变化，同时，在变化的过程中，测试集上的精度会不会有所改变。

