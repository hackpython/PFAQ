# 情感分析

## 背景介绍
在自然语言处理中，情感分析一般是指判断一段文本所表达的情绪状态。其中，一段文本可以是一个句子，一个段落或一个文档。情绪状态可以是两类，如（正面，负面），（高兴，悲伤）；也可以是三类，如（积极，消极，中性）等等。情感分析的应用场景十分广泛，如把用户在购物网站（亚马逊、天猫、淘宝等）、旅游网站、电影评论网站上发表的评论分成正面评论和负面评论；或为了分析用户对于某一产品的整体使用感受，抓取产品的用户评论并进行情感分析等等。

## 情感分析PaddlePaddle-fluid版代码：
https://github.com/PaddlePaddle/book/tree/develop/06.understand_sentiment

## `已审核`1.问题：使用训练好的情感分析模型预测句子结果都是一样的

 + 关键字：`数据字典`，`字符编码`
 

 + 问题描述：使用循环神经网络训练一个IMDB数据集得到一个模型，使用这个模型进行预测句子，无论句子是正面还是负面的，预测的结果都是一样。


 + 报错信息：

```
[[5146, 5146, 5146, 5146, 5146, 5146], [5146, 5146, 5146, 5146, 5146], [5146, 5146, 5146, 5146]]
Predict probability of  0.54538333  to be positive and  0.45461673  to be negative for review ' read the book forget the movie '
Predict probability of  0.54523355  to be positive and  0.45476642  to be negative for review ' this is a great movie '
Predict probability of  0.54504114  to be positive and  0.45495886  to be negative for review ' this is very bad '
```

 + 问题复现：在预测是，使用`Inferencer`接口创建一个预测器，然后把句子里的每个单词转换成列表形式，然后使用`word_dict.get(words, UNK)`根据数据集的字典把单词转换成标签，然后使用这些标签进行预测，最后预测的都是错误的。错误代码如下：

```python
inferencer = Inferencer(
    infer_func=partial(inference_program, word_dict),
    param_path=params_dirname,
    place=place)
reviews_str = ['read the book forget the movie', 'this is a great movie', 'this is very bad']
reviews = [c.split() for c in reviews_str]
UNK = word_dict['<unk>']
lod = []
for c in reviews:
    lod.append([word_dict.get(words, UNK) for words in c])
print(lod)
base_shape = [[len(c) for c in lod]]
tensor_words = fluid.create_lod_tensor(lod, base_shape, place)
results = inferencer.infer({'words': tensor_words})
```

 + 解决问题：错误的原因是没使用正确的编码，所以在使用`word_dict.get(words, UNK)`转换编码时，程序理解里面都是`<unk>`，所以句子都是`<unk>`对应的编码。需要对里面的单词转换成UTF-8的字符编码，例子这样`word_dict.get(words.encode('utf-8')`。正确代码如下：

```python
inferencer = Inferencer(
    infer_func=partial(inference_program, word_dict),
    param_path=params_dirname,
    place=place)
reviews_str = ['read the book forget the movie', 'this is a great movie', 'this is very bad']
reviews = [c.split() for c in reviews_str]
UNK = word_dict['<unk>']
lod = []
for c in reviews:
    lod.append([word_dict.get(words.encode('utf-8'), UNK) for words in c])
print(lod)
base_shape = [[len(c) for c in lod]]
tensor_words = fluid.create_lod_tensor(lod, base_shape, place)
results = inferencer.infer({'words': tensor_words})
```



## `已审核`2.问题：使用句子做感情分析预测时出现结果不正确

 + 关键字：`数据字典`
 

 + 问题描述：使用3个句子进行预测，预测该句子的正面和负面的概率，在执行预测时大多数的结果都不正确，而且每个句子的编码都很长。


 + 报错信息：

```
['read the book forget the movie', 'this is a great movie', 'this is very bad']
[[2237, 4008, 2, 2062, 5146, 3602, 3752, 4008, 5146, 951, 2903, 2903, 5146, 5146, 2414, 2903, 2237, 3316, 4008, 3602, 5146, 3602, 3752, 4008, 5146, 4136, 2903, 5146, 8, 4008], [3602, 3752, 8, 2551, 5146, 8, 2551, 5146, 2, 5146, 3316, 2237, 4008, 2, 3602, 5146, 4136, 2903, 5146, 8, 4008], [3602, 3752, 8, 2551, 5146, 8, 2551, 5146, 5146, 4008, 2237, 5146, 5146, 951, 2, 2062]]
Predict probability of  0.59143597  to be positive and  0.4085641  to be negative for review ' read the book forget the movie '
Predict probability of  0.73750913  to be positive and  0.26249087  to be negative for review ' this is a great movie '
Predict probability of  0.55495805  to be positive and  0.445042  to be negative for review ' this is very bad '
```

 + 问题复现：在预测时需要把句子转换成单词列表，在把单词转换成编码。把句子转换成列表时使用`reviews = [c for c in reviews_str]`进行转换，然后使用这个结果通过数据集字典转换成编码进行预测，预测结果几乎都是错误的。错误代码如下：

```python
inferencer = Inferencer(
    infer_func=partial(inference_program, word_dict),
    param_path=params_dirname,
    place=place)
reviews_str = ['read the book forget the movie', 'this is a great movie', 'this is very bad']
reviews = [c for c in reviews_str]
print(reviews)
UNK = word_dict['<unk>']
lod = []
for c in reviews:
    lod.append([word_dict.get(words.encode('utf-8'), UNK) for words in c])
print(lod)
base_shape = [[len(c) for c in lod]]
tensor_words = fluid.create_lod_tensor(lod, base_shape, place)
results = inferencer.infer({'words': tensor_words})
```

 + 解决问题：上面错误的原因是数据预处理时，没有正确把句子中的单词拆开，导致在使用数据字典把字符串转换成编码的时候，使用的是句子的字符，所以导致错误出现。在处理的时候应该是`reviews = [c.split() for c in reviews_str]`。正确代码如下：

```python
inferencer = Inferencer(
    infer_func=partial(inference_program, word_dict),
    param_path=params_dirname,
    place=place)
reviews_str = ['read the book forget the movie', 'this is a great movie', 'this is very bad']
reviews = [c.split() for c in reviews_str]
print(reviews)
UNK = word_dict['<unk>']
lod = []
for c in reviews:
    lod.append([word_dict.get(words.encode('utf-8'), UNK) for words in c])
print(lod)
base_shape = [[len(c) for c in lod]]
tensor_words = fluid.create_lod_tensor(lod, base_shape, place)
results = inferencer.infer({'words': tensor_words})
```
正确的输出情况：
```
[['read', 'the', 'book', 'forget', 'the', 'movie'], ['this', 'is', 'a', 'great', 'movie'], ['this', 'is', 'very', 'bad']]
[[325, 0, 276, 818, 0, 16], [9, 5, 2, 78, 16], [9, 5, 51, 81]]
Predict probability of  0.44390476  to be positive and  0.55609524  to be negative for review ' read the book forget the movie '
Predict probability of  0.83933955  to be positive and  0.16066049  to be negative for review ' this is a great movie '
Predict probability of  0.35688713  to be positive and  0.64311296  to be negative for review ' this is very bad '
```



## `已审核`3.问题：在使用情感分析模型预测句子是出现数量类型错误

 + 关键字：`数据字典`，`自定义`
 

 + 问题描述：通过自己写一个句子，使用训练好的模型进行预测。在使用`fluid.create_lod_tensor`接口准备把数据转换成张量数据进行预测时，出现数据类型错误。


 + 报错信息：

```
[['paddlepaddle', 'from', 'baidu'], ['this', 'is', 'a', 'great', 'movie'], ['this', 'is', 'very', 'bad', 'fack']]
[[None, 34, None], [9, 5, 2, 78, 16], [9, 5, 51, 81, None]]
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-22-9af2e9ab36a2> in <module>
      9 if __name__ == '__main__':
     10     use_cuda = True  # set to True if training with GPU
---> 11     main(use_cuda)

<ipython-input-22-9af2e9ab36a2> in main(use_cuda)
      4     params_dirname = "understand_sentiment_stacked_lstm.inference.model"
      5 #     train(use_cuda, train_program, params_dirname)
----> 6     infer(use_cuda, inference_program, params_dirname)
      7 
      8 

<ipython-input-21-4bf435dffc3f> in infer(use_cuda, inference_program, params_dirname)
     15     print(lod)
     16     base_shape = [[len(c) for c in lod]]
---> 17     tensor_words = fluid.create_lod_tensor(lod, base_shape, place)
     18     results = inferencer.infer({'words': tensor_words})
     19 

/opt/conda/envs/py35-paddle1.0.0/lib/python3.5/site-packages/paddle/fluid/lod_tensor.py in create_lod_tensor(data, recursive_seq_lens, place)
     75             new_recursive_seq_lens
     76         ] == recursive_seq_lens, "data and recursive_seq_lens do not match"
---> 77         flattened_data = np.concatenate(data, axis=0).astype("int64")
     78         flattened_data = flattened_data.reshape([len(flattened_data), 1])
     79         return create_lod_tensor(flattened_data, recursive_seq_lens, place)

TypeError: int() argument must be a string, a bytes-like object or a number, not 'NoneType'
```

 + 问题复现：通过自己定义一句话，然后使用`word_dict.get(words.encode('utf-8'))`转换成整数编码，使用这些编码创建一个张量数据的时候，就出现以上的错误。错误代码如下： 

```python
inferencer = Inferencer(
    infer_func=partial(inference_program, word_dict),
    param_path=params_dirname,
    place=place)
reviews_str = ['paddlepaddle from baidu', 'this is a great movie', 'this is very bad fack']
reviews = [c.split() for c in reviews_str]
print(reviews)
lod = []
for c in reviews:
    lod.append([word_dict.get(words.encode('utf-8')) for words in c])
print(lod)
base_shape = [[len(c) for c in lod]]
tensor_words = fluid.create_lod_tensor(lod, base_shape, place)
results = inferencer.infer({'words': tensor_words})
```

 + 解决问题：上面出现的错误是因为使用到了数据集字典中没有出现过的单词，在使用`word_dict.get(words.encode('utf-8'))`转换成整数编码时，就会出现结果为`None`的情况。如果需要使用`UNK = word_dict['<unk>']`和`word_dict.get(words.encode('utf-8'), UNK)`把未知的单词转换成同一个整数编码就不会出现上述问题。正确代码如下：

```python
inferencer = Inferencer(
    infer_func=partial(inference_program, word_dict),
    param_path=params_dirname,
    place=place)
reviews_str = ['paddlepaddle from baidu', 'this is a great movie', 'this is very bad fack']
reviews = [c.split() for c in reviews_str]
UNK = word_dict['<unk>']
print(reviews)
lod = []
for c in reviews:
    lod.append([word_dict.get(words.encode('utf-8'), UNK) for words in c])
print(lod)
base_shape = [[len(c) for c in lod]]
tensor_words = fluid.create_lod_tensor(lod, base_shape, place)
results = inferencer.infer({'words': tensor_words})
```




## `已审核`4.问题：在使用感情分析模型预测句子时出现转换成张量数据出错

 + 关键字：`数据类型`，`预测`
 

 + 问题描述：使用一个训练好的模型进行预测，出现错误，错误提示您的一些提要数据保存LoD信息。它们不能完全从Python ndarray列表转换为lod张量。在输入数据之前，请直接将数据转换为lod张量。


 + 报错信息：

```
<ipython-input-27-54d4dcebc66b> in infer(use_cuda, inference_program, params_dirname)
      9     reviews_str = ['paddlepaddle from baidu', 'this is a great movie', 'this is very bad fack']
     10     reviews = [c.split() for c in reviews_str]
---> 11     results = inferencer.infer({'words': reviews})
     12 
     13     for i, r in enumerate(results[0]):

/opt/conda/envs/py35-paddle1.0.0/lib/python3.5/site-packages/paddle/fluid/contrib/inferencer.py in infer(self, inputs, return_numpy)
    102             results = self.exe.run(feed=inputs,
    103                                    fetch_list=[self.predict_var.name],
--> 104                                    return_numpy=return_numpy)
    105 
    106         return results

/opt/conda/envs/py35-paddle1.0.0/lib/python3.5/site-packages/paddle/fluid/executor.py in run(self, program, feed, fetch_list, feed_var_name, fetch_var_name, scope, return_numpy, use_program_cache)
    467                 fetch_var_name=fetch_var_name)
    468 
--> 469         self._feed_data(program, feed, feed_var_name, scope)
    470         self.executor.run(program.desc, scope, 0, True, True)
    471         outs = self._fetch_data(fetch_list, fetch_var_name, scope)

/opt/conda/envs/py35-paddle1.0.0/lib/python3.5/site-packages/paddle/fluid/executor.py in _feed_data(self, program, feed, feed_var_name, scope)
    341                 cur_feed = feed[feed_target_name]
    342                 if not isinstance(cur_feed, core.LoDTensor):
--> 343                     cur_feed = _as_lodtensor(cur_feed, self.place)
    344                 idx = op.desc.attr('col')
    345                 core.set_feed_variable(scope, cur_feed, feed_var_name, idx)

/opt/conda/envs/py35-paddle1.0.0/lib/python3.5/site-packages/paddle/fluid/executor.py in _as_lodtensor(data, place)
    247                 ndarray to LoDTensor. Please convert data to LoDTensor \
    248                 directly before feeding the data.\
--> 249                 ")
    250     # single tensor case
    251     tensor = core.LoDTensor()

RuntimeError: Some of your feed data hold LoD information.                 They can not be completely cast from a list of Python                 ndarray to LoDTensor. Please convert data to LoDTensor                 directly before feeding the data.               
```

 + 问题复现：使用训练好的模型和网络生成要给预测器，使用这个预测器进行预测句子，预测句子之前先把句子转换成一个列表，然后使用之前创建的预测器进行预测，就会出现以上的错误，错误代码如下：

```python
inferencer = Inferencer(
    infer_func=partial(inference_program, word_dict),
    param_path=params_dirname,
    place=place)
reviews_str = ['paddlepaddle from baidu', 'this is a great movie', 'this is very bad fack']
reviews = [c.split() for c in reviews_str]
results = inferencer.infer({'words': reviews})
```

 + 解决问题：错误的原因是直接把句子作为一个列表进行预测，PaddlePaddle不支持这种字符串类型的数据输入，所以数据集提供了一个数据字典，把句子中的单词转换成整数列表。得到一个张量数据，使用这个张量数据再进行预测。正确代码如下：

```python
inferencer = Inferencer(
    infer_func=partial(inference_program, word_dict),
    param_path=params_dirname,
    place=place)
reviews_str = ['paddlepaddle from baidu', 'this is a great movie', 'this is very bad fack']
reviews = [c.split() for c in reviews_str]
UNK = word_dict['<unk>']
print(reviews)
lod = []
for c in reviews:
    lod.append([word_dict.get(words.encode('utf-8'), UNK) for words in c])
print(lod)
base_shape = [[len(c) for c in lod]]
tensor_words = fluid.create_lod_tensor(lod, base_shape, place)
results = inferencer.infer({'words': tensor_words})
```



## `已审核`5.问题：在使用训练好的模型预测句子时出现数据类型的错误

 + 关键字：`数据类型`，`张量`
 

 + 问题描述：使用一个训练好的模型进行预测，在使用数据字典把单词转换成整数编码之后，使用这谢整数列表进行预测，出现错误数据类型错误。


 + 报错信息：

```
<ipython-input-33-340d192d6a07> in infer(use_cuda, inference_program, params_dirname)
     15         lod.append([word_dict.get(words.encode('utf-8'), UNK) for words in c])
     16     print(lod)
---> 17     results = inferencer.infer({'words': lod})
     18 
     19     for i, r in enumerate(results[0]):

/opt/conda/envs/py35-paddle1.0.0/lib/python3.5/site-packages/paddle/fluid/contrib/inferencer.py in infer(self, inputs, return_numpy)
    102             results = self.exe.run(feed=inputs,
    103                                    fetch_list=[self.predict_var.name],
--> 104                                    return_numpy=return_numpy)
    105 
    106         return results

/opt/conda/envs/py35-paddle1.0.0/lib/python3.5/site-packages/paddle/fluid/executor.py in run(self, program, feed, fetch_list, feed_var_name, fetch_var_name, scope, return_numpy, use_program_cache)
    467                 fetch_var_name=fetch_var_name)
    468 
--> 469         self._feed_data(program, feed, feed_var_name, scope)
    470         self.executor.run(program.desc, scope, 0, True, True)
    471         outs = self._fetch_data(fetch_list, fetch_var_name, scope)

/opt/conda/envs/py35-paddle1.0.0/lib/python3.5/site-packages/paddle/fluid/executor.py in _feed_data(self, program, feed, feed_var_name, scope)
    341                 cur_feed = feed[feed_target_name]
    342                 if not isinstance(cur_feed, core.LoDTensor):
--> 343                     cur_feed = _as_lodtensor(cur_feed, self.place)
    344                 idx = op.desc.attr('col')
    345                 core.set_feed_variable(scope, cur_feed, feed_var_name, idx)

/opt/conda/envs/py35-paddle1.0.0/lib/python3.5/site-packages/paddle/fluid/executor.py in _as_lodtensor(data, place)
    247                 ndarray to LoDTensor. Please convert data to LoDTensor \
    248                 directly before feeding the data.\
--> 249                 ")
    250     # single tensor case
    251     tensor = core.LoDTensor()

RuntimeError: Some of your feed data hold LoD information.                 They can not be completely cast from a list of Python                 ndarray to LoDTensor. Please convert data to LoDTensor                 directly before feeding the data.             
```

 + 问题复现：使用时列表，通过`lod.append([word_dict.get(words.encode('utf-8'), UNK) for words in c])`把句子转换成整数列表，使用这个整数列表进行预测，在执行预测的时候就会报以上的错误。错误代码如下：

```python
inferencer = Inferencer(
    infer_func=partial(inference_program, word_dict),
    param_path=params_dirname,
    place=place)
reviews_str = ['paddlepaddle from baidu', 'this is a great movie', 'this is very bad fack']
reviews = [c.split() for c in reviews_str]
UNK = word_dict['<unk>']
print(reviews)
lod = []
for c in reviews:
    lod.append([word_dict.get(words.encode('utf-8'), UNK) for words in c])
print(lod)
results = inferencer.infer({'words': lod})
```

 + 解决问题：PaddlePaddle虽然是支持整型数据，但是在使用使用数据预测时，需要把数据转换成PaddlePaddle的张量，使用的接口是`fluid.create_lod_tensor`。

```python
inferencer = Inferencer(
    infer_func=partial(inference_program, word_dict),
    param_path=params_dirname,
    place=place)
reviews_str = ['paddlepaddle from baidu', 'this is a great movie', 'this is very bad fack']
reviews = [c.split() for c in reviews_str]
UNK = word_dict['<unk>']
print(reviews)
lod = []
for c in reviews:
    lod.append([word_dict.get(words.encode('utf-8'), UNK) for words in c])
print(lod)
base_shape = [[len(c) for c in lod]]
print(base_shape)
tensor_words = fluid.create_lod_tensor(lod, base_shape, place)
results = inferencer.infer({'words': tensor_words})
```




## `已审核`6.问题：使用长短期记忆模型的时候出现维度错误

 + 关键字：`数据维度`，`词向量`
 

 + 问题描述：在使用`fluid.layers.dynamic_lstm`建立一个长短期记忆网络时，出现数据维度或者权重不一致的错误。


 + 报错信息：

```
<ipython-input-3-3c2a355f1576> in stacked_lstm_net(data, input_dim, class_dim, emb_dim, hid_dim, stacked_num)
      6 
      7 #     fc1 = fluid.layers.fc(input=emb, size=hid_dim)
----> 8     lstm1, cell1 = fluid.layers.dynamic_lstm(input=emb, size=hid_dim)
      9 
     10     inputs = [fc1, lstm1]

/opt/conda/envs/py35-paddle1.0.0/lib/python3.5/site-packages/paddle/fluid/layers/nn.py in dynamic_lstm(input, size, h_0, c_0, param_attr, bias_attr, use_peepholes, is_reverse, gate_activation, cell_activation, candidate_activation, dtype, name)
    434             'gate_activation': gate_activation,
    435             'cell_activation': cell_activation,
--> 436             'candidate_activation': candidate_activation
    437         })
    438     return hidden, cell

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

EnforceNotMet: Enforce failed. Expected w_dims[0] == frame_size, but received w_dims[0]:128 != frame_size:32.
The first dimension of Input(Weight) should be 32. at [/paddle/paddle/fluid/operators/lstm_op.cc:63]
PaddlePaddle Call Stacks: 
```

 + 问题复现：使用`fluid.layers.embedding`接口把输入的转换成词向量，然后使用这些词向量传入到`fluid.layers.dynamic_lstm`接口中，计划使用`fluid.layers.dynamic_lstm`接口创建一个长短期记忆网络。但是在执行训练时就报以上的错误，错误代码如下：

```python
emb = fluid.layers.embedding(
    input=data, size=[input_dim, emb_dim], is_sparse=True)
lstm1, cell1 = fluid.layers.dynamic_lstm(input=emb, size=hid_dim)
```

 + 解决问题：上面的错误是因为使用`fluid.layers.embedding`创建的词向量和`fluid.layers.dynamic_lstm`所需的输入的维度不一致，为了解决这个问题，可以在中间加一个全连接层统一大小。正确代码如下：

```python
emb = fluid.layers.embedding(
    input=data, size=[input_dim, emb_dim], is_sparse=True)
fc1 = fluid.layers.fc(input=emb, size=hid_dim)
lstm1, cell1 = fluid.layers.dynamic_lstm(input=fc1, size=hid_dim)
```




## `已审核`7.问题：在使用长短期记忆网络训练时出现出现输入形状错误

 + 关键字：`序列池化`，`长短期记忆网络`
 

 + 问题描述：使用一个长短期记忆网络训练IMDB数据集时，出现输入形状错误，错误提示：输入(X)和输入(标签)应具有相同的形状。


 + 报错信息：

```
<ipython-input-7-fd22a596e844> in train(use_cuda, train_program, params_dirname)
     41         event_handler=event_handler,
     42         reader=train_reader,
---> 43         feed_order=feed_order)

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

EnforceNotMet: Enforce failed. Expected framework::slice_ddim(x_dims, 0, rank - 1) == framework::slice_ddim(label_dims, 0, rank - 1), but received framework::slice_ddim(x_dims, 0, rank - 1):31673 != framework::slice_ddim(label_dims, 0, rank - 1):128.
Input(X) and Input(Label) shall have the same shape except the last dimension. at [/paddle/paddle/fluid/operators/cross_entropy_op.cc:37]
PaddlePaddle Call Stacks: 
```

 + 问题复现：在构建一个长短期记忆网络时，首先使用`fluid.layers.fc`定义了一个全连接层，然后又使用`fluid.layers.dynamic_lstm`创建了一个长短期记忆单元，最后使用使用这个两个进行分类输出，结果就会出现上面的错误，错误代码如下：

```python
emb = fluid.layers.embedding(input=data, size=[input_dim, emb_dim], is_sparse=True)
fc1 = fluid.layers.fc(input=emb, size=hid_dim)
lstm1, cell1 = fluid.layers.dynamic_lstm(input=fc1, size=hid_dim)
prediction = fluid.layers.fc(input=[fc1, lstm1], size=class_dim, act='softmax')
```

 + 解决问题：搭建一个长短期记忆网络时，在执行最好一层分类器前还要经过一个序列进行池化的接口，将上面的全连接层和长短期记忆单元的输出全部时间步的特征进行池化，最后才执行分类器输出。正确代码如下：

```python
emb = fluid.layers.embedding(input=data, size=[input_dim, emb_dim], is_sparse=True)
fc1 = fluid.layers.fc(input=emb, size=hid_dim)
lstm1, cell1 = fluid.layers.dynamic_lstm(input=fc1, size=hid_dim)
fc_last = fluid.layers.sequence_pool(input=fc1, pool_type='max')
lstm_last = fluid.layers.sequence_pool(input=lstm1, pool_type='max')
prediction = fluid.layers.fc(input=[fc_last, lstm_last], size=class_dim, act='softmax')
```



## `已审核`8.问题：使用PaddlePaddle搭建一个循环神经网络出现输入的宽度和高度不一致

 + 关键字：`记忆单元`，`循环神经网络`
 

 + 问题描述：使用`fluid.layers.DynamicRNN`创建一个循环神经网络时，在执行训练的时候出现错误，错误提示：第一个矩阵的宽度必须等于第二个矩阵的高度。


 + 报错信息：

```
<ipython-input-7-fd22a596e844> in train(use_cuda, train_program, params_dirname)
     41         event_handler=event_handler,
     42         reader=train_reader,
---> 43         feed_order=feed_order)

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

EnforceNotMet: Enforce failed. Expected x_mat_dims[1] == y_mat_dims[0], but received x_mat_dims[1]:64 != y_mat_dims[0]:128.
First matrix's width must be equal with second matrix's height.  at [/paddle/paddle/fluid/operators/mul_op.cc:59]
PaddlePaddle Call Stacks: 
```

 + 问题复现：使用`rnn.block`定义一个循环神经网络块，使用`rnn.memory`定义一个记忆单元，大小设置为128，接着使用`fluid.layers.fc`创建要给全连接层，大小设置为64。然后在执行训练的时就会出现上述的错误。错误代码如下：

```python
emb = fluid.layers.embedding(input=ipt, size=[input_dim, 128], is_sparse=True)
sentence = fluid.layers.fc(input=emb, size=128, act='tanh')
rnn = fluid.layers.DynamicRNN()
with rnn.block():
    word = rnn.step_input(sentence)
    prev = rnn.memory(shape=[128])
    hidden = fluid.layers.fc(input=[word, prev], size=64, act='relu')
    rnn.update_memory(prev, hidden)
    rnn.output(hidden)
last = fluid.layers.sequence_last_step(rnn())
out = fluid.layers.fc(input=last, size=2, act='softmax')
```

 + 解决问题：上面的错误是因为记忆单元的大小和全连接层的大小不一致，有些用户会错误理解上一层和下一层的大小是互补相干的，这是错误的，它们的大小必须是一样的。正确代码如下：

```python
emb = fluid.layers.embedding(input=ipt, size=[input_dim, 128], is_sparse=True)
sentence = fluid.layers.fc(input=emb, size=128, act='tanh')
rnn = fluid.layers.DynamicRNN()
with rnn.block():
    word = rnn.step_input(sentence)
    prev = rnn.memory(shape=[128])
    hidden = fluid.layers.fc(input=[word, prev], size=128, act='relu')
    rnn.update_memory(prev, hidden)
    rnn.output(hidden)
last = fluid.layers.sequence_last_step(rnn())
out = fluid.layers.fc(input=last, size=2, act='softmax')
```


## `已审核`9.问题：情感分析的dome，怎么用自定义的训练集和测试集？

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



