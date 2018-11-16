# 情感分析

# 1.问题：

 + 问题描述：


 + 报错信息：

```
[[5146, 5146, 5146, 5146, 5146, 5146], [5146, 5146, 5146, 5146, 5146], [5146, 5146, 5146, 5146]]
Predict probability of  0.54538333  to be positive and  0.45461673  to be negative for review ' read the book forget the movie '
Predict probability of  0.54523355  to be positive and  0.45476642  to be negative for review ' this is a great movie '
Predict probability of  0.54504114  to be positive and  0.45495886  to be negative for review ' this is very bad '
```

 + 问题复现：

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

 + 解决问题：

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



# 2.问题：

 + 问题描述：


 + 报错信息：

```
['read the book forget the movie', 'this is a great movie', 'this is very bad']
[[2237, 4008, 2, 2062, 5146, 3602, 3752, 4008, 5146, 951, 2903, 2903, 5146, 5146, 2414, 2903, 2237, 3316, 4008, 3602, 5146, 3602, 3752, 4008, 5146, 4136, 2903, 5146, 8, 4008], [3602, 3752, 8, 2551, 5146, 8, 2551, 5146, 2, 5146, 3316, 2237, 4008, 2, 3602, 5146, 4136, 2903, 5146, 8, 4008], [3602, 3752, 8, 2551, 5146, 8, 2551, 5146, 5146, 4008, 2237, 5146, 5146, 951, 2, 2062]]
Predict probability of  0.59143597  to be positive and  0.4085641  to be negative for review ' read the book forget the movie '
Predict probability of  0.73750913  to be positive and  0.26249087  to be negative for review ' this is a great movie '
Predict probability of  0.55495805  to be positive and  0.445042  to be negative for review ' this is very bad '
```

 + 问题复现：

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

 + 解决问题：

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

```
[['read', 'the', 'book', 'forget', 'the', 'movie'], ['this', 'is', 'a', 'great', 'movie'], ['this', 'is', 'very', 'bad']]
[[325, 0, 276, 818, 0, 16], [9, 5, 2, 78, 16], [9, 5, 51, 81]]
Predict probability of  0.44390476  to be positive and  0.55609524  to be negative for review ' read the book forget the movie '
Predict probability of  0.83933955  to be positive and  0.16066049  to be negative for review ' this is a great movie '
Predict probability of  0.35688713  to be positive and  0.64311296  to be negative for review ' this is very bad '
```



# 3.问题：

 + 问题描述：


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

 + 问题复现：

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

 + 解决问题：

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




# 4.问题：

 + 问题描述：


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

 + 问题复现：

```python
inferencer = Inferencer(
    infer_func=partial(inference_program, word_dict),
    param_path=params_dirname,
    place=place)
reviews_str = ['paddlepaddle from baidu', 'this is a great movie', 'this is very bad fack']
reviews = [c.split() for c in reviews_str]
results = inferencer.infer({'words': reviews})
```

 + 解决问题：

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




# 5.问题：

 + 问题描述：


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

 + 问题复现：

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

 + 解决问题：

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




# 6.问题：

 + 问题描述：


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

 + 问题复现：

```python
emb = fluid.layers.embedding(
    input=data, size=[input_dim, emb_dim], is_sparse=True)
lstm1, cell1 = fluid.layers.dynamic_lstm(input=emb, size=hid_dim)
```

 + 解决问题：

```python
emb = fluid.layers.embedding(
    input=data, size=[input_dim, emb_dim], is_sparse=True)
fc1 = fluid.layers.fc(input=emb, size=hid_dim)
lstm1, cell1 = fluid.layers.dynamic_lstm(input=fc1, size=hid_dim)
```




# 7.问题：

 + 问题描述：


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

 + 问题复现：

```python
emb = fluid.layers.embedding(input=data, size=[input_dim, emb_dim], is_sparse=True)
fc1 = fluid.layers.fc(input=emb, size=hid_dim)
lstm1, cell1 = fluid.layers.dynamic_lstm(input=fc1, size=hid_dim)
prediction = fluid.layers.fc(input=[fc1, lstm1], size=class_dim, act='softmax')
```

 + 解决问题：

```python
emb = fluid.layers.embedding(input=data, size=[input_dim, emb_dim], is_sparse=True)
fc1 = fluid.layers.fc(input=emb, size=hid_dim)
lstm1, cell1 = fluid.layers.dynamic_lstm(input=fc1, size=hid_dim)
fc_last = fluid.layers.sequence_pool(input=fc1, pool_type='max')
lstm_last = fluid.layers.sequence_pool(input=lstm1, pool_type='max')
prediction = fluid.layers.fc(input=[fc_last, lstm_last], size=class_dim, act='softmax')
```




# 8.问题：

 + 问题描述：


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

 + 问题复现：

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

 + 解决问题：

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


