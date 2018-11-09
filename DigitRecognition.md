## 1.问题：使用MNIST数据训练时出现张量类型错误

+ 关键字：`张量`，`数据类型`

+ 问题描述：使用卷积神经网络训练MNIST数据集，由于输入数据的数据类型设置为float32，在训练时直接报错，报错信息提示张量类型错误。

+ 报错信息：

```bash
/usr/local/lib/python3.5/dist-packages/paddle/fluid/executor.py in run(self, program, feed, fetch_list, feed_var_name, fetch_var_name, scope, return_numpy, use_program_cache)
    468 
    469         self._feed_data(program, feed, feed_var_name, scope)
--> 470         self.executor.run(program.desc, scope, 0, True, True)
    471         outs = self._fetch_data(fetch_list, fetch_var_name, scope)
    472         if return_numpy:

EnforceNotMet: Tensor holds the wrong type, it holds f at [/paddle/paddle/fluid/framework/tensor_impl.h:29]
PaddlePaddle Call Stacks: 
```

+ 问题复现：
使用MNIST数据集，使用卷积神经网络进行训练，并定义输入层数据类型设置为float64，即将dtype参数的值设置为float64，如下代码片段：

```
image = fluid.layers.data(name='image', shape=[1, 28, 28], dtype='float64')
label = fluid.layers.data(name='label', shape=[1], dtype='int64')
```

+ 问题解决：
由于如果数据为float32，而定义的输入层数据类型为float64，导致的数量类型不正确。把输入数据的类型设置为float32，即将dtype参数值设置为float32。如下代码片段：

```
image = fluid.layers.data(name='image', shape=[1, 28, 28], dtype='float32')
label = fluid.layers.data(name='label', shape=[1], dtype='int64')
```

+ 问题拓展：PaddlePaddle支持多种数据类型，比如上面使用的`float32`，这个是主要实数类型，`float64`是次要实数类型，支持大部分操作。我们使用的标签是int64，这个是主要标签类型，也有次要标签类型`int32`。也有一些控制流的数据类型`bool`。


## 2.问题：在优化方法处报错 EnforceNotMet: Enforce failed

+ 关键字：`rank`，`优化方法`，`损失函数`

+ 问题描述：
执行定义损失函数代码后，再执行优化方法就报错，提示执行失败，dy_dims.size():1 != rank:2。

+ 报错信息：

```
/usr/local/lib/python3.5/dist-packages/paddle/fluid/optimizer.py in minimize(self, loss, startup_program, parameter_list, no_grad_set)
    253         """
    254         params_grads = append_backward(loss, parameter_list, no_grad_set,
--> 255                                        [error_clip_callback])
    256 
    257         params_grads = sorted(params_grads, key=lambda x: x[0].name)

/usr/local/lib/python3.5/dist-packages/paddle/fluid/backward.py in append_backward(loss, parameter_list, no_grad_set, callbacks)
    588     _rename_grad_(root_block, fwd_op_num, grad_to_var, {})
    589 
--> 590     _append_backward_vars_(root_block, fwd_op_num, grad_to_var, grad_info_map)
    591 
    592     program.current_block_idx = current_block_idx

/usr/local/lib/python3.5/dist-packages/paddle/fluid/backward.py in _append_backward_vars_(block, start_op_idx, grad_to_var, grad_info_map)
    424         # infer_shape and infer_type
    425         op_desc.infer_var_type(block.desc)
--> 426         op_desc.infer_shape(block.desc)
    427         # ncclInit dones't need to set data_type
    428         if op_desc.type() == 'ncclInit':

EnforceNotMet: Enforce failed. Expected dy_dims.size() == rank, but received dy_dims.size():1 != rank:2.
Input(Y@Grad) and Input(X) should have the same rank. at [/paddle/paddle/fluid/operators/cross_entropy_op.cc:82]
PaddlePaddle Call Stacks: 
```


+ 问题复现：
定义一个交叉熵损失函数，直接使用这个损失函数传给优化方法，再执行到这一行代码就出现这个问题。错误代码如下：

```
cost = fluid.layers.cross_entropy(input=model, label=label)

optimizer = fluid.optimizer.AdamOptimizer(learning_rate=0.001)
opts = optimizer.minimize(cost)
```


+ 问题解决：
训练是一个Batch进行训练的，所以计算的损失值也是计算一个Batch的损失值。优化方法参数使用的是一个平均的损失函数，所以不能直接使用损失函数，还需要对损失函数求平均值。正确代码如下：

```
cost = fluid.layers.cross_entropy(input=model, label=label)
avg_cost = fluid.layers.mean(cost)

optimizer = fluid.optimizer.AdamOptimizer(learning_rate=0.001)
opts = optimizer.minimize(avg_cost)
```

+ 问题拓展：
如果在训练的时候，`fetch_list`参数使用的是`cost`，而不是`avg_cost`的话，训练输出的也会是一个Batch的损失值。所以在训练的时候，`fetch_list`参数的值最好使用`avg_cost`，输出的是平均损失值，从而更方便观察训练情况。



## 3.问题：在训练时报错 Expected label_dims[rank - 1] == 1UL

+ 关键字：`标签维度`，`label`
 
+ 问题描述：使用MNIST数据集训练分类模型报错，提示label的维度不正确。

+ 报错信息：

```
/usr/local/lib/python3.5/dist-packages/paddle/fluid/layers/nn.py in cross_entropy(input, label, soft_label, ignore_index)
   1126         outputs={'Y': [out]},
   1127         attrs={"soft_label": soft_label,
-> 1128                "ignore_index": ignore_index})
   1129     return out
   1130 

/usr/local/lib/python3.5/dist-packages/paddle/fluid/layer_helper.py in append_op(self, *args, **kwargs)
     48 
     49     def append_op(self, *args, **kwargs):
---> 50         return self.main_program.current_block().append_op(*args, **kwargs)
     51 
     52     def multiple_input(self, input_param_name='input'):

/usr/local/lib/python3.5/dist-packages/paddle/fluid/framework.py in append_op(self, *args, **kwargs)
   1205         """
   1206         op_desc = self.desc.append_op()
-> 1207         op = Operator(block=self, desc=op_desc, *args, **kwargs)
   1208         self.ops.append(op)
   1209         return op

/usr/local/lib/python3.5/dist-packages/paddle/fluid/framework.py in __init__(***failed resolving arguments***)
    654         if self._has_kernel(type):
    655             self.desc.infer_var_type(self.block.desc)
--> 656             self.desc.infer_shape(self.block.desc)
    657 
    658     def _has_kernel(self, op_type):

EnforceNotMet: Enforce failed. Expected label_dims[rank - 1] == 1UL, but received label_dims[rank - 1]:10 != 1UL:1.
If Attr(softLabel) == false, the last dimension of Input(Label) should be 1. at [/paddle/paddle/fluid/operators/cross_entropy_op.cc:45]
PaddlePaddle Call Stacks: 
```


+ 问题复现：
使用卷积神经网络训练MNIST数据集，定义label输出层设置形状为`[10]`。在执行开始训练的时候就会报错。错误代码如下：

```
image = fluid.layers.data(name='image', shape=[1, 28, 28], dtype='float32')
label = fluid.layers.data(name='label', shape=[10], dtype='int64')
```


+ 问题解决：
因为每一条数据对应的label只有一个值，所以label的形状应该是`(1)`。label的形状是值label的维度，而不是label的类别数量。正确代码如下：

```
image = fluid.layers.data(name='image', shape=[1, 28, 28], dtype='float32')
label = fluid.layers.data(name='label', shape=[1], dtype='int64')
```

+ 问题分析：
在PaddlePaddle的旧版本中，在定义label的大小需要在输入层设置label的数量。而在新版本Fluid的中，定义label输入层是设置label数量的形状，而不是label的数量。




## 4.问题：训练过程中损失值突然全部为0

+ 关键字：`损失值`，`梯度消失`
 
+ 问题描述：使用卷积神经网络在训练MNIST数据集时，在训练过程中损失值突然为0，并且识别准确率开始下降。

+ 报错信息：

```
Pass:0, Batch:0, Cost:1.41644, Accuracy:0.16406
Pass:0, Batch:100, Cost:0.00000, Accuracy:0.07812
Pass:0, Batch:200, Cost:0.00000, Accuracy:0.07812
Pass:0, Batch:300, Cost:0.00000, Accuracy:0.11719
```


+ 问题复现：
在卷积神经网络最后的输出层的激活函数使用Sigmoid，然后作为图像分类的神经网络进行训练。在训练过程中就会出现损失值突然为0的情况。

```
def convolutional_neural_network(input):
    conv1 = fluid.layers.conv2d(input=input,
                                num_filters=32,
                                filter_size=3,
                                stride=1)

    pool1 = fluid.layers.pool2d(input=conv1,
                                pool_size=2,
                                pool_stride=1,
                                pool_type='max')

    conv2 = fluid.layers.conv2d(input=pool1,
                                num_filters=64,
                                filter_size=3,
                                stride=1)

    pool2 = fluid.layers.pool2d(input=conv2,
                                pool_size=2,
                                pool_stride=1,
                                pool_type='max')

    fc = fluid.layers.fc(input=pool2, size=10, act='sigmoid')
    return fc
```


+ 问题解决：
Sigmoid函数比较常用于二分类任务上，在手写数字识别任务上，MNIST手写数据集有10个类别。所以这里使用Sigmoid函数不适合，可以使用Softmax作为激活函数，可以使用的模型正常收敛。正确代码如下：

```
def convolutional_neural_network(input):
    conv1 = fluid.layers.conv2d(input=input,
                                num_filters=32,
                                filter_size=3,
                                stride=1)

    pool1 = fluid.layers.pool2d(input=conv1,
                                pool_size=2,
                                pool_stride=1,
                                pool_type='max')

    conv2 = fluid.layers.conv2d(input=pool1,
                                num_filters=64,
                                filter_size=3,
                                stride=1)

    pool2 = fluid.layers.pool2d(input=conv2,
                                pool_size=2,
                                pool_stride=1,
                                pool_type='max')

    fc = fluid.layers.fc(input=pool2, size=10, act='softmax')
    return fc
```


+ 问题拓展：当全连接层的激活函数是`sigmoid`或者`softmax`时，那么这个全连接层相当于一个分类器，`sigmoid`激活函数多用于二分类任务，`softmax`多用于多分类任务。当全连接层的激活函数是`relu`、`tanh`，可以增强网络的非线性能力。





## 5.问题：在测试数据集进行测试时出错：Cannot find fetch variable in scope

+ 关键词：`测试程序`
 
+ 问题描述：从主程序中克隆一个程序作为测试程序，使用这个测试程序在训练之后使用测试数据集进行测试，在执行测试程序时报错，错误提示找不到fetch变量。

+ 报错信息：

```
/usr/local/lib/python3.5/dist-packages/paddle/fluid/executor.py in run(self, program, feed, fetch_list, feed_var_name, fetch_var_name, scope, return_numpy, use_program_cache)
    468 
    469         self._feed_data(program, feed, feed_var_name, scope)
--> 470         self.executor.run(program.desc, scope, 0, True, True)
    471         outs = self._fetch_data(fetch_list, fetch_var_name, scope)
    472         if return_numpy:

EnforceNotMet: Cannot find fetch variable in scope, fetch_var_name is mean_0.tmp_0 at [/paddle/paddle/fluid/operators/fetch_op.cc:37]
PaddlePaddle Call Stacks: 
```


+ 问题复现：
在定义优化方法之前就从主程序`default_main_program()`克隆一个测试程序，使用这个测试程序，通过执行器运行测试程序就出现错误。错误代码如下：

```
test_program = fluid.default_main_program().clone(for_test=True)

cost = fluid.layers.cross_entropy(input=model, label=label)
avg_cost = fluid.layers.mean(cost)
acc = fluid.layers.accuracy(input=model, label=label)
```


+ 问题解决：在定义损失函数和优化方法都会添加都主程序`default_main_program()`中，而测试不需要使用到训练时用到的一些操作，所以在克隆测试程序时，需要定义在损失函数之后，优化方法之前。正确代码如下：

```
cost = fluid.layers.cross_entropy(input=model, label=label)
avg_cost = fluid.layers.mean(cost)
acc = fluid.layers.accuracy(input=model, label=label)

test_program = fluid.default_main_program().clone(for_test=True)
```

+ 问题拓展：PaddlePaddle的`Program`是Fluid程序主要组成部分之一， Fluid程序中通常存在 2段`Program`。`fluid.default_startup_program`是定义了创建模型参数，输入输出，以及模型中可学习参数的初始化等各种操作。而`fluid.default_main_program`是定义了神经网络模型，前向反向计算，以及优化算法对网络中可学习参数的更新。




## 6.问题：训练时出现错误 y_dims.size():1 <= y_num_col_dims:1

+ 关键字：`初试化`，`执行器`
 
+ 问题描述：在定义执行器之后，就直接使用执行器进行训练，就出现错误，提示错误 y_dims.size():1 <= y_num_col_dims:1。

+ 报错信息：

```
/usr/local/lib/python3.5/dist-packages/paddle/fluid/executor.py in run(self, program, feed, fetch_list, feed_var_name, fetch_var_name, scope, return_numpy, use_program_cache)
    468 
    469         self._feed_data(program, feed, feed_var_name, scope)
--> 470         self.executor.run(program.desc, scope, 0, True, True)
    471         outs = self._fetch_data(fetch_list, fetch_var_name, scope)
    472         if return_numpy:

EnforceNotMet: Enforce failed. Expected y_dims.size() > y_num_col_dims, but received y_dims.size():1 <= y_num_col_dims:1.
The input tensor Y's rank of MulOp should be larger than y_num_col_dims. at [/paddle/paddle/fluid/operators/mul_op.cc:52]
PaddlePaddle Call Stacks: 
```


+ 问题复现：编写一个图像分类程序，在定义执行器之后，使用执行器`exe`执行`run`函数，就会出现这个问题。错误代码如下：

```
place = fluid.CPUPlace()
exe = fluid.Executor(place)
for batch_id, data in enumerate(train_reader()):
    train_cost, train_acc = exe.run(program=fluid.default_main_program(),
                                    feed=feeder.feed(data),
                                    fetch_list=[avg_cost, acc])
```


+ 问题解决：定义执行器之后，因为还没有执行初始化模型参数，所以缺少初始化数据，导致出现这个问题。在定义执行器之后，还执行初始化参数程序`exe.run(fluid.default_startup_program())`，之后再执行训练程序。正确代码如下：

```
place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())
for batch_id, data in enumerate(train_reader()):
    train_cost, train_acc = exe.run(program=fluid.default_main_program(),
                                    feed=feeder.feed(data),
                                    fetch_list=[avg_cost, acc])
```

+ 问题分析：在定义网络之后，Fluid内部有大量的参数需要进行初始化才能正常运行，网络也才能正确使用，所以在执行训练之前需要执行`exe.run(fluid.default_startup_program())`初始化参数。




## 7.问题：训练时出现错误：ValueError: var image not in this block

+ 关键字：`初始化`，`主程序`，`program`
 
+ 问题描述：使用卷积神经网络训练MNIST数据集，再执行训练程序时出现错误，错误提示var image not in this block。

+ 报错信息：

```
/usr/local/lib/python3.5/dist-packages/paddle/fluid/executor.py in run(self, program, feed, fetch_list, feed_var_name, fetch_var_name, scope, return_numpy, use_program_cache)
    465                 fetch_list=fetch_list,
    466                 feed_var_name=feed_var_name,
--> 467                 fetch_var_name=fetch_var_name)
    468 
    469         self._feed_data(program, feed, feed_var_name, scope)

/usr/local/lib/python3.5/dist-packages/paddle/fluid/executor.py in _add_feed_fetch_ops(self, program, feed, fetch_list, feed_var_name, fetch_var_name)
    313         if not has_feed_operators(global_block, feed, feed_var_name):
    314             for i, name in enumerate(feed):
--> 315                 out = global_block.var(name)
    316                 global_block._prepend_op(
    317                     type='feed',

/usr/local/lib/python3.5/dist-packages/paddle/fluid/framework.py in var(self, name)
   1038         v = self.vars.get(name, None)
   1039         if v is None:
-> 1040             raise ValueError("var %s not in this block" % name)
   1041         return v
   1042 

ValueError: var image not in this block
```


+ 问题复现：在执行训练程序时，`run`函数的program参数的值设置为`fluid.default_startup_program()`，当执行到这一行的时就会出错。错误代码如下：

```
for batch_id, data in enumerate(train_reader()):
    train_cost, train_acc = exe.run(program=fluid.default_startup_program(),
                                    feed=feeder.feed(data),
                                    fetch_list=[avg_cost, acc])
```


+ 问题解决：在执行训练时，`run`函数的program参数的值应该时`fluid.default_main_program()`的，错误的原因是使用了初始化参数的程序来进行训练而导致的错误。正确代码如下：

```
for batch_id, data in enumerate(train_reader()):
    train_cost, train_acc = exe.run(program=fluid.default_main_program(),
                                    feed=feeder.feed(data),
                                    fetch_list=[avg_cost, acc])
```

+ 问题分析：`fluid.default_main_program`是定义了神经网络模型，前向反向计算，以及优化算法对网络中可学习参数的更新，所以在训练的时候，使用的Program应该是`fluid.default_main_program`。而不是用于初始化的Program。




## 8.问题：使用测试程序预测图片时出错：rank:2 != label_dims.size():1

+ 关键字：`测试程序`，`feed`

+ 问题描述：在使用测试程序预测自己的图片的时候，在执行`run`函数的时候出错，错误提示rank:2 != label_dims.size():1。

+ 报错信息：

```
/usr/local/lib/python3.5/dist-packages/paddle/fluid/executor.py in run(self, program, feed, fetch_list, feed_var_name, fetch_var_name, scope, return_numpy, use_program_cache)
    468 
    469         self._feed_data(program, feed, feed_var_name, scope)
--> 470         self.executor.run(program.desc, scope, 0, True, True)
    471         outs = self._fetch_data(fetch_list, fetch_var_name, scope)
    472         if return_numpy:

EnforceNotMet: Enforce failed. Expected rank == label_dims.size(), but received rank:2 != label_dims.size():1.
Input(X) and Input(Label) shall have the same rank. at [/paddle/paddle/fluid/operators/cross_entropy_op.cc:33]
PaddlePaddle Call Stacks: 
```


+ 问题复现：使用克隆得到的测试程序`test_program`来预测自己的图片，图片通过`feed`参数以键值对的方式传入到预测程序中，但不传入label值。然后执行`run`函数。错误代码如下：

```
results = exe.run(program=test_program,
                  feed={'image': img},
                  fetch_list=[model])
```


+ 问题解决：因为测试程序是为了用于测试克隆得到的，在测试需要图片数据的同时，还需要label数据。所以我们使用测试程序预测图片时，还要输入label值。为了能够让程序运行，最简单的方式是模拟一个假的label值传入到程序中，就可以解决这个错误了。正确代码如下：

```
results = exe.run(program=test_program,
                  feed={'image': img, "label": np.array([[1]]).astype("int64")},
                  fetch_list=[model])
```

+ 问题拓展：测试程序是从主程序`fluid.default_main_program`中克隆得到的，所以也继承了主程序的输入数据的格式，需要同时输入图像数据和label数据。但真实的预测是不会使用label作为输入的。在真实预测中，还要对模型进行修剪，去掉label的输入。



## 9.问题：迭代数据时出现错误：TypeError: 'function' object is not iterable

+ `reader`，`数据读取`
 
+ 问题描述：在读取使用reader读取训练数据时，出现错误，错误提示TypeError: 'function' object is not iterable。

+ 报错信息：

```
TypeError                                 Traceback (most recent call last)
<ipython-input-12-0b74c209241b> in <module>
      2 for pass_id in range(1):
      3     # 进行训练
----> 4     for batch_id, data in enumerate(train_reader):
      5         train_cost, train_acc = exe.run(program=fluid.default_main_program(),
      6                                         feed=feeder.feed(data),

TypeError: 'function' object is not iterable

```


+ 问题复现：在循环中读取数据时，通过`paddle.batch()`定义的reader对数据进行迭代，`enumerate()`使用的是定义的变量。当调用该函数的时候，就会报错，错误代码如下：

```
for batch_id, data in enumerate(train_reader):
    train_cost, train_acc = exe.run(program=fluid.default_main_program(),
                                    feed=feeder.feed(data),
                                    fetch_list=[avg_cost, acc])
```

+ 问题解决：同过`paddle.batch()`得到的一个读取数据的函数，返回值是一个reader，上面之所以错误是因为直接`train_reader`变量，这变量是指一个函数，所以需要加一个括号，得到这个函数的返回值reader。

```
for batch_id, data in enumerate(train_reader()):
    train_cost, train_acc = exe.run(program=fluid.default_main_program(),
                                    feed=feeder.feed(data),
                                    fetch_list=[avg_cost, acc])
```


+ 在Python的变量中，不带括号时，调用的是这个函数本身，是一个函数对象，不须等该函数执行完成。带括号时，调用的是函数的执行结果，须等该函数执行完成的结果。




## 10.问题：在定义训练器是出现NameError: name 'Trainer' is not defined

+ 关键字：`训练器`，`contrib`，`Trainer`
 
+ 问题描述：在使用Trainer函数创建训练器的时候，出现错误，错误提示NameError: name 'Trainer' is not defined。

+ 报错信息：

```
<ipython-input-7-4908863d4852> in main()
      9     place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
     10 
---> 11     trainer = Trainer(
     12         train_func=train_program, place=place, optimizer_func=optimizer_program)
     13 

NameError: name 'Trainer' is not defined
```


+ 问题复现：安装PaddlePaddle 1.0以上的版本，然后通过`from paddle.fluid.trainer import *`导入PaddlePaddle的高级API，之后使用`Trainer`创建一个训练器，就会报错。错误代码如下：

```
from paddle.fluid.trainer import *
from paddle.fluid.inferencer import *
······
trainer = Trainer(
    train_func=train_program, place=place, optimizer_func=optimizer_program)
```


+  问题解决：在PaddlePaddle 1.0以上的版本，高级API已经迁移到`paddle.fluid.contrib`目录下，所以在导包的是应该要使用`from paddle.fluid.contrib.trainer import *`的导包方式。正确代码如下：

```
from paddle.fluid.contrib.trainer import *
from paddle.fluid.contrib.inferencer import *
······
trainer = Trainer(
    train_func=train_program, place=place, optimizer_func=optimizer_program)
```

+  问题分析：在对于高层API，PaddlePaddle在版本1.0之后做了很大的改动，比如就修改了高层API所在的位置。同时也完善了高层API的很多功能。高层API虽然没有底层API灵活，但高层API使用更简单，非常适合初学者使用。



## 11.问题：使用高级API训练时出现ValueError: var img not in this block

+ 关键字：`输入层name`
 
+ 问题描述：在使用PaddlePaddle 1.0以上的版本，通过使用高级API进行训练，在执行训练的时候出现错误，错误提示ValueError: var img not in this block。

+ 报错信息：

```
<ipython-input-7-3294b0f46718> in main()
     41         event_handler=event_handler,
     42         reader=train_reader,
---> 43         feed_order=['img', 'label'])
     44 
     45     # find the best pass

/usr/local/lib/python3.5/dist-packages/paddle/fluid/contrib/trainer.py in train(self, num_epochs, event_handler, reader, feed_order)
    403         else:
    404             self._train_by_executor(num_epochs, event_handler, reader,
--> 405                                     feed_order)
    406 
    407     def test(self, reader, feed_order):

/usr/local/lib/python3.5/dist-packages/paddle/fluid/contrib/trainer.py in _train_by_executor(self, num_epochs, event_handler, reader, feed_order)
    476         """
    477         with self._prog_and_scope_guard():
--> 478             feed_var_list = build_feed_var_list(self.train_program, feed_order)
    479             feeder = data_feeder.DataFeeder(
    480                 feed_list=feed_var_list, place=self.place)

/usr/local/lib/python3.5/dist-packages/paddle/fluid/contrib/trainer.py in build_feed_var_list(program, feed_order)
    634     if isinstance(feed_order, list):
    635         feed_var_list = [
--> 636             program.global_block().var(var_name) for var_name in feed_order
    637         ]
    638     else:

/usr/local/lib/python3.5/dist-packages/paddle/fluid/contrib/trainer.py in <listcomp>(.0)
    634     if isinstance(feed_order, list):
    635         feed_var_list = [
--> 636             program.global_block().var(var_name) for var_name in feed_order
    637         ]
    638     else:

/usr/local/lib/python3.5/dist-packages/paddle/fluid/framework.py in var(self, name)
   1038         v = self.vars.get(name, None)
   1039         if v is None:
-> 1040             raise ValueError("var %s not in this block" % name)
   1041         return v
   1042 

ValueError: var img not in this block

```


+  问题复现：在使用`fluid.layers.data()`接口定义输入层时，其中`name`参数的值是`image`，在`trainer.train()`接口的`feed_order`参数中设置输入层的值为`img`，之后在执行训练程序时会出错。错误代码如下：

```
def convolutional_neural_network():
    img = fluid.layers.data(name='image', shape=[1, 28, 28], dtype='float32')
    # first conv pool
    conv_pool_1 = fluid.nets.simple_img_conv_pool(
        input=img,
        filter_size=5,
        num_filters=20,
        pool_size=2,
        pool_stride=2,
        act="relu")
    conv_pool_1 = fluid.layers.batch_norm(conv_pool_1)
    prediction = fluid.layers.fc(input=conv_pool_1, size=10, act='softmax')
    return prediction
······    
trainer.train(
    num_epochs=1,
    event_handler=event_handler,
    reader=train_reader,
    feed_order=['img', 'label'])
```


+ 问题解决：在训练时是根据`feed_order`参数定义的输入数据的维度，对应定义输入层的`name`的，所以`feed_order`参数的值必须要对应输入层的`name`参数的值。正确代码如下：
```
def convolutional_neural_network():
    img = fluid.layers.data(name='img', shape=[1, 28, 28], dtype='float32')
    # first conv pool
    conv_pool_1 = fluid.nets.simple_img_conv_pool(
        input=img,
        filter_size=5,
        num_filters=20,
        pool_size=2,
        pool_stride=2,
        act="relu")
    conv_pool_1 = fluid.layers.batch_norm(conv_pool_1)
    prediction = fluid.layers.fc(input=conv_pool_1, size=10, act='softmax')
    return prediction
······    
trainer.train(
    num_epochs=1,
    event_handler=event_handler,
    reader=train_reader,
    feed_order=['img', 'label'])
```





## 12.问题：在预测图像时出现错误EnforceNotMet: Conv intput should be 4-D or 5-D tensor

+ 关键字：`数据预处理`，`预测图片`
 
+ 问题描述：使用训练好的模型，预测图片。图片经过预处理之后，再调用预测接口`infer()`对图片进行预测，出现输入数据维度不正确。错误提示：EnforceNotMet: Conv intput should be 4-D or 5-D tensor

+ 报错信息：

```
EnforceNotMet                             Traceback (most recent call last)
<ipython-input-17-27415360cecb> in <module>
     23     place=place)
     24 
---> 25 results = inferencer.infer({'img': img})
     26 lab = np.argsort(results)
     27 print("Inference result of image/infer_3.png is: %d" % lab[0][0][-1])

/usr/local/lib/python3.5/dist-packages/paddle/fluid/contrib/inferencer.py in infer(self, inputs, return_numpy)
    102             results = self.exe.run(feed=inputs,
    103                                    fetch_list=[self.predict_var.name],
--> 104                                    return_numpy=return_numpy)
    105 
    106         return results

/usr/local/lib/python3.5/dist-packages/paddle/fluid/executor.py in run(self, program, feed, fetch_list, feed_var_name, fetch_var_name, scope, return_numpy, use_program_cache)
    468 
    469         self._feed_data(program, feed, feed_var_name, scope)
--> 470         self.executor.run(program.desc, scope, 0, True, True)
    471         outs = self._fetch_data(fetch_list, fetch_var_name, scope)
    472         if return_numpy:

EnforceNotMet: Conv intput should be 4-D or 5-D tensor. at [/paddle/paddle/fluid/operators/conv_op.cc:47]
PaddlePaddle Call Stacks: 
```


+ 问题复现：使用训练好保存的模型和网络结构创建一个预测器，然后图片经过预处理，再使用预处理后的图片进行预测，然后就出现错误。错误代码如下：

```
params_dirname = "recognize_digits_network.inference.model"
use_cuda = False  # set to True if training with GPU
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    
def load_image(file):
    im = Image.open(file).convert('L')
    im = im.resize((28, 28), Image.ANTIALIAS)
    im = np.array(im).astype(np.float32).flatten()
    im = im / 255.0
    return im
    
inferencer = Inferencer(
    infer_func=convolutional_neural_network,
    param_path=params_dirname,
    place=place)

img = load_image('infer_3.png')
results = inferencer.infer({'img': img})
lab = np.argsort(results)
print("Inference result of image/infer_3.png is: %d" % lab[0][0][-1])
```


+ 问题解决：上面错误的原因是数据预处理出错，上面数据预处理后的数据维度是`(784,)`，但实际需要的数据维度是`(1, 1, 28, 28)`，结构分别是Batch大小，图片通道数，图片的宽和图片的高。所以要通过`reshape(1, 1, 28, 28)`函数对图片维度进行变换。正确代码如下：

```
params_dirname = "recognize_digits_network.inference.model"
use_cuda = False  # set to True if training with GPU
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

def load_image(file):
    im = Image.open(file).convert('L')
    im = im.resize((28, 28), Image.ANTIALIAS)
    im = np.array(im).reshape(1, 1, 28, 28).astype(np.float32)
    im = im / 255.0 * 2.0 - 1.0
    return im

img = load_image('infer_3.png')
inferencer = Inferencer(
    infer_func=convolutional_neural_network,
    param_path=params_dirname,
    place=place)

results = inferencer.infer({'img': img})
lab = np.argsort(results)
print("Inference result of image/infer_3.png is: %d" % lab[0][0][-1])
```





## 13.问题：在训练时出现TypeError: simple_img_conv_pool() got an unexpected keyword argument 'stride'

+ 关键字：`stride`，`步长`
 
+ 问题描述：在使用`fluid.nets.simple_img_conv_pool()`接口建立一个卷积神经网络时，当通过参数`stride`设置卷积操作的滑动步长，在训练的时候报错，提示`stride`参数不存在。

+ 报错信息：

```
<ipython-input-7-b3ae5da446df> in main()
     10 
     11     trainer = Trainer(
---> 12         train_func=train_program, place=place, optimizer_func=optimizer_program)
     13 
     14     # Save the parameter into a directory. The Inferencer can load the parameters from it to do infer

/usr/local/lib/python3.5/dist-packages/paddle/fluid/contrib/trainer.py in __init__(self, train_func, optimizer_func, param_path, place, parallel, checkpoint_config)
    257 
    258         with framework.program_guard(self.train_program, self.startup_program):
--> 259             program_func_outs = train_func()
    260             self.train_func_outputs = program_func_outs if isinstance(
    261                 program_func_outs, list) else [program_func_outs]

<ipython-input-6-e0d473e7889c> in train_program()
      5     # predict = softmax_regression() # uncomment for Softmax
      6     # predict = multilayer_perceptron() # uncomment for MLP
----> 7     predict = convolutional_neural_network()  # uncomment for LeNet5
      8 
      9     # Calculate the cost from the prediction and label.

<ipython-input-4-0966b62f60c9> in convolutional_neural_network()
      9         pool_size=2,
     10         pool_stride=2,
---> 11         act="relu")
     12     conv_pool_1 = fluid.layers.batch_norm(conv_pool_1)
     13     # second conv pool

TypeError: simple_img_conv_pool() got an unexpected keyword argument 'stride'
```


+ 问题复现：使用`fluid.nets.simple_img_conv_pool()`定义一个卷积神经网络，并使用`stride`参数设置卷积操作的滑动步长。最后使用这个卷积神经网络进行训练，便出现该问题。错误代码如下：

```
def convolutional_neural_network():
    img = fluid.layers.data(name='img', shape=[1, 28, 28], dtype='float32')
    # first conv pool
    conv_pool_1 = fluid.nets.simple_img_conv_pool(
        input=img,
        filter_size=5,
        num_filters=20,
        stride=1,
        pool_size=2,
        pool_stride=2,
        act="relu")
    conv_pool_1 = fluid.layers.batch_norm(conv_pool_1)
    # second conv pool
    conv_pool_2 = fluid.nets.simple_img_conv_pool(
        input=conv_pool_1,
        filter_size=5,
        num_filters=50,
        stride=1,
        pool_size=2,
        pool_stride=2,
        act="relu")
    prediction = fluid.layers.fc(input=conv_pool_2, size=10, act='softmax')
    return prediction
```


+ 问题解决：错误的原因是`fluid.nets.simple_img_conv_pool()`接口没有`stride`这个参数，如果需要设置卷积操作的滑动步长，可以使用这个`paddle.fluid.layers.conv2d()`接口，这个接口有`stride`参数可以设置卷积操作的步长。

```
def convolutional_neural_network():
    img = fluid.layers.data(name='img', shape=[1, 28, 28], dtype='float32')
    # first conv pool
    conv_pool_1 = fluid.nets.simple_img_conv_pool(
        input=img,
        filter_size=5,
        num_filters=20,
        pool_size=2,
        pool_stride=2,
        act="relu")
    conv_pool_1 = fluid.layers.batch_norm(conv_pool_1)
    # second conv pool
    conv_pool_2 = fluid.nets.simple_img_conv_pool(
        input=conv_pool_1,
        filter_size=5,
        num_filters=50,
        pool_size=2,
        pool_stride=2,
        act="relu")
    prediction = fluid.layers.fc(input=conv_pool_2, size=10, act='softmax')
    return prediction
```

