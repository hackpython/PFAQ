# 1.问题：

 - 问题描述：


 - 报错信息：
```
ImportError                               Traceback (most recent call last)
<ipython-input-1-600eb39503dc> in <module>
     16     from paddle.fluid.inferencer import *
     17 
---> 18 from vgg import vgg_bn_drop
     19 from resnet import resnet_cifar10

ImportError: No module named 'vgg'
```

 - 问题复现：
```
from vgg import vgg_bn_drop
```

 - 解决问题：



# 2.问题：

 - 问题描述：


 - 报错信息：
```
<ipython-input-5-fb9e47c67b84> in train(use_cuda, train_program, params_dirname)
     37         num_epochs=EPOCH_NUM,
     38         event_handler=event_handler,
---> 39         feed_order=['pixel', 'label'])

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

EnforceNotMet: Enforce failed. Expected framework::slice_ddim(x_dims, 0, rank - 1) == framework::slice_ddim(label_dims, 0, rank - 1), but received framework::slice_ddim(x_dims, 0, rank - 1):384 != framework::slice_ddim(label_dims, 0, rank - 1):128.
Input(X) and Input(Label) shall have the same shape except the last dimension. at [/paddle/paddle/fluid/operators/cross_entropy_op.cc:37]
PaddlePaddle Call Stacks: 
```

 - 问题复现：
```
def inference_network():
    images = fluid.layers.data(name='pixel', shape=[1, 32, 32], dtype='float32')
    predict = vgg_bn_drop(images)
    return predict
```


 - 解决问题：
```
def inference_network():
    images = fluid.layers.data(name='pixel', shape=[3, 32, 32], dtype='float32')
    predict = vgg_bn_drop(images)
    return predict
```


# 3.问题：

 - 问题描述：


 - 报错信息：
```
/usr/local/lib/python3.5/dist-packages/paddle/fluid/nets.py in img_conv_group(input, conv_num_filter, pool_size, conv_padding, conv_filter_size, conv_act, param_attr, conv_with_batchnorm, conv_batchnorm_drop_rate, pool_stride, pool_type, use_cudnn)
    229             param_attr=param_attr[i],
    230             act=local_conv_act,
--> 231             use_cudnn=use_cudnn)
    232 
    233         if conv_with_batchnorm[i]:

/usr/local/lib/python3.5/dist-packages/paddle/fluid/layers/nn.py in conv2d(input, num_filters, filter_size, stride, padding, dilation, groups, param_attr, bias_attr, use_cudnn, act, name)
   1639             'groups': groups,
   1640             'use_cudnn': use_cudnn,
-> 1641             'use_mkldnn': False
   1642         })
   1643 

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

EnforceNotMet: Conv intput should be 4-D or 5-D tensor. at [/paddle/paddle/fluid/operators/conv_op.cc:47]
PaddlePaddle Call Stacks: 
0       0x7f8683d586b6p paddle::platform::EnforceNotMet::EnforceNotMet(std::__exception_ptr::exception_ptr, char const*, int) + 486
1       0x7f86845cf940p paddle::operators::ConvOp::InferShape(paddle::framework::InferShapeContext*) const + 3440
2       0x7f8683e00f86p paddle::framework::OpDesc::InferShape(paddle::framework::BlockDesc const&) const + 902
```

 - 问题复现：
```
def inference_network():
    images = fluid.layers.data(name='pixel', shape=[3072], dtype='float32')
    predict = vgg_bn_drop(images)
    return predict
```

 - 解决问题：
```
def inference_network():
    images = fluid.layers.data(name='pixel', shape=[3, 32, 32], dtype='float32')
    predict = vgg_bn_drop(images)
    return predict
```

# 4.问题：

 - 问题描述：


 - 报错信息：
```
infer results:  5
```

 - 问题复现：
```
img = load_image('dog.png') 
results = inferencer.infer({'pixel': img})
print("infer results: ", np.argmax(results[0]))
```

 - 解决问题：
```
img = load_image('dog.png')
results = inferencer.infer({'pixel': img})

label_list = [
    "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse",
    "ship", "truck"]
print("infer results: ", label_list[np.argmax(results[0])])
```

# 5.问题：

 - 问题描述：


 - 报错信息：
```
<ipython-input-17-246b35b3c3dc> in infer(use_cuda, inference_program, params_dirname)
     27 
     28     # inference
---> 29     results = inferencer.infer({'pixel': img})
     30 
     31     label_list = [

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

EnforceNotMet: Enforce failed. Expected in_dims[1] == filter_dims[1] * groups, but received in_dims[1]:32 != filter_dims[1] * groups:3.
The number of input channels should be equal to filter channels * groups. at [/paddle/paddle/fluid/operators/conv_op.cc:60]
PaddlePaddle Call Stacks: 
0       0x7ff682f386b6p paddle::platform::EnforceNotMet::EnforceNotMet(std::__exception_ptr::exception_ptr, char const*, int) + 486
1       0x7ff6837b01c6p paddle::operators::ConvOp::InferShape(paddle::framework::InferShapeContext*) const + 5622
```

 - 问题复现：
```
def load_image(file):
    im = Image.open(file)
    im = im.resize((32, 32), Image.ANTIALIAS)
    im = np.array(im).astype(np.float32)
    im = im / 255.0
    im = numpy.expand_dims(im, axis=0)
    return im
```

 - 解决问题：
需要的是`(3, 32, 32)`，但PIL打开方式是`(32, 32, 3)`
```
def load_image(file):
    im = Image.open(file)
    im = im.resize((32, 32), Image.ANTIALIAS)
    im = np.array(im).astype(np.float32)
    im = im.transpose((2, 0, 1))  # CHW
    im = im / 255.0
    im = numpy.expand_dims(im, axis=0)
    return im
```

# 6.问题：

 - 问题描述：


 - 报错信息：
```
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
0       0x7ff682f386b6p paddle::platform::EnforceNotMet::EnforceNotMet(std::__exception_ptr::exception_ptr, char const*, int) + 486
1       0x7ff6837af940p paddle::operators::ConvOp::InferShape(paddle::framework::InferShapeContext*) const + 3440
2       0x7ff683ab514dp paddle::framework::OperatorWithKernel::RunImpl(paddle::framework::Scope const&, boost::variant<paddle::platform::CUDAPlace, paddle::platform::CPUPlace, paddle::platform::CUDAPinnedPlace, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_> const&) const + 77
```

 - 问题复现：
```
def load_image(file):
    im = Image.open(file)
    im = im.resize((32, 32), Image.ANTIALIAS)
    im = np.array(im).astype(np.float32)
    im = im.transpose((2, 0, 1))  # CHW
    im = im / 255.0
    return im
```

 - 解决问题：
`(3, 32, 32)`，应该是`(1, 3, 32, 32)`
```
def load_image(file):
    im = Image.open(file)
    im = im.resize((32, 32), Image.ANTIALIAS)
    im = np.array(im).astype(np.float32)
    im = im.transpose((2, 0, 1))  # CHW
    im = im / 255.0
    im = numpy.expand_dims(im, axis=0)
    return im
```




