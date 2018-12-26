# v2迁移到Fluid

## `已审核` 1.问题：如何移植v2模型到fluid?

+ 问题描述：deep fm从paddlepaddle v2 版模型`https://github.com/PaddlePaddle/models/tree/develop/legacy/deep_fm`移植到fluid。但是我找不到移植指导文档。特别是v2中的数据类型如何对应到fluid中：`paddle.data_type.dense_vector`，`paddle.data_type.sparse_binary_vector`，`paddle.data_type.integer_value`

+ 问题解答：
	1.layers.data只需要指定lod_level即可，dense_vector, interge_value默认都可以不用制定lod_level
	2.reader的返回还可以复用v2的reader， dense_vector返回numpy.array

	具体的代码细节可以参考：https://github.com/PaddlePaddle/models/tree/develop/fluid/PaddleRec/ctr

## `已审核`2.问题：Fluid版本中中数据格式与v2版本接口有区别？

+ 问题描述：v2中数据格式是 N C H W kernel参数格式是 Cout Cin H W
tensorflow中数据格式是 N H W C kernel参数格式是 H W Cin Cout
fluid中跟v2区别？

+ 问题解答：

在Fluid版本的PaddlePaddle中可以使用data_layout参数来表示数据的格式，即该参数可以指定tensor的格式，示例写法如下：

```python
hidden1 = fluid.layers.fc(input=x, size=1, param_attr='fc1.w')
hidden2 = fluid.layers.batch_norm(input=hidden1, data_layout='NHWC')
hidden3 = fluid.layers.batch_norm(input=hidden1, data_layout='NCHW')
```

data_layout可以指定两种类型的数据格式，其含义如下：

NHWC [batch，in_height，in_width，in_channels]
NCHW [batch，in_channels，in_height，in_width]