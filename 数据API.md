# 数据API

## `待审核`1.paddle_batch不能保证数据每次输出的问题？

+ 版本号：`1.1.0`

+ 标签：`数据输出`

+ 问题描述：paddle_batch不能保证数据每次输出的问题

+ 相关代码：
```
import paddle
import paddle.fluid as fluid
import numpy as np
import math
import random
import sys
import os

cluster_train_dir = "./tra"
cluster_test_dir = "./tes"

dict_size = 30
emb_dim = 300
fc_dim = 128
margin = 0.5
PASS_NUM = 100
TRAIN_BATCH_SIZE = 1
TEST_BATCH_SIZE = 1


def cluster_data_reader(file_dir):
    """
    cluster data reader
    """
    def data_reader():
        """
        data reader
        """
        files = os.listdir(file_dir)
        for fi in files:
            with open(file_dir + '/' + fi, "r") as f:
                for line in f:
                    line = line.strip()
                    ls = line.split(";")
                    pos_num = int(ls[1].split(" ")[0])
                    neg_num = int(ls[1].split(" ")[1])
                    if neg_num == 0 or pos_num == 0:
                        continue
                    if neg_num > 500:
                        neg_num = 500
                    for pos in range(3, 3+pos_num):
                        for neg in range(3+pos_num, 3+pos_num+neg_num):
                            r = random.random()
                            if r < 0.8:
                    	        query = np.fromstring(ls[2], dtype=np.int64, sep=" ")
                    	        pos_title = np.fromstring(ls[pos], dtype=np.int64, sep=" ")
                    	        neg_title = np.fromstring(ls[neg], dtype=np.int64, sep=" ")
                    	        yield query, pos_title, neg_title
    return data_reader

def train(use_cuda, save_dirname, is_local):
    """
    train
    """
    query = fluid.layers.data(name = "Query", shape = [1], dtype="int64", lod_level=1)
    pos_ans = fluid.layers.data(name = "Pos", shape = [1], dtype="int64", lod_level=1)
    neg_ans = fluid.layers.data(name = "Neg", shape = [1], dtype="int64", lod_level=1)

    query_emb = fluid.layers.embedding(input=query, size=[dict_size, emb_dim], is_sparse=True, is_distributed=True, param_attr=fluid.param_attr.ParamAttr(name="emb"))
    pos_emb = fluid.layers.embedding(input=pos_ans, size=[dict_size, emb_dim], is_sparse=True, is_distributed=True, param_attr=fluid.param_attr.ParamAttr(name="emb"))
    neg_emb = fluid.layers.embedding(input=neg_ans, size=[dict_size, emb_dim], is_sparse=True, is_distributed=True, param_attr=fluid.param_attr.ParamAttr(name="emb"))

    query_sum = fluid.layers.sequence_pool(input=query_emb, pool_type="sum")
    pos_sum = fluid.layers.sequence_pool(input=pos_emb, pool_type="sum")
    neg_sum = fluid.layers.sequence_pool(input=neg_emb, pool_type="sum")

    query_softsign = fluid.layers.softsign(query_sum)
    pos_softsign = fluid.layers.softsign(pos_sum)
    neg_softsign = fluid.layers.softsign(neg_sum)

    query_fc = fluid.layers.fc(input=query_softsign, size=fc_dim, param_attr=fluid.param_attr.ParamAttr(name="fc1.w"), \
            bias_attr=fluid.param_attr.ParamAttr(name="fc1.b"), name="fc1")
    query_fc_relu = fluid.layers.relu(query_fc)
    
    pos_fc = fluid.layers.fc(input=pos_softsign, size=fc_dim, param_attr=fluid.param_attr.ParamAttr(name="fc1.w"), \
            bias_attr=fluid.param_attr.ParamAttr(name="fc1.b"), name="fc1")
    pos_fc_relu = fluid.layers.relu(pos_fc)
    
    neg_fc = fluid.layers.fc(input=neg_softsign, size=fc_dim, param_attr=fluid.param_attr.ParamAttr(name="fc1.w"), \
            bias_attr=fluid.param_attr.ParamAttr(name="fc1.b"), name="fc1")
    neg_fc_relu = fluid.layers.relu(neg_fc)
    
    q_p_cosine = fluid.layers.cos_sim(query_fc_relu, pos_fc_relu)
    q_n_cosine = fluid.layers.cos_sim(query_fc_relu, neg_fc_relu)
    q_p_nan = fluid.layers.has_nan(q_p_cosine)
    q_n_nan = fluid.layers.has_nan(q_n_cosine)
    zero_constant = fluid.layers.fill_constant_batch_size_like(q_n_cosine, q_n_cosine.shape, "float32", 0.0)
    margin_constant = fluid.layers.fill_constant_batch_size_like(q_n_cosine, q_n_cosine.shape, "float32", margin)
    sub = fluid.layers.elementwise_sub(q_n_cosine, q_p_cosine)
    add = fluid.layers.elementwise_add(sub, margin_constant) # q_n_cosine - q_p_cosine + margin
    hinge_loss = fluid.layers.elementwise_max(zero_constant, add) # max(0, q_n_cosine - q_p_cosine + margin)
    hinge_loss_nan = fluid.layers.has_nan(hinge_loss)
    avg_cost = fluid.layers.reduce_mean(hinge_loss) # average cost
    avg_cost_nan = fluid.layers.has_nan(avg_cost)
    inference_program = fluid.default_main_program().clone(for_test=True)
    #inference_program = fluid.io.get_inference_program([avg_cost], fluid.default_main_program())
    sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.001)
    sgd_optimizer.minimize(avg_cost)

    train_reader = paddle.batch(
        paddle.reader.shuffle(
            cluster_data_reader(cluster_train_dir), buf_size=500),
        batch_size=TRAIN_BATCH_SIZE)

    test_reader = paddle.batch(
        paddle.reader.shuffle(
            cluster_data_reader(cluster_test_dir), buf_size=500),
        batch_size=TEST_BATCH_SIZE)

    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    training_role = os.getenv("TRAINING_ROLE", "TRAINER")
    if training_role == "PSERVER":
        place = fluid.CPUPlace()
    exe = fluid.Executor(place)

    def train_loop(main_program):
        """
        train_loop
        """
        feeder = fluid.DataFeeder(place=place, feed_list=[query, pos_ans, neg_ans])
        exe.run(fluid.default_startup_program())

        for pass_id in range(PASS_NUM):
            train_losses = []
            total_loss = 0.0
            print "start %s" % pass_id
            for iter_id, data in enumerate(train_reader()):
                avg_loss_value = exe.run(main_program, feed=feeder.feed(data), fetch_list=[avg_cost, q_p_nan, q_n_nan, hinge_loss_nan, avg_cost_nan])
                total_loss += avg_loss_value[0]
                print "data  %s" % data
                print "q_p_nan  %s" % avg_loss_value[1]
                print "q_n_nan  %s" % avg_loss_value[2]
                print "hinge_loss_nan  %s" % avg_loss_value[3]
                print "avg_loss_nan  %s" % avg_loss_value[4] 
                
                if (iter_id + 1) % 100 == 0:
                    print "epoch: %d, iter: %d, loss: %f" % (pass_id, iter_id, total_loss / 100)
                    total_loss = 0.0
                train_losses.append(avg_loss_value[0])
            print "epoch: %d, train_loss: %f" % (pass_id, np.mean(train_losses))
            test_losses = []
            for iter_id, data in enumerate(test_reader()):
                test_cost = exe.run(inference_program, feed=feeder.feed(data), fetch_list=[avg_cost])
                test_losses.append(test_cost[0])
            print "epoch: %d, valid_loss: %f" % (pass_id, np.mean(test_losses))
            save_dirname_epochid = "%s%s" % (save_dirname, pass_id)
            if save_dirname_epochid is not None:
                fluid.io.save_inference_model(save_dirname_epochid, ['Query'], [query_fc], exe)
                
    if is_local:
        train_loop(fluid.default_main_program())
    else:
        port = os.getenv("PADDLE_PORT", "6174")
        pserver_ips = os.getenv("PADDLE_PSERVERS")  # ip,ip...
        eplist = []
        for ip in pserver_ips.split(","):
            eplist.append(':'.join([ip, port]))
        pserver_endpoints = ",".join(eplist)  # ip:port,ip:port...
        trainers = int(os.getenv("PADDLE_TRAINERS_NUM", "0"))
        current_endpoint = os.getenv("POD_IP") + ":" + port
        trainer_id = int(os.getenv("PADDLE_TRAINER_ID", "0"))
        config = fluid.DistributeTranspilerConfig()
        config.slice_var_up = False
        t = fluid.DistributeTranspiler(config=config)
        t.transpile(trainer_id, pservers=pserver_endpoints, trainers=trainers)
        if training_role == "PSERVER":
            pserver_prog = t.get_pserver_program(current_endpoint)
            pserver_startup = t.get_startup_program(current_endpoint, pserver_prog)
            exe.run(pserver_startup)
            exe.run(pserver_prog)
        elif training_role == "TRAINER":
            train_loop(t.get_trainer_program())


def main(use_cuda, is_local=True):
    """
    main
    """
    if use_cuda and not fluid.core.is_compiled_with_cuda():
        return
    save_dirname = "./output/model/"
    if not os.path.isdir(save_dirname):
        os.makedirs(save_dirname)
    train(use_cuda, save_dirname, is_local)


if __name__ == '__main__':
    use_cuda = False
    is_local = True
    main(use_cuda, is_local)
```

+ 代码输出：

输入数据：只有一行：1 1;1 2;1 2 3 4 5 6 7;1 2 3 4 5 6 7;1 3 2 1 1 1;4 5 2 1 3 4
迭代过程中发现有的轮次没有输出数据：

```
epoch: 1, train_loss: 0.039884
epoch: 1, valid_loss: 0.296880
start 2
data  [(array([1, 2, 3, 4, 5, 6, 7]), array([1, 2, 3, 4, 5, 6, 7]), array([1, 3, 2, 1, 1, 1]))]
q_p_nan  [False]
q_n_nan  [False]
hinge_loss_nan  [False]
avg_loss_nan  [False]
data  [(array([1, 2, 3, 4, 5, 6, 7]), array([1, 2, 3, 4, 5, 6, 7]), array([4, 5, 2, 1, 3, 4]))]
q_p_nan  [False]
q_n_nan  [False]
hinge_loss_nan  [False]
avg_loss_nan  [False]
epoch: 2, train_loss: 0.186649
epoch: 2, valid_loss: 0.285172
start 3
epoch: 3, train_loss: nan
epoch: 3, valid_loss: 0.333816
```

+ 解决方法：

```
r = random.random()
if r < 0.8:
    query = np.fromstring(ls[2], dtype=np.int64, sep=" ")
    pos_title = np.fromstring(ls[pos], dtype=np.int64, sep=" ")
    neg_title = np.fromstring(ls[neg], dtype=np.int64, sep=" ")
    yield query, pos_title, neg_title
```

从源码看，r = random.random()有随机性，只有小于0.8的情况下，才会返回数据。
所以出现有时候没有数据的问题，可以去掉 if r < 0.8，再试下。




## `待审核`2.问题：py_func接口当遇到输出值维度会变化的如何处理

+ 版本号：`1.2.0`

+ 标签：`py_func` `数据处理`

+ 问题描述：如何输出的cond是可变的[2, ?]，这种情况怎么处理？我尝试使用reshape，会得到`AssertionError: Only one dimension in shape can be unknown.`错误

+ 报错输出：

```
Traceback (most recent call last):
  File "/media/test/5C283BCA283BA1C6/yeyupiaoling/PyCharm/PaddlePaddle-MTCNN/train_PNet/train_PNet.py", line 14, in <module>
    image, label, bbox_target, landmark_target, label_cost, bbox_loss, landmark_loss, accuracy, conv4_1, conv4_2, conv4_3 = P_Net()
  File "/media/test/5C283BCA283BA1C6/yeyupiaoling/PyCharm/PaddlePaddle-MTCNN/train_PNet/model.py", line 75, in P_Net
    label_cost = cls_ohem(cls_prob=cls_prob, label=label)
  File "/media/test/5C283BCA283BA1C6/yeyupiaoling/PyCharm/PaddlePaddle-MTCNN/train_PNet/model.py", line 282, in cls_ohem
    cls_prob_reshpae = fluid.layers.reshape(num_cls_prob, [num_cls_prob, -1])
  File "/home/test/PaddlePaddle_Python3.5/lib/python3.5/site-packages/paddle/fluid/layers/nn.py", line 5979, in reshape
    "Only one dimension in shape can be unknown.")
AssertionError: Only one dimension in shape can be unknown.
```

+ 相关代码：

```
    def my_size(x):
        num_cls_prob = np.size(x)
        return num_cls_prob

    num_cls_prob = create_tmp_var(name='num_cls_prob', dtype='int32', shape=[1])
    num_cls_prob = fluid.layers.py_func(func=my_size, x=cls_prob, out=num_cls_prob)
    cls_prob_reshpae = fluid.layers.reshape(num_cls_prob, [num_cls_prob, -1])
    print(cls_prob_reshpae)
```

+ 解决方法：

根据py_func文档的解释，在使用这个接口时是需要指定输出的shape和data type的，但不意味着这个函数在构建模型时就已经被执行。如果你只是希望将可变长shape从[-1, 2]变成[-1, 1]，可以尝试试，

```
import paddle.fluid as fluid
import numpy as np

def create_tmp_var(name, dtype, shape):
       return fluid.default_main_program().current_block().create_var(
           name=name, dtype=dtype, shape=shape)

def my_size(x):
        num_cls_prob = np.size(np.array(x))
        cls_prob_reshpae = np.reshape(np.array(x), [num_cls_prob, -1])
        return cls_prob_reshpae

cls_prob = create_tmp_var(name='num_cls', dtype='int64', shape=[-1, 2])
cls_prob_reshpae = create_tmp_var(name='num_cls_prob', dtype='int64', shape=[-1, 1])
cls_prob_reshpae = fluid.layers.py_func(func=my_size, x=cls_prob, out=cls_prob_reshpae)
print(cls_prob_reshpae)
```

这样print的cls_prob_reshpae如下：

```
name: "num_cls_prob"
type {
  type: LOD_TENSOR
  lod_tensor {
    tensor {
      data_type: INT64
      dims: -1
      dims: 1
    }
  }
}
```

## `待审核`3.问题：如何只取loss的70%的数据呢？


+ 版本号：`1.2.0`

+ 标签：`MTCNN` `loss`

+ 问题描述：

我通过以下得到了一个损失函数，但是我只想取前70%的数据（MTCNN模型要求的），那么该如果做呢？

loss = fluid.layers.cross_entropy(input=cls_prob, label=label_filter_invalid)

+ 报错输出：

```
  File "/media/test/5C283BCA283BA1C6/yeyupiaoling/PyCharm/TestMTCNN/train/model.py", line 108, in cls_ohem
    loss, _ = fluid.layers.topk(input=loss, k=22)
  File "/media/test/5C283BCA283BA1C6/yeyupiaoling/PyCharm/TestMTCNN/train/model.py", line 75, in P_Net
    label_cost = cls_ohem(cls_prob=cls_prob, label=label)
  File "/media/test/5C283BCA283BA1C6/yeyupiaoling/PyCharm/TestMTCNN/train/train_PNet.py", line 14, in <module>
    image, label, bbox_target, landmark_target, label_cost, bbox_loss, landmark_loss, conv4_1, conv4_2, conv4_3 = P_Net()
C++ Callstacks: 
Enforce failed. Expected input_dims[input_dims.size() - 1] >= k, but received input_dims[input_dims.size() - 1]:1 < k:22.
input must have >= k columns at [/paddle/paddle/fluid/operators/top_k_op.cc:40]
PaddlePaddle Call Stacks: 
```

+ 相关代码：

```
    loss = fluid.layers.cross_entropy(input=cls_prob, label=label_filter_invalid)
    # 选取70%的数据, batch_size * 0.7
    loss, _ = fluid.layers.topk(input=loss, k=22)
    return fluid.layers.reduce_sum(loss)
```

+ 解决方法：

```
Enforce failed. Expected input_dims[input_dims.size() - 1] >= k, but received input_dims[input_dims.size() - 1]:1 < k:22.
```

该报错通常是维度不匹配造成的错误，坚持loss对应的维度，确保维度匹配。






















