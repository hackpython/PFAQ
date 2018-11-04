# Paddle深度讨论

## 1.Paddle 应该是什么形态

### 1.问题
如果要允许用户在iPython和Jupiter之类的界面里写Paddle程序，那么Paddle得是一个library（提供本地函数调用）或者一个RPC server（提供远程函数调用），而不能是目前的executable command line tool的形式。

在library和RPC server之间的选择是个问题.

上述问题实际上是两个问题：

+ 1.API应该是底层（或者说细粒度）的，还是高层（粗粒度）的？
+ 2.API的形式应该是本地调用（library）还是远程调用（RPC）？

当然，如果选择是RPC，那么就应该是粗粒度API了，因为用RPC描述细粒度API的运行时开销太大了。

### 2. 讨论
Tensorflow的API很底层。一方面允许Python程序描述很细节的概念，另一方面让用户学习曲线比较高。很多用户实际上用的是Tensorflow的wrapper Keras。Paddle的API应该是Tensorflow API这个层次的呢，还是Keras这个层次的呢？

感觉先要确定试图达到的目标， 目标是让用户好用，还是让用户灵活使用？ 还是让系统工程师更容易集成PaddlePaddle到不同的平台，抑或是其他目标。

+ 从普通算法用户角度，肯定重点关心好用，对模型的灵活处理能使用他们更快速实现原型，也可以作为高级选项，但是他们对pserver、gpu等怎么使用可能几乎不关心了。
+ 从系统角度，灵活可能更重要，对trainer和pserver层面的抽象api，如果能wrapper出高效的组件那最好，不能完全为了灵活性牺牲性能，毕竟deep learning对速度的追求也是目标之一。

对于paddle是抽象出api，还是抽象出rpc，还是现在的command line进程方式的讨论，关乎到paddlepaddle能潜在的渗透到哪些不同平台，哪些不同领域，这非常赞。目前单一的在mpi、ssh多机、k8s等等有限的场景下，确实还有待进一步开发。

### 3.整体设计code

master.py

```python
# Master是整体全局上的一个注册机制。所有的信息会注册到这个Master里面，master是一个信息同步的节点。
import paddle

# redundancy 是说这个世界中的pserver需要有2个副本。
# 也就是每一个ParameterBlock存放到两个不同的pserver上。默认是1
master = paddle.Master(addr=":8000", redundancy=2) 

# 开始监听，直到退出
master.start_and_join()
```


client.py

```python
import paddle

# Context是使用设备的上下文。Paddle究竟使用多少设备，在这里指定
context = paddle.Context(devices=[paddle.cpu_all, paddle.gpu_all])  # use all device in one node.

# Context可以连接到一个Master。连接到master就是集群训练。否则就是单机训练。
context.connect("master_ip",  role=paddle.WORKER)

# 定义一个网络。前面的注解说明这个函数是一个网络定义。
@context.network()
def simple_network(network):
      # network参数是一个网络定义的函数集合，包括了我们支持的layers
      ipt = network.data_layer(name="input", size=784)
      hidden = network.fc_layer(input=ipt, size=200)
      predict = network.fc_layer(input=hidden, size=10, act=SoftmaxActivation())
      cost = network.classification_cost(input=predict, label=network.data_layer(name="input", size=10))
      return cost  # 返回优化的目标。相当于现在paddle的outputs

# define a data provider, same as current Paddle process.
@paddle.provider()
def process_data(settings, filename):
    for sample in read_from_file(filename):
         yield sample

# train all networks in current context.
context.with_train_data(train_files=["a.txt", "b.txt"], method=process_data)  # set train data, and data provider
     .with_test_data(test_files=["c.txt"], test_period=Batch(1000), method=process_data) # set test data
     .use_optimizer(SgdOptimizer())  # set optimizer.
     .standard_sgd_train(num_passes=100)  # set use standard sgd strategy to train 100 pass.


context.exit(0)  # send exit message to scheduler, to kill all pserver.
```

pserver.py

```python
import paddle
context = paddle.Context(devices=[paddle.cpu_all])  # pserver only use all cpu is enough.

# Hello master, I'm a pserver.
context.connect("master_ip", role=paddle.PSERVER)

# Create a standard pserver.
pserver = context.new_standard_pserver()

# Start and join.
pserver.start_and_join()
```

**实现重点**
这里实现重点在于

Master保存全局的拓扑结构。即知道所有的work是谁，也知道所有的pserver是谁。也知道对于每一个ParamBlock，对应的PServer有哪些
所有的『同步结构』都在Master单节点进行，所有的同步均抽象成Barrier进行。
所有的控制命令，均由work通知master执行。master再去请求pserver执行控制。
所有的梯度推送，参数拉取，均先通过work，查询参数在哪些pserver上，然后work直接和pserver通信。


### 4.Paddle 集群训练重构和API

**重构后可以解决的问题**

+ Paddle可以单机使用Python驱动训练。
	+ Paddle的安装再也不是问题，一个pip install Paddle搞定
+ 用户可以在本地新建集群训练任务。
	+ 只需要在单机代码的基础上，加上一个 connect("master_ip", role=paddle.Client)即可
+ 集群可以支持动态扩容，动态去掉节点，节点漂移等功能
	+ 更好的支持现代集群的管理方式。
	+ 对于百度内部客户也可以尽可能的服务。比如我们可以完全独占一个集群，然后内部分配节点数。
		+ 比如独占了所有集群机器。对于一个用户开始训练，那么我们就占用所有节点训练。对于两个用户，我们就按照优先级重新分配对应比例的节点。
+ 用户可以实时变更运行时的参数，比如学习率。或者实时读取到训练的日志，例如cost，错误率等等。也可以实时更换学习算法。例如先用adam训练，然后用sgd训练。

![](https://camo.githubusercontent.com/8866b856326da2258ccad8cbc6b01d461f43336c/687474703a2f2f626f732e6e6a2e6270632e62616964752e636f6d2f76312f6167726f75702f32336331346337613732623233656138383835373637636438373437633831373761373766613536)

**新的集群训练拓扑结构**
![](https://camo.githubusercontent.com/b0a38ea5977ebd5c307f7f7a92626c81a35e8949/687474703a2f2f626f732e6e6a2e6270632e62616964752e636f6d2f76312f6167726f75702f31333933653634623032643536633734616330336266393438353265663130653364653139623663)

**角色**

新的集群训练的角色共有四种，每一种角色的简单定义如下:

+ Client: 用户端。真正发起训练任务的节点。客户端将训练文件(python)和数据地址传输给Master。并持续和Master通信，查询训练的状态。
+ Master: 集群管理的主节点。主要为集群提供: K-V数据库、每一个ParamBlock的哈希表、全局锁、对PServer的操作
+ PServer: 实际存储每一个ParamBlock。接受Master的操作请求(创建ParamBlock，创建/删除ParamBlock的Buffer，优化某个ParamBlock)
+ Worker: 相当于目前的trainer。训练神经网络，从PServer上请求Value，将Gradient送给PServer，向Master提交PServer的Op

其中，基本的原则是:

+ 所有的Client只与Master通信，控制整体训练的参数(因为Master是KV数据库，所以可以直接修改某些值就好了)
+ 所有的Worker将控制信息都发送给Master，将参数更新推送信息都发送给PServer
+ Worker端控制所有锁变量，这样Master和PServer就可以写的非常非常薄。

**Parameter和Parameter Block**
在说明各个角色的使用方法之前，我们先说明一些Paddle基本的概念。

一个Parameter指的是神经网络某一层的参数和参数同维度的东西。例如，参数值(Value), 参数梯度(Gradient)，参数动量(Momentum)。每一个神经网络中的每一个参数都有一个唯一的名字。

ParameterBlock是指，每一个神经网络的参数被切分成的子块。这些子块均匀分布在各个PServer上，整体构成一个Parameter。每一个ParameterBlock包含Parameter的id，和Parameter的区间段 [0,200], [200, 400]之类的。

PServer能看到的基本数据是ParameterBlock

**Client => Master API**

Client会将模型配置和dataprovider的函数，全部序列字符串送交到Master上执行(可以使用python的pickle包)。序列化上去之后，如果需要哪些第三方库，也可以将依赖的名字使用字符串传递给Master。

Client可以读写Master上的任意KV数据。client通过读写这些KV数据来控制训练进程Master上面会根据客户端传过来的模型配置，dataprovider函数(其实应该是一个main.py)，和依赖脚本，打包成一个Dockerfile，分发给各个worker启动。

```python
FROM paddle:latest
COPY main.py /
COPY requirements.txt / 
COPY packages.txt /
RUN pip install -U /requirements.txt # 安装依赖
RUN apt install -y $(cat packages.txt) # 安装依靠的debian包，这个可以去掉
ENTRYPOINT python /main.py --master_addr="192.168.170.1:54321"  # connect to master.
```

执行流程是:

+ Client => API (我要启动一个任务啦)

	+ 启动Master。
	+ 启动Pserver，PServer全部连接到Master。
	+ 启动Worker，Worker全部连接到Master
	+ 返回Master的IP，还有private key之类的东西
+ Client => Master

	+ Pause任务
	+ Kill任务
	+ 看一看训练日志
	+ 下载当前的参数值。

**Worker => PServer API**

Worker和PServer之间的沟通非常简单。只有两点，

+ push_gradient
+ pull_value
并且这两点完全无锁。也就是只要worker向PServer请求，PServer就把最新的东西给Worker

** Master => PServer API**
+ Master到PServer之间的沟通非常简单, 只有几个操作。

+ block_id = create_param_block(size) 新建一个ParamBlock
+ alloc_buffer(block_id, buffer_id, init=[Random or Zero or Default]) 新建一个buffer(随机初始化，0初始化，不初始化)
+ free_buffer(block_id, buffer_id) 删除一个buffer
+ push_value(block_id) 更新一个block的value
+ pull_value(block_id) 获得一个block的value
+ do_ops(ops)

**PServer => Master API**
+ register 注册一个PServer

**Worker => Master API**

+ register 注册一个Worker
+ get_data_list获得每个pass需要读取的数据列表(这个可以先不实现)
+ create_barrier 创建一个同步barrier
	+ 如果barrier存在，但没有被wait，计数+1
	+ 如果barrier不存在，创建barrier
	+ 如果barrier存在，但是正在被wait。等待这个barrier wait完毕。
+ wait_barrier 等待一个barrier
+ set_value 设置某一个值
+ set_value_ne 当某一个值不存在，设置某一个值。返回true即设置成功
+ get_value获得某一个值
+ wait_value(name, expected_value) 等待某一个值等于expected_value
+ get_pservers_by_param_blocks(param_name, param_offset)获得ParamBlocks对应的Pserver，返回一个dict， block => [PServers]
+ do_ops(ops, lazy_ops, before_wait_bar, after_notify_value) 向pserver执行一系列ops。在执行这些ops之前，等待barrier。在这些ops执行之后，设置after_notify_value=True。 同时记录下来lazy_ops，这些lazy_ops会在获得参数的前再对每个参数执行(主要是稀疏+正则的实现)

**Worker的多机训练逻辑**

```python
master = ...  # master api
master.set_value_ne("num_passes", num_passes)
passes = master.get_value("num_passes")
while passes != 0:
    data = master.get_data_list()
    download_and_remove_data(data)  # get data from hdfs if data in data_list, remove data not in this list.
	data_provider.feed(data)  # data provider use data.
	pass_bar = master.create_barrier("start_pass")
    for each_batch in data_provider:
        master.wait_value("pause", False)  # wait training is not paused. Training will paused when optimizer
                                           # are changed, some pserver is down and we need backup param block to 
                                           # other pserver
                                           
        prefetch_bar = master.create_barrier("prefetch")
        param_blocks = gradient_machine.prefetch(each_batch)  # get all param_blocks used in this batch.
		# Get latest params from pservers.
        blocks_pserver_dict = master.get_pservers_from_param_blocks(param_blocks)
        parallel_for each_block in param_blocks:  # parallel for means we could use multiple thread for it.
	        master.create_barrier("merge_gradient_"+ each_block.id)  # create barrier for merge gradient.
	        pservers = blocks_pserver_dict[each_block]
	        master.wait_value("value_ready_%s"%each_block.id, True)  # wait block value ready.
	        for pserver in pservers:
	          try:
	            pserver.update(each_block)  # get value from pserver for each_block
	            break
	          except: continue
        
        prefetch_bar.wait()  # prefetch should be synced, because here we determined "merge_gradient" 
                             # barrier size.
        
        # local neural network has the lastest value.
        gradient_machine.forward_backward(lambda param: {
		   # backward on each parameter callback.
		   async { # do this job in background
		     parallel_for each_block in param.blocks():
		       for pserver in each_block.pservers():  # each block could associate to many pserver
		         pserver.push_gradient(each_block)  # push gradient, sync call. No matter it is success or not.
		       
		       # all gradient is pushed to pserver, we can reset local gradient here.
		       
		       async { each_block.reset_gradient() }
		       master.do_ops(ops=master.get_value("Optimizer"),  # if all client after push gradient.
		                    before_wait_bar="merge_gradient_%s" % each_block.id,  # wait all gradient pushed.
		                    after_notify_value="value_ready_%s"%each_block.id)  # notify value complete 
		                                                                        # if sgd done.
		  }
		})

        gradient_machine.finish_batch()
    pass_bar.wait()  # sync all worker to each pass end.
```

Worker由于每一个pass都是从master重新获得训练数据列表，每一个batch都是向pserver重新获得训练参数，每一个barrier都是在worker端向master申请，worker退出时，网络连接断开，master可以增加析构掉对应的barrier。避免死锁。

所以Worker可以随便挂。

**Master的实现逻辑**

Master预先启动一堆PServer，然后启动的时候有一定的冗余性，比如同一个ParamBlock存储在两个或者两个以上的PServer上。

当worker断线时候的流程，是:

```python
def on_worker_disconnected():
  barriers = get_barrers_from_worker()
  destroy_barriers(barriers)
```

当PServer断线时候的流程是(可以先不实现):

```python
def on_pserver_disconnected():
  blocks = get_myself_blocks()
  set_value("pause", true)
  remove_myself_in_pserver_hash()
  rerange_blocks_to_pservers()
  set_value("pause", false)
```

当新增PServer的时候(可以先不实现):

```python
def on_new_pserver():
   blocks = get_pserver_heavy_blocks()  # rerange blocks.
   copy_blocks_to_new_pserver()
```

综上，我看了下Master希望存在的功能。我们是不是可以简化成三件事情完成？或者，在这三件事情的框架内，能否完成当前功能？

+ 直接用Redis做K-V database。也就是每个Master同时启动一个Redis，或者类似于Redis的K-V database.

	+ 控制训练流程，PServer的布局，反馈工作进度，都通过向这个K-V 数据库写数据来完成。
		+ 比如，当前的错误率是多少？等等，都写到redis里
		(线下的，批量的学习)比如，每个节点应该训练哪些数据列表。也写到redis里。注意,数据肯定还是存储在hdfs之类的地方的，但是每个节点应该训练的数据列表，还是存在redis里面的。
		+ (online的，实时的数据学习)可以使用redis的pub/sub功能，每一个worker可以监听一个Pub/Sub Channel。当然，这对Paddle也只是一个特殊的文件路径而已。
		+ 例如批量的文件可能是 "hdfs://blah/blah/file"，而实时的文件可能是 "redis://10.0.0.1/pubsub/blah"。之类的。
+ 全局的锁服务。

	+ 这个锁服务是神经网络算法自身需要的东西，将锁服务完全放到一个master节点上，可以让开发变得更简单一些。别的进程挂了，锁服务也可以正确的被释放。 我们也不需要分布式锁，因为其实一个master节点并不会管理非常多的pserver、worker。所以，这个锁应该并不会是性能瓶颈。
+ 启动，杀死 pserver、worker

	+ 对于每一个job，master由另一个进程启动。但是master启动之后，pserver和worker的生死，由master自身掌控。
	+ 或者这一块逻辑也可以完全独立出来，如果中心是一个K-V数据库的话，那么可以监听这个K-V数据库的某一些key，来控制世界的进程数量。

主体的思路是，能用一些标准的K-V数据库解决的工作，就直接引入K-V数据库。然后，将所有操作都和这个数据库紧密相联就好了。将控制，推送数据，汇报进度，都变成读写数据库操作就好了。

### 5.讨论
Paddle的新体系结构里要有一个master process，它负责把一个job拆分成tasks，并且组织workers和pservers来完成这些tasks。这个角色和很多分布式框架（包括MapReduce）类似。

目前Paddle只支持offline learning。所以每个task的很可能是一组minibatches。Paddle master通过把不同的task分给不同的worker processes去做，来实现data parallelism。

未来Paddle应该支持online learning，这样才能支持好reinforcement learning。这样的情况下，master会成为数据流的入口——master负责把流入的数据分组成minibatches，然后分发给workers执行。

除了分发（dispatch）tasks，master还应该是和客户程序交互的接入点——master应该是一个Restful API server，和运行在用户机器上的Paddle程序交互，听从安排，和反馈工作进度以及状态。

以上内容源自：https://github.com/PaddlePaddle/Paddle/issues/594

## 2.优化多机同步SGD训练与pserver设计建议 

### 1. 问题

对于数据并行的分布式同步SGD训练，我们会把每一个mini-batch分成多份，分别交给trainer做训练。假设$z^{(i)}$是第i个trainer的输入，根据 $z^{(i)}$ 我们计算出梯度 $g^{(i)}$ 。pserver收集所有trainer的梯度，并取平均值，如下：

$$\overline g = \frac 1{N} \sum_{i}^N g^{(i)}$$

然后，各个trainer从pserver上拉取$\overline g$，并更新当前trainer的parameter. 在多机训练过程中，机器之前的通信是非常大的瓶颈，主要体现在pserver和trainer之间float类型梯度的传递。如果我们可以对float类型的梯度信息进行转换压缩，可以一定程度上优化整体训练性能。


### 2. 梯度量化

trainer在把梯度交给pserver之间，为了减少传输代价，先将梯度其从float类型量化为三元组{-1， 0， 1}，具体量化公式如下:

$$\widetilde {g_t^k} = quantize(g_t^k) = s_t^k * sign(g_t^k) \circ b_t^k  $$

其中：

$$s_t^k \triangleq \max (abs(g_t^k)) $$

$g^k_t$ 为第t个batch的第k个layer(所有trainer)算出来的梯度; $sign(g_t^k)$和$abs(g_t^k)$分别取$g^k_t$的符号和绝对值。∘符号是Hadamard product运算。 $b^k_t$是一个随机的二进制向量，它的每一个元素服从伯努利分布：

$$P(b_t^k[j] =1 | b_t^k) = \frac {|(g_t^k[j] |} {s_t^k} $$

$$P(b_t^k[j] =0 | b_t^k) = 1 - \frac {|(g_t^k[j] |} {s_t^k} $$

其中，$b^k_t[j]$和$g^k_t[j]$是分别是$b^k_t$和$g^k_t$的第j个元素.

通过等式(1)得到量化后的梯度$\widetilde {g_t^k}$，是一个元素取值范围为 {−1, 0, +1}的向量;

### 3.分析

假设z是一个batch data, w是模型的parameter, loss function为Q(z,w)，那么我们的优化目标就是最小化：

$$C(w)  \triangleq E(Q(z, w)) $$

我们一般按以下方式更新梯度:

$$w_{t+1} = w_t - \eta_t X $$

对于普通方法:

$$X = g_t = \bigtriangledown _wQ(z_t, w_t) $$

对于量化方法：

$$X = \widetilde g_t = s_t * sign(g_t) \circ b_t$$

根据等式(3)(4)有：

$$E(\widetilde g_t) = E(s_t * sign(g_t) \circ b_t) \\ 
= E(s_t * sign(g_t) \circ E(b_t|z_t)) \\
= E(g_t)\\
= E(\bigtriangledown _wQ(z_t, w_t))\\
= \bigtriangledown _wE(Q(z_t, w_t))\\
=  \bigtriangledown _w C(w)$$

所以我们量化后梯度的期望就等于我们要优化目标函数的对w的微分。

### 4. 优化效果

通过等式(1)得到量化后的梯度$\widetilde {g_t^k}$，是一个元素取值范围为 {−1, 0, +1}的向量，其中每个元素对应原来的一个float类型的梯度。 也就是我们可以用两个bit来编码表示一个梯度了，比如00表示-1， 01表示0， 10表示1，从4个字节缩减到2个bit，网络传输数据量减少为原来的1/16。

**5. paddle pserver的限制**
如上节分析，我们可以将4个梯度的量化结果编码到一个uint8_t中，然后pserver收集并计算以uint8_t类型存储的梯度信息。 而且，在量化各个trainer的梯度前，我们还要根据等式(2)收集所有trainer上的绝对值最大梯度。 但是，当前parameter server只能收集trainer的weight和gradients。 在设计实现新版pserver时，我们应该考虑到上述需求，比如实现如下功能

```python
pclient.init_parameter(key="max_abs_grad", shape=[1], type="float32") // 初始化参数
pclient.push_parameter(key="max_abs_grad", value=0.5) // 传本地parameter到pserver
pclient.pull_parameter(key="max_abs_grad", reduce_method="max") // 拉取reduce之后的parameters

pclient.init_parameter(key="op_id_grads", shape=[2,2], type="uint8")
pclient.push_parameter(key="op_id_grad", value=[[1,5],[8,2]])
pclient.pull_parameter(key="op_id_grad", reduce_method="decode_average")
```

以上内容源自：`https://github.com/PaddlePaddle/Paddle/issues/6599`



