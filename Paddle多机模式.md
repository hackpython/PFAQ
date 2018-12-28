# Paddle多机模式

## `待审核`1.问题：Fluid版本的Paddle中，如何实现多机的模型重加载恢复训练？

+ 问题描述：使用Fluid分布式的训练模型时，如何保存已训练的模型？如何加载已训练的模型？

+ 问题分析：

Checkpoint功能能够在训练中途对训练数据中间数据进行保存，出现异常恢复训练的时候能够加载中途保存的数据继续训练， 实现单机/多机的容错训练的功能。

在Fluid.Trainer对象的声明中加入Fluid.CheckpointConfig的声明，Trainer的init方法包含CheckpointConfig参数， 需要传入在声明Trainer前声明的CheckpointConfig对。

分布式训练的过程中：每个Trainer都会在checkpoint_dir目录中保存当前Trainer的参数（只有Trainer 0会保存模型的参数），需要分布式文件系统(HDFS等)将同checkpoint_dir目录的数据进行合并才能得到完整的数据，恢复训练的时候需要用完整的数据进行恢复。

+ 解决方法：

```python
config = CheckpointConfig(
    checkpoint_dir = "/tmp/ckpt", max_num_checkpoints = 2, 
    epoch_interval = 2, step_interval = 10)
trainer = Trainer(..., checkpoint_config=config)
```

更多细节，参考：http://staging.paddlepaddle.org/documentation/docs/zh/0.14.0/new_docs/user_guides/howto/training/checkpoint_doc_cn.html


## `待审核`2.问题：Fluid版的Paddle进行多机训练时，每个机器上的参数是否相同？

+ 问题描述：在使用Fluid版的Paddle进行多机训练时，存在一个疑惑，即每个机器上的参数是相同的吗？

+ 问题分析：
分布式的模型训练又两种训练模式，分别是数据并行训练模式与模型并行训练模式，目前1.2版的Fluid只支持数据并行模型，模型并行将在后续版本中集成

+ 问题解答：

在数据并行模式的训练中，Fluid使用了两种通信模式，用于应对不同训练任务对分布式训练的要求，分别为RPC通信和Collective 通信。

使用RPC通信方式的数据并行分布式训练，会启动多个pserver进程和多个trainer进程，每个pserver进程会保存一部分模型参数，并负责接收从trainer发送的梯度并更新这些模型参数；每个trainer进程会保存一份完整的模型，并使用一部分数据进行训练，然后向pserver发送梯度，最后从pserver拉取更新后的参数。

使用NCCL2（Collective通信方式）进行分布式训练，是不需要启动pserver进程的，每个trainer进程都保存一份完整的模型参数，在完成计算梯度之后通过trainer之间的相互通信，Reduce梯度数据到所有节点的所有设备然后每个节点在各自完成参数更新。

更多细节，参考：http://www.paddlepaddle.org/documentation/docs/zh/1.2/user_guides/howto/training/cluster_howto.html

## `待审核`3.问题：分布时训练时，PaddlePaddle是否支持多机开启内存优化？

+ 问题描述：在多机上使用PaddlePaddle进行训练时，如何开启多机开启内存优化？Paddle是否支持该功能？

+ 问题解答：
Paddle支持多机内存优化，但与常见的单机内容优化有所不同，需要注意如下规则：

1.在pserver端，不要执行 memory_optimize
2.在trainer端，先执行 fluid.memory_optimize 再执行 t.transpile()
3.在trainer端，调用 memory_optimize 需要增加 skip_grads=True 确保发送的梯度不会被重命名： fluid.memory_optimize(input_program, skip_grads=True)

+ 解决方法：

实例代码如下：

```
if role == "TRAINER":
    fluid.memory_optimize(fluid.default_main_program(), skip_grads=True)
t = fluid.DistributeTranspiler()
t.transpile(trainer_id, pservers=pserver_endpoints, trainers=trainers)
if role == "PSERVER":
    # start pserver here
elif role == "TRAINER":
    # start trainer here
```


## `待审核`4.问题：Fluid如何实现分布式网络架构？

+ 问题描述：目前有多个物理主机，现在想通过Fluid来构建一个分布式的训练网络，如何实现？

+ 问题分析：可以使用`paddle.fluid.Distribut`类，该类可以把fluid program转变为分布式数据并行计算程序（distributed data-parallelism programs）,可以有Pserver和NCCL2两种模式。


+ 解决方法：

`paddle.fluid.Distribut`类pserver模式的实例代码如下：

```
#pserver模式下
pserver_endpoints = "192.168.0.1:6174,192.168.0.2:6174"
trainer_endpoints = "192.168.0.1:6174,192.168.0.2:6174"
current_endpoint = "192.168.0.1:6174"
trainer_id = 0
trainers = 4
role = os.getenv("PADDLE_TRAINING_ROLE")

t = fluid.DistributeTranspiler()
t.transpile(
     trainer_id, pservers=pserver_endpoints, trainers=trainers)
if role == "PSERVER":
     pserver_program = t.get_pserver_program(current_endpoint)
     pserver_startup_program = t.get_startup_program(current_endpoint,
                                             pserver_program)
elif role == "TRAINER":
     trainer_program = t.get_trainer_program()
```

关于该类更多的内容，可以参考：http://www.paddlepaddle.org/documentation/api/en/0.15.0/transpiler.html#distributetranspiler

+ 问题拓展：

分布式训练深度学习模型在技术实现上是存在挑战的，其抽象整体结构如下图：

![](https://raw.githubusercontent.com/ayuLiao/images/master/distributedeeplearning.png)

其实设计了很多技术细节，可以参考下面文章：

http://joerihermans.com/ramblings/distributed-deep-learning-part-1-an-introduction/



## `待审核`5.问题：Fluid版Paddle多机训练时，batch_size大小实际是多少？

+ 问题描述：使用Fluid版的Paddle多机训练时，设置的batch_size与实际训练时使用batch_size是否不同？实际使用的batch_size是多少？

+ 问题分析：Fluid在进行多机训练时，会将数据根据相应的规则传递到不同的机器上，然后再进行训练。

+ 问题解答：
多机训练时，即Paddle的集群模式下，实际使用的batch_size = 配置中的batch_size * 结点数目，一个具体的例子：
现在有10个节点使用Fluid进行训练，配置的batch_size为128，那么实际训练时使用的batch_size为128\*10=1280


## `待审核`6.问题：如何将参数分别传递给不同的机器设备进行训练？

+ 问题描述：Fluid版如果实现将数据指定给不同的设备，因为每个设备强弱不同，想将较多的数据分配给算力较强的设备进行模型的训练。

+ 问题分析：

其实多机训练时数据的传递与单机时是类似的，在单机情况，数据传递到模型的形式如下：

```python
exe = fluid.Executor(cpu)

exe.run(fluid.default_startup_program())

outs = exe.run(
        feed = {'x': train_data, 'y': y_true},
        fetch_list=[y_predict.name, avg_cost.name]
    )
```

+ 解决方法：

形式与单机时传参类似，只是不再使用Executor而是使用ParallelExecutor来进行训练，同样通过feed传递参数，形式如下：

```python
import paddle.fluid as fluid
import numpy

parallel_executor = fluid.ParallelExecutor()
parallel_executor.run(
  feed=[
     {
       "image": numpy.random.random(size=(32, 3, 224, 224)).astype('float32'),
       "label": numpy.random.random(size=(32, 1)).astype('int64')
     },
     {
       "image": numpy.random.random(size=(16, 3, 224, 224)).astype('float32'),
       "label": numpy.random.random(size=(16, 1)).astype('int64')
     },
  ]
)
```

在上面的代码中，会分别将数据传递到不同的GPU上，GPU 0会训练32为一batch的样本，而GPU 1只会训练16为一batch的样本。

## `待审核`7.问题：fluid启动pserver时卡住了，无法进行执行后面的逻辑。

+ 问题描述：自建k8s, 使用paddle镜像：1.0.1-gpu-cuda8.0-cudnn7
启动pserver相关代码：

```python
if training_role == "PSERVER":
        place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    if is_local:
        train_loop(fluid.default_main_program())
    else:
        eplist = []
        for ip in pserver_ips.split(","):
            eplist.append(':'.join([ip, port]))
        pserver_endpoints = ",".join(eplist)  # ip:port,ip:port...
        print(pserver_endpoints)
        print(trainers)
        print(trainer_id)
        t = fluid.DistributeTranspiler()
        t.transpile(
            trainer_id,
            pservers=pserver_endpoints,
            trainers=trainers)

        if training_role == "PSERVER":
            print("pserver")
            print("current_endpoint: %s" % current_endpoint)
            pserver_prog = t.get_pserver_program(current_endpoint)
            pserver_startup = t.get_startup_program(current_endpoint,
                                                    pserver_prog)
            print("start pserver...")
            exe.run(pserver_startup)
            print("pserver start up ok: %s" % pserver_startup)
            exe.run(pserver_prog)
            print("pserver ended")
        elif training_role == "TRAINER":
            print("trainer")
            train_loop(t.get_trainer_program())
```

输出日志中发现输出了start pserver...之后再无新日志输出，而trainer也一直在等待pserver启动成功, 不停的输出“pserver not ready, wait 3 sec to retry...”，一段时间后trainer也失败退出。

+ 问题分析：

`pserver not ready, wait 3 sec to retry...`的报错输出说明是通信问题，即还没有进入Paddle Fluid的执行逻辑就已经报错了。

+ 解决方法：
从相关代码中可以看出，使用了Executor，但在多机模式下，要使用pserver需要通过ParallelExecutor来使用，示例代码如下：

```python
 if args.update_method == "pserver":
        # parameter server mode distributed training, merge
        # gradients on local server, do not initialize
        # ParallelExecutor with multi server all-reduce mode.
        num_trainers = 1
        trainer_id = 0

    exe = fluid.ParallelExecutor(
        True,
        avg_loss.name,
        main_program=train_prog,
        exec_strategy=strategy,
        build_strategy=build_strategy,
        num_trainers=num_trainers,
        trainer_id=trainer_id)
```

更多细节可以参考：https://github.com/PaddlePaddle/models/blob/develop/fluid/PaddleCV/image_classification/dist_train/dist_train.py#L326











