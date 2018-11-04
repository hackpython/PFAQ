# PaddlePaddle集群模式

## 1.问题：关于PaddlePaddle集群模式参数设置的一些疑惑

+ 关键字：`集群模式` `参数设置`

+ 问题描述：
按照cluster_train的步骤跑的集群demo，关于参数设置方面有几个问题，如下：

	1.下面命令里的trainer_count是跟所有机器上的pserver总数一样吗？还有每台机器上的pserver个数跟机器配置有关系吧？
	python paddle.py 
	--job_dispatch_package="${PATH_TO_LOCAL_WORKSPACE}" 
	--dot_period=10 
	--ports_num_for_sparse=2 
	--log_period=50 
	--num_passes=10 
	--trainer_count=4 
	--saving_period=1 
	--local=0 
	--config=./trainer_config.py 
	--save_dir=./output 
	--use_gpu=0

	2.感觉单机和集群模式训练模型使用的时间是一样的，是参数哪里设置的不对吗？

	3.训练完了程序好久没有退出？

	4.pserver的端口数设置主要作用是什么呢？

+ 问题解答：
>1.下面命令里的trainer_count是跟所有机器上的pserver总数一样吗？还有每台机器上的pserver个数跟机器配置有关系吧？

	trainer_count指示一个trainer会起trainer_count个并行线程，共同完成一个mini-batch数据的forward和backward过程，所以这个参数跟pserver数目没有关系。我们提供的简易脚本里，提供N个trainer（可能包含trainer_count个trainer thread）和N个pserver，且一个物理节点对应一个trainer和一个pserver，所有的trainer将利用所有的pserver进行通信，同步参数。

	>2.感觉单机和集群模式训练模型使用的时间是一样的，是参数哪里设置的不对吗？

	根据guide，如果你没有对训练数据进行手动划分，那么所有的节点将分别对所有全量的数据进行处理，所以时间上和单机没有区别，因为你相当于每个节点都对完整数据做了训练（这其实是不对的）。 从你使用的--job_dispatch_package来看，你应该是将本地所有样本直接推送到了所有节点，因此每个节点都重复做了一次完整的训练。 文档中提到这个--job_dispatch_package主要辅助用户完成调试。 实际训练中，需要用户手动划分数据，并配置train.list和test.list，然后利用--job_workspace 完成实际训练。
	关于集群训练，很快我们将推出更加完备的的集群训练方法。

	>3.训练完了程序好久没有退出？

	我们提供的简易脚本，是一个阻塞式的脚本，训练完成后，用户直接ctrl + c就能停止训练，并清除所有节点上训练进程。

	>4.pserver的端口数设置主要作用是什么呢？
	
	优化连接并发度，对于短肥通信管道可以起到提高带宽利用率和pserver计算并发度的作用

## 2.问题：集群训练的时候，每台机器上的训练数据是不同的么？

+ 关键字：`集群训练` `训练数据`

+ 问题描述：在使用PaddlePaddle集群进行训练时，有几个疑惑，如下：

	1.集群训练的时候，每台机器上的训练数据是不同的么？
	比如两台机器参加集群训练，训练数据总共有1000条，测试数据总共有100条。
	那在每台机器上放置500条训练数据和50条测试数据，对么？

	2.Paddle有没有自动分发数据的机制？
	比如，全部1000条训练数据和100条测试数据都在trainer_id=0的机器上，启动trainer后，自动分出500条训练数据和50条测试数据到另一台训练机器上。

	3.测试数据需要分发到集群每台机器上么？

	4.训练完成（所有pass都训练完了）之后，pserver虽然还在后台运行，但是已经不能再次训练是吧？我再次训练的时候，总是报错，需要把pserver都kill掉重启，才能开始新的训练。我想知道这是不是正常的。

+ 问题解答：

	对于问题1、2、3，可以参考上一题的解答

	>训练完成（所有pass都训练完了）之后，pserver虽然还在后台运行，但是已经不能再次训练是吧？

	训练完成之后是要停掉所有Parameter Server的进程的，因为要清空Parameter Server上的参数数据。

