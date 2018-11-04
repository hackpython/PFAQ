# 个性化推荐

## 背景介绍

在网络技术不断发展和电子商务规模不断扩大的背景下，商品数量和种类快速增长，用户需要花费大量时间才能找到自己想买的商品，这就是信息超载问题。为了解决这个难题，推荐系统（Recommender System）应运而生。

个性化推荐系统是信息过滤系统（Information Filtering System）的子集，它可以用在很多领域，如电影、音乐、电商和 Feed 流推荐等。推荐系统通过分析、挖掘用户行为，发现用户的个性化需求与兴趣特点，将用户可能感兴趣的信息或商品推荐给用户。与搜索引擎不同，推荐系统不需要用户准确地描述出自己的需求，而是根据分析历史行为建模，主动提供满足用户兴趣和需求的信息


深度学习具有优秀的自动提取特征的能力，能够学习多层次的抽象特征表示，并对异质或跨域的内容信息进行学习，可以一定程度上处理推荐系统冷启动问题。

## 个性化推进系统PaddlePaddle-fluid版代码：
https://github.com/PaddlePaddle/book/tree/develop/05.recommender_system


## 1.模型训练随机出core的问题

+ 关键字：`推荐系统` `参数随机`

+ 问题描述：我使用PaddlePaddle构建了一个模型，模型的结构与官方文档中推荐系统的的模型结构类似，该模型训练到不同的pass之后随机报错退出,请问训练过程中随机退出都可能有什么原因呢？这个报错与train.sh的参数设置有关系么？

+ 报错输出：

```bash
I0728 00:45:40.022342  1868 Stat.cpp:132] Stat=BackwardTimer                  TID=1875   total=101.164    avg=0.046      max=0.461      min=0          count=2160      
                                          Stat=BackwardTimer                  TID=1876   total=105.213    avg=0.048      max=0.739      min=0          count=2160      
                                          Stat=BackwardTimer                  TID=1871   total=110.272    avg=0.051      max=0.553      min=0          count=2160      
                                          Stat=BackwardTimer                  TID=1877   total=110.925    avg=0.051      max=0.546      min=0          count=2160      
                                          Stat=BackwardTimer                  TID=1872   total=105.79     avg=0.048      max=0.673      min=0          count=2160      
                                          Stat=BackwardTimer                  TID=1874   total=107.7      avg=0.049      max=0.536      min=0          count=2160      
                                          Stat=BackwardTimer                  TID=1873   total=104.855    avg=0.048      max=0.727      min=0          count=2160      
I0728 00:45:40.022374  1868 Stat.cpp:140] ======= BarrierStatSet status ======
I0728 00:45:40.022379  1868 Stat.cpp:153] --------------------------------------------------
I0728 00:46:22.938666  1868 Tester.cpp:127]  Test samples=886436 cost=0.619395 Eval: 
I0728 00:46:22.953542  1868 GradientMachine.cpp:112] Saving parameters to ./output2/pass-00030
I0728 00:46:23.030747  1868 Util.cpp:230] copy trainer_config_feed.py to ./output2/pass-00030
*** Aborted at 1501173992 (unix time) try "date -d @1501173992" if you are using GNU date ***
PC: @     0x7fe550fdac98 (unknown)
*** SIGSEGV (@0x0) received by PID 1868 (TID 0x7fe507fff700) from PID 0; stack trace: ***
    @     0x7fe55123e160 (unknown)
    @     0x7fe550fdac98 (unknown)
/home/iknow/lianjie/paddle/paddle_internal_release_tools/idl/paddle/output/bin/paddle_local: line 109:  1868 Segmentation fault      (core dumped) ${DEBUGGER} $MYDIR/../opt/paddle/bin/paddle_trainer ${@:2}
```

+ 相关代码：

```python
cfg=trainer_config_feed.py

paddle train \
  --config=$cfg \
  --save_dir=./output2 \
  --trainer_count=7 \
  --log_period=2000 \
  --num_passes=150 \
  --use_gpu=false \
  --show_parameter_stats_period=500 \
  --test_all_data_in_one_period=1 \
  --dot_period=30 \
  --saving_period=1 \
  --num_gradient_servers=1 \
  2>&1 | tee 'train.log.2'
```


+ 解决方法：
从报错信息来看像是遇到了计算异常，但从中看不出来是哪一层出现了计算异常。最好能固定数据（去掉一切随机性）稳定复杂，定位到具体的计算步骤，现在报错输出给出的信息比较少。可以使用paddle train的init_model_path启动参数，该参数可以用来指定初始化模型的若干参数。

+ 问题研究：

    + 1.如果确实需要彻底追查计算异常，必须要去掉随机性，稳定复现。否则很难找到精确的原因。
    + 2.计算异常和网络处理样本顺序会有关系。
    	+ 第一种情况：A 样本处理完毕会更新网络，这时候再计算B样本；第二种情况：B样本处理完更新网络，再计算A样本。（1）和 （2）效果是不同的。
    	+ 第一情况失败了，而第二种情况没有失败的可能性是存在的，如果想精确找到原因，还是尽量稳定的复现，否则只能靠不全的信息去猜测推理，无法从根本的定位问题，就难以从根本上解决问题
    	+ 数值异常往往是因为：（1）输入数据异常；或者（2）神经网络参数设置不当，而引起的。如果想彻底解决，请在这两个方向上逐一确定，靠随机性去避过问题，下次可能还会遇。
    + 3.从 init_model_path 指定的模型加载训练好的参数，初始化网络不会有随机初始化参数的过程。而 seed 用来设置随机种子，随机初始化要学习的参数。



## 2.关于推荐双塔模型输出的问题

+ 关键字：`双塔模型` `推荐系统`

+ 问题描述：
我使用PaddlePaddle构建的双塔模型，如下图：
![](https://raw.githubusercontent.com/ayuLiao/images/master/Jietu20181020-153224.jpg)
实际使用的时候，希望输出用户向量，即L2 物品向量，R2因为模型训练的时候是先读入的用户数据，再读入的物品数据。所以如果想获得L2输出时，模型仅仅加载L1数据即可，但是如果要获得R2的输出，只加载物品的网络结构就报错了，必须先加载L1的网络结构（可以没有数据输入），再加在R1才能成功。请问必须要这样么？或者通过怎么配置可以只加载右半部分网络就可以输出结果呢？

+ 相关代码：
问题描述中对应的代码。

    1.我认为如果获得L2输出，模型仅仅加载L1数据即可，代码如图：
    ![](https://user-images.githubusercontent.com/22980558/38727666-3f2d0462-3f40-11e8-91e0-60b718ab5ca8.png)

    2.但是如果要获得R2的输出，只加载物品的网络结构就报错了，必须先加载L1的网络结构（可以没有数据输入），再加在R1才能成功，代码如图：
    ![](https://user-images.githubusercontent.com/22980558/38727724-6d5e7e60-3f40-11e8-9ab1-c603ad7e87df.png)

+ 报错输出：
将先加载L1的网络结构的代码注销，如下：
![](https://user-images.githubusercontent.com/22980558/38788576-2425e93a-4167-11e8-85d1-56e164fa93ec.png)

    再次运行，则获得报错输出：
    ![](https://user-images.githubusercontent.com/22980558/38788624-631639c4-4167-11e8-869e-d92326aacdad.png)

+ 解决方法：
从报错输出的信息可以看出，出错的地方应该是加载的模型结构和保存的模型参数不一致，造成模型在使用时，参数对不上，导致报错，可以试下从保存的模型里把两部分各自的参数拿出来手工的保存两个模型分别加载，如果无法确定这两部分各自所使用的参数时可以在定义网络结构时为每一层设置layer name来进行判断。



## 3.推荐模型准确度问题

+ 关键字：`推荐模型` `准确度`

+ 问题描述：
请问movielens这个数据集合上的这个深度模型的误差是不是算很大？正常情况下，即使我用lr，fm，或者矩阵分解来预测，误差值比这个要好很多。

+ 问题讨论：
官方文档中，利用movielens数据集构建出推荐模型的示例只是一个教学例子，其准确性与实用性并没有经过严格的验证，更多是以演示如何使用PaddlePaddle构建推荐系统模型，既如何使用Paddle构建CNN模型来处理文本数据，如果寻找fine tuned模型，可以参考：https://github.com/PaddlePaddle/models 下的ltr, ctr, deep_fm等相关模型。

+ 解决方法：
官方文档中的模型更多以演示使用PaddlePadd1为主，而使用PaddlePaddle构建准确性与实用性高的模型可以参考https://github.com/PaddlePaddle/models ，使用models下通过的模型来构建相应的推荐系统



## 4.个性化推荐训练和测试结果不匹配问题

+ 关键字：`推荐系统` `结果不匹配`

+ 问题描述：参考官方文档中个性化推荐系统的内容，使用了个性化推荐系统的参考代码，有3个疑惑，分别是：
1.每次执行训练后执行预测，输入参数保持不变的情况 prediction 输出每次都不一样，4-4.2之间随机变化，这是什么原因？
2.保存训练结果：parameters.to_tar(f)， 预测时读取训练参数 parameters.init_from_tar(f) ，输入参数保持不变的情况下， 预测结果和第1步预测结果总差0.1左右，这又是什么原因?
3.如果用户信息增加或修改了，或者电影信息增加修改了，整个训练过程需要完整重新执行一次吗？有没增量执行的方法？

+ 解决方法：
官方文档中推荐系统只是一个示例，主要用于演示如果使用PaddlePaddle构建一个推荐系统模型，在该示例代码中使用了random做随机，这就会导致你问题1与问题2的情况，相关的代码如下：

    ```python
    for line in rating:
                    if (rand.random() < test_ratio) == is_test:
                        uid, mov_id, rating, _ = line.strip().split("::")
                        uid = int(uid)
                        mov_id = int(mov_id)
                        rating = float(rating) * 2 - 5.0

                        mov = MOVIE_INFO[mov_id]
                        usr = USER_INFO[uid]
                        yield usr.value() + mov.value() + [[rating]]
    ```

    从代码中可以看出random的使用，完整代码链接：https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/v2/dataset/movielens.py

    在这个推荐系统示例中，电影信息增减，可以增量训练，不需要完整重新train。用户id则不能，因为模型里记下id了。内容和用户增量训练框架本身是支持的，都可以实现



## 5.PaddlePaddle个性化推荐代码速度问题

+ 关键字：`个性化推荐` `运行速度`

+ 问题描述：仿照文档中个性推荐系统的代码，构建了自己的模型，完全代码如下：

```python
def cluster_data_reader(file_dir, node_id):
    def data_reader():
        files = os.listdir(file_dir)
        t1 = time.time()
        print "\n*****begin to reader*****"
        for fi in files:
            with open(file_dir + '/' + fi, "r") as f:
                for line in f:
                    user_id, time_period, user_location, app_id, content_id, cate_id, title, brief, quality_level, check_in_period, read_time = line.strip().split('\t')
                    user_id_code = feature_dict['user_id'].gen(user_id)
                    user_location_code = [feature_dict['user_location'].gen(ul) for ul in user_location.split('|')]
                    content_id_code = feature_dict['content_id'].gen(int(content_id))
                    cate_id_code = feature_dict['cate_id'].gen(cate_id)
                    title_code = [feature_dict['word'].gen(w.lower()) for w in title.split('^')]
                    brief_code = [feature_dict['word'].gen(w.lower()) for w in brief.split('^')]
                    check_in_period_code = feature_dict['check_in_period'].gen(int(check_in_period))
                    record = [user_id_code, int(time_period), user_location_code, content_id_code, cate_id_code, title_code, brief_code, check_in_period_code]
                    yield record + [[float(read_time)]]
        print "cost time of reader is %s" % (time.time()-t1)
        print "*****end reader*****"
    return data_reader

def get_usr_combined_features():
    print "user_id size_%d" % feature_dict['user_id'].size()
    print "user_location_%d" % feature_dict['user_location'].size()
    uid = paddle.layer.data(
        name='user_id',
        type=paddle.data_type.integer_value(
            feature_dict['user_id'].size()))
    usr_emb = paddle.layer.embedding(input=uid, size=16)
    usr_fc = paddle.layer.fc(input=usr_emb, size=16)

    time_period = paddle.layer.data(
        name='time_period',
        type=paddle.data_type.integer_value(24))
    time_period_emb = paddle.layer.embedding(input=time_period, size=16)
    time_period_fc = paddle.layer.fc(input=time_period_emb, size=16)

    usr_location = paddle.layer.data(
        name='user_location',
        type=paddle.data_type.sparse_binary_vector(
        feature_dict['user_location'].size()))
    usr_location_fc = paddle.layer.fc(input=usr_location, size=32)

    usr_combined_features = paddle.layer.fc(
        input=[usr_fc, time_period_fc, usr_location_fc],
        size=200,
        act=paddle.activation.Tanh())
    return usr_combined_features

def get_content_combined_features():
    content_word_dict = feature_dict['word'].dic
    print "content_id size_%d" % feature_dict['content_id'].size()
    print "cate_id size_%d" % feature_dict['cate_id'].size()
    print "content_word_dict length_%d" % len(content_word_dict)
    print "check_in_period size_%d" % feature_dict['check_in_period'].size()

    content_id = paddle.layer.data(
        name='content_id',
        type=paddle.data_type.integer_value(
            feature_dict['content_id'].size()))
    content_emb = paddle.layer.embedding(input=content_id, size=32)
    content_fc = paddle.layer.fc(input=content_emb, size=32)

    content_categories = paddle.layer.data(
        name='cate_id',
        type=paddle.data_type.integer_value(
            feature_dict['cate_id'].size()))
    content_categories_emb = paddle.layer.embedding(input=content_categories, size=16)
    content_categories_fc = paddle.layer.fc(input=content_categories_emb, size=16)

    content_title_id = paddle.layer.data(
        name='title',
        type=paddle.data_type.integer_value_sequence(len(content_word_dict)))
    content_title_emb = paddle.layer.embedding(input=content_title_id, size=128)
    content_title_conv = paddle.networks.sequence_conv_pool(
        input=content_title_emb, hidden_size=128, context_len=2)

    content_brief_id = paddle.layer.data(
        name='brief',
        type=paddle.data_type.integer_value_sequence(len(content_word_dict)))
    content_brief_emb = paddle.layer.embedding(input=content_brief_id, size=128)
    content_brief_conv = paddle.networks.sequence_conv_pool(
        input=content_brief_emb, hidden_size=128, context_len=2)

    check_in_period = paddle.layer.data(
        name='check_in_period',
        type=paddle.data_type.integer_value(
            feature_dict['check_in_period'].size()))
    check_in_period_emb = paddle.layer.embedding(input=check_in_period, size=32)
    check_in_period_fc = paddle.layer.fc(input=check_in_period_emb, size=32)

    content_combined_features = paddle.layer.fc(
        input=[content_fc, content_categories_fc, content_title_conv, content_brief_conv, check_in_period_fc],
        size=200,
        act=paddle.activation.Tanh())
    return content_combined_features

usr_combined_features = get_usr_combined_features()
content_combined_features = get_content_combined_features()

inference = paddle.layer.cos_sim(
        a=usr_combined_features, b=content_combined_features, size=1)
cost = paddle.layer.square_error_cost(
        input=inference,
        label=paddle.layer.data(
            name='read_time', type=paddle.data_type.dense_vector(1)))


def train_model(num_pass):
    print("Begin to train model...")

    train_reader = paddle.batch(
            paddle.reader.shuffle(
                cluster_data_reader(cluster_train_dir, node_id), buf_size=8192),
            batch_size=100)
    test_reader = paddle.batch(
            paddle.reader.shuffle(
                cluster_data_reader(cluster_test_dir, node_id), buf_size=8192),
            batch_size=100)

    parameters = paddle.parameters.create(cost)

    trainer = paddle.trainer.SGD(
        cost=cost,
        parameters=parameters,
        update_equation=paddle.optimizer.Adam(learning_rate=1e-3))

    global t
    t = time.time()

    def event_handler(event):
        if isinstance(event, paddle.event.EndPass):
            if not os.path.exists("model_params"):
                os.makedirs("model_params")
            with gzip.open("model_params/video_recomm_pass_%03d.tar.gz" % event.pass_id, 'w') as f:
                parameters.to_tar(f)

        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id % 1 == 0:
                global t
                print "Pass %d Batch %d Cost %.5f Time %s" % (
                    event.pass_id, event.batch_id, event.cost, time.time()-t)
                t = time.time()

    trainer.train(
        reader=train_reader,
        event_handler=event_handler,
        feeding=feeding,
        num_passes=num_pass)

    print("Training finished!")
```

测试数据：500条，batch_size＝100，pass_num=20，将每个batch的耗时打印出来，具体如下：

```bash
begin to reader
cost time of reader is 0.0190041065216
end reader
Pass 0 Batch 0 Cost 0.00802 Time 0.0562610626221
Pass 0 Batch 1 Cost 0.01444 Time 2.72469496727
Pass 0 Batch 2 Cost 0.00655 Time 2.34903788567
Pass 0 Batch 3 Cost 0.00880 Time 2.33675003052
Pass 0 Batch 4 Cost 0.00874 Time 2.34233188629

begin to reader
cost time of reader is 0.0191640853882
end reader
Pass 1 Batch 0 Cost 0.00708 Time 70.6922171116
Pass 1 Batch 1 Cost 0.00345 Time 2.46361804008
Pass 1 Batch 2 Cost 0.00581 Time 2.33501601219
Pass 1 Batch 3 Cost 0.00240 Time 2.33918118477
Pass 1 Batch 4 Cost 0.00394 Time 2.33651804924

begin to reader
cost time of reader is 0.0201029777527
end reader
Pass 2 Batch 0 Cost 0.00428 Time 64.1795651913
Pass 2 Batch 1 Cost 0.00132 Time 2.34728193283
Pass 2 Batch 2 Cost 0.00151 Time 2.3494591713
Pass 2 Batch 3 Cost 0.00295 Time 2.35163593292
Pass 2 Batch 4 Cost 0.00090 Time 2.34288620949

begin to reader
cost time of reader is 0.203039884567
end reader
Pass 3 Batch 0 Cost 0.00317 Time 66.1391232014
Pass 3 Batch 1 Cost 0.00065 Time 2.39867901802
Pass 3 Batch 2 Cost 0.00048 Time 2.33365297318
Pass 3 Batch 3 Cost 0.00172 Time 2.33354210854
Pass 3 Batch 4 Cost 0.00055 Time 2.33912920952
```

为什么第一个pass的第一个batch耗时很短（0.056s），后面每个pass的第一个batch耗时都非常长(60+s)，核心问题点在哪里呢？是哪个地方的代码写的不好呢？

+ 解决方法：

    根据上面的提供的代码

    ```python
    def event_handler(event):
            if isinstance(event, paddle.event.EndPass):
                if not os.path.exists("model_params"):
                    os.makedirs("model_params")
                with gzip.open("model_params/video_recomm_pass_%03d.tar.gz" % event.pass_id, 'w') as f:
                    parameters.to_tar(f)

            if isinstance(event, paddle.event.EndIteration):
                if event.batch_id % 1 == 0:
                    global t
                    print "Pass %d Batch %d Cost %.5f Time %s" % (
                        event.pass_id, event.batch_id, event.cost, time.time()-t)
                    t = time.time()
    ```

    在EndPass的时候执行了保存模型的操作，由于变量`t`是global，所以这部分的耗时也算在下一个pass的第一个batch了，这就造成了第一个pass的第一个batch耗时很短（0.056s），后面每个pass的第一个batch耗时都非常长(60+s)的现象

+ 问题研究：
当训练模型训练时涉及的数据量很大时，模型读取数据的耗时就是要考虑的一个因素，当数据量很大，如果读入数据消耗的时间是秒基本的，那么模型的训练就会非常耗时，在商业环境下，这通常是难以接受的，对于需要提升大量数据的总训练时间的需求，可以通过增加并行度，比如每个节点增加trainer_count，增加节点个数，这样训练每GB数据的总时间仍然是可以缩短的。提升训练速度的目标我理解无非是能在同样的时间，训练更多的数据达到更好的效果。

    或者换个思路，使用增量的方式训练模型，不必一次去读入当前全局的数据，比如拿1个月的数据，训练一个基础的模型，然后用每天的数据更新这个基础模型，这样每天训练的数据量小。基础的模型训练花费时间长也可以接受了。

+ 问题拓展：

    增量方式支持词典表的增加吗？比如增加了新的id。

    对于这个问题，可以尝试一些方法，比如，base模型预留足够多的`<unk>`，增量模型逐步使用预留的id增量训练，下一个base模型，再用最新的词典训练并进一步扩展预留id，依次类推。但难以判断通过这种方式训练后，对获得模型的效果有什么影响。



## 6.PaddlePaddle中DSSM模型的实现方式与DSSM相关论文的结构不相同

+ 关键字：`DSSM模型`

+ 问题描述：看了下paddle提供的dssm实现代码，最上层是用的square_error_cost，这个和论文里面的最后一层区别还是很大的。请问paddle有按照dssm论文里面的实现方式吗？

    其中DSSM相关论文为：Learning deep structured semantic models for web search using clickthrough data

+ 问题解答：
DSSM 作为一种框架，有若干相关的论文和变种。Paddle Models 中的DSSM有若干流程分支，我们按照 DSSM 的框架和思想来实现，并对模型验证过。Models 中的DSSM 模型通过指定model_type （分类0，排序1，回归2）可以在：softmax + cross entropy、rank loss，和 square error 三种损失函数之间切换。

    阅读了你提及的论文，PaddlePaddle提供的Models中，DSSM使用的优化准则和您提到的这篇论文中的优化准则不同。

    论文中定义的这种损失函数从策略上接近于 LTR 中的 “Listwise” 策略。基于同query下所有检索到候选，来对单个候选进行归一化，再用极大似然估计作为优化准则。
    
    而 DSSM 目前默认的 softmax/regression 都是 "Pointwise"，不考虑到query检索出的其它候选。PaddlePaddle目前没有直接对应于论文中的损失函数（sequence softmax 可以基于一个list进行归一化，但没有损失函数能够利用softmax的计算结果）。
    
    如果学习目标是得到item的表示，损失函数应该确实会起到举足轻重的作用。可以尝试使用其他损失函数来代替它，但不能确定其他损失函数是否也可以获得同样的学习效果。

    后期PaddlePaddle会考虑添加上该损失函数。

+ 问题拓展：
《Learning deep structured semantic models for web search using clickthrough data》论文中这个损失函数最核心的部分应该是能够考虑整个List的信息。比较接近 LTR 中的Listwise策略。Paddle中目前只有LambdaRank是Listwise类型的损失函数。但DSSM与LTR追求“序”还是有所不同，DSSM 追求特征空间的相似性度量。LambdaRank 对学习相似性表示，从原理上不像是一个特别好的选择。



## 7.在使用PaddlePaddle实现推荐系统时，使用了sequence_conv_pool方法，该方法是怎么实现的？有什么作用？

+ 关键字：`推荐系统` `sequence_conv_pool方法`

+ 问题描述：我看了一下paddle的代码，sequence_conv_pool层的介绍是这样的：
    Text input => Context Projection => FC Layer => Pooling => Output.
    参考：
    sequence_conv_pool中Context_projection（）的定义，如果context_len = 3
    原始输入是：
    [A B C D E F G]
    那么卷积 后的结果：
    [ 0AB ABC BCD CDE DEF EFG FG0 ]
    我理解上述过程完成的应该是从 input 到 FC Layer的操作

    在fc layer之后要做pooling，也就是对每次卷积结果0AB, ABC ... 做默认的max pooling
    输出应该是：
    [ m0, m1, m2, m3, m4, m5, m6 ]
    然后pooling之后的结果，再通过一个全链接层做softmax得到分类结果，即下图中红框的部分：

    ![](https://user-images.githubusercontent.com/22980558/28765899-99878302-75ff-11e7-8342-146c0e8bf0c8.png)

    但是paddle中并没有参数对pooling层的节点数量做设置，请问这个设置在paddle中是如何实现的呢？
    如果context_len都是：3
    当输入是[ A B C ]时，卷积是[ 0AB ABC BC0] pooling后是 [ m0, m1, m2]
    当输入是[ A B C D E ]时，卷积是[ 0AB ABC BCD CDE DE0] pooling后是 [ m0, m1, m2, m3, m4] 这个参数肯定是要固定吧？

+ 问题讨论：
从问题描述的内容可以知道提问者的理解错了，而且问题描述中提及的`sequence_conv_pool`是旧版PaddlePaddle中的，这里基于新版PaddlePaddle来解答，最新的PaddlePaddle中`sequence_conv_pool`的参数更易理解，代码如下：

    ```python
    def sequence_conv_pool(input,
                           num_filters,
                           filter_size,
                           param_attr=None,
                           act="sigmoid",
                           pool_type="max"):
    ```

    其中input是输入，num_filters是卷积核的个数，熟悉CNN结构的话，可知上一层使用卷积核的个数决定了下一层的深度，filter_size为卷积核的滑动窗口，param_attr是`Sequence_conv`层的参数，默认是None，act为激活函数，pool_type为池化层要的操作，默认为最大池化。

    `sequence_conv_pool`做的事情，简单而言就是有机的将序列卷积操作与池化操作连接在一起，为Paddle使用者提供一个更高层的API，加快神经网络的构建速度。





## 8.问题：关于个性化推荐网络结构的一些疑问

+ 关键字：`个性化推荐网络`

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



## 9.问题：integer_value_sequence与sparse_binary_vector的区别

+ 关键字：`integer_value_sequence` `sparse_binary_vector` 

+ 问题描述：
在recommender_system的[train.py](https://github.com/PaddlePaddle/book/tree/develop/05.recommender_system)代码中category_id与movie_title的数据形式应该是一致的，都是有多个nominal的可能值。具体来说就是，category_id对应category_dict中的多个取值，movie_title类似。
但是这两个选择了不同的数据类型integer_value_sequence与sparse_binary_vector，这个有什么区别。如果没有区别的话，在实际中要怎么选择。
在我看来sparse_binary_vector和integer_value_sequence完全就是一样的。

+ 相关代码：

```python
def get_mov_combined_features():
    movie_title_dict = paddle.dataset.movielens.get_movie_title_dict()
    mov_id = paddle.layer.data(
        name='movie_id',
        type=paddle.data_type.integer_value(
            paddle.dataset.movielens.max_movie_id() + 1))
    mov_emb = paddle.layer.embedding(input=mov_id, size=32)
    mov_fc = paddle.layer.fc(input=mov_emb, size=32)

    mov_categories = paddle.layer.data(
        name='category_id',
        type=paddle.data_type.sparse_binary_vector( #Datatype: sparse_binary_vector
            len(paddle.dataset.movielens.movie_categories())))
    mov_categories_hidden = paddle.layer.fc(input=mov_categories, size=32)

    mov_title_id = paddle.layer.data(
        name='movie_title',
        type=paddle.data_type.integer_value_sequence(len(movie_title_dict))) #Datatype: integer_value_sequence
    mov_title_emb = paddle.layer.embedding(input=mov_title_id, size=32)
    mov_title_conv = paddle.networks.sequence_conv_pool(
        input=mov_title_emb, hidden_size=32, context_len=3)
```


+ 问题解答：

    假如有2条样本；词典大小为100；第一条样本的词ID为：0 3 5， 第二条样本词ID为：0，10，19；隐层神经元个数为128。

    两者区别：

    sparse_binary_vector 没有序列信息， 接FC层，属于稀疏连接，

    FC层的weight的大小为：100 * 128，
    经过FC层之后：为2 * 128的输出激活值，对于每一个样本，输入的多个神经元链接到一个神经元。
    integer_value_sequence包含序列信息，一般用在RNN模型中, 后面一般接embedding层，

    embedding层的weight大小为：100 * 128。
    经过embedding层之后为：6 * 128，每个词得到一个词向量。
    然后每个样本可以做各种各样的pooling，比如max, average, sum等，得到2 * 128的输出激活值。 而只有sum这种pooling类型和上述sparse_binary_vector + Fc层才等价。
    在实际中如果需要RNN模型(简单的RNN，LSTM, GRU)，或序列卷积等，需要使用integer_value_sequence类型。



## 10.在推荐系统实现中，是否可以使用简单的hash得到index，而不用经过建dict，找唯一index这个步骤？

+ 关键字：`hash index`

+ 问题描述：
我现有的数据的user_id并没有编码为从0开始递增到len-1的形式，所以user_id取值范围并不是[0, len-1]。所以问是否有必要进行重新编码为[0, len-1]，还是原始的编码顺序就是可以的?

+ 相关代码：

```python
user_id = paddle.layer.data(
        name='user',
        type=paddle.data_type.integer_value(100)) #使用的数据类型是integer_value
```

+ 问题解答：
需要重新编码到[0, len-1]，上述词典大小为len(=100)，后面接embedding层，需要依据编码的ID，在table中选出对应词的向量，所以编码必须是 [0, len -1]。




## 11.问题：使用PaddlePaddle实现推荐系统模型时，是否可以使用sparse_binary_vector代替integer_value，两者的区别是什么？

+ 问题描述：使用PaddlePaddle实现推荐系统模型时，是否可以使用sparse_binary_vector代替integer_value，两者的区别是什么？

+ 相关代码：

```python
user_id = paddle.layer.data(
        name='user',
        type=paddle.data_type.integer_value(100)) #使用的数据类型是integer_value
```

+ 问题解答：
integer_value: 一般用来表示label，是个整数值。每个样本的label只有一个整数，一般只作为label用，接入cost层，或者evalutor层。
sparse_binary_vector: 可以用来表示输入特征，在程序内部存储为稀疏矩阵，后面可以接入FC层，或者支持稀疏输入的层。














