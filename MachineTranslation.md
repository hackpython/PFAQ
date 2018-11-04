# 机器翻译

## 背景环境
机器翻译（machine translation, MT）是用计算机来实现不同语言之间翻译的技术。被翻译的语言通常称为源语言（source language），翻译成的结果语言称为目标语言（target language）。机器翻译即实现从源语言到目标语言转换的过程，是自然语言处理的重要研究领域之一。

将深度学习应用于机器翻译任务的方法大致分为两类：1）仍以统计机器翻译系统为框架，只是利用神经网络来改进其中的关键模块，如语言模型、调序模型等（见图1的左半部分）；2）不再以统计机器翻译系统为框架，而是直接用神经网络将源语言映射到目标语言，即端到端的神经网络机器翻译（End-to-End Neural Machine Translation, End-to-End NMT）（见图1的右半部分），简称为NMT模型。

## 机器学习PaddlePaddle-fluid版代码：

https://github.com/PaddlePaddle/book/tree/develop/08.machine_translation

## 1.paddle book机器翻译BLEU值与官方文档中给出的BLEU值偏差过大

+ 关键字：`机器翻译` `BLEU`

+ 问题描述：我按照paddle book上机器翻译的例子用作者提供的训练好的模型进行预测，并计算了一下BLEU值，发现这个BLEU值与作者说的26.92差距太大，我测出来是3.37。请问是什么问题呢？

	我是将is_generating改成True，trainer_count设为1，并将只预测前3条样本的限制给去掉了，我修后的代码如下：

	```python
	    gen_num = 3
	        for item in paddle.dataset.wmt14.gen(dict_size)():
	            gen_data.append([item[0]])
	#            if len(gen_data) == gen_num:
	#                break
	```

	之后使用的是下载的wmt14_model.tar.gz模型，打印出来的生成日志如下所示：

	```bash
	Les <unk> se <unk> au sujet de la largeur des sièges alors que de grosses commandes sont en jeu
	-19.0179        The <unk> will be rotated about the width of the seats , while large orders are at stake . <e>
	-19.1114        The <unk> will be rotated about the width of the seats , while large commands are at stake . <e>
	-19.5112        The <unk> will be rotated about the width of the seats , while large commands are at play . <e>

	La dispute fait rage entre les grands constructeurs aéronautiques �|  propos de la largeur des sièges de la classe touriste sur les vols <uu
	nk> , ouvrant la voie �|  une confrontation <unk> lors du salon aéronautique de Dubaï qui a lieu de mois-ci .
	-28.1139        The dispute is between the large aviation manufacturers about the width of the tourist seats on the <unk> flights , paving the way for a <unk> confrontation at the Dubai aviation fair , which is a month . <e>
	-28.7137        The dispute is between the large aviation manufacturers about the width of the tourist seats on the <unk> flights , paving the way for a <unk> confrontation at the Dubai aviation fair , which takes place months . <e>
	-29.3381        The dispute is between the large aviation manufacturers about the width of the tourist seats on the <unk> flights , paving the way for a <unk> confrontation at the Dubai aviation fair , which takes place in month . <e>
	```

	并且我已经将该结果处理成了下面的格式：

	```
	0
	0       -19.0179        The <unk> will be rotated about the width of the seats , while large orders are at stake . <e>
	1       -19.1114        The <unk> will be rotated about the width of the seats , while large commands are at stake . <e>
	2       -19.5112        The <unk> will be rotated about the width of the seats , while large commands are at play . <e>

	1
	0       -28.1139        The dispute is between the large aviation manufacturers about the width of the tourist seats on the <unk> flights , paving the way for a <unk> confrontation at the Dubai aviation fair , which is a month . <e>
	1       -28.7137        The dispute is between the large aviation manufacturers about the width of the tourist seats on the <unk> flights , paving the way for a <unk> confrontation at the Dubai aviation fair , which takes place months . <e>
	2       -29.3381        The dispute is between the large aviation man
	````

+ 问题讨论：
从问题描述中给出的打印信息来看，其结果是是正确的，请问你是怎么计算BLEU值呢？

	我是在所有的gen样本上计算BLEU的，大概有3003条文本吧。计算BLEU的脚本是之前，计算使用的是paddle老版本里面demo下面的一个脚本eval_bleu.sh，里面调用的是multi-bleu.perl。

	eval_bleu.sh脚本的内容：

	```bash
	set -e
	gen_file=$1
	beam_size=$2

	# find top1 generating result
	top1=$(printf '%s_top1.txt' `basename $gen_file .txt`)
	if [ $beam_size -eq 1 ]; then
	    awk -F "\t" '{sub(" <e>","",$2);sub(" ","",$2);print $2}' $gen_file >$top1
	else
	    awk 'BEGIN{
	        FS="\t";
	        OFS="\t";
	        read_pos = 2} {
	        if (NR == read_pos){
	            sub(" <e>","",$3);
	            sub(" ","",$3);
	            print $3;
	            read_pos += (2 + res_num);
	      }}' res_num=$beam_size $gen_file >$top1
	fi

	# evalute bleu value
	bleu_script=multi-bleu.perl
	standard_res=./data/wmt14/gen/en.trg
	bleu_res=`perl $bleu_script $standard_res <$top1`

	echo $bleu_res
	rm $top1

	```

	调用方法：`sh eval_bleu.sh gen_result_formatted 3`

	代码中中`multi-bleu.perl`文件是mosesdecoder提供的，eval_bleu.sh打印的结果如下：

	```
	It is in-advisable to publish scores from multi-bleu.perl.  The scores depend on your tokenizer, which is unlikely to be reproducible from your paper or consistent across research groups.  Instead you should detokenize then use mteval-v14.pl, which has a standard tokenization.  Scores from multi-bleu.perl can still be used for internal purposes when you have a consistent tokenizer.
	BLEU = 3.37, 21.0/4.0/1.8/1.0 (BP=0.952, ratio=0.954, hyp_len=67792, ref_len=71094)
	```

	结果为：BLEU=3.37，而官方文档中处理的结果为26.92，差距太大了。

+ 问题解答：
从问题讨论给出的`eval_bleu.sh`脚本看出，该脚本是一一对应进行比对的，所以少了2条后，会导致比对顺序全乱了。可以注释掉那行后，重新跑下gen结果，再进行BLEU计算。即将target中没有结构的两行去除掉，计算的BLEU值应该就会与官方文档中的值接近。

+ 问题拓展：
BLEU全称为bilingual evaluation understudy，即双语互译质量评估辅助工具，简单而言，它是一个用于衡量机器翻译质量的指标和工具，通常而言，评估机器翻译的质量这件事情应有人去做，因为只有人才能判断机器翻译的质量，但人来判断会引入主观性，而且重复的工作量太大，所以就引入了BLEU方法来辅助评判机器翻译的质量。

	BLEU的设计思想与人工评判机器翻译质量优差的思想是一致的，即机器翻译的结果越接近专业人工翻译的结果，则认为机器翻译的质量越好，当然这种方法也只是一种指标，并没有作为绝对的判断标准，因为可能机器对于常用语翻译不错，但在翻译论文等专业领域的内容时，质量就有失偏颇了。


## 2.问题：MPI集群跑机器翻译模型demo，中途报错`Forwarding __bidirectional_gru_0___fw`

+ 关键字：`MPI集群` `机器翻译` 

+ 问题描述：我用集群跑paddle给的机器翻译的demo，刚开始跑得好好的，后来不知怎的就报上面这个错，不知是什么原因，还请老师帮忙看一下

+ 报错输出：

```bash
Sat Aug 11 15:35:27 2018[1,1]:
Sat Aug 11 15:35:27 2018[1,1]:Pass 0, Batch 368, Cost 50.880733, {'classification_error_evaluator': 0.7685352563858032}
Sat Aug 11 15:35:27 2018[1,2]:
Sat Aug 11 15:35:27 2018[1,2]:Pass 0, Batch 368, Cost 53.013538, {'classification_error_evaluator': 0.7777777910232544}
Sat Aug 11 15:35:42 2018[1,19]:Thread [139910583289600] Forwarding __bidirectional_gru_0___fw,
Sat Aug 11 15:35:42 2018[1,19]:*** Aborted at 1533972942 (unix time) try "date -d @1533972942" if you are using GNU date ***
Sat Aug 11 15:35:42 2018[1,13]:Thread [140696326326016] Forwarding __bidirectional_gru_0___fw,
Sat Aug 11 15:35:42 2018[1,13]:*** Aborted at 1533972942 (unix time) try "date -d @1533972942" if you are using GNU date ***
Sat Aug 11 15:35:42 2018[1,19]:PC: @ 0x0 (unknown)
Sat Aug 11 15:35:42 2018[1,13]:
Sat Aug 11 15:35:42 2018[1,13]:PC: @ 0x0 (unknown)
Sat Aug 11 15:35:42 2018[1,20]:Thread [140598679189248] Forwarding __bidirectional_gru_0___fw,
Sat Aug 11 15:35:42 2018[1,20]:*** Aborted at 1533972942 (unix time) try "date -d @1533972942" if you are using GNU date ***
Sat Aug 11 15:35:42 2018[1,24]:Thread [139964018677504] Forwarding __bidirectional_gru_0___fw,
Sat Aug 11 15:35:42 2018[1,24]:*** Aborted at 1533972942 (unix time) try "date -d @1533972942" if you are using GNU date ***
Sat Aug 11 15:35:42 2018[1,11]:Thread [140135036057344] Forwarding __bidirectional_gru_0___fw,
Sat Aug 11 15:35:42 2018[1,11]:*** Aborted at 1533972942 (unix time) try "date -d @1533972942" if you are using GNU date ***
Sat Aug 11 15:35:42 2018[1,10]:Thread [139842902214400] Forwarding __bidirectional_gru_0___fw,
Sat Aug 11 15:35:42 2018[1,10]:*** Aborted at 1533972942 (unix time) try "date -d @1533972942" if you are using GNU date ***
Sat Aug 11 15:35:42 2018[1,20]:
Sat Aug 11 15:35:42 2018[1,20]:PC: @ 0x0 (unknown)
Sat Aug 11 15:35:42 2018[1,24]:
Sat Aug 11 15:35:42 2018[1,24]:PC: @ 0x0 (unknown)
Sat Aug 11 15:35:42 2018[1,11]:
Sat Aug 11 15:35:42 2018[1,11]:PC: @ 0x0 (unknown)
Sat Aug 11 15:35:42 2018[1,10]:
Sat Aug 11 15:35:42 2018[1,10]:PC: @ 0x0 (unknown)
Sat Aug 11 15:35:42 2018[1,18]:Thread [139962781460224] Forwarding __bidirectional_gru_0___fw,
Sat Aug 11 15:35:42 2018[1,18]:*** Aborted at 1533972942 (unix time) try "date -d @1533972942" if you are using GNU date ***
Sat Aug 11 15:35:42 2018[1,24]:
Sat Aug 11 15:35:42 2018[1,24]:*** SIGFPE (@0x7f4d8904b460) received by PID 5470 (TID 0x7f4be99d5700) from PID 18446744071713371232; stack trace: ***
Sat Aug 11 15:35:42 2018[1,20]:
Sat Aug 11 15:35:42 2018[1,20]:*** SIGFPE (@0x7fdfff589460) received by PID 20103 (TID 0x7fdfae543700) from PID 18446744073698579552; stack trace: ***
Sat Aug 11 15:35:42 2018[1,24]:
Sat Aug 11 15:35:42 2018[1,24]: @ 0x7f4d8eb9f160 (unknown)
Sat Aug 11 15:35:42 2018[1,20]:
Sat Aug 11 15:35:42 2018[1,20]: @ 0x7fe0050dd160 (unknown)
Sat Aug 11 15:35:42 2018[1,13]:
Sat Aug 11 15:35:42 2018[1,13]:*** SIGFPE (@0x7ff6b4518460) received by PID 48799 (TID 0x7ff66a8d4700) from PID 18446744072439825504; stack trace: ***
Sat Aug 11 15:35:42 2018[1,11]:
Sat Aug 11 15:35:42 2018[1,11]:*** SIGFPE (@0x7f73f16fc460) received by PID 27779 (TID 0x7f73bb0b7700) from PID 18446744073465218144; stack trace: ***
Sat Aug 11 15:35:42 2018[1,13]:
Sat Aug 11 15:35:42 2018[1,13]: @ 0x7ff6ba06c160 (unknown)
Sat Aug 11 15:35:42 2018[1,3]:Thread [140542075557632] Forwarding __bidirectional_gru_0___fw,
Sat Aug 11 15:35:42 2018[1,3]:*** Aborted at 1533972942 (unix time) try "date -d @1533972942" if you are using GNU date ***
Sat Aug 11 15:35:42 2018[1,19]:*** SIGFPE (@0x7f3fb9a26460) received by PID 51743 (TID 0x7f3f789e0700) from PID 18446744072529011808; stack trace: ***
Sat Aug 11 15:35:42 2018[1,3]:PC: @ 0x0 (unknown)
Sat Aug 11 15:35:42 2018[1,19]: @ 0x7f3fbf57a160 (unknown)
Sat Aug 11 15:35:42 2018[1,21]:Thread [140391670904576] Forwarding __bidirectional_gru_0___fw,
Sat Aug 11 15:35:42 2018[1,21]:*** Aborted at 1533972942 (unix time) try "date -d @1533972942" if you are using GNU date ***
Sat Aug 11 15:35:42 2018[1,17]:Thread [140468229388032] Forwarding __bidirectional_gru_0___fw,
Sat Aug 11 15:35:42 2018[1,17]:*** Aborted at 1533972942 (unix time) try "date -d @1533972942" if you are using GNU date ***
Sat Aug 11 15:35:42 2018[1,11]:
Sat Aug 11 15:35:42 2018[1,11]: @ 0x7f73f7250160 (unknown)
Sat Aug 11 15:35:42 2018[1,10]:
Sat Aug 11 15:35:42 2018[1,10]:*** SIGFPE (@0x7f3156ea8460) received by PID 35278 (TID 0x7f2fb682c700) from PID 1458209888; stack trace: ***
Sat Aug 11 15:35:42 2018[1,10]:
Sat Aug 11 15:35:42 2018[1,10]: @ 0x7f315c9fc160 (unknown)
Sat Aug 11 15:35:42 2018[1,18]:
Sat Aug 11 15:35:42 2018[1,18]:PC: @ 0x0 (unknown)
Sat Aug 11 15:35:42 2018[1,20]:
Sat Aug 11 15:35:42 2018[1,20]: @ 0x7fdfff589460 (unknown)
Sat Aug 11 15:35:42 2018[1,20]:
Sat Aug 11 15:35:42 2018[1,20]: @ 0x7fdfff2fc20b hl_cpu_gru_forward<>()
Sat Aug 11 15:35:42 2018[1,12]:Thread [140489601419008] Forwarding __bidirectional_gru_0___fw,
Sat Aug 11 15:35:42 2018[1,12]:*** Aborted at 1533972942 (unix time) try "date -d @1533972942" if you are using GNU date ***
Sat Aug 11 15:35:42 2018[1,20]:
Sat Aug 11 15:35:42 2018[1,20]: @ 0x7fdfff2fbce7 paddle::GruCompute::forward<>()
Sat Aug 11 15:35:42 2018[1,21]:
Sat Aug 11 15:35:42 2018[1,21]:PC: @ 0x0 (unknown)
Sat Aug 11 15:35:42 2018[1,24]:
Sat Aug 11 15:35:42 2018[1,24]: @ 0x7f4d8904b460 (unknown)
Sat Aug 11 15:35:42 2018[1,19]:
Sat Aug 11 15:35:42 2018[1,19]: @ 0x7f3fb9a26460 (unknown)
Sat Aug 11 15:35:42 2018[1,12]:
Sat Aug 11 15:35:42 2018[1,12]:PC: @ 0x0 (unknown)
Sat Aug 11 15:35:42 2018[1,20]:
Sat Aug 11 15:35:42 2018[1,20]: @ 0x7fdfff2ff9ae paddle::GatedRecurrentLayer::forwardBatch()
Sat Aug 11 15:35:42 2018[1,24]:
Sat Aug 11 15:35:42 2018[1,24]: @ 0x7f4d88dbe20b hl_cpu_gru_forward<>()
Sat Aug 11 15:35:42 2018[1,19]:
Sat Aug 11 15:35:42 2018[1,19]: @ 0x7f3fb979920b hl_cpu_gru_forward<>()
......
```

+ 问题讨论：
从报错输出中看到`SIGFPE (@0x7f3156ea8460) received by PID 35278 (TID 0x7f2fb682c700) from PID 1458209888; stack trace:`，出现了`SIGFPE`，通常都属于浮点数异常。

+ 问题分析：

	Paddle二进制在运行时捕获了浮点数异常，只要出现浮点数异常(即训练过程中出现NaN或者Inf)，立刻退出。浮点异常通常的原因是浮点数溢出、除零等问题。 主要原因包括两个方面:

	+ 训练过程中参数或者训练过程中的梯度尺度过大，导致参数累加，乘除等时候，导致了浮点数溢出。
	+ 模型一直不收敛，发散到了一个数值特别大的地方。
	+ 训练数据有问题，导致参数收敛到了一些奇异的情况。或者输入数据尺度过大，有些特征的取值达到数百万，这时进行矩阵乘法运算就可能导致浮点数溢出。

+ 解决方法：

	有两种有效的解决方法：

	1.设置 `gradient_clipping_threshold` 参数，示例代码如下：

	```python
	optimizer = paddle.optimizer.RMSProp(
	    learning_rate=1e-3,
	    gradient_clipping_threshold=10.0,
	    regularization=paddle.optimizer.L2Regularization(rate=8e-4))
	```

	更详细的代码可以参考PaddlePaddle models/legacy/nmt_without_attention下的train.py

	https://github.com/PaddlePaddle/models/blob/59adc0d6f38cd2351e16608d6c9d4e72dd5e7fea/legacy/nmt_without_attention/train.py#32

	2.具体可以参考 `nmt_without_attention` 示例。

	设置 error_clipping_threshold 参数，示例代码如下：

	```python
	decoder_inputs = paddle.layer.fc(
	    act=paddle.activation.Linear(),
	    size=decoder_size * 3,
	    bias_attr=False,
	    input=[context, current_word],
	    layer_attr=paddle.attr.ExtraLayerAttribute(
	        error_clipping_threshold=100.0))
	```

	完整代码可以参考
	https://github.com/PaddlePaddle/book/blob/develop/08.machine_translation/train.py#L66

+ 问题研究：
两种方法的区别：

	+ 1.两者都是对梯度的截断，但截断时机不同，前者在 optimzier 更新网络参数时应用；后者在激活函数反向计算时被调用；
	+ 2.截断对象不同：前者截断可学习参数的梯度，后者截断回传给前层的梯度;
	除此之外，还可以通过减小学习率或者对数据进行归一化处理来解决这类问题。

## 3.判别式的机器翻译任务出core

+ 关键字：`生成式机器翻译` `判别式机器翻译`

+ 问题描述：现在我们的seq2seq中的机器翻译例子是产生式的（is_generating=True），我想把这个任务改成判别式的（就是只是给出源语言翻译到给定目标语言的得分），训练的时候遵循https://github.com/PaddlePaddle/Paddle/blob/develop/doc/tutorials/text_generation/index_en.md 里面的步骤训练出一个翻译模型， 在test阶段在gen.conf 中设置 is_generating=False ， 结果出core了，错误信息如下：

	```bash
	I /home/img/baidu/idl/paddle/paddle/trainer/Trainer.cpp:148] trainer: in testing mode
	I /home/img/baidu/idl/paddle/paddle/trainer/Trainer.cpp:155] trainer mode: Testing
	I /home/img/baidu/idl/paddle/paddle/gserver/dataproviders/PyDataProvider2.cpp:247] loading dataprovider dataprovider::process
	[INFO 2016-12-15 13:57:33,367 dataprovider.py:27] src dict len : 30000
	[INFO 2016-12-15 13:57:33,367 dataprovider.py:37] trg dict len : 30000
	I /home/img/baidu/idl/paddle/paddle/gserver/dataproviders/PyDataProvider2.cpp:247] loading dataprovider dataprovider::process
	[INFO 2016-12-15 13:57:33,416 dataprovider.py:27] src dict len : 30000
	[INFO 2016-12-15 13:57:33,416 dataprovider.py:37] trg dict len : 30000
	F /home/img/baidu/idl/paddle/paddle/trainer/Tester.cpp:286] Check failed: (paramUtil_->loadParameters(passId, true , true )) == (true)
	/home/img/liushui/bin/paddle: line 81: 4510 Aborted (core dumped) ${DEBUGGER} $MYDIR/../opt/paddle/bin/paddle_trainer ${@:2}
	```

	训练，我使用的配置是：
		
	```bash
	is_generating = False

	data_dir = "./data/pre-wmt14"
	train_conf = seq_to_seq_data(data_dir = data_dir,
	is_generating = is_generating)

	settings(
	learning_method = AdamOptimizer(),
	batch_size = 50,
	learning_rate = 5e-4)

	gru_encoder_decoder(train_conf, is_generating)
	```

	测试，我使用的配置是：

	```
	is_generating = False

	gen_conf = seq_to_seq_data(data_dir = "./data/pre-wmt14",
	is_generating = is_generating,
	gen_result = "./translation/gen_result")

	settings(
	learning_method = AdamOptimizer(),
	batch_size = 1,
	learning_rate = 0)

	gru_encoder_decoder(gen_conf, is_generating)
	```

+ 问题解答：
根据问题描述中提供的信息，你的需求是不是这样 ： 使用seq2seq model 训练，不生成，而是为两个序列打分，计算 P(Y|X) 这个条件概率呢？X 是源语言序列，Y 是目标语言序列

	如果是上面的需要的话。

	使用训练的配置，而不是生成，走 paddle_trainer 的 job=test 流程
	配置的Outputs 选择 softmax 。这时会输出每个时间步词典中每一个词是下一个词的概率，需要自己做后处理，把目标词的概率选择出来，logProb 相加，得到整个句子的得分
	以上方法有两个缺点：

	PaddlePaddle 目前木有现成的Layer 能够选择一个向量的某一维输出，上面的方法会令输出文件非常大。
	softmax 当词表比较大的时候，计算得会比较慢。
	其实，如果只是计算两个序列的相似性，下一个词总是知道的，如果可以不去计算全词表中每一个词的概率，那么 score 2 sequence 这个过程是可以极大加速的。

	推荐训练的时候使用 PaddlePaddle 的 multi_class_cross_entropy_with_selfnorm 损失函数，预测时使用 SelectiveFc ，可以只输出目标词那一个词的概率，同时避免上面提到的两个问题，极大地加速运算。


## 4.为什么trainer.train中没有feeding参数也可以保持数据输入与模型中input的关系

+ 关键字：`机器翻译` `feeding参数`

+ 问题描述：
根据paddle文档的介绍，“ Reader返回的数据可以包括多列，我们需要一个Python dict把列 序号映射到网络里的数据层。”
参考：https://github.com/PaddlePaddle/book/blob/develop/01.fit_a_line/train.py

但是我发现在机器翻译例子中，并没有使用feeding方式，没有指定数据与模型input的关系，
参考：https://github.com/PaddlePaddle/book/blob/develop/08.machine_translation/train.py
请问程序是如何保证数据与模型input对应一致的呢

+ 问题解答：
Paddle会对网络配置进行解析，解析出来的 数据层 的顺序和定义顺序一致，而Paddle默认的input顺序和解析出的数据层的顺序一致。所以在不使用feeding方式时，input顺序须和数据层定义顺序保持一致。


## 5.修改seqToseq的目标函数(cost function) 

+ 关键字：`seqToseq`

+ 问题描述：
我之前试用过demo中的seqToseq示例，成功训练了机器翻译nmt模型，工作正常。

最近看了一篇将align信息引入到cost function中的论文：
Wenhu Chen, “guided alignment training for topic-aware neural machine translation”.

这篇论文的思路其实比较简单：
seqToseq中的attention可以看做对齐信息，但这个对齐没有fast align的强对齐效果好，作者希望模型在训练过程中能够参考fast align的强对齐结果，对attention进行调整。基于这个思路，作者先在线下用fast align对语料进行了对齐，然后定义了fast align结果和网络attention之间的cost，将其加入到cross entropy的cost中，即：

![](https://cloud.githubusercontent.com/assets/18412946/21791214/46c43b38-d71c-11e6-944e-46d78519891f.png)

![](https://cloud.githubusercontent.com/assets/18412946/21791224/5b9ae2b4-d71c-11e6-9bb7-cc8889c95318.png)

其中：
HD就是目前seqToseq demo中使用的cross entropy
G是作者定义的align cost。
可以看到，最后cost function是HD和G两者的加权和。w1、w2控制了两者的比例。

G作为align cost，其实就是mean squared error。G中的A是二维矩阵，即fast align的对齐结果，Ati是target sentence第t个token和src sentence第i个token的对齐情况，若fast align的结果认为这两者是对齐的，则Ati被置为1(实际计算时会对A进行归一化，使得每一行的和是1)。
alpha则是网络中的attention。

上面说完了背景，现在说说我怎么在paddle中尝试实现这一方案。

1.首先用fast align对语料做了对齐，为每个sentence pair生成了对应的A，A的行数为该pair中target sentence的token数，A的列数为src sentence的token数。根据fast align的结果，将A中相应的位置置为1，最后再对每一行进行了归一化。

2.完成1后，就要将A作为训练信息通过dataprovider传入到paddle中。
由于后面要计算A和attention的align cost，所以我先看了下demo中的attention，其代码在simple_attention中：

```
attention_weight = fc_layer(input=m, size=1, act=SequenceSoftmaxActivation(), param_attr=softmax_param_attr, name="%s_softmax" % name, bias_attr=False)
```

我的理解是：simple_attention方法在解码端每个time step都会执行一次，在当前t时刻时，这里的attention_weight是一个序列，序列长度是当前sentence pair的src len，序列中每个元素是一个一维的浮点数，第i个元素表示当前解码时刻t的target token和src第i个token的attention值。

相应地，我也将fast align的结果A设置为类似的格式，采用了
dense_vector_sub_sequence(1)
这种格式，假设训练样本sentence pair的src包含3个token，target包含2个token，则A的形式举例如下：
[
[ [0.5],[0.5],[0] ], //target第1个token和src每个token的对齐结果
[ [0],[0.5],[0.5] ], //target第2个token和src每个token的对齐结果
]

我按照这种格式将A传进了paddle，具体如下：

```
a = data_layer(name='target_source_align', size=1)

decoder = recurrent_group(name=decoder_group_name,
                              step=gru_decoder_with_attention,
                              input=[
                                  StaticInput(input=encoded_vector,
                                              is_seq=True),
                                  StaticInput(input=encoded_proj,
                                              is_seq=True),
                                  trg_embedding
                              ])
```

现在我有几个问题：

1.我上述的理解和处理流程是否正确？
2.如何将align_info这种sub_sequence传入到recurrent_group->gru_decoder_with_attention中？
3.如果attention_weight是长度为src len的序列，那么怎么与a计算上面式子中定义的align cost(即G)
4.如何将两个cost加权在一起进行训练？

我尝试和trg_embedding一样，直接将a传入到recurrent_group的input中：

```
decoder = recurrent_group(name=decoder_group_name,

                          step=gru_decoder_with_attention,
                          input=[
                              StaticInput(input=encoded_vector,
                                          is_seq=True),
                              StaticInput(input=encoded_proj,
                                          is_seq=True),
                              trg_embedding,
                              a
                          ])
```

相应地，在gru_decoder_with_attention中也添加了a变量
`def gru_decoder_with_attention(enc_vec, enc_proj, current_word, a):`

结果报错：

```
F /home/nmt/paddle/paddle/gserver/gradientmachines/RecurrentGradientMachine.cpp:407] Check failed: ((size_t)input1.getNumSequences()) == (numSequences)
/home/nmt/paddle/install/bin/paddle: line 81: 29166 Aborted ${DEBUGGER} $MYDIR/../opt/paddle/bin/paddle_trainer ${@:2}
```

我把a改成SubsequenceInput，也有问题：

```
[CRITICAL 2017-01-10 11:16:42,404 layers.py:2493] The sequence type of in_links should be the same in RecurrentLayerGroup
```

我看了下layers.py中对应位置，应该是因为：

```
trg_embedding是LayerOutput
a是SubsequenceInput,
导致RecurrentLayerGroupWithoutOutLinksBegin中的assert检查失败：
config_assert(in_links_has_subseq == has_subseq, "The sequence type of in_links should be the same in RecurrentLayerGroup")
```

另外，为了使用simple_attention中的attention_weight，我修改了networks.py中的simple_attention返回值，将attention_weight也返回了：

```
return pooling_layer(input=scaled, pooling_type=SumPooling(), name="%s_pooling" % name), attention_weight
```

然后在seqToseq_net.py中调用simple_attention的位置定义了attention_weight:

```
context,attention_weight = simple_attention(encoded_sequence=enc_vec, encoded_proj=enc_proj, decoder_state=decoder_mem, )
```

我是打算在gru_decoder_with_attention中计算a和attention_weight的cost，然后作为返回值和out变量(即softmax)一起传出去。不知道这样理解对吗？

+ 问题解答：

	对流程的理解，大致是没有问题的。

	先说结论，你需要的模型如果不修改Paddle的 C++ 代码，无法通过配置直接配出来。

	下面是关于上面提到的 4 个问题。

	1.正确的做法是把 attention weight 拿到 recurrent_layer_group 外面，然后和通过data_provider 给进去的fast_align 得到的“正确的”对齐信息，通过 cost layer （paddle 有 MSE 这个cost layer）G 这一部分 error。但是 attention weight 目前无法拿到 recurrent_layer_group 外面。下面会详细说明。

	2.在你的这个例子里面，align info 是无法传进 recurrent_layer_group 里面的。

	3.有了2 的答案，3 这种做法就行不通了。

	4.接 2 个 cost paddle 是可以直接支持的，直接定义 2 个cost 就可以，cost layer 的接口中，可以指定权值。

+ 问题分析：

	我大致解释一下 recurrent_layer_group 的流程和现在的一些限制。
	你需要的功能非常简单，但是改代码可能需要花一些时间去看一看 layer group 的处理逻辑。

	1.recurrent_layer_group 是一种支持自定义 rnn 的机制，可以看作是一个被 “打开的RNN”，通过定义step 函数，来自定义 rnn 在一个时间步内的计算，recurrent_layer_group 框架本身会完成在整个序列上的展开。虽然 recurrent_layer_group 是被打开的RNN，但是定义在 step 函数中定义的layer 在layer group 外部却不是总能够直接引用的。

	2.recurrent_layer_group 这个调用接口中的 inputs 是layer group 的输入，step 函数 return 的layer 的输出是 recurrent_layer_group 这个自定义RNN 单元的输出；

	3.虽然 recurrent_layer_group 是被打开的RNN，不是什么样的序列都可以塞给 recurrent_layer_group 。 recurrent_layer_group 依然需要遵循普通 rnn 的一些原则。RNN 接受一个序列作为输入，每个时间步会有一个输出，也就是“多长进多长出”。这是目前 recurrent_layer_group 一个很重要的原则。

	4.在 nmt 的例子中，target embedding 序列是 recurrent_layer_group 真正的输入，输出是decoder 的hidden 向量序列，以这个例子为例，会发现 layer group 需要“多长进多长出”；这一个限制导致 attention weight 不能拿出 layer_group

	5.attention 的例子中，源语言序列对 recurrent_layer_group 来说应该叫做 unbounded memory，这是一种特殊的输入。每个时刻会引用 unbounded memory 中存储的内容；也就是说，recurrent_layer_group 和普通rnn 一样，可以有一个或者多个输入序列，是rnn 真正的数据输入，step 函数在数据输入反复被展开计算。这些数据输入必须等长，如果有不等长的输入，只能通过 read-only memory 的形式拿进来。

	6.在你的例子里面，出现了 3 种不等长的输入。 recurrent layer group 还有一个限制，所有 input 的序列类型都必须是相同的，也就是要么都是sequence，要么都是 subsequence；这一个限制导致 fast_align 的 weight 不能给进 recurrent_layer_group


## 6.问题：docker版本paddledev/paddle:0.10.0rc3-noavx，运行机器翻译的例子出现'Floating point exception'

+ 关键字：`Floating point exception` `机器翻译`

+ 问题描述：使用docker镜像安装好了Paddle后，运行机器翻译的示例出现`Floating point exception`，Docker版本为paddledev/paddle:0.10.0rc3-noavx

+ 问题解答：
用户使用的话，一般用https://hub.docker.com/r/paddlepaddle/paddle/tags/ 这里面的docker， https://hub.docker.com/r/paddledev/paddle/tags/ 中列的是开发版的。
	同时，paddle paddle也提供了paddle book的镜像https://hub.docker.com/r/paddlepaddle/book/tags/ ，最新的paddlebook是没有问题

## 7.浮点数向量序列向量序列该如何定义输入？

+ 关键字：`浮点数`

+ 问题描述：我的预测任务需要用到浮点数向量序列输入，查看了一下文档和机器翻译的例子，没有弄懂如何定义。

	yield的数据样例：
	[[0.0, 1.0, 0.2903], [0.0, 1.0, 0.291], [0.0, 1.0, 0.2917]] [0]
	data = paddle.layer.data("status",paddle.data_type.dense_vector(params))
	这里面的params该如何填写？

+ 问题解答：
定义 data_layer如下：

	```python
	input = paddle.layer.data(
	        name='input',
	        type=paddle.data_type.dense_vector_sequence(3))
	```

	python 端读数据接口返回如下格式的 python list：

	```
	 [[0.0, 1.0, 0.2903], [0.0, 1.0, 0.291], [0.0, 1.0, 0.2917]]
	```

	还需要提一下：
	1.data 的type 指定错误 paddle.data_type.dense_vector --> paddle.data_type.dense_vector_sequence
	2.实例中 params 变量表示的是一个时间步实向量的维度，对应以上的示例数据，params 的大小为 3



## 8.问题：出现AssertionError的Bug 

+ 关键字：`机器翻译` `AssertionError`

+ 问题描述：
使用paddlepaddle进行机器翻译训练,使用的语料是中英文语料,已分词已对齐
	引用训练集直接更改的.cache/paddle/wmt/wmt14.tgz里的内容,其他代码参数未修改,在开始程序后出现AssertionError错误.

+ 报错输出：

```bash
I0201 20:07:38.419838 13473 Util.cpp:166] commandline: --use_gpu=False --trainer_count=2 I0201 20:07:38.691911 13473 GradientMachine.cpp:94] Initing parameters.. I0201 20:07:41.030560 13473 GradientMachine.cpp:101] Init parameters done. Traceback (most recent call last): File "train.py", line 163, in <module> main() File "train.py", line 159, in main train() File "train.py", line 154, in train feeding=feeding) File "/usr/local/lib/python2.7/dist-packages/paddle/v2/trainer.py", line 162, in train for batch_id, data_batch in enumerate(reader()): File "/usr/local/lib/python2.7/dist-packages/paddle/v2/minibatch.py", line 33, in batch_reader for instance in r: File "/usr/local/lib/python2.7/dist-packages/paddle/v2/reader/decorator.py", line 70, in data_reader for e in reader(): File "/home/yanmengqi/桌面/mt_with_external_memory/data_utils.py", line 12, in new_reader for ins in reader(): File "/usr/local/lib/python2.7/dist-packages/paddle/v2/dataset/wmt14.py", line 73, in reader src_dict, trg_dict = __read_to_dict__(tar_file, dict_size) File "/usr/local/lib/python2.7/dist-packages/paddle/v2/dataset/wmt14.py", line 60, in __read_to_dict__ assert len(names) == 1 AssertionError
```

+ 问题解答：
因为不知道你使用中英文语料数据的细节，所以无法给出具体的解决操作，但该报错通常是因为压缩包里存在多个以src.dict或trg.dict结尾的文件，具体可以参考

	https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/v2/dataset/wmt14.py#L68



## 9.demo machine translate "beam_search() got multiple values for keyword argument 'end_id'"

+ 关键字：`机器翻译` `beam_search`

+ 问题描述：使用Fluid 0.14 CPU版的PaddlePaddle运行机器翻译的例子时，出现`beam_search() got multiple values for keyword argument 'end_id'`

+ 报错输出：

```bash
Traceback (most recent call last):
  File "infer.py", line 196, in <module>
    main(use_cuda)
  File "infer.py", line 191, in main
    decode_main(False)  # Beam Search does not support CUDA
  File "infer.py", line 133, in decode_main
    translation_ids, translation_scores = decode(context)
  File "infer.py", line 98, in decode
    beam_size, end_id=10, level=0)
TypeError: beam_search() got multiple values for keyword argument 'end_id'
```

+ 问题解答：
这个问题很有可能是PaddlePaddle版本问题，因为`beamsearch`相关的内容在Fluid 0.14版后依旧有相应的修改，所以造成该问题的原因可能是你使用机器学习示例中的用法与Paddle版本中提供的api不匹配。

+ 解决方法：
将PaddlePaddle更新成最新的版本，参考官方最新的机器翻译的实例，如下

	http://www.paddlepaddle.org/documentation/docs/zh/1.0/beginners_guide/basics/machine_translation/index.html

	或者直接参考PaddlePaddle提供的book中关于机器翻译的代码，如下

	https://github.com/PaddlePaddle/book/tree/develop/08.machine_translation
