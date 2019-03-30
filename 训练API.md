# 数据API

## `未审核`训练时sum op上报错Expected in_dim == x_dim, but received in_dim:10, 512 != x_dim:548, 512 

+ 版本号：`1.1.0`

+ 标签：`sum op` `训练`

+ 问题描述：训练时sum op上报错Expected in_dim == x_dim, but received in_dim:10, 512 != x_dim:548, 512 

+ 报错输出：

```
   cost = exe.run(main_program, feed=feeder.feed(data), fetch_list=[avg_cost])
  File "./tools/paddle_fluid/paddle_release_home/python/lib/python2.7/site-packages/paddle/fluid/executor.py", line 472, in run
    self.executor.run(program.desc, scope, 0, True, True)
paddle.fluid.core.EnforceNotMet: Enforce failed. Expected in_dim == x_dim, but received in_dim:10, 512 != x_dim:548, 512.
Input tensors must have same shape at [/paddle/paddle/fluid/operators/sum_op.cc:59]
PaddlePaddle Call Stacks:
0       0x7f78eff19ed2p paddle::platform::EnforceNotMet::EnforceNotMet(std::__exception_ptr::exception_ptr, char const*, int) + 482
1       0x7f78f044c7a5p paddle::operators::SumOp::InferShape(paddle::framework::InferShapeContext*) const + 1237
2
```

+ 相关代码：

```
def get_p_attention(p_emb_layer, lstm_layer, hidden_dim):
    p_fc = fluid.layers.fc(input=p_emb_layer, size=hidden_dim, act='tanh')
    plstm_0, _ = fluid.layers.dynamic_lstm(input=p_fc, \
        size=hidden_dim,
        candidate_activation='relu',
        gate_activation='sigmoid',
        cell_activation='sigmoid',
        is_reverse=True)
    p_lstm_layer = fluid.layers.sequence_last_step(input=plstm_0)
    p_expand = fluid.layers.sequence_expand(x=p_lstm_layer, y=lstm_layer)
    combined_input = fluid.layers.elementwise_mul(x=lstm_layer,
                                        y=p_expand)
    attention_weight = fluid.layers.fc(input=combined_input,
                                        size=1,
                                        act='tanh',
                                        bias_attr=False)
    #得到归一化权重
    normed_attention_weight = fluid.layers.sequence_softmax(input=attention_weight)
    assist_info = {'unnormalized_p_attention_weight': attention_weight}
    return normed_attention_weight, assist_info

def att_model():
    word_emb_fixed = True if conf_dict['word_emb_fixed'] == "True" else False
    emb_distributed = not conf_dict['is_local']
    conf_dict['is_sparse'] = bool(conf_dict['is_sparse'])
    char_param = fluid.ParamAttr(name=conf_dict['emb_name'],
                                 trainable=(not char_emb_fixed))
    
    char_embedding = fluid.layers.embedding(
        input=char,
        size=[data_reader.get_dict_size('charemb_dict'),
            conf_dict['char_dim']],
        dtype='float32',
        is_distributed=emb_distributed,
        is_sparse=emb_distributed,
        param_attr=char_param)

    p_embedding = fluid.layers.embedding(
        input=p_word,
        size=[data_reader.get_dict_size('charemb_dict'),
            conf_dict['char_dim']],
        dtype='float32',
        is_distributed=emb_distributed,
        is_sparse=emb_distributed,
        param_attr=char_param)

    emb_layers = [char_embedding, p_embedding]
    # input hidden
    char_fc = fluid.layers.fc(input=char_embedding, size=hidden_dim, act='tanh')
    lstm_layer, _ = fluid.layers.dynamic_lstm(
        input=char_fc,
        size=hidden_dim,
        candidate_activation='relu',
        gate_activation='sigmoid',
        cell_activation='sigmoid',
        is_reverse=0)
    position_weight_layer, position_attention_assist_info = \
        get_p_attention(p_embedding, lstm_layer, hidden_dim)
    p_scaled_lstm_layer = fluid.layers.elementwise_mul(x=lstm_layer,
                                    y=position_weight_layer)
    p_lstm_layer = fluid.layers.sequence_pool(input=p_scaled_lstm_layer,
                                            pool_type='sum')
  
    lstm_layers = [p_lstm_layer, lstm_layer]
    hidden_0_layers = [
        fluid.layers.fc(input=l_layer, size=hidden_dim, act='tanh')
            for l_layer in lstm_layers
        ]
    hidden_0 = fluid.layers.sums(input=hidden_0_layers)
    lstm_0 = fluid.layers.dynamic_lstm(
        input=hidden_0/4,
        size=hidden_dim,
        candidate_activation='relu',
        gate_activation='sigmoid',
        cell_activation='sigmoid')

    # stack L-LSTM and R-LSTM with direct edges
    input_tmp = [hidden_0, lstm_0]

    for i in range(1, depth):
        mix_hidden = fluid.layers.sums(input=[
            fluid.layers.fc(input=input_tmp[0], size=hidden_dim, act='tanh'),
            fluid.layers.fc(input=input_tmp[1], size=hidden_dim, act='tanh')
        ])

        lstm = fluid.layers.dynamic_lstm(
            input=mix_hidden,
            size=hidden_dim,
            candidate_activation='relu',
            gate_activation='sigmoid',
            cell_activation='sigmoid',
            is_reverse=((i % 2) == 1))

        input_tmp = [mix_hidden, lstm]

    # output
    feature_out = fluid.layers.sums(input=[
        fluid.layers.fc(input=input_tmp[0], size=label_dict_len, act='tanh'),
        fluid.layers.fc(input=input_tmp[1], size=label_dict_len, act='tanh')
    ])

    return feature_out
```

+ 解决方法：

```
paddle.fluid.core.EnforceNotMet: Enforce failed. Expected in_dim == x_dim, but received in_dim:10, 512 != x_dim:548, 512.
Input tensors must have same shape at [/paddle/paddle/fluid/operators/sum_op.cc:59]
```

sum op报错维度没对上。可以先把网络中sum_op的input维度都打印出来看下，定位到具体是哪个sum_op出错。


## `待审核`3.问题：Input(Out@GRAD) shouldn't be null错误怎么排查 

+ 版本号：`1.1.0`

+ 标签：`input null`

+ 问题描述：程序还没有开始运行，加上optimizer以后报错


+ 报错输出：

```
Input(Out@GRAD) shouldn't be null. at [/paddle/paddle/fluid/operators/reshape_op.cc:314]
PaddlePaddle Call Stacks:
0       0x7fd071284455p void paddle::platform::EnforceNotMet::IniTraceback (most recent call last):
  File "main.py", line 24, in <module>
    autodl_exe = autodl.AutoDL()
  File "/paddle/parl/paddle-demo/autodl/autodl.py", line 40, in __init__
    self.controller = AutoDLController(self.algorithm, self.parse_args)#self.parse_args.num_nodes,
  File "/paddle/parl/paddle-demo/autodl/autodl_controller.py", line 34, in __init__
    super(AutoDLController, self).__init__(algorithm)
  File "/usr/local/lib/python2.7/dist-packages/parl-1.0-py2.7.egg/parl/framework/agent_base.py", line 46, in __init__
    self.build_program()
  File "/paddle/parl/paddle-demo/autodl/autodl_controller.py", line 74, in build_program
    self.update_op = self.alg.define_learn(obs=inputs, reward=reward, action=None)
  File "/paddle/parl/paddle-demo/autodl/reinforce_policy_gradient.py", line 71, in define_learn
    train_op = optimizer.minimize(cost)
  File "/usr/local/lib/python2.7/dist-packages/paddle/fluid/optimizer.py", line 404, in minimize
    parameter_list, no_grad_set)
  File "/usr/local/lib/python2.7/dist-packages/paddle/fluid/optimizer.py", line 316, in backward
    return append_backward(loss, parameter_list, no_grad_set, callbacks)
  File "/usr/local/lib/python2.7/dist-packages/paddle/fluid/backward.py", line 518, in append_backward
    _append_backward_vars_(root_block, fwd_op_num, grad_to_var, grad_info_map)
  File "/usr/local/lib/python2.7/dist-packages/paddle/fluid/backward.py", line 354, in _append_backward_vars_
    op_desc.infer_shape(block.desc)
EnforceNotMet: Input(Out@GRAD) shouldn't be null. at [/paddle/paddle/fluid/operators/reshape_op.cc:314]
```

+ 相关代码：

```
optimizer = fluid.optimizer.Adam(learning_rate=self.lr)
train_op = optimizer.minimize(cost)
```

+ 问题分析：

从报错中，可以发现是基于 parl.layers 组网，而不是 fluid.layers，PARL.layers 只是一个对fluid.layers 的二次封装，为了更好的进行参数复用。

+ 解决方法：

训练代码中用到了softmax_with_cross_entropy和sigmoid_cross_entropy_with_logits，这两个op针对label不计算梯度，这点与tf对应接口实现不同，所以在具体的训练时，Input为空。








