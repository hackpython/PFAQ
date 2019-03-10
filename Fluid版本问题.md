# Fuild版本问题

## `待审核`1.问题：Fluid模型run startup报错 

+ 版本号：`1.0.1`

+ 标签：`run startup`

+ 问题描述：Fluid模型run startup报错 

+ 报错输出：

```
`*** Aborted at 1550832133 (unix time) try "date -d @1550832133" if you are using GNU date ***
PC: @     0x7f226a787312 paddle::framework::VisitDataType<>()
*** SIGILL (@0x7f226a787312) received by PID 32275 (TID 0x7f22da922700) from PID 1786278674; stack trace: ***
    @     0x7f22da4f9160 (unknown)
    @     0x7f226a787312 paddle::framework::VisitDataType<>()
    @     0x7f226a76f7ae paddle::operators::math::set_constant_with_place<>()
    @     0x7f2269e0bdda _ZNK6paddle9operators14FillConstantOp7RunImplERKNS_9framework5ScopeERKN5boost7variantINS_8platform9CUDAPlaceEJNS8_8CPUPlaceENS8_15CUDAPinnedPlaceEEEE
    @     0x7f226a723b58 paddle::framework::OperatorBase::Run()
    @     0x7f2269d3eb51 paddle::framework::Executor::RunPreparedContext()
    @     0x7f2269d3f380 paddle::framework::Executor::Run()
    @     0x7f2269c674fd _ZZN8pybind1112cpp_function10initializeIZN6paddle6pybindL13pybind11_initEvEUlRNS2_9framework8ExecutorERKNS4_11ProgramDescEPNS4_5ScopeEibbE63_vIS6_S9_SB_ibbEINS_4nameENS_9is_methodENS_7siblingEEEEvOT_PFT0_DpT1_EDpRKT2_ENUlRNS_6detail13function_callEE1_4_FUNEST_
    @     0x7f2269cb7672 pybind11::cpp_function::dispatcher()
    @           0x4b4bfa PyEval_EvalFrameEx
    @           0x4b6b28 PyEval_EvalCodeEx
    @           0x4b5d10 PyEval_EvalFrameEx
    @           0x4b5fb8 PyEval_EvalFrameEx
    @           0x4b6b28 PyEval_EvalCodeEx
    @           0x4b6c52 PyEval_EvalCode
    @           0x4e1c7d PyRun_FileExFlags
    @           0x4e3501 PyRun_SimpleFileExFlags
    @           0x4159dd Py_Main
    @     0x7f22d9a53bd5 __libc_start_main
    @           0x414b71 (unknown)
非法指令 (core dumped)`
```


+ 相关代码：

```
import paddle.fluid as fluid
import paddle.fluid.framework as framework
import os
import sys

fea_input_dim = 1224

def user_model(is_test = False):
    """ 
    define user model
    """

    x = fluid.layers.data(name='fea_input', shape=[fea_input_dim], dtype='float32')
    y = fluid.layers.data(name='label_target', shape=[1], dtype='float32')

    zero_data = fluid.layers.zeros(shape=[fea_input_dim], dtype='float32')
    xz = fluid.layers.elementwise_add(x, zero_data)

    xz_bn = fluid.layers.batch_norm(input=xz, act=None, epsilon=1e-06, momentum=0.0, is_test=is_test, \
        name='x_bn', moving_mean_name='xz_bn_moving_mean', moving_variance_name='xz_bn_moving_var')
    xz_droped = fluid.layers.dropout(xz_bn, dropout_prob=0.15, is_test=is_test, name='x_droped')

    h1_bn = fluid.layers.batch_norm(input=xz_droped, act='relu', epsilon=1e-06, momentum=0.0, is_test=is_test, \
        name='h1_bn', moving_mean_name='h1_bn_moving_mean', moving_variance_name='h1_moving_var')
    y_predict = fluid.layers.fc(input=h1_bn, size=1, act='sigmoid', name='y_predict')

    cost = fluid.layers.square_error_cost(input=y_predict, label=y)
    avg_loss = fluid.layers.mean(cost)
    if not is_test:
        fluid.backward.append_backward(avg_loss)

    return (x, y, y_predict, avg_loss)

def dump_model_conf():

    startup_program = framework.Program()
    train_program = framework.Program()

    os.makedirs('./test_bug')
    with framework.program_guard(train_program, startup_program):
        fea_input, label_target, predq, sum_loss = user_model(True)

        with open("./test_bug/startup.pb", "w") as f:
            f.write(startup_program.desc.serialize_to_string())
        with open("./test_bug/main.pb", "w") as f:
            f.write(train_program.desc.serialize_to_string())

def load_model_test():
    # main program
    program_desc_str = ''
    with open('./test_bug/main.pb', "rb") as f:
        program_desc_str = f.read()
    program = framework.Program.parse_from_string(program_desc_str)

    # startup program
    startup_prog = None
    with open('./test_bug/startup.pb', "rb") as f:
        program_desc_str = f.read()
    startup_prog = framework.Program.parse_from_string(program_desc_str)

    print('run startup program')
    exe = fluid.Executor(fluid.CPUPlace())
    if startup_prog is not None:
        exe.run(startup_prog)

if __name__ == '__main__':
    dump_model_conf()
    load_model_test()
```


+ 解决方法：

尝试升级一下Fluid，在Fluid 1.2.1的Linux版下，不会出现如上问题。
































