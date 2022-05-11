# covid_emr

## Model Zoo

### Machine Learning Models

- [ ] Logistic regression (LR)
- [ ] Random forest (RF)
- [ ] Decision tree (DT)
- [ ] Gradient Boosting Decision Tree (GBDT)
- [ ] XGBoost

### Deep Learning Models

- [ ] Multi-layer perceptron (MLP)
- [x] Recurrent neural network (RNN)
- [x] Long-short term memory network (LSTM)
- [x] Gated recurrent units (GRU)
- [x] Temporal convolutional networks
- [x] Transformer

#### Temporal convolutional networks

```python
model = models.TemporalConvNet(num_inputs=76, num_channels=76, num_classes=1, max_seq_length=48)
x = torch.randn(batch_size, 48, 76)
output = model(x)
```

TODO: check the meaning of `max_seq_length`, and whether `num_inputs = num_channels`

### EHR Predictive Models

- [x] RETAIN
- [x] StageNet
- [ ] Dr. Agent
- [x] AdaCare
- [x] ConCare
- [ ] GRASP
- [ ] (CovidCare)

#### RETAIN

```python
model = models.RETAIN(76)
x = torch.randn(batch_size, 48, 76)
lengths = torch.ones_like(x[:,0,0])
output, alpha, beta = model(x, lengths)
```

TODO: See PyHealth Version [RETAIN](https://github.com/zzachw/PyHealth/blob/master/pyhealth/models/sequence/retain.py), [Current one](https://github.com/ast0414/pytorch-retain/blob/master/retain.py)

#### AdaCare

```python
model = models.AdaCare()
"""
default AdaCare hyperparameter:

hidden_dim=128, kernel_size=2, kernel_num=64, input_dim=76, output_dim=1, dropout=0.5, r_v=4, r_c=4, activation='sigmoid', device='cuda')

In mimic-iii dataset, the input shape is T=48, H=76
"""
x = torch.randn(batch_size, 48, 76)
output, inputse_att = model(x, device)
```

TODO: 不需要attention matrix的输出

#### StageNet

```python
model = models.StageNet(input_dim=76, hidden_dim=384, conv_size=10, output_dim=1, levels=3, dropconnect=0.5, dropout=0.5, dropres=0.3)
x = torch.randn(batch_size, 48, 76)
time = torch.randn(batch_size, 48)
output = model(x, time, device)
```

TODO: check `time` tensor

See [stagenet in pyhealth](https://github.com/zzachw/PyHealth/blob/master/pyhealth/models/sequence/stagenet.py)

### Dr.Agent

```python
model = models.Agent(
    cell="gru",
    use_baseline=True,
    n_actions=10,
    n_units=32,
    n_input=76,
    demo_dim=12,
    fusion_dim=20,
    n_hidden=32,
    n_output=1,
    dropout=0.3,
    lamda=1.0,
    device=device,
).to(device)
x = torch.randn(batch_size, 48, 76)
demo = torch.randn(batch_size, 12)
lengths = torch.ones_like(x[:, 0, 0])
output = model(x, demo)
```

[Link](https://github.com/v1xerunt/Dr.Agent/blob/master/train_mortality.py)

TODO: check `_mortality` / `_los` 2 models, and `loss_fn`

### GRASP

```python
device = torch.device("cpu")
batch_size = 64

input_dim = 76
hidden_dim = 32
output_dim = 1
cluster_num = 32  # should be smaller than batch_size
dropout = 0.5
block = "GRU"

model = models.MAPLE(
    device=device,
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    output_dim=output_dim,
    cluster_num=cluster_num,
    dropout=dropout,
    block=block,
)

x = torch.randn(batch_size, 48, 76)
demo = torch.randn(batch_size, 12)
lengths = torch.randint(1, 20, (batch_size,))
# print(lengths)
output, _ = model(x, 0, lengths)
```

- [Link](https://github.com/choczhang/GRASP/blob/main/MAPLE-code.ipynb)
- TODO: check the model/training pipeline


## TODOs

### Data preprocessing

- [x] 在patient-wise上计算特征的缺失率
- [ ] 特征的记录频率、次数统计 (确定筛选规则, missing rate的阈值)
- [ ] 在各数据集上导出相同格式(shape)和含义的x/y
  - [ ] x: [patient_number, max_timestep_length, [x_demo | x_lab]] 即对于x中的feature在数据预处理阶段即做好demographic和labtest的concat工作 (写模型时更简单, 只需要输入x即可)。假设有100个患者，其中患者最多有13次visit，demographic有2维，lab test有10维，则该tensor shape为：(100, 13, 2+10)
  - [ ] y: [patient_number, max_timestep_length, [outcome, los]] 若有100个患者，他们的最长就诊次数是10次，则y的shape为[100, 10, 2]
  - [ ] visit_length: [patient_number] 记录每个患者的就诊次数。假设有100个患者，则该tensor shape为：(100, )

**分工:**

- [ ] tj: zyh
- [ ] hm: wwq
- [ ] some basic observation (需要 code and document, 以jupyter notebook format就蛮好的): wwq

**备注:**

- 就实现上，我认为比较好的方法就是在之前的基础上做一下转换。之前是每个都拆分了出来，算出了单独的demo tensor, lab test tensor, visit length tensor, outcome/los tensor。这次我们在该基础上把concat的步骤写进data preprocessing 的 pipeline中，这样对于读者而言可能可读性会更好。
- 做完之后再交叉check一遍，并统一代码风格。
- 暂定这部分在5月10号前确认好。
- 因为之前做过一遍能得到以上所有信息的数据处理工作，所以应该是挺快的。
- 不确定的点是: 上述设计的input shape有没有更优的方案，但这部分问题不大，因为总能通过简单的切片 + reshape等方法转成各种format。

### Models

ml models:

- [ ] Logistic regression (LR)
- [ ] Random forest (RF)
- [ ] Decision tree (DT)
- [ ] Gradient Boosting Decision Tree (GBDT)
- [ ] XGBoost

dl models:

- [ ] Multi-layer perceptron (MLP)  wwq
- [x] Recurrent neural network (RNN)
- [x] Long-short term memory network (LSTM)
- [x] Gated recurrent units (GRU)
- [x] Temporal convolutional networks
- [ ] Transformer  zyh 加个mask

ehr models:

- [ ] RETAIN wwq
- [ ] StageNet zyh
- [ ] Dr. Agent zyh
- [ ] AdaCare wwq
- [ ] ConCare wwq
- [ ] GRASP zyh

**分工:**

- 具体的model分配见上
- ml models的整个pipeline: zyh
- balanced metrics: wwq

**备注:**

对于dl/ehr models，分成以下几步:

- 能够将模型的input/output shape都对上，知道模型的输入是哪些量，输出的是什么量
- (检测model hidden state中是否有用到future information。检测是否有做过转置，去计算feature-wise attention，以及attention部分是否有做mask处理。) 其实也可以先不用太纠这个点，即假设model的output中的每个time step上的hidden state所代表的均是当前/结合了之前的信息。我们主要是在calculate loss和evaluate的部分保证了没有泄露future information。
- model的output统一输出2 value: outcome(1st) and los(2nd)。
- 对不同的model，对一些hyper parameter和命名的规则进行统一的调整。如act_layer / activation_function, dropout/ drop / drop_rate / dropout_rate 等命名convention做规范。

### Toolkit Architecture Design

- 6月前的基本目标: 单机单卡，能完整跑完实验pipeline的architecture
- 后续迭代目标:
  - checkpoints load and resume
  - distributed training
  - 实验结果的monitoring and management
  - attention visualization
  - ... etc

**分工:**

- [ ] 在上述规范后的x/y tensor shape上，能完整跑multi-target task的pipeline的basic architecture: zyh
- [ ] 后续的模块化/抽象封装/继承等future work都不着急

### Done

- [x] 评估时按每一次visit对应的预测结果 (mortality outcome / los) 的结果进行评估
- [x] 计算 loss 时才取平均，评估时将patient每一次visit时的结果取平均没有意义
- [x] Multi-task learning: 2 branches: mortality and LOS prediction (2 tasks)
- [x] 检查 predict LOS 时为何不收敛。(因为metrics写反了，选了worst model，已解决)
