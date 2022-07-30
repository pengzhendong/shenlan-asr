## 基于 DNN-HMM 的语音识别系统作业

本次课程共有 2 个作业，分别如下所示。

### 作业 1

#### 数据说明

本次实验所用的数据为 0-9（其中0的标签为 Z（Zero））和 O 这 11 个字符的英文录音所提取的 39 维的 MFCC 特征。其中

* 训练数据：330 句话，11 个字符，每个字符 30 句话，训练数据位于 train 目录下。
* 测试数据：110 句话，11 个字符，每个字符 10 句话，测试数据位于 test 目录下。

train/test 目录下各有 3 个文件，分别如下：

* text: 标注文件，每一行第一列为句子id，第二列为标注。
* feats.scp: 特征索引文件，每一行第一列为句子 id，第二列为特征的索引表示。
* feats.ark: 特征实际存储文件，该文件为二进制文件。

#### 实验内容

本实验实现了一个简单的 DNN 的框架，使用 DNN 进行 11 个数字的训练和识别。
实验中使用以上所述的训练和测试数据分别对该 DNN 进行训练和测试。
请阅读 `dnn.py` 中的代码，理解该 DNN 框架，完善 ReLU 激活函数和 FullyConnect 全连接层的前向后向算法。
可以参考 Softmax 的前向和后向实现。`dnn.py` 中代码插入位置为。

1. L 层神经网络，网络结构为：[Linear -> ReLU] * (L - 1) -> Linear -> Softmax
2. 前向传播

  * Linear 层：$Z^{[l]}=W^{[l]}A^{[l-1]}+b^{[l]}$，其中 $A^{[0]}=X$
    ``` python
    def forward(self, A_prev):
        return np.dot(input, self.w.T) + self.b
    ```
  * ReLU 激活层：$A=g(Z)=\mathrm{max}(0,Z)$
    ``` python
    def forward(self, z):
        return np.maximum(z, 0)
    ```
  * Softmax 激活层：$A=g(Z)=\sigma(Z)$
    ``` python
    def forward(self, z):
        z -= z.max(axis=1).reshape(z.shape[0], 1)
        return np.exp(z) / np.sum(np.exp(z), axis=1).reshape(z.shape[0], 1)
    ```
  * DNN 前向传播：
    ``` python
    def forward(self, x)
        self.forward_buf = []
        out = x
        self.forward_buf.append(x)
        for i in range(len(self.layers)):
            out = self.layers[i].forward(x)
            self.forward_buf.append(out)
        return out
    ```

3. 反向传播

  * Linear 层：
    $$
    dW^{[l]}=\frac{\partial\mathcal L}{\partial W^{[l]}}=\frac{1}{m}dZ^{[l]}A^{[l-1]\mathrm T}
    $$

    $$
    db^{[l]}=\frac{\partial\mathcal L}{\partial b^{[l]}}=\frac{1}{m}\sum_{i=1}^mdZ^{[l](i)}
    $$

    $$
    dA^{[l-1]}=\frac{\partial\mathcal L}{\partial A^{[l-1]}}=W^{[l]\mathrm T}dZ^{[l]}
    $$

    ``` python
    def backward(self, A_prev, z, dz):
        m = A_prev.shape[0]
        self.dw = np.dor(dz.T, A_prev) / m
        self.db = np.sum(dz, axis=0) / m
        dA_prev = np.dot(dz, self.w)
        return dA_prev
    ```
  * ReLU 激活层：
    $$
    dZ^{[l]}=dA^{[l]}g'(Z^{[l]})
    $$

    $$
    g'(Z^{[l](i)})=\begin{cases}
    0 & Z^{[l](i)}\leq 0 \\
    1 & \text{otherwise}
    \end{cases}
    $$

    ``` python
    def backward(self, z, A, dA):
        dz = np.array(dA, copy=True)
        dz[A <= 0] = 0
        return dz
    ```
  * DNN 反向传播
    ``` python
    def backward(self, grad):
      L = len(self.layers)
        self.backward_buf = [None] * L
        self.backward_buf[L - 1] = grad
        for i in range(L - 2, -1, -1):
            grad = self.layers[i].backward(self.forward_buf[i],
                                           self.forward_buf[i + 1],
                                           self.backward_buf[i + 1])
            self.backward_buf[i] = grad
    ```

#### 运行和检查

使用如下命令运行该实验，该程序末尾会打印出在测试集上的准确率。假设实现正确，应该得到 95% 以上的准确率，作者的实现分类准确率为 98.18%。

``` sh
python dnn.py
```

#### 拓展

除了跑默认参数之外，读者还可以自己尝试调节一些超参数，并观察这些超参数对最终准确率的影响。如

* 学习率
* 隐层结点数
* 隐层层数

读者还可以基于该框架实现神经网络中的一些基本算法，如：

* sigmoid 和 tanh 激活函数
* dropout
* L2 regularization
* optimizer(Momentum/Adam)
* ...

实现后读者可以在该数字识别任务上应用这些算法，并观察对识别率的影响。

通过调节这些超参数和实现其他的一些基本算法，读者可以进一步认识和理解神经网络。

### 作业 2

基于 Kaldi 理解基于 DNN-HMM 的语音识别系统。请安装 kaldi，并运行 kaldi 下的标准数据集 THCHS30 的实验，该实验如链接所示，

https://github.com/kaldi-asr/kaldi/blob/master/egs/thchs30/s5/run.sh

[THCHS30](http://www.openslr.org/18)是清华大学开源的一个中文数据集，总共 30 小时。请基于该数据集，基于 kaldi 下该数据集的标注脚本，梳理基于 DNN-HMM 的语音识别系统的**流程，其有哪些步骤，每一步的输入、输出，步骤间的相互关系**等，可以把自己的理解流程化、图形化、文字化的记录下来，写下来。
