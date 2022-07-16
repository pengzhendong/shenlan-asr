## 第三章作业

## 数据

本次实验所用的数据为 0-9（其中 0 的标签为 `Z`ero）和 O 这 11 个字符的英文录音，每个录音的原始录音文件和 39 维的 MFCC 特征都已经提供。

实验中，每个字符用一个 5 分量的 GMM 来建模。在测试阶段，对于某句话，对数似然最大的模型对应的字符为当前语音数据的预测的标签（target）。

* 训练数据：330 句话，11 个字符，每个字符 30 句话
* 测试数据：110 句话，11 个字符，每个字符 10 句话

`digit_test/digit_train` 里面包含了测试和训练用数据，包括：

1. `wav.scp`：句子 id 到 wav 的路径的映射，所用到的数据 wav 文件的相对路径
2. `feats.scp`：语音识别工具 kaldi 提取的特征文件之一，句子 id 到特征数据真实路径和位置的映射
3. `feats.ark`：语音识别工具 kaldi 提取的特征文件之一，特征实际存储在二进制 ark 文件中
4. `text`：句子 id 到标签的映射，本实验中标签（语音对应的文本）只能是 0-9，O 这 11 个字符

## 程序：

* `kaldi_io.py`：提供了读取 kaldi 特征的功能
* `utils.py`：提供了一个特征读取工具
* `gmm_estimatior.py`：核心代码，提供了 GMM 训练和测试的代码，需要自己完成 GMM 类中 `em_estimator` 和 `calc_log_likelihood` 函数

### em_estimator

#### E 步

即给定观测数据 `X` 和当前估计的参数 $\pi$, $\mu$ 和 $\Sigma$，对每个高斯分量计算数据似然的期望（**E**xpectation）。输出似然期望的维度为 $K\times N$：

* `K`：高斯混合分量数，默认为 5
* `N`：观测数据的样本数（帧数），一个样本对应 39 维 MFCC 特征

似然期望矩阵中的元素 $\gamma(z_{nk})$ 表示第 $n$ 个样本属于第 $k$ 个分量似然的期望（即属于第 $k$ 个分量的似然除以属于每一个分量似然的和）。计算公式如下：

$$
\gamma(\mathrm z_{nk})=\frac{\pi_k\mathcal N(\mathrm x_n|\mu_k,\Sigma_k)}{\sum_k^{K}\pi_k\mathcal N(\mathrm x_n|\mu_k,\Sigma_k)}
$$

``` python
N = len(X)
gamma = np.zeros((self.K, N))
for k in range(self.K):
    for n in range(N):
        gamma[k, n] = self.pi[k] * self.gaussian(X[n, :], k)
gamma /= np.sum(gamma, axis=0)
```

#### M 步

重新估计参数 $\pi$, $\mu$ 和 $\Sigma$，以最大化似然的期望。其中 $N_k$ 表示所有样本属于第 $k$ 个分量的似然期望的和。

$$
N_k=\sum_{n=1}^N\gamma(z_{nk})
$$

``` python
Nk = np.sum(gamma[k, :])
```

$$
\pi_k^\mathrm{new}=\frac{N_k}{N}
$$


``` python
pi_k = Nk / N
```

$$
\mu_k^\mathrm{new}=\frac{1}{N_k}\sum_{n=1}^N\gamma(z_{nk})\mathrm x_n
$$

``` python
mu_k = np.dot(gamma[k, :], X) / Nk
```

$$
\Sigma_k^\mathrm{new}=\frac{1}{N_k}\sum_{n=1}^N\gamma(z_{nk})(\mathrm x_n-\mu_k^\mathrm{new})(\mathrm x_n-\mu_k^\mathrm{new})^\mathrm T
$$

``` python
sigma_k = np.dot(gamma[k, :] * (X - self.mu[k]).T, X - self.mu[k]) / Nk
```

### calc_log_likelihood

重新计算对数似然函数。

``` python
N = len(X)
gamma = np.zeros((self.K, N))
for k in range(self.K):
    for n in range(N):
        gamma[k, n] = self.pi[k] * self.gaussian(X[n, :], k)
log_llh = np.sum(np.log(np.sum(gamma, axis=0)))
```

## 输出：

程序最终输出一个 `acc.txt` 文件，里面记录了识别准确率。
