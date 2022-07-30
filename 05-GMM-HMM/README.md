## 基于GMM-HMM的语音识别系统

### 要求和注意事项

1. 认真读 `lab2.pdf`, 思考 `lab2.txt` 中的问题
2. 理解数据文件
3. ref 文件作为参考输出，用 `diff` 命令检查自己的实现得到的输出和 ref 是否完全一致
4. 实验中实际用的 GMM 其实都是单高斯
5. 阅读 `util.h` 里面的注释，Graph 的注释有如何遍历 graph 中 state 上所有的 arc 的方法
6. 完成代码
    * `lab2_vit.C` 中一处代码
    * `gmm_util.C` 中两处代码
    * `lab2_fb.C` 中两处代码

### 作业说明

### 安装

该作业依赖 g++, boost 库和 make 命令，按如下方式安装：
* MacOS: `brew install boost` (MacOS 下 g++/make 已内置）
* Linux(Ubuntu): `sudo apt-get install make g++ libboost-all-dev`
* Windows: 请自行查阅如何安装作业环境。

### 编译

对以下三个问题，均使用该方法编译。

``` bash
$ make -C src
```

### p1

完成 `lab2_vit.C` 中的用 viterbi 解码代码。比较程序运行结果 `p1a.chart` 和参考结果 `p1a.chart.ref`，浮点数值差在一定范围内即可。

``` bash
$ bash lab2_p1a.sh
$ bash lab2_p1b.sh
```

1. 初始化 Chart，每个元素包含 $\delta_t(i)$ 和 $\psi_t(i)$

| frmIdx \ stateIdx | 0       | 1       | ... | stateCnt - 1 |
| ----------------- | ------- | ------- | --- | ------------ |
| 0                 | (0, -1) | (0, -1) | ... | (0, -1)      |
| 1                 | (0, -1) | (0, -1) | ... | (0, -1)      |
| ...               | ...     | ...     | ... | ...          |
| frmCnt            | (0, -1) | (0, -1) | ... | (0, -1)      |

``` c++
for (int stateIdx = 0; stateIdx < stateCnt; ++stateIdx) {
  chart(0, stateIdx).assign(g_zeroLogProb, -1);
}
chart(0, graph.get_start_state()).assign(log(1.0), -1);
```

2. 递推。对 $t=1,2,...,T$

$$
\delta_t(i)=\max\limits_{1\leq j\leq N}[\delta_{t-1}(j)a_{ji}]b_i(o_t),\quad i=1,2,...,N
$$

$$
\psi_t(i)=\arg\max\limits_{1\leq j\leq N}[\delta_{t-1}(j)a_{ji}],\quad i=1,2,...,N
$$

* $T$：frmCnt
* $N$：stateCnt

``` c++
for (int frmIdx = 1; frmIdx <= frmCnt; ++frmIdx) {
  for (int stateIdx = 0; stateIdx < stateCnt; ++stateIdx) {
    int arcCnt = graph.get_arc_count(stateIdx);
    int curArcId = graph.get_first_arc_id(stateIdx);
    for (int arcIdx = 0; arcIdx < arcCnt; ++arcIdx) {
      Arc arc;
      int nextArcId = graph.get_arc(curArcId, arc);
      int dstState = arc.get_dst_state();
      double logProb = chart(frmIdx - 1, stateIdx).get_log_prob() +
                        arc.get_log_prob() +
                        gmmProbs(frmIdx - 1, arc.get_gmm());

      if (logProb > chart(frmIdx, dstState).get_log_prob()) {
        chart(frmIdx, dstState).assign(logProb, curArcId);
      }
      curArcId = nextArcId;
    }
  }
}
```

### p2

估计模型参数,不使用前向后向算法计算统计量，而是用 viterbi 解码得到的最优的一条序列来计算统计量，叫做 viterbi-EM。

给定 align（viterbi 解码的最优状态序列)，原始语音和 GMM 的初始值，更新 GMM 参数。完成 `gmm_util.C` 中两处代码。比较 `p2a.gmm` 和 `p2a.gmm.ref`。

``` bash
$ bash lab2_p2a.sh
```

1. 遍历 $N$ 个样本的过程中，对于高斯混合分量 $k$，计算统计量：

$$
N_k=\sum_{n=1}^N\gamma(z_{nk})
$$

对于样本 $\mathrm x_n$ 的每个维度，计算一阶统计量 $\sum_{n=1}^N\gamma(z_{nk})\mathrm x_n$ 和二阶统计量 $\sum_{n=1}^N\gamma(z_{nk})\mathrm x_n^2$。

``` c++
m_gaussCounts[gaussIdx] += posterior;
for (int dimIdx = 0; dimIdx < dimCnt; ++dimIdx) {
  m_gaussStats1(gaussIdx, dimIdx) += posterior * feats[dimIdx];
  m_gaussStats2(gaussIdx, dimIdx) += posterior * pow(feats[dimIdx], 2);
}
```

2. 计算每个维度的均值和方差

$$
\mu_k=\frac{1}{N_k}\sum_{n=1}^N\gamma(z_{nk})\mathrm x_n
$$

$$
\begin{aligned}
\sigma_k&=\frac{1}{N_k}\sum_{n=1}^N\gamma(z_{nk})(\mathrm x_n-\mu_k)^2 \\
&=\frac{1}{N_k}\sum_{n=1}^N\gamma(z_{nk})(\mathrm x_n^2-2\mathrm x_n\mu_k+\mu_k^2) \\
&=\frac{1}{N_k}\sum_{n=1}^N\gamma(z_{nk})\mathrm x_n^2-2\mu_k\frac{1}{N_k}\sum_{n=1}^N\gamma(z_{nk})\mathrm x_n+\mu_k^2\frac{1}{N_k}\sum_{n=1}^N\gamma(z_{nk}) \\
&=\frac{1}{N_k}\sum_{n=1}^N\gamma(z_{nk})\mathrm x_n^2-2\mu_k^2+\mu_k^2\frac{1}{N_k}N_k \\
&=\frac{1}{N_k}\sum_{n=1}^N\gamma(z_{nk})\mathrm x_n^2-\mu_k^2
\end{aligned}
$$

``` c++
double newMean = 0;
double newVar = 0;
for (int gaussIdx = 0; gaussIdx < gaussCnt; ++gaussIdx) {
  double occupancy = m_gaussCounts[gaussIdx];
  for (int dimIdx = 0; dimIdx < dimCnt; ++dimIdx) {
    double gaussState1 = m_gaussStats1(gaussIdx, dimIdx);
    double gaussState2 = m_gaussStats2(gaussIdx, dimIdx);
    newMean = gaussState1 / occupancy;
    newVar = gaussState2 / occupancy - pow(newMean, 2);

    m_gmmSet.set_gaussian_mean(gaussIdx, dimIdx, newMean);
    m_gmmSet.set_gaussian_var(gaussIdx, dimIdx, newVar);
  }
}
```

### p3

用前向后向算法来估计参数，完成 `lab2_fb.C` 中的两处代码。分别比较 `p3a_chart.dat` 和 `p3a_chart.ref`；`p3b.gmm` 和 `p3b.gmm.ref`。

运行：
* `bash lab2_p3a.sh`: 1 条数据，1 轮迭代
* `bash lab2_p3b.sh`: 22 条数据，1 轮迭代

1. 初始化 Chart，每个元素包含前向概率和后向概率

| frmIdx \ stateIdx | 0      | 1      | ... | stateCnt - 1 |
| ----------------- | ------ | ------ | --- | ------------ |
| 0                 | (0, 0) | (0, 0) | ... | (0, 0)       |
| 1                 | (0, 0) | (0, 0) | ... | (0, 0)       |
| ...               | ...    | ...    | ... | ...          |
| frmCnt            | (0, 0) | (0, 0) | ... | (0, 0)       |

``` c++
for (int stateIdx = 0; stateIdx < stateCnt; ++stateIdx) {
  chart(0, stateIdx).set_forw_log_prob(g_zeroLogProb);
  chart(0, stateIdx).set_back_log_prob(g_zeroLogProb);
}
chart(0, graph.get_start_state()).set_forw_log_prob(0);
```

2. 前向递推，对于 $t=1,2,...,T$

$$
\alpha_t(i)=\Bigg[\sum_{j=1}^N\alpha_{t-1}(j)a_{ji}\Bigg]b_i(o_t),\quad i=1,2,...,N
$$

``` c++
for (int frmIdx = 1; frmIdx <= frmCnt; ++frmIdx) {
  for (int stateIdx = 0; stateIdx < stateCnt; ++stateIdx) {
    int arcCnt = graph.get_arc_count(stateIdx);
    int arcId = graph.get_first_arc_id(stateIdx);
    for (int arcIdx = 0; arcIdx < arcCnt; ++arcIdx) {
      Arc arc;
      arcId = graph.get_arc(arcId, arc);
      int dstState = arc.get_dst_state();
      double logProb = chart(frmIdx - 1, stateIdx).get_forw_log_prob() +
                        arc.get_log_prob() +
                        gmmProbs(frmIdx - 1, arc.get_gmm());

      logProb = add_log_probs(vector<double>{
          logProb, chart(frmIdx, dstState).get_forw_log_prob()});
      chart(frmIdx, dstState).set_forw_log_prob(logProb);
    }
  }
}
```

3. 反向递推，对于 $t=T-1,T-2,...,0$

$$
\beta_t(i)=\sum_{j=1}^Na_{ij}b_j(o_{t+1})\beta_{t+1}(j),\quad i=1,2,...,N
$$

``` c++
for (int frmIdx = frmCnt - 1; frmIdx >= 0; --frmIdx) {
  for (int stateIdx = 0; stateIdx < stateCnt; ++stateIdx) {
    int arcCnt = graph.get_arc_count(stateIdx);
    int arcId = graph.get_first_arc_id(stateIdx);
    for (int arcIdx = 0; arcIdx < arcCnt; ++arcIdx) {
      Arc arc;
      arcId = graph.get_arc(arcId, arc);
      int dstState = arc.get_dst_state();
      double logProb = chart(frmIdx + 1, dstState).get_back_log_prob() +
                        arc.get_log_prob() + gmmProbs(frmIdx, arc.get_gmm());

      logProb = add_log_probs(vector<double>{
          logProb, chart(frmIdx, stateIdx).get_back_log_prob()});
      chart(frmIdx, stateIdx).set_back_log_prob(logProb);
    }
  }
}
```

4. 计算后验概率

``` c++
for (int frmIdx = frmCnt - 1; frmIdx >= 0; --frmIdx) {
  for (int stateIdx = 0; stateIdx < stateCnt; ++stateIdx) {
    int arcCnt = graph.get_arc_count(stateIdx);
    int arcId = graph.get_first_arc_id(stateIdx);
    for (int arcIdx = 0; arcIdx < arcCnt; ++arcIdx) {
      Arc arc;
      arcId = graph.get_arc(arcId, arc);
      int dstState = arc.get_dst_state();
      double logProb = chart(frmIdx, stateIdx).get_forw_log_prob() +
                        arc.get_log_prob() + gmmProbs(frmIdx, arc.get_gmm()) +
                        chart(frmIdx + 1, dstState).get_back_log_prob();

      double arcPosterior = exp(logProb - uttLogProb);
      gmmCountList.push_back(GmmCount(arc.get_gmm(), frmIdx, arcPosterior));
    }
  }
}
```