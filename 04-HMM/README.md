## 第四章作业

考虑盒子和球模型 $\lambda=(A,B,\pi)$，状态集合 $Q=\{1,2,3\}$，观测集合 $V=\{红,白\}$，

$$
A=\begin{bmatrix}
0.5 & 0.2 & 0.3 \\
0.3 & 0.5 & 0.2 \\
0.2 & 0.3 & 0.5
\end{bmatrix},
B=\begin{bmatrix}
0.5 & 0.5 \\
0.4 & 0.6 \\
0.7 & 0.3
\end{bmatrix},
\pi=(0.2,0.4,0.4)^\mathrm T
$$

设 $T=3,O=(红,白,红)$。

* 请用 Python 编程实现前向算法和后向算法，分别计算 $P(O|\lambda)$；
* 请用 Python 编程实现 Viterbi 算法，求最优状态序列，即最优路径 $I^\*=(i_1^\*,i_2^\*,i_3^\*)$。

### 前向算法

给定隐马尔可夫模型 $\lambda$，定义时刻 $t$ 部分观测序列为 $o_1,o_2,...,o_t$ 且状态为 $q_i$ 的概率为前向概率，记作（可省略 $\lambda$）：

$$
\alpha_t(i)=P(o_1,o_2,...,o_t,i_t=q_i|\lambda)
$$

* 输入：隐马尔可夫模型 $\lambda$，观测序列 $O$；
* 输出：观测序列概率 $P(O|\lambda)$。

1. 初值

$$
\alpha_1(i)=\pi_ib_i(o_1),\quad i=1,2,...,N
$$

``` python
alpha[0, :] = pi * B[:, O[0]]
```

2. 递推，对于 $t=1,2,...,T-1$

$$
\alpha_{t+1}(i)=\Bigg[\sum_{j=1}^N\alpha_t(j)a_{ji}\Bigg]b_i(o_{t+1}),\quad i=1,2,...,N
$$

``` python
for t in range(1, T):
    alpha[t, :] = np.sum(alpha[t - 1, :] * A.T, axis=1) * B[:, O[t]]
```

3. 终止

$$
P(O|\lambda)=\sum_{i=1}^N\alpha_T(i)
$$

``` python
p = np.sum(alpha[T - 1, :])
```

### 后向算法

给定隐马尔可夫模型 $\lambda$，定义时刻 $t$ 状态为 $q_i$ 的条件下，从 $t+1$ 到 $T$ 的部分观测序列为 $o_{t+1},o_{t+2},...,o_T$ 的概率为后向概率，记作（可省略 $\lambda$）：

$$
\beta_t(i)=P(o_{t+1},o_{t+2},...,o_T|i_t=q_i,\lambda)
$$

* 输入：隐马尔可夫模型 $\lambda$，观测序列 $O$；
* 输出：观测序列概率 $P(O|\lambda)$。

1. 初值

$$
\beta_T(i)=1,\quad i=1,2,...,N
$$

``` python
beta[T - 1, :] = 1
```

2. 递推，对于 $t=T-1,T-2,...,1$

$$
\beta_t(i)=\sum_{j=1}^Na_{ij}b_j(o_{t+1})\beta_{t+1}(j),\quad i=1,2,...,N
$$

``` python
for t in reversed(range(T - 1)):
    beta[t, :] = np.sum(A * B[:, O[t + 1]].T * beta[t + 1, :], axis=1)
```

3. 终止

$$
P(O|\lambda)=\sum_{i=1}^N\pi_ib_i(o_1)\beta_1(i)
$$

``` python
p = np.sum(pi * B[:, O[0]] * beta[0, :])
```

### Viterbi 算法

已知模型 $\lambda=(A,B,\pi)$ 和观测序列 $O=(o_1,o_2,...,o_T)$，计算使概率 $P(I|O)$ 最大的状态序列 $I=(i_1,i_2,...,i_T)$。

首先导入两个变量 $\delta$ 和 $\psi$，定义在时刻 $t$ 状态为 $i$ 的所有单个路径 $(i_1,i_2,...i_t)$ 中概率最大值为

$$
\delta_t(i)=\max\limits_{i_1,i_2,...,i_{t-1}}P(i_t=i,i_{t-1},...,i_1,o_t,
...,o_1|\lambda),\quad i=1,2,...,N
$$

由定义可得变量 $\delta$ 的递推公式：

$$
\begin{aligned}
\delta_{t+1}(i)&=\max\limits_{i_1,i_2,...,i_t}P(i_{t+1}=i,i_t,...,i_1,o_{t+1},
...,o_1|\lambda) \\
&=\max\limits_{1\leq j\leq N}[\delta_t(j)a_{ji}]b_i(o_{t+1}),\quad i=1,2,...,N;t=1,2,...,T-1
\end{aligned}
$$

定义在时刻 $t$ 状态为 $i$ 的所有单个路径 $(i_1,i_2,...,i_{t-1},i)$ 中概率最大的路径的第 $t-1$ 个结点为

$$
\psi_t(i)=\arg\max\limits_{1\leq j\leq N}[\delta_{t-1}(j)a_{ji}],\quad i=1,2,...,N
$$

* 输入：模型 $\lambda=(A,B,\pi)$ 和观测 $O=(o_1,o_2,..,o_T)$；
* 输出：最优路径 $I^\*=(i_1^\*,i_2^\*,...i_T^\*)$。

1. 初始化

$$
\delta_1(i)=\pi_ib_i(o_1),\quad i=1,2,...,N
$$

$$
\psi_1(i)=0,\quad i=1,2,...,N
$$

``` python
delta[0, :] = pi * B[:, O[0]]
psi = np.zeros(delta.shape, dtype=int)
```

2. 递推，对于 $t=2,3,...,T$

$$
\delta_t(i)=\max\limits_{1\leq j\leq N}[\delta_{t-1}(j)a_{ji}]b_i(o_t),\quad i=1,2,...,N
$$

$$
\psi_t(i)=\arg\max\limits_{1\leq j\leq N}[\delta_{t-1}(j)a_{ji}],\quad i=1,2,...,N
$$

``` python
for t in range(1, T):
    delta[t, :] = np.max(delta[t - 1, :] * A.T * B[:, O[t]], axis=1)
    psi[t, :] = np.argmax(delta[t - 1, :] * A.T, axis=1) + 1
```

3. 终止

$$
P^*=\max\limits_{1\leq i\leq N}\delta_T(i)
$$

$$
i_T^*=\arg\max\limits_{1\leq i\leq N}\delta_T(i)
$$

``` python
P = np.max(delta[T - 1, :])
i[T - 1] = np.argmax(delta[T - 1, :]) + 1
```

4. 最优路径回溯，对于 $t=T-1,T-2,...,1$

$$
i_t^\*=\psi_{i+1}(i_{t+1}^\*)
$$

``` python
for t in reversed(range(T - 1)):
    i[t] = psi[t + 1, i[t + 1] - 1]
```

求得最优路径 $I^\*=(i_1^\*,i_2^\*,...i_T^\*)$。
