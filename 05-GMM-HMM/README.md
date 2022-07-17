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

### p2

估计模型参数,不使用前向后向算法计算统计量，而是用 viterbi 解码得到的最优的一条序列来计算统计量，叫做 viterbi-EM。

给定 align（viterbi 解码的最优状态序列)，原始语音和 GMM 的初始值，更新 GMM 参数。完成 `gmm_util.C` 中两处代码。比较 `p2a.gmm` 和 `p2a.gmm.ref`。

``` bash
$ bash lab2_p2a.sh
```

### p3

用前向后向算法来估计参数，完成 `lab2_fb.C` 中的两处代码。分别比较 `p3a_chart.dat` 和 `p3a_chart.ref`；`p3b.gmm` 和 `p3b.gmm.ref`。

运行：
* `bash lab2_p3a.sh`: 1 条数据，1 轮迭代
* `bash lab2_p3b.sh`: 22 条数据，1 轮迭代
* `bash lab2_p3c.sh`: 22 条数据，20 轮迭代
* `bash lab2_p3d.sh`: 使用 p3c 的训练的模型，使用 viterbi 算法解码，结果应该和 p1b 的结果一样
