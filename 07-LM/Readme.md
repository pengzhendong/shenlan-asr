## 第七章作业

### 说明
请根据 `实验指导书.pdf`: part2.2 和 part4 分别完成 N-gram 计数和 Witten-Bell 算法的编写。

编译文件：Makefile

### 提供的 C++ 文件介绍：

1. `main.C`：入口函数
2. `util.{H,C}`：提供命令行解析，读取和写出数据等功能,不必仔细阅读，可以掠过。
3. `lang_model.{H,C}`：LM 类定义，本实验主要部分内容，需要完成 .C 文件中 count_sentence_ngrams() 和 get_prob_witten_bell() 函数。
4. `lab3_lm.{H,C}`：语言模型实验的 wrapper 函数。

#### count_sentence_ngrams

* `m_n`：Ngram model 中的 N
* `m_predCounts`：例如：$\mathrm{count}(\text{OF THE})$
* `m_histCount`：例如：$\mathrm{count_{hisf}}(\text{OF})=\sum_w\text{OF THE }w$
* `m_histOnePlusCounts`：$N_{1+}(w_{i-1})$，出现在 $w_{i-1}$ 后面至少一次的词的总次数

``` c++
assert(m_n > 0);
int wordCnt = wordList.size();
for (int wordIdx = m_n - 1; wordIdx < wordCnt; ++wordIdx) {
  for (int n = 1; n <= m_n; ++n) {
    auto end = wordList.begin() + wordIdx;
    auto begin = end - m_n + n;
    vector<int> histNgram(begin, end);
    vector<int> ngram(begin, end + 1);

    m_histCounts.incr_count(histNgram);
    if (m_predCounts.incr_count(ngram) == 1) {
      m_histOnePlusCounts.incr_count(histNgram);
    }
  }
}
```

#### get_prob_witten_bell

$$
P_{WB}(w_i|w_{i-1})=\frac{c_h(w_{i-1})}{c_h(w_{i-1})+N_{1+}(w_{i-1})}P_{MLE}(w_i|w_{i-1})+\frac{N_{1+}(w_{i-1})}{c_h(w_{i-1})+N_{1+}(w_{i-1})}P_{\text{backoff}}(w_i)
$$

$$
\begin{aligned}
P_{\text{backoff}}(w_i)&=P_{WB}(w_i) \\
&=\frac{c_h(\epsilon)}{c_h(\epsilon)+N_{1+}(\epsilon)}P_{MLE}(w_i)+\frac{N_{1+}(\epsilon)}{c_h(\epsilon)+N_{1+}(\epsilon)}\frac{1}{|V|}
\end{aligned}
$$

* $c_h(w_{i-1})$：`m_histCounts`
* $N_{1+}(w_{i-1})$：`m_histOnePlusCounts`
* $P_{MLE}(w_i|w_{i-1})$：`m_predCounts / m_histCounts`

``` c++
int predCnt = m_predCounts.get_count(ngram);
vector<int> histNgram(ngram.begin(), ngram.end() - 1);
int histCnt = m_histCounts.get_count(histNgram);
int histOnePlusCnt = m_histOnePlusCounts.get_count(histNgram);

double lambda = 0.0, PMLE = 0.0;
if (histCnt > 0) {
  lambda = (double)histCnt / (histCnt + histOnePlusCnt);
  PMLE = (double)predCnt / histCnt;
}

double PBackoff;
if (ngram.size() == 1) {
  PBackoff = 1.0 / vocSize;
} else {
  PBackoff =
      get_prob_witten_bell(vector<int>(ngram.begin() + 1, ngram.end()));
}
retProb = lambda * PMLE + (1 - lambda) * PBackoff;
```

### 数据文件：

* 字典：lab3.syms
* 训练集：minitrain.txt 和 minitrain2.txt
* 测试集：test1.txt 和 test2.txt

### bash 文件：

* lab3_p1{a,b}.sh：测试 N-gram 计数
* lab3_p3{a,b}.sh：测试 Witten-Bell smoothing 算法
