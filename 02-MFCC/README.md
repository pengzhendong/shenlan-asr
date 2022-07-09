## 第二章作业

给定一段音频，请提取 12 维 MFCC 特征和 23 维 FBank。

阅读代码预加重、分帧、加窗部分，完善作业代码中 FBank 特征提取和 MFCC 特征提取部分。

### 运行

``` bash
$ pip install -r requirements.txt
$ python mfcc.py
```

### 代码说明

#### FBank

``` python
def fbank(spectrum, num_filter=23):
    feats = np.zeros(spectrum.shape[0], num_filter)
    """
        FINISH by YOURSELF
    """
    return feats
```

#### MFCC

``` python
def mfcc(fbank, num_mfcc=12):
    feats = np.zeros((fbank.shape[0], num_mfcc))
    """
        FINISH by YOURSELF
    """
    return feats
```
