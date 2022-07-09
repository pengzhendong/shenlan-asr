import os
import sys
import cv2
import numpy as np
import matplotlib
from mfcc import plot_spectrogram
from skimage.metrics import structural_similarity as ssim

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read_feats(file, dim):
    return np.loadtxt(file,
                      dtype=float,
                      converters={0: lambda s: s[1:]},
                      usecols=np.r_[range(0, dim)])


def images_ssim(img_a, img_b):
    a = cv2.cvtColor(cv2.imread(img_a), cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(cv2.imread(img_b), cv2.COLOR_BGR2GRAY)
    return ssim(a, b)


if __name__ == '__main__':
    output_dir = sys.argv[1]
    fbank_sm = read_feats(f'{output_dir}/test.fbank', 23)
    mfcc_sm = read_feats(f'{output_dir}/test.mfcc', 12)
    assert fbank_sm.shape[0] == mfcc_sm.shape[0] == 356
    exists_fbank = os.path.exists(f'{output_dir}/fbank.png')
    exists_mfcc = os.path.exists(f'{output_dir}/mfcc.png')

    # 提交了 fbank 的图片，直接计算结构相似度
    fbank_ssim = 0
    if exists_fbank:
        fbank_ssim = images_ssim(f'{output_dir}/fbank.png', 'fbank.png')
        print('fbank structural similarity: ', fbank_ssim)

    # 没有提交 mfcc 的图片，则生成 mfcc 图片
    if not exists_mfcc:
        plot_spectrogram(mfcc_sm.T, 'MFCC', f'{output_dir}/mfcc.png')
    # 计算 mfcc 图片的结构相似度
    mfcc_ssim = images_ssim(f'{output_dir}/mfcc.png', 'mfcc.png')
    print('mfcc structural similarity: ', mfcc_ssim)

    # 没有提交 fbank 的图片或者提交的相似度低于 0.9，转置后重新生成 fbank 图片再计算相似度
    if not exists_fbank or fbank_ssim < 0.9:
        plot_spectrogram(fbank_sm.T, 'Filter Bank',
                         f'{output_dir}/fbank_T.png')
        fbank_ssim = images_ssim(f'{output_dir}/fbank_T.png', 'fbank.png')
        print('fbank structural similarity after transpose: ', fbank_ssim)
