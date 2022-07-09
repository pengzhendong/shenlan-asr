import librosa
import numpy as np
from scipy.fftpack import dct
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_spectrogram(spec, note, file_name):
    """Draw the spectrogram picture
        :param spec: a feature_dim by num_frames array(real)
        :param note: title of the picture
        :param file_name: name of the file
    """
    fig = plt.figure(figsize=(20, 5))
    heatmap = plt.pcolor(spec)
    fig.colorbar(mappable=heatmap)
    plt.xlabel('Time(s)')
    plt.ylabel(note)
    plt.tight_layout()
    plt.savefig(file_name)


def preemphasis(signal, coeff=0.97):
    """Perform preemphasis on the input signal
        :param signal: The signal to filter
        :param coeff: The preemphasis coefficient. 0 is no filter
        :returns: the filtered signal
    """
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])


def enframe(signal, frame_len=400, frame_shift=160, win=None):
    """Enframe with Hamming widow function
        :param signal: The signal be enframed
        :param frame_len: default is 400 / 16000 * 1000 = 25ms
        :param frame_shift: default is 160 / 16000 * 1000 = 10ms
        :param win: window function, default Hamming
        :returns: the enframed signal, num_frames by frame_len array
    """
    if win is None:
        win = np.hamming(frame_len)
    num_samples = signal.size
    num_frames = np.floor((num_samples - frame_len) / frame_shift) + 1
    frames = np.zeros((int(num_frames), frame_len))
    for i in range(int(num_frames)):
        frames[i, :] = signal[i * frame_shift:i * frame_shift + frame_len]
        frames[i, :] = frames[i, :] * win
    return frames


def get_spectrum(frames, fft_len=512):
    """Get spectrum using fft
        :param frames: the enframed signal, num_frames by frame_len array
        :param fft_len: FFT length
        :returns: spectrum, a num_frames by fft_len/2+1 array (real)
    """
    cFFT = np.fft.fft(frames, n=fft_len)
    valid_len = int(fft_len / 2) + 1
    spectrum = np.abs(cFFT[:, 0:valid_len])
    return spectrum


def fbank(spectrum, num_filter=23):
    """Get mel filter bank feature from spectrum
        :param spectrum: a num_frames by fft_len/2+1 array(real)
        :param num_filter: mel filters number
        :returns: fbank feature, a num_frames by num_filter array
        DON'T FORGET LOG OPRETION AFTER MEL FILTER!
    """

    feats = np.zeros(spectrum.shape[0], num_filter)
    """
        FINISH by YOURSELF
    """
    return feats


def mfcc(fbank, num_mfcc=12):
    """Get mfcc feature from fbank feature
        :param fbank: a num_frames by  num_filter array(real)
        :param num_mfcc: mfcc number
        :returns: mfcc feature, a num_frames by num_mfcc array
    """

    feats = np.zeros((fbank.shape[0], num_mfcc))
    """
        FINISH by YOURSELF
    """
    return feats


def write_file(feats, file_name):
    """Write the feature to file
        :param feats: a num_frames by feature_dim array(real)
        :param file_name: name of the file
    """
    f = open(file_name, 'w')
    (row, col) = feats.shape
    for i in range(row):
        f.write('[')
        for j in range(col):
            f.write(str(feats[i, j]) + ' ')
        f.write(']\n')
    f.close()


def main():
    wav, fs = librosa.load('test.wav', sr=None)
    signal = preemphasis(wav)
    frames = enframe(signal)
    spectrum = get_spectrum(frames)

    fbank_feats = fbank(spectrum)
    plot_spectrogram(fbank_feats.T, 'Filter Bank', 'fbank.png')
    write_file(fbank_feats, 'test.fbank')

    mfcc_feats = mfcc(fbank_feats)
    plot_spectrogram(mfcc_feats.T, 'MFCC', 'mfcc.png')
    write_file(mfcc_feats, 'test.mfcc')


if __name__ == '__main__':
    main()
