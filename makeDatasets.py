import librosa

import glob
from tqdm import tqdm
import random
import math

from scipy.signal import stft
import scipy as sp
import numpy as np


c_files = glob.glob("/disk107/Datasets/CMU_ARCTIC/*/*wav/*.wav")
n_files = glob.glob("/disk107/Datasets/UrbanSound8K/audio/fold*/*.wav")

random.shuffle(c_files)
random.shuffle(n_files)

print("clear voice is",len(c_files),"files")
print("Urban Noise is",len(n_files),"files")

audio_len = 2**15
usedata_num = 15000
sample_rate = 16000

def makePSM(addnoise,clean):
    X, S = addnoise, clean #観測信号、所望信号stft型
    A = (np.abs(S) / np.abs(X)) * np.cos((np.angle(S)-np.angle(X)))
    B = np.maximum(A,0)
    G = np.minimum(B,1)
    return G

def addnoise(c_data,n_data,SNR = 0.1):
    """
    c_data:音声データ
    n_data:ノイズデータ
    
    クリーンな音声データに任意の雑音をつけます。
    
    音声データの長さがノイズデータの長さより長い時→ノイズデータをリピートして音声データの長さに合わせる
    音声データの長さがノイズデータの長さより短い時→ノイズデータを音声な音声データの長さに合わせて切り捨て
    """

    c_data_s = c_data * SNR #ernieノイズの音量が小さいので調整してます。

    if len(c_data) == len(n_data):
        noise_data = c_data_s + n_data

    elif len(c_data) > len(n_data):

        q, mod = divmod(len(c_data), len(n_data))

        if q == 1:
            new = np.append(n_data,n_data[:mod])

        else:
            new = np.append(n_data,n_data)
            for i in range(q-2):
                new = np.append(new,n_data)
            new = np.append(new,n_data[:mod])

        noise_data = c_data_s + new

    else:
        noise_data = c_data_s + n_data[:len(c_data)]
        
    return noise_data

# STFT test

c_data,_ = librosa.load(str(c_files[0]),sr=sample_rate)

if len(c_data)<audio_len:
    _c_data = np.zeros([audio_len])
    _c_data[:len(c_data)] = c_data
    c_data = _c_data
    
else:
    c_data = c_data[:audio_len]
    
n_data,_ = librosa.load(n_files[random.randint(0,len(n_files)-1)],sr=sample_rate)
c_n_data = addnoise(c_data,n_data,SNR = random.uniform(0.7, 1.2))

# stft
_f,_t,stft_data = stft(c_n_data,fs=sample_rate,window='hamming')
_f,_t,stft_label= stft(c_data,fs=sample_rate,window='hamming')
stft_label_psm = makePSM(stft_data,stft_label)

print("data",stft_data.shape,stft_data.dtype,"freq",_f.shape,"time",_t.shape)
print("label",stft_label_psm.shape,stft_label_psm.dtype)

#main
data = np.zeros([usedata_num,stft_data.shape[0],stft_data.shape[1]],dtype = "complex64")
label = np.zeros([usedata_num,stft_data.shape[0],stft_data.shape[1]],dtype = "float32")

for i in tqdm(range(usedata_num)):
    # audioデータを作る
    c_data,_ = librosa.load(str(c_files[i]),sr=sample_rate)
    if len(c_data)<audio_len:
        _c_data = np.zeros([audio_len])
        _c_data[:len(c_data)] = c_data
        c_data = _c_data

    else:
        c_data = c_data[:audio_len]

    """
    雑音データが足りないので、雑音はランダムに選択し、SNRも0.7-1.2の間からランダムに選択してつけてる
    """
    n_data,_ = librosa.load(n_files[random.randint(0,len(n_files)-1)],sr=sample_rate)
    c_n_data = addnoise(c_data,n_data,SNR = random.uniform(0.7, 1.2))

    # stft
    _f,_t,stft_data = stft(c_n_data,fs=sample_rate,window='hamming')
    _f,_t,stft_label= stft(c_data,fs=sample_rate,window='hamming')
    stft_label_psm = makePSM(stft_data,stft_label)

    # append
    data[i] = stft_data
    label[i] = stft_label_psm

np.savez("datasets129x257", data=data, label=label)
