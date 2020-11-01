#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 04:08:03 2020

@author: zankyo
"""

import glob
from tqdm import tqdm
# from tqdm import tqdm_notebook as tqdm
import librosa
import soundfile as sf
import numpy as np
import os
import random

# Input data
wave_dir = "cleardata"
noise_dir= "UrbanSound8K/audio"
# Output data
outdir_cleardata = "label"
outdir_addnoisedata = "data"

if(os.path.isdir(outdir_cleardata)==False):
    os.mkdir(outdir_cleardata)
    
if(os.path.isdir(outdir_addnoisedata)==False):
    os.mkdir(outdir_addnoisedata)


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

def main():

    # Lord data
    
    print("Loading Cleandata")
    clean_wave_files = glob.glob(wave_dir+"/*.wav")
    n_sources=len(clean_wave_files)
    clean_data=[]
    n_samples=0
    
    for wave_file in tqdm(clean_wave_files):
        data, rate = librosa.load(wave_file,sr=None)
        if n_samples<len(data):
            n_samples=len(data)
        clean_data.append(data)
        
    print("Loading Noisedata")
    noise_files = glob.glob(noise_dir+"/*/*.wav")
    noise_data=[]
    for f in tqdm(noise_files):
        data, rate = librosa.load(f,sr=rate)
        noise_data.append(data)
        
    print(n_sources," wave files")
    print(len(noise_files)," noise files")
    
    # zero pudding
    
    output = np.zeros([n_sources,n_samples])
    for i in tqdm(range(n_sources)):
        output[i,:len(clean_data[i])] = clean_data[i]
    
    # write wave file
    """
    雑音データが足りないので、雑音はランダムに選択し、SNRも0.7-1.2の間からランダムに選択してつけてる
    """
    for i in tqdm(range(n_sources)):
        sf.write("label/"+str(i)+"clean.wav", output[i], rate, subtype="PCM_16")
        
        noise_num = random.randint(0,len(noise_files)-1)
        noise_select = noise_data[noise_num]
        addnoisedata = addnoise(output[i],noise_select,SNR = random.uniform(0.7, 1.2))
        
        sf.write("data/"+str(i)+"noise.wav", addnoisedata, rate, subtype="PCM_16")
        
        
main()
    
