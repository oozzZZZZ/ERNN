_,_,stft_mix=stft(add_noise[n_noise_only:],fs=rate,window="hann",nperseg=512,noverlap=256)
_,_,stft_sample=stft(sample,fs=rate,window="hann",nperseg=512,noverlap=256)

#PSM
X, S = stft_mix, stft_sample #観測信号、所望信号

A = (np.abs(S) / np.abs(X)) * np.cos((np.angle(S)-np.angle(X)))
B = np.maximum(A,0)
G = np.minimum(B,1)

#G = np.ones(G.shape) - G

# マスクの適応
processed_stft_data = stft_mix * G # アダマール積

#時間領域の波形に戻す
_,processed_stft_post=istft(processed_stft_data,fs=rate,window="hann",nperseg=512,noverlap=256)

_,iG = istft(G,fs=rate,window="hann",nperseg=512,noverlap=256)


print("add noise")
display(Audio(add_noise[n_noise_only:], rate=rate))
print("original")
display(Audio(sample, rate=rate))
print("psm")
display(Audio(processed_stft_post, rate=rate))
disp_spectrum(processed_stft_post,rate,size=(10,4))
