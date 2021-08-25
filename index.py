import sys
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# Load a wav file
y, sr = librosa.load(sys.argv[1], sr=None)

# extract mel spectrogram feature
melspec = librosa.feature.melspectrogram(y, sr, n_fft=1024, hop_length=512, n_mels=128)

# convert to log scale
logmelspec = librosa.power_to_db(melspec)

# chroma
S = np.abs(librosa.stft(y))
chroma = librosa.feature.chroma_stft(S=S, sr=sr)

plt.figure()

# plot a wavform
plt.subplot(3, 1, 1)
librosa.display.waveplot(y, sr)
plt.title('Beat Wavform')

# plot mel spectrogram
plt.subplot(3, 1, 2)
librosa.display.specshow(logmelspec, sr=sr, x_axis='time', y_axis='mel')
plt.title('Mel Spectrogram')

# chroma
ax = plt.subplot(3, 1, 3)
librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', ax=ax)
plt.title('Chromagram')


plt.tight_layout() #保证图不重叠
plt.show()
