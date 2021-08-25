import matplotlib.pyplot as plt
import librosa
import librosa.display
# Load a wav file
y, sr = librosa.load(sys.argv[1], sr=None)
# extract mel spectrogram feature
melspec = librosa.feature.melspectrogram(y, sr, n_fft=1024, hop_length=512, n_mels=128)
# convert to log scale
logmelspec = librosa.power_to_db(melspec)
plt.figure()
# plot a wavform
plt.subplot(2, 1, 1)
librosa.display.waveplot(y, sr)
plt.title('Beat wavform')
# plot mel spectrogram
plt.subplot(2, 1, 2)
librosa.display.specshow(logmelspec, sr=sr, x_axis='time', y_axis='mel')
plt.title('Mel spectrogram')
plt.tight_layout() #保证图不重叠
plt.show()
