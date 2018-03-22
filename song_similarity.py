import numpy as np
from scipy.signal import spectrogram
import glob
from scipy import stats
from pydub import AudioSegment
import matplotlib.pyplot as plt

datadir = '/Users/jamalw/Desktop/PNI/music_event_structures/'
genres = ["classical","jazz"]
#classical_fn = glob.glob(datadir + genres[0] + '/*.wav')
#jazz_fn = glob.glob(datadir + genres[1] + '/*.wav')
#all_songs_fn = classical_fn + jazz_fn
all_songs_fn = ['/jukebox/norman/jamalw/MES/data/songs/Moonlight_Sonata.wav']
spects = []
audio_array_holder = []

# load data
FFMPEG_BIN = "ffmpeg"

for j in range(len(all_songs_fn)):
    raw_audio = AudioSegment.from_wav(all_songs_fn[j]).raw_data
    
    # convert raw_audio to audio arrays
    audio_array = np.fromstring(raw_audio, dtype="int16")
    audio_array = audio_array.reshape((int(len(audio_array)/2),2))
    audio_array_holder.append(audio_array)    

    # combine channels
    audio_array = (audio_array[:,0] * audio_array[:,1])/2
    print('computing spectrogram')
    f,t,spect = spectrogram(audio_array,44100)
    spects.append(spect)
    print('spectrogram computed')
#spects_corr = np.corrcoef(spect.T,spect.T)[35303:,:35303]

win_size = np.round(len(t)/(len(audio_array)/44101))


np.save('Moonlight_Sonata_Similarity',spects_corr)
plt.imshow(spects_corr)
plt.colorbar()
plt.savefig('Moonlight_Sonata_Similarity')
