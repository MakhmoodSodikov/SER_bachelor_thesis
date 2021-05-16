import librosa
import soundfile
import os, glob, pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from pydub import AudioSegment
from tqdm.notebook import tqdm_notebook
import matplotlib.pyplot as plt
from pydub import AudioSegment


emotions = {
  '01': 'neutral',
  '02': 'calm',
  '03': 'happy',
  '04': 'sad',
  '05': 'angry',
  '06': 'fearful',
  '07': 'disgust',
  '08': 'surprised'
}


# Emotions to observe
observed_emotions = ['calm', 'happy', 'sad']
# observed_emotions = list(emotions.values())
# print(observed_emotions)


# Extract features (mfcc, chroma, mel) from a sound file
def extract_feature(file_name, mfcc=True, chroma=False, mel=False):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        # plt.plot(X[:])
        # print(X.shape)
        if X.ndim == 2:
            X = X[:, 0]
        sample_rate = sound_file.samplerate

        if chroma:
            stft = np.abs(librosa.stft(X))

        result = np.array([])

        if mfcc:
            mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = librosa.feature.chroma_stft(S=stft, sr=sample_rate)
            result = np.hstack((result, chroma))
        if mel:
            mel = librosa.feature.melspectrogram(X, sr=sample_rate)
            result = np.hstack((result, mel))

    return result


def load_data(test_size=0.2, path="data/Actor_*/*.wav"):
    x, y = [], []
    for file in tqdm_notebook(glob.glob(path)):
        file_name = os.path.basename(file)
        emotion = emotions[file_name.split("-")[2]]
        if emotion not in observed_emotions:
            continue
        feature = extract_feature(file, mfcc=True, chroma=False, mel=False)
        x.append(feature)
        y.append(emotion)
    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)


def train():
    x_train, x_test, y_train, y_test = load_data(test_size=0.25)
    model = MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,),
                          learning_rate='adaptive', max_iter=500)
    model.fit(x_train, y_train)

    print('model created!')
    return model
