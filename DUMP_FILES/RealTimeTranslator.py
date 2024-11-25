import io
import streamlit as st 
import numpy as np
import librosa
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
from scipy.io.wavfile import write, read as wav_read   
from st_audiorec import st_audiorec

# Function to convert audio to spectrogram image.
def audio_to_spectrogram(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=512)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    plt.figure(figsize=(4, 4))
    plt.axis('off')
    plt.imshow(mel_spec_db, aspect='auto', origin='lower')
    plt.tight_layout()
    plt.savefig("spectrogram.png")
    plt.close()
    
# Just so we can show u how it looks.


