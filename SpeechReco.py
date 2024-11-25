# Importing the required libraries
import streamlit as st
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import librosa
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import io

# Define the target speakers in a list.
# ALSO THESE ARE FILES THAT WERE NOT USED TO TEST NOR TRAIN THE MODEL
# I PULLED THEM OUT JUST SO WE CAN SEE HOW WELL THE MODEL PERFORMS ON 
# UNSEEN AUDIO FILES. 
# 
# There is 4 more for each that still havent been added
# here but i already tested it in the notebook. 98% Accuracy

target_dictionary = {
    0 : ["p225", 'AUDIO_FILES/p225_358.wav'], # Label, Speaker_id, Wav file
    1 : ["p226", 'AUDIO_FILES/p226_366.wav'], 
    2 : ["p228", 'AUDIO_FILES/p228_367.wav'], 
    3 : ["p236", 'AUDIO_FILES/p236_500.wav'], 
    4 : ["p237", 'AUDIO_FILES/p237_348.wav'], 
    5 : ["p241", 'AUDIO_FILES/p241_370.wav'], 
    6 : ["p249", 'AUDIO_FILES/p249_351.wav'], 
    7 : ["p257", 'AUDIO_FILES/p257_430.wav'], 
    8 : ["p304", 'AUDIO_FILES/p304_420.wav'], 
    9 : ["p326", 'AUDIO_FILES/p326_400.wav']
}

# Function to extract features from audio file... (same function from notebook)
def extract_feature(file_name):
    """ Extract features from audio file
    Args:
      file_name (str): Path to audio file

    return:
      np.array: Feature vector
    """
    X, sample_rate = librosa.core.load(file_name) # load audio file
    result = np.array([]) # array that stores features
    mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T,axis=0) # calc mel spectogram
    result = np.hstack((result, mel)) # insert the mel spect into results arr
    return result # return the feature vector

# Function to classify gender (NOT MY CODE)

###################################################
# shout out: https://github.com/https://github.com/JoyBis48
# Link to Hugging Face Space: https://huggingface.co/spaces/Cosmos48/Gender-Voice-Recognition

# Function to convert audio to spectrogram image. Just so u can see it 2.
def audio_to_spectrogram(file_path):
    y, sr = librosa.load(file_path)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=512)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    plt.figure(figsize=(4, 4))
    plt.axis('off')
    plt.imshow(mel_spec_db, aspect='auto', origin='lower')
    plt.tight_layout()
    plt.savefig("spectrogram.png")
    plt.close()

# Function to classify the speaker
def classify_speaker(file_path):
    # Extract features from the user recording
    features = extract_feature(file_path).reshape(1, -1)  # Reshaping to match the model input

    # Predict the probabilities for each of the 10 speakers
    speaker_probs = model.predict(features, verbose=0)[0]

    # Identify the most likely speaker by finding the index of the highest probability
    most_likely_speaker = np.argmax(speaker_probs)  
    probability = speaker_probs[most_likely_speaker]  # Probability of the most likely speaker

    # Map the index to the speaker label
    speaker = f"Speaker {target_dictionary[most_likely_speaker][0]}"

    # For users to hear what the voice sounds like if they use their actual voice.
    wav_file = target_dictionary[most_likely_speaker][1]
    
    # Format the probability for better readability
    probability = "{:.2f}".format(probability)

    return speaker, probability, wav_file

# Load Speaker Reco Model
model = load_model('NEW_MODELS/CUR_speaker_model.h5')

# Record audio with sounddevice
def record_audio(duration=8, samplerate=22050):
    st.write("Recording...")
    audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished
    write("recorded_audio.wav", samplerate, audio_data)  # Save as WAV file
    return "recorded_audio.wav"
