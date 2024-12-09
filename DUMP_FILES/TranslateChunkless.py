# Importing the required libraries
import streamlit as st
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import librosa
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout 
import io
from translate import Translator
from gradio_client import Client, handle_file
import speech_recognition as sr

# Set the tts model to F5-TTS
client = Client("http://127.0.0.1:7860/")
result = client.predict(
		new_choice="F5-TTS",
		api_name="/switch_tts_model"
)
print(result)

# Define the target speakers in a list.
# ALSO THESE ARE FILES THAT WERE NOT USED TO TEST NOR TRAIN THE MODEL
# I PULLED THEM OUT JUST SO WE CAN SEE HOW WELL THE MODEL PERFORMS ON 
# UNSEEN AUDIO FILES. 
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
cloning_directory = {
    'p225': 'FILES_CLONING/p225_023.wav',
    'p226': 'FILES_CLONING/p226_023.wav',
    'p228': 'FILES_CLONING/p228_023.wav',
    'p236': 'FILES_CLONING/p236_023.wav',
    'p237': 'FILES_CLONING/p237_023.wav',
    'p241': 'FILES_CLONING/p241_023.wav',
    'p249': 'FILES_CLONING/p249_023.wav',
    'p257': 'FILES_CLONING/p257_023.wav',
    'p304': 'FILES_CLONING/p304_023.wav',
    'p326': 'FILES_CLONING/p326_023.wav'
}

#function to turn spanish speach into text
def transcribe_audio(file_path):
    recognizer = sr.Recognizer()

    with sr.AudioFile(file_path) as source:
        audio = recognizer.record(source)

        # Recognize speech in Spanish
        text = recognizer.recognize_google(audio, language="es-ES")
        #text = recognizer.recognize_google(audio)

        st.write(text)
        return text


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

# Streamlit app
st.title("Voice Correlation Recognition")
st.write("This application is still undergoing fixes & updates ;-;")

# Option to upload a file
uploaded_file = st.file_uploader("Upload an audio file", type=['wav', 'mp3'])

if uploaded_file is not None:
    with open("uploaded_audio.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.audio(uploaded_file, format='audio/wav')

    if st.button("Submit"):
        try:
            audio_to_spectrogram("uploaded_audio.wav")
            st.image("spectrogram.png", caption="Mel Spectrogram of the uploaded audio file", use_container_width ="auto", width=200)
            speaker, probability, _ = classify_speaker("uploaded_audio.wav")
            
            # Which speaker is it?
            st.write(f"Predicted Speaker: {speaker}")

            # Whats the chances of being speaker?
            st.write(f"Speaker Probability: {probability}")



            transcribed_audio = transcribe_audio("uploaded_audio.wav")
            translated = 'error please try again'

            try:
                translator = Translator(from_lang="es", to_lang="en")
                translated = translator.translate(transcribed_audio)
                st.write('Translated Text: ' + translated)
            except Exception as e:
                st.write(f"An error occurred: {e}")

            result = client.predict(
		        ref_audio_input=handle_file(cloning_directory[speaker[8:12]]),
		        ref_text_input="If the red of the second bow falls upon the green of the first, the result is to give a bow with an abnormally wide yellow band, since red and green light when mixed form yellow.",
		        gen_text_input=translated,
		        remove_silence=True,
		        cross_fade_duration_slider=0.15,
		        speed_slider=1,
		        api_name="/basic_tts"
            )

            st.write('Output Audio:')
            st.audio(result[0])

            
        except Exception as e:
            st.error(f"Error occurred: {e}")

# Record audio with sounddevice
def record_audio(duration=8, samplerate=22050):
    st.write("Recording...")
    audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished
    write("recorded_audio.wav", samplerate, audio_data)  # Save as WAV file
    return "recorded_audio.wav"

st.write('Or, record audio')

if st.button("Record Audio"):
    wav_file_path = record_audio(duration=5)
    st.write(f"Audio recorded and saved to {wav_file_path}") # save auido
    st.audio(wav_file_path) # show audio
    audio_to_spectrogram(wav_file_path) # melspectogram of recorded audio
    st.image("spectrogram.png", caption="Mel Spectrogram of the recorded audio file", use_container_width ="auto", width=200)
    speaker, probability, wav_file = classify_speaker(wav_file_path)

    # who do u sound like the most?
    st.write(f"Predicted Speaker: {speaker}")
    
    # how much do u sound like this speaker?
    st.write(f"Speaker Probability: {probability}")
    
    # speaker audio
    st.audio(wav_file)
    
    transcribed_audio = transcribe_audio(wav_file_path)
    translated = 'error please try again'

    try:
        translator = Translator(from_lang="es", to_lang="en")
        translated = translator.translate(transcribed_audio)
        st.write('Translated Text: ' + translated)
    except Exception as e:
        st.write(f"An error occurred: {e}")

    result = client.predict(
		ref_audio_input=handle_file(cloning_directory[speaker[8:12]]),
		ref_text_input="If the red of the second bow falls upon the green of the first, the result is to give a bow with an abnormally wide yellow band, since red and green light when mixed form yellow.",
		gen_text_input=translated,
		remove_silence=True,
		cross_fade_duration_slider=0.15,
		speed_slider=1,
		api_name="/basic_tts"
    )
    st.write('Output Audio:')
    st.audio(result[0])
    
    