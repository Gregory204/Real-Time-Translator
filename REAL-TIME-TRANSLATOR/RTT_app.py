import streamlit as st  # our interface
import SpeechReco
import SpeechTrans          
import tensorflow as tf  # TensorFlow import
import time             # Grace Period
import os               # DUMP USAGE

# Title of the app
st.title("Real Time Translator")

st.write("Choose a foreign language of your choice (e.g., Spanish, French)")

# Define a dictionary mapping language names to their corresponding codes
language_map = {
    "Japanese": "ja",
    "Russian": "ru",
    "French": "fr",
    "Spanish": "es",
    "Italian": "it"
}

# Create a selectbox to let the user choose a language
selected_language = st.selectbox(
    "Select a language:",
    options=list(language_map.keys()),  # Show language names
)

# Show the corresponding language code in the selected format
language_code = language_map[selected_language]

if selected_language:
    # Display the language and its code
    st.write(f"Current selected language: {selected_language}")

    # First phase: Voice Correlation
    st.title("Phase 1: Voice Correlation")

    uploaded_file = st.file_uploader("Upload an audio file", type=['wav', 'mp3'])

    if uploaded_file:
        file_path = os.path.join("DUMP_FILES", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.audio(uploaded_file, format='audio/wav')

        if st.button("Submit"):
            try:
                SpeechReco.audio_to_spectrogram(file_path)
                st.image("spectrogram.png", caption="Mel Spectrogram of the uploaded audio file", use_column_width="auto", width=200)
                speaker, probability, _ = SpeechReco.classify_speaker(file_path)
                
                # Which speaker is it?
                st.write(f"Predicted Speaker: {speaker}")

                # Whats the chances of being speaker?
                st.write(f"Speaker Probability: {float(probability):.2f}")
                
                        # Wait for 5 seconds before moving to Phase 2
                st.write("Waiting for Phase 2 to start...")
                time.sleep(5)  # Wait for 5 seconds before proceeding
                    
            
                # Now start the second phase (Speech Translation)
                st.title("Phase 2: Speech-to-English Translation")
                st.write("Speak in a foreign language (e.g., Spanish, French), and the translation in English will be spoken back to you. Also if you want to stop say 'stop'")

                # st.session_state.chunks = SpeechTrans.speech_to_english_and_translate(chunk_duration=2, source_language=language_code)
        
                # Adding chunks list to session state
                if "chunks" not in st.session_state:
                    st.session_state.chunks = []
                    
                # Adding translated chunks list to session state
                if "translated_chunks" not in st.session_state:
                    st.session_state.translated_chunks = []

                # Adding listening flag to session state
                if "listening" not in st.session_state:
                    st.session_state.listening = False

                # Start Listening Button -> it's like our loop :)
                if not st.session_state.listening:
                    st.session_state.listening = True
                    st.session_state.chunks = SpeechTrans.speech_to_english_and_translate(chunk_duration=2, source_language=language_code)

                # Stop Listening Button
                if st.session_state.listening:
                    if st.button("Stop Listening"):
                        st.session_state.listening = False

                # Display recognized chunks
                if st.session_state.chunks:
                    st.write("Recognized Chunks:")
                    st.text_area("Chunks:", value="\n".join(st.session_state.chunks), height=200)
                    st.text_area("Translated Chunks:", value="\n".join(st.session_state.translated_chunks), height=200)

            except Exception as e:
                st.error(f"Error occurred in Phase 1: {e}")

    # Option to record audio
    if st.button("Record Audio"):
        wav_file_path = SpeechReco.record_audio(duration=5)
        st.write(f"Audio recorded and saved to {wav_file_path}")  # save audio
        st.audio(wav_file_path)  # show audio
        SpeechReco.audio_to_spectrogram(wav_file_path)  # melspectrogram of recorded audio
        st.image("spectrogram.png", caption="Mel Spectrogram of the recorded audio file", use_column_width="auto", width=200)
        speaker, probability, wav_file = SpeechReco.classify_speaker(wav_file_path)

        # Who do you sound like the most?
        st.write(f"Predicted Speaker: {speaker}")
        
        # How much do you sound like this speaker?
        st.write(f"Speaker Probability: {float(probability):.2f}")
        
        # Speaker audio
        st.audio(wav_file)
        
        # Wait for 5 seconds before moving to Phase 2
        st.write("Waiting for Phase 2 to start...")
        time.sleep(5)  # Wait for 5 seconds before proceeding
            
        # Now start the second phase (Speech Translation)
        st.title("Phase 2: Speech-to-English Translation")
        st.write("Speak in a foreign language (e.g., Spanish, French), and the translation in English will be spoken back to you. Also, if you want to stop, say 'stop'")

        st.session_state.chunks = SpeechTrans.speech_to_english_and_translate(chunk_duration=2, source_language=language_code)
        
        # Adding chunks list to session state
        if "chunks" not in st.session_state:
            st.session_state.chunks = []
            
        # Adding translated chunks list to session state
        if "translated_chunks" not in st.session_state:
            st.session_state.translated_chunks = []

        # Adding listening flag to session state
        if "listening" not in st.session_state:
            st.session_state.listening = False

        # ** NOT NEEDED **
        # # Start Listening Button -> it's like our loop :)
        # if not st.session_state.listening:
        #     if st.button("Start Listening"):
        #         st.session_state.listening = True
        #         st.session_state.chunks = speech_to_english_and_translate(chunk_duration=2, source_language=language_code)
        # ** NOT NEEDED **
        
        # Stop Listening Button
        if st.session_state.listening:
            if st.button("Stop Listening"):
                st.session_state.listening = False

        # Display recognized chunks
        if st.session_state.chunks:
            st.text_area("Chunks:", value="\n".join(st.session_state.chunks), height=200)
            st.text_area("Translated Chunks:", value="\n".join(st.session_state.translated_chunks), height=200)

# TensorFlow optimization (GETS RID OF ERROR MESSAGES)
@tf.function(reduce_retracing=True)
def process_audio_input(input_data):
    return some_tensorflow_function(input_data)
