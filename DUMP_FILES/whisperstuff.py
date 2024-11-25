import streamlit as st
import speech_recognition as sr
from gtts import gTTS
from googletrans import Translator
import os
import time

def speech_to_english_and_translate(chunk_duration=2, source_language="es"):
    """
    Function to recognize speech in a different language, 
    translate it to English, and convert it to speech.
    """
    recognizer = sr.Recognizer()    # speech recognition
    microphone = sr.Microphone()    # user mic
    translator = Translator()       # google translator

    chunks = []  # Store chunks of recognized speech
    translated_chunks = [] # store chunks of translated speech
    stop_flag = False

    try:
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source)  # Adjust for background noise (like guests in a nyc library speaking)

            while not stop_flag:
                st.write(f"Listening for {chunk_duration} seconds...")
                audio = recognizer.record(source, duration=chunk_duration)  # Record for a fixed duration

                try:
                    # Recognize speech in the recorded audio (in the source language)
                    speech_text = recognizer.recognize_google(audio, language=source_language)
                    st.write(f"Chunk recognized: {speech_text}")    #

                    # Stop condition
                    if "stop" in speech_text.lower().split():   # just say stop to end it
                        stop_flag = True
                        break

                    # Append recognized chunk to the list
                    chunks.append(speech_text)

                    # Translate the recognized text to English
                    translated_text = translator.translate(speech_text, src=source_language, dest="en").text
                    st.write(f"Translated Text (English): {translated_text}")

                    # Append recognized translated chunk to translated list
                    translated_chunks.append(translated_text)
                    
                    # Convert translated text (English) to speech using gTTS
                    tts = gTTS(translated_text, lang="en")
                    output_file = f"output_{int(time.time())}.mp3"
                    tts.save(output_file)

                    # Play the converted speech
                    os.system(f"start {output_file}" if os.name == "nt" else f"afplay {output_file}")

                except sr.UnknownValueError:
                    st.warning("Could not understand audio, skipping...")
                except sr.RequestError as e:
                    st.error(f"Request error: {e}")
                    stop_flag = True

    except Exception as e:
        st.error(f"An error occurred: {e}")

    return chunks

# # Streamlit App Title and Stuff!
# st.title("Speech-to-English Translation and Text-to-Speech App")
# st.write("Speak in a foreign language (e.g., Spanish, French), and the translation in English will be spoken back to you. Also if you want to stop say 'stop'")

# # Select source language (spanish, french, russian)
# language_code = st.text_input("Enter source language code (e.g., 'es' for Spanish, 'fr' for French):", value="es")

# # Adding chunks list to session state
# if "chunks" not in st.session_state:
#     st.session_state.chunks = []
    
# # Adding translated chunks list to session state
# if "translated_chunks" not in st.session_state:
#     st.session_state.translated_chunks = []

# # Adding listening flag to session state
# if "listening" not in st.session_state:
#     st.session_state.listening = False

# # Start Listening Button -> its like our loop :)
# if not st.session_state.listening:
#     if st.button("Start Listening"):
#         st.session_state.listening = True
#         st.session_state.chunks = speech_to_english_and_translate(chunk_duration=2, source_language=language_code)

# # Stop Listening Button
# if st.session_state.listening:
#     if st.button("Stop Listening"):
#         st.session_state.listening = False

# # Display recognized chunks
# if st.session_state.chunks:
#     st.write("Recognized Chunks:")
#     st.text_area("Chunks:", value="\n".join(st.session_state.chunks), height=200)
#     st.text_area("Translated Chunks:", value="\n".join(st.session_state.translated_chunks), height=200)