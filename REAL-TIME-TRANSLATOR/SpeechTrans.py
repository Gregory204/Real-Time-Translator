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

                    # Stop condition
                    if "stop" in translated_text.lower().split():   # just say stop to end it
                        stop_flag = True
                        break
                    
                    # Append recognized translated chunk to translated list
                    translated_chunks.append(translated_text)
                    
                    # Convert translated text (English) to speech using gTTS
                    tts = gTTS(translated_text, lang="en")
                    output_file = os.path.join("DUMP_FILES", f"output_{int(time.time())}.mp3")
                    tts.save(output_file)

                    # Play the converted speech. (WORKS FOR WINDOWS AND MAC)
                    os.system(f"start {output_file}" if os.name == "nt" else f"afplay {output_file}")

                except sr.UnknownValueError:
                    st.warning("Could not understand audio, skipping...")
                except sr.RequestError as e:
                    st.error(f"Request error: {e}")
                    stop_flag = True

    except Exception as e:
        st.error(f"An error occurred: {e}")

    return chunks