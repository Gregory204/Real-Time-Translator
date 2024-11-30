# 🌍 **Real-Time Translator**

## Members:
Gregory Grullon, Nakib Abedin, Gabriel Menkoff, Ofir Bitton

## 🚀 **Elevator Pitch**
Real-time voice translator to make communication easier for everyone.

## 💡 **Project Idea**
There are billions of people in the world, and unfortunately, language barriers prevent us from communicating with many of them. This project aims to build a real-time translator to facilitate communication between people speaking different languages. Here's the idea:

🎙️ **Input:** Capture the user's vocal input.  
🔁 **Translate:** Automatically translate it to English.  
🗣️ **Output:** Play the English translation as a voice-over in near real-time.

Think of interviews where a translator speaks over the original speaker—our tool automates that process to bring live translation to the masses.

---

## 🔧 **How It Works**
1. 🎤 **Receive Input:** Record speech in a foreign language using a microphone.
2. 📈 **Analyze Vocal Inflection:** Identify pitch, tones, and pauses to segment audio.
3. 🧑‍🤝‍🧑 **Match Patterns:** Compare speaking patterns to 10 pre-recorded speakers.
4. 🌐 **Translate:** Convert speech to English using a model or translation API.
5. 🎧 **Voice-Over:** Generate an English voice-over that matches the user's tone and sync it with the original audio.

---

## 📊 **Datasets**
- [English Multispeaker Corpus for Voice Cloning](https://www.kaggle.com/datasets/mfekadu/english-multispeaker-corpus-for-voice-cloning/data)  
- [Speech Accent Archive](https://www.kaggle.com/datasets/rtatman/speech-accent-archive)  

---

## 🔨 **Project Blueprint**

### 🛠️ **Input Processing**
- Receive speech input in a foreign language.
- Use a microphone to capture the user's voice.
### 🗣️ **Languages**
  - Japansese
  - Russian
  - French
  - Spanish
  - Italian
### 🎙️ **Speech Recognition**
- Use Speech Recognition Library for speech-to-text (STT).
- Pre-process audio (e.g., sampling rate, normalization).
- Transcribe audio using Google TTS.

### 🎵 **Vocal Inflection**
- Extract features like pitch & tone using **Librosa**.
- Segment audio into meaningful phrases or sentences.

### 🗺️ **Speaker Classification**
- Train a model with 10 pre-recorded speakers for speaker recognition.
- Extract features like **Mel Spectrograms**.
- Use Artificial Nerual Network for classification.

### 🌐 **Translation**
- Translate speech to English using the Google Translate API.

### 🗣️ **Voice Generation**
- Use a **TTS (Text-to-Speech)** system to generate an English voice-over.
- Aim for the TTS to output audio correlated to users voice.
- Synchronize TTS output with the original audio (allowing for a slight delay).

### 🔗 **Integration**
- Develop a robust pipeline to connect all components seamlessly.

### 🖥️ **User Interface**
- Build a simple and intuitive interface for the tool. **STREAMLIT**

---

## ✅ **Testing Phase**
- Conduct extensive testing with diverse inputs to ensure accuracy and reliability.

---

## 🌐 **Deployment**
- Host the application on the cloud or make it available for local use (curr plans).

---

## 🚧 **Construction**
- Currently working on generating voice-overs in English for non-English speakers.

---

## 🚀 **Future Improvements**
- Collect user feedback to refine the tool.
- Expand support for more languages and accents.
- Reduce latency for real-time processing.

---
