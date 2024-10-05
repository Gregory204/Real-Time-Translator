# Real-Time-Translator

## Members:
Gregory Grullon, Nakib Abedin, Gabriel Menkoff

## Elevator Pitch:
Real-time voice translator to make communication easier for everyone

## Project Idea:
There are billions of people in the world and it is unfortunate that we can’t communicate with many of them because of language barriers. Our project idea is to build a real-time translator in order to facilitate communication between different people. What we will do at a high level is take in a sample vocal input from the user, translate what they’re saying to English, and then play an English translation of what they’re saying as a voice-over. If you’ve ever seen an interview of a person speaking a different language, that’s exactly what we’re going for. Instead of having a person do the translation, we want to do it in an automated manner to make a live-translation tool available to the masses.

## How it Will Work:
Receive input in a foreign language
Identify vocal inflection points for the input speaker
Match the user’s speaking patterns with the patterns for the 109 pre-recorded speakers we have
Translate what the user is saying into English
Produce a voice-over in English that plays at the same time as what the input user is saying

## Datasets:
English Multispeaker Corpus for Voice Cloning: https://www.kaggle.com/datasets/mfekadu/english-multispeaker-corpus-for-voice-cloning/data
Speech Accent Archive: https://www.kaggle.com/datasets/rtatman/speech-accent-archive

## Project Blueprint: Real-Time Voice Translator

### Input Processing:
Receive input in a foreign language.
Use a microphone to capture user speech

### Speech Recognition:
Implement Whisper model for STT
Pre-Process audio input (such as sampling rate, normalization
Use a whisper model to transcribe audio.

### Vocal Inflection:
Use audio processing libraries such as librosa to get the pitch, tones, and pauses. (Segment the audio into phrases or sentences)

### Accent Classification
Train a model using the 109 pre-recorded speakers for accent classification.
Will require MFCC.
Use Neural Networks (ex: CNNs, RNNs)

### Translate:
Foreign Language to English:
Use google translate api or a pre-trained model

### Voice Generating:
Have a tts system to create a english voice over.
Check if the TTS output can almost mimic the user’s o.g. Vocals.
Sync the TTS output with o.g. Audio in somewhat real time possibly with 1 second or sentence delay.

### Integrating:
Develop a pipeline to bring all these steps together.

### User Interface:
Create a User Interface.. More Details Later

### Testing Phase:

### Deployment:

### Further improvement and Feedback.
