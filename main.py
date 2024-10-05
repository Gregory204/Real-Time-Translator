import pyaudio 
import wave 
import time

def write_to_file(curr_time, frames, audio) -> None:
    filename = f"{curr_time}.wav"
    sound_file = wave.open(filename, "wb")
    sound_file.setnchannels(1)
    sound_file.setsampwidth( audio.get_sample_size(pyaudio.paInt16) )
    sound_file.setframerate(44100)
    sound_file.writeframes(b''.join(frames))
    sound_file.close()

def main() -> None:
    # generate stream of audio from input
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)
    prev_checkpoint = time.time()

    try:
        frames = []
        while True:
            # record the audio
            frames.append( stream.read(1024) )

            # if a 2 seconds elapse, save it audio to a file for translation
            curr_time = time.time()
            time_elapsed = curr_time - prev_checkpoint
            if time_elapsed > 2:
                prev_checkpoint = curr_time
                write_to_file( curr_time, frames, audio )
                frames = [] # reset for next window

    except KeyboardInterrupt:
        pass
    
    write_to_file(time.time(), frames, audio)
    stream.stop_stream()
    stream.close()
    audio.terminate()


if __name__ == "__main__":
    main()