import tkinter as tk
import soundfile as sf
import numpy as np
import librosa
import wave
from scipy.signal import butter, filtfilt
import pyworld as pw
import pyaudio
import time


# Global variables for recording
RECORDING = False
OUTPUT_FILENAME = "recorded_audio.wav"      # Output filename for the recorded audio
SAMPLE_RATE = 16000                         # Sample rate for the recorded audio
CHANNELS = 1                                # Number of audio channels for the recorded audio
CHUNK_SIZE = 1024                           # Size of each audio chunk


# Functions required for Effects

# This is required for Resonance
def butter_highpass(cutoff, fs, order=10):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

# This is required for Resonance
def apply_highpass_filter(data, cutoff, fs, order=10):
    b, a = butter_highpass(cutoff, fs, order=order)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

# This is required for Resonance
def adjust_resonance(input_file, output_file, cutoff):
    # Load the audio file
    audio_data, sample_rate = librosa.load(input_file, sr=None)

    # Apply the high-pass filter
    filtered_data = apply_highpass_filter(audio_data, cutoff, sample_rate)

    # Save the filtered audio
    sf.write(output_file, filtered_data, sample_rate)

# This is required for Echo
def apply_echo(signal, delay, decay):
    echo = np.zeros_like(signal)
    echo[delay:] = signal[:-delay] * decay
    return signal + echo

# This is required for Chorus
def apply_chorus(input_file, output_file, delay_time, modulation_depth, modulation_rate):
    # Load the input voice signal
    voice, sr = librosa.load(input_file, sr=None)
    
    # Create a time vector
    t = np.arange(len(voice)) / sr
    
    # Compute the modulation signal
    mod_signal = np.sin(2 * np.pi * modulation_rate * t)
    
    # Scale and center the modulation signal
    mod_signal = modulation_depth * mod_signal + 1
    
    # Create the delayed copies with varying amplitude
    delays = [0, delay_time/4, delay_time/2, 3*delay_time/4]
    amplitudes = [0.6, 0.4, 0.2, 0.1]  
    delayed_voices = np.zeros_like(voice)
    
    for d, a in zip(delays, amplitudes):
        delay_samples = int(d * sr)
        delayed_voice = np.concatenate((np.zeros(delay_samples), voice[:-delay_samples]))
        delayed_voice *= a
        delayed_voices += np.pad(delayed_voice, (0, len(voice) - len(delayed_voice)), 'constant')
    
    # Mix the original and delayed voices
    mixed_voice = voice + delayed_voices
    
    # Apply the modulation to the mixed voice
    modulated_voice = mixed_voice * mod_signal
    
    # Normalize the output
    modulated_voice /= np.max(np.abs(modulated_voice))
    
    # Save the output as a new audio file
    sf.write(output_file, modulated_voice, sr)

'''Functions to run GUI'''

# Function to change voice to female
def Female():
    print('Changing male voice to female voice...')

    # Pitch_Shifting 

    wr = wave.open('recorded_audio.wav','r')
    # Set the parameters for the output file.
    par = list(wr.getparams())
    par[3] = 0
    par = tuple(par)
    wf = wave.open('recorded_fourier.wav', 'w')
    wf.setparams(par)

    fr = 20
    sz = wr.getframerate()//fr  # Read and process 1/fr second at a time.
    c = int(wr.getnframes()/sz)  # count of the whole file
    shift = 100//fr  
    for num in range(c):
        da = np.fromstring(wr.readframes(sz), dtype=np.int16)
        left, right = da[0::2], da[1::2]  # left and right channel
        lf, rf = np.fft.rfft(left), np.fft.rfft(right)
        lf, rf = np.roll(lf, shift), np.roll(rf, shift)
        lf[0:shift], rf[0:shift] = 0, 0
        nl, nr = np.fft.irfft(lf), np.fft.irfft(rf)
        ns = np.column_stack((nl, nr)).ravel().astype(np.int16)
        wf.writeframes(ns.tostring())
    wr.close()
    wf.close()

    # Resonance

    input_file = "recorded_fourier.wav"
    output_file = "recorded_resonance.wav"
    cutoff = 180.0  # High-pass filter cutoff frequency in Hz

    adjust_resonance(input_file, output_file, cutoff)

    # Formant_Shifting
    
    audio, sr = librosa.load('recorded_resonance.wav', sr=None) #Load audio file

    # Convert audio to double precision
    audio = audio.astype(float)

    # Extract fundamental frequency (F0), spectral envelope, and aperiodicity using WORLD
    f0, t = pw.dio(audio, sr)
    f0 = pw.stonemask(audio, f0, t, sr)
    sp = pw.cheaptrick(audio, f0, t, sr)
    ap = pw.d4c(audio, f0, t, sr)

    # Shift formants
    formant_shift_ratio = 100  # Adjust this value to control the degree of formant shift
    sp_shifted = sp / formant_shift_ratio

    # Synthesize shifted voice
    audio_shifted = pw.synthesize(f0, sp_shifted, ap, sr)

    # Save shifted voice to file
    output_file = 'recorded_formant.wav'
    sf.write(output_file, audio_shifted, sr)

    # Open the audio file
    wf = wave.open(output_file, 'rb')

    # Initialize PyAudio
    p = pyaudio.PyAudio()

    # Open the audio stream
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    # Read and play the audio file in chunks
    chunk_size = 1024
    data = wf.readframes(chunk_size)
    while data:
        stream.write(data)
        data = wf.readframes(chunk_size)

    # Stop and close the audio stream
    stream.stop_stream()
    stream.close()

    # Terminate PyAudio
    wf.close()
    p.terminate()


# Function to change voice to Alien-like
def Alien():
    print('Changing voice to Alien-like...')
    wr = wave.open('recorded_audio.wav','rb')  
    # Parameters for the output file.
    par = list(wr.getparams())
    par[3] = 0
    par = tuple(par)
    wf = wave.open('output_Alien.wav', 'w') 
    wf.setparams(par)

    fr = 20
    sz = wr.getframerate()//fr      # Read and process 1/fr second at a time.
    c = int(wr.getnframes()/sz)     # count of the whole file
    shift = 400//fr                 # shifting 20 Hz
    for num in range(c):
        da = np.fromstring(wr.readframes(sz), dtype=np.int16)
        left, right = da[0::2], da[1::2]  # left and right channel
        lf, rf = np.fft.rfft(left), np.fft.rfft(right)
        lf, rf = np.roll(lf, shift), np.roll(rf, shift)
        lf[0:shift], rf[0:shift] = 0, 0
        nl, nr = np.fft.irfft(lf), np.fft.irfft(rf)
        ns = np.column_stack((nl, nr)).ravel().astype(np.int16)
        wf.writeframes(ns.tostring())
    wr.close()
    wf.close()

    # Open the audio file
    wf = wave.open('output_Alien.wav', 'rb')

    # Initialize PyAudio
    p = pyaudio.PyAudio()

    # Open the audio stream
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    # Read and play the audio file in chunks
    chunk_size = 1024
    data = wf.readframes(chunk_size)
    while data:
        stream.write(data)
        data = wf.readframes(chunk_size)

    # Stop and close the audio stream
    stream.stop_stream()
    stream.close()

    # Terminate PyAudio
    wf.close()
    p.terminate()


# Function to add Echo effect to voice
def Echo():
    print('Adding Echo effect...')
    # Load the audio file
    audio_path = 'recorded_audio.wav' 
    audio, sr = sf.read(audio_path)

    # Set the parameters for the echo effect
    delay_samples = int(sr * 0.5)   # Delay in seconds (adjust as desired)
    decay = 0.5                     # Decay factor (adjust as desired)

    # Apply the echo effect to the audio signal
    echo_audio = apply_echo(audio, delay_samples, decay)

    # Save the modified audio to a new file
    output_file = 'output_Echo.wav'
    sf.write(output_file, echo_audio, sr)

    # Open the audio file
    wf = wave.open(output_file, 'rb')

    # Initialize PyAudio
    p = pyaudio.PyAudio()

    # Open the audio stream
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    # Read and play the audio file in chunks
    chunk_size = 1024
    data = wf.readframes(chunk_size)
    while data:
        stream.write(data)
        data = wf.readframes(chunk_size)

    # Stop and close the audio stream
    stream.stop_stream()
    stream.close()

    # Terminate PyAudio
    wf.close()
    p.terminate()


# Function to add chorus effect to voice
def Chorus():
    print('Adding chorus effect...')

    input_file = 'recorded_audio.wav' 
    output_file = 'output_Chorus.wav'  
    delay_time = 1                  # Delay time in seconds
    modulation_depth = 0.3          # Amplitude scaling factor for modulation
    modulation_rate = 2.0           # Modulation frequency in Hz

    apply_chorus(input_file, output_file, delay_time, modulation_depth, modulation_rate)

    # Open the audio file
    wf = wave.open(output_file, 'rb')

    # Initialize PyAudio
    p = pyaudio.PyAudio()

    # Open the audio stream
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    # Read and play the audio file in chunks
    chunk_size = 1024
    data = wf.readframes(chunk_size)
    while data:
        stream.write(data)
        data = wf.readframes(chunk_size)

    # Stop and close the audio stream
    stream.stop_stream()
    stream.close()

    # Terminate PyAudio
    wf.close()
    p.terminate()


# Function to start recording
def Start():
    global RECORDING
    if RECORDING:
        print("Already recording...") # To prevent multiple pressings of Start Button
        return
    RECORDING = True

    p = pyaudio.PyAudio() 

    # Open the output WAV file for writing
    wf = wave.open(OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(SAMPLE_RATE)

    # Start audio stream
    stream = p.open(format=pyaudio.paInt16,
                    channels=CHANNELS,
                    rate=SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=CHUNK_SIZE)

    start_time = time.time()
    while RECORDING:
        data = stream.read(CHUNK_SIZE)
        wf.writeframes(data)
        if time.time() - start_time >= 10:  # Stop recording after 10 seconds
            print('Recording Stopped')
            break
    

    # Stop and close the audio stream
    stream.stop_stream()
    stream.close()

    # Close the output WAV file
    wf.close()

    p.terminate()  # Terminate PyAudio

# Function to stop recording
def Stop():
    global RECORDING
    RECORDING = False


# Function to quit the GUI
def quit_gui():
    window.destroy()

# Create the GUI window
window = tk.Tk()
window.title("Voice Changer")
window.geometry("400x400")
window.resizable(False, False)
window.configure(bg="white")

# Creating label for recording functions
lbl_recording_functions = tk.Label(window, text="Recording Functions", font=("Helvetica", 16, "bold"), bg="white")
lbl_recording_functions.pack(pady=20)

# Create buttons for recording
btn_start = tk.Button(window, text="Start Recording", command=Start, width=30)
btn_start.pack(pady=5)

btn_stop = tk.Button(window, text="Stop Recording", command=Stop, width=30)
btn_stop.pack(pady=5)

# Creating label for voice changing functions
lbl_voice_functions = tk.Label(window, text="Voice Changing Functions", font=("Helvetica", 16, "bold"), bg="white")
lbl_voice_functions.pack(pady=20)

# Creating buttons for voice changing functions
btn_female = tk.Button(window, text="Female", command=Female, width=30)
btn_female.pack(pady=5)

btn_alien = tk.Button(window, text="Alien", command=Alien, width=30)
btn_alien.pack(pady=5)

btn_echo = tk.Button(window, text="Echo", command=Echo, width=30)
btn_echo.pack(pady=5)

btn_chorus = tk.Button(window, text="Chorus", command=Chorus, width=30)
btn_chorus.pack(pady=5)

# Quit button
btn_quit = tk.Button(window, text="Quit", command=quit_gui)
btn_quit.pack(pady=10)


# Start the GUI event loop
window.mainloop()
