import yaml
import torchaudio
import librosa
import librosa.display
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from IPython.display import Audio

def load_audio(file_path,sr=16000):
    waveform, sample_rate = librosa.load(file_path, sr=sr)
    return waveform, sample_rate

def extract_segment(waveform, sample_rate, start_time, duration):
    start_sample = int(start_time * sample_rate)
    end_sample = int((start_time + duration) * sample_rate)
    return waveform[start_sample:end_sample]

def plot_spectrogram(waveform, sample_rate, metadata,window_length=1024,hop_length=512, save_path="spectrogram.png"):
    spec = librosa.stft(waveform,n_fft=window_length, hop_length=hop_length, win_length=window_length, window='hann', pad_mode='constant', center=True)
    T_coef = np.arange(spec.shape[1]) * hop_length / sample_rate
    K = window_length // 2
    F_coef = np.arange(K + 1) * sample_rate / window_length
    Y = np.abs(spec) ** 2
    extent = [T_coef[0], T_coef[-1], F_coef[0], F_coef[-1]]
    Y_compressed = log_compression(Y,gamma=1000)
    # Calculate the start and end times of the detection
    recording_start_time = datetime.strptime(metadata["recording_start"], "%Y-%m-%d %H:%M:%S")
    detection_time = datetime.strptime(metadata["timestamp"], "%Y-%m-%d %H:%M:%S")
    time_difference = (detection_time - recording_start_time).total_seconds()

    # Convert time to spectrogram frames
    frame_rate = sample_rate / hop_length
    start_frame = int(time_difference * frame_rate)
    end_frame = int((time_difference + metadata["duration"]) * frame_rate)


    # Plot the spectrogram
    plt.figure(figsize=(10, 4))
    plt.imshow(Y_compressed, cmap='inferno', aspect='auto', origin='lower', extent=extent)
    plt.xlabel('Time (seconds)')
    plt.ylim([0, 1000])
    plt.xlim([0,60])
    plt.clim([0, Y_compressed.max()])
    plt.ylabel('Frequency (Hz)')
    plt.colorbar()
    plt.title(f'Spectogram for {metadata["path"]} with detection highlighted')
    
    # Annotate the detection region
    plt.axvline(x=start_frame / frame_rate, color='cyan', linestyle='--', label='Detection Start')
    # plt.axvline(x=end_frame / frame_rate, color='red', linestyle='--', label='Detection End')
    # plt.axhline(y=freqs[low_idx], color='cyan', linestyle='--', label='Low Frequency')
    # plt.axhline(y=freqs[high_idx], color='red', linestyle='--', label='High Frequency')
    plt.legend()
    
    # Save and show the plot
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

def plot_mel_spectrogram(waveform, sample_rate, metadata, n_mels=128, hop_length=512, save_path="mel_spectrogram.png"):
    mel_spec = librosa.feature.melspectrogram(y=waveform, sr=sample_rate, n_mels=n_mels, hop_length=hop_length)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spec_db, sr=sample_rate, hop_length=hop_length, x_axis='time', y_axis='mel', cmap='inferno')
    plt.colorbar(label='dB')
    plt.title(f'Mel Spectrogram for {metadata["path"]}')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Mel Frequency (Hz)')
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def load_config(config_path):
    with open(config_path, 'r') as fp:
        config = yaml.safe_load(fp)
    return config


def plot_waveform(waveform, sample_rate, title="Waveform"):
    plt.figure(figsize=(10, 4))
    plt.plot(waveform.t().numpy())
    plt.title(title)
    plt.xlabel("Time Steps")
    plt.ylabel("Amplitude")
    plt.show()

def play_audio_segment(segment, sample_rate):
    return Audio(segment.numpy(), rate=sample_rate)

# def extract_segment(waveform, sample_rate, start_time, duration):
#     start_sample = int(start_time * sample_rate)
#     end_sample = int((start_time + duration) * sample_rate)
#     return waveform[:, start_sample:end_sample]

def log_compression(v, gamma=1.0):
    """Logarithmically compresses a value or array

    Notebook: C3/C3S1_LogCompression.ipynb

    Args:
        v (float or np.ndarray): Value or array
        gamma (float): Compression factor (Default value = 1.0)

    Returns:
        v_compressed (float or np.ndarray): Compressed value or array
    """
    return np.log(1 + gamma * v)