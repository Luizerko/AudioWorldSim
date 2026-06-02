import argparse

import librosa
import librosa.display

import matplotlib.pyplot as plt
import numpy as np

# Processing binaural audio and (potentially) plotting sound waves and spectograms for sanity check 
def process_binaural_audio(file: str, sr: int = 44100, plot: bool = False):
    # Reading audio file
    audio, _ = librosa.load(file, sr=sr, mono=False)
    left, right = audio[0], audio[1]

    # (Potentially) Plotting sound waves
    if plot:
        plt.figure(figsize=(9, 9))
       
        plt.subplot(2, 1, 1)
        plt.title("Left ear waveform")
        librosa.display.waveshow(left, sr=sr, color='blue')

        plt.subplot(2, 1, 2)
        plt.title("Right ear waveform")
        librosa.display.waveshow(right, sr=sr, color='orange')

        plt.tight_layout()
        plt.show()

    # Computing Mel spectograms
    n_samples = 2048
    hop_len = 294
    n_bands = 128

    mel_left = librosa.feature.melspectrogram(y=left, sr=sr, n_fft=n_samples, hop_length=hop_len, n_mels=n_bands)
    mel_right = librosa.feature.melspectrogram(y=right, sr=sr, n_fft=n_samples, hop_length=hop_len, n_mels=n_bands)
    max_power = np.max([mel_left, mel_right])

    # Converting from power to dB's
    mel_left = librosa.power_to_db(mel_left, ref=max_power)
    mel_right = librosa.power_to_db(mel_right, ref=max_power)

    # (Potentially) Plotting Mel spectograms
    if plot:
        plt.figure(figsize=(9, 9))

        plt.subplot(2, 1, 1)
        librosa.display.specshow(mel_left, sr=sr, hop_length=hop_len, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')

        plt.subplot(2, 1, 2)
        librosa.display.specshow(mel_right, sr=sr, hop_length=hop_len, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')

        plt.tight_layout()
        plt.show()

    return mel_left, mel_right


if __name__ == "__main__":
    # Parsing arguments
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--file", help="Path to the audio file to be processed", type=str, required=True)
    parser.add_argument("--sample_rate", help="Sample rate for sound simulation and later for audio spatialization.", type=int, default=44100)
    parser.add_argument("--verbose", help="Print and plot everything. Meant for debugging.", action='store_true')

    args = parser.parse_args()

    # Processing audio
    process_binaural_audio(args.file, args.sample_rate, args.verbose)