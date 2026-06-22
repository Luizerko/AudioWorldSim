import argparse
import os

import librosa
import librosa.display

import matplotlib.pyplot as plt
import numpy as np

# Processing binaural audio with mel spectrograms and (potentially) plotting sound waves and spectograms for sanity check
def process_binaural_audio_mel(file: str, sr: int = 44100, hop_len: int = 294, plot: bool = False):
    # Reading audio file
    audio, _ = librosa.load(file, sr=sr, mono=False)
    left, right = audio[0], audio[1]

    # (Potentially) Plotting sound waves
    if plot:
        fig, axes = plt.subplots(2, 1, figsize=(9, 9), sharey=True)

        axes[0].set_title("Left ear waveform")
        librosa.display.waveshow(left, sr=sr, color='blue', ax=axes[0])

        axes[1].set_title("Right ear waveform")
        librosa.display.waveshow(right, sr=sr, color='orange', ax=axes[1])

        plt.tight_layout()
        plt.show()
        plt.close()

    # Computing Mel spectrograms. We leave n_samples at 2048 here because we don't have ITD to work with, so we sitck to better frequency resolution 
    n_samples = 2048
    n_bands = 128

    mel_left = librosa.feature.melspectrogram(y=left, sr=sr, n_fft=n_samples, hop_length=hop_len, n_mels=n_bands)
    mel_right = librosa.feature.melspectrogram(y=right, sr=sr, n_fft=n_samples, hop_length=hop_len, n_mels=n_bands)

    # Converting from power to dB's
    max_power = np.max([mel_left, mel_right])
    mel_left = librosa.power_to_db(mel_left, ref=max_power)
    mel_right = librosa.power_to_db(mel_right, ref=max_power)

    # Saving full mel spectrograms
    np.savez_compressed(file.replace('output.wav', 'full_mel.npz'), left=mel_left, right=mel_right)

    # (Potentially) Plotting Mel spectrograms
    if plot:
        plt.figure(figsize=(9, 9))

        plt.subplot(2, 1, 1)
        plt.title("Left ear Mel spectrogram")
        librosa.display.specshow(mel_left, sr=sr, hop_length=hop_len, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')

        plt.subplot(2, 1, 2)
        plt.title("Right ear Mel spectrogram")
        librosa.display.specshow(mel_right, sr=sr, hop_length=hop_len, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')

        plt.tight_layout()
        plt.show()
        plt.close()

    return mel_left, mel_right


# Processing binaural audio with raw STFT and (potentially) plotting sound waves and spectograms for sanity check. The idea of using this method instead of the mel spectrogram is to capture phase so we don't lose ITD
def process_binaural_audio_stft(file: str, sr: int = 44100, hop_len: int = 147, plot: bool = False):
    # Reading audio file
    audio, _ = librosa.load(file, sr=sr, mono=False)
    left, right = audio[0], audio[1]

    # (Potentially) Plotting sound waves
    if plot:
        fig, axes = plt.subplots(2, 1, figsize=(9, 9), sharey=True)

        axes[0].set_title("Left ear waveform")
        librosa.display.waveshow(left, sr=sr, color='blue', ax=axes[0])

        axes[1].set_title("Right ear waveform")
        librosa.display.waveshow(right, sr=sr, color='orange', ax=axes[1])

        plt.tight_layout()
        plt.show()
        plt.close()

    # Computing raw complex spectrogram. We use a lower n_samples than Mel (1024 instead of 2048) becuase we do preserve phase information with raw STFT, so we need better time resolution to make good use of ITD
    n_samples = 1024
    stft_left = librosa.stft(left, n_fft=n_samples, hop_length=hop_len)
    stft_right = librosa.stft(right, n_fft=n_samples, hop_length=hop_len)

    # Saving full spetrogram
    np.savez_compressed(file.replace('output.wav', 'full_stft'), left=stft_left, right=stft_right)

    # Extracting magnitude and phase components, as well as computing sine and cosine for smooth phase visualization
    mag_left, phase_left = np.abs(stft_left), np.angle(stft_left)
    mag_right, phase_right = np.abs(stft_right), np.angle(stft_right)
    cos_left, sin_left = np.cos(phase_left), np.sin(phase_left)
    cos_right, sin_right = np.cos(phase_right), np.sin(phase_right)

    # Converting from amplitude to dB's
    max_amplitude = np.max([mag_left, mag_right])
    mag_left = librosa.amplitude_to_db(mag_left, ref=max_amplitude)
    mag_right = librosa.amplitude_to_db(mag_right, ref=max_amplitude)

    # (Potentially) Plotting raw complex spectograms
    if plot:
        fig, axes = plt.subplots(4, 2, figsize=(9, 9), sharex=True)

        # Magnitude plots
        axes[0, 0].set_title('Left ear magnitude')
        img_0 = librosa.display.specshow(mag_left, sr=sr, hop_length=hop_len, x_axis='time', y_axis='log', ax=axes[0, 0])
        fig.colorbar(img_0, ax=axes[0, 0], format='%+2.0f dB')

        axes[0, 1].set_title('Right ear magnitude')
        img_1 = librosa.display.specshow(mag_right, sr=sr, hop_length=hop_len, x_axis='time', y_axis='log', ax=axes[0, 1])
        fig.colorbar(img_1, ax=axes[0, 1], format='%+2.0f dB')

        # Raw phase plots
        axes[1, 0].set_title('Left ear raw phase')
        img_2 = librosa.display.specshow(phase_left, sr=sr, hop_length=hop_len, x_axis='time', y_axis='linear', ax=axes[1, 0])
        fig.colorbar(img_2, ax=axes[1, 0], label='Radians')

        axes[1, 1].set_title('Right ear raw phase')
        img_3 = librosa.display.specshow(phase_right, sr=sr, hop_length=hop_len, x_axis='time', y_axis='linear', ax=axes[1, 1])
        fig.colorbar(img_3, ax=axes[1, 1], label='Radians')

        # Cosine of phase plots
        axes[2, 0].set_title('Left ear cosine phase')
        img_4 = librosa.display.specshow(cos_left, sr=sr, hop_length=hop_len, x_axis='time', y_axis='linear', ax=axes[2, 0])
        fig.colorbar(img_4, ax=axes[2, 0])

        axes[2, 1].set_title('Right ear cosine phase')
        img_5 = librosa.display.specshow(cos_right, sr=sr, hop_length=hop_len, x_axis='time', y_axis='linear', ax=axes[2, 1])
        fig.colorbar(img_5, ax=axes[2, 1])

        # Sine of phase plots
        axes[3, 0].set_title('Left ear sine phase')
        img_6 = librosa.display.specshow(sin_left, sr=sr, hop_length=hop_len, x_axis='time', y_axis='linear', ax=axes[3, 0])
        fig.colorbar(img_6, ax=axes[3, 0])

        axes[3, 1].set_title('Right ear sine phase')
        img_7 = librosa.display.specshow(sin_right, sr=sr, hop_length=hop_len, x_axis='time', y_axis='linear', ax=axes[3, 1])
        fig.colorbar(img_7, ax=axes[3, 1])

        plt.tight_layout()
        plt.show()
        plt.close()

    return stft_left, stft_right, mag_left, mag_right, phase_left, phase_right, cos_left, cos_right, sin_left, sin_right

# Given the processed audio data, we split it so we can later input it to our machine learning model
def split_data(data_list: list, file: str, sr: int = 44100, hop_len: int = 294, time_step: float = 0.2, plot: bool = False, method: str = 'mel'):
    # Computing image resolution and number of images
    samples_per_action = sr * time_step
    frames_per_action = int(samples_per_action/hop_len)
    n_images = int(data_list[0].shape[1]/frames_per_action)

    # Iterating data and saving images for dataset
    if method == 'mel':
        images = []
        for i in range(n_images):
            image = []
            for data in data_list:
                image.append(data[:, i*frames_per_action:(i+1)*frames_per_action])

            images.append(np.stack(image, axis=-1))
        images = np.array(images)

    elif method == 'stft':
        images = []
        complex_images = []
        for i in range(n_images):
            image = []
            complex_image = []
            for data in data_list[2:]:
                image.append(data[:, i*frames_per_action:(i+1)*frames_per_action])
            for data in data_list[:2]:
                complex_image.append(data[:, i*frames_per_action:(i+1)*frames_per_action])

            images.append(np.stack(image, axis=-1))
            complex_images.append(np.stack(complex_image, axis=-1))
        
        images = np.array(images)
        complex_images = np.array(complex_images)

    # Saving split spectrograms
    if method == 'mel':
        np.savez_compressed(file, left=images[:, :, :, 0], right=images[:, :, :, 1])
    elif method == 'stft':
        np.savez_compressed(file, left=complex_images[:, :, :, 0], right=complex_images[:, :, :, 1])
    
    # (Potentialy) Plot the first and second images
    if plot:
        if method == 'mel':
            fig, axes = plt.subplots(2, 2, figsize=(9, 9))
        elif method == 'stft':
            fig, axes = plt.subplots(4, 2, figsize=(9, 9))

        for i, image in enumerate([images[0], images[1]]):
            # Generating correct time coordiantes for current slice
            start_time = i * time_step
            times = librosa.times_like(image[:, :, 0], sr=sr, hop_length=hop_len) + start_time
            
            if method == 'mel':
                axes[i, 0].set_title('Left ear Mel spectrogram')
                img_0 = librosa.display.specshow(image[:, :, 0], sr=sr, hop_length=hop_len, x_coords=times, x_axis='time', y_axis='mel', ax=axes[i, 0])
                fig.colorbar(img_0, ax=axes[i, 0], format='%+2.0f dB')

                axes[i, 1].set_title('Right ear Mel spectrogram')
                img_1 = librosa.display.specshow(image[:, :, 1], sr=sr, hop_length=hop_len, x_coords=times, x_axis='time', y_axis='mel', ax=axes[i, 1])
                fig.colorbar(img_1, ax=axes[i, 1], format='%+2.0f dB')

            elif method == 'stft':
                # Magnitude plots
                axes[2*i, 0].set_title('Left ear magnitude')
                img_0 = librosa.display.specshow(image[:, :, 0], sr=sr, hop_length=hop_len, x_coords=times, x_axis='time', y_axis='log', ax=axes[2*i, 0])
                fig.colorbar(img_0, ax=axes[2*i, 0], format='%+2.0f dB')

                axes[2*i, 1].set_title('Right ear magnitude')
                img_1 = librosa.display.specshow(image[:, :, 2], sr=sr, hop_length=hop_len, x_coords=times, x_axis='time', y_axis='log', ax=axes[2*i, 1])
                fig.colorbar(img_1, ax=axes[2*i, 1], format='%+2.0f dB')

                # Raw phase plots
                axes[2*i+1, 0].set_title('Left ear raw phase')
                img_2 = librosa.display.specshow(image[:, :, 1], sr=sr, hop_length=hop_len, x_coords=times, x_axis='time', y_axis='linear', ax=axes[2*i+1, 0])
                fig.colorbar(img_2, ax=axes[2*i+1, 0], label='Radians')

                axes[2*i+1, 1].set_title('Right ear raw phase')
                img_3 = librosa.display.specshow(image[:, :, 3], sr=sr, hop_length=hop_len, x_coords=times, x_axis='time', y_axis='linear', ax=axes[2*i+1, 1])
                fig.colorbar(img_3, ax=axes[2*i+1, 1], label='Radians')

        plt.tight_layout()
        plt.show()
        plt.close()

    return images

if __name__ == "__main__":
    # Parsing arguments
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--file", help="Path to the audio file to be processed.", type=str, required=True)
    parser.add_argument("--method", help="Choosing a method to process the audio.", type=str, default='mel', choices=['mel', 'stft'])
    
    parser.add_argument("--sample_rate", help="Sample rate for sound simulation and later for audio spatialization.", type=int, default=44100)
    parser.add_argument("--hop_len", help="Length of hop for STFT computation. This is particularly important for the image resolution that we ara using later on a machine learning model.", type=int, default=294)
    parser.add_argument("--time_step", help="Time step used in the simulation.", type=float, default=0.2)

    
    parser.add_argument("--verbose", help="Print and plot everything. Meant for debugging.", action='store_true')

    args = parser.parse_args()

    # Processing audio
    if args.method == 'mel':
        mel_left, mel_right = process_binaural_audio_mel(args.file, args.sample_rate, args.hop_len, args.verbose)

        images = split_data([mel_left, mel_right], args.file.replace('output.wav', 'split_mel.npz'), args.sample_rate, args.hop_len, args.time_step, args.verbose, args.method)

    elif args.method == 'stft':
        stft_left, stft_right, mag_left, mag_right, phase_left, phase_right, cos_left, cos_right, sin_left, sin_right = process_binaural_audio_stft(args.file, args.sample_rate, args.hop_len, args.verbose)
        
        images = split_data([stft_left, stft_right, mag_left, phase_left, mag_right, phase_right], args.file.replace('output.wav', 'split_stft.npz'), args.sample_rate, args.hop_len, args.time_step, args.verbose, args.method)