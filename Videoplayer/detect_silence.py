import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
from scipy.ndimage import maximum_filter1d
import moviepy.editor as mp


def detect_silence(file_path, window_size_ms=100, silence_threshold=0.002, merge_threshold=0.3, min_silence_duration=0.1):
    """
    Function detects intervals of silence in a video.

    Parameters:
    file_path (string):
        filename of the audio to be processed    
    window_size_ms=100 (int):
        size of windows used to test in the mel spectogram    
    silence_threshold=0.002 (float):
        maximum sound level before considered silent    
    merge_threshold=0.3 (float):
        WHAT EXACTLY DOES THIS DO?
    min_silence_duration=0.1 (float):
        minimum length of silence interval. If silence is shorter, it gets ignored.

    Returns:
    y (int):
        ?
    sr (int):
        Sampling rate of the processed audio
    silent_segments (tuple):
        number of start end points, one for each silent segment
    silenceSum (float):
        sum of the length of all silent moments
    """

    # Load audio file
    y, sr               = sf.read(file_path)    
    if len(y.shape) > 1:
        y = np.mean(y, axis=-1)
    # Calculate the window size in samples
    window_size_samples = int(sr * window_size_ms / 1000)    
    # Compute the moving maximum
    moving_max          = maximum_filter1d(y, size=window_size_samples)    
    # Identify silence windows
    is_silence          = moving_max < silence_threshold
    num_windows         = len(moving_max) // window_size_samples
    silence_windows     = np.split(is_silence[:num_windows * window_size_samples], num_windows)
    silence_flags       = [np.mean(window) > 0.95 for window in silence_windows]    
    # Find start and end points of silent segments
    silent_segments     = []
    in_silence          = False
    start_time          = 0    
    for i, flag in enumerate(silence_flags):
        if flag and not in_silence:
            start_time  = i * window_size_ms / 1000
            in_silence  = True
        elif not flag and in_silence:
            end_time    = i * window_size_ms / 1000
            silent_segments.append((start_time, end_time))
            in_silence  = False
    if in_silence:  # Handle case where file ends with silence
        end_time = len(y) / sr
        silent_segments.append((start_time, end_time))    
    # Post-process to merge close silence periods
    merged_silent_segments = []
    current_start, current_end = silent_segments[0]    
    for start, end in silent_segments[1:]:
        if start - current_end < merge_threshold:
            current_end = end
        else:
            merged_silent_segments.append((current_start, current_end))
            current_start, current_end = start, end
    merged_silent_segments.append((current_start, current_end))  # Append the last segment
    
    # Further process to keep only silent segments longer than min_silence_duration
    final_silent_segments = [(start, end) for start, end in merged_silent_segments if (end - start) >= min_silence_duration]
    
    # Create silent_segments directory if it does not exist
    silent_segments_dir = os.path.join('silent_segments', os.path.dirname(file_path))
    os.makedirs(silent_segments_dir, exist_ok=True)
    
    # Save filtered silent segments to a txt file in silent_segments directory
    output_file_path = os.path.join(silent_segments_dir, os.path.splitext(os.path.basename(file_path))[0] + '.txt')
    silenceSum = 0.0
    with open(output_file_path, 'w') as f:
        for start, end in final_silent_segments:
            dur = end - start
            silenceSum = silenceSum + dur
            print(f"{start:.2f} {end:.2f} {dur:.2f}\n")
            f.write(f"{start:.2f} {end:.2f}\n")
    print("\n\n")
    return y, sr, final_silent_segments, silenceSum


def plot_waveform_and_melspectrogram(y, sr, file_path, silent_segments):
    """
    Function detects intervals of silence in a video.

    Parameters:
    y (?):
        ?    
    sr (int):
        sampling rate of the original audio file     
    file_path (string):
        Name to be displayed on the window showing the output    
    silent_segments (tuples):
        Start and end points of the silent segments
    """

    # Plot waveform and mel spectrogram
    plt.figure(figsize=(14, 10))    
    # Plot waveform
    plt.subplot(2, 1, 1)
    plt.plot(np.linspace(0, (len(y) - 1) * 1/sr, len(y)), y)
    plt.title('Waveform - ' + f'{os.path.basename(file_path)}')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')    
    # Draw vertical lines for silence segments
    start_label_done    = False
    end_label_done      = False    
    for start, end in silent_segments:
        plt.axvline(x=start, color='red', linestyle='--', label='Start of Silence' if not start_label_done else "")
        start_label_done = True
        plt.axvline(x=end, color='black', linestyle='--', label='End of Silence' if not end_label_done else "")
        end_label_done = True    
    # Show legend
    plt.legend()    
    # Compute mel spectrogram
    S    = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    print("shape of S_dB",S_dB.shape)    
    # Plot mel spectrogram
    plt.subplot(2, 1, 2)
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram - ' + f'{os.path.basename(file_path)}')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')    
    plt.tight_layout()
    plt.savefig("afasfasf.png")
    plt.show()


def process_folder(folder_path):
    """
    Function loops over all the files in the folder and displayes the wav with
    the silent intervalls overlapped.

    Parameters:
    folder_path (string):
        folder to be processed
    """
    
    tmpWav = "E:/Datasets_Crafter/tmp/tmp.wav"
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.MXF'):
            print("Testing file: ", file_name)
            file_path               = os.path.join(folder_path, file_name)    
            clip                    = mp.VideoFileClip(file_path)
            clip.audio.write_audiofile(tmpWav)
            y, sr, silent_segments  = detect_silence(tmpWav)
            print("Y and sr are: ", y, sr)
            plot_waveform_and_melspectrogram(y, sr, tmpWav, silent_segments)


def visualizeSingleAudioFile(file):
    """
    Function processes the audio of a single video and displays the wav with
    the silent intervalls ovrelapped.

    Parameters:
    file (string):
        video/audio file to be processed and displayed
    """
    
    tmpWav = "E:/Datasets_Crafter/tmp/tmp.wav"
    print("Testing file: ", file)   
    clip = mp.VideoFileClip(file)
    clip.audio.write_audiofile(tmpWav)
    y, sr, silent_segments = detect_silence(tmpWav)
    print("Y and sr are: ", y, sr)
    plot_waveform_and_melspectrogram(y, sr, tmpWav, silent_segments)

