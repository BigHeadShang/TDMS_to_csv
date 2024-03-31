import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
from scipy.signal import stft, cwt, morlet
import librosa.display
from scipy.signal import spectrogram
from tftb.processing import WignerVilleDistribution
from datetime import datetime
from nptdms import TdmsFile
import concurrent.futures

def sanitize_filename(filename):
    """Remove spaces from the filename."""
    return filename.replace(' ', '')

def convert_tdms_to_csv(tdms_path, output_folder):
    """Convert a single TDMS file to a CSV file and return the CSV file path."""
    base_filename = sanitize_filename(os.path.basename(tdms_path))
    csv_filename = base_filename.replace('.tdms', '.csv')
    csv_path = os.path.join(output_folder, csv_filename)
    tdms_file = TdmsFile.read(tdms_path)
    df = tdms_file.as_dataframe()
    df.to_csv(csv_path, index=False)
    print(f'Converted {tdms_path} to {csv_path}')
    return csv_path

def merge_csv_files(files, output_folder, timestamp):
    """Merge all CSV files into a single file, standardizing column headers."""
    dataframes = []
    for i, file in enumerate(files):
        try:
            df = pd.read_csv(file)

            # 检查是否为空
            if df.empty:
                print(f"Skipping empty file: {file}")
                continue

            # 重命名列以确保一致性，例如使用通用列名：CH1, CH2, ..., CHn
            column_count = len(df.columns)
            standardized_columns = ['CH' + str(i) for i in range(column_count)]
            df.columns = standardized_columns

            dataframes.append(df)
            print(f"Processed file: {file} | Columns = {len(df.columns)}, Rows = {len(df)}")
        except pd.errors.EmptyDataError:
            print(f"No data in {file}, skipping.")
        except Exception as e:
            print(f"Error reading {file}: {e}")
            continue

    # 合并所有DataFrame
    if dataframes:
        merged_df = pd.concat(dataframes, ignore_index=True)
        print(f"After Processed file: {file} | Columns = {len(merged_df.columns)}, Rows = {len(merged_df)}")
        # 再次检查列名一致性，此时所有DataFrame应该有相同的列名
        merged_filename = os.path.join(output_folder, f"merge_csv_{timestamp}.csv")
        merged_df.to_csv(merged_filename, index=False)
        print(f"Merged CSV files into {merged_filename}")
        return merged_filename
    else:
        print("No valid CSV files to merge.")
        return None


def stft_transform(csv_file, channel):
    """Perform Short-Time Fourier Transform on selected channel."""
    df = pd.read_csv(csv_file)
    data = df[channel].values
    f, t, Zxx = stft(data)
    plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
    plt.title(f'STFT Magnitude - {channel}')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar()
    plt.savefig(csv_file.replace('.csv', f'_{channel}_stft.png'))
    plt.close()
    print(f"STFT transformation completed for {channel}. Image saved.")

def cwt_transform(csv_file, channel):
    """Perform Continuous Wavelet Transform on selected channel."""
    df = pd.read_csv(csv_file)
    data = df[channel].values
    widths = np.arange(1, 31)
    cwt_matr = cwt(data, morlet, widths)
    plt.imshow(np.abs(cwt_matr), extent=[-1, 1, 1, 31], cmap='PRGn', aspect='auto')
    plt.title(f'Continuous Wavelet Transform - {channel}')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.savefig(csv_file.replace('.csv', f'_{channel}_cwt.png'))
    plt.close()
    print(f"CWT transformation completed for {channel}. Image saved.")


def mel_spectrogram_transform(csv_file, channel):
    """Perform Mel Spectrogram transformation on selected channel."""
    df = pd.read_csv(csv_file)
    data = df[channel].values
    S = librosa.feature.melspectrogram(y=data, sr=22050, n_mels=128, fmax=8000)
    S_DB = librosa.power_to_db(S, ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_DB, sr=22050, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.tight_layout()
    plt.savefig(csv_file.replace('.csv', f'_{channel}_mel_spectrogram.png'))
    plt.close()
    print(f"Mel Spectrogram transformation completed for {channel}. Image saved.")

def s_transform(csv_file, channel):
    """Perform a simplified S-Transform using spectrogram."""
    df = pd.read_csv(csv_file)
    data = df[channel].values
    f, t, Sxx = spectrogram(data, fs=1.0)
    plt.pcolormesh(t, f, 10 * np.log10(Sxx))
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title('Spectrogram (Simplified S-Transform)')
    plt.savefig(csv_file.replace('.csv', f'_{channel}_s_transform.png'))
    plt.close()
    print(f"S-Transform (simplified) completed for {channel}. Image saved.")

def wigner_ville_distribution(csv_file, channel):
    """Perform a Wigner-Ville Distribution transformation on selected channel."""
    df = pd.read_csv(csv_file)
    data = df[channel].values
    wvd = WignerVilleDistribution(data)
    tfr, _, _ = wvd.run()
    plt.imshow(abs(tfr), extent=[0, 1, 0, 0.5], aspect='auto', origin='lower')
    plt.title('Wigner-Ville Distribution')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.savefig(csv_file.replace('.csv', f'_{channel}_wvd.png'))
    plt.close()
    print(f"Wigner-Ville Distribution transformation completed for {channel}. Image saved.")


def perform_transformation(merge_csv_path, channels, algorithm):
    """Perform transformation based on selected algorithm."""
    for channel in channels:
        if algorithm == 1:  # Mel Spectrogram
            mel_spectrogram_transform(merge_csv_path, channel)
        elif algorithm == 2:  # STFT
            stft_transform(merge_csv_path, channel)
        elif algorithm == 3:  # CWT
            cwt_transform(merge_csv_path, channel)
        elif algorithm == 4:  # Wigner-Ville Distribution
            wigner_ville_distribution(merge_csv_path, channel)
        elif algorithm == 5:  # Simplified S-Transform
            s_transform(merge_csv_path, channel)
        else:
            print(f"Algorithm {algorithm} not implemented for {channel}.")

def process_folder(folder_path, output_folder, should_merge, transform, channels, algorithm):
    """Process all TDMS files in a given folder concurrently, merge the converted CSV files, and optionally transform them."""
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_csv = {executor.submit(convert_tdms_to_csv, os.path.join(folder_path, filename), output_folder): filename
                         for filename in sorted(os.listdir(folder_path))
                         if filename.lower().endswith('.tdms')}

        csv_files = []
        for future in concurrent.futures.as_completed(future_to_csv):
            csv_path = future.result()
            if csv_path:
                csv_files.append(csv_path)
                print(f"Converted: {csv_path}")

    merged_csv_path = ""
    if should_merge and csv_files:
        csv_files_sorted = sorted(csv_files, key=lambda x: os.path.getmtime(x))
        merged_csv_path = merge_csv_files(csv_files_sorted, output_folder, datetime.now().strftime('%Y%m%d%H%M'))

    if transform and channels and algorithm:
        if should_merge and merged_csv_path:
            perform_transformation(merged_csv_path, channels, algorithm)
        elif not should_merge:
            for csv_file in csv_files:
                perform_transformation(csv_file, channels, algorithm)

def main():
    parser = argparse.ArgumentParser(description="Convert TDMS files to CSV format, optionally merge and transform them.")
    parser.add_argument('path', type=str, help='The path to the TDMS file or folder to process.')
    parser.add_argument('-o', '--output', type=str, help='The output folder for the CSV files.')
    parser.add_argument('--merge', type=lambda x: (str(x).lower() == 'true'), default=True, help='Whether to merge the CSV files.')
    parser.add_argument('--transform', type=lambda x: (str(x).lower() == 'true'), default=False, help='Whether to transform the merged CSV into a 2D image.')
    parser.add_argument('--channels', type=str, nargs='+', help='List of channels to be used for transformation.')
    parser.add_argument('--algorithm', type=int, choices=range(1, 28), help='The algorithm number to be used for transformation.')

    args = parser.parse_args()

    # 处理输入路径，转换为绝对路径
    tdms_path = os.path.abspath(args.path)

    # 处理输出路径，如果未指定则创建默认目录，并转换为绝对路径
    output_folder = os.path.abspath(args.output if args.output else os.path.join(os.getcwd(), datetime.now().strftime('%Y%m%d%H%M') + '_tdms'))

    # 确保输出目录存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 检查路径是文件还是目录，并进行相应处理
    if os.path.isdir(tdms_path):
        process_folder(tdms_path, output_folder, args.merge, args.transform, args.channels, args.algorithm)
    elif os.path.isfile(tdms_path) and tdms_path.lower().endswith('.tdms'):
        csv_path = convert_tdms_to_csv(tdms_path, output_folder)
        if args.transform and args.channels and args.algorithm:
            perform_transformation(csv_path, args.channels, args.algorithm)

if __name__ == "__main__":
    main()