# TDMS to CSV Converter with Data Transformation

This Python script, developed by Zhanhao Shang, is designed to convert TDMS (Technical Data Management Streaming) files to CSV format and provide post-processing options, including merging and transforming data into different 2D image forms for further analysis.

## Features

- **TDMS to CSV Conversion**: Convert single or multiple TDMS files in a folder to CSV format.
- **CSV Merging**: Optionally merge multiple CSV files resulting from the conversion into a single file.
- **Data Transformation**: Apply different algorithms to selected channels from the merged (or individual) CSV file to transform them into 2D image representations. Supported transformations include:
  - Mel Spectrogram
  - Short-Time Fourier Transform (STFT)
  - Continuous Wavelet Transform (CWT)
  - Wigner-Ville Distribution
  - Simplified S-Transform

## Usage

Assume you have one or more TDMS files that you wish to convert to CSV and then apply an STFT transformation on specific channels. Follow the steps below:

### Preparation

Ensure all your TDMS files are located in the same directory, e.g., `/path/to/tdms/files`.

### Executing the Script

Run the script from the command line, specifying the path to the TDMS file(s), output path, whether to merge CSV files, and the transformation to apply, along with the specific channels:

## Results

After executing the command, the converted CSV files and the 2D images from the applied transformation on specified channels will be available in the specified output directory.

You can adjust the `--algorithm` parameter in the command line to choose different data transformation algorithms: `1` for Mel Spectrogram, `3` for Continuous Wavelet Transform, `4` for Wigner-Ville Distribution, and `5` for the Simplified S-Transform.

## Developer

- Zhanhao Shang

```bash
python convert_to_csv.py /path/to/tdms/files -o /path/to/output --merge true --transform true --channels 'CH0' 'CH1' --algorithm 2

### Parameters:

- `/path/to/tdms/files`: Path to the TDMS file or directory containing multiple TDMS files.
- `-o /path/to/output`: Output directory for the converted CSV files.
- `--merge true`: Indicates the script should merge the output CSV files.
- `--transform true`: Indicates the script should apply a transformation algorithm to the output data.
- `--channels '/DAQ/CH0' '/DAQ/CH1'`: Specifies the channels to transform. List any of `'/DAQ/CH0'` to `'/DAQ/CH13'`.
- `--algorithm 2`: Specifies the transformation algorithm to use, where `2` is for STFT.


