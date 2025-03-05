import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import kurtosis, skew

# Define paths
output_directory = "write your directory"
design_matrix_file = "write your directory"

# Load the design matrix from the CSV file
design_df = pd.read_csv(design_matrix_file)

# Initialize a list to store results
results = []

# Loop through all parameter sets defined in the design matrix
for index, row in design_df.iterrows():
    try:
        # Load results based on the parameter set
        run_number = index + 1
        npy_file_path = os.path.join(output_directory, f'sample_{run_number}_matrix.npy')
        MSroi = np.load(npy_file_path)

        # Calculate relevant statistics
        mean_value = np.mean(MSroi)
        noise_ratio = np.std(MSroi) / mean_value if mean_value != 0 else np.inf
        signal_to_noise_ratio = mean_value / noise_ratio if noise_ratio != 0 else np.inf
        median_mz = np.median(MSroi)
        num_roi_peaks = np.count_nonzero(np.sum(MSroi, axis=0))
        peak_skewness = skew(MSroi, axis=0, nan_policy='omit')
        peak_kurtosis = kurtosis(MSroi, axis=0, nan_policy='omit')
        avg_skewness = np.nanmean(peak_skewness)
        avg_kurtosis = np.nanmean(peak_kurtosis)

        # Append the results along with the parameter set
        results.append({
            'Run': f'run{run_number}',
            'MeanValue': mean_value,
            'NoiseRatio': noise_ratio,
            'SignalToNoiseRatio': signal_to_noise_ratio,
            'MedianMZ': median_mz,
            'NumROIPeaks': num_roi_peaks,
            'AvgSkewness': avg_skewness,
            'AvgKurtosis': avg_kurtosis,
            'SignalIntensity': row['SignalIntensity'],
            'MassError': row['MassError'],
            'Occurrences': row['Occurrences']
        })

    except Exception as e:
        print(f"Error processing run {run_number}: {e}")

# Convert the results to a DataFrame
results_df = pd.DataFrame(results)
print("Processed Results DataFrame:")
print(results_df)

# Normalize the factors
scaler = MinMaxScaler()
results_df[['NoiseRatio', 'SignalToNoiseRatio', 'NumROIPeaks', 'AvgSkewness', 'AvgKurtosis']] = scaler.fit_transform(
    results_df[['NoiseRatio', 'SignalToNoiseRatio', 'NumROIPeaks', 'AvgSkewness', 'AvgKurtosis']]
)

# Define weights for each factor (adjust these weights according to importance)
weight_noise_ratio = 0.2
weight_signal_to_noise_ratio = 0.3
weight_num_roi_peaks = 0.3
weight_gaussianity = 0.2  # Considering both skewness and kurtosis

# Calculate a composite score for each run
results_df['CompositeScore'] = (
    weight_noise_ratio * (1 - results_df['NoiseRatio']) +  # Minimize NoiseRatio
    weight_signal_to_noise_ratio * results_df['SignalToNoiseRatio'] +  # Maximize SignalToNoiseRatio
    weight_num_roi_peaks * results_df['NumROIPeaks'] +  # Maximize NumROIPeaks
    weight_gaussianity * ((1 - np.abs(results_df['AvgSkewness'])) + (1 - np.abs(results_df['AvgKurtosis'] - 3))) / 2  # Gaussianity
)

# Select the run with the highest composite score
optimal_run = results_df.loc[results_df['CompositeScore'].idxmax()]

# Extract optimal settings
optimal_settings = {
    'SignalIntensity': optimal_run['SignalIntensity'],
    'MassError': optimal_run['MassError'],
    'Occurrences': optimal_run['Occurrences']
}

# Print the optimal settings and run ID
print("Optimal Settings:")
print(optimal_settings)
optimal_run_id = optimal_run['Run']
print(f"The run with the optimal parameters is: {optimal_run_id}")

# Visualization 1: Plot relationships between experimental factors and Number of ROI Peaks
sns.pairplot(results_df, x_vars=['SignalIntensity', 'MassError', 'Occurrences'],
             y_vars='NumROIPeaks', height=4, aspect=1.2)
plt.suptitle("Relationships between Experimental Factors and Number of ROI Peaks")
plt.show()

# Visualization 2: Plot Noise Ratio vs Number of ROI Peaks with Run Labels and Highlight Optimal Run
plt.figure(figsize=(10, 6))
plt.scatter(results_df['NoiseRatio'], results_df['NumROIPeaks'], color='blue', label='Runs')
plt.scatter(optimal_run['NoiseRatio'], optimal_run['NumROIPeaks'], color='red', s=100, marker='*', label='Optimal Run')
plt.title('Noise Ratio vs Number of ROI Peaks')
plt.xlabel('Noise Ratio')
plt.ylabel('Number of ROI Peaks')
for i in range(results_df.shape[0]):
    plt.text(results_df['NoiseRatio'].iloc[i], results_df['NumROIPeaks'].iloc[i],
             results_df['Run'].iloc[i], fontsize=9, ha='right')
plt.legend()
plt.show()

# Visualization 3: Plot Signal-to-Noise Ratio vs Number of ROI Peaks with Run Labels and Highlight Optimal Run
plt.figure(figsize=(10, 6))
plt.scatter(results_df['SignalToNoiseRatio'], results_df['NumROIPeaks'], color='blue', label='Runs')
plt.scatter(optimal_run['SignalToNoiseRatio'], optimal_run['NumROIPeaks'], color='red', s=100, marker='*', label='Optimal Run')
plt.title('Signal-to-Noise Ratio vs Number of ROI Peaks')
plt.xlabel('Signal-to-Noise Ratio')
plt.ylabel('Number of ROI Peaks')
for i in range(results_df.shape[0]):
    plt.text(results_df['SignalToNoiseRatio'].iloc[i], results_df['NumROIPeaks'].iloc[i],
             results_df['Run'].iloc[i], fontsize=9, ha='right')
plt.legend()
plt.show()
