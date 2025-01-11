import numpy as np
from scipy import signal
from scipy.integrate import simpson
from typing import Optional, Dict
from dataclasses import dataclass
import logging
import os
import pickle
from eeg_data import EEGData

CACHE_DIR = r'temp'

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define frequency bands
FREQUENCY_BANDS = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30)
}

# @dataclass
# class EEGData:
#     """Class to encapsulate EEG data and metadata."""
#     data: np.ndarray  # EEG data array
#     labels: np.ndarray  # Corresponding labels
#     fs: int  # Sampling frequency
#     channel_names: Optional[list] = None  # Optional metadata
#     features: Optional[Dict[str, np.ndarray]] = None  # Extracted features

def compute_band_features(data: np.ndarray, fs: int, band: tuple, feature_type: str) -> np.ndarray:
    """
    Compute features (PSD or DE) for a specific frequency band.

    Args:
        data: EEG data with shape (n_samples, n_channels, n_timepoints).
        fs: Sampling frequency.
        band: Frequency band (low, high) in Hz.
        feature_type: Type of feature to compute ('psd' or 'de').

    Returns:
        Feature values with shape (n_samples, n_channels).
    """
    if feature_type == 'psd':
        # Compute power spectral density (PSD) using Welch's method
        freqs, psd = signal.welch(data, fs, nperseg=fs, axis=-1)  # Shape: (n_samples, n_channels, n_freqs)
        band_mask = (freqs >= band[0]) & (freqs <= band[1])
        # Use keyword arguments for simpson
        psd_band = simpson(y=psd[:, :, band_mask], x=freqs[band_mask], axis=-1)  # Shape: (n_samples, n_channels)
        return psd_band
    elif feature_type == 'de':
        # Bandpass filter the data
        sos = signal.butter(4, band, btype='bandpass', fs=fs, output='sos')
        filtered_data = signal.sosfiltfilt(sos, data, axis=-1)
        # Compute differential entropy
        return 0.5 * np.log(2 * np.pi * np.exp(1) * np.var(filtered_data, axis=-1))
    else:
        raise ValueError(f"Unsupported feature type: {feature_type}")

def extract_features(
    eeg_data: EEGData,
    feature_list: list = ['psd', 'de'],  # 默认提取 PSD 和 DE
    cache_path: Optional[str] = 'temp'
) -> EEGData:
    """
    Extract specified features (PSD and/or DE) for each frequency band from EEG data.

    Args:
        eeg_data: EEGData object containing EEG data.
        feature_list: List of features to extract ('psd' and/or 'de').
        cache_file: Path to the cache file. If None, no caching is used.

    Returns:
        EEGData object with extracted features.
    """
    cache_file = os.path.join(cache_path, 'eeg_features_{}.pkl'.format('_'.join(feature_list))) if cache_path else None
    if cache_file and os.path.exists(cache_file):
        logging.info(f'Loading cached features from {cache_file}...')
        with open(cache_file, 'rb') as f:
            eeg_data.features = pickle.load(f)
        return eeg_data

    logging.info('Extracting features from EEG data...')
    features = {}

    for feature_type in feature_list:
        if feature_type not in ['psd', 'de']:
            raise ValueError(f"Unsupported feature type: {feature_type}. Supported types are 'psd' and 'de'.")

        # Initialize a list to store feature data for all frequency bands
        feature_data = []

        for band_range in FREQUENCY_BANDS.values():
            # Compute the feature for the current frequency band
            band_feature = compute_band_features(eeg_data.data, eeg_data.fs, band_range, feature_type)
            feature_data.append(band_feature)

        # Stack feature data along the last dimension (n_samples, n_channels, n_bands)
        feature_data = np.stack(feature_data, axis=-1)  # Shape: (n_samples, n_channels, n_bands)
        features[feature_type] = feature_data

    # Add features to the EEGData object
    eeg_data.features = features

    if cache_file:
        logging.info(f'Saving features to cache file {cache_file}...')
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(eeg_data.features, f)

    logging.info('Feature extraction completed.')
    return eeg_data

if __name__ == '__main__':
    # Example usage
    from preprocess import load_data, slice_data

    # Load data
    eeg_data = load_data(cache_path='temp', dataname='deap')

    sliced_data = slice_data(eeg_data)

    # Extract features
    eeg_data_with_features = extract_features(
        sliced_data,
        # cache_file=os.path.join(CACHE_DIR, 'eeg_features.pkl')
    )

    # Print extracted features
    # for band_name in FREQUENCY_BANDS.keys():
    print(f'psd:', eeg_data_with_features.features[f'psd'].shape)
    print(f'de:', eeg_data_with_features.features[f'de'].shape)