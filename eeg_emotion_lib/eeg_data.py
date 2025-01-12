# eeg_data.py
import numpy as np
from dataclasses import dataclass
from typing import Optional
import logging
import os
import pickle

@dataclass
class EEGData:
    """Class to encapsulate EEG data and metadata."""
    dataname: str
    data: np.ndarray  # EEG data array (n_samples, n_channels, n_timepoints)
    labels: np.ndarray  # Corresponding labels (n_samples,)
    fs: int  # Sampling frequency
    channel_names: Optional[list] = None  # Optional metadata
    features: Optional[dict] = None  # Extracted features
    cache_path: Optional[str] = "temp"

    def save_to_cache(self, cache_file: str) -> None:
        """Save EEGData object to a cache file."""
        try:
            logging.info(f'Saving data to cache file {cache_file}...')
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump(self, f)
        except (OSError, IOError) as e:
            logging.error(f'Failed to save cache file {cache_file}: {e}')

    @staticmethod
    def load_from_cache(cache_file: str) -> Optional['EEGData']:
        """Load EEGData object from a cache file if it exists."""
        if os.path.exists(cache_file):
            try:
                logging.info(f'Loading cached data from {cache_file}...')
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except (pickle.UnpicklingError, IOError) as e:
                logging.error(f'Failed to load cache file {cache_file}: {e}')
        return None
