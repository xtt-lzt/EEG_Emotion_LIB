import os
import pickle
import logging
from typing import Optional, Tuple
from dataclasses import dataclass
from eeg_data import EEGData

import numpy as np

# Constants
FS = 128  # Sampling frequency
EEG_PATH = r'D:\zitao\torch_eeg\data_preprocessed_python'
CACHE_DIR = 'temp'
BASELINE_WINDOW_WIDTH = 1  # in seconds

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Channel positions for mapping EEG data to a 9x9 matrix
CHANNEL_POSITIONS = [
    (0, 3), (1, 3), (2, 2), (2, 0), (3, 1), (3, 3), (4, 2), (4, 1),
    (5, 1), (5, 3), (6, 2), (6, 0), (7, 3), (8, 3), (8, 4), (6, 4),
    (0, 5), (1, 5), (2, 4), (2, 6), (2, 8), (3, 7), (3, 5), (4, 4),
    (4, 6), (4, 8), (5, 7), (5, 5), (6, 6), (6, 8), (7, 5), (8, 5)
]


# @dataclass
# class EEGData:
#     """Class to encapsulate EEG data and metadata."""
#     dataname: str
#     data: np.ndarray  # EEG data array (n_samples, n_channels, n_timepoints)
#     labels: np.ndarray  # Corresponding labels (n_samples,)
#     fs: int  # Sampling frequency
#     channel_names: Optional[list] = None  # Optional metadata

#     # def __post_init__(self):
#     #     """
#     #     Validate the shape of the data and labels after initialization.
#     #     """
#     #     # Validate data shape
#     #     if self.data.ndim != 4:
#     #         raise ValueError(f"Data must be 3-dimensional (n_samples, n_channels, n_timepoints). Got shape: {self.data.shape}")
        
#     #     # Validate labels shape
#     #     if self.labels.ndim != 3:
#     #         raise ValueError(f"Labels must be 1-dimensional (n_samples,). Got shape: {self.labels.shape}")
        
#     #     # Check if the number of samples matches between data and labels
#     #     if self.data.shape[0] != self.labels.shape[0]:
#     #         raise ValueError(
#     #             f"Number of samples in data ({self.data.shape[0]}) does not match number of labels ({self.labels.shape[0]})."
#     #         )

#     def save_to_cache(self, cache_file: str) -> None:
#         """Save EEGData object to a cache file."""
#         logging.info(f'Saving data to cache file {cache_file}...')
#         os.makedirs(os.path.dirname(cache_file), exist_ok=True)
#         with open(cache_file, 'wb') as f:
#             pickle.dump(self, f)

#     @staticmethod
#     def load_from_cache(cache_file: str) -> Optional['EEGData']:
#         """Load EEGData object from a cache file if it exists."""
#         if os.path.exists(cache_file):
#             logging.info(f'Loading cached data from {cache_file}...')
#             with open(cache_file, 'rb') as f:
#                 return pickle.load(f)
#         return None


def load_data(
    eeg_path: str = EEG_PATH,
    dataname: Optional[str] = None,
    cache_path: Optional[str] = "temp"
) -> EEGData:
    """
    Load EEG data from files or cache.

    Args:
        eeg_path: Path to the directory containing EEG data files.
        dataname: Name of the dataset (e.g., 'deap'). If None, assumes default processing.
        cache_file: Path to the cache file. If None, no caching is used.

    Returns:
        EEGData object containing EEG data and labels.
    """
    cache_file = os.path.join(cache_path, f'{dataname}_data.pkl') if cache_path else None

    if cache_file:
        cached_data = EEGData.load_from_cache(cache_file)
        if cached_data:
            return cached_data

    logging.info('Extracting EEG data from files...')

    if dataname == 'deap':
        # DEAP dataset processing
        eeg_files = [os.path.join(eeg_path, f) for f in os.listdir(eeg_path) if os.path.isfile(os.path.join(eeg_path, f))]
        eeg_datas, labels = zip(*[
            (np.array(data['data'][:, :32, :]), np.array(data['labels']))
            for file in eeg_files
            for data in [pickle.load(open(file, 'rb'), encoding='latin1')]
        ])

        eeg_data = EEGData(dataname=dataname, data=np.array(eeg_datas), labels=np.array(labels), fs=128)
    else:
        # Placeholder for other datasets
        raise NotImplementedError(f"Data processing for dataset '{dataname}' is not implemented yet.")

    if cache_file:
        eeg_data.save_to_cache(cache_file)

    logging.info(f'EEG data shape: {eeg_data.data.shape}')
    logging.info(f'Labels shape: {eeg_data.labels.shape}')
    return eeg_data


def slice_data(
    eeg_data: EEGData,
    window_width: int = BASELINE_WINDOW_WIDTH,
    step_len: int = 1,
    remove_baseline: bool = True,
    cache_path: Optional[str] = CACHE_DIR
) -> EEGData:
    """
    Slice EEG data into windows with a specified step length, optionally removing baseline.

    Args:
        eeg_data: EEGData object containing raw EEG data and labels.
        window_width: Width of the slicing window in seconds.
        step_len: Step length for sliding the window in seconds.
        remove_baseline: Whether to remove baseline from the data.
        cache_file: Path to the cache file. If None, no caching is used.

    Returns:
        EEGData object containing sliced EEG data and labels.
    """
    cache_file = os.path.join(cache_path, f'sliced_{eeg_data.dataname}_{window_width}s_{step_len}s_data.pkl') if cache_path else None
    if cache_file:
        cached_data = EEGData.load_from_cache(cache_file)
        if cached_data:
            return cached_data

    logging.info('Slicing data...')
    eeg_datas = eeg_data.data[:, :, :, 3 * FS:]  # Remove the first 3 seconds as baseline

    sliced_eeg_datas = []
    sliced_labels = []

    for trial in range(eeg_datas.shape[0]):
        for video in range(eeg_datas.shape[1]):
            # Compute baseline if remove_baseline is True
            baseline = eeg_datas[trial, video, :, :3 * FS].mean(axis=-1, keepdims=True) if remove_baseline else 0

            # Slice the data into windows with the specified step length
            for start in range(0, eeg_datas.shape[3] - window_width * FS + 1, step_len * FS):
                end = start + window_width * FS
                sliced_eeg_datas.append(eeg_datas[trial, video, :, start:end] - baseline)
                sliced_labels.append(eeg_data.labels[trial, video])

    sliced_eeg_data = EEGData(
        dataname=eeg_data.dataname,
        data=np.array(sliced_eeg_datas),
        labels=np.array(sliced_labels),
        fs=eeg_data.fs,
        channel_names=eeg_data.channel_names
    )

    if cache_file:
        sliced_eeg_data.save_to_cache(cache_file)

    logging.info(f'Sliced EEG data shape: {sliced_eeg_data.data.shape}')
    logging.info(f'Sliced labels shape: {sliced_eeg_data.labels.shape}')
    return sliced_eeg_data


# def map_data_to_matrix(eeg_data: EEGData) -> np.ndarray:
#     """
#     Map EEG data to a 9x9 matrix based on channel positions.

#     Args:
#         eeg_data: EEGData object containing EEG data.

#     Returns:
#         Mapped EEG data with shape (n_samples, 9, 9, n_timepoints).
#     """
#     logging.info('Mapping EEG data to 9x9 matrix...')
#     mapped_data = np.zeros((eeg_data.data.shape[0], 9, 9, eeg_data.data.shape[2]))
#     for channel_idx, (row, col) in enumerate(CHANNEL_POSITIONS):
#         if 0 <= row < 9 and 0 <= col < 9:
#             mapped_data[:, row, col, :] = eeg_data.data[:, channel_idx, :]
#         else:
#             raise ValueError(f"Invalid position ({row}, {col}) for channel {channel_idx}")

#     logging.info(f'Mapped EEG data shape: {mapped_data.shape}')
#     return mapped_data


if __name__ == '__main__':
    # Load data
    eeg_data = load_data(cache_path='temp', dataname='deap')

    # Slice data
    sliced_eeg_data = slice_data(
        eeg_data,
        cache_path=CACHE_DIR
    )
    print(sliced_eeg_data.data.shape)
    print(sliced_eeg_data.labels.shape)

    # Map data to 9x9 matrix
    # mapped_data = map_data_to_matrix(sliced_eeg_data)