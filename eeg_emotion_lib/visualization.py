import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns
from typing import Optional

def plot_eeg_signals(eeg_data: np.ndarray, fs: int, channel_names: Optional[list] = None, title: str = "EEG Signals"):
    """
    Plot the time-domain EEG signals for each channel.

    Args:
        eeg_data: EEG data with shape (n_samples, n_channels, n_timepoints).
        fs: Sampling frequency.
        channel_names: List of channel names. If None, channels are labeled as 0, 1, 2, etc.
        title: Title of the plot.
    """
    n_samples, n_channels, n_timepoints = eeg_data.shape
    time = np.arange(n_timepoints) / fs

    plt.figure(figsize=(12, 6))
    for channel in range(n_channels):
        label = channel_names[channel] if channel_names else f"Channel {channel}"
        plt.plot(time, eeg_data[0, channel, :], label=label)  # Plot the first sample for clarity
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()

def plot_psd_features(psd_data: np.ndarray, freqs: np.ndarray, band_ranges: dict, title: str = "PSD Features"):
    """
    Plot the PSD features for each frequency band.

    Args:
        psd_data: PSD data with shape (n_samples, n_channels, n_bands).
        freqs: Frequency values corresponding to the PSD data.
        band_ranges: Dictionary of frequency bands (e.g., {'delta': (1, 4), 'theta': (4, 8)}).
        title: Title of the plot.
    """
    n_samples, n_channels, n_bands = psd_data.shape
    band_names = list(band_ranges.keys())

    plt.figure(figsize=(12, 6))
    for band_idx, band_name in enumerate(band_names):
        plt.plot(freqs, psd_data[0, :, band_idx].T, label=f"{band_name} band")  # Plot the first sample for clarity
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectral Density")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()

def plot_feature_matrix(feature_matrix: np.ndarray, title: str = "Feature Matrix"):
    """
    Plot the feature matrix as a heatmap.

    Args:
        feature_matrix: Feature matrix with shape (n_samples, n_features).
        title: Title of the plot.
    """
    plt.figure(figsize=(10, 6))
    sns.heatmap(feature_matrix, cmap="viridis", cbar=True)
    plt.xlabel("Features")
    plt.ylabel("Samples")
    plt.title(title)
    plt.show()

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, title: str = "Confusion Matrix"):
    """
    Plot the confusion matrix for classification results.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        title: Title of the plot.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.show()

def plot_roc_curve(y_true: np.ndarray, y_scores: np.ndarray, title: str = "ROC Curve"):
    """
    Plot the ROC curve for classification results.

    Args:
        y_true: True labels.
        y_scores: Predicted scores (probabilities).
        title: Title of the plot.
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()