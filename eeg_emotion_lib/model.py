import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import logging
import matplotlib.pyplot as plt
from typing import Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CACHE_DIR = r'temp'

FREQUENCY_BANDS = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30)
}

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DeepLearningModel(tf.keras.Model):
    """
    A deep learning model for EEG classification.
    """
    def __init__(self, input_shape: tuple, num_classes: int):
        """
        Initialize the model.

        Args:
            input_shape: Shape of the input data (n_channels, n_timepoints).
            num_classes: Number of output classes.
        """
        super(DeepLearningModel, self).__init__()
        self.conv1 = layers.Conv1D(32, kernel_size=3, strides=1, padding='same', activation='relu')
        self.conv2 = layers.Conv1D(64, kernel_size=3, strides=1, padding='same', activation='relu')
        self.pool = layers.MaxPooling1D(pool_size=2, strides=2)
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(128, activation='relu')
        self.dropout = layers.Dropout(0.5)
        self.fc2 = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs, training=False):
        """
        Forward pass of the model.

        Args:
            inputs: Input tensor with shape (batch_size, n_channels, n_timepoints).
            training: Whether the model is in training mode.

        Returns:
            Output tensor with shape (batch_size, num_classes).
        """
        x = self.conv1(inputs)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x, training=training)
        return self.fc2(x)

def create_dataset(data: np.ndarray, labels: np.ndarray, batch_size: int = 32, shuffle: bool = True) -> tf.data.Dataset:
    """
    Create a TensorFlow Dataset from numpy arrays.

    Args:
        data: Input data with shape (n_samples, n_channels, n_timepoints).
        labels: Labels with shape (n_samples,).
        batch_size: Batch size for the dataset.
        shuffle: Whether to shuffle the dataset.

    Returns:
        A TensorFlow Dataset.
    """
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(data))
    dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


def train_model(
    model: tf.keras.Model,
    train_dataset: tf.data.Dataset,
    val_dataset: tf.data.Dataset,
    num_epochs: int = 10,
    learning_rate: float = 0.001
) -> dict:
    """
    Train the model.

    Args:
        model: The Keras model to train.
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        num_epochs: Number of epochs to train.
        learning_rate: Learning rate for the optimizer.

    Returns:
        Dictionary containing training and validation metrics.
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=num_epochs,
        verbose=1
    )

    return history.history

def evaluate_model(model: tf.keras.Model, test_dataset: tf.data.Dataset) -> Tuple[float, float]:
    """
    Evaluate the model on a test set.

    Args:
        model: The Keras model to evaluate.
        test_dataset: Test dataset.

    Returns:
        test_loss: Average loss on the test set.
        test_accuracy: Accuracy on the test set.
    """
    test_loss, test_accuracy = model.evaluate(test_dataset, verbose=0)
    return test_loss, test_accuracy


def save_model(model: tf.keras.Model, filepath: str) -> None:
    """
    Save the model to a file.

    Args:
        model: The Keras model to save.
        filepath: Path to save the model.
    """
    model.save(filepath)
    logging.info(f'Model saved to {filepath}')

def load_model(filepath: str) -> tf.keras.Model:
    """
    Load the model from a file.

    Args:
        filepath: Path to load the model from.

    Returns:
        The loaded Keras model.
    """
    model = tf.keras.models.load_model(filepath)
    logging.info(f'Model loaded from {filepath}')
    return model



def visualize_metrics(metrics: dict) -> None:
    """
    Visualize training and validation metrics.

    Args:
        metrics: Dictionary containing training and validation metrics.
    """
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(metrics['loss'], label='Train Loss')
    plt.plot(metrics['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(metrics['accuracy'], label='Train Accuracy')
    plt.plot(metrics['val_accuracy'], label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()


def main():
    """
    Main function to train and evaluate the deep learning model.
    """
    from preprocess import load_data, slice_data
    from feature_extraction import extract_features

    # Load and preprocess data
    eeg_data = slice_data(load_data(dataname='deap'))
    eeg_data = extract_features(eeg_data)

    # Prepare datasets
    X = eeg_data.data  # Shape: (n_samples, n_channels, n_timepoints)
    y = (eeg_data.labels[:, 1]>5).astype(int) # Shape: (n_samples,)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_dataset = create_dataset(X_train, y_train, batch_size=32, shuffle=True)
    test_dataset = create_dataset(X_test, y_test, batch_size=32, shuffle=False)

    # Initialize model
    input_shape = X_train.shape[1:]  # (n_channels, n_timepoints)
    model = DeepLearningModel(input_shape=input_shape, num_classes=2)

    # Train model
    metrics = train_model(model, train_dataset, test_dataset, num_epochs=10, learning_rate=0.001)

    # Evaluate model
    test_loss, test_accuracy = evaluate_model(model, test_dataset)
    logging.info(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

    # Visualize metrics
    visualize_metrics(metrics)

    # Save model
    save_model(model, os.path.join(CACHE_DIR, 'deep_learning_model'))


if __name__ == '__main__':
    main()