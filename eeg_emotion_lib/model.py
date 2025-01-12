import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Conv2D, Dense, BatchNormalization, Dropout, Conv1D, Permute, Reshape, Flatten, DepthwiseConv2D, Activation, AveragePooling2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import logging
import matplotlib.pyplot as plt
from typing import Optional, Tuple
from abc import ABC, abstractmethod

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
class MyDEAPModel(tf.keras.Model):
    def __init__(self, input_shape, num_classes):
        super(MyDEAPModel, self).__init__()
        # 第一层 Conv2D
        self.conv2d_1 = Conv2D(128, kernel_size=(1, 128), strides=(1, 1), padding='same', activation='relu', input_shape=input_shape)
        self.bn_1 = BatchNormalization()
        self.dropout_1 = Dropout(0.3)

        # 第二层 Conv2D
        self.conv2d_2 = Conv2D(128, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')
        self.bn_2 = BatchNormalization()
        self.dropout_2 = Dropout(0.3)

        # 第三层 Conv2D
        self.conv2d_3 = Conv2D(64, kernel_size=(input_shape[0], 1), strides=(1, 1), padding='valid', activation='relu')
        self.bn_3 = BatchNormalization()
        self.dropout_3 = Dropout(0.3)

        # Permute 和 Reshape 层
        self.permute = Permute((2, 1, 3))
        self.reshape = Reshape((input_shape[1], -1))

        # 第一层 Conv1D
        self.conv1d_1 = Conv1D(64, kernel_size=3, strides=1, padding='same', activation='relu')
        self.bn_4 = BatchNormalization()
        self.dropout_4 = Dropout(0.3)

        # 第二层 Conv1D
        self.conv1d_2 = Conv1D(64, kernel_size=3, strides=1, padding='same', activation='relu')
        self.bn_5 = BatchNormalization()
        self.dropout_5 = Dropout(0.3)

        # Flatten 层
        self.flatten = Flatten()

        # 全连接层
        self.dense_1 = Dense(128, activation='relu')
        self.bn_6 = BatchNormalization()
        self.dropout_6 = Dropout(0.5)

        self.dense_2 = Dense(64, activation='relu')
        self.bn_7 = BatchNormalization()
        self.dropout_7 = Dropout(0.5)

        # 输出层
        self.dense_output = Dense(num_classes, activation='softmax')

    def call(self, inputs):
        # 前向传播过程
        x = self.conv2d_1(inputs)
        x = self.bn_1(x)
        x = self.dropout_1(x)

        x = self.conv2d_2(x)
        x = self.bn_2(x)
        x = self.dropout_2(x)

        x = self.conv2d_3(x)
        x = self.bn_3(x)
        x = self.dropout_3(x)

        x = self.permute(x)
        x = self.reshape(x)

        x = self.conv1d_1(x)
        x = self.bn_4(x)
        x = self.dropout_4(x)

        x = self.conv1d_2(x)
        x = self.bn_5(x)
        x = self.dropout_5(x)

        x = self.flatten(x)

        x = self.dense_1(x)
        x = self.bn_6(x)
        x = self.dropout_6(x)

        x = self.dense_2(x)
        x = self.bn_7(x)
        x = self.dropout_7(x)

        return self.dense_output(x)

class EEGNet(tf.keras.Model):
    def __init__(self, input_shape, num_classes, F1=8, D=2, F2=16, kernel_length=64, dropout_rate=0.5):
        """
        EEGNet model for EEG signal classification.

        Args:
            input_shape: Shape of the input data (e.g., (n_channels, n_timepoints, 1)).
            num_classes: Number of output classes.
            F1: Number of temporal filters.
            D: Depth multiplier for depthwise convolution.
            F2: Number of pointwise filters.
            kernel_length: Length of the temporal convolution kernel.
            dropout_rate: Dropout rate.
        """
        super(EEGNet, self).__init__()

        # Block 1: Temporal Convolution
        self.conv2d_1 = Conv2D(F1, (1, kernel_length), padding='same', use_bias=False, input_shape=input_shape)
        self.bn_1 = BatchNormalization()
        self.depthwise_conv2d = DepthwiseConv2D((input_shape[0], 1), depth_multiplier=D, use_bias=False, padding='valid')
        self.bn_2 = BatchNormalization()
        self.activation_1 = Activation('elu')
        self.avg_pool_1 = AveragePooling2D((1, 4))
        self.dropout_1 = Dropout(dropout_rate)

        # Block 2: Spatial Convolution
        self.conv2d_2 = Conv2D(F2, (1, 16), padding='same', use_bias=False)
        self.bn_3 = BatchNormalization()
        self.activation_2 = Activation('elu')
        self.avg_pool_2 = AveragePooling2D((1, 8))
        self.dropout_2 = Dropout(dropout_rate)

        # Flatten and Dense layers
        self.flatten = Flatten()
        self.dense = Dense(num_classes, activation='softmax')

    def call(self, inputs, training=False):
        # Block 1: Temporal Convolution
        x = self.conv2d_1(inputs)
        x = self.bn_1(x, training=training)
        x = self.depthwise_conv2d(x)
        x = self.bn_2(x, training=training)
        x = self.activation_1(x)
        x = self.avg_pool_1(x)
        x = self.dropout_1(x, training=training)

        # Block 2: Spatial Convolution
        x = self.conv2d_2(x)
        x = self.bn_3(x, training=training)
        x = self.activation_2(x)
        x = self.avg_pool_2(x)
        x = self.dropout_2(x, training=training)

        # Flatten and Dense layers
        x = self.flatten(x)
        x = self.dense(x)

        return x
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
    data = np.expand_dims(data, axis=-1)  # Add a single channel dimension
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
    # model = MyDEAPModel(input_shape=input_shape, num_classes=2)

    model = EEGNet(input_shape=input_shape, num_classes=2)

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