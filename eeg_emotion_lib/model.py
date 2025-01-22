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
    

import tensorflow as tf
from tensorflow.keras.layers import GRU, BatchNormalization, Dropout, Permute, Reshape, Conv1D, Flatten, Dense

class MyRNNModel(tf.keras.Model):
    def __init__(self, input_shape, num_classes):
        super(MyRNNModel, self).__init__()
        self.reshape_1 = Reshape((input_shape[0], input_shape[1]))  # Reshape input to (timesteps, features)

        # 一层 GRU
        self.gru_1 = GRU(128, return_sequences=True, activation='relu')
        self.bn_1 = BatchNormalization()
        self.dropout_1 = Dropout(0.3)

        # Reshape for Conv2D
        self.reshape_2 = Reshape((input_shape[0], 128, 1))  # Convert to 4D for Conv2D

        # 一层 Conv2D
        self.conv2d_1 = Conv2D(64, kernel_size=(input_shape[0], 1), strides=(1, 1), padding='valid', activation='relu')
        self.bn_2 = BatchNormalization()
        self.dropout_2 = Dropout(0.3)

        # Flatten 层
        self.flatten = Flatten()

        # 全连接层
        self.dense_1 = Dense(128, activation='relu')
        self.bn_3 = BatchNormalization()
        self.dropout_3 = Dropout(0.5)

        # 输出层
        self.dense_output = Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.reshape_1(inputs) 

        # 一层 GRU
        x = self.gru_1(x)
        x = self.bn_1(x)
        x = self.dropout_1(x)

        # Reshape for Conv2D
        x = self.reshape_2(x)

        # 一层 Conv2D
        x = self.conv2d_1(x)
        x = self.bn_2(x)
        x = self.dropout_2(x)

        # Flatten and Dense layers
        x = self.flatten(x)
        x = self.dense_1(x)
        x = self.bn_3(x)
        x = self.dropout_3(x)

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
    
class CNN3DModel(tf.keras.Model):
    def __init__(self, input_shape, num_classes=10, dropout_rate=0.3):
        super(CNN3DModel, self).__init__()
        self.conv1 = layers.Conv3D(32, (3, 3, 4), padding='same', activation='relu')
        self.bn1 = layers.BatchNormalization()
        self.pool1 = layers.MaxPooling3D((1, 1, 2))

        self.conv2 = layers.Conv3D(64, (3, 3, 4), padding='same', activation='relu')
        self.bn2 = layers.BatchNormalization()
        self.pool2 = layers.MaxPooling3D((1, 1, 2))

        self.conv3 = layers.Conv3D(128, (3, 3, 4), padding='same', activation='relu')
        self.bn3 = layers.BatchNormalization()
        self.pool3 = layers.MaxPooling3D((1, 1, 2))

        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.dropout = layers.Dropout(dropout_rate)
        self.output_layer = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = self.pool3(x)

        x = self.flatten(x)
        x = self.dense1(x)
        if training:
            x = self.dropout(x)
        return self.output_layer(x)




def visualize_metrics(metrics: dict, model_type) -> None:
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

    plt.savefig(f'{model_type}_metrics.png')

def train_and_evaluate_deep_model(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = 'EEGNet',
    model: Optional[tf.keras.Model] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    num_epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    cache_path: Optional[str] = CACHE_DIR
) -> Tuple[tf.keras.Model, float]:
    """
    Train and evaluate a deep learning model for EEG emotion recognition.

    Args:
        X: Feature matrix with shape (n_samples, n_channels, n_timepoints).
        y: Label vector with shape (n_samples,).
        model_type: Type of model to train ('EEGNet' or 'MyDEAPModel').
        test_size: Proportion of the dataset to include in the test split.
        random_state: Random seed for reproducibility.
        num_epochs: Number of epochs to train.
        batch_size: Batch size for training.
        learning_rate: Learning rate for the optimizer.
        cache_path: Directory to cache the trained model.

    Returns:
        model: Trained deep learning model.
        test_accuracy: Accuracy score on the test set.
    """
    logging.info(f'Training and evaluating {model_type} model...')

    # Split the data into training and testing sets
    X = np.expand_dims(X, axis=-1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(buffer_size=len(X_train)).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)

    # Initialize the model
    input_shape = X_train.shape[1:]  # (n_channels, n_timepoints, 1)
    num_classes = len(np.unique(y))
    if model is None:
        if model_type == 'EEGNet':
            model = EEGNet(input_shape=input_shape, num_classes=num_classes)
        elif model_type == 'MyDEAPModel':
            model = MyDEAPModel(input_shape=input_shape, num_classes=num_classes)
        elif model_type == 'CNN3DModel':
            model = CNN3DModel(input_shape=input_shape, num_classes=num_classes)
        elif model_type == 'MyRNNModel':
            model = MyRNNModel(input_shape=input_shape, num_classes=num_classes)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train the model
    history = model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=num_epochs,
        verbose=1
    )

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(test_dataset, verbose=0)
    logging.info(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

    # Generate classification report
    y_pred = np.argmax(model.predict(test_dataset), axis=1)
    logging.info('Classification report:\n' + classification_report(y_test, y_pred))

    # Save the model
    model_save_path = os.path.join(cache_path, f'{model_type}_model')
    model.save(model_save_path, save_format="tf")
    logging.info(f'Model saved to {model_save_path}')
    visualize_metrics(history.history, model_type)

    return model, test_accuracy

def main():
    """
    Main function to train and evaluate the deep learning model.
    """
    from preprocess import load_data, slice_data
    from feature_extraction import extract_features

    

    

    # Load and preprocess data
    eeg_data = slice_data(load_data(dataname='deap'))
    # eeg_data = extract_features(eeg_data)

    # Prepare datasets
    X = eeg_data.data  # Shape: (n_samples, n_channels, n_timepoints)
    y = (eeg_data.labels[:, 1]>5).astype(int) # Shape: (n_samples,)
    model = EEGNet(input_shape=X.shape[1:], num_classes=2)
    model, test_accuracy = train_and_evaluate_deep_model(X, y)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # train_dataset = create_dataset(X_train, y_train, batch_size=32, shuffle=True)
    # test_dataset = create_dataset(X_test, y_test, batch_size=32, shuffle=False)

    # # Initialize model
    # input_shape = X_train.shape[1:]  # (n_channels, n_timepoints)
    # # model = MyDEAPModel(input_shape=input_shape, num_classes=2)

    

    # # Train model
    # metrics = train_model(model, train_dataset, test_dataset, num_epochs=10, learning_rate=0.001)

    # # Evaluate model
    # test_loss, test_accuracy = evaluate_model(model, test_dataset)
    # logging.info(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

    # # Visualize metrics
    # visualize_metrics(metrics)

    # Save model
    # save_model(model, os.path.join(CACHE_DIR, 'deep_learning_model'))


if __name__ == '__main__':
    main()