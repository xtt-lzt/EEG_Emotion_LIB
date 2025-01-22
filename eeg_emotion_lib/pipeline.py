import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# 导入自定义模块
from eeg_data import EEGData
from preprocess import load_data, slice_data, map_data_to_matrix
from feature_extraction import extract_features
from classification import prepare_data, train_and_evaluate
from model import train_and_evaluate_deep_model
from visualization import plot_confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers

def DEAP_main(data_path):
    # Load data
    eeg_data = slice_data(load_data(dataname='deap'))

    # Extract features
    eeg_data_with_features = extract_features(
        eeg_data,
        feature_list=['psd', 'de'],  # Extract both PSD and DE
    )

    # Prepare data for machine learning
    X, y = prepare_data(eeg_data_with_features)
    y_1 = (y[:, 0]>5).astype(int)  # Use valence as binary label
    y_2 = (y[:, 1]>5).astype(int)

    # Train and evaluate a model
    # ml_model, accuracy = train_and_evaluate(X, y, model_type='svm')
    # train_and_evaluate(X, y, model_type='logistic_regression')
    # train_and_evaluate(X, y, model_type='random_forest')
    # train_and_evaluate(X, y, model_type='xgboost')

    data = eeg_data.data
    # train_and_evaluate_deep_model(data, y_2, model_type='EEGNet', test_size=0.2, random_state=42, num_epochs=50, batch_size=32, learning_rate=0.001)
    # data_3d = map_data_to_matrix(eeg_data)

    # model_3d, accuracy = train_and_evaluate_deep_model(
    #     data_3d, y, model_type='CNN3DModel', test_size=0.2, random_state=42, num_epochs=20, batch_size=32, learning_rate=0.001
    # )
    # train_and_evaluate_deep_model(data, y, model_type='MyDEAPModel', test_size=0.2, random_state=42, num_epochs=50, batch_size=32, learning_rate=0.001)
    # train_and_evaluate_deep_model(data, y_1, model_type='MyRNNModel', test_size=0.2, random_state=42, num_epochs=10, batch_size=32, learning_rate=0.001)
    # train_and_evaluate_deep_model(data, y_2, model_type='MyRNNModel', test_size=0.2, random_state=42, num_epochs=10, batch_size=32, learning_rate=0.001)

    # train_and_evaluate_deep_model(data, y, model_type='MyDEAPModel', test_size=0.2, random_state=42, num_epochs=50, batch_size=32, learning_rate=0.001)
    # train_and_evaluate_deep_model(data, y, model_type='EEGNet', test_size=0.2, random_state=42, num_epochs=50, batch_size=32, learning_rate=0.001)

    data_3d = map_data_to_matrix(eeg_data)
    data_3d = (data_3d - np.mean(data_3d, axis=(0, 1, 2, 3), keepdims=True)) / np.std(data_3d, axis=(0, 1, 2, 3), keepdims=True)


    model_3d, accuracy = train_and_evaluate_deep_model(
        data_3d, y_1, model_type='CNN3DModel', test_size=0.2, random_state=42, num_epochs=10, batch_size=240, learning_rate=0.001)




if __name__ == "__main__":
    data_path = "path_to_your_data"
    DEAP_main(data_path)
