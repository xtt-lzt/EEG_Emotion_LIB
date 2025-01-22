import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from typing import Optional, Tuple
import logging
from eeg_data import EEGData
import os

CACHE_DIR = 'temp'

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def prepare_data(eeg_data) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare features and labels for machine learning.

    Args:
        eeg_data: EEGData object containing features and labels.

    Returns:
        X: Feature matrix with shape (n_samples, n_features).
        y: Label vector with shape (n_samples,).
    """
    logging.info('Preparing data for machine learning...')
    
    # Combine all features into a single feature matrix
    feature_list = []
    for feature_type, feature_data in eeg_data.features.items():
        # Reshape feature data from (n_samples, n_channels, n_bands) to (n_samples, n_channels * n_bands)
        n_samples, n_channels, n_bands = feature_data.shape
        feature_reshaped = feature_data.reshape(n_samples, -1)  # Shape: (n_samples, n_channels * n_bands)
        feature_list.append(feature_reshaped)
    
    # Stack features along the feature dimension
    X = np.hstack(feature_list)  # Shape: (n_samples, n_channels * n_bands * n_feature_types)
    
    # Flatten labels to match the number of samples
    y = eeg_data.labels
    # y = (eeg_data.labels[:, 1]>5).astype(int)  # Shape: (n_samples,)
    
    logging.info(f'Feature matrix shape: {X.shape}')
    logging.info(f'Label vector shape: {y.shape}')
    return X, y

def train_and_evaluate(X: np.ndarray, y: np.ndarray, model_type: str = 'svm', test_size: float = 0.2, random_state: int = 42, cache_path: Optional[str] = CACHE_DIR):
    """
    Train and evaluate a machine learning model.

    Args:
        X: Feature matrix with shape (n_samples, n_features).
        y: Label vector with shape (n_samples,).
        model_type: Type of model to train ('svm' or 'random_forest').
        test_size: Proportion of the dataset to include in the test split.
        random_state: Random seed for reproducibility.

    Returns:
        model: Trained machine learning model.
        accuracy: Accuracy score on the test set.
    """
    logging.info(f'Training and evaluating {model_type} model...')
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Standardize features (important for SVM)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Initialize the model
    if model_type == 'svm':
        from sklearn.svm import LinearSVC
        model = LinearSVC()
    elif model_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, random_state=random_state, verbose=True)
    elif model_type == 'logistic_regression':
        model = LogisticRegression(random_state=random_state, verbose=True)
    elif model_type == 'xgboost':
        import xgboost as xgb
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=random_state, verbosity=2)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f'Test accuracy: {accuracy:.4f}')
    logging.info('Classification report:\n' + classification_report(y_test, y_pred))
    
    import joblib
    joblib.dump(model, os.path.join(cache_path, f'trained_model: {model_type}.pkl'))

    return model, accuracy

if __name__ == '__main__':
    # Example usage
    from feature_extraction import extract_features
    from preprocess import load_data, slice_data
    import os

    # Load data
    eeg_data = slice_data(load_data(dataname='deap'))

    # Extract features
    eeg_data_with_features = extract_features(
        eeg_data,
        feature_list=['psd', 'de'],  # Extract both PSD and DE
        # cache_file=os.path.join(CACHE_DIR, 'eeg_features.pkl')
    )

    # Prepare data for machine learning
    X, y = prepare_data(eeg_data_with_features)
    y = (y[:, 1]>5).astype(int)  # Use valence as binary label

    # Train and evaluate a model
    model, accuracy = train_and_evaluate(X, y, model_type='svm')