import pandas as pd
import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE

"""
Data Preprocessing Utilities

These functions are designed to be executed within each fold of a cross-validation procedure,
which is considered best practice to prevent data leakage between training and validation sets.

This module maintains a consistent structure: it defines preprocessing functions that are
invoked during cross-validation, accompanied by corresponding unit tests located in
`unittests/Unittests_Data_Preprocessing.ipynb`.
"""

def handle_missing_categorical(X_train, X_test):
    """
    Handle missing values in categorical columns by introducing an 'Unknown' category.

    This is done based on the assumption that missingness is informative — for example, some participants 
    might have skipped certain questions in a questionnaire due to discomfort, making 
    the absence of a response itself meaningful.

    Parameters:
    ----------
    X_train : pandas.DataFrame
        The training dataset containing categorical columns with potential missing values.
    X_test : pandas.DataFrame
        The test dataset containing the same categorical columns as X_train.

    Returns:
    -------
    X_train : pandas.DataFrame
        The training dataset with missing categorical values filled.
    X_test : pandas.DataFrame
        The test dataset with missing categorical values filled.
    """
    # Identify categorical columns in X_train
    categorical_columns = X_train.select_dtypes(include=['object', 'category']).columns

    # Fill missing values in categorical columns with 'Unknown'
    for col in categorical_columns:
        if X_train[col].dtype.name == 'category':
            # Add 'Unknown' as a new category
            new_categories = X_train[col].cat.categories.union(['Unknown'])
            X_train[col] = X_train[col].cat.set_categories(new_categories)
            X_test[col] = X_test[col].cat.set_categories(new_categories)

        # Fill missing values in both datasets
        X_train[col] = X_train[col].fillna('Unknown')
        X_test[col] = X_test[col].fillna('Unknown')

    return X_train, X_test


def handle_missing_numericals(X_train, X_test):
    """
    Handle missing values in numerical columns using the median of the training set.

    This function (like any other function in this class) is intended to be called during cross-validation 
    to ensure that missing values in categorical features are properly handled. 
    Now, the median for each fold is computed and used, rather than the median of the entire dataset.

    Parameters:
    ----------
    X_train : pandas.DataFrame
        The training dataset containing numerical columns with potential missing values.
    X_test : pandas.DataFrame
        The test dataset with the same numerical columns as X_train.

    Returns:
    -------
    X_train : pandas.DataFrame
        The training dataset with missing numerical values filled.
    X_test : pandas.DataFrame
        The test dataset with missing numerical values filled.
    """
    # Identify numerical columns
    numerical_columns = X_train.select_dtypes(include=['float64', 'int64']).columns

    # Fill missing values in the numerical columns with the median of X_train
    for col in numerical_columns:
        median = X_train[col].median()  # Get the median from X_train
        X_train[col] = X_train[col].fillna(median)  # Fill missing values in X_train
        X_test[col] = X_test[col].fillna(median)  # Fill missing values in X_test using the same median

    return X_train, X_test


def handle_numeric_outliers(X_train, y_train, contamination=0.05, random_state=42):
    """
    Remove numerical outliers from the training dataset using Isolation Forest and Local Outlier Factor (LOF).

    The `handle_numeric_outliers` function removes numerical outliers from the training dataset using two 
    unsupervised anomaly detection techniques: **Isolation Forest** and **Local Outlier Factor (LOF)**. 
    In this project, we found that combining both methods yields good results by first applying Isolation 
    Forest to identify and remove broad anomalies, followed by LOF to detect more local irregularities.

    We chose not to search for outliers in categorical features, as the preprocessing step already corrected 
    all typos, and all remaining categories were predefined. If categories are clean and predefined, then a 
    rarely occurring category should not be considered an outlier.

    Parameters:
    ----------
    X_train : pandas.DataFrame
        The training feature set.
    y_train : pandas.Series or pandas.DataFrame
        The corresponding target values for training.
    contamination : float, default=0.05
        The expected proportion of outliers in the data. This value is used by both Isolation Forest and LOF.
    random_state : int, default=42
        Random seed for reproducibility in Isolation Forest.

    Returns:
    -------
    X_train : pandas.DataFrame
        Filtered training feature set with outliers removed.
    y_train : pandas.Series or pandas.DataFrame
        Filtered target values corresponding to the cleaned feature set.
    """
    # Select only numeric features from X_train
    X_numeric = X_train.select_dtypes(include=['number'])

    # Apply Isolation Forest on numeric features
    isolation_forest = IsolationForest(random_state=random_state, contamination=contamination)
    is_inlier_if = isolation_forest.fit_predict(X_numeric) == 1

    # Filter all data accordingly
    X_train, y_train = X_train[is_inlier_if], y_train[is_inlier_if]
    X_numeric = X_numeric[is_inlier_if]

    # Apply Local Outlier Factor on the filtered numeric features
    lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
    is_inlier_lof = lof.fit_predict(X_numeric) == 1

    # Final filtering based on LOF results
    X_train, y_train = X_train[is_inlier_lof], y_train[is_inlier_lof]

    return X_train, y_train


def encode_categorical_features(X_train, X_test):
    """
    Perform one-hot encoding on categorical features.

    Parameters:
    ----------
    X_train : pandas.DataFrame
        The training dataset containing both categorical and numerical features.
    X_test : pandas.DataFrame
        The test dataset with the same structure as X_train.

    Returns:
    -------
    X_train : pandas.DataFrame
        The training dataset with categorical features one-hot encoded and concatenated with numerical features.
    X_test : pandas.DataFrame
        The test dataset with categorical features one-hot encoded and concatenated with numerical features.
    """
    # Automatically identify categorical columns (columns with 'object' or 'category' dtype)
    categorical_columns = X_train.select_dtypes(include=['object', 'category']).columns

    # Initialize the OneHotEncoder with drop='first' to avoid a dummy variable trap
    encoder = OneHotEncoder(drop='first', sparse_output=False)

    # Apply encoder to X_train
    encoded_train_features = encoder.fit_transform(X_train[categorical_columns])
    encoded_train_df = pd.DataFrame(encoded_train_features, columns=encoder.get_feature_names_out(categorical_columns))

    # Apply encoder to X_test
    encoded_test_features = encoder.transform(X_test[categorical_columns])
    encoded_test_df = pd.DataFrame(encoded_test_features, columns=encoder.get_feature_names_out(categorical_columns))

    # Drop the original categorical columns and reset indices to ensure alignment
    X_train = X_train.drop(columns=categorical_columns).reset_index(drop=True)
    X_test = X_test.drop(columns=categorical_columns).reset_index(drop=True)

    # Concatenate the encoded features to the DataFrames
    X_train = pd.concat([X_train, encoded_train_df], axis=1)
    X_test = pd.concat([X_test, encoded_test_df], axis=1)

    return X_train, X_test


def scale_and_transform(X_train, X_test, scaler):
    """
    Scale and transform training and testing datasets using the provided scaler.

    Parameters:
    ----------
    X_train : pandas.DataFrame
        The training dataset to be scaled.
    X_test : pandas.DataFrame
        The test dataset to be scaled using the fitted scaler.
    scaler : scaler object
        An instantiated scaler object from scikit-learn (or compatible), e.g., StandardScaler.

    Returns:
    -------
    X_train_scaled : pandas.DataFrame
        The scaled training dataset with original column names.
    X_test_scaled : pandas.DataFrame
        The scaled test dataset with original column names.
    """
    # Fit the scaler on the training data and transform it
    X_train_scaled = scaler.fit_transform(X_train)

    # Transform the test data using the already fitted scaler
    X_test_scaled = scaler.transform(X_test)

    # Convert the scaled NumPy arrays back into DataFrames to retain column names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    return X_train_scaled, X_test_scaled


def apply_smote(X_train, y_train, random_state=42):
    """
    Handle class imbalance by applying SMOTE (Synthetic Minority Over-sampling Technique) to generate synthetic minority class samples.

    Parameters:
    ----------
    X_train : pandas.DataFrame or numpy.ndarray
        The training feature set.
    y_train : pandas.Series or numpy.ndarray
        The corresponding target labels.
    random_state : int, default=42
        Random seed for reproducibility of synthetic sample generation.

    Returns:
    -------
    X_train_resampled : numpy.ndarray or pandas.DataFrame
        The training features after applying SMOTE, including synthetic minority samples.
    y_train_resampled : numpy.ndarray or pandas.Series
        The target labels corresponding to the resampled training set.
    """
    smote = SMOTE(random_state=random_state)
    
    # Fit the SMOTE on the training data and generate synthetic samples
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train) 
    return X_train_resampled, y_train_resampled
