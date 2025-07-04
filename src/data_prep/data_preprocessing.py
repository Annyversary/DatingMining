import pandas as pd
import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE
from scipy.stats.mstats import winsorize

"""
Data Preprocessing Utilities

These functions are designed to be executed within each fold of a cross-validation procedure,
which is considered best practice to prevent data leakage between training and validation sets.

This module maintains a consistent structure: it defines preprocessing functions that are
invoked during cross-validation, accompanied by corresponding unit tests located in
`unittests\data_prep`.
"""

def handle_missing_categorical(X_train, X_test, cast_to_category=True, fill_value='Unknown'):
    """
    Handle missing values in categorical columns by introducing a custom category (default: 'Unknown').

    Parameters
    ----------
    X_train : pd.DataFrame
        Training dataset containing categorical columns.
    X_test : pd.DataFrame
        Test dataset with the same structure.
    cast_to_category : bool, default=True
        Whether to convert object-type columns to 'category' dtype before processing.
    fill_value : str, default='Unknown'
        The value to fill in for missing entries.

    Returns
    -------
    X_train : pd.DataFrame
        Training dataset with missing categorical values filled.
    X_test : pd.DataFrame
        Test dataset with missing categorical values filled.
    """
    # Identify object/category columns
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns

    for col in cat_cols:
        # Optionally cast to category dtype for memory efficiency and proper category management
        if cast_to_category:
            if X_train[col].dtype != 'category':
                X_train[col] = X_train[col].astype('category')
            if X_test[col].dtype != 'category':
                X_test[col] = X_test[col].astype('category')

        if X_train[col].dtype == 'category':
            # Add 'Unknown' (or fill_value) to category list if not present
            new_categories = X_train[col].cat.categories.union([fill_value])
            X_train[col] = X_train[col].cat.set_categories(new_categories)
            X_test[col] = X_test[col].cat.set_categories(new_categories)

        # Fill missing values in both datasets
        X_train.loc[:, col] = X_train[col].fillna(fill_value)
        X_test.loc[:, col] = X_test[col].fillna(fill_value)

    return X_train, X_test

def handle_missing_numericals(X_train, X_test, strategy='median'):
    """
    Handle missing values in numerical columns using a specified strategy ('median' or 'mean').

    Parameters
    ----------
    X_train : pd.DataFrame
        Training dataset with numerical columns.
    X_test : pd.DataFrame
        Test dataset with the same numerical columns as X_train.
    strategy : str, default='median'
        The strategy to use for imputation. Options: 'median', 'mean'.

    Returns
    -------
    X_train_filled : pd.DataFrame
        X_train with missing values filled.
    X_test_filled : pd.DataFrame
        X_test with missing values filled using X_train's statistics.
    """
    # Identify numerical columns
    numerical_columns = X_train.select_dtypes(include=['float64', 'int64']).columns

    # Select imputation strategy
    if strategy == 'median':
        imputers = X_train[numerical_columns].median()
    elif strategy == 'mean':
        imputers = X_train[numerical_columns].mean()
    else:
        raise ValueError("strategy must be either 'median' or 'mean'")

    # Apply imputers
    X_train_filled = X_train.copy()
    X_test_filled = X_test.copy()

    X_train_filled[numerical_columns] = X_train_filled[numerical_columns].fillna(imputers)
    X_test_filled[numerical_columns] = X_test_filled[numerical_columns].fillna(imputers)

    return X_train_filled, X_test_filled


def handle_numeric_outliers(
    X_train,
    y_train,
    use_isolation_forest=True,
    use_lof=True,
    strategy="remove",  # or "winsorize"
    contamination_if=0.05,
    contamination_lof=0.05,
    random_state=42,
    n_neighbors=20,
    winsorize_limits=(0.01, 0.01)
):
    """
    Detect and optionally remove or winsorize numerical outliers using Isolation Forest and/or LOF.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series or pd.DataFrame
        Corresponding targets.
    use_isolation_forest : bool, default=True
        Whether to apply Isolation Forest.
    use_lof : bool, default=True
        Whether to apply Local Outlier Factor.
    strategy : str, default="remove"
        Strategy to handle outliers: 'remove' or 'winsorize'.
    contamination_if : float, default=0.05
        Expected proportion of outliers for Isolation Forest.
    contamination_lof : float, default=0.05
        Expected proportion of outliers for LOF.
    random_state : int, default=42
        Random state for reproducibility.
    n_neighbors : int, default=20
        Number of neighbors for LOF.
    winsorize_limits : tuple of floats, default=(0.01, 0.01)
        Lower and upper limits for winsorization (used if strategy='winsorize').

    Returns
    -------
    X_train_out : pd.DataFrame
        Cleaned or adjusted training features.
    y_train_out : pd.Series or pd.DataFrame
        Corresponding cleaned target values.
    """
    X_numeric = X_train.select_dtypes(include='number').copy()
    X_out = X_train.copy()
    y_out = y_train.copy()

    # Init outlier mask
    outlier_mask = pd.Series(False, index=X_train.index)

    # Isolation Forest
    if use_isolation_forest:
        iso = IsolationForest(contamination=contamination_if, random_state=random_state)
        mask_if = iso.fit_predict(X_numeric) == -1  # -1 = outlier
        outlier_mask |= mask_if

    # Local Outlier Factor
    if use_lof:
        lof = LocalOutlierFactor(contamination=contamination_lof, n_neighbors=n_neighbors)
        mask_lof = lof.fit_predict(X_numeric) == -1  # -1 = outlier
        outlier_mask |= mask_lof

    if strategy == "remove":
        # Drop outliers
        X_out = X_out.loc[~outlier_mask].copy()
        y_out = y_out.loc[~outlier_mask].copy()

    elif strategy == "winsorize":
        for col in X_numeric.columns:
            col_data = X_out[col]
            # Winsorize only the outlier rows in this column
            non_outliers = col_data[~outlier_mask]
            lower = np.percentile(non_outliers, winsorize_limits[0] * 100)
            upper = np.percentile(non_outliers, 100 - winsorize_limits[1] * 100)
            col_data[outlier_mask] = col_data[outlier_mask].clip(lower=lower, upper=upper)
            X_out[col] = col_data
    else:
        raise ValueError("strategy must be either 'remove' or 'winsorize'.")

    return X_out, y_out


def encode_categorical_features(
    X_train,
    X_test,
    drop='first',
    handle_unknown='ignore',
    sparse_output=False
):
    """
    Perform one-hot encoding on categorical features using scikit-learn's OneHotEncoder.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training dataset with mixed types.
    X_test : pd.DataFrame
        Test dataset with same structure as X_train.
    drop : str or None, default='first'
        Whether to drop the first level to avoid multicollinearity. Set to None to keep all.
    handle_unknown : str, default='ignore'
        How to handle unknown categories in test data. 'ignore' prevents errors.
    sparse_output : bool, default=False
        Whether to return a sparse matrix or dense output (False = dense DataFrame).

    Returns
    -------
    X_train_encoded : pd.DataFrame
        Transformed training set with one-hot encoded categorical features.
    X_test_encoded : pd.DataFrame
        Transformed test set with one-hot encoded categorical features.
    """
    # Identify categorical columns
    categorical_columns = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    if not categorical_columns:
        # Nothing to encode — return input unchanged
        return X_train.copy(), X_test.copy()

    # Initialize encoder
    encoder = OneHotEncoder(
        drop=drop,
        handle_unknown=handle_unknown,
        sparse_output=sparse_output
    )

    # Fit encoder on training data
    encoder.fit(X_train[categorical_columns])

    # Transform train and test data
    X_train_encoded = encoder.transform(X_train[categorical_columns])
    X_test_encoded = encoder.transform(X_test[categorical_columns])

    # Convert to DataFrame with correct column names
    encoded_columns = encoder.get_feature_names_out(categorical_columns)
    X_train_encoded = pd.DataFrame(X_train_encoded, columns=encoded_columns, index=X_train.index)
    X_test_encoded = pd.DataFrame(X_test_encoded, columns=encoded_columns, index=X_test.index)

    # Drop original categorical columns
    X_train_dropped = X_train.drop(columns=categorical_columns)
    X_test_dropped = X_test.drop(columns=categorical_columns)

    # Concatenate encoded with numeric
    X_train_final = pd.concat([X_train_dropped, X_train_encoded], axis=1)
    X_test_final = pd.concat([X_test_dropped, X_test_encoded], axis=1)

    return X_train_final, X_test_final


def scale_and_transform(X_train, X_test, scaler, return_scaler=False):
    """
    Fit the provided scaler on training data and apply the transformation to both train and test sets.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training dataset.
    X_test : pd.DataFrame
        Test dataset.
    scaler : scikit-learn compatible scaler
        Must implement fit(X), transform(X), and optionally fit_transform(X).
    return_scaler : bool, default=False
        If True, return the fitted scaler object.

    Returns
    -------
    X_train_scaled : pd.DataFrame
        Scaled training data.
    X_test_scaled : pd.DataFrame
        Scaled test data.
    scaler (optional) : fitted scaler object
        Returned only if return_scaler=True.
    """
    # Check input types
    if not isinstance(X_train, pd.DataFrame) or not isinstance(X_test, pd.DataFrame):
        raise TypeError("X_train and X_test must be pandas DataFrames.")

    # Fit scaler on train set and transform both sets
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Preserve column names and index
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

    if return_scaler:
        return X_train_scaled, X_test_scaled, scaler
    return X_train_scaled, X_test_scaled

def apply_smote(X_train, y_train, random_state=42, sampling_strategy='auto', return_as_dataframe=True):
    """
    Apply SMOTE to balance the class distribution in the training data.

    Parameters
    ----------
    X_train : pd.DataFrame or np.ndarray
        Training feature matrix.
    y_train : pd.Series or np.ndarray
        Training labels.
    random_state : int, default=42
        Random seed for reproducibility.
    sampling_strategy : str, float, dict or callable, default='auto'
        Sampling strategy passed to SMOTE. See imbalanced-learn docs for options.
    return_as_dataframe : bool, default=True
        If True, returns X_resampled and y_resampled as pandas DataFrame/Series
        (with preserved column names and index starting at 0).

    Returns
    -------
    X_resampled : pd.DataFrame or np.ndarray
        Resampled training features.
    y_resampled : pd.Series or np.ndarray
        Resampled training labels.
    """
    smote = SMOTE(random_state=random_state, sampling_strategy=sampling_strategy)

    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    if return_as_dataframe:
        # Restore DataFrame structure if original was DataFrame
        if isinstance(X_train, pd.DataFrame):
            X_resampled = pd.DataFrame(X_resampled, columns=X_train.columns)
        if isinstance(y_train, pd.Series):
            y_resampled = pd.Series(y_resampled, name=y_train.name)

    return X_resampled, y_resampled
