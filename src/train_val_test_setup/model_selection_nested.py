from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from src.data_prep.data_preprocessing import (
    handle_missing_categorical, 
    handle_missing_numericals,
    handle_numeric_outliers,
    encode_categorical_features,
    scale_and_transform,
    apply_smote)
from src.train_val_test_setup.setup import (
    initialize_results_storage,
    evaluate_fold
)

def perform_nested_cv(
    features, target, model, param_distributions, 
    scaler=StandardScaler(), contamination=0.1, random_state=42,
    outer_splits=5, inner_splits=3, n_iter=20,
    bool_handleMissingValues=True, bool_scaling=True, bool_upsampling=True
):
    """
    Perform nested cross-validation with hyperparameter tuning using RandomizedSearchCV.

    This function conducts a nested cross-validation procedure to estimate the
    generalization performance of a model while tuning hyperparameters within
    the inner folds. It handles preprocessing steps inside each outer fold to
    avoid data leakage, including missing value imputation, outlier removal,
    categorical encoding, scaling, and optional SMOTE for class imbalance.

    Parameters:
    ----------
    features : pandas.DataFrame
        Feature dataset (predictors).
    target : pandas.Series or numpy.ndarray
        Target variable.
    model : estimator object
        Instantiated sklearn-compatible model to be tuned and evaluated.
    param_distributions : dict
        Dictionary with parameters names (string) as keys and distributions
        or lists of parameters to try.
    scaler : scaler object, default=StandardScaler()
        Scaler instance to apply to features.
    contamination : float, default=0.1
        Expected proportion of outliers for outlier detection.
    random_state : int, default=42
        Seed for reproducibility.
    outer_splits : int, default=5
        Number of splits for the outer cross-validation.
    inner_splits : int, default=3
        Number of splits for the inner cross-validation (hyperparameter tuning).
    n_iter : int, default=20
        Number of parameter settings that are sampled in RandomizedSearchCV.
    bool_handleMissingValues : bool, default=True
        Whether to apply missing value handling.
    bool_scaling : bool, default=True
        Whether to apply feature scaling.
    bool_upsampling : bool, default=True
        Whether to apply SMOTE for class imbalance.

    Returns:
    -------
    nested_scores : dict
        Dictionary containing lists of evaluation metrics (accuracy, precision, recall, f1)
        aggregated over the outer folds.
    """
    outer_cv = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=random_state)
    inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=random_state)

    nested_scores = initialize_results_storage(features)

    for outer_train_idx, outer_test_idx in outer_cv.split(features, target):
        X_train, X_test = features.iloc[outer_train_idx].copy(), features.iloc[outer_test_idx].copy()
        y_train, y_test = target[outer_train_idx], target[outer_test_idx]

        # Preprocessing (only on train and test split)
        if bool_handleMissingValues:
            X_train, X_test = handle_missing_categorical(X_train, X_test)
            X_train, X_test = handle_missing_numericals(X_train, X_test)
            X_train, y_train = handle_numeric_outliers(X_train, y_train, contamination=contamination, random_state=random_state)

        X_train, X_test = encode_categorical_features(X_train, X_test)

        if bool_scaling:
            X_train, X_test = scale_and_transform(X_train, X_test, scaler)

        if bool_upsampling:
            X_train, y_train = apply_smote(X_train, y_train, random_state=random_state)

        # Inner loop: RandomizedSearchCV for hyperparameter tuning on outer train set
        randomized_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=inner_cv,
            scoring='f1',  # adjust metric if desired
            random_state=random_state,
            n_jobs=-1,
            verbose=1
        )
        randomized_search.fit(X_train, y_train)

        best_model = randomized_search.best_estimator_

        # Evaluate best model on outer test set
        nested_scores = evaluate_fold(X_test, y_test, nested_scores, best_model)

    return nested_scores
