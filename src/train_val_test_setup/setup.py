from sklearn.model_selection import StratifiedKFold
from typing import Tuple, List, Dict


def create_cross_validation_splits(
    outer_splits: int = 5,
    inner_splits: int = 3,
    random_state: int = 42,
    shuffle: bool = True
) -> Tuple[StratifiedKFold, StratifiedKFold]:
    """
    Create StratifiedKFold cross-validation splitters for nested CV.

    Parameters:
    ----------
    outer_splits : int, default=5
        Number of splits for the outer cross-validation loop.
    inner_splits : int, default=3
        Number of splits for the inner cross-validation loop (hyperparameter tuning).
    random_state : int, default=42
        Seed for reproducibility of the splits.
    shuffle : bool, default=True
        Whether to shuffle data before splitting.

    Returns:
    -------
    outer_cv : StratifiedKFold
        Outer cross-validation splitter.
    inner_cv : StratifiedKFold
        Inner cross-validation splitter.
    """
    if outer_splits < 2 or inner_splits < 2:
        raise ValueError("Number of splits must be at least 2 for both outer and inner CV.")

    outer_cv = StratifiedKFold(n_splits=outer_splits, shuffle=shuffle, random_state=random_state)
    inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=shuffle, random_state=random_state)
    return outer_cv, inner_cv


def initialize_results_storage(
    metrics: List[str] = ['accuracy', 'precision', 'recall', 'f1']
) -> Dict[str, List[float]]:
    """
    Initialize storage dictionary for nested cross-validation results.

    Parameters:
    ----------
    metrics : list of str, optional
        List of metric names to store (default: ['accuracy', 'precision', 'recall', 'f1']).

    Returns:
    -------
    nested_scores : dict
        Dictionary with keys like 'test_accuracy' initialized to empty lists.
    """
    nested_scores = {f'test_{metric}': [] for metric in metrics}
    return nested_scores


from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import logging

def evaluate_fold(X_test, y_test, nested_scores, model, average='binary', return_confusion=False):
    """
    Evaluate model performance on a test fold and update nested_scores dict.

    Parameters:
    ----------
    X_test : pd.DataFrame or np.ndarray
        Test feature set.
    y_test : pd.Series or np.ndarray
        True labels for test set.
    nested_scores : dict
        Dictionary storing accumulated fold metrics.
    model : sklearn-like estimator
        Trained model with predict method.
    average : str, default='binary'
        Averaging method for precision, recall, f1 (useful for multiclass).
    return_confusion : bool, default=False
        Whether to return the confusion matrix along with nested_scores.

    Returns:
    -------
    nested_scores : dict
        Updated dictionary with appended fold metrics.
    conf_matrix : np.ndarray (optional)
        Confusion matrix for the fold (only if return_confusion=True).
    """

    y_pred_fold = model.predict(X_test)

    conf_matrix = confusion_matrix(y_test, y_pred_fold)
    
    # Safe attempt to unpack confusion matrix, fallback if shape != (2,2)
    try:
        tn, fp, fn, tp = conf_matrix.ravel()
    except ValueError:
        tn = fp = fn = tp = None

    logging.debug(f"Confusion Matrix:\n{conf_matrix}")

    fold_scores = {
        'accuracy': accuracy_score(y_test, y_pred_fold),
        'precision': precision_score(y_test, y_pred_fold, average=average, zero_division=0),
        'recall': recall_score(y_test, y_pred_fold, average=average, zero_division=0),
        'f1': f1_score(y_test, y_pred_fold, average=average, zero_division=0)
    }

    for metric, score in fold_scores.items():
        nested_scores.setdefault(f'test_{metric}', []).append(score)

    if return_confusion:
        return nested_scores, conf_matrix
    else:
        return nested_scores

