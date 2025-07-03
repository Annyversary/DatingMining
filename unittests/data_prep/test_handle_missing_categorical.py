import unittest
import pandas as pd
import numpy as np

from pandas.api.types import is_categorical_dtype
from src.data_prep.data_preprocessing import handle_missing_categorical

class TestHandleMissingCategorical(unittest.TestCase):

    def setUp(self):
        self.X_train = pd.DataFrame({
            'cat1': pd.Series(['A', np.nan, 'B', 'A'], dtype='category'),
            'cat2': ['X', 'Y', np.nan, 'Z'],
            'num': [1, 2, 3, 4]
        })

        self.X_test = pd.DataFrame({
            'cat1': pd.Series([np.nan, 'B', 'C'], dtype='category'),
            'cat2': [np.nan, 'Y', 'X'],
            'num': [10, 20, 30]
        })

    def test_missing_values_filled(self):
        X_train_result, X_test_result = handle_missing_categorical(self.X_train.copy(), self.X_test.copy())

        # Check that there are no missing values in categorical columns
        cat_train = X_train_result.select_dtypes(include=['category', 'object'])
        cat_test = X_test_result.select_dtypes(include=['category', 'object'])

        self.assertFalse(cat_train.isnull().values.any(), "NaNs in X_train after handling")
        self.assertFalse(cat_test.isnull().values.any(), "NaNs in X_test after handling")

    def test_unknown_category_added(self):
        X_train_result, X_test_result = handle_missing_categorical(self.X_train.copy(), self.X_test.copy())

        for col in ['cat1', 'cat2']:
            # Ensure 'Unknown' is now a category in both train and test
            self.assertIn('Unknown', X_train_result[col].unique(), f"'Unknown' not in X_train[{col}]")
            self.assertIn('Unknown', X_test_result[col].unique(), f"'Unknown' not in X_test[{col}]")

    def test_data_types_unchanged(self):
        X_train_result, X_test_result = handle_missing_categorical(self.X_train.copy(), self.X_test.copy())

        # cat1 is originally categorical
        self.assertTrue(is_categorical_dtype(X_train_result['cat1']))
        self.assertTrue(is_categorical_dtype(X_test_result['cat1']))

        # cat2 is originally object; dtype may still be object
        self.assertTrue(X_train_result['cat2'].dtype in ['object', 'category'])
        self.assertTrue(X_test_result['cat2'].dtype in ['object', 'category'])

if __name__ == '__main__':
    unittest.main()
    
