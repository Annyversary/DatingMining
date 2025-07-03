import unittest
import pandas as pd
import numpy as np

from src.data_prep.data_preprocessing import handle_missing_numericals

class TestHandleMissingNumericals(unittest.TestCase):

    def setUp(self):
        self.X_train = pd.DataFrame({
            'num1': [1, 2, np.nan, 4, 5],
            'num2': [10.0, np.nan, 30.0, 40.0, 50.0],
            'cat': ['a', 'b', 'c', np.nan, 'e']
        })

        self.X_test = pd.DataFrame({
            'num1': [np.nan, 3, 6],
            'num2': [20.0, np.nan, 60.0],
            'cat': ['b', 'c', 'd']
        })

    def test_missing_values_filled_with_median(self):
        X_train_result, X_test_result = handle_missing_numericals(self.X_train.copy(), self.X_test.copy())

        # Median values for train columns (manually calculated)
        median_num1 = self.X_train['num1'].median()  # 2.5
        median_num2 = self.X_train['num2'].median()  # 30.0

        # Assert no NaNs in numerical columns
        num_train = X_train_result.select_dtypes(include=['float64', 'int64'])
        num_test = X_test_result.select_dtypes(include=['float64', 'int64'])
        self.assertFalse(num_train.isnull().values.any(), "NaNs in X_train after handling")
        self.assertFalse(num_test.isnull().values.any(), "NaNs in X_test after handling")

        # Check that NaNs replaced by median in train
        self.assertTrue((X_train_result['num1'] == median_num1).any())
        self.assertTrue((X_train_result['num2'] == median_num2).any())

        # Check that NaNs replaced by median in test
        self.assertTrue((X_test_result['num1'] == median_num1).any())
        self.assertTrue((X_test_result['num2'] == median_num2).any())

    def test_non_numerical_columns_unchanged(self):
        X_train_result, X_test_result = handle_missing_numericals(self.X_train.copy(), self.X_test.copy())

        # Non-numerical columns should remain unchanged (except for NaNs which this func does not touch)
        pd.testing.assert_series_equal(X_train_result['cat'], self.X_train['cat'])
        pd.testing.assert_series_equal(X_test_result['cat'], self.X_test['cat'])

if __name__ == '__main__':
    unittest.main()
