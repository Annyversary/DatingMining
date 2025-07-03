import unittest
import pandas as pd

from src.data_prep.data_preprocessing import encode_categorical_features

class TestEncodeCategoricalFeatures(unittest.TestCase):

    def setUp(self):
        # Sample training data with categorical and numeric columns
        self.X_train = pd.DataFrame({
            'cat1': ['A', 'B', 'A', 'C'],
            'cat2': ['X', 'Y', 'X', 'Z'],
            'num': [1, 2, 3, 4]
        })

        # Sample test data with same categorical columns and numeric column
        self.X_test = pd.DataFrame({
            'cat1': ['B', 'C', 'A'],
            'cat2': ['Y', 'Z', 'X'],
            'num': [10, 20, 30]
        })

    def test_encoding_removes_original_categorical_columns(self):
        # Check that original categorical columns are dropped after encoding
        X_train_enc, X_test_enc = encode_categorical_features(self.X_train.copy(), self.X_test.copy())

        self.assertNotIn('cat1', X_train_enc.columns)
        self.assertNotIn('cat2', X_train_enc.columns)
        self.assertNotIn('cat1', X_test_enc.columns)
        self.assertNotIn('cat2', X_test_enc.columns)

    def test_encoding_adds_expected_columns(self):
        # Check that expected one-hot encoded columns are present
        X_train_enc, X_test_enc = encode_categorical_features(self.X_train.copy(), self.X_test.copy())

        expected_columns = [
            'num',
            'cat1_B', 'cat1_C',  # 'cat1_A' is dropped due to drop='first'
            'cat2_Y', 'cat2_Z'   # 'cat2_X' dropped as baseline
        ]

        for col in expected_columns:
            self.assertIn(col, X_train_enc.columns)
            self.assertIn(col, X_test_enc.columns)

        # Check no extra columns beyond expected
        self.assertEqual(len(X_train_enc.columns), len(expected_columns))
        self.assertEqual(len(X_test_enc.columns), len(expected_columns))

    def test_numeric_column_unchanged(self):
        # Numeric columns should remain unchanged
        X_train_enc, X_test_enc = encode_categorical_features(self.X_train.copy(), self.X_test.copy())

        pd.testing.assert_series_equal(X_train_enc['num'], self.X_train['num'].reset_index(drop=True))
        pd.testing.assert_series_equal(X_test_enc['num'], self.X_test['num'].reset_index(drop=True))

    def test_encoded_values_are_correct(self):
        # Verify the one-hot encoding values for training data
        X_train_enc, _ = encode_categorical_features(self.X_train.copy(), self.X_test.copy())

        expected_cat1_B = [0, 1, 0, 0]
        expected_cat1_C = [0, 0, 0, 1]
        self.assertListEqual(X_train_enc['cat1_B'].tolist(), expected_cat1_B)
        self.assertListEqual(X_train_enc['cat1_C'].tolist(), expected_cat1_C)

        expected_cat2_Y = [0, 1, 0, 0]
        expected_cat2_Z = [0, 0, 0, 1]
        self.assertListEqual(X_train_enc['cat2_Y'].tolist(), expected_cat2_Y)
        self.assertListEqual(X_train_enc['cat2_Z'].tolist(), expected_cat2_Z)


if __name__ == '__main__':
    unittest.main()
