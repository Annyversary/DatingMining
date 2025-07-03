import unittest
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.data_prep.data_preprocessing import scale_and_transform

class TestScaleAndTransform(unittest.TestCase):

    def setUp(self):
        # Create sample train and test DataFrames
        self.X_train = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50]
        })
        self.X_test = pd.DataFrame({
            'A': [6, 7],
            'B': [60, 70]
        })
        self.scaler = StandardScaler()

    def test_scaled_shape_and_columns(self):
        # Test if the scaled data keeps the same shape and columns
        X_train_scaled, X_test_scaled = scale_and_transform(self.X_train, self.X_test, self.scaler)
        self.assertEqual(X_train_scaled.shape, self.X_train.shape)
        self.assertEqual(X_test_scaled.shape, self.X_test.shape)
        self.assertListEqual(list(X_train_scaled.columns), list(self.X_train.columns))
        self.assertListEqual(list(X_test_scaled.columns), list(self.X_test.columns))

    def test_train_scaled_mean_zero(self):
        # Test that the scaled training data has approximately zero mean (due to StandardScaler)
        X_train_scaled, _ = scale_and_transform(self.X_train, self.X_test, self.scaler)
        means = X_train_scaled.mean().round(decimals=6)
        for mean in means:
            self.assertAlmostEqual(mean, 0.0, places=5)

    def test_test_scaled_values(self):
        # Test that the test data is scaled using the parameters from the training data
        X_train_scaled, X_test_scaled = scale_and_transform(self.X_train, self.X_test, self.scaler)
        expected = (self.X_test - self.X_train.mean()) / self.X_train.std(ddof=0)
        pd.testing.assert_frame_equal(X_test_scaled, expected)

if __name__ == "__main__":
    unittest.main()
