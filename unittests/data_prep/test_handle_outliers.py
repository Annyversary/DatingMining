import unittest
import pandas as pd
import numpy as np

from sklearn.datasets import make_blobs
from src.data_prep.data_preprocessing import handle_numeric_outliers

class TestHandleNumericOutliers(unittest.TestCase):

    def setUp(self):
        # Create a dataset with numeric and categorical features
        X_blob, y_blob = make_blobs(n_samples=100, centers=1, n_features=3, random_state=42)
        self.X_train = pd.DataFrame(X_blob, columns=["num1", "num2", "num3"])
        self.X_train["cat"] = ["A"] * 95 + ["B"] * 5  # Add categorical column
        self.y_train = pd.Series(y_blob)

    def test_output_shapes_reduced(self):
        # Run outlier removal
        X_clean, y_clean = handle_numeric_outliers(self.X_train, self.y_train, contamination=0.1)

        # Expect fewer samples than original (some removed as outliers)
        self.assertLess(len(X_clean), len(self.X_train))
        self.assertEqual(len(X_clean), len(y_clean))

    def test_no_categorical_column_used_for_outlier_detection(self):
        # Run function and ensure categorical column is preserved
        X_clean, _ = handle_numeric_outliers(self.X_train, self.y_train)
        
        # The 'cat' column should still exist
        self.assertIn("cat", X_clean.columns)

    def test_only_numeric_features_used_in_detection(self):
        # Extract numeric subset
        X_numeric = self.X_train.select_dtypes(include=['number'])
        
        # Apply IF and LOF directly for manual reference
        from sklearn.ensemble import IsolationForest
        from sklearn.neighbors import LocalOutlierFactor
        
        if_mask = IsolationForest(contamination=0.05, random_state=42).fit_predict(X_numeric) == 1
        X_if = X_numeric[if_mask]
        lof_mask = LocalOutlierFactor(contamination=0.05).fit_predict(X_if) == 1
        final_numeric = X_if[lof_mask]

        # Run original function
        X_clean, _ = handle_numeric_outliers(self.X_train, self.y_train, contamination=0.05)
        
        # Check that result contains only those filtered by numeric logic
        self.assertLessEqual(len(X_clean), len(final_numeric))

    def test_return_type_is_dataframe(self):
        # Run the function
        X_clean, y_clean = handle_numeric_outliers(self.X_train, self.y_train)
        
        # Check that return types are correct
        self.assertIsInstance(X_clean, pd.DataFrame)
        self.assertIsInstance(y_clean, pd.Series)

if __name__ == '__main__':
    unittest.main()
