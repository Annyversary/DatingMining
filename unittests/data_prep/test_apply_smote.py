import unittest
import pandas as pd
import numpy as np

from src.data_prep.data_preprocessing import apply_smote

class TestApplySMOTE(unittest.TestCase):

    def setUp(self):
        # Create imbalanced dataset
        self.X_train = pd.DataFrame({
            'feature1': np.concatenate([np.random.normal(0, 1, 50), np.random.normal(5, 1, 10)]),
            'feature2': np.concatenate([np.random.normal(0, 1, 50), np.random.normal(5, 1, 10)])
        })
        # Imbalanced target: class 0 has 50 samples, class 1 has 10 samples
        self.y_train = np.array([0]*50 + [1]*10)

    def test_balanced_classes_after_smote(self):
        X_res, y_res = apply_smote(self.X_train, self.y_train)
        # Check that classes are balanced after SMOTE
        unique, counts = np.unique(y_res, return_counts=True)
        counts_dict = dict(zip(unique, counts))
        self.assertEqual(counts_dict[0], counts_dict[1], "Classes are not balanced after SMOTE")

    def test_no_data_loss(self):
        X_res, y_res = apply_smote(self.X_train, self.y_train)
        # Check that number of samples increased or stayed the same
        self.assertGreaterEqual(len(X_res), len(self.X_train))
        self.assertGreaterEqual(len(y_res), len(self.y_train))

    def test_return_types(self):
        X_res, y_res = apply_smote(self.X_train, self.y_train)
        # Check output types
        self.assertTrue(hasattr(X_res, "shape"), "X_res should be array-like")
        self.assertTrue(hasattr(y_res, "shape") or isinstance(y_res, np.ndarray), "y_res should be array-like or ndarray")

if __name__ == "__main__":
    unittest.main()
