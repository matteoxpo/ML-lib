# tests/test_metrics.py

import unittest
import numpy as np
from sklearn.metrics import confusion_matrix
from utils.metrics import (
    accuracy,
    precision,
    recall,
    f1_score,
    mean_squared_error,
    root_mean_squared_error,
    r2_score,
    roc_auc_score,
    log_loss,
    confusion_matrix_1,
    mean_absolute_error,
    explained_variance_score
)

class TestMetrics(unittest.TestCase):

    def setUp(self):
        self.y_true_binary = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
        self.y_pred_binary = np.array([1, 0, 1, 0, 0, 1, 1, 0, 0, 0])

        self.y_true_regression = np.array([3.0, -0.5, 2.0, 7.0])
        self.y_pred_regression = np.array([2.5, 0.0, 2.0, 8.0])

    def test_accuracy(self):
        result = accuracy(self.y_true_binary, self.y_pred_binary)
        expected = 0.7
        self.assertAlmostEqual(result, expected, places=5)

    def test_precision(self):
        result = precision(self.y_true_binary, self.y_pred_binary)
        expected = 0.6
        self.assertAlmostEqual(result, expected, places=5)

    def test_recall(self):
        result = recall(self.y_true_binary, self.y_pred_binary)
        expected = 0.75
        self.assertAlmostEqual(result, expected, places=5)

    def test_f1_score(self):
        result = f1_score(self.y_true_binary, self.y_pred_binary)
        expected = 0.6666666666666666
        self.assertAlmostEqual(result, expected, places=5)

    def test_mean_squared_error(self):
        result = mean_squared_error(self.y_true_regression, self.y_pred_regression)
        expected = 0.375
        self.assertAlmostEqual(result, expected, places=5)

    def test_root_mean_squared_error(self):
        result = root_mean_squared_error(self.y_true_regression, self.y_pred_regression)
        expected = np.sqrt(0.375)
        self.assertAlmostEqual(result, expected, places=5)

    def test_r2_score(self):
        result = r2_score(self.y_true_regression, self.y_pred_regression)
        expected = 0.9486081370449679
        self.assertAlmostEqual(result, expected, places=5)

    def test_confusion_matrix(self):
        result = confusion_matrix_1(self.y_true_binary, self.y_pred_binary)
        print("ASDSAD")
        print(type(confusion_matrix(self.y_true_binary, self.y_pred_binary)))
        print(confusion_matrix(self.y_true_binary, self.y_pred_binary))
        print(result)
        # matrix = [ [tp, fn],
        #            [fp, tn]]
        # self.assertEqual(result, matrix)

    def test_mean_absolute_error(self):
        result = mean_absolute_error(self.y_true_regression, self.y_pred_regression)
        expected = 0.5
        self.assertAlmostEqual(result, expected, places=5)

    def test_explained_variance_score(self):
        result = explained_variance_score(self.y_true_regression, self.y_pred_regression)
        expected = 0.9571111111111112
        self.assertAlmostEqual(result, expected, places=5)

if __name__ == "__main__":
    unittest.main()
