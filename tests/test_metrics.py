# tests/test_metrics.py

import unittest
import numpy as np
import sklearn
import sklearn.metrics
import utils
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
    confusion_matrix,
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
        expected = sklearn.metrics.accuracy_score(self.y_true_binary, self.y_pred_binary)
        self.assertAlmostEqual(result, expected, places=5)

    def test_precision(self):
        result = precision(self.y_true_binary, self.y_pred_binary)
        expected = sklearn.metrics.precision_score(self.y_true_binary, self.y_pred_binary)
        self.assertAlmostEqual(result, expected, places=5)

    def test_recall(self):
        result = recall(self.y_true_binary, self.y_pred_binary)
        expected = sklearn.metrics.recall_score(self.y_true_binary, self.y_pred_binary)
        self.assertAlmostEqual(result, expected, places=5)

    def test_f1_score(self):
        result = f1_score(self.y_true_binary, self.y_pred_binary)
        expected = sklearn.metrics.f1_score(self.y_true_binary, self.y_pred_binary)
        self.assertAlmostEqual(result, expected, places=5)

    def test_mean_squared_error(self):
        result = mean_squared_error(self.y_true_regression, self.y_pred_regression)
        expected = sklearn.metrics.mean_squared_error(self.y_true_regression, self.y_pred_regression)
        self.assertAlmostEqual(result, expected, places=5)

    def test_root_mean_squared_error(self):
        result = root_mean_squared_error(self.y_true_regression, self.y_pred_regression)
        expected = sklearn.metrics.root_mean_squared_error(self.y_true_regression, self.y_pred_regression)
        # np.sqrt(0.375)
        self.assertAlmostEqual(result, expected, places=5)

    def test_r2_score(self):
        result = r2_score(self.y_true_regression, self.y_pred_regression)
        expected = sklearn.metrics.r2_score(self.y_true_regression, self.y_pred_regression)
        # 0.9486081370449679
        self.assertAlmostEqual(result, expected, places=5)

    def test_confusion_matrix(self):
        result = np.array(utils.metrics.confusion_matrix(self.y_true_binary, self.y_pred_binary))
        matrix = sklearn.metrics.confusion_matrix(self.y_true_binary, self.y_pred_binary)
        self.assertTrue(np.array_equal(result, matrix))

    def test_mean_absolute_error(self):
        result = mean_absolute_error(self.y_true_regression, self.y_pred_regression)
        expected = sklearn.metrics.mean_absolute_error(self.y_true_regression, self.y_pred_regression)
        self.assertAlmostEqual(result, expected, places=5)

    def test_explained_variance_score(self):
        
        result = explained_variance_score(self.y_true_regression, self.y_pred_regression)
        expected = sklearn.metrics.explained_variance_score(self.y_true_regression, self.y_pred_regression)
        # 0.9571111111111112
        self.assertAlmostEqual(result, expected, places=5)

if __name__ == "__main__":
    unittest.main()
