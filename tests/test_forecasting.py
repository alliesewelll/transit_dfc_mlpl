import unittest
import numpy as np
import pandas as pd
from src.features import training_matrix, feature_row, recursive_predict
from src.data import validate_target
from src.modeling import predict


class ForecastTests(unittest.TestCase):
    def setUp(self):
        self.y = pd.Series(np.arange(100, 196, dtype=float), index=pd.date_range("2010-01-01", periods=96, freq="MS"))

    def test_training_features_cannot_see_current_or_future_target(self):
        before, _ = training_matrix(self.y)
        changed = self.y.copy()
        changed.iloc[50:] += 1_000_000
        after, _ = training_matrix(changed)
        pd.testing.assert_frame_equal(before.loc[:self.y.index[50]], after.loc[:self.y.index[50]])

    def test_recursive_steps_use_predictions(self):
        class AddTen:
            def predict(self, x):
                return [x["lag_1"].iloc[0] + 10]
        np.testing.assert_allclose(recursive_predict(AddTen(), self.y, 3), [205, 215, 225])

    def test_seasonal_naive_repeats_correct_months(self):
        expected = np.tile(self.y.iloc[-12:].to_numpy(), 2)
        np.testing.assert_array_equal(predict("seasonal_naive", None, self.y, 24), expected)

    def test_missing_month_rejected(self):
        with self.assertRaises(ValueError):
            validate_target(self.y.drop(self.y.index[30]))

    def test_missing_value_rejected(self):
        self.y.iloc[20] = np.nan
        with self.assertRaises(ValueError):
            validate_target(self.y)

    def test_api_empty_and_invalid_download(self):
        from fastapi.testclient import TestClient
        from api.main import app
        with TestClient(app) as client:
            self.assertEqual(client.get("/health").status_code, 200)
            self.assertEqual(client.get("/").status_code, 200)
            self.assertEqual(client.get("/download/secrets").status_code, 404)


if __name__ == "__main__":
    unittest.main()
