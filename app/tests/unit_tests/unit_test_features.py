import unittest
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from app.forecast.core import feature_engineering_time, feature_engineering_naive


class TestFeatureEngineeringSequential(unittest.TestCase):

    def setUp(self):
        """
        Set up test data for feature engineering, including `evse_id`.
        """
        # Input DataFrame with `start_datetime` as datetime objects
        self.test_data = pd.DataFrame(
            {
                "session_id": [1, 2, 3, 4],
                "evse_id": [101, 102, 103, 104],  # Added `evse_id`
                "start_datetime": pd.to_datetime(
                    [
                        "2024-11-01T08:00:00",
                        "2024-11-01T10:00:00",
                        "2024-11-01T12:00:00",
                        "2024-11-01T14:00:00",
                    ]
                ),
                "end_datetime": pd.to_datetime(
                    [
                        "2024-11-01T09:00:00",
                        "2024-11-01T11:00:00",
                        "2024-11-01T13:00:00",
                        "2024-11-01T15:00:00",
                    ]
                ),
                "total_energy_transfered": [15.0, 20.0, 25.0, 30.0],
            }
        )
        self.target_feature = "total_energy_transfered"

    def test_feature_engineering_sequence(self):
        """
        Test sequential processing of feature engineering functions.
        """
        # Apply time-based feature engineering
        df_time_features = feature_engineering_time(self.test_data)

        # Verify time features
        self.assertIn("start_year", df_time_features.columns)
        self.assertIn("start_month", df_time_features.columns)
        self.assertIn("start_day", df_time_features.columns)
        self.assertIn("day_of_week", df_time_features.columns)

        # Ensure `start_datetime` is preserved as datetime
        self.assertTrue(
            pd.api.types.is_datetime64_any_dtype(df_time_features["start_datetime"])
        )

        # Apply naive feature engineering on the output of time-based feature engineering
        df_naive_features, lowest_mse_column = feature_engineering_naive(
            dataframe=df_time_features, target_feature=self.target_feature
        )

        # Verify naive features
        naive_features = [
            "naive_avg",
            "naive_avg_row5",
            "naive_med",
            "naive_med_row5",
            "naive_avg_period",
            "naive_med_period",
            "persistence",
        ]
        for feature in naive_features:
            self.assertIn(feature, df_naive_features.columns)

        # Verify lowest MSE column
        self.assertIn(lowest_mse_column, naive_features)

    def test_integration_of_features(self):
        """
        Test the final DataFrame output after both functions are applied.
        """
        # Apply both functions in sequence
        df_time_features = feature_engineering_time(self.test_data)
        df_naive_features, _ = feature_engineering_naive(
            dataframe=df_time_features, target_feature=self.target_feature
        )

        # Check the combined DataFrame contains all expected features
        combined_features = [
            "start_year",
            "start_month",
            "start_day",
            "naive_avg",
            "persistence",
            "evse_id",
        ]
        for feature in combined_features:
            self.assertIn(feature, df_naive_features.columns)

        # Verify `evse_id` column is preserved in the final DataFrame
        self.assertTrue("evse_id" in df_naive_features.columns)


if __name__ == "__main__":
    unittest.main()
