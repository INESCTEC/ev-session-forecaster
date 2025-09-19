import unittest
import pandas as pd
from app.forecast.processing import parse_data, parse_validate_data


class TestProcessingFunctions(unittest.TestCase):

    def setUp(self):
        """
        Set up test data for the tests.
        """
        self.raw_data = [
            {
                "user_id": 1,
                "start_time": "2024-11-21T10:00:00",
                "end_time": None,
                "states": [
                    {"state": "charging", "datetime": "2024-11-21T10:05:00"},
                    {"state": "not_plugged", "datetime": "2024-11-21T10:30:00"},
                ],
                "total_energy_transfered": 15.5,
            },
            {
                "user_id": 2,
                "start_time": "2024-11-21T12:00:00",
                "end_time": "2024-11-21T12:30:00",
                "states": [
                    {"state": "charging", "datetime": "2024-11-21T12:05:00"},
                ],
                "total_energy_transfered": 10.0,
            },
        ]

    def test_parse_data(self):
        """
        Test the parse_data function to ensure it correctly converts JSON data to a DataFrame.
        """
        df = parse_data(self.raw_data)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 2)
        self.assertIn("user_id", df.columns)
        self.assertIn("start_time", df.columns)
        self.assertIn("end_time", df.columns)
        self.assertIn("states", df.columns)
        self.assertIn("total_energy_transfered", df.columns)

    def test_parse_validate_data(self):
        """
        Test the parse_validate_data function to ensure it validates and cleans the data correctly.
        """
        df = parse_data(self.raw_data)
        validated_df = parse_validate_data(df)

        # Ensure that the DataFrame returned is not None
        self.assertIsNotNone(validated_df)

        # Ensure that the 'end_time' field is populated for the first row
        self.assertIsNotNone(validated_df.iloc[0]["end_time"])

        # Ensure no rows with 'total_energy_transfered' <= 0
        self.assertFalse((validated_df["total_energy_transfered"] <= 0).any())

        # Ensure no duplicate rows exist
        self.assertFalse(validated_df.duplicated().any())

        # Ensure 'start_time' column has no null or zero values
        self.assertFalse(validated_df["start_time"].isnull().any())
        self.assertFalse((validated_df["start_time"] == 0).any())


if __name__ == "__main__":
    unittest.main()
