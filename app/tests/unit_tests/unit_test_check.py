import unittest
import pandas as pd
import os
from unittest.mock import patch, MagicMock
from app.forecast.processing import check_user_train
from app.forecast.core import model_exists


class TestForecastFunctions(unittest.TestCase):
    def setUp(self):
        """
        Set up test data for the tests.
        """
        self.test_data = pd.DataFrame(
            {
                "user_id": [1, 1, 1, 2, 2, 3],
                "session_id": [101, 102, 103, 201, 202, 301],
            }
        )

    def test_check_user_train(self):
        """
        Test the check_user_train function to ensure it identifies users with more than n sessions.
        """
        # Test for users with more than 2 sessions
        users_to_train = check_user_train(data=self.test_data, n_sessions=2)
        self.assertEqual(users_to_train, [1])  # Only user_id 1 has more than 2 sessions

        # Test for users with more than 1 session
        users_to_train = check_user_train(data=self.test_data, n_sessions=1)
        self.assertEqual(
            set(users_to_train), {1, 2}
        )  # Users 1 and 2 have more than 1 session

        # Test for users with more than 3 sessions
        users_to_train = check_user_train(data=self.test_data, n_sessions=3)
        self.assertEqual(users_to_train, [])  # No users have more than 3 sessions

    @patch("os.path.exists")
    @patch("os.listdir")
    def test_model_exists(self, mock_listdir, mock_exists):
        """
        Test the model_exists function to check if it correctly identifies existing models.
        """
        mock_exists.return_value = True  # Mock the directory existence
        mock_listdir.return_value = [
            "1_target_feature1.pkl",
            "1_target_feature2.pkl",
            "2_target_feature1.pkl",
        ]  # Mock the directory files

        # Test for an existing model
        exists = model_exists(
            path_folder="test_folder",
            target_feature="target_feature1",
            user_id=1,
        )
        self.assertTrue(exists)

        # Test for a non-existing model
        exists = model_exists(
            path_folder="test_folder",
            target_feature="target_feature3",
            user_id=1,
        )
        self.assertFalse(exists)

        # Test for a specific file match
        exists = model_exists(
            path_folder="test_folder",
            target_feature="target_feature1",
            user_id=1,
            define_file="1_target_feature1.pkl",
        )
        self.assertTrue(exists)

        # Test for a file mismatch
        exists = model_exists(
            path_folder="test_folder",
            target_feature="target_feature1",
            user_id=1,
            define_file="non_matching_file.pkl",
        )
        self.assertFalse(exists)

    @patch("os.path.exists")
    def test_model_exists_no_directory(self, mock_exists):
        """
        Test the model_exists function when the directory does not exist.
        """
        mock_exists.return_value = False  # Mock the directory does not exist
        exists = model_exists(
            path_folder="non_existent_folder",
            target_feature="target_feature1",
            user_id=1,
        )
        self.assertFalse(exists)  # Should return False if directory doesn't exist


if __name__ == "__main__":
    unittest.main()
