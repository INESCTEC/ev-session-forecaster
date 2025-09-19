import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta
import requests


# Mock settings
class MockSettings:
    API_URL = "https://mock-api.com/ev-sessions"
    API_KEY = "mock_api_key"


# Mock logger
class MockLogger:
    @staticmethod
    def debug(msg):
        pass


settings = MockSettings()
logger = MockLogger()


# The function to test
def get_ev_session_data(user_id=None):
    url = settings.API_URL
    api_key = settings.API_KEY

    # header creation
    header = {"x-api-key": api_key}
    # end_datetime creation with today dates
    current_datetime = datetime.now() + timedelta(hours=1)

    # Format it as "YYYY-MM-DDTHH:MM:SS"
    today_datetime = current_datetime.strftime("%Y-%m-%dT%H:%M:%S")

    # params creation
    params = {
        "start_datetime": "2020-01-01T09:00:00",
        "end_datetime": today_datetime,
        "user_id": user_id,
    }

    response = requests.get(url, headers=header, params=params, verify=True)
    if response.status_code == 200:
        logger.debug("Response successful")
        data = response.json()
    else:
        raise Exception(f"Error getting EV session data: {response.text}")
    return data


class TestGetEvSessionData(unittest.TestCase):
    @patch("requests.get")
    def test_get_ev_session_data_success(self, mock_get):
        # Arrange
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"sessions": []}
        mock_get.return_value = mock_response

        # Act
        result = get_ev_session_data(user_id=17)

        # Assert
        self.assertEqual(result, {"sessions": []})
        mock_get.assert_called_once()

        # Verify request parameters
        called_args, called_kwargs = mock_get.call_args
        self.assertEqual(called_kwargs["headers"], {"x-api-key": "mock_api_key"})
        self.assertIn("params", called_kwargs)
        self.assertEqual(called_kwargs["params"]["user_id"], 17)

    @patch("requests.get")
    def test_get_ev_session_data_failure(self, mock_get):
        # Arrange
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.text = "Not Found"
        mock_get.return_value = mock_response

        # Act and Assert
        with self.assertRaises(Exception) as context:
            get_ev_session_data(user_id=17)

        self.assertIn(
            "Error getting EV session data: Not Found", str(context.exception)
        )
        mock_get.assert_called_once()


if __name__ == "__main__":
    unittest.main()
