import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime
import requests


# Mock settings
class MockSettings:
    API_URL_POST = "https://mock-api.com/post-ev-sessions"
    API_KEY_POST = "mock_api_key"


# Mock logger
class MockLogger:
    @staticmethod
    def debug(msg):
        pass


settings = MockSettings()
logger = MockLogger()


# The function to test
def post_ev_session_data(predictions):
    """
    Connection to REST API to retrieve EV session information.

    :param: predictions - Session characteristics including forecast predictions.
    :return: json response - EV session information based on user.
    """
    url = settings.API_URL_POST
    api_key = settings.API_KEY_POST

    # header creation
    header = {"x-api-key": api_key}

    response = requests.post(url, headers=header, json=predictions, verify=True)
    if response.status_code == 200:
        logger.debug("POST request successful")
        return response.json()
    else:
        raise Exception(f"Error posting EV session data: {response.text}")


def create_json_forecast(
    launch_time,
    duration_value,
    energy_value,
    returnal_date,
    returnal_time,
    model_id,
    evse_id,
    session_id,
    user_id,
):
    return {
        "launch_time": launch_time.isoformat(),
        "forecasts": {
            "duration": {
                "value": int(round(duration_value, 0)),
                "unit": "minute",
            },
            "energy_consumption": {
                "value": round(energy_value / 1000, 4),
                "unit": "kWh",
            },
            "returnal": {
                "date": returnal_date,
                "time": returnal_time,
                "d0": 0,
                "d1": 0,
                "d2": 0,
                "d3": 0,
                "d4": 0,
                "d5": 0,
                "d6": 0,
                "d7": 0,
                "d7_plus": 0,
            },
        },
        "model_id": model_id,
        "evse_id": evse_id,
        "session_id": session_id,
        "user_id": user_id,
    }


class TestPostEvSessionData(unittest.TestCase):
    @patch("requests.post")
    def test_post_ev_session_data_success(self, mock_post):
        # Arrange
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "success"}
        mock_post.return_value = mock_response

        launch_time = datetime(2024, 11, 21, 15, 30)
        duration_value = 120.5
        energy_value = 15500  # in Wh
        returnal_date = "2024-11-22"
        returnal_time = "10:00:00"
        model_id = "ml"
        evse_id = "EVSE123"
        session_id = "SESSION456"
        user_id = "USER789"

        predictions = create_json_forecast(
            launch_time,
            duration_value,
            energy_value,
            returnal_date,
            returnal_time,
            model_id,
            evse_id,
            session_id,
            user_id,
        )

        # Act
        result = post_ev_session_data(predictions)

        # Assert
        self.assertEqual(result, {"status": "success"})
        mock_post.assert_called_once()

        # Verify request parameters
        called_args, called_kwargs = mock_post.call_args
        self.assertEqual(called_kwargs["headers"], {"x-api-key": "mock_api_key"})
        self.assertEqual(called_kwargs["json"], predictions)

    @patch("requests.post")
    def test_post_ev_session_data_failure(self, mock_post):
        # Arrange
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"
        mock_post.return_value = mock_response

        launch_time = datetime(2024, 11, 21, 15, 30)
        duration_value = 120.5
        energy_value = 15500  # in Wh
        returnal_date = "2024-11-22"
        returnal_time = "10:00:00"
        model_id = "ml"
        evse_id = "EVSE123"
        session_id = "SESSION456"
        user_id = "USER789"

        predictions = create_json_forecast(
            launch_time,
            duration_value,
            energy_value,
            returnal_date,
            returnal_time,
            model_id,
            evse_id,
            session_id,
            user_id,
        )

        # Act and Assert
        with self.assertRaises(Exception) as context:
            post_ev_session_data(predictions)

        self.assertIn(
            "Error posting EV session data: Bad Request", str(context.exception)
        )
        mock_post.assert_called_once()

        # Verify request parameters
        called_args, called_kwargs = mock_post.call_args
        self.assertEqual(called_kwargs["headers"], {"x-api-key": "mock_api_key"})
        self.assertEqual(called_kwargs["json"], predictions)


if __name__ == "__main__":
    unittest.main()
