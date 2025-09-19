import requests
from loguru import logger
import app.conf.settings as settings


@logger.catch(level="ERROR", message="Error posting EV session data:")
def post_ev_session_data(predictions):
    """
    Connection to REST API to retrive EV session information.

    :param: predicitons - Session characteristics including forecast predicitons.
    :return: json response - EV session information based on user.
    """
    url = settings.API_URL_POST
    api_key = settings.API_KEY_POST

    # header creation
    header = {"x-api-key": api_key}

    response = requests.post(url, headers=header, json=predictions, verify=True)
    if response.status_code == 200:
        logger.debug("POST request successful")
    # request to REST API (get data for EV owner X)
    else:
        raise Exception(f"Error posting EV session data: {response.text}")
