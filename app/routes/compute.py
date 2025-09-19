# from typing import Optional
from fastapi import APIRouter, HTTPException  # , Depends

from ..schemas import ForecastComputePayload

# from .helpers.auth import oauth2_scheme, verify_token
from ..forecast.pipeline import compute_forecast
from ..forecast.processing import send_error_email, send_success_email
from ..forecast.database import post_ev_session_data

from loguru import logger
import os
import sys


router = APIRouter()


@router.post("/forecast/session-data")
def post_compute_all_forecast(payload: ForecastComputePayload):
    """
    Launch forecast computation of session data (duration, energy consumption
    and next session start datetime).
    Admin privileges required.
    """

    # verify_token(token)
    logger.info("Initiating calculation of Forecasts")
    user_id = payload.user_id
    session_id = payload.session_id
    evse_id = payload.evse_id
    launch_time = payload.launch_time

    if (not session_id) and (not launch_time):
        raise HTTPException(
            detail="You must declare either 'session_id' or 'launch_time' fields on the payload.",
            status_code=400,
        )
    if session_id and launch_time:
        raise HTTPException(
            detail="You cannot declare both 'session_id' and 'launch_time' as payload fields.",
            status_code=400,
        )
    recipients_debug = os.environ.get("EMAIL_RECIPIENTS_DEBUG", "").split(",")

    logger.info(
        f"user_id: {user_id}\n" f"session_id: {session_id}\n" f"evse_id: {evse_id}"
    )

    try:
        response = compute_forecast(
            user_id=user_id,
            session_id=session_id,
            evse_id=evse_id,
            launch_time=launch_time,
        )
        logger.info("Forecast computed successfully")
    except Exception as e:
        # Log the exception with details
        logger.exception(f"Failed to compute forecast: {e}", exc_info=True)
        response = None
    # Save json in a file
    if response and session_id:
        logger.info(f"Forecast result: {response}")
        try:
            post_ev_session_data(predictions=response)
        except Exception as e:
            # Log the exception with details
            logger.error(f"Failed post data to database: {e}", exc_info=True)

        try:
            logger.info("Sending success email...")
            send_success_email(json_final=response)
            logger.info("Success email was sent.")
        except Exception as e:
            # Log the exception with details
            logger.exception(f"Failed to send success email: {e}", exc_info=True)
        # todo: add request to EV management interface (Jacinta)
    elif session_id is None:
        pass
    else:
        logger.info("Sending error email...")  # noqa
        try:
            send_error_email(error_email_recipients=recipients_debug)
            logger.info("Success email was sent.")
        except Exception as e:
            # Log the exception with details
            logger.exception(f"Failed to send error email: {e}", exc_info=True)
    return response


