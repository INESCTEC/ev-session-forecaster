import requests
import os
from datetime import datetime, timedelta
import pandas as pd
from loguru import logger
import app.conf.settings as settings
import json
from app.sendEmail.sendEmailClient.EmailClient import EmailClient
import numpy as np
from joblib import dump, load


@logger.catch(level="ERROR", message="Error getting EV session data:")
def get_ev_session_data(user_id=None):
    """
    Connection to REST API to retrive EV session information.

    :param: int - Ev owner.
    :return: json response - EV session information based on user.
    """
    url = settings.API_URL
    api_key = settings.API_KEY

    # header creation
    header = {"x-api-key": api_key}
    # end_datetime creation with today dates

    # params creation
    params = {
        "start_datetime": "2020-01-01T09:00:00",
        "user_id": user_id,
    }

    response = requests.get(url, headers=header, params=params, verify=True)
    if response.status_code == 200:
        logger.debug("Response successful")
        data = response.json()
    # request to REST API (get data for EV owner X)
    else:
        raise Exception(f"Error getting EV session data: {response.text}")
    return data


@logger.catch(level="ERROR", message="Error getting EV session data:")
def get_ev_session_all_sessions(start_datetime=None, user_id=None, end_datetime=None):
    """
    Connection to REST API to retrive EV session information.

    :param: int - Ev owner.
    :return: json response - EV session information based on user.
    """
    url = settings.API_URL
    api_key = settings.API_KEY

    # header creation
    header = {"x-api-key": api_key}
    # end_datetime creation with today dates
    current_datetime = datetime.now()

    # Format it as "YYYY-MM-DDTHH:MM:SS"
    if end_datetime is None:
        today_datetime = current_datetime.strftime("%Y-%m-%dT%H:%M:%S")
    else:
        today_datetime = end_datetime.strftime("%Y-%m-%dT%H:%M:%S")
    if start_datetime is None:
        logger.debug("start_datetime none")
        start_datetime = "2020-01-01T09:00:00"

    logger.debug(start_datetime)
    # params creation
    params = {
        "start_datetime": start_datetime,
        "end_datetime": today_datetime,
        "user_id": user_id,
    }

    response = requests.get(url, headers=header, params=params, verify=True)
    if response.status_code == 200:
        logger.debug("Response successful")
        data = response.json()
    # request to REST API (get data for EV owner X)
    else:
        raise Exception(f"Error getting EV session data: {response.text}")
    return data


def calculate_mad_md(group, target):
    # Step 2: Calculate the median of the column
    median = group[target].median()

    # Step 3: Calculate the absolute deviation from the median
    absolute_deviation = np.abs(group[target] - median)

    # Step 4: Calculate the MAD (median of absolute deviations)
    mad = np.median(absolute_deviation)

    # Step 5: Return the MAD value
    return median, mad

@logger.catch(level="ERROR", message="Parsing Data Error:")
def parse_data(data):
    """
    Parse JSON Data and convert to Dataframe

    :param data: json response - JSON containing data to be converted to Dataframe.
    :return: dataframe - - Converted dataframe.
    """
    # parse JSON data and convert to dataframe
    data_parsed = pd.DataFrame(data)
    return data_parsed


@logger.catch(level="ERROR", message="Validating Data Error:")
def parse_validate_data(data):
    """
    Parse and validate data from a DataFrame before processing.

    This function handles null values, corrects data types, and removes duplicate rows
    to ensure data integrity before further processing.

    :param data: dataframe - Input data to be parsed and validated.
    :return: dataframe - Processed DataFrame with cleaned and validated data.
    """
    # validate data and convert to dataframe
    # Define default corrections
    df = data.copy()
    logger.debug(f"Initial dataframe shape: {df.shape}")
    default_fill_value = 0
    inferred_column_types = {
        "user_id": "int"  # Example type inference, adjust as needed
    }

    for index, row in df.iterrows():
        logger.debug(f"index:{index}")
        logger.debug(f"row:{row}")
        end_date = (
            pd.to_datetime(row["end_time"]) if row["end_time"] is not None else None
        )

        if end_date is None:
            # If end_date is null, find the first 'not_plugged' datetime or the last datetime
            first_not_plugged = None
            last_datetime = None

            for event in row["states"]:
                logger.debug(f"event:{event}")
                # Iterate over events to find the first 'not_plugged'
                event_datetime = pd.to_datetime(event["datetime"])
                logger.debug(f"event_datetime:{event_datetime}")
                if last_datetime is None:
                    last_datetime = event_datetime
                elif (event_datetime - last_datetime) <= pd.Timedelta(days=1):
                    last_datetime = event_datetime
                if event["state"] == "not_plugged" and first_not_plugged is None:
                    first_not_plugged = event_datetime
                    break  # Exit the loop after finding the first 'not_plugged'

            # Use the first 'not_plugged' datetime if it exists, otherwise use the last datetime
            end_date = first_not_plugged if first_not_plugged else last_datetime
            df.at[index, "end_time"] = end_date

    # Iterate over each column in the DataFrame
    for column in df.columns:

        # 1. Check and correct for null values
        logger.debug("Check and correct for null values")

        if df[column].isnull().any():
            if column == "start_time":
                # Remove rows with null values in 'start time' column
                logger.debug(
                    f"Null values found in column '{column}'. Removing rows with null values."
                )
                df = df.dropna(subset=[column])
            else:
                # Fill null values with default value for other columns
                logger.debug(
                    f"Null values found in column '{column}'. Filling with default value '{default_fill_value}'."
                )
                df[column] = df[column].fillna(default_fill_value)

        if column == "start_time":
            if (df[column] == 0).any():
                logger.debug(
                    "Rows with 'start_time' equal to 0 found. Removing these rows."
                )
                df = df[df[column] != 0]
        # 2. Check and correct for data type
        logger.debug("Check and correct for data type")
        if column in inferred_column_types:
            expected_dtype = inferred_column_types[column]
            if df[column].dtype != expected_dtype:
                print(
                    f"Column '{column}' is not of type {expected_dtype}. Converting to {expected_dtype}."
                )
                try:
                    df[column] = df[column].astype(expected_dtype)
                except ValueError as e:
                    print(
                        f"Error converting column '{column}' to {expected_dtype}: {e}"
                    )

    # 3. Check and remove duplicate rows
    if "states" in df.columns:
        df = df.drop(columns=["states"])

    if df.duplicated().any():
        df.drop_duplicates()

    df = df[df["total_energy_transfered"] > 0]

    return df


@logger.catch(level="ERROR", message="Filtering dataset Error:")
def filtering_sessions(data, column, filter):
    """
    Filtering sessions for value higher than filter .

    :param data: dataframe - Input data to be parsed and validated.
    :param column: str - Column wich will be filtered
    :param filter: int - value that will be filtered
    :return: dataframe - Processed DataFrame with cleaned and validated data.
    """

    df = data.copy()
    # Filter rows where 'duration' > 20, excluding the last row
    filtered_df = df[df[column] > filter]

    # Append the last row back if it was removed by the filter
    if df.iloc[-1][column] <= filter:
        filtered_df = filtered_df.append(df.iloc[-1])

    return filtered_df.reset_index(drop=True)


@logger.catch(level="ERROR", message="Creating Json Error:")
def compute_json(data, file_path):
    """
    Convert DataFrame to JSON and save it to a file.

    :param data: dataframe - DataFrame containing data to be converted to JSON.
    """
    json_data = {}

    data = data.drop_duplicates()
    # Iterate over the DataFrame rows
    for _, row in data.iterrows():
        # Create a key based on the row index (adding 1 to start indexing from 1)
        key = f"user_{row['user_id']}"
        # Extract the row values and convert them to a dictionary
        values = row.to_dict()
        # Add the dictionary to the json_data with the custom key
        json_data[key] = values

    # Convert the dictionary to a JSON string
    target_directory = os.path.join(settings.BASE_PATH, file_path)
    try:
        # Save the JSON data to a file
        logger.debug("Saving JSON file")
        with open(target_directory, "w") as json_file:
            json.dump(json_data, json_file, indent=2)
        logger.debug("JSON file saved successfully at {}".format(target_directory))
    except Exception as e:
        logger.error(f"Error saving JSON file: {e}")


@logger.catch(level="ERROR", message="Error saving json:")
def save_json(json_data, file_path):
    """
    Save json in a particular path.

    :param json_data: json - Input json data that will be saved.
    :param file_path: str - path in wich the json will be saved.
    """
    target_file_path = os.path.join(settings.BASE_PATH, file_path)
    target_directory = os.path.dirname(target_file_path)
    os.makedirs(target_directory, exist_ok=True)

    try:
        # Save the JSON data to a file
        logger.debug("Saving JSON file")
        with open(target_file_path, "w") as json_file:
            json.dump(json_data, json_file, indent=2)
        logger.debug("JSON file saved successfully at {}".format(target_file_path))
    except Exception as e:
        logger.error(f"Error saving JSON file: {e}")


@logger.catch(level="ERROR", message="Creating json forecast:")
def create_json_forecast(
    launch_time,
    duration_value,
    energy_value,
    returnal_date_full,
    model_id,
    evse_id,
    session_id,
    user_id,
    return_probabilities=None
):
    """
    Create a JSON object based on the provided parameters.

    :param launch_time: str - Launch time in ISO 8601 format.
    :param duration_value: float - Forecast duration value.
    :param energy_value: float - Forecast energy consumption value.
    :param returnal_date: str - Returnal date.
    :param model_id: str - Model identifier.
    :param evse_id: str - EVSE identifier.
    :param session_id: str - Session identifier.
    :param user_id: str - User identifier.
    :return: str - JSON string.
    """

    # datetime_returnal = datetime.strptime(returnal_date_full, "%Y-%m-%d %H:%M:%S")
    datetime_returnal=returnal_date_full
    
    logger.debug(f"datime_returnal {datetime_returnal}")

    returnal_date = datetime_returnal.date().isoformat()
    returnal_time = datetime_returnal.time().isoformat()

    # Construct the JSON object
    json_data = {
        "launch_time": launch_time.isoformat(),
        "forecasts": {
            "duration": {
                "value": int(round(duration_value, 0)),
                "unit": "minute",  # Assuming duration unit is fixed to minute
            },
            "energy_consumption": {
                "value": round(energy_value / 1000, 4),
                "unit": "kWh",  # Assuming energy unit is fixed to kWh
            },
            "returnal": {
                "date": returnal_date,
                "time": returnal_time,
                "d0": round(float(return_probabilities.get(0, 0)) * 100, 2),
                "d1": round(float(return_probabilities.get(1, 0)) * 100, 2),
                "d2": round(float(return_probabilities.get(2, 0)) * 100, 2),
                "d3": round(float(return_probabilities.get(3, 0)) * 100, 2),
                "d4": round(float(return_probabilities.get(4, 0)) * 100, 2),
                "d5": round(float(return_probabilities.get(5, 0)) * 100, 2),
                "d6": round(float(return_probabilities.get(6, 0)) * 100, 2),
                "d7": round(float(return_probabilities.get(7, 0)) * 100, 2),
                "d7_plus": round(float(return_probabilities.get(8, 0)) * 100, 2),
                "unit": "%",
            },
        },
        "model_id": model_id,
        "evse_id": evse_id,
        "session_id": session_id,
        "user_id": user_id,
    }

    # Convert the dictionary to a JSON string
    # json_str = json.dumps(json_data, indent=2)
    return json_data


@logger.catch(level="ERROR", message="Error sending sucess email:")
def send_success_email(json_final):
    """
    Send an email with forecsat information of the session.

    :param json_final: json - json with the forecasts of session.
    """

    email_client = EmailClient()
    email_client.ensure_connection()
    email_client.get_email_instance()  # Ensure we're connected before sending
    email_client.compose_and_send(
        subject="EV forecasting session",
        msg=f"Hello,\n\n The session will have the following characteristics \n\n {json_final}",
        signature="GreenDAT.AI team",
    )
    email_client.close()


@logger.catch(level="ERROR", message="Error sending error email:")
def send_error_email(error_email_recipients):
    """
    Send an email with error message.

    :param error_email_recipients: list - recipients of the email with error message.
    """
    email_client = EmailClient()
    email_client.ensure_connection()
    email_client.get_email_instance()
    email_client.update_recipient(new_recipients_debug=error_email_recipients)
    email_client.compose_and_send(
        subject="Error in EV Forecasting Pipeline",
        msg="Hello,\n\n There was an error in the EV forecasting pipeline.",
        signature="GreenDAT.AI team",
        file_to_send="debug_log.log",
        path_file=os.path.join(settings.LOGS_DIR, "debug_log.log"),
    )
    email_client.close()


@logger.catch(level="ERROR", message="Error loading json:")
def load_json(file_path):
    """
    Load a json file.

    :param file_path: str - string wit the path for the json file.
    :return: data: json - json with loaded information
    """
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


@logger.catch(level="ERROR", message="Error getting nested values:")
def get_nested_value(data, path):
    """
    Saves a comparison result to a specified file in JSON format.

    Parameters:
        comparison_result (dict): The result of a comparison that needs to be saved as a JSON file.
        output_file_path (str): The file path where the JSON will be saved.

    Functionality:
        - Serializes the comparison result object into JSON format with an indentation level of 4.
        - If any error occurs during file writing, it logs the error with a custom message.
    """
    try:
        for key in path:
            logger.debug(f"Accessing key: {key}")
            data = data.get(key, {})
            logger.debug(f"Current data at key {key}: {data}")
        return data
    except Exception as e:
        logger.error(f"Error while accessing path {path}: {e}")
        return {}


# @logger.catch(level='ERROR', message='Error saving comparisson jason:')
# def save_json(comparison_result, output_file_path):
#     with open(output_file_path, 'w') as outfile:
#         json.dump(comparison_result, outfile, indent=4)


@logger.catch(level="ERROR", message="Error merging dataframes:")
def merge_dataframes(dataframe_first, dataframe_second):
    """
    Merging dataframes.

    :param dataframe_first: dataframe - First dataframe for merging.
    :param dataframe_second: dataframe - Second dataframe for merging.

    :return: dataframe - merged dataframes.
    """
    df1 = dataframe_first.copy()
    df2 = dataframe_second.copy()
    # Add missing columns to df2 and fill them with NaN
    df2 = df2.reindex(columns=df1.columns, fill_value=np.nan)

    # Append df2 to df1
    df1 = pd.concat([df1, df2], ignore_index=True)

    return df1


@logger.catch(level="ERROR", message="Error checking method:")
def check_user_train(data, n_sessions):
    """
    Get users who have more than 20 sessions.

    :param data: dataframe - DataFrame with previous sessions, containing 'user_id' column.
    :param n_sessions: int - number of sessions
    :return: list - List of user_ids who have more than n sessions.
    """

    try:
        # Get the number of sessions for each user
        user_session_counts = data.groupby("user_id").size()

        # Filter users who have more than 20 sessions
        users_with_more_than_n_sessions = user_session_counts[
            user_session_counts > n_sessions
        ].index.tolist()

        if users_with_more_than_n_sessions:
            logger.info(
                f"Users with more than {n_sessions} sessions found: {len(users_with_more_than_n_sessions)} users."
            )
        else:
            logger.info(f"No users with more than {n_sessions} sessions found.")

        return users_with_more_than_n_sessions

    except Exception as e:
        logger.error(f"Error occurred while checking users: {e}")
        return []


@logger.catch(level="ERROR", message="Error saving model:")
def save_model(
    model_params, path_folder, target_feature, user_id, file_type, define_file
):
    """
    save model in a specific path

    :param model_params: model.
    :param path_folder: str - path in wich we want to save the model .
    :param target_feature: str - target to save.
    :param user_id: str - user id.
    :param file_type: str - type of fily to save.
    :param define_file: str - variable to define the file .
    """
    file_path = os.path.join(
        "files",
        "models",
        f"{path_folder}",
        f"{user_id}_{target_feature}_{define_file}.{file_type}",
    )
    target_file_path = os.path.join(settings.BASE_PATH, file_path)
    target_directory = os.path.dirname(target_file_path)
    os.makedirs(target_directory, exist_ok=True)
    dump(model_params, target_file_path)


@logger.catch(level="ERROR", message="Error loading model:")
def load_model(path_folder, target_feature, user_id):
    """
    Load model.

    :param path_folder: str - path to the folder where the model is saved.
    :param target_feature: str - feature to be saved.
    :param: user_id - int - user.
    """
    # Define the folder path
    models_folder = os.path.join(settings.BASE_PATH, "files", "models", path_folder)

    # Iterate over files in the directory to find the correct file
    for file_name in os.listdir(models_folder):
        if f"{user_id}_{target_feature}" in file_name:
            # Load and return the model
            file_path = os.path.join(models_folder, file_name)
            model_params = load(file_path)
            return model_params

    # If no model is found, return None
    return None


logger.catch(level="ERROR", message="Error saving csv file:")


def saving_csv(dataframe, path_csv, csv_name):
    """
    Function that saves a datframe into a csv in specific path.

    :param dataframe: dataframe - Input data to be saved in csv file
    :param path_csv: str - Path wich the csv file will be saved.
    :param csv_name: str - Csv name.
    """
    if not os.path.exists(path_csv):
        os.makedirs(path_csv)

    # Now you can save the file in the directory
    file_path = os.path.join(path_csv, csv_name)
    try:
        # Save the JSON data to a file
        logger.debug("Saving csv file")
        dataframe.to_csv(file_path, index=False)

    except Exception as e:
        logger.error(f"Error saving csv file: {e}")


@logger.catch(level="ERROR", message="Error sending error email:")
def send_email(email_recipients, file_to_send, path_file):
    """
    Send an email with a file.

    :param email_recipients: list - recipients of the email with error message.
    :param file_to_send: string - file to send to the recipient
    :param path_file: - path for the file that will be sent
    """

    # email_recipients = "josejoaocdias@gmail.com"
    email_client = EmailClient()
    email_client.ensure_connection()
    email_client.get_email_instance()
    email_client.update_recipient(new_recipients_debug=email_recipients)
    email_client.compose_and_send(
        subject="Validation data",
        msg="Hello,\n\n This is the validation file.",
        signature="GreenDAT.AI team",
        file_to_send=file_to_send,
        path_file=os.path.join(path_file, file_to_send),
    )
    email_client.close()


def combine_json_and_dataframe(
    df_ml, df_naive_duration, df_naive_energy, columns=["session_id", "evse_id"]
):
    """
    Combines data from two JSON files and a DataFrame, then saves the result to a CSV file.

    Parameters:
    - json_file1: Path to the first JSON file.
    - json_file2: Path to the second JSON file.
    - df: The DataFrame to combine with the JSON data.
    - output_csv: The file path for the output CSV file.

    Returns:
    - None: Saves the combined data as a CSV.
    """
    try:

        # Combine all DataFrames (the two from JSON files and the provided DataFrame)
        result_df = pd.merge(
            df_naive_duration, df_naive_energy, how="inner", on=columns
        )
        result_df_final = pd.merge(result_df, df_ml, how="left", on=columns)

        # Save the combined DataFrame to a CSV file
        return result_df_final

    except Exception as e:
        print(f"An error occurred: {e}")
