"""
This script runs a local data processing pipeline for an CSV file provided by the user.
It loads the file, processes the data, and stores the results locally.

The file must have the following structure:
session_id              - Unique identifier for each charging session.
evse_id                 - Identifier for the EVSE (Electric Vehicle Supply Equipment) used in the session.
user_id                 - Identifier for the user who initiated the charging session.
start_time              - Timestamp indicating when the charging session began (format: YYYY-MM-DDTHH:MM:SS, 2024-08-12T17:58:09).
end_time                - Timestamp indicating when the charging session ended (format: YYYY-MM-DDTHH:MM:SS,2024-08-12T17:58:09).
total_energy_transfered - Total amount of energy transferred during the session,wH.

No internet access or external dependencies are required on runtime.

The file should be inside of the files/local/data folder, named as username_data.csv, for example, 17_data.csv

The result will be saved inside the fodler files/local/results.
To run this script: 
    intall python 3.12.3:
        winget install Python.Python.3.12
    install requirements.txt:
        cd ev-session-forecacaster
        pip install -r requirements.txt
    Go the root folder of the project and run:
        python -m compute_forecast_local

"""
import os
import datetime as dt
from loguru import logger
from app.forecast.processing import (
    parse_validate_data,
    create_json_forecast,
    merge_dataframes,
    filtering_sessions,
    check_user_train,
    compute_json
)
from app.forecast.core import (
    naive_models_forecast,
    naive_models_returnal,
    calculation_duration,
    calculation_delta_startime,
    calculation_time_next_plug_in,
    forecast,
    model_exists,
    compute_next_session,
    filtered_sorted_return_day_offsets_for_last_row,
    rowwise_most_common_return_weekday,
    compute_next_hour,
    day_next_session,
    probability_day_next_plug_in,
    analyze_charging_patterns,
    statistical_analysis,

)
import pandas as pd
import app.conf.settings as settings


def compute_forecast_local(user_id, evse_id,dataframe, session_id=None, launch_time=None):

    """
    Compute forecast for EV charging session

    :return: (dict) - forecast results
    """
    # Session information
    logger.info("-" * 79)
    logger.info("Forecasting ...")
    logger.info(
        f"Call details:"
        f"\n\t- User ID: {user_id}"
        f"\n\t- Session ID: {session_id}"
        f"\n\t- EVSE ID: {evse_id}"
    )

    logger.info("Getting EV session data...")
 
    launch_time_ = launch_time
    # Convert launch_time to srt
    launch_time_str = launch_time_.strftime("%Y-%m-%dT%H:%M:%S")
    logger.info("Ev session data was retrieved")

    # Convert json data into dataframe
    logger.info("Parsing Data...")
    data = dataframe.copy()
    # Checking if launch time is inferior to the minimum start date
    if launch_time_str < data["start_time"].min():
        raise Exception(
            f"launch_time '{launch_time_}' is inferior to the first session for the user."
        )
    else:
        data = data[data["start_time"] < launch_time_str]
        user_session = check_user_train(data=data, n_sessions=52)
    logger.info(user_session is not None)
    logger.success("Data Parsed successfully")
    logger.info("Validating Data...")
    # Validating data
    data = parse_validate_data(data=data)

    logger.success("Data validated")
    # Variable that will be used to determine which model will be used (naive, ml)
    check_model = False

    # Data analysis for user
    logger.info('Starting statistical analysis...')
    data_analyzed = statistical_analysis(data)
    logger.success('Statistical analysis ended')

    logger.info('Computing json with statistical analysis...')

    json_file_path = os.path.join(
        settings.BASE_PATH,
        "files",
        "statiscal_analysis"
    )
    os.makedirs(json_file_path, exist_ok=True)
    json_file_path = os.path.join(json_file_path, f"{user_id}.json")

    compute_json(data_analyzed, file_path=json_file_path)
    logger.success('Done')

    logger.info('Number of Sessions')
    # user_data_analysis = analyze_charging_patterns(user_id, file_path=json_file_path)
    # Note that you can use this user_data_analysis for further analysis if needed
    # e.g., to define if naive models should be executed instead of the ML approach
    # right now, we are calculating both approaches.

    logger.info("Current session data...")
    launch_time_str = launch_time_.strftime("%Y-%m-%dT%H:%M:%S")
    logger.debug(f"value of start_time:{launch_time_str}")

    # creation of the current session
    data_current_session = {
        "evse_id": evse_id,
        "user_id": user_id,
        "session_id": session_id,
        "start_time": launch_time_str,
    }
    df_current_session = pd.DataFrame([data_current_session])
   
    logger.debug("Merging data of session with historic data ...")
    df_all = merge_dataframes(dataframe_first=data, dataframe_second=df_current_session)
    logger.success("Merging data of session with historic data ... Ok!")

    # Calculation of information regarding session
    logger.debug("Computing duration of session ...")
    df_all = calculation_duration(dataframe=df_all)
    logger.success("Computing duration of session ... Ok!")

    logger.info("Calculating time since previous session ...")
    df_all = calculation_delta_startime(dataframe=df_all)
    df_all = calculation_time_next_plug_in(dataframe=df_all)
    prob_day_next_plug_in = probability_day_next_plug_in(dataframe=df_all)
    df_all = day_next_session(data=df_all)
    df_all = compute_next_hour(dataframe=df_all,column_date='start_datetime')
    df_all = rowwise_most_common_return_weekday(df_all)
    logger.debug("Calculating time since previous session ... Ok!")
    return_day_list = filtered_sorted_return_day_offsets_for_last_row(df=df_all)
    # filtering session with less than 20 minutes
    df_all = filtering_sessions(data=df_all, column="duration", filter=20)
    json_final_ml = None
    json_final_naive = None

    # Check if a pre-trained ML model exists (i.e., created with the train.py script)
    # and call it
    # -- Duration forecast:
    logger.info("Checking if the model exists for target duration...")
    if (
        model_exists(
            path_folder="ML models", target_feature="duration", user_id=user_id
        )
    ) and (user_session is not None):
        check_model = True
        logger.debug("Checking if the model exists for target duration... Ok!")
        logger.info("Computing duration forecast ...")
        # Duration forecast calculation
        ml_duration_value = forecast(
            dataframe=df_all, target_feature="duration", path_folder="ML models"
        )
        logger.success("Computing duration forecast ... Ok!")

    # -- Energy consumption forecast:
    logger.info("Checking if the model exists for target energy consumption ...")
    if model_exists(
        path_folder="ML models",
        target_feature="total_energy_transfered",
        user_id=user_id,
    ) and (user_session is not None):
        logger.success("Checking if the model exists for target energy consumption ... Ok!")
        logger.info("Computing energy consumption forecast ...")
        # Energy forecast calculation
        ml_energy_value = forecast(
            dataframe=df_all,
            target_feature="total_energy_transfered",
            path_folder="ML models",
        )
        logger.success("Computing energy consumption forecast ... Ok!")

    # -- Returnal forecast:
    logger.info("Checking if the model exists for target returnal date ...")
    if model_exists(
        path_folder="ML models", target_feature="hour_minute", user_id=user_id
    ) and (user_session is not None):
        logger.success("Checking if the model exists for target returnal date ...")
        logger.info("Computing returnal date forecast ...")
        # Time until next forecast calculation
        ml_returnal_value = forecast(
            dataframe=df_all,
            target_feature="hour_minute",
            path_folder="ML models",
            return_day_list=return_day_list,
            launch_time=launch_time_,
        )
        logger.success("Computing returnal date forecast ... Ok!")
        # logger.info("Computing next session...")
        # Computation of next forecast calculation
        # next_session = compute_next_session(
        #     start_time=launch_time_str, forecast_time=ml_returnal_value
        # )
        # logger.debug(f"next_session{ml_returnal_value[0]["Next Session - Forecast ML"]}")
        # logger.success("Next session was computed successfully")

    # -- Next day plugin:
    if model_exists(
        path_folder="ML models", target_feature="day_next_plug_in", user_id=user_id
    ) and (user_session is not None):
        logger.success("Model for returnal date exists")
        logger.info("Returnal date forecasting...")
        # Time untill next forecast calculation
        ml_returnal_value_next_day = forecast(
            dataframe=df_all,
            target_feature="day_next_plug_in",
            path_folder="ML models",
        )

    if check_model:
        logger.info("Json creation for ML forecast...")
        logger.debug(f"launch_time:{launch_time_}")
        logger.debug(f"duration:{ml_duration_value}")
        logger.debug(f"energy:{ml_energy_value}")
        # logger.debug(f"plug-in:{ml_returnal_value[0]["Next Session - Forecast ML"]}")
        logger.debug(f"session:{session_id}")

        # Json creation with session and forecast information
        json_final_ml = create_json_forecast(
            launch_time=launch_time_,
            duration_value=ml_duration_value,
            energy_value=ml_energy_value,
            returnal_date_full=ml_returnal_value[0]["Next Session - Forecast ML"],
            model_id="ml",
            evse_id=evse_id,
            session_id=session_id,
            user_id=user_id,
            return_probabilities=ml_returnal_value_next_day[0],
        )

        logger.success("Json for machine learning models forecast was created")
        logger.info("Creation of session id")
        # session_id_ = (
        #     launch_time_.strftime("%Y%m%d%H%M%S") if session_id is None else session_id
        # )
        logger.success("session id was calculated...")
        logger.info("file path creation...")
        # Saving session and forecasts
        # file_path = os.path.join("files", "results", "ml", f"{session_id_}.json")
        logger.success("path for json file was created")

        logger.info("saving json ...")
        # save_json(json_data=json_final_ml, file_path=file_path)
        logger.success("json was saved")

    logger.info("Computing naive models for forecast...")
    logger.debug(f"data for naive models: {df_all.info()}")
    logger.info("Computing naive models for duration...")
    # Computing Duration Naive Forecast
    naive_duration_value = naive_models_forecast(
        data=df_all, target="duration", time=launch_time_
    )
    logger.success("Duration Forecast computed successfully")
    logger.info("Computing naive models for total energy transfered...")
    # Computing Energy Naive Forecast
    naive_energy_value = naive_models_forecast(
        data=df_all, target="total_energy_transfered", time=launch_time_
    )
    logger.success("Total energy transfered Forecast computed successfully")
    logger.info("Computing naive models for Returnal...")
    # # Computing Next Plug-in Naive Forecast
    # naive_returnal_value = naive_models_returnal(df_all, launch_time_).strftime(
    #     "%Y-%m-%d %H:%M:%S"
    # )  # noqa
    naive_hour_value = naive_models_forecast(
        data=df_all, target="hour_minute", time=launch_time_
    )
    
    hour_value=int(naive_hour_value)
    minutes_value = int(round((naive_hour_value - hour_value) * 60))
    if minutes_value  == 60:
        minutes_value = 0
    forecast_time_str = f"{hour_value:02d}:{minutes_value:02d}"
    launch_time_start = pd.to_datetime(launch_time_str)
    start_date = launch_time_start.date()
    if not return_day_list:
        return_day=1
    else:
        return_day=return_day_list[0]
    return_date = start_date + dt.timedelta(days=return_day)

    forecast_datetime_str = f"{return_date.strftime('%d/%m/%Y')} {forecast_time_str}"

    # Final datetime
    full_forecast_datetime = pd.to_datetime(forecast_datetime_str, format="%d/%m/%Y %H:%M")

    logger.success("Returnal Forecast computed successfully")
    logger.info("Json Creation for naive forecast...")
    # Creating json for naive forecast
    json_final_naive = create_json_forecast(
        launch_time=launch_time_,
        duration_value=naive_duration_value,
        energy_value=naive_energy_value,
        returnal_date_full=full_forecast_datetime,
        model_id="naive",
        evse_id=evse_id,
        session_id=session_id,
        user_id=user_id,
        return_probabilities=prob_day_next_plug_in
    )
    logger.success("Json created successfully")

    logger.info("Saving Json naive...")

    logger.success("Json saved.")
    logger.info("Checking which json to return...")

    if check_model:
        logger.info("Returning ML json")
        logger.debug(json_final_ml)
        ml_df = pd.DataFrame([{
            "launch_time": json_final_ml["launch_time"],
            "evse_id": json_final_ml["evse_id"],
            "user_id": json_final_ml["user_id"],
            "duration": json_final_ml["forecasts"]["duration"]["value"],
            "total_energy_transferred": json_final_ml["forecasts"]["energy_consumption"]["value"],
            "model": json_final_ml["model_id"],
            "returnal_date": json_final_ml["forecasts"]["returnal"]["date"],
            "returnal_time": json_final_ml["forecasts"]["returnal"]["time"],
            "d0": json_final_ml["forecasts"]["returnal"]["d0"],
            "d1": json_final_ml["forecasts"]["returnal"]["d1"],
            "d2": json_final_ml["forecasts"]["returnal"]["d2"],
            "d3": json_final_ml["forecasts"]["returnal"]["d3"],
            "d4": json_final_ml["forecasts"]["returnal"]["d4"],
            "d5": json_final_ml["forecasts"]["returnal"]["d5"],
            "d6": json_final_ml["forecasts"]["returnal"]["d6"],
            "d7": json_final_ml["forecasts"]["returnal"]["d7"],
            "d7_plus": json_final_ml["forecasts"]["returnal"]["d7_plus"]
        }])
        return ml_df
    else:
        logger.info("Returning Naive json")
        naive_df = pd.DataFrame([{
            "launch_time": json_final_naive["launch_time"],
            "evse_id": json_final_naive["evse_id"],
            "user_id": json_final_naive["user_id"],
            "duration": json_final_naive["forecasts"]["duration"]["value"],
            "total_energy_transferred": json_final_naive["forecasts"]["energy_consumption"]["value"],
            "model": json_final_naive["model_id"],
            "returnal_date": json_final_naive["forecasts"]["returnal"]["date"],
            "returnal_time": json_final_naive["forecasts"]["returnal"]["time"],
            "d0": json_final_naive["forecasts"]["returnal"]["d0"],
            "d1": json_final_naive["forecasts"]["returnal"]["d1"],
            "d2": json_final_naive["forecasts"]["returnal"]["d2"],
            "d3": json_final_naive["forecasts"]["returnal"]["d3"],
            "d4": json_final_naive["forecasts"]["returnal"]["d4"],
            "d5": json_final_naive["forecasts"]["returnal"]["d5"],
            "d6": json_final_naive["forecasts"]["returnal"]["d6"],
            "d7": json_final_naive["forecasts"]["returnal"]["d7"],
            "d7_plus": json_final_naive["forecasts"]["returnal"]["d7_plus"]
        }])
        logger.debug("\n{}", naive_df.to_string(index=False))
        return naive_df


folder_path_data = os.path.join(
    settings.BASE_PATH,
    "files",
    "local",
    "data",
    )

results_folder = os.path.join(
    settings.BASE_PATH,
    "files",
    "local",
    "results",
    )

# Define a default date range
default_start_datetime = dt.datetime(2025, 1, 1, 9, 0)
default_end_datetime = dt.datetime(2025, 3, 1, 9, 0)

# Optional: Map filename prefixes to custom date ranges
custom_date_ranges = {
    '17': (dt.datetime(2025, 1, 1, 9, 0), dt.datetime(2025, 3, 1, 9, 0)),
}

for filename in os.listdir(folder_path_data):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path_data, filename)
        # Read Excel
        data = pd.read_csv(file_path, sep=',', decimal='.')
        logger.info(f"Read {filename} with shape {data.shape}")

        data['total_energy_transfered'] = pd.to_numeric(data['total_energy_transfered'], errors='coerce').fillna(0).astype(int)
        # Convert 'start_time' to datetime
        data['start_time_datetime'] = pd.to_datetime(data['start_time'])

        # Extract prefix (e.g., "17" from "17_data.csv")
        prefix = filename.split('_')[0]

        # Use custom date range if available, else use default
        start_datetime_first_date, start_datetime_second_date = custom_date_ranges.get(
            prefix, (default_start_datetime, default_end_datetime)
        )

        # Filter start_times within the range
        filtered_dates = data[
            (data['start_time_datetime'] >= start_datetime_first_date) &
            (data['start_time_datetime'] <= start_datetime_second_date)
        ]['start_time'].sort_values().unique()
        user_id = data['user_id'].unique()[0]
        # Collect results in a list
        results_df = pd.DataFrame()
        data=data.drop('start_time_datetime',axis=1)
        # Loop through each unique filtered start_time
        for launch_time in filtered_dates:
            launch_data = data[data['start_time'] == launch_time]

            # Extract evse_id for that specific launch time
            evse_id = launch_data['evse_id'].unique()[0] if 'evse_id' in launch_data else "unknown_evse_id"

            result_row_df=compute_forecast_local(
                user_id=user_id,       
                evse_id=f"{evse_id}",         
                dataframe=data,
                session_id=None,
                launch_time=dt.datetime.fromisoformat(launch_time)
            )
            results_df = pd.concat([results_df, result_row_df], ignore_index=True)

        os.makedirs(results_folder, exist_ok=True)

        # Format the dates for the filename (e.g., 20250101_0900)
        start_str = start_datetime_first_date.strftime('%Y%m%d_%H%M')
        end_str = start_datetime_second_date.strftime('%Y%m%d_%H%M')

        # Create the result filename with dates
        result_filename = f"{prefix}_result_{start_str}_to_{end_str}.csv"
        result_path = os.path.join(results_folder, result_filename)

        # Save the report to CSV (merge observed values):
        results_df["launch_time"] = pd.to_datetime(results_df["launch_time"], format="%Y-%m-%dT%H:%M:%S")
        data["start_time"] = pd.to_datetime(data["start_time"], format="%Y-%m-%dT%H:%M:%S")
        data["end_time"] = pd.to_datetime(data["end_time"], format="%Y-%m-%dT%H:%M:%S")
        data["duration_real"] = (data["end_time"] - data["start_time"]).dt.total_seconds() / 60
        
        results_df = results_df.set_index("launch_time").join(data[["start_time", "end_time", "total_energy_transfered", "duration_real"]].set_index("start_time"))
        results_df.to_csv(result_path, sep=';', decimal='.', index=False)
        
        logger.info(f"Saved: {result_filename}")