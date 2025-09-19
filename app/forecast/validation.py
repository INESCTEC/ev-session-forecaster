"""
This script validates both the naive and the machine learning approach, saving the results into a csv in files/results/validation.
"""
import os
import datetime as dt
from loguru import logger
import app.conf.settings as settings
from app.forecast.processing import (
    get_ev_session_data,
    save_json,
    parse_data,
    parse_validate_data,
    create_json_forecast,
    merge_dataframes,
    filtering_sessions,
    check_user_train,
    get_ev_session_all_sessions,
    saving_csv,
    send_email
)
from app.forecast.core import (
    naive_models_forecast,
    calculation_duration,
    calculation_delta_startime,
    calculation_time_next_plug_in,
    forecast,
    model_exists,
    probability_day_next_plug_in,
    day_next_session,
    compute_next_hour,
    rowwise_most_common_return_weekday,
    filtered_sorted_return_day_offsets_for_last_row
)
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta


def compute_forecast_validation(user_id, evse_id, session_id=None, launch_time=None):

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
    response = get_ev_session_data(user_id=user_id)
    logger.info(response)
    # Check if the user exists in the database
    if len(response) == 0:
        raise Exception(f"User ID '{user_id}' does not exists in database.")
    # If a session id is not declared, uses launch time specified by user
    if session_id is not None:
        current_session = [x for x in response if x["session_id"] == session_id and x["evse_id"]==evse_id]
        if len(current_session) == 0:
            raise Exception(
                f"Session ID '{session_id}' was not found in user session data."
            )
        launch_time_ = current_session[0]["start_time"]
        launch_time_ = dt.datetime.strptime(launch_time_, "%Y-%m-%dT%H:%M:%S")
    else:
        launch_time_ = launch_time

    # Convert launch_time to srt
    launch_time_str = launch_time_.strftime("%Y-%m-%dT%H:%M:%S")
    logger.info(type(launch_time_))
    logger.info("Ev session data was retrieved")

    # Convert json data into dataframe
    logger.info("Parsing Data...")
    data = parse_data(data=response)
    logger.info(type(launch_time_str))

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
    # logger.info('Starting statistical analysis...')
    # data_analyzed = statistical_analysis(data)
    # logger.success('Statistical analysis ended')

    # logger.info('Computing json with statistical analysis...')
    # compute_json(data_analyzed)
    # logger.success('Done')

    # logger.info('Number of Sessions')
    # bool_=analyze_charging_patterns(user_id)

    logger.info("Current session data...")
    launch_time_str = launch_time_.strftime("%Y-%m-%dT%H:%M:%S")
    logger.debug(f"value os start_time:{launch_time_str}")

    # creation of the current session
    data_current_session = {
        "evse_id": evse_id,
        "user_id": user_id,
        "session_id": session_id,
        "start_time": launch_time_str,
    }
    df_current_session = pd.DataFrame([data_current_session])
    logger.success("Current session data was successfull")
    # data=data[data['start_time']<launch_time_str]
    logger.info("Merging data of session with historic data ...")
    df_all = merge_dataframes(dataframe_first=data, dataframe_second=df_current_session)
    logger.success("Merging data was successfull")

    # Calculation of information regarding session
    logger.info("Computing duration of session ...")
    df_all = calculation_duration(dataframe=df_all)
    logger.success("Duration was successfully calculated")

    logger.info("Time since last session calculating ...")
    df_all = calculation_delta_startime(dataframe=df_all)
    df_all = calculation_time_next_plug_in(dataframe=df_all)
    prob_day_next_plug_in = probability_day_next_plug_in(dataframe=df_all)
    df_all = day_next_session(data=df_all)
    df_all = compute_next_hour(dataframe=df_all,column_date='start_datetime')
    df_all = rowwise_most_common_return_weekday(df_all)
    logger.success("Time since last session was successfully calculated...")
    return_day_list = filtered_sorted_return_day_offsets_for_last_row(df=df_all)
    logger.info("Checking if the model exists for target duration...")
    # filtering session with less than 20 minutes
    df_all = filtering_sessions(data=df_all, column="duration", filter=20)
    json_final_ml = None
    json_final_naive = None

    # Checking if the model for machine learning forecast exists
    if (
        model_exists(
            path_folder="ML models", target_feature="duration", user_id=user_id
        )
    ) and (user_session is not None):
        check_model = True
        logger.success("Model for duration exists")
        logger.info("duration forecast...")
        # Duration forecast calculation
        ml_duration_value = forecast(
            dataframe=df_all, target_feature="duration", path_folder="ML models"
        )
        logger.success("Duration forecast was computed")

    if model_exists(
        path_folder="ML models",
        target_feature="total_energy_transfered",
        user_id=user_id,
    ) and (user_session is not None):
        logger.success("Model for total energy transfered exists")
        logger.info("Total energy transfered forecast...")
        # Energy forecast calculation
        ml_energy_value = forecast(
            dataframe=df_all,
            target_feature="total_energy_transfered",
            path_folder="ML models",
        )
        logger.success("Total energy forecast was computed")

    if model_exists(
        path_folder="ML models", target_feature="hour_minute", user_id=user_id
    ) and (user_session is not None):
        logger.success("Model for returnal date exists")
        logger.info("Returnal date forecasting...")
        # Time untill next forecast calculation
        ml_returnal_value = forecast(
            dataframe=df_all,
            target_feature="hour_minute",
            path_folder="ML models",
        )
        logger.success("Returnal forecast was computed")
        logger.info("Computing next session...")
        # Computation of next forecast calculation
        # next_session = compute_next_session(
        #     start_time=launch_time_str, forecast_time=ml_returnal_value
        # )
        logger.debug(f"next_session{ml_returnal_value[0]["Next Session - Forecast ML"]}")
        logger.success("Next session was computed successfully")
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
        logger.debug(f"plug-in:{ml_returnal_value[0]["Next Session - Forecast ML"]}")
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

    # Checking to see if the return should be Machine Learning Forecast or Naive Forecast
    # Safe extraction from json_final_naive
    duration_naive = json_final_naive.get("forecasts", {}).get("duration", {}).get("value") if json_final_naive else None
    energy_naive = json_final_naive.get("forecasts", {}).get("energy_consumption", {}).get("value") if json_final_naive else None
    returnal_date_naive = json_final_naive.get("forecasts", {}).get("returnal", {}).get("date") if json_final_naive else None
    returnal_time_naive = json_final_naive.get("forecasts", {}).get("returnal", {}).get("time") if json_final_naive else None
    launch_time = json_final_naive.get("launch_time") if json_final_naive else None

    # Safe extraction from json_final_ml
    duration_ml = json_final_ml.get("forecasts", {}).get("duration", {}).get("value") if json_final_ml else None
    energy_ml = json_final_ml.get("forecasts", {}).get("energy_consumption", {}).get("value") if json_final_ml else None
    returnal_date_ml = json_final_ml.get("forecasts", {}).get("returnal", {}).get("date") if json_final_ml else None
    returnal_time_ml = json_final_ml.get("forecasts", {}).get("returnal", {}).get("time") if json_final_ml else None

    # Create the DataFrame
    final_df = pd.DataFrame([{
        "launch_time": launch_time,
        "evse_id": evse_id,
        "user_id": user_id,
        "session_id": session_id,
        "duration_naive_forecast": duration_naive,
        "duration_ml_forecast": duration_ml,
        "total_energy_transferred_naive_forecast": energy_naive,
        "total_energy_transferred_ml_forecast": energy_ml,
        "returnal_date_naive_forecast": returnal_date_naive,
        "returnal_time_naive_forecast": returnal_time_naive,
        "returnal_date_ml_forecast": returnal_date_ml,
        "returnal_time_ml_forecast": returnal_time_ml,
    }])

    # Combine date + time into datetime columns
    final_df["returnal_datetime_naive_forecast"] = final_df.apply(
        lambda row: datetime.combine(
            pd.to_datetime(row["returnal_date_naive_forecast"]).date(),
            pd.to_datetime(row["returnal_time_naive_forecast"]).time()
        ) if pd.notnull(row["returnal_date_naive_forecast"]) and pd.notnull(row["returnal_time_naive_forecast"]) else pd.NaT,
        axis=1
    )

    final_df["returnal_datetime_ml_forecast"] = final_df.apply(
        lambda row: datetime.combine(
            pd.to_datetime(row["returnal_date_ml_forecast"]).date(),
            pd.to_datetime(row["returnal_time_ml_forecast"]).time()
        ) if pd.notnull(row["returnal_date_ml_forecast"]) and pd.notnull(row["returnal_time_ml_forecast"]) else pd.NaT,
        axis=1
    )

    # Drop intermediate date/time fields
    final_df = final_df.drop([
        "returnal_date_naive_forecast",
        "returnal_time_naive_forecast",
        "returnal_date_ml_forecast",
        "returnal_time_ml_forecast"
    ], axis=1)

    return final_df


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


current_datetime = datetime.now()

start_date_validation = current_datetime - relativedelta(days=29, months=2)
start_date_validation = start_date_validation.strftime("%Y-%m-%dT%H:%M:%S")
logger.remove()

logger.info("Getting EV session data...")
response = get_ev_session_all_sessions(start_datetime=start_date_validation)

logger.info(response)

# Convert json data into dataframe
logger.info("Parsing Data...")
data = parse_data(data=response)
logger.success("Data Parsed successfully")
logger.info("Validating Data...")
# Validating data
data = parse_validate_data(data=data)
data = calculation_duration(dataframe=data)
data = filtering_sessions(data=data, column="duration", filter=20)


unique_user_ids_list = data["user_id"].unique().tolist()



# Collect results in a list
results_df = pd.DataFrame()
# Loop through each unique filtered 
for user_id in unique_user_ids_list:
        data_user = (
            data[data["user_id"] == user_id]
            .copy()
            .reset_index(drop=True)
            )    
        data_user.sort_values(by='start_time',ascending=True)
        data_user=data_user.drop(['duration'],axis=1)
        for i, row in data_user.iterrows():
            # start_time = row['start_datetime']
            evse_id = row["evse_id"]
            session_id = row["session_id"]


            result_row_df=compute_forecast_validation(
                user_id=user_id,       
                evse_id=f"{evse_id}",         
                session_id=f"{session_id}"
            )
            results_df = pd.concat([results_df, result_row_df], ignore_index=True)



final_df_merge = pd.merge(
    results_df,
    data[['session_id','evse_id','user_id','start_datetime','duration','total_energy_transfered']],
    on=["evse_id", "session_id", "user_id"],
    how="inner"  # or "left", "right", "outer" depending on your use case
)
final_df_merge=final_df_merge.sort_values(by='start_datetime',ascending=True)
final_df_merge['start_datetime_target']=final_df_merge['start_datetime'].shift(-1)
logger.success("Dataframe Create successfully")
output_csv_dir = os.path.join(settings.BASE_PATH, "files", "results", "validation")
logger.info("Saving csv ...")
previous_month_csv = current_datetime.strftime("%Y-%m-%d")
saving_csv(
    dataframe=final_df_merge,
    path_csv=output_csv_dir,
    csv_name=f"{previous_month_csv}_validation.csv",
)
logger.success("Csv saved successfully.")
logger.info("Sending email...")
send_email(
    email_recipients="jose.j.dias@inesctec.pt",
    file_to_send=f"{previous_month_csv}_validation.csv",
    path_file=output_csv_dir,
)
logger.success("Email was sent.")


# Create the result filename with dates
