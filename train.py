from app.forecast.core import (
    train_model,
    calculation_duration,
    compute_tx_lag_energy,
    train_model_day_next_plugin
)
from app.forecast.processing import (
    get_ev_session_all_sessions,
    parse_data,
    parse_validate_data,
    check_user_train,
    calculate_mad_md,
)
from loguru import logger
from datetime import datetime

logger.info("Starting training script...")

# Extracting session information for all the users
current_datetime = datetime.now()

# Note that you can also ensure that the models are trained only with data until a specific date.
# This will condition the requests to the API to return data until that date.
# Uncomment the line below to set a specific date for training
# current_datetime = datetime.strptime("2025-01-01 9:00", "%Y-%m-%d %H:%M")
logger.debug(f"Data for training until:{current_datetime}")

logger.info("Retrieving EV session data ...")
response = get_ev_session_all_sessions(
    start_datetime=None, user_id=None, end_datetime=current_datetime
)

logger.success("Data was retrieved successfully")
logger.info("Parsing Data...")
# Converting and validating data from json to dataframe
data = parse_data(data=response)
logger.debug(f"data session: \n {data}")
logger.success("Data Parsed successfully")
logger.info("Validating Data...")
data = parse_validate_data(data=data)
logger.success("Data validated")
logger.info("Duration target calculating..")
# Computing sessions information duration
df_all = calculation_duration(dataframe=data)
logger.debug(df_all[df_all["duration"] < 0])
logger.success("Duration was computed successfully")
logger.info("Time until next session target calculating...")
logger.success("Time until next session targat was computed successfully")
logger.info("Filtering energy and duration")
logger.success("Filtering was a success")
logger.info("training model for each user...")
df_all = compute_tx_lag_energy(dataframe=df_all)
logger.debug(f"Type of column 'A': {df_all['duration'].dtype}")

# Data pre-processing to elimate outliers
grouped = df_all.groupby("user_id")[["user_id", "duration"]].apply(
    calculate_mad_md, target="duration"
)
logger.debug(f"grouped dataframe:{grouped.head(5)}")
media = grouped.apply(lambda x: x[0])
mad = grouped.apply(lambda x: x[1])
logger.debug(f"media:{media.head(5)}")
logger.debug(f"mad:{mad.head(5)}")
df_all["media"] = df_all["user_id"].map(media)
df_all["mad"] = df_all["user_id"].map(mad)
c = 1.4826 * 12
df_all["threshold_max"] = df_all["media"] + c * df_all["mad"]
df_all["threshold_min"] = df_all["media"] - c * df_all["mad"]

# Step 7: Calculate the max value for column 'B' for each group in column 'A'
max_values_per_user = df_all.groupby("user_id")["duration"].max()
min_values_per_user = df_all.groupby("user_id")["duration"].min()

logger.debug(f"before removing outliers: {df_all.info()}")

# Step 8: Create the 'is_outlier' column based on whether the max value exceeds the threshold
df_all["is_outlier"] = df_all["duration"] > df_all["threshold_max"]


df_all_final = df_all[~df_all["is_outlier"]].copy()


logger.debug(df_all.head(10))
logger.success("MAD sucess")

df_all_final = df_all_final[df_all["duration"] > 20]
df_all_final = df_all_final[df_all_final["total_energy_transfered"] > 10]
logger.debug(f"training database: {df_all_final.info()}")
# Check which users need to be trainned
users_to_train = check_user_train(data=df_all_final, n_sessions=52)
users_to_train = [17]
# For each user train duration,energy and next plug-in model
for user in users_to_train:
    # For each user in users with more than 52 sessions, train a regression model:
    # - for session duration forecasting
    # - for total energy transferred forecasting
    # - for next plug-in time forecasting
    logger.info(f"user: {user}")
    logger.info("training model for duration forecasting")
    # Training session duration forecasting model for particular user
    train_model(
        dataframe=df_all_final, target_feature="duration", user_id=user
    )
    logger.success("Duration model was trained successfully")

    logger.info("training model for total energy transferred")
    # # Training energy model for particular user
    train_model(
        dataframe=df_all_final, target_feature="total_energy_transfered", user_id=user
    )
    logger.info("Total energy transfered model was trained successfully")

    # logger.info("training model for next session")
    # Training next plug_in model for particular user
    train_model(
        dataframe=df_all_final, target_feature="time_next_plug_in", user_id=user
    )
    # logger.info("Next session model was trained successfully")
    logger.info("training model for next session hour")
    train_model(
        dataframe=df_all_final, target_feature="hour_minute", user_id=user
    )
    logger.info("Next session hour model was trained successfully")

    logger.info("training model for next day plug in")
    train_model_day_next_plugin(
        dataframe=df_all_final, user_id=user
    )
    logger.info("Next day plug in model was trained successfully")
