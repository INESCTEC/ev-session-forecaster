import pandas as pd
from loguru import logger
import app.conf.settings as settings
from app.forecast.processing import get_nested_value, save_json, save_model, load_model
import os
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
import numpy as np
from datetime import timedelta, datetime
from dateutil.relativedelta import relativedelta
from imblearn.over_sampling import SMOTE,RandomOverSampler
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from collections import Counter
from bayes_opt import BayesianOptimization
from sklearn.feature_selection import RFE


def is_weekday(day_of_week):
    """
    :param day_of_week: featured variable from 0 to 6. Saturday is 5; Sunday is 6.
    :return: 1 for weekday, 0 for weekend
    """
    if (day_of_week == 5) | (day_of_week == 6):
        return 0
    else:
        return 1


@logger.catch(level="ERROR", message="Error calculating includes_weekend")
def includes_weekend(start_datetime, end_datetime=None):
    """
    Function return 0 or 1 if the the session occurs between a weekend.

    :param data: dataframe - Input data to be parsed and validated.
    :return: bool - Return 0 is the start date and end date has a weekedn between them or 1 if not
    """
    if end_datetime is None:
        # If end_datetime is None, treat the day as weekend if start_datetime is after 19:00
        if start_datetime.hour >= 19:
            return 0
        else:
            return 1

    # Generate range of dates if end_datetime is provided
    all_dates = pd.date_range(start=start_datetime, end=end_datetime, freq="D")
    if any(date.weekday() >= 5 for date in all_dates):  # 5 is Saturday, 6 is Sunday
        return 0
    else:
        return 1


def tri_tariff(start_datetime):
    """
    Function that calculates the tri_tariff based on a start_datetime.

    :param start_datetime: datetime - start datetime used to calculate tri tariff.
    :return: bool - Return 0,1 or 2 according to the the date
    """
    logger.debug(start_datetime)
    current_year = start_datetime.year
    winter_start = pd.Timestamp("{}-12-01 00:00:00".format(current_year))
    winter_end = pd.Timestamp(
        "{}-02-28 23:59:59".format(
            current_year + 1 if start_datetime.month >= 12 else current_year
        )
    )
    current_hour = start_datetime.hour
    current_minute = start_datetime.minute

    if (start_datetime >= winter_start) & (
        start_datetime <= winter_end
    ):  # Winter scheme
        logger.debug("winter start")
        if (current_hour >= 0) & (
            (current_hour < 9) or (current_hour == 9 and current_minute == 0)
        ):
            return 0
        elif ((current_hour >= 9) & (current_hour < 10)) or (
            (current_hour == 10) & (current_minute < 30)
        ):
            return 2
        elif ((current_hour >= 10) & (current_hour < 12)) or (
            current_hour == 12 and current_minute == 0
        ):
            return 1
        elif (current_hour >= 12) & (
            (current_hour < 18) or ((current_hour == 18) & (current_minute < 30))
        ):
            return 0
        elif ((current_hour >= 18) & (current_hour < 21)) or (
            current_hour == 21 and current_minute == 0
        ):
            return 1
        else:
            return 2
    else:  # Summer scheme
        if (current_hour >= 0) & (
            (current_hour < 9) or (current_hour == 9 and current_minute == 0)
        ):
            return 0
        elif ((current_hour >= 9) & (current_hour < 10)) or (
            (current_hour == 10) & (current_minute < 30)
        ):
            return 2
        elif ((current_hour >= 10) & (current_hour < 11)) or (
            current_hour == 11 and current_minute == 0
        ):
            return 1
        elif (current_hour >= 11) & (
            (current_hour < 20) or ((current_hour == 20) & (current_minute < 30))
        ):
            return 0
        elif ((current_hour >= 20) & (current_hour < 22)) or (
            current_hour == 22 and current_minute == 0
        ):
            return 1
        else:
            return 2


def occupancy_period(start_datetime):
    """
    Function that gives the occupancy period of the session

    :param start_datetime: datetime - Start time of the session
    :return: (bool) - 0,1 and 2 based on the occupancy period
    """
    current_hour = start_datetime.hour
    current_minute = start_datetime.minute

    if (current_hour >= 8) & (
        (current_hour < 10) or ((current_hour == 10) & (current_minute <= 30))
    ):
        return 1  # entrada = categoria 1
    elif (current_hour >= 14) & (
        (current_hour < 15) or (current_hour == 15 and current_minute == 0)
    ):
        return 1  # entrada = categoria 1
    elif (current_hour >= 12) & (
        (current_hour < 14) or (current_hour == 14 and current_minute == 0)
    ):
        return 2  # saída = categoria 2
    elif current_hour >= 18:
        return 2  # saída = categoria 2
    else:
        return 0  # restantes = categoria 0


def var_discretization(start_datetime, bin_size):
    """
    Function that discretes a variable

    :param start_datetime: datetime - Start time of the session
    :param bin_size: str - discretization bin
    :return: float -  discrete value
    """
    if bin_size == "20min":
        category = (start_datetime.hour * 60 + start_datetime.minute) // 20
    elif bin_size == "40min":
        category = (start_datetime.hour * 60 + start_datetime.minute) // 40
    elif bin_size == "1h":
        category = start_datetime.hour
    elif bin_size == "2h":
        category = start_datetime.hour // 2
    elif bin_size == "4h":
        category = start_datetime.hour // 4
    else:
        raise ValueError("Invalid bin_size")
    return category


def var_discretization_target(dataframe, target, bin_size):
    # Define bin size of 30 minutes
    bin_size = bin_size
    df = dataframe.copy()

    # Create bins starting from min_value to max_value in steps of bin_size
    df["discretization_duration"] = (df[target] / bin_size).round().astype(int)

    return df


def compute_tx_lag_energy(dataframe):
    df = dataframe.copy()
    df["tx_lag_energy"] = df["total_energy_transfered"].shift(1)

    return df


def analyze_charging_patterns(user_id, file_path):
    """
    Analyze charging patterns of EV charging sessions

    :return: (bool) - True if charging pattern is consistent, False otherwise
    """

    user_key = f"user_{user_id}"

    # Read the JSON file
    try:
        with open(file_path, "r") as file:
            data = json.load(file)
        logger.debug("JSON File was read successfully")
    except FileNotFoundError:
        logger.error(f"The file '{file_path}' does not exist.")
    except json.JSONDecodeError:
        logger.error(f"The file '{file_path}' is not a valid JSON file.")

    # Check if user_id exists in the JSON data
    if user_key not in data:
        logger.error(f"User ID '{user_id}' not found in the JSON file.")

    logger.debug("Get ev owner charging patterns ")
    total_energy_rating = data[user_key].get("energy_rating", None)
    # todo: check why these variables are not used
    # Validate the logic to assess if a user will be using ml or naive
    charging_duration_rating = data[user_key].get('duration_rating', None)
    # return_rating = data[user_key].get('return_period', None)
    start_time_rating = data[user_key].get('start_time_rating', None)

    return {
        "energy_rating": total_energy_rating,
        "duration_rating": charging_duration_rating,
        "start_time_rating": start_time_rating,
    }


def analyze_threshold(pattern, threshold):
    """
    Analyze charging patterns of EV charging sessions

    :return: (bool) - similar if charging pattern is consistent, not_similar otherwise
    """
    if pattern < threshold:
        return "similar"
    else:
        return "not_similar"


@logger.catch(level="ERROR", message="Error Data Analysis:")
def statistical_analysis(data):
    """
    Analyze charging patterns of EV charging sessions in 5 groups:
        - Attedance Analysis
        - Charging Duration Analysis
        - Returnal Analysis
        - Start time Analysis
        - Total Energy charged Analysis

    :param data: dataframe - DataFrame containing data to be analyzed.
    :return: dataframe - characteristics of EV owners of charging session
    """
    # Transform date strings to datetime
    data["start_time"] = pd.to_datetime(data["start_time"])
    data["end_time"] = data["end_time"].fillna(pd.Timestamp.today())
    data["end_time"] = pd.to_datetime(data["end_time"])

    # Attendance analysis
    logger.info("Starting Attedance Analysis")
    data["day_of_week"] = data["start_time"].dt.dayofweek

    # Get the total number of sessions by user_id
    total_sessions_by_user = (
        data.groupby("user_id").size().reset_index(name="total_sessions")
    )  # noqa

    # Merge the results
    sessions_by_day = (
        data.groupby(["user_id", "day_of_week"])
        .size()
        .reset_index(name="count_sessions_week_day")
    )
    sessions_by_day = pd.merge(
        sessions_by_day, total_sessions_by_user, on="user_id"
    )  # noqa

    day_num_to_name = {
        0: "Monday",
        1: "Tuesday",
        2: "Wednesday",
        3: "Thursday",
        4: "Friday",
        5: "Saturday",
        6: "Sunday",
    }
    logger.info("mapping_day_num_to_name:")
    sessions_by_day["day_of_week"] = sessions_by_day["day_of_week"].map(day_num_to_name)
    sessions_by_day["rating"] = (
        sessions_by_day["count_sessions_week_day"] / sessions_by_day["total_sessions"]
    )  # noqa
    sessions_by_day["rating_frequency"] = sessions_by_day["rating"].apply(
        lambda x: "Frequent" if x > 0.14 else "Rare"
    )
    sessions_by_day_transpose = sessions_by_day[
        ["user_id", "day_of_week", "rating_frequency"]
    ]
    unique_days = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    # Reorder the columns and reset the index
    sessions_by_day_transpose = sessions_by_day_transpose.pivot(
        index="user_id", columns="day_of_week", values="rating_frequency"
    )
    available_days = [
        day for day in unique_days if day in sessions_by_day_transpose.columns
    ]

    sessions_by_day_transpose = (
        sessions_by_day_transpose[available_days].reset_index().fillna("No_sessions")
    )

    logger.info("Attendance Analysis Completed")

    ## Charging Duration and Total Energy Charged analysis

    logger.info("Starting Charging Duration and Total Energy Charged analysis")

    data["parked_time"] = (
        data["end_time"] - data["start_time"]
    ).dt.total_seconds() / 60
    data["end_time"] = data["end_time"].fillna(data["start_time"])

    ev_owner_analysis = (
        data.groupby("user_id")
        .agg(
            std_total_energy_transfered=("total_energy_transfered", "std"),
            std_parked_time=("parked_time", "std"),
            count_sessions=("session_id", "count"),
        )
        .reset_index()
    )

    threshold_std_parked = data["parked_time"].std()
    threshold_std_energy = data["total_energy_transfered"].std()

    ev_owner_analysis["duration_rating"] = ev_owner_analysis["std_parked_time"].apply(
        analyze_threshold, threshold=threshold_std_parked
    )
    ev_owner_analysis["energy_rating"] = ev_owner_analysis[
        "std_total_energy_transfered"
    ].apply(analyze_threshold, threshold=threshold_std_energy)

    logger.info("Charging Duration and Total Energy Charged analysis Completed")

    # Start time analysis
    logger.info("Starting Start time analysis")

    data["hour"] = data["start_time"].dt.strftime("%H:%M:%S")

    def get_time_of_day(hour):
        if "05:00:00" <= hour < "12:00:00":
            return "Morning"
        elif "12:00:00" <= hour < "17:00:00":
            return "Afternoon"
        elif "17:00:00" <= hour < "21:00:00":
            return "Evening"
        else:
            return "Night"

    data["time_of_day"] = data["hour"].apply(get_time_of_day)

    # Count sessions by 'time_of_day'
    time_of_day_sessions = (
        data.groupby(["user_id", "time_of_day"])
        .size()
        .reset_index(name="count_sessions_time_of_day")
    )

    # Merge and calculate session percentages
    merged_data = (
        data.groupby("user_id")["session_id"]
        .count()
        .reset_index()
        .rename(columns={"session_id": "count_sessions_total"})
    )
    merged_data = pd.merge(merged_data, time_of_day_sessions, on="user_id")
    merged_data["percentage_sessions"] = (
        merged_data["count_sessions_time_of_day"] / merged_data["count_sessions_total"]
    )  # noqa

    # Apply flag rating
    merged_data["flag_rating"] = merged_data["percentage_sessions"].apply(
        lambda x: "Consistent" if x > 0.7 else "Inconsistent"
    )

    # Determine start time rating
    merged_data["start_time_rating"] = merged_data.groupby("user_id")[
        "flag_rating"
    ].transform(
        lambda x: (
            "returns_at_same_period"
            if "Consistent" in x.values
            else "returns_at_different_periods"
        )
    )

    logger.info("Start time analysis Completed")

    # Returnal Analysis
    logger.info("Starting Returnal analysis")

    # Calculate time difference between consecutive visits
    data = data.sort_values(by=["user_id", "start_time"])
    data["timestamp"] = data["start_time"].dt.normalize()
    data["time_diff"] = data.groupby("user_id")["timestamp"].diff()

    def classify_time_diff(td):
        if pd.isna(td):
            return "NAN"
        elif td < pd.Timedelta(days=1):
            return "returns_at_d"
        elif td < pd.Timedelta(days=2):
            return "returns_at_d+1"
        elif td < pd.Timedelta(days=3):
            return "returns_at_d+2"
        elif td < pd.Timedelta(days=4):
            return "returns_at_d+3"
        elif td < pd.Timedelta(days=5):
            return "returns_at_d+4"
        elif td < pd.Timedelta(days=6):
            return "returns_at_d+5"
        elif td < pd.Timedelta(days=7):
            return "returns_at_d+7"
        else:
            return "returns_at_d>7 "

    data["return_period"] = data["time_diff"].apply(classify_time_diff)

    # Determine predominant return period
    return_periods = (
        data.groupby(["user_id", "return_period"]).size().reset_index(name="count")
    )
    max_return_periods = return_periods.loc[
        return_periods.groupby("user_id")["count"].idxmax()
    ]

    logger.info("Returnal analysis Completed")

    # # Merge and finalize the analysis
    logger.info("Merging all the analysis")

    final_data = ev_owner_analysis.merge(
        merged_data[["user_id", "start_time_rating"]], on="user_id", how="left"
    )
    final_data = final_data.merge(
        max_return_periods[["user_id", "return_period"]], on="user_id", how="left"
    )
    final_data = final_data.merge(
        sessions_by_day_transpose[["user_id"] + available_days],
        on="user_id",
        how="left",
    )
    required_columns = [
        "user_id",
        "count_sessions",
        "energy_rating",
        "duration_rating",
        "start_time_rating",
        "return_period",
    ]

    logger.info("Merging was successful")

    return final_data[required_columns + available_days]


@logger.catch(level="ERROR", message="Error Check Sessions:")
def check_sessions(user_id):
    """
    Retrive the number of sessions for the EV owner

    :return: (int) - number of sessions for the Ev owner
    """

    user_key = f"user_{user_id}"

    json_file_path = os.path.join(settings.BASE_PATH, "files/statiscal_analysis")

    # Read the JSON file
    try:
        with open(json_file_path, "r") as file:
            data = json.load(file)
        logger.debug("JSON File was read successfully")
    except FileNotFoundError:
        logger.error(f"The file '{json_file_path}' does not exist.")
    except json.JSONDecodeError:
        logger.error(f"The file '{json_file_path}' is not a valid JSON file.")

    # Check if user_id exists in the JSON data
    if user_key not in data:
        logger.error(f"User ID '{user_id}' not found in the JSON file.")

    # Get the sessions for the specified user_id
    logger.debug("Get sessions count")
    session_count = data[user_key].get("count_sessions", None)

    # Return the number of sessions
    return session_count


def day_period(start_datetime):
    """
    Compute period of day baed on a date.

    :param start_datetime: datetime - Input datetime.
    :return: string - period of the day
    """
    if start_datetime.hour < 6:
        return "dp_0"
    elif (start_datetime.hour >= 6) & (start_datetime.hour < 13):
        return "dp_1"
    elif (start_datetime.hour >= 13) & (start_datetime.hour < 19):
        return "dp_2"
    elif (start_datetime.hour >= 19) & (start_datetime.hour <= 23):
        return "dp_3"
    else:
        return "dp_0"  # redundancy


@logger.catch(level="ERROR", message="Error naive models forecast:")
def naive_models_forecast(data, target, time=None):
    """
    Compute naive models (naive average, naive median, persistence) and evaluate their MSE.

    :param data: pd.DataFrame - Input DataFrame containing the data.
    :param target_feature: str - Name of the target feature column in the DataFrame.
    :return: float - prevision based on the naive model with lowest MSE.

    """
    logger.info(f"Calculation of naive {target}")
    data = data.copy()
    target_feature = target
    data["start_datetime"] = pd.to_datetime(data["start_time"], errors="coerce")
    logger.debug(type(data["start_datetime"]))

    data = data[data["start_datetime"].notna()]

    data = data.sort_values(by="start_datetime")
    data["includes_weekend"] = data.apply(
        lambda row: includes_weekend(row["start_datetime"], row["end_datetime"]), axis=1
    )

    # TXCX_dataset[var].reset_index(drop=True, inplace=True)
    # data['end_datetime'] = pd.to_datetime(data['end_time'], errors='coerce')

    # data['duration'] = (data['end_datetime'] - data[
    #     'start_datetime']).dt.total_seconds() / 60

    logger.info("Initializing naive models")
    # Separate data into weekday and weekend sets
    weekday_df = data[data["includes_weekend"] != 0].copy()
    weekend_df = data[data["includes_weekend"] == 0].copy()

    logger.info("Initializing naive models with weekday and weekend differentiation")

    # Define placeholders for means/medians
    weekday_avg, weekday_median = 0, 0

    for i in range(data.shape[0]):
        logger.info("start cicle")
        if i == 0:
            logger.info("first_row")
            data.loc[i, "naive_avg"] = 0  # first row shall be deleted later
            data.loc[i, "naive_avg_row5"] = 0  # first row shall be deleted later
            data.loc[i, "naive_avg_period"] = 0  # first row shall be deleted later
            data.loc[i, "naive_med"] = 0  # first row shall be deleted later
            data.loc[i, "naive_med_row5"] = 0  # first row shall be deleted later
            data.loc[i, "naive_med_period"] = 0  # first row shall be deleted later
            data.loc[i, "persistence"] = 0  # first row shall be deleted later

        else:
            logger.info("else")
            logger.debug(data[target_feature].iloc[i - 1])
            is_weekend = data.iloc[i]["includes_weekend"] == 0
            if not is_weekend:
                # Calculate weekday averages and medians
                weekday_avg = (
                    data[target_feature]
                    .iloc[:i][~data["includes_weekend"].iloc[:i].eq(0)]
                    .mean()
                )
                weekday_median = (
                    data[target_feature]
                    .iloc[:i][~data["includes_weekend"].iloc[:i].eq(0)]
                    .median()
                )
                data.loc[i, "persistence"] = data[target_feature].iloc[i - 1]
                data.loc[i, "naive_avg"] = data[target_feature].iloc[:i].mean()
                data.loc[i, "naive_med"] = data[target_feature].iloc[:i].median()
            else:
                logger.debug(f"i{i}")
                logger.debug(data["start_time"][i])
                if i == weekend_df.index[0]:
                    # For the first weekend row, use weekday cumulative average
                    logger.debug(weekday_avg)
                    logger.debug(weekday_median)

                    data.loc[i, "persistence"] = weekday_avg
                    data.loc[i, "naive_avg"] = weekday_avg
                    data.loc[i, "naive_med"] = weekday_median
                    logger.debug(data.loc[i, "naive_avg"])
                else:
                    # For other weekend rows, use prior weekend averages
                    logger.debug(weekday_avg)
                    logger.debug(weekday_median)
                    data.loc[i, "persistence"] = data[target_feature].iloc[i - 1]
                    data.loc[i, "naive_avg"] = data[target_feature].iloc[:i].mean()
                    data.loc[i, "naive_med"] = data[target_feature].iloc[:i].median()

    # Calculate the cumulative mean and median per day_period
    # Initialize the columns with NaN or 0
    logger.info("initializing naive period")

    # Calculate separate 5-row rolling values for weekends and weekdays
    weekend_roll_avg = (
        data[target_feature]
        .where(data["includes_weekend"] == 0)
        .rolling(window=5, min_periods=1)
        .mean()
        .shift(1)
    )
    weekend_roll_median = (
        data[target_feature]
        .where(data["includes_weekend"] == 0)
        .rolling(window=5, min_periods=1)
        .median()
        .shift(1)
    )

    weekday_roll_avg = (
        data[target_feature]
        .where(data["includes_weekend"] == 1)
        .rolling(window=5, min_periods=1)
        .mean()
        .shift(1)
    )
    weekday_roll_median = (
        data[target_feature]
        .where(data["includes_weekend"] == 1)
        .rolling(window=5, min_periods=1)
        .median()
        .shift(1)
    )
    logger.debug(weekday_median)
    # Replace all NaN values in weekend_roll_avg with weekday_avg
    if weekend_roll_avg.isna().any():
        logger.debug(
            f"NaN values found in weekend_roll_avg, replacing all with {weekday_avg}"
        )
        weekend_roll_avg.fillna(weekday_avg, inplace=True)

    # Replace all NaN values in weekend_roll_median with weekday_median
    if weekend_roll_median.isna().any():
        logger.debug(
            f"NaN values found in weekend_roll_median, replacing all with {weekday_median}"
        )
        weekend_roll_median.fillna(weekday_median, inplace=True)
        # Assign rolling values based on whether it's a weekend or weekday
    data["naive_avg_row5"] = weekday_roll_avg.where(
        data["includes_weekend"] == 1, weekend_roll_avg
    )
    logger.debug(data[data["session_id"] == "0000000000000023"]["naive_avg_row5"])
    logger.debug(f"weekend_roll_value:{weekend_roll_median}")
    data["naive_med_row5"] = weekday_roll_median.where(
        data["includes_weekend"] == 1, weekend_roll_median
    )
    logger.debug(data[data["session_id"] == "0000000000000023"]["naive_med_row5"])

    data["naive_avg_period"] = np.nan
    data["naive_med_period"] = np.nan
    data["day_period"] = data["start_datetime"].apply(
        lambda x: day_period(x)
    )  # categorical datetime feature

    logger.info("Calculation period naive")
    # Loop over each day_period
    for period in data["day_period"].unique():
        # Select the rows for the current period
        logger.debug(f"period {period}")
        mask = data["day_period"] == period
        logger.debug(f"mask {period}")

        # Calculate expanding mean/median for the current period,
        # excluding the current row
        logger.debug("mean calculation")
        data.loc[mask, "naive_avg_period"] = (
            data.loc[mask, target_feature].expanding().mean().shift(1)
        )
        logger.debug("mean calculation")
        data.loc[mask, "naive_med_period"] = (
            data.loc[mask, target_feature].expanding().median().shift(1)
        )

    logger.info("naive variables")
    # fill the NaN values with 0 - might need to keep nan and exclude
    data.fillna(0, inplace=True)

    logger.info("mean_squared_error")
    mse_naive_avg = mean_squared_error(data[target_feature], data["naive_avg"]) ** 0.05
    mse_naive_avg_row5 = (
        mean_squared_error(data[target_feature], data["naive_avg_row5"]) ** 0.05
    )
    mse_naive_med = mean_squared_error(data[target_feature], data["naive_med"]) ** 0.05
    mse_naive_med_row5 = (
        mean_squared_error(data[target_feature], data["naive_med_row5"]) ** 0.05
    )
    mse_naive_med_period = (
        mean_squared_error(data[target_feature], data["naive_med_period"]) ** 0.05
    )
    mse_naive_avg_period = (
        mean_squared_error(data[target_feature], data["naive_avg_period"]) ** 0.05
    )
    mse_persistence = (
        mean_squared_error(data[target_feature], data["persistence"]) ** 0.05
    )

    logger.debug(f"MSE for naive_avg: {mse_naive_avg}")
    logger.debug(f"MSE for naive_avg_row5: {mse_naive_avg_row5}")
    logger.debug(f"MSE for naive_med: {mse_naive_med}")
    logger.debug(f"MSE for naive_med_row5: {mse_naive_med_row5}")
    logger.debug(f"MSE for naive_med_period: {mse_naive_med_period}")
    logger.debug(f"MSE for naive_avg_period: {mse_naive_avg_period}")
    logger.debug(f"MSE for naive_avg_row5: {mse_naive_avg_row5}")
    logger.debug(f"MSE for persitence: {mse_persistence}")

    mse_values = {
        "naive_avg": mse_naive_avg,
        "naive_avg_row5": mse_naive_avg_row5,
        "naive_med": mse_naive_med,
        "naive_med_row5": mse_naive_med_row5,
        "naive_med_period": mse_naive_med_period,
        "naive_avg_period": mse_naive_avg_period,
        "persistence": mse_persistence,
    }

    # Find the column with the lowest MSE
    logger.info("lowest column with min mse error")

    lowest_mse_column = min(mse_values, key=mse_values.get)
    logger.debug(f"column with lowest mse: {lowest_mse_column}")

    # Drop the columns that do not correspond to the lowest MSE
    columns_to_drop = [
        column for column in mse_values.keys() if column != lowest_mse_column
    ]
    logger.debug(f"drop columns: {columns_to_drop}")

    period_start = day_period(time)
    data.drop(columns=columns_to_drop, inplace=True)
    df = data
    logger.debug(f"day period : {period_start}")
    period_data = df[df["day_period"] == period_start]

    if ("period" in lowest_mse_column) & (not period_data.empty):
        forecast_period = df[df["day_period"] == period_start][lowest_mse_column].iloc[
            -1
        ]
        logger.debug(f"forecast: {forecast_period}")
        return forecast_period
    else:
        forecast = df[lowest_mse_column].iloc[-1]
        logger.debug(f"forecast: {forecast}")
        return forecast


@logger.catch(level="ERROR", message="Error naive models forecast returnal:")
def naive_models_returnal(data, time=None):
    """
    Compute naive models for returnal forecast (naive average, naive median,
    persistence) and evaluate their MSE.

    :param data: pd.DataFrame - Input DataFrame containing the data.
    :param target_feature: str - Name of the target feature column in the DataFrame.
    :return: float - prevision based on the naive model with lowest MSE.

    """
    data = data.copy()
    logger.info("Start date time calculation")
    # Ensure start_time is recognized as datetime
    data["start_datetime"] = pd.to_datetime(data["start_time"], errors="coerce")
    data = data[data["start_datetime"].notna()]

    # Sort data by start_datetime to ensure correct order for calculations
    data = data.sort_values(by="start_datetime")
    # data['time_since_last_tx'] = 0.0
    logger.info("calculation of time since last session")

    # Initialize new columns with NaT (Not a Time)
    data["naive_avg"] = 0.0
    data["naive_avg_row5"] = 0.0
    data["naive_avg_period"] = 0.0
    data["naive_med"] = 0.0
    data["naive_med_row5"] = 0.0
    data["naive_med_period"] = 0.0
    data["persistence"] = 0.0

    logger.info("Naive calculation")

    # Convert datetime to Unix timestamp in seconds for calculations
    for i in range(1, len(data)):
        data.loc[i, "persistence"] = data["time_since_last_tx"].iloc[i - 1]

        if i >= 1:
            data.loc[i, "naive_avg"] = data["time_since_last_tx"].iloc[:i].mean()
            data.loc[i, "naive_med"] = np.median(data["time_since_last_tx"].iloc[:i])

        if i >= 5:
            data.loc[i, "naive_avg_row5"] = (
                data["time_since_last_tx"].iloc[i - 5 : i].mean()
            )
            data.loc[i, "naive_med_row5"] = np.median(
                data["time_since_last_tx"].iloc[i - 5 : i]
            )

    data["day_period"] = data["start_datetime"].apply(day_period)

    logger.info("Naive calculation for period")

    for period in data["day_period"].unique():
        mask = data["day_period"] == period
        data.loc[mask, "naive_avg_period"] = (
            data.loc[mask, "time_since_last_tx"].expanding().mean().shift(1)
        )
        data.loc[mask, "naive_med_period"] = (
            data.loc[mask, "time_since_last_tx"].expanding().median().shift(1)
        )

    data.fillna(0, inplace=True)

    # Calculate MSE
    mse_values = {
        "naive_avg": mean_squared_error(data["time_since_last_tx"], data["naive_avg"])
        ** 0.5,
        "naive_avg_row5": mean_squared_error(
            data["time_since_last_tx"], data["naive_avg_row5"]
        )
        ** 0.5,
        "naive_avg_period": mean_squared_error(
            data["time_since_last_tx"], data["naive_avg_period"]
        )
        ** 0.5,
        "naive_med": mean_squared_error(data["time_since_last_tx"], data["naive_med"])
        ** 0.5,
        "naive_med_row5": mean_squared_error(
            data["time_since_last_tx"], data["naive_med_row5"]
        )
        ** 0.5,
        "naive_med_period": mean_squared_error(
            data["time_since_last_tx"], data["naive_med_period"]
        )
        ** 0.5,
        "persistence": mean_squared_error(
            data["time_since_last_tx"], data["persistence"]
        )
        ** 0.5,
    }

    logger.debug(f'MSE for naive_avg: {mse_values["naive_avg"]}')
    logger.debug(f'MSE for naive_avg_row5: {mse_values["naive_avg_row5"]}')
    logger.debug(f'MSE for naive_avg_period: {mse_values["naive_avg_period"]}')
    logger.debug(f'MSE for naive_med: {mse_values["naive_med"]}')
    logger.debug(f'MSE for naive_med_row5: {mse_values["naive_med_row5"]}')
    logger.debug(f'MSE for naive_med_period: {mse_values["naive_med_period"]}')
    logger.debug(f'MSE for persistence: {mse_values["persistence"]}')
    lowest_mse_column = min(mse_values, key=mse_values.get)
    df = data.copy()
    if time is not None:
        period_start = day_period(time)
        period_data = df[df["day_period"] == period_start]
        logger.debug(f"periodo: {period_data}")
        if ("period" in lowest_mse_column) and (not period_data.empty):
            delta_time_hours = df[df["day_period"] == period_start][
                lowest_mse_column
            ].iloc[-1]
        else:
            delta_time_hours = df[lowest_mse_column].iloc[-1]

        # Convert delta time from hours to timedelta
        delta_time = timedelta(hours=delta_time_hours)

    # Calculate the forecasted time
    data["start_datetime"] = pd.to_datetime(data["start_time"], errors="coerce")

    # Sort data by start_datetime to ensure correct order for calculations
    data = data.sort_values(by="start_datetime")
    last_start_time = data["start_datetime"].iloc[-1]

    logger.debug(f"start_time:{last_start_time}")
    forecasted_time = last_start_time + delta_time
    logger.debug(f"forecast: {forecasted_time}")

    return forecasted_time


def compute_next_session(start_time, forecast_time):
    """
    Calculates next session.

    :param start_time: datetime - start time of the session.
    :param forecast_time: datetime - forecast time of differenc between start sessions.

    :return: nex_session : datetime - return the value of the start_datetime of the next session

    """
    start_time_dt = datetime.strptime(start_time, "%Y-%m-%dT%H:%M:%S")

    next_session = start_time_dt + timedelta(hours=forecast_time)

    return next_session.strftime("%Y-%m-%d %H:%M:%S")


def normalize_dataset(original_value, min, max):
    """
    normalize dataset.

    :param original_value: int - orginal value.
    :param min: int - minimum value.
    :param: max: it - maximium value.

    :return: scaled : int - return the value normalized

    """
    epsilon = 1e-10
    if max != min:
        scaled = (original_value - min) / (max - min)
    else:
        scaled = (original_value - min) / (max - min + epsilon)

    return scaled


def denormalize_dataset(predicted_normalized_value, min, max):
    """
    denormalize dataset.

    :param predicted_normalized_value: int - normalized value.
    :param min: int - minimum value.
    :param: max: it - maximium value.

    :return: descaled : int - return the value not normalized

    """
    descaled = (predicted_normalized_value * (max - min)) + min
    return descaled


# MODEL OPTIMIZATION
def xgb_cv(
    learning_rate,
    n_estimators,
    max_depth,
    min_child_weight,
    gamma,
    subsample,
    colsample_bytree,
    reg_alpha,
    reg_lambda,
    X_train,
    y_train,
):
    """
    Perform cross-validation using an XGBoost regressor model.

    :param learning_rate: float - Step size shrinkage used in the update to prevent overfitting. Controls how quickly the model adapts.
    :param n_estimators: int - Number of boosting rounds or trees to build.
    :param max_depth: int - Maximum depth of a tree. Increasing this value makes the model more complex.
    :param min_child_weight: float - Minimum sum of instance weight (hessian) needed in a child node.
    :param gamma: float - Minimum loss reduction required to make a further partition on a leaf node.
    :param subsample: float - Fraction of samples (rows) used for training each tree. Helps prevent overfitting.
    :param colsample_bytree: float - Fraction of features (columns) used to build each tree.
    :param reg_alpha: float - L1 regularization term (Lasso) on weights. Increases model robustness by penalizing large coefficients.
    :param reg_lambda: float - L2 regularization term (Ridge) on weights. Reduces model complexity by discouraging large weights.
    :param X_train: pd.DataFrame or np.ndarray - The training dataset features.
    :param y_train: pd.Series or np.ndarray - The training dataset target values.

    :return: float - The mean of the cross-validated negative mean squared error (CV NRMSE) for the XGBoost model.
    """
    params = {
        "learning_rate": learning_rate,
        "n_estimators": int(n_estimators),
        "max_depth": int(max_depth),
        "min_child_weight": min_child_weight,
        "gamma": gamma,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "reg_alpha": reg_alpha,
        "reg_lambda": reg_lambda,
        "objective": "reg:squarederror",
    }
    model = xgb.XGBRegressor(**params)

    return cross_val_score(
        model, X_train, y_train, cv=5, scoring="neg_mean_squared_error"
    ).mean()


# CROSS VALIDATION TRAIN TEST SPLIT
logger.catch(level="ERROR", message="Error with cv:")


def cv_train_and_test(X_train, y_train, X_test, y_test, best_params, check, eval):
    """
    Train and test an XGBoost regressor model using the provided training and test sets.

    :param X_train: pd.DataFrame or np.ndarray - Training dataset features.
    :param y_train: pd.Series or np.ndarray - Training dataset target values.
    :param X_test: pd.DataFrame or np.ndarray - Test dataset features.
    :param y_test: pd.Series or np.ndarray - Test dataset target values.
    :param best_params: dict - Dictionary of the best hyperparameters to use for training the XGBoost regressor.

    :return: tuple - Mean Absolute Error (MAE) and Mean Squared Error (MSE) for the test set predictions.
    """
    # Create and train the model
    model = xgb.XGBRegressor(random_state=42, early_stopping_rounds=10)
    logger.debug(f"X_train:{X_train.shape}")
    logger.debug(f"y_train:{y_train.shape}")

    # weight_features = np.array(weight_features)
    model.fit(X_train, y_train, eval_set=eval, verbose=True)

    # Predict on the test set
    logger.info("predict cv model")
    y_pred_cv = model.predict(X_test)
    if check:
        y_pred_cv = np.expm1(y_pred_cv)
        y_test = np.expm1(y_test)

    # Calculate and return the metrics
    mae = mean_absolute_error(y_test, y_pred_cv)
    mse = mean_squared_error(y_test, y_pred_cv)
    return mae, mse


# CROSS VALIDATION
def k_fold_split(X, y, k=5):
    """
    Perform K-fold split of the dataset for cross-validation.

    :param X: pd.DataFrame or np.ndarray - The feature dataset to split.
    :param y: pd.Series or np.ndarray - The target dataset to split.
    :param k: int - Number of folds for cross-validation (default is 5).

    :return: tuple - Two lists containing the split features (X_folds) and target (y_folds) datasets.
    """
    n = len(X)
    fold_size = n // k
    remainder = n % k

    X_folds = []
    y_folds = []

    start = 0
    for i in range(k):
        end = start + fold_size
        if remainder > 0:
            end += 1
            remainder -= 1
        X_folds.append(X[start:end])
        y_folds.append(y[start:end])
        start = end

    return X_folds, y_folds


@logger.catch(level="ERROR", message="Error comparing dataframes values:")
def compare_dataframe_json(df, forecasts, column_mapping, threshold_mapping):
    """
    Compare values between a DataFrame and a JSON object based on a given column-to-JSON mapping.

    :param df: pd.DataFrame - The input DataFrame containing actual values.
    :param forecasts: dict - Nested JSON data to compare against the DataFrame.
    :param column_mapping: dict - Dictionary mapping DataFrame columns to the corresponding JSON path.
    :param threshold_mapping: dict - Dictionary defining acceptable difference thresholds for numeric comparisons.

    :return: dict - A detailed comparison result, including differences, percentage differences, and MSE value errors.
    """
    comparison_result = {}

    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        comparison_result[index] = {}

        # Iterate over each column and its corresponding JSON path
        for df_column, json_path in column_mapping.items():
            # Navigate to the correct nested JSON value using the path
            json_value = get_nested_value(forecasts, json_path)
            logger.debug(f"json_value:{json_value}")

            df_value = row[df_column]

            # Get the specific threshold for the current column
            threshold = threshold_mapping.get(
                df_column, 0
            )  # Default to 0 if no threshold is specified

            # Check if the value is numeric and the difference exceeds the threshold
            if isinstance(json_value, (int, float)) and isinstance(
                df_value, (int, float)
            ):
                difference = abs(json_value - df_value)
                if difference > threshold:
                    comparison_result[index][df_column] = {
                        "DataFrame_value": df_value,
                        "JSON_value": json_value,
                        "Difference": difference,
                        "Difference percentage": (
                            "NA" if df_value == 0 else difference / df_value
                        ),
                        "MSE_value_error": (df_value - json_value) ** 2,
                        "Status": "Exceeded Threshold",
                    }
                else:
                    comparison_result[index][df_column] = {
                        "DataFrame_value": df_value,
                        "JSON_value": json_value,
                        "Difference": difference,
                        "Difference percentage": (
                            "NA" if df_value == 0 else difference / df_value
                        ),
                        "MSE_value_error": (df_value - json_value) ** 2,
                        "Status": "Within Threshold",
                    }
            else:
                comparison_result[index][df_column] = {
                    "DataFrame_value": df_value,
                    "JSON_value": json_value,
                    "Status": "Non-numeric Comparison",
                }

    return comparison_result


@logger.catch(level="ERROR", message="Error calculating duration:")
def calculation_duration(dataframe):
    """
    Calculus of session duration.

    :param df: pd.DataFrame - The input DataFrame containing actual values.

    :return: dataframe - Dataframe with duration values of session.
    """
    logger.info("Calculation of duration")
    data = dataframe.copy()
    # Convert to datetime without timezone (tz-naive)
    logger.info("start_datetime")
    data["start_datetime"] = pd.to_datetime(data["start_time"], errors="coerce")
    data = data.sort_values(by="start_datetime")
    logger.info("end_datetime")

    data["end_datetime"] = pd.to_datetime(data["end_time"], errors="coerce")
    logger.info("Duration calculation ...")
    data["duration"] = (
        data["end_datetime"] - data["start_datetime"]
    ).dt.total_seconds() / 60
    return data


@logger.catch(level="ERROR", message="Error calculating delta startime:")
def calculation_delta_startime(dataframe):
    """
    Calculate the time until next session.

    :param df: pd.DataFrame - The input DataFrame containing actual values.

    :return: dataframe - Dataframe with delta values.
    """
    df = dataframe.copy()
    df.sort_values(by="start_datetime", ascending=True, inplace=False)
    df["time_since_last_tx"] = 0
    for i in np.arange(1, len(df)):  # -1 to not go for last row (no prior information)
        delta_tx = df["start_datetime"].iloc[i] - df["start_datetime"].iloc[i - 1]
        logger.debug(f"delta_tx: {delta_tx}")
        df.iloc[i, df.columns.get_loc("time_since_last_tx")] = (
            abs(delta_tx.total_seconds()) / 3600
        )
    return df


@logger.catch(level="ERROR", message="Error calculating delta startime:")
def calculation_time_next_plug_in(dataframe):
    """
    Calculus of time until next session charger plugin (session start)

    :param df: pd.DataFrame - The input DataFrame containing actual values.

    :return: dataframe - Dataframe with delta _startime values.
    """
    df = dataframe.copy()
    df.sort_values(by="start_datetime", ascending=True, inplace=False)
    df["time_next_plug_in"] = 0
    for i in np.arange(
        0, len(df) - 1
    ):  # -1 to not go for last row (no prior information)
        delta_tx = df["start_datetime"].iloc[i + 1] - df["start_datetime"].iloc[i]
        logger.debug(f"delta_tx: {delta_tx}")
        df.iloc[i, df.columns.get_loc("time_next_plug_in")] = (
            abs(delta_tx.total_seconds()) / 3600
        )
    return df


@logger.catch(level="ERROR", message="Error normalizating data:")
def data_normalization(
    dataframe, target_feature, save_path="normalization_params.json"
):
    """
    Normalize numerical features in the dataset for consistent scaling.

    This function applies min-max normalization to selected numerical features in the input DataFrame.
    Categorical variables like 'day_period' and 'tri_tariff' are excluded from normalization.

    Parameters:
    ----------
    dataframe : pd.DataFrame - Dataframe containing features to be normalized
    target_feature : str - Feature to be normalized
    save_path : str - Path where the normalization parameters should be saved (default: 'normalization_params.json')

    Returns:
    --------
    pd.DataFrame - DataFrame with normalized numerical features.
    """

    X = dataframe.copy()

    if target_feature != "time_since_last_tx":
        # Setting min-max limits
        normalization_params = {
            "start_year": {
                "min": X["start_year"].min().item(),
                "max": X["start_year"].max().item(),
            },
            "start_month": {
                "min": X["start_month"].min().item(),
                "max": X["start_month"].max().item(),
            },
            "start_day": {
                "min": X["start_day"].min().item(),
                "max": X["start_day"].max().item(),
            },
            "start_hour": {
                "min": X["start_hour"].min().item(),
                "max": X["start_hour"].max().item(),
            },
            "start_minute": {
                "min": X["start_minute"].min().item(),
                "max": X["start_minute"].max().item(),
            },
            "tx_lag": {
                "min": X[target_feature].min().item(),
                "max": X[target_feature].max().item(),
            },
            "time_since_last_tx": {
                "min": X["time_since_last_tx"].min().item(),
                "max": X["time_since_last_tx"].max().item(),
            },
            "occupancy_period": {
                "min": X["occupancy_period"].min().item(),
                "max": X["occupancy_period"].max().item(),
            },
            "var_1h": {
                "min": X["var_1h"].min().item(),
                "max": X["var_1h"].max().item(),
            },
            "var_20min": {
                "min": X["var_20min"].min().item(),
                "max": X["var_20min"].max().item(),
            },
            "var_40min": {
                "min": X["var_40min"].min().item(),
                "max": X["var_40min"].max().item(),
            },
        }
    else:
        normalization_params = {
            "start_year": {
                "min": X["start_year"].min().item(),
                "max": X["start_year"].max().item(),
            },
            "start_month": {
                "min": X["start_month"].min().item(),
                "max": X["start_month"].max().item(),
            },
            "start_day": {
                "min": X["start_day"].min().item(),
                "max": X["start_day"].max().item(),
            },
            "start_hour": {
                "min": X["start_hour"].min().item(),
                "max": X["start_hour"].max().item(),
            },
            "start_minute": {
                "min": X["start_minute"].min().item(),
                "max": X["start_minute"].max().item(),
            },
            "tx_lag": {
                "min": X[target_feature].min().item(),
                "max": X[target_feature].max().item(),
            },
            "occupancy_period": {
                "min": X["occupancy_period"].min().item(),
                "max": X["occupancy_period"].max().item(),
            },
            "var_1h": {
                "min": X["var_1h"].min().item(),
                "max": X["var_1h"].max().item(),
            },
            "var_20min": {
                "min": X["var_20min"].min().item(),
                "max": X["var_20min"].max().item(),
            },
            "var_40min": {
                "min": X["var_40min"].min().item(),
                "max": X["var_40min"].max().item(),
            },
        }

    # Save normalization parameters to JSON file
    logger.debug(f"save path: {save_path}")
    if save_path is not None:
        save_json(json_data=normalization_params, file_path=save_path)

    # Data normalization using min-max normalization
    for feature, params in normalization_params.items():
        min_value = params["min"]
        max_value = params["max"]
        X[feature] = X[feature].apply(
            lambda x: normalize_dataset(x, min_value, max_value)
        )

    return X


@logger.catch(level="ERROR", message="Error checking if model exists:")
def model_exists(path_folder, target_feature, user_id, define_file=None):
    """
    Calculus of session duration.

    :param df: pd.DataFrame - The input DataFrame containing actual values.

    :return: dataframe - Dataframe with duration values of session.
    """

    # Define the folder path
    models_folder = os.path.join(settings.BASE_PATH, "files", "models", path_folder)
    logger.debug(f"target:{target_feature}")
    logger.debug(f"user id:{user_id}")

    if isinstance(target_feature, str):
        target_feature = [target_feature]
    # Check if the directory exists
    if not os.path.exists(models_folder):
        return False

    # Iterate over files in the directory to find a match
    for file_name in os.listdir(models_folder):
        logger.debug(f"file name:{file_name}")
        if all(f"{user_id}_{feature}" in file_name for feature in target_feature):
            logger.debug("true")
            if define_file is None or define_file in file_name:
                return True

    return False


@logger.catch(level="ERROR", message="Error during feature selection:")
def feature_selection(dataframe, target_feature, lowest_mse_column):
    """
    Select important features using XGBoost feature importance and predefined feature groups.

    :param dataframe: pd.DataFrame - The input dataset containing features and the target variable.
    :param target_feature: str - The target variable to predict.
    :param lowest_mse_column: str - The column with the lowest mean squared error from previous evaluations.

    :return: list - List of selected features for modeling.
    """

    df_features_selection = dataframe.copy()
    if target_feature == "total_energy_transfered":
        df_features_selection = df_features_selection[
            [
                "day_period",
                "start_year",
                "start_month",
                "start_day",
                "start_hour",
                "start_minute",
                "is_weekday",
                "day_of_week",
                "tx_lag",
                "includes_weekend",
                "tri_tariff",
                "occupancy_period",
                "var_20min",
                "var_40min",
                "var_1h",
                # "tx_lag_energy",
                # "Sum_Last_72_Hours",
                target_feature,
            ]
        ]
    elif target_feature == "hour_minute" or target_feature == "start_datetime":
        df_features_selection = df_features_selection[
            [
                "day_period",
                "start_year",
                "start_month",
                "start_day",
                "start_hour",
                "start_minute",
                "is_weekday",
                "day_of_week",
                "tx_lag",
                "includes_weekend",
                "tri_tariff",
                "occupancy_period",
                "var_20min",
                "var_40min",
                "var_1h",
                "most_probable_return_day",
                "return_day",
                "end_day",
                "end_hour",
                "transaction_same_day",
                # "tx_lag_energy",
                # "Sum_Last_72_Hours",
                target_feature,
            ]
        ]
    else:
        df_features_selection = df_features_selection[
            [
                "day_period",
                "start_year",
                "start_month",
                "start_day",
                "start_hour",
                "start_minute",
                "is_weekday",
                "day_of_week",
                "tx_lag",
                "includes_weekend",
                "tri_tariff",
                "occupancy_period",
                "var_20min",
                "var_40min",
                "var_1h",
                # "tx_lag_energy",
                target_feature,
            ]
        ]

    #################################################
    #       FILTER BY FEATURE IMPORTANCE            ####################################
    #################################################

    # FEATURE GROUP CATEGORIZATION
    feature_lag = ["tx_lag"]
    feature_special_group_1 = ["var_20min"]  # top_feature_most_common[0][0]
    if target_feature != "time_since_last_tx":
        feature_special_group_2 = [
            "time_since_last_tx"
        ]  # top_feature_most_common[1][0]

    feature_special_group_3 = ["start_hour"]  # top_feature_most_common[2][0]

    feature_group_calendar = ["start_month", "start_day", "day_period", "start_minute"]
    feature_group_naive_1 = ["naive_avg_row5", "naive_med_row5"]
    feature_group_naive_2 = ["naive_med_period", "naive_avg_period"]
    feature_group_naive_3 = ["naive_med", "naive_avg"]
    feature_common_group_1 = ["tri_tariff", "occupancy_period"]
    feature_common_group_2 = ["start_year", "is_weekday"]  # newly added
    feature_discretization = ["var_40min", "var_1h"]

    feature_last_to_drop = [target_feature]

    # set up for optimization feature combination
    if target_feature != "time_since_last_tx":
        logger.info("target feature is not time_since_last_tx")
        opt_feat_dict = {
            "feature_lag": feature_lag,
            "feature_special_group_1": feature_special_group_1,
            "feature_special_group_2": feature_special_group_2,
            "feature_special_group_3": feature_special_group_3,
            "feature_group_calendar": feature_group_calendar,
            "feature_group_naive_1": feature_group_naive_1,
            "feature_group_naive_2": feature_group_naive_2,
            "feature_group_naive_3": feature_group_naive_3,
            "feature_common_group_1": feature_common_group_1,
            "feature_common_group_2": feature_common_group_2,
            "feature_discretization": feature_discretization,
        }

        opt_feat_dict_list = [
            "feature_lag",
            "feature_special_group_1",
            "feature_special_group_2",
            "feature_special_group_3",
            "feature_group_calendar",
            "feature_group_naive_1",
            "feature_common_group_1",
            "feature_common_group_2",
            "feature_discretization",
            "feature_group_naive_2",
            "feature_group_naive_3",
        ]
    else:
        opt_feat_dict = {
            "feature_lag": feature_lag,
            "feature_special_group_1": feature_special_group_1,
            "feature_special_group_3": feature_special_group_3,
            "feature_group_calendar": feature_group_calendar,
            "feature_group_naive_1": feature_group_naive_1,
            "feature_group_naive_2": feature_group_naive_2,
            "feature_group_naive_3": feature_group_naive_3,
            "feature_common_group_1": feature_common_group_1,
            "feature_common_group_2": feature_common_group_2,
            "feature_discretization": feature_discretization,
        }

        opt_feat_dict_list = [
            "feature_lag",
            "feature_special_group_1",
            "feature_special_group_3",
            "feature_group_calendar",
            "feature_group_naive_1",
            "feature_common_group_1",
            "feature_common_group_2",
            "feature_discretization",
            "feature_group_naive_2",
            "feature_group_naive_3",
        ]

    opt_features_selected = []

    for opt_index in np.arange(len(opt_feat_dict)):
        if (
            len(opt_features_selected) > 0
        ):  # list must not be empty to remove target from it
            opt_features_selected.pop()  # removes the last element which is the target feature

        opt_features_selected.extend(opt_feat_dict[opt_feat_dict_list[opt_index]])
        # case_step = f'case_1_to_{opt_index + 1}'  # for saving/storing the files for correct comparison

    # # Add target for modeling
    # feature_naive= [lowest_mse_column]
    # opt_features_selected.extend(feature_naive)

    opt_features_selected.extend(feature_last_to_drop)
    logger.debug(opt_features_selected)
    # features_to_be_removed = [feature[0] for feature in features_to_be_removed[0]]

    # opt_features_selected = [feature for feature in opt_features_selected if feature not in features_to_be_removed]
    # logger.debug(f"feature to be removed: {features_to_be_removed}")
    logger.debug(opt_features_selected)
    if target_feature == "total_energy_transfered":
        final_features = (
            [lowest_mse_column]
            # feature_special_group_3
            + feature_group_calendar
            # + ["Sum_Last_72_Hours"]
            # + feature_common_group_1
            # + feature_common_group_2
            # + feature_discretization
            # + feature_group_naive_1
            # + feature_group_naive_2
            # + feature_group_naive_3
            # + ["includes_weekend"]
            # +["occupancy_period"]
            # +["start_hour"] # better for energy
            # +["tri_tariff"] # better for energy
            # +["tx_lag"]
            # + ["tx_lag_energy"] # better for energy
            + [target_feature]
        )
    elif target_feature == "hour_minute" or target_feature == "start_datetime":
        final_features = (
            ["start_month"]
            + ["start_day"]
            + ["start_hour"]
            # + ["end_day"]         
            # + ["end_hour"]
            # + ["transaction_same_day"]
            # feature_group_calendar  +
            + [lowest_mse_column]
            # feature_special_group_3
            # + feature_group_calendar
            # + feature_common_group_1
            # + feature_common_group_2
            # + feature_discretization
            # + feature_group_naive_1
            # + feature_group_naive_2
            # + feature_group_naive_3
            + ["includes_weekend"]
            + ["day_of_week"]
            + ["most_probable_return_day"]
            +["return_day"]
            # + ["is_weekday"]
            # +["occupancy_period"]
            # +["start_hour"] # better for energy
            # # +["tri_tariff"] # better for energy
            # # +["tx_lag"]
            # + ["tx_lag_energy"] # better for energy
            + [target_feature]
        )
    elif target_feature == "time_next_plug_in":
        final_features = (
            ["start_month"]
            + ["start_day"]
            # feature_group_calendar  +
            + [lowest_mse_column]
            # feature_special_group_3
            # + feature_group_calendar
            # + feature_common_group_1
            # + feature_common_group_2
            # + feature_discretization
            # + feature_group_naive_1
            # + feature_group_naive_2
            # + feature_group_naive_3
            # + ["includes_weekend"]
            + ["is_weekday"]
            # +["occupancy_period"]
            # +["start_hour"] # better for energy
            # # +["tri_tariff"] # better for energy
            # # +["tx_lag"]
            # + ["tx_lag_energy"] # better for energy
            + [target_feature]
        )
    elif target_feature=='day_next_plug_in':
        final_features = (
            # ["start_month"]
            # + ["start_day"]
            feature_group_calendar 
            + [lowest_mse_column]
            + feature_special_group_3
            # + feature_group_calendar
            # + feature_common_group_1
            # + feature_common_group_2
            # + feature_discretization
            # + feature_group_naive_1
            # + feature_group_naive_2
            # + feature_group_naive_3
            + ["includes_weekend"]
            + ["is_weekday"]
            +["occupancy_period"]
            # +["start_hour"] # better for energy
            +["tri_tariff"] # better for energy
            # + ["tx_lag"]
            # + ["tx_lag_energy"] # better for energy
            + [target_feature]
        )
    else:
        final_features = (
            feature_group_calendar
            + [lowest_mse_column]
            # feature_special_group_3
            # + feature_group_calendar
            # + feature_common_group_1
            # + feature_common_group_2
            # + feature_discretization
            # + feature_group_naive_1
            # + feature_group_naive_2
            # + feature_group_naive_3
            + ["includes_weekend"]
            + ["occupancy_period"]
            # +["start_hour"] # better for energy
            # +["tri_tariff"] # better for energy
            # +["tx_lag"]
            # + ["tx_lag_energy"] # better for energy
            + [target_feature]
        )
    opt_features_selected = final_features

    return opt_features_selected


@logger.catch(level="ERROR", message="Error preparing dataset for forecast:")
def prepare_dataset(dataframe, target):
    """
    Prepares the dataset for training or forecasting by splitting it into features (X) and target (y).

    :param dataframe: pd.DataFrame - The input dataset.
    :param target: str - The target variable.

    :return: tuple (X, y) - The feature matrix (X) and the target array (y).
    """
    df = dataframe.copy()
    Xfs = df.copy()
    yfs = Xfs[target].copy()
    logger.debug(f"Target dataset: {yfs}")
    logger.debug(yfs.info())

    Xfs = Xfs.drop(target, axis=1)
    X = Xfs.values  # convert to array
    y = yfs.values
    return X, y

@logger.catch(level="ERROR", message="Error converting prediction to dictionary:")
def predict_proba_as_dict(model, X):
    """
    A wrapper for model.predict_proba to return probabilities as a dictionary.

    
    param model: Trained machine learning model with `predict_proba`.
    param X: Test data for predictions.
    return:    List of dictionaries where each dictionary maps class labels to probabilities.
    """
    # Get class labels
    classes = model.classes_

    # Predict probabilities
    y_pred_proba = model.predict_proba(X)

    # Convert each prediction's probabilities to a dictionary
    y_pred_proba_dict = [
        {cls: prob for cls, prob in zip(classes, probs)} for probs in y_pred_proba
    ]

    return y_pred_proba_dict
@logger.catch(level="ERROR", message="Error calculating forecasting:")
def forecast_model(dataframe, model, target_feature):
    """
    Predicts the target variable using the provided model and prepared dataset.

    :param dataframe: pd.DataFrame - The input dataset for prediction.
    :param model: Model - The machine learning model used for forecasting.
    :param target_feature: str - The target feature to predict.

    :return: np.array - The forecasted values.
    """
    df = dataframe.copy()
    X, y = prepare_dataset(dataframe=df, target=target_feature)
    logger.debug("Input X shape: {}".format(X.shape))
    logger.debug("Input y shape: {}".format(y.shape))
    forecast = model.predict(X)
    if target_feature == "day_next_plug_in":
        y_pred_proba = predict_proba_as_dict(model=model, X=X)
        return y_pred_proba

    logger.debug(f"forecast:{forecast}")
    logger.debug(f"forecast type:{type(forecast)}")
    return forecast


@logger.catch(level="ERROR", message="Error data cleaning:")
def data_cleaning(dataframe, target_feature):
    """
    Cleans the dataset by removing outliers that exceed three standard deviations from the median.

    :param dataframe: pd.DataFrame - The input dataset.
    :param target_feature: str - The feature for outlier detection (default: 'duration').

    :return: pd.DataFrame - The cleaned dataset without outliers.
    """
    df = dataframe.copy()
    dataset_std = df[target_feature].std()
    dataset_median = df[target_feature].median()
    dataset_limit = dataset_median + dataset_std * 3
    client_dataset = df[df[target_feature] <= dataset_limit].reset_index(drop=True)
    return client_dataset


@logger.catch(level="ERROR", message="Error creating xgb cv:")
def create_xgb_cv(X_train, y_train):
    """
    Creates an XGBoost cross-validation function for tuning hyperparameters.

    :param X_train: np.array - Training feature matrix.
    :param y_train: np.array - Training target array.

    :return: function - A cross-validation function for use in hyperparameter tuning.
    """

    def xgb_cv(
        learning_rate,
        n_estimators,
        max_depth,
        min_child_weight,
        gamma,
        subsample,
        colsample_bytree,
        reg_alpha,
        reg_lambda,
    ):
        """
        Inner function for performing XGBoost cross-validation with the provided hyperparameters.

        :param learning_rate: float - Learning rate for XGBoost.
        :param n_estimators: int - Number of trees in the XGBoost model.
        :param max_depth: int - Maximum tree depth.
        :param min_child_weight: float - Minimum sum of instance weight needed in a child.
        :param gamma: float - Minimum loss reduction required to make a further partition.
        :param subsample: float - Subsample ratio of the training instance.
        :param colsample_bytree: float - Subsample ratio of columns when constructing each tree.
        :param reg_alpha: float - L1 regularization term on weights.
        :param reg_lambda: float - L2 regularization term on weights.

        :return: float - Mean cross-validated negative mean squared error.
        """

        params = {
            "learning_rate": learning_rate,
            "n_estimators": int(n_estimators),
            "max_depth": int(max_depth),
            "min_child_weight": min_child_weight,
            "gamma": gamma,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "objective": "reg:squarederror",
        }
        model = xgb.XGBRegressor(**params)

        return cross_val_score(model, X_train, y_train, cv=5, scoring="r2").mean()

    return xgb_cv


@logger.catch(level="ERROR", message="Error in hyperameter optimization")
def hyperparameter_optimization(X_train, y_train, optimization_check="yes"):
    """
    Function that optimizes the hyperparameters

    :param X_train: dataframe - Features for training
    :param y_train: dataframe - Target features of training
    :param optimization_check: str - Str to optimize

    :return: df_param - dataframe - best parameters
    :return: best_param - list - best params

    """
    # Define the bounds for Bayesian Optimization
    bounds = {
        "learning_rate": (0.01, 0.3),
        "n_estimators": (50, 500),
        "max_depth": (1, 10),
        "min_child_weight": (1, 40),
        "gamma": (0.1, 1),
        "subsample": (0.1, 1),
        "colsample_bytree": (0.1, 1),
        "reg_alpha": (0.1, 20),
        "reg_lambda": (0.1, 200),
    }

    # Create the xgb_cv function using the closure
    xgb_cv = create_xgb_cv(X_train, y_train)

    if optimization_check == "yes":
        # Create the Bayesian Optimizer
        optimizer = BayesianOptimization(f=xgb_cv, pbounds=bounds, random_state=42)

        # Optimizer
        n_iterations = 250
        optimizer.maximize(init_points=5, n_iter=n_iterations)

        # Best hyperparameters
        best_params = optimizer.max["params"]
        best_params["n_estimators"] = int(best_params["n_estimators"])
        best_params["max_depth"] = int(best_params["max_depth"])

        # Create a DataFrame from the best_params dictionary
        df_param = pd.DataFrame(
            list(best_params.items()), columns=["Parameter", "Value"]
        )

    return df_param, best_params


@logger.catch(level="ERROR", message="Error cross validation:")
def cross_validation(X, y, best_params, check=False):
    """
    Perform cross-validation using the given features, target, and best hyperparameters.

    :param X: np.array - The feature matrix.
    :param y: np.array - The target array.
    :param best_params: dict - Best hyperparameters for XGBoost.

    :return: tuple - Average MAE and RMSE across the folds.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )

    X_folds, y_folds = k_fold_split(X_train, y_train)

    mae_list = []
    mse_list = []
    logger.debug(f"y_train :{y_train.shape}")
    logger.debug(f"X_train in cv :{X_train.shape}")

    for i in range(5):

        # Concatenate the other folds to form the training set
        X_train_cv = np.concatenate([X_folds[j] for j in range(5) if j != i])
        y_train_cv = np.concatenate([y_folds[j] for j in range(5) if j != i])
        X_val_cv = X_folds[i]
        y_val_cv = y_folds[i]

        # Set evals for this fold
        evals_fold = [(X_train_cv, y_train_cv), (X_val_cv, y_val_cv)]
        # model training cv

        mae, mse = cv_train_and_test(
            X_train_cv, y_train_cv, X_test, y_test, best_params, check, evals_fold
        )

        logger.debug(f"mae:{mae}")
        mae_list.append(mae)
        mse_list.append(mse)

    # Average of the metrics during CV
    avg_mae_cv = np.mean(mae_list)
    avg_rmse_cv = np.mean(mse_list) ** 0.5
    return avg_mae_cv, avg_rmse_cv


logger.catch(level="ERROR", message="Error creating features:")


def feature_engineering_time(dataframe):
    """
    Perform feature engineering on a dataset containing EV charging session data.

    This function processes the input dataframe by creating new features that can be useful for
    forecasting EV charging sessions. The features are derived from the start and end times of
    each charging session, along with other time-related information.

    Parameters:
    ----------
    dataframe : pd.DataFrame - dataframe with EV charging session data
    is_history : bool . Boolean variable to indetify if is the new session or the historic data

    returns : pd.DataFrame - dataframe with time variables
    """

    # copy of the dataset
    data = dataframe.copy()

    data["start_time"] = pd.to_datetime(data["start_datetime"], errors="coerce")
    data = data[data["start_time"].notna()]

    logger.debug(f"start_time:{data['start_datetime']}")
    data = data.sort_values(by="start_time")
    data.reset_index(drop=True, inplace=True)

    # # calculates how long since last recharge
    # for i in np.arange(1, len(data)-1):  # -1 to not go for last row (no prior information)
    #     delta_tx = data['start_time'].iloc[i] - data['end_time'].iloc[i - 1]
    #     logger.debug(f'delta_tx: {delta_tx}')
    #     data['time_since_last_tx'][i] = abs(delta_tx.total_seconds()) / 3600

    # calendar variables
    data["start_year"] = data["start_time"].apply(lambda x: x.year)
    data["start_month"] = data["start_time"].apply(lambda x: x.month)
    data["start_day"] = data["start_time"].apply(lambda x: x.day)
    data["start_hour"] = data["start_time"].apply(lambda x: x.hour)
    data["start_minute"] = data["start_time"].apply(lambda x: x.minute)
    data["end_day"] = data["end_datetime"].apply(lambda x: x.day)
    data["end_hour"] = data["end_datetime"].apply(lambda x: x.hour)

    data["day_of_week"] = data["start_time"].dt.dayofweek
    data["is_weekday"] = data["day_of_week"].apply(lambda x: is_weekday(x))
    data["includes_weekend"] = data.apply(
        lambda row: includes_weekend(row["start_datetime"], row["end_datetime"]), axis=1
    )

    data["tri_tariff"] = data["start_time"].apply(lambda x: tri_tariff(x))
    data["day_period"] = data["start_time"].apply(lambda x: day_period(x))
    data["occupancy_period"] = data["start_time"].apply(lambda x: occupancy_period(x))
    data["var_20min"] = data["start_time"].apply(
        lambda x: var_discretization(x, "20min")
    )
    data["var_40min"] = data["start_time"].apply(
        lambda x: var_discretization(x, "40min")
    )
    data["var_1h"] = data["start_time"].apply(lambda x: var_discretization(x, "1h"))
    data["var_2h"] = data["start_time"].apply(lambda x: var_discretization(x, "2h"))
    data["var_4h"] = data["start_time"].apply(lambda x: var_discretization(x, "4h"))
    logger.debug(data.head(5))
    return data


def feature_engineering_naive(dataframe, target_feature):
    """
    Perform naive feature engineering on a dataset for forecasting purposes.

    This function processes the input DataFrame by creating several naive forecasting features.
    It computes various naive statistics (mean, median, and persistence) based on the target feature.
    These features are useful as baseline models or input to more sophisticated forecasting models.

    Parameters:
    ----------
    dataframe : pd.DataFrame - dataframe the input .
    target_feature : feature for naive calculation - duration or total_energy_transfered

    returns : pd.DataFrame - dataframe with naive calculation
    """
    # copy of the dataset
    data = dataframe.copy()
    for i in range(data.shape[0]):
        logger.info("start cicle")
        if i == 0:
            logger.info("first_row")
            data.loc[i, "naive_avg"] = 0  # first row shall be deleted later
            data.loc[i, "naive_avg_row5"] = 0  # first row shall be deleted later
            data.loc[i, "naive_avg_period"] = 0  # first row shall be deleted later
            data.loc[i, "naive_med"] = 0  # first row shall be deleted later
            data.loc[i, "naive_med_row5"] = 0  # first row shall be deleted later
            data.loc[i, "naive_med_period"] = 0  # first row shall be deleted later
            data.loc[i, "persistence"] = 0  # first row shall be deleted later

        else:
            logger.info("else")
            logger.info("calculating persitence")
            logger.debug(f"target_feature {target_feature}")
            data.iloc[i, data.columns.get_loc("persistence")] = data[
                target_feature
            ].iloc[i - 1]
            logger.success("Persistence calcualted correctly")
            data.iloc[i, data.columns.get_loc("naive_avg")] = (
                data[target_feature].iloc[:i].mean()
            )
            data.iloc[i, data.columns.get_loc("naive_med")] = (
                data[target_feature].iloc[:i].median()
            )

        logger.info("creation_naive_avg")
        data["naive_avg_row5"] = (
            data[target_feature].rolling(window=5, min_periods=1).mean().shift(1)
        )  # noqa
        data["naive_max_row5"] = (
            data[target_feature].rolling(window=5, min_periods=1).max().shift(1)
        )  # noqa
        data["naive_min_row5"] = (
            data[target_feature].rolling(window=5, min_periods=1).min().shift(1)
        )  # noqa
        data["naive_med_row5"] = (
            data[target_feature].rolling(window=5, min_periods=1).median().shift(1)
        )  # noqa

    # new variable Naive sum rolling -
    # Calculate the cumulative mean and median per day_period
    # Initialize the columns with NaN or 0
    logger.info("initializing naive period")

    data["naive_avg_period"] = np.nan
    data["naive_med_period"] = np.nan
    data["day_period"] = data["start_datetime"].apply(
        lambda x: day_period(x)
    )  # categorical datetime feature

    logger.info("Calculation period naive")
    # Loop over each day_period
    for period in data["day_period"].unique():
        # Select the rows for the current period
        logger.debug(f"period {period}")
        mask = data["day_period"] == period
        logger.debug(f"mask {period}")

        # Calculate expanding mean/median for the current period,
        # excluding the current row
        logger.debug("mean calculation")
        data.loc[mask, "naive_avg_period"] = (
            data.loc[mask, target_feature].expanding().mean().shift(1)
        )
        logger.debug("mean calculation")
        data.loc[mask, "naive_med_period"] = (
            data.loc[mask, target_feature].expanding().median().shift(1)
        )

    logger.info("naive variables")
    # fill the NaN values with 0 - might need to keep nan and exclude
    data.fillna(0, inplace=True)
    data["persistence_rename"] = data["persistence"]
    data.rename(columns={"persistence_rename": "tx_lag"}, inplace=True)
    day_period_list = ["dp_0", "dp_1", "dp_2", "dp_3", "dp_4"]
    day_period_category_dict = {
        category: index for index, category in enumerate(day_period_list)
    }
    data["day_period"] = data["day_period"].apply(
        lambda x: day_period_category_dict.get(x)
    )

    mse_naive_avg = mean_squared_error(data[target_feature], data["naive_avg"]) ** 0.05
    mse_naive_avg_row5 = (
        mean_squared_error(data[target_feature], data["naive_avg_row5"]) ** 0.05
    )
    mse_naive_med = mean_squared_error(data[target_feature], data["naive_med"]) ** 0.05
    mse_naive_med_row5 = (
        mean_squared_error(data[target_feature], data["naive_med_row5"]) ** 0.05
    )
    mse_naive_med_period = (
        mean_squared_error(data[target_feature], data["naive_med_period"]) ** 0.05
    )
    mse_naive_avg_period = (
        mean_squared_error(data[target_feature], data["naive_avg_period"]) ** 0.05
    )
    mse_persistence = (
        mean_squared_error(data[target_feature], data["persistence"]) ** 0.05
    )

    logger.debug(f"MSE for naive_avg: {mse_naive_avg}")
    logger.debug(f"MSE for naive_avg_row5: {mse_naive_avg_row5}")
    logger.debug(f"MSE for naive_med: {mse_naive_med}")
    logger.debug(f"MSE for naive_med_row5: {mse_naive_med_row5}")
    logger.debug(f"MSE for naive_med_period: {mse_naive_med_period}")
    logger.debug(f"MSE for naive_avg_period: {mse_naive_avg_period}")
    logger.debug(f"MSE for naive_avg_row5: {mse_naive_avg_row5}")
    logger.debug(f"MSE for persitence: {mse_persistence}")

    mse_values = {
        "naive_avg": mse_naive_avg,
        "naive_avg_row5": mse_naive_avg_row5,
        "naive_med": mse_naive_med,
        "naive_med_row5": mse_naive_med_row5,
        "naive_med_period": mse_naive_med_period,
        "naive_avg_period": mse_naive_avg_period,
        "persistence": mse_persistence,
    }

    # Find the column with the lowest MSE
    logger.info("lowest column with min mse error")

    lowest_mse_column = min(mse_values, key=mse_values.get)
    logger.debug(f"column with lowest mse: {lowest_mse_column}")

    # Drop the columns that do not correspond to the lowest MSE
    columns_to_drop = [
        column for column in mse_values.keys() if column != lowest_mse_column
    ]
    logger.debug(f"drop columns: {columns_to_drop}")

    # data.drop(columns=columns_to_drop, inplace=True)
    df = data.copy()
    duplicates = df.duplicated(subset=["session_id", "evse_id"], keep=False)
    if duplicates.any():
        logger.debug(f"Duplicates found in final df:\n {df[duplicates]}")
    logger.debug(f"forecast: {forecast}")
    return data, lowest_mse_column


@logger.catch(level="ERROR", message="Error computing consistent sorted return day offsets for last row:")
def filtered_sorted_return_day_offsets_for_last_row(df, start_col='start_datetime'):
    """
    For the final row, computes a filtered, descending-frequency list of return day offsets (integers),
    based on past rows that started on the same weekday. Any offset that breaks descending order of frequency
    is excluded.

    :param df: DataFrame with a datetime column.
    :param start_col: name of the datetime column.
    :returns: List of filtered day offsets as integers.
    """
    df = df.copy()
    df[start_col] = pd.to_datetime(df[start_col])
    df.sort_values(by=start_col, inplace=True)
    df.reset_index(drop=True, inplace=True)

    df['start_weekday'] = df[start_col].dt.weekday
    df['return_datetime'] = df[start_col].shift(-1)
    df['day_offset'] = (df['return_datetime'] - df[start_col]).dt.days
    previous_rows = df.iloc[:-1]
    final_start_weekday = df.iloc[-1]['start_weekday']
    filtered = previous_rows[previous_rows['start_weekday'] == final_start_weekday]
    offset_counts = filtered['day_offset'].value_counts()

    # Sort offsets by frequency descending, then by offset value ascending
    sorted_offsets = offset_counts.sort_values(ascending=False)
    valid_offsets = []
    last_offset = None

    for offset, count in sorted_offsets.items():
        if pd.isnull(offset):
            continue
        if last_offset is None or offset >= last_offset:
            valid_offsets.append(int(offset))
            last_offset = offset
    return valid_offsets

def filtered_sorted_return_day_offsets_for_row(index, df, start_col='start_datetime'):
    """
    For the given row index, computes a filtered list of return day offsets (integers),
    based on previous rows that started on the same weekday. Offsets are ordered by
    descending frequency, breaking ties by ascending offset. Stops if order of frequency breaks.

    :param index: Index of the row to compute for.
    :param df: Full DataFrame with datetime column.
    :param start_col: Name of datetime column.
    :returns: List of filtered day offsets as integers.
    """
    if index == 0:
        return []

    df = df.copy()
    df[start_col] = pd.to_datetime(df[start_col])
    df.sort_values(by=start_col, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Limit to previous rows only
    current_row = df.iloc[index]
    previous_rows = df.iloc[:index]

    previous_rows['start_weekday'] = previous_rows[start_col].dt.weekday
    previous_rows['return_datetime'] = previous_rows[start_col].shift(-1)
    previous_rows['day_offset'] = (previous_rows['return_datetime'] - previous_rows[start_col]).dt.days

    current_weekday = current_row[start_col].weekday()
    filtered = previous_rows[previous_rows['start_weekday'] == current_weekday]

    offset_counts = filtered['day_offset'].value_counts()
    sorted_offsets = offset_counts.sort_values(ascending=False)

    valid_offsets = []
    last_offset = None

    for offset, count in sorted_offsets.items():
        if pd.isnull(offset):
            continue
        if last_offset is None or offset >= last_offset:
            valid_offsets.append(int(offset))
            last_offset = offset
    return valid_offsets


@logger.catch(level="ERROR", message="Error most probable return weekday:")
def rowwise_most_common_return_weekday(df, start_col='start_datetime'):
    """
    For each row, computes the most common return weekday (as an integer 0–6)
    based on past rows that started on the same weekday as the current row.

    :param df: DataFrame with a datetime column.
    :param start_col: name of the datetime column.
    :returns: The original DataFrame with a new column 'most_probable_return_day' (int weekday).
    """
    df = df.copy()
    df[start_col] = pd.to_datetime(df[start_col])
    df.sort_values(by=start_col, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Add numeric weekday: 0=Monday, 6=Sunday
    df['start_weekday'] = df[start_col].dt.weekday

    # Compute return datetime
    df['return_datetime'] = df[start_col].shift(-1)
    df['return_weekday'] = df['return_datetime'].dt.weekday

    most_probable_weekday = []

    for i in range(len(df)):
        current_day = df.loc[i, 'start_weekday']
        previous_rows = df.loc[:i-1]

        # Filter previous rows that started on the same weekday
        filtered = previous_rows[previous_rows['start_weekday'] == current_day]
        mode = filtered['return_weekday'].mode()
        most_common = mode.iloc[0] if not mode.empty else None
        most_probable_weekday.append(most_common)

    df['most_probable_return_day'] = most_probable_weekday

    # Drop helper columns if needed
    df.drop(columns=['return_datetime', 'return_weekday', 'start_weekday'], inplace=True)

    return df

@logger.catch(level="ERROR", message="Error training forecasting model:")
def train_model(dataframe, target_feature, user_id):
    """
    Train the XGBoost model after data cleaning, feature selection, and hyperparameter optimization.

    :param dataframe: pd.DataFrame - Input dataset for training.
    :param target_feature: str - The target variable to predict.

    :return: None - The trained model and hyperparameters are saved.
    """
    df = dataframe.copy()
    df = df[df["user_id"] == user_id].copy().reset_index(drop=True)
    df = calculation_delta_startime(dataframe=df)
    df = calculation_time_next_plug_in(dataframe=df)
    df = compute_next_hour(dataframe=df,column_date='start_datetime')
    df = rowwise_most_common_return_weekday(df)

    df["next_start_datetime"] = df["start_datetime"].shift(-1)
    df["return_day"] = df.apply(
        lambda row: get_day_offset(row["start_datetime"], row["next_start_datetime"]),
        axis=1,
    )
    df.drop(columns=["next_start_datetime"], inplace=True)

    # Time difference between consecutive start_datetime entries in hours
    df["time_diff_hours"] = (
        df["start_datetime"] - df["start_datetime"].shift(1)
    ).dt.total_seconds() / 3600

    # Filter rows where the time difference is greater than or equal to 2 hours
    # -- Avoids issues with sessions that are too close together
    # -- This threshold can be adjusted based on domain knowledge
    filtered_df = df[(df["time_diff_hours"] >= 2)].copy()

    # Drop the helper column if not needed
    filtered_df.drop(columns=["time_diff_hours"], inplace=True)

    df_features = feature_engineering_time(dataframe=filtered_df)
    logger.debug(df_features[["start_datetime", "day_period", "var_20min"]])
    df_all, lowest_mse_column = feature_engineering_naive(
        dataframe=df_features, target_feature=target_feature
    )
    file_path_normalize = os.path.join(
        "files",
        "models",
        "Normalization",
        "Training",
        f"{user_id}_{target_feature}_normalization.json",
    )

    df_features_normalized = data_normalization(
        dataframe=df_all, target_feature=target_feature, save_path=file_path_normalize
    )
    if target_feature == "time_next_plug_in":
        df_features_normalized = df_features_normalized.drop(columns=["duration"])

    # logger.info('Data Cleaning...')
    df_cleaned = data_cleaning(
        dataframe=df_features_normalized, target_feature=target_feature
    )
    if target_feature == "total_energy_transfered":
        df_sum = df_cleaned.copy()
        df_sum["start_datetime_aux"] = pd.to_datetime(df_sum["start_datetime"])

        # Initialize a column to store the sum
        df_sum["Sum_Last_72_Hours"] = 0

        # Compute sum of values for the last 72 hours using itertuples
        for i, row in enumerate(df_sum.itertuples(index=False)):
            current_time = row.start_datetime_aux
            # Filter the rows within the 72-hour window
            mask = (df_sum["start_datetime_aux"] <= current_time) & (
                df_sum["start_datetime_aux"] > current_time - timedelta(hours=72)
            )
            df_sum.loc[i, "Sum_Last_72_Hours"] = df_sum.loc[
                mask, "total_energy_transfered"
            ].sum()
        df_cleaned = df_sum.copy()
    df_features_cleaned = feature_selection(
        dataframe=df_cleaned,
        target_feature=target_feature,
        lowest_mse_column=lowest_mse_column,
    )
    logger.debug(df_features_cleaned)
    df_cleaned = df_cleaned[df_features_cleaned]

    logger.info("Cleaned data")
    logger.debug(df_cleaned.head(5))
    logger.info("Preparing dataset for ML models")
    logger.debug("Info data cleaned")
    logger.debug(df_cleaned.describe())

    # X and y for training
    X, y = prepare_dataset(dataframe=df_cleaned, target=target_feature)

    logger.info("Dataset ready for ML models")
    logger.info("Determining best hyperparameters for model")
    logger.debug(f"target feature {target_feature}")
    logger.debug("Input X shape: {}".format(X.shape))
    logger.debug("Input y shape: {}".format(y.shape))

    best_params_df, best_params = hyperparameter_optimization(
        X_train=X, y_train=y
    )

    logger.info("Training XGBoost model with best hyperparameters")
    mvp_xgboost = xgb.XGBRegressor(**best_params, random_state=42)
    mvp_xgboost.fit(X, y)
    logger.success("Training XGBoost model with best hyperparameters ... Ok!")

    save_model(
        model_params=mvp_xgboost,
        path_folder="ML models",
        target_feature=target_feature,
        user_id=user_id,
        file_type="joblib",
        define_file="trained",
    )

    file_path_metrics = os.path.join("files", "models", "Metrics")
    file_name_parameters = f"{user_id}_{target_feature}_parameters.json"

    file_path = os.path.join(file_path_metrics, file_name_parameters)
    save_json(json_data=best_params, file_path=file_path)


@logger.catch(level="ERROR", message="Error calculating forecast:")
def forecast(dataframe, target_feature, path_folder, launch_time=None, return_day_list=None):
    """
    Generate forecasts using a pre-trained model or train a new model if not available.

    :param dataframe: pd.DataFrame - The input dataset for forecasting.
    :param target_feature: str - The target variable to forecast.
    :param path_folder: str - The folder path where the model is stored.

    :return: np.array - The forecasted values.
    """
    # Get user id from dataframe
    df = dataframe.copy()
    if return_day_list is not None:
        return_day_list_=return_day_list
        if launch_time.hour > 14 and 0 in return_day_list_:
            return_day_list_.remove(0)
        if not return_day_list_:
            return_day_list_=[1]
    else:
        return_day_list_=[]
    # df = rowwise_most_common_return_weekday(df)
    logger.info("calculating user_id")
    user_id = df.at[0, "user_id"]
    logger.info("calculating if")
    if target_feature == "time_next_plug_in":
        df = df.drop(
            columns=["duration"],
        )
    logger.debug(f"database: \n {df}")
    logger.info("Calculating time features...")
    df_feature_time = feature_engineering_time(dataframe=df)
    logger.success("Time features computed successfully ")
    logger.info("Calculating naive features...")
    df_all_featues, lowest_mse_column = feature_engineering_naive(
        dataframe=df_feature_time, target_feature=target_feature
    )
    logger.success("Naive features computed successfully ")

    logger.info("Normalizing Data...")
    file_path_normalize = os.path.join(
        "files",
        "models",
        "Normalization",
        "Forecast",
        f"{user_id}_{target_feature}_normalization.json",
    )

    if len(df_all_featues) > 1:
        df_features_normalized = data_normalization(
            dataframe=df_all_featues,
            target_feature=target_feature,
            save_path=file_path_normalize,
        )
    else:
        df_features_normalized = df_all_featues

    logger.debug(f"df_features_normalized:\n {df_features_normalized.info()}")
    logger.success("Data normalized successfully ")

    logger.info("loading model...")
    xgboost_model = load_model(
        path_folder=path_folder, target_feature=target_feature, user_id=user_id
    )
    logger.success("Load model sucessfull ...")
    
    if target_feature=='hour_minute':
        
        results = []

        for return_day in return_day_list_:
            session_information = df_features_normalized.tail(1).copy()
            
            # Assign the current return day to the session
            session_information["return_day"] = return_day
            start_date = launch_time.date()
            logger.info("Feature selection...")
            df_features = feature_selection(
                dataframe=session_information,
                target_feature=target_feature,
                lowest_mse_column=lowest_mse_column,
            )
            logger.success("Feature selection was a success")

            session_information = session_information[df_features]

            logger.info("Forecasting machine learning model...")
            forecast = forecast_model(
                dataframe=session_information,
                model=xgboost_model,
                target_feature=target_feature,
            )
            return_date = start_date + timedelta(days=return_day)
            # Convert forecast to HH:MM string if it's a float (e.g., 13.5 = 13:30)
            if isinstance(forecast[0], (float, int,np.floating)):
                hours = int(forecast[0])
                minutes = int(round((forecast[0] - hours) * 60))
                forecast_time_str = f"{hours:02d}:{minutes:02d}"
            else:
                forecast_time_str = str(forecast[0])  # already a string like "13:30"

            # Combine return day and forecast time into datetime
            try:
                # Step 3: Combine return_date and forecast_time into full datetime
                forecast_datetime_str = f"{return_date.strftime('%d/%m/%Y')} {forecast_time_str}"

                # Final datetime
                full_forecast_datetime = pd.to_datetime(forecast_datetime_str, format="%d/%m/%Y %H:%M")
            except Exception as e:
                logger.error(f"Error combining return_day and forecast time: {e}")
                full_forecast_datetime = None

            # Store the result
            results.append({
                "return_day": return_day,
                "forecast_time": forecast_time_str,
                "Next Session - Forecast ML": full_forecast_datetime
            })
        return results
    else:
        session_information = df_features_normalized.tail(1)
        logger.info("Feature selection...")
        df_features = feature_selection(
            dataframe=session_information,
            target_feature=target_feature,
            lowest_mse_column=lowest_mse_column,
        )
        logger.success("Feature selection was a success")
        session_information = session_information[df_features]
        # logger.debug(f'session information: {session_information['time_since_last_tx']}')
        logger.info("Forecasting machine learning model...")
        forecast = forecast_model(
            dataframe=session_information,
            model=xgboost_model,
            target_feature=target_feature,
        )
        # if target_feature=='duration':
        #     forecast = np.expm1(forecast)

        logger.success("Forecasting machine learning model was a success")
        logger.debug(f"forecast value:{forecast}")

        if target_feature == "day_next_plug_in":
            return forecast
        else:
            return float(forecast)

def add_return_date_feature(df):
    """
    Adds a 'return_date' column to the DataFrame based on the most probable return day offset.

    For each row, the return date is calculated by adding the 'most_probable_return_day' value
    (interpreted as a day offset) to the 'start_datetime'. If the value is 8 (a special case, 
    possibly indicating "after D+7" or undefined return timing), a fixed offset of 8 days is used,
    though this logic can be customized.

    Assumes the DataFrame contains:
    - 'start_datetime': the original date of the event (as a datetime)
    - 'most_probable_return_day': the predicted or derived return offset in days

    :param df: Input DataFrame containing 'start_datetime' and 'most_probable_return_day'
    :return: DataFrame with a new 'return_date' column
    """
    # Calculate the return date based on the most probable return day
    def calculate_return_date(row):
        most_probable_day = row['most_probable_return_day']
        start_datetime = row['start_datetime']
        
        if most_probable_day == 8:
            # You can define your own logic here, like adding 14 days if it's 'after D+7'
            return start_datetime + pd.Timedelta(days=8)  # Example: 14 days after
        else:
            return start_datetime + pd.Timedelta(days=most_probable_day)

    # Apply the function to calculate the return date
    df['return_date'] = df.apply(calculate_return_date, axis=1)

    return df

@logger.catch(level="ERROR", message="Error naive models forecast:")
def naive_simple_models_forecast_validation(data, target):
    """
    Compute naive models (naive average, naive median, persistence) and evaluate their MSE.

    :param data: pd.DataFrame - Input DataFrame containing the data.
    :param target_feature: str - Name of the target feature column in the DataFrame.
    :return: float - prevision based on the naive model with lowest MSE.

    """
    logger.info(f"Calculation of naive {target}")
    data_df = data.copy()
    target_feature = target
    data_df["start_datetime"] = pd.to_datetime(data_df["start_time"], errors="coerce")
    logger.debug(type(data_df["start_datetime"]))
    data_df["includes_weekend"] = data_df.apply(
        lambda row: includes_weekend(row["start_datetime"], row["end_datetime"]), axis=1
    )
    data_df = data_df[data_df["start_datetime"].notna()]

    data_df = data_df.sort_values(by="start_datetime")
    # TXCX_dataset[var].reset_index(drop=True, inplace=True)
    # data['end_datetime'] = pd.to_datetime(data['end_time'], errors='coerce')

    # data['duration'] = (data['end_datetime'] - data[
    #     'start_datetime']).dt.total_seconds() / 60

    logger.info("Initializing naive models")

    for i in range(data_df.shape[0]):
        logger.info("start cicle")
        logger.debug(range(data_df.shape[0]))
        if i == 0:
            logger.info("first_row")
            data_df.loc[i, "naive_avg"] = 0  # first row shall be deleted later
            data_df.loc[i, "naive_avg_row5"] = 0  # first row shall be deleted later
            data_df.loc[i, "naive_avg_period"] = 0  # first row shall be deleted later
            data_df.loc[i, "naive_med"] = 0  # first row shall be deleted later
            data_df.loc[i, "naive_med_row5"] = 0  # first row shall be deleted later
            data_df.loc[i, "naive_med_period"] = 0  # first row shall be deleted later
            data_df.loc[i, "persistence"] = 0  # first row shall be deleted later

        else:
            logger.info("else")
            logger.info("calculating persitence")
            logger.debug(f"target_feature {target_feature}")
            data_df.iloc[i, data_df.columns.get_loc("persistence")] = data_df[
                target_feature
            ].iloc[i - 1]
            logger.success("Persistence calcualted correctly")
            data_df.iloc[i, data_df.columns.get_loc("naive_avg")] = (
                data_df[target_feature].iloc[:i].mean()
            )
            data_df.iloc[i, data_df.columns.get_loc("naive_med")] = (
                data_df[target_feature].iloc[:i].median()
            )

        logger.info("creation_naive_avg")
        data_df["naive_avg_row5"] = (
            data_df[target_feature].rolling(window=5, min_periods=1).mean().shift(1)
        )  # noqa
        data_df["naive_med_row5"] = (
            data_df[target_feature].rolling(window=5, min_periods=1).median().shift(1)
        )  # noqa

    duplicates = data_df.duplicated(subset=["session_id", "evse_id"], keep=False)
    if duplicates.any():
        logger.debug(
            f"Duplicates found in df_naive_energy_value before window:\n {data_df[duplicates]}"
        )
    logger.info("creation_naive_avg")
    data_df["naive_avg_row5"] = (
        data_df[target_feature].rolling(window=5, min_periods=1).mean().shift(1)
    )  # noqa
    data_df["naive_med_row5"] = (
        data_df[target_feature].rolling(window=5, min_periods=1).median().shift(1)
    )  # noqa
    data_df = data_df.drop(index=0).reset_index(drop=True)

    # Calculate the cumulative mean and median per day_period
    # Initialize the columns with NaN or 0
    duplicates = data_df.duplicated(subset=["session_id", "evse_id"], keep=False)
    if duplicates.any():
        logger.debug(
            f"Duplicates found in df_naive_energy_value:\n {data_df[duplicates]}"
        )
    logger.info("initializing naive period")

    data_df["naive_avg_period"] = np.nan
    data_df["naive_med_period"] = np.nan
    data_df["day_period"] = data["start_datetime"].apply(
        lambda x: day_period(x)
    )  # categorical datetime feature

    logger.info("Calculation period naive")
    # Loop over each day_period
    for period in data_df["day_period"].unique():
        # Select the rows for the current period
        logger.debug(f"period {period}")
        mask = data_df["day_period"] == period
        logger.debug(f"mask {period}")

        # Calculate expanding mean/median for the current period,
        # excluding the current row
        logger.debug("mean calculation")
        data_df.loc[mask, "naive_avg_period"] = (
            data_df.loc[mask, target_feature].expanding().mean().shift(1)
        )
        logger.debug("mean calculation")
        data_df.loc[mask, "naive_med_period"] = (
            data_df.loc[mask, target_feature].expanding().median().shift(1)
        )

    duplicates = data_df.duplicated(subset=["session_id", "evse_id"], keep=False)
    if duplicates.any():
        logger.debug(f"Duplicates found in period:\n {data_df[duplicates]}")
    logger.info("initializing naive period")
    logger.debug(data.duplicated(subset=["session_id", "evse_id"], keep=False))
    logger.info("naive variables")
    # fill the NaN values with 0 - might need to keep nan and exclude
    data_df.fillna(0, inplace=True)
    logger.info("fill na.")
    duplicates = data_df.duplicated(subset=["session_id", "evse_id"], keep=False)
    if duplicates.any():
        logger.debug(f"Duplicates found in after filling NA:\n {data_df[duplicates]}")
    logger.info("initializing naive period")
    logger.info("mean_squared_error")
    mse_naive_avg = (
        mean_squared_error(data_df[target_feature], data_df["naive_avg"]) ** 0.05
    )
    mse_naive_avg_row5 = (
        mean_squared_error(data_df[target_feature], data_df["naive_avg_row5"]) ** 0.05
    )
    mse_naive_med = (
        mean_squared_error(data_df[target_feature], data_df["naive_med"]) ** 0.05
    )
    mse_naive_med_row5 = (
        mean_squared_error(data_df[target_feature], data_df["naive_med_row5"]) ** 0.05
    )
    mse_naive_med_period = (
        mean_squared_error(data_df[target_feature], data_df["naive_med_period"]) ** 0.05
    )
    mse_naive_avg_period = (
        mean_squared_error(data_df[target_feature], data_df["naive_avg_period"]) ** 0.05
    )
    mse_persistence = (
        mean_squared_error(data_df[target_feature], data_df["persistence"]) ** 0.05
    )

    logger.debug(f"MSE for naive_avg: {mse_naive_avg}")
    logger.debug(f"MSE for naive_avg_row5: {mse_naive_avg_row5}")
    logger.debug(f"MSE for naive_med: {mse_naive_med}")
    logger.debug(f"MSE for naive_med_row5: {mse_naive_med_row5}")
    logger.debug(f"MSE for naive_med_period: {mse_naive_med_period}")
    logger.debug(f"MSE for naive_avg_period: {mse_naive_avg_period}")
    logger.debug(f"MSE for naive_avg_row5: {mse_naive_avg_row5}")
    logger.debug(f"MSE for persitence: {mse_persistence}")

    mse_values = {
        "naive_avg": mse_naive_avg,
        "naive_avg_row5": mse_naive_avg_row5,
        "naive_med": mse_naive_med,
        "naive_med_row5": mse_naive_med_row5,
        "naive_med_period": mse_naive_med_period,
        "naive_avg_period": mse_naive_avg_period,
        "persistence": mse_persistence,
    }

    # Find the column with the lowest MSE
    logger.info("lowest column with min mse error")

    lowest_mse_column = min(mse_values, key=mse_values.get)
    logger.debug(f"column with lowest mse: {lowest_mse_column}")

    # Drop the columns that do not correspond to the lowest MSE
    columns_to_drop = [
        column for column in mse_values.keys() if column != lowest_mse_column
    ]
    logger.debug(f"drop columns: {columns_to_drop}")

    data_df.drop(columns=columns_to_drop, inplace=True)
    df = data_df
    duplicates = df.duplicated(subset=["session_id", "evse_id"], keep=False)
    if duplicates.any():
        logger.debug(f"Duplicates found in final df:\n {df[duplicates]}")
    logger.debug(f"forecast: {forecast}")
    # if target_feature=='time_since_last_tx':
    #         df["start_datetime"] = pd.to_datetime(data["start_time"], errors="coerce")
    #         df[lowest_mse_column] = df['start_datetime'].shift(1, fill_value=0) + df[lowest_mse_column]

    return df, lowest_mse_column


@logger.catch(level="ERROR", message="Error naive models forecast:")
def naive_models_forecast_validation(data, target):
    """
    Compute naive models (naive average, naive median, persistence) for both weekday and weekend data and evaluate their MSE.

    :param data: pd.DataFrame - Input DataFrame containing the data.
    :param target: str - Name of the target feature column in the DataFrame.
    :return: tuple - DataFrame with selected naive model and the name of the model with lowest MSE.
    """
    logger.info(f"Calculation of naive {target}")
    data_df = data.copy()
    target_feature = target

    # Convert start_time to datetime and flag weekend days
    data_df["start_datetime"] = pd.to_datetime(data_df["start_time"], errors="coerce")
    logger.debug(type(data_df["start_datetime"]))
    data_df["includes_weekend"] = data_df.apply(
        lambda row: includes_weekend(row["start_datetime"], row["end_datetime"]), axis=1
    )
    data_df = data_df[data_df["start_datetime"].notna()]
    data_df = data_df.sort_values(by="start_datetime")

    # Separate data into weekday and weekend sets
    weekday_df = data_df[data_df["includes_weekend"] != 0].copy()
    weekend_df = data_df[data_df["includes_weekend"] == 0].copy()

    logger.info("Initializing naive models with weekday and weekend differentiation")

    # Define placeholders for means/medians
    weekday_avg, weekday_median = 0, 0

    for i in range(data_df.shape[0]):
        if i == 0:
            # Initialize the first row with zeros, to be dropped later
            data_df.loc[i, "naive_avg"] = data_df.loc[i, "naive_med"] = data_df.loc[
                i, "persistence"
            ] = 0
            data_df.loc[i, "naive_avg_row5"] = data_df.loc[i, "naive_med_row5"] = 0
            data_df.loc[i, "naive_avg_period"] = data_df.loc[i, "naive_med_period"] = 0
        else:
            is_weekend = data_df.iloc[i]["includes_weekend"] == 0
            logger.debug(is_weekend)
            if not is_weekend:
                # Calculate weekday averages and medians
                weekday_avg = (
                    data_df[target_feature]
                    .iloc[:i][~data_df["includes_weekend"].iloc[:i].eq(0)]
                    .mean()
                )
                weekday_median = (
                    data_df[target_feature]
                    .iloc[:i][~data_df["includes_weekend"].iloc[:i].eq(0)]
                    .median()
                )
                data_df.loc[i, "persistence"] = data_df[target_feature].iloc[i - 1]
                data_df.loc[i, "naive_avg"] = data_df[target_feature].iloc[:i].mean()
                data_df.loc[i, "naive_med"] = data_df[target_feature].iloc[:i].median()
            else:
                logger.debug(f"i{i}")
                logger.debug(data_df["start_time"][i])
                logger.debug(weekend_df.index[0])
                if i == weekend_df.index[0]:
                    # For the first weekend row, use weekday cumulative average
                    logger.debug(weekday_avg)
                    logger.debug(weekday_median)

                    data_df.loc[i, "persistence"] = weekday_avg
                    data_df.loc[i, "naive_avg"] = weekday_avg
                    data_df.loc[i, "naive_med"] = weekday_median
                    logger.debug(data_df.loc[i, "naive_avg"])
                else:
                    # For other weekend rows, use prior weekend averages
                    logger.debug(weekday_avg)
                    logger.debug(weekday_median)
                    data_df.loc[i, "persistence"] = data_df[target_feature].iloc[i - 1]
                    data_df.loc[i, "naive_avg"] = (
                        data_df[target_feature].iloc[:i].mean()
                    )
                    data_df.loc[i, "naive_med"] = (
                        data_df[target_feature].iloc[:i].median()
                    )

    # Calculate separate 5-row rolling values for weekends and weekdays
    weekend_roll_avg = (
        data_df[target_feature]
        .where(data_df["includes_weekend"] == 0)
        .rolling(window=5, min_periods=1)
        .mean()
        .shift(1)
    )
    weekend_roll_median = (
        data_df[target_feature]
        .where(data_df["includes_weekend"] == 0)
        .rolling(window=5, min_periods=1)
        .median()
        .shift(1)
    )

    weekday_roll_avg = (
        data_df[target_feature]
        .where(data_df["includes_weekend"] == 1)
        .rolling(window=5, min_periods=1)
        .mean()
        .shift(1)
    )
    weekday_roll_median = (
        data_df[target_feature]
        .where(data_df["includes_weekend"] == 1)
        .rolling(window=5, min_periods=1)
        .median()
        .shift(1)
    )
    logger.debug(weekday_median)
    # Replace all NaN values in weekend_roll_avg with weekday_avg
    if weekend_roll_avg.isna().any():
        logger.debug(
            f"NaN values found in weekend_roll_avg, replacing all with {weekday_avg}"
        )
        weekend_roll_avg.fillna(weekday_avg, inplace=True)

    # Replace all NaN values in weekend_roll_median with weekday_median
    if weekend_roll_median.isna().any():
        logger.debug(
            f"NaN values found in weekend_roll_median, replacing all with {weekday_median}"
        )
        weekend_roll_median.fillna(weekday_median, inplace=True)
        # Assign rolling values based on whether it's a weekend or weekday
    data_df["naive_avg_row5"] = weekday_roll_avg.where(
        data_df["includes_weekend"] == 1, weekend_roll_avg
    )
    logger.debug(data_df[data_df["session_id"] == "0000000000000023"]["naive_avg_row5"])
    logger.debug(f"weekend_roll_value:{weekend_roll_median}")
    data_df["naive_med_row5"] = weekday_roll_median.where(
        data_df["includes_weekend"] == 1, weekend_roll_median
    )
    logger.debug(data_df[data_df["session_id"] == "0000000000000023"]["naive_med_row5"])

    logger.info("First row initialization and 5-row rolling calculations done.")

    # Initialize columns for period-based cumulative means and medians
    data_df["naive_avg_period"] = np.nan
    data_df["naive_med_period"] = np.nan
    data_df["day_period"] = data["start_datetime"].apply(lambda x: day_period(x))

    # Calculate expanding mean/median by day_period
    for period in data_df["day_period"].unique():
        mask = data_df["day_period"] == period
        data_df.loc[mask, "naive_avg_period"] = (
            data_df.loc[mask, target_feature].expanding().mean().shift(1)
        )
        data_df.loc[mask, "naive_med_period"] = (
            data_df.loc[mask, target_feature].expanding().median().shift(1)
        )

    logger.info("Expanding mean/median period-based calculations completed.")

    data_df["naive_avg_period"].fillna(data_df["naive_avg"], inplace=True)
    data_df["naive_med_period"].fillna(data_df["naive_med"], inplace=True)

    data_df.fillna(0, inplace=True)
    # MSE Calculations
    mse_naive_avg = (
        mean_squared_error(data_df[target_feature], data_df["naive_avg"]) ** 0.05
    )
    mse_naive_avg_row5 = (
        mean_squared_error(data_df[target_feature], data_df["naive_avg_row5"]) ** 0.05
    )
    mse_naive_med = (
        mean_squared_error(data_df[target_feature], data_df["naive_med"]) ** 0.05
    )
    mse_naive_med_row5 = (
        mean_squared_error(data_df[target_feature], data_df["naive_med_row5"]) ** 0.05
    )
    mse_naive_med_period = (
        mean_squared_error(data_df[target_feature], data_df["naive_med_period"]) ** 0.05
    )
    mse_naive_avg_period = (
        mean_squared_error(data_df[target_feature], data_df["naive_avg_period"]) ** 0.05
    )
    mse_persistence = (
        mean_squared_error(data_df[target_feature], data_df["persistence"]) ** 0.05
    )

    mse_values = {
        "naive_avg": mse_naive_avg,
        "naive_avg_row5": mse_naive_avg_row5,
        "naive_med": mse_naive_med,
        "naive_med_row5": mse_naive_med_row5,
        "naive_med_period": mse_naive_med_period,
        "naive_avg_period": mse_naive_avg_period,
        "persistence": mse_persistence,
    }

    # Select the column with the lowest MSE
    lowest_mse_column = min(mse_values, key=mse_values.get)
    logger.info(lowest_mse_column)
    columns_to_drop = [col for col in mse_values.keys() if col != lowest_mse_column]
    data_df.drop(columns=columns_to_drop, inplace=True)

    # # Drop the initial row and reset index
    data_df = data_df.drop(index=0).reset_index(drop=True)

    return data_df, lowest_mse_column


# @logger.catch(level="ERROR", message="Error calculating forecast:")
# def forecast_validation(dataframe, target_feature, path_folder):
#     """
#     Generate forecasts using a pre-trained model or train a new model if not available.

#     :param dataframe: pd.DataFrame - The input dataset for forecasting.
#     :param target_feature: str - The target variable to forecast.
#     :param path_folder: str - The folder path where the model is stored.

#     :return: np.array - The forecasted values.
#     """
#     # Get user id from dataframe
#     df = dataframe.copy()
#     logger.info(df.shape)
#     logger.info("calculating user_id")
#     user_id = df.at[0, "user_id"]
#     logger.info("calculating if")
#     if target_feature == "time_next_plug_in":
#         df = df.drop(
#             columns=["duration"],
#         )
#     original_columns = ["evse_id", "session_id"]
#     logger.info("type of original columns")
#     logger.info(type(original_columns))
#     logger.debug(f"database: \n {df}")
#     logger.info("Calculating time features...")
#     df_feature_time = feature_engineering_time(dataframe=df)
#     logger.success("Time features computed successfully ")
#     logger.debug(df_feature_time.head(5))
#     logger.debug(
#         f"Rows with null start_year:{df_feature_time[df_feature_time['start_year'].isnull()]}"
#     )
#     logger.info("Calculating naive features...")
#     df_all_featues, lowest_mse_column = feature_engineering_naive(
#         dataframe=df_feature_time, target_feature=target_feature
#     )

#     logger.success("Naive features computed successfully ")

#     logger.info("Normalizing Data...")
#     file_path_normalize = os.path.join(
#         "files",
#         "models",
#         "Normalization",
#         "Training",
#         f"{user_id}_{target_feature}_normalization.json",
#     )
#     normalization_params = load_normalization_params(file_path_normalize)

#     if len(df_all_featues) > 1:
#         df_features_normalized = normalize_new_data(
#             new_dataframe=df_all_featues, normalization_params=normalization_params
#         )
#     else:
#         df_features_normalized = df_all_featues

#     logger.debug(
#         f"Rows with null start_year data normalization:{df_features_normalized[df_features_normalized['start_year'].isnull()]}"
#     )

#     logger.debug(f"df_features_normalized:\n {df_features_normalized.tail(5)}")
#     logger.success("Data normalized successfully ")

#     logger.info("loading model...")
#     xgboost_model = load_model(
#         path_folder=path_folder, target_feature=target_feature, user_id=user_id
#     )
#     if target_feature == "total_energy_transfered":
#         df_sum = df_features_normalized.copy()
#         df_sum["start_datetime_aux"] = pd.to_datetime(df_sum["start_datetime"])

#         # Initialize a column to store the sum
#         df_sum["Sum_Last_72_Hours"] = 0

#         # Compute sum of values for the last 72 hours using itertuples
#         for i, row in enumerate(df_sum.itertuples(index=False)):
#             current_time = row.start_datetime_aux
#             # Filter the rows within the 72-hour window
#             mask = (df_sum["start_datetime_aux"] <= current_time) & (
#                 df_sum["start_datetime_aux"] > current_time - timedelta(hours=72)
#             )
#             df_sum.loc[i, "Sum_Last_72_Hours"] = df_sum.loc[
#                 mask, "total_energy_transfered"
#             ].sum()
#         df_features_normalized = df_sum.copy()
#     # logger.debug(xgboost_model.get_booster().feature_names)
#     logger.success("Load model sucessfull ...")
#     logger.info("Feature selection...")
#     df_features = feature_selection(
#         dataframe=df_features_normalized,
#         target_feature=target_feature,
#         lowest_mse_column=lowest_mse_column,
#     )

#     logger.info(type(df_features))
#     logger.debug(f"selected features are {df_features}")

#     logger.success("Feature selection was a success")

#     logger.debug(df_features_normalized["start_datetime"].min())
#     selected_columns = df_features
#     session_information = df_features_normalized[selected_columns]
#     logger.info("type of original columns")
#     logger.info(type(df_features))
#     # logger.debug(f'session information: {session_information['time_since_last_tx']}')
#     logger.info("Forecasting machine learning model...")
#     forecast_ = session_information.copy()
#     logger.info("df_feature_normalized describe:")
#     logger.debug(forecast_.describe())
#     forecast_[f"{target_feature} - Prevision ML"] = forecast_.apply(
#         lambda row: forecast_model(
#             dataframe=row.to_frame().T,  # Convert the row into a dataframe
#             model=xgboost_model,
#             target_feature=target_feature,
#         )[
#             0
#         ],  # Get the first element of the prediction array
#         axis=1,
#     )
#     logger.info(forecast_.info())
#     df_features_normalized[f"{target_feature} - Prevision ML"] = forecast_[
#         f"{target_feature} - Prevision ML"
#     ]

#     #  Display the result
#     df_all = df_features_normalized
#     logger.success("Forecasting machine learning model was a success")
#     logger.debug(f"forecast value:{forecast}")
#     return df_all


def load_normalization_params(json_path="normalization_params.json"):
    """
    Load normalize parameters.

    :param json_path: string - path for the normalize parameters.

    :return: normalization_params : json - normalized parameters

    """
    with open(json_path, "r") as json_file:
        normalization_params = json.load(json_file)
    return normalization_params


def normalize_new_data(new_dataframe, normalization_params):
    """
    Normalize the new data using previously saved normalization parameters.

    Parameters:
    ----------
    new_dataframe : pd.DataFrame - New dataframe containing features to be normalized
    normalization_params : dict - Dictionary of normalization parameters loaded from JSON

    Returns:
    --------
    pd.DataFrame - DataFrame with normalized numerical features.
    """
    X = new_dataframe.copy()

    # Apply normalization using the saved parameters
    for feature, params in normalization_params.items():
        if feature in X.columns:
            min_value = params["min"]
            max_value = params["max"]
            X[feature] = X[feature].apply(
                lambda x: normalize_dataset(x, min_value, max_value)
            )

    return X

def get_day_offset(current_date: str, next_session_date: str) -> int:
    """
    Calculate the offset in days between current_date and next_session_date.

    Args:
        current_date (str): The current date in 'YYYY-MM-DD' format.
        next_session_date (str): The next session date in 'YYYY-MM-DD' format.

    Returns:
        int: 0 if the dates are the same (D+0), 1 for D+1, ..., 7 for D+7,
             or -1 if the date is not within the range D+0 to D+7.
    """
    # # Parse input dates
    # current = datetime.strptime(current_date, '%Y-%m-%d')
    # next_session = datetime.strptime(next_session_date, '%Y-%m-%d')

    # Calculate the difference in days
    # logger.info(logger.info(type(next_session_date)))
    # next_session_date_day = next_session_date.date()
    # current_date_day = current_date.date()

    current_date_normalized = current_date.replace(hour=0, minute=0, second=0, microsecond=0)
    next_session_date_normalized = next_session_date.replace(hour=0, minute=0, second=0, microsecond=0)

    # logger.info(next_session_date_day)
    day_difference = (next_session_date_normalized - current_date_normalized).days

    # Return the offset if within the range 0-7, otherwise -1
    if 0 <= day_difference <= 7:
        return day_difference
    elif day_difference < 0:
        return 12
    return 8


@logger.catch(level="ERROR", message="Error calculating forecast:")
def probability_day_next_plug_in(dataframe):
    """
    Calculate the probability distribution of the number of days 
    between consecutive 'start_datetime' events in the dataframe.

    Args:
        dataframe (pd.DataFrame): A DataFrame containing a 'start_datetime' column.

    Returns:
        dict: A dictionary where keys are the number of days between plug-ins 
              and values are the corresponding probabilities.
    """

    # Create a copy of the original DataFrame to avoid modifying it in place
    df = dataframe.copy()

    # Create a new column with the 'start_datetime' of the next row (i.e., the next plug-in event)
    df["next_start_datetime"] = df["start_datetime"].shift(-1)

    # Calculate the number of days between current and next 'start_datetime' for each row
    df["day_next_plug_in"] = df.apply(
        lambda row: get_day_offset(row["start_datetime"], row["next_start_datetime"]),
        axis=1,
    )

    # Drop the helper column as it is no longer needed
    df.drop(columns=["next_start_datetime"], inplace=True)

    # Count how many times each day difference occurs
    day_diff_counts = df["day_next_plug_in"].value_counts().sort_index()

    # Compute the total number of transitions (non-null day differences)
    total_transitions = day_diff_counts.sum()

    # Calculate the probability for each day difference
    day_probabilities = day_diff_counts / total_transitions

    # Convert the result to a dictionary format
    return_probabilities = day_probabilities.to_dict()

    return return_probabilities

@logger.catch(level="ERROR", message="Error training forecasting model:")
def train_model_day_next_plugin(dataframe, user_id):
    """
    Train the XGBoost model after data cleaning, feature selection, and hyperparameter optimization.

    :param dataframe: pd.DataFrame - Input dataset for training.
    :param target_feature: str - The target variable to predict.

    :return: None - The trained model and hyperparameters are saved.
    """
    df = dataframe.copy()
    df = df[df["user_id"] == user_id].copy().reset_index(drop=True)
    df = calculation_delta_startime(dataframe=df)
    df = df.sort_values(by=["user_id", "start_time"])
    df["next_start_datetime"] = df["start_datetime"].shift(-1)
    df["day_next_plug_in"] = df.apply(
        lambda row: get_day_offset(row["start_datetime"], row["next_start_datetime"]),
        axis=1,
    )
    df.drop(columns=["next_start_datetime"], inplace=True)
    df["most_probable_return_day"]=df["day_next_plug_in"]
    df["time_diff_hours"] = (
        df["start_datetime"] - df["start_datetime"].shift(1)
    ).dt.total_seconds() / 3600

    # Filter rows where the time difference is greater than or equal to 2 hours
    filtered_df = df[(df["time_diff_hours"] >= 2)].copy()

    # Drop the helper column if not needed
    filtered_df.drop(columns=["time_diff_hours"], inplace=True)


    df_features = feature_engineering_time(dataframe=filtered_df)
    logger.debug(df_features[["start_datetime", "day_period", "var_20min"]])
    df_all, lowest_mse_column = feature_engineering_naive(
        dataframe=df_features, target_feature="day_next_plug_in"
    )
    file_path_normalize = os.path.join(
        "files",
        "models",
        "Normalization",
        "Training",
        f"{user_id}_{'day_next_plug_in'}_normalization.json",
    )
    df_features_normalized = data_normalization(
        dataframe=df_all,
        target_feature="day_next_plug_in",
        save_path=file_path_normalize,
    )

    # logger.info('Data Cleaning...')
    df_cleaned = data_cleaning(
        dataframe=df_features_normalized, target_feature="day_next_plug_in"
    )

    df_features_cleaned = feature_selection(
        dataframe=df_cleaned,
        target_feature="day_next_plug_in",
        lowest_mse_column=lowest_mse_column,
    )
    logger.debug(df_features_cleaned)
    df_cleaned = df_cleaned[df_features_cleaned]

    logger.info("Cleaned data")
    logger.debug(df_cleaned.head(5))
    logger.info("Preparing dataset for ML models")
    logger.debug("Info data cleaned")
    logger.debug(df_cleaned.describe())

    X, y = prepare_dataset(dataframe=df_cleaned, target="day_next_plug_in")

    class_counts = Counter(y)
    # Handle classes with only 1 sample using Random Oversampling
    rare_classes = [cls for cls, count in class_counts.items() if count == 1]
    ros = RandomOverSampler(sampling_strategy={cls: 5 for cls in rare_classes})  # Oversample to 5 samples
    X_resampled, y_resampled = ros.fit_resample(X, y)
    # valid_classes = {cls: count for cls, count in class_counts.items() if count > 1}

    # SMOTE for classes with at least 2 samples
    smote = SMOTE(k_neighbors=1)
    X_resampled, y_resampled = smote.fit_resample(X_resampled, y_resampled)


    logger.info("Dataset ready for ML models")
    logger.info("Determining best hyperparameters for model")
    logger.debug(f"target feature {'day_next_plug_in'}")
    logger.debug("Input X shape: {}".format(X.shape))
    logger.debug("Input y shape: {}".format(y.shape))

    unique_classes, counts = np.unique(y, return_counts=True)
    class_weights = {cls: len(y) / count for cls, count in zip(unique_classes, counts)}

    # Assign sample weights
    sample_weights = np.array([class_weights[label] for label in y])
    logger.info(f"Class weights:{class_weights}")
    logger.info(f"Sample weights:{sample_weights}")
    best_params_df, best_params = hyperparameter_optimization(X_train=X_resampled, y_train=y_resampled)
    num_classes = len(np.unique(y_resampled))  
    model = xgb.XGBClassifier(
        **best_params, use_label_encoder=False, eval_metric="mlogloss",num_class=num_classes,  random_state=42
    )
    encoder = LabelEncoder()
    y_resampled = encoder.fit_transform(y_resampled)
    logger.info(f"Unique labels in y:{np.unique(y_resampled)}")
    model.fit(X_resampled, y_resampled)
    logger.info("predicting XGBoost")
    logger.debug("Input X_train shape: {}".format(X.shape))
    logger.debug("Input y_train shape: {}".format(y.shape))
    num_classes = len(np.unique(y))


    save_model(
        model_params=model,
        path_folder="ML models",
        target_feature="day_next_plug_in",
        user_id=user_id,
        file_type="joblib",
        define_file="trained",
    )

    file_name = f"{user_id}_{'day_next_plug_in'}_metrics.json"
    file_path_metrics = os.path.join("files", "models", "Metrics")
    file_name_parameters = f"{user_id}_{'day_next_plug_in'}_parameters.json"

    file_path = os.path.join(file_path_metrics, file_name)

    file_path = os.path.join(file_path_metrics, file_name_parameters)

    save_json(json_data=best_params, file_path=file_path)

@logger.catch(level="ERROR", message="Error day of next session:")
def day_next_session(data):
    """
    Calculate the number of days between each session's start time and the next session's start time.

    :param data: pd.DataFrame - Input dataset containing a 'start_datetime' column with session start times.

    :return: pd.DataFrame - Modified DataFrame with an added 'day_next_plug_in' column indicating
                            the day difference to the next session.
    """
    df=data.copy()
    df["next_start_datetime"] = df["start_datetime"].shift(-1)

    # Apply the function to calculate the day offset
    df["day_next_plug_in"] = df.apply(
        lambda row: get_day_offset(row["start_datetime"], row["next_start_datetime"]),
        axis=1,
    )
    # Drop the 'next_start_datetime' column if no longer needed
    df.drop(columns=["next_start_datetime"], inplace=True)
    return df


logger.catch(level='ERROR', message='Error computing next starting hour')
def compute_next_hour(dataframe,column_date):
    """
    Compute the hour and day offset to the next transaction in a DataFrame.

    This function shifts the given datetime column to align each row with the next transaction's timestamp.
    It then calculates the fractional hour of the next transaction, checks whether the next transaction occurred
    on the same day, and computes the number of days between the current and next session.

    :param dataframe: pd.DataFrame - Input DataFrame containing datetime information for transactions.
    :param column_date: str - Name of the datetime column used to calculate the next transaction time.

    :return: pd.DataFrame - DataFrame with additional columns:
                            - '<column_date>_next': Timestamp of the next transaction.
                            - 'hour_minute': Fractional hour of the next transaction.
                            - 'transaction_same_day': Indicator (0 or 1) of whether the next transaction occurred on the same day.
                            - 'day_next_plug_in': Number of days to the next session, calculated using a custom function.
    """
    # todo: @Ricardo - check this
    dataframe[f'{column_date}_next'] = dataframe[column_date].shift(-1)
    dataframe['hour_minute'] = dataframe[f'{column_date}_next'].dt.hour + dataframe[f'{column_date}_next'].dt.minute / 60
    dataframe['transaction_same_day'] = (dataframe[f'{column_date}_next'].dt.date == dataframe[column_date].dt.date).astype(int)
    dataframe["day_next_plug_in"] = dataframe.apply(
        lambda row: get_day_offset(row["start_datetime"], row[f'{column_date}_next']),
        axis=1,
    )
    return dataframe