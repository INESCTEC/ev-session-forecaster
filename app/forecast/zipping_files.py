from app.conf.settings import BASE_PATH
from loguru import logger
import os
import zipfile
from datetime import date


logger.info("Zipping json files")
logger.info("Creating path for json directory...")
json_directory = os.path.join(BASE_PATH, "files", "logs")
logger.debug(f"json directory is : {json_directory}")
logger.success("json directory was extracted successfully")

logger.info("extracting json file...")
json_files = [f for f in os.listdir(json_directory) if f.endswith(".json")]
logger.success("json files were successfully extracted")

today = date.today()


# 3. Define the output zip file name and location
logger.info("creation of path for zip file ")
zip_filename = os.path.join(json_directory, f"{today}_logs.zip")
logger.success("path to zip file was successfully created")


# 4. Create a zip file and add all JSON files to it
logger.info("Zipping files...")
with zipfile.ZipFile(zip_filename, "w") as zipf:
    for json_file in json_files:
        file_path = os.path.join(json_directory, json_file)
        zipf.write(file_path, arcname=json_file)
        # os.remove(file_path)
logger.success("Files zipped")
