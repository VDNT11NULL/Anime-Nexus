import os
import pandas as pd
from google.cloud import storage
from src.logger import get_logger
from src.custom_exception import CustomException
from utils.common_functions import *
from config.paths_config import *

logger = get_logger(__name__)

class DataIngestion:
    def __init__(self, config):
        try:
            self.config = config.get("data_ingestion", {})
            if not self.config:
                raise CustomException("No 'data_ingestion' section found in config.yaml")
            self.project_id = self.config.get("project_id")
            self.bucket_name = self.config.get("bucket_name")
            self.file_names = self.config.get("bucket_file_names")
            self.nrows = self.config.get("num_rows_to_select")

            if not all([self.bucket_name, self.file_names, self.nrows]):
                raise CustomException("Missing required config parameters: bucket_name, bucket_file_names, or num_rows_to_select")

            os.makedirs(RAW_DIR, exist_ok=True)

            logger.info(f"Data Ingestion started with bucket {self.bucket_name} for files {', '.join(self.file_names)}")

        except KeyError as e:
            logger.error(f"Missing key in config: {e}")
            raise CustomException("Configuration error: ",e)
        except Exception as e:
            logger.error(f"Error in DataIngestion initialization: {e}")
            raise CustomException("Initialization failed: ",e)

    def download_from_GCP(self):
        try:
            client = storage.Client()  # the GOOGLE_APPLICATION_CREDENTIALS is set
            bucket = client.bucket(self.bucket_name)

            for file_name in self.file_names:
                file_path = os.path.join(RAW_DIR, file_name)
                if file_name == "animelist.csv":
                    blob = bucket.blob(file_name)
                    blob.download_to_filename(file_path)

                    data = pd.read_csv(file_path, nrows=self.nrows)
                    data.to_csv(file_path, index=False)

                    logger.info(f"Large file {file_name} detected, using only {self.nrows} data points")
                else:
                    blob = bucket.blob(file_name)
                    blob.download_to_filename(file_path)

                    logger.info(f"Downloaded {file_name} from GCP bucket {self.bucket_name}")

        except Exception as e:
            logger.error(f"Error while downloading CSV files from GCP bucket {self.bucket_name}: {e}")
            raise CustomException("Failed to download CSV files from GCP bucket: ",e)
        
    def run(self):
        try:
            logger.info("Starting Data Ingestion process")
            self.download_from_GCP()
            logger.info("Data Ingestion finished successfully!")

        except Exception as e:
            logger.error(f"Error during data ingestion stage from GCP bucket {self.bucket_name}: {e}")
            raise CustomException("Failed to ingest data from GCP bucket: ",e)

        finally:
            logger.info("Finally, data ingestion finished")

if __name__ == "__main__":
    print(f"Looking for config at: {os.path.abspath(CONFIG_PATH)}")
    config = read_yaml(CONFIG_PATH)
    print(f"Config contents: {config}")
    data_ingestion = DataIngestion(read_yaml(CONFIG_PATH))
    data_ingestion.run()