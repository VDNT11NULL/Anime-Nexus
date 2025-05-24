import os
import pandas as pd
from google.cloud import storage
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from utils.common_functions import read_yaml

logger = get_logger(__name__)

class DataInjestion:
    def __init__(self, config):
        self.config = config["data_injestion"]
        self.bucket_name = self.config["bucket_name"]
        self.file_names  = self.config["bucket_file_names"]
        self.nrows = self.config["num_rows_to_select"]

        os.makedirs(RAW_DIR, exist_ok=True)

        logger.info(f"Data Ingestion started with {self.bucket_name} on {self.file_name}")

    def download_from_GCP(self):
        try:
            client = storage.Client() # we just did `set GOOGLE_APPLICATION_CREDENTIALS`
            bucket = client.bucket(self.bucket_name)
                
            for file_name in self.file_names:
                file_path = os.path.join(RAW_DIR, file_name)
                ## Selective Data Injestion as we wont be able to use 70M data points from animelist.csv
                if(file_name=="animelist.csv"):
                    blob = bucket.blob(file_name)
                    blob.download_to_filename(file_path)

                    data = pd.read_csv(file_path, nrows=self.nrows)
                    data.to_csv(file_path, index=False)

                    logger.info(f"Large {file_name} detected, using only {self.nrows} data points")

                else:
                    blob = bucket.blob(file_name)
                    blob.download_to_filename(file_path)

                    logger.info(f"Downlaoded {file_name} from GCP bucket {self.bucket_name}")
        
        except Exception as e:
            logger.error(f"Error while downloading csv files from GCP {self.bucket_name}")
            raise CustomException("Failed to download csv files from GCP buckets")
        
    def run(self):
        try:
            logger.info(f"Starting Data Injestion process")
            self.download_from_GCP()
            logger.info("Data Injestion fininshed successfully!!")

        except Exception as e:
            logger.error(f"Error while data injestion satge from GCP {self.bucket_name}")
            raise CustomException("Failed to injestdata from GCP buckets")
        
        finally:
            logger.info(f"Finally data injestion finished")


if __name__=="__main__":
    config = read_yaml(CONFIG_PATH)
    print(config)
    data_injestion = DataInjestion(read_yaml(CONFIG_PATH))
    data_injestion.run()