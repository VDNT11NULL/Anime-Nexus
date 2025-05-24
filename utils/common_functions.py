import os 
import pandas as pd
from src.logger import get_logger
from src.custom_exception import CustomException
import yaml

logger = get_logger(__name__)

def read_yaml(file_path):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found at {file_path}")
        
        with open(file_path, 'r') as yaml_file:
            config = yaml.safe_load(yaml_file)
            logger.info("Successfully read the YAML CONFIG file")
            return config

    except Exception as e:
        logger.error(f"Error while reading YAML file at {file_path}: {str(e)}")
        raise CustomException(f"Failed to read YAML at {file_path}", e)

def load_data(csv_path):
    try:
        logger.info(f"Reading CSV at {csv_path}")
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        logger.error(f"Error while loading CSV data from {csv_path}: {str(e)}")
        raise CustomException(f"Failed loading CSV at {csv_path}", e)