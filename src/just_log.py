import os
import pandas as pd
from google.cloud import storage
from src.logger import get_logger
from src.custom_exception import CustomException
from utils.common_functions import *
from config.paths_config import *

logger = get_logger(__name__)


if __name__ == "__main__":
    config = read_yaml("config/config.yaml")
    