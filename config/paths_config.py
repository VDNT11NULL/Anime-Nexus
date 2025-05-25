import os

# Define BASE_DIR as the project root (directory containing paths_config.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(BASE_DIR)  # Go up one level to the project root (Anime_Recommendation_Sys)

# Define paths
CONFIG_PATH = os.path.join(BASE_DIR, "config", "config.yaml")
RAW_DIR = os.path.join(BASE_DIR, "artifacts", "raw")