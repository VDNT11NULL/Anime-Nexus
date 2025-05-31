import os

# Define BASE_DIR as the project root (directory containing paths_config.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(BASE_DIR)  # Go up one level to the project root (Anime_Recommendation_Sys)
CONFIG_PATH = os.path.join(BASE_DIR, "config", "config.yaml")

################ RAW DATA PATHS ################
RAW_DIR = os.path.join(BASE_DIR, "artifacts", "raw")

ANIMELIST_CSV = os.path.join(RAW_DIR, "animelist.csv") ## rating df
ANIME_CSV = os.path.join(RAW_DIR, "anime.csv") ## anime df
SYNOPSIS_CDV = os.path.join(RAW_DIR, "anime_with_synopsis.csv") ## synopsis df

################# PROCESSED DATA PATHS ################
PROCESSED_DIR = os.path.join(BASE_DIR, "artifacts", "processed")
X_TRAIN_ARRAY = os.path.join(PROCESSED_DIR, "Xtrain_array.npy")
X_TEST_ARRAY = os.path.join(PROCESSED_DIR, "Xtest_array.npy")
Y_TRAIN = os.path.join(PROCESSED_DIR, "Ytrain.npy")
Y_TEST = os.path.join(PROCESSED_DIR, "Ytest.npy")

################# MAPPINGS PATHS ################
