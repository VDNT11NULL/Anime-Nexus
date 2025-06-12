import os

# Define BASE_DIR as the project root (directory containing paths_config.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(BASE_DIR)  # Go up one level to the project root (Anime_Recommendation_Sys)
CONFIG_PATH = os.path.join(BASE_DIR, "config", "config.yaml")

################ RAW DATA PATHS ################
RAW_DIR = os.path.join(BASE_DIR, "artifacts", "raw")
ANIMELIST_CSV = os.path.join(RAW_DIR, "animelist.csv") ## rating df
ANIME_CSV = os.path.join(RAW_DIR, "anime.csv") ## anime df
SYNOPSIS_CSV = os.path.join(RAW_DIR, "anime_with_synopsis.csv") ## synopsis df

################# PROCESSED DATA PATHS ################
PROCESSED_DIR = os.path.join(BASE_DIR, "artifacts", "processed")
PROCESSED_RATING_DF = os.path.join(PROCESSED_DIR, "processed_rating_df.csv")
PROCESSED_SYNOPSIS_DF = os.path.join(PROCESSED_DIR, "processed_synopsis_df.csv")
PROCESSED_ANIME_DF = os.path.join(PROCESSED_DIR, "processed_anime_df.csv")

X_TRAIN_ARRAY = os.path.join(PROCESSED_DIR, "Xtrain_array.npy")
X_TEST_ARRAY = os.path.join(PROCESSED_DIR, "Xtest_array.npy")
Y_TRAIN = os.path.join(PROCESSED_DIR, "Ytrain.npy")
Y_TEST = os.path.join(PROCESSED_DIR, "Ytest.npy")

################# MAPPINGS PATHS ################
ANIMEID_2_ENCODEDANIMEID_MAPPING = os.path.join(PROCESSED_DIR, "animeId_2_encodedAnimeId_mapping.pkl")
USERID_2_ENCODEDUSERID_MAPPING = os.path.join(PROCESSED_DIR, "userId_2_encodedUserId_mapping.pkl")
ENCODEDANIMEID_2_ANIMEID_MAPPING = os.path.join(PROCESSED_DIR, "encodedAnimeId_2_animeId_mapping.pkl")  
ENCODEDUSERID_2_USERID_MAPPING = os.path.join(PROCESSED_DIR, "encodedUserId_2_userId_mapping.pkl")

################ MODEL DIR ######################
MODEL_DIR = os.path.join(BASE_DIR, "artifacts", "model")
MODEL_CKPT_FILE = os.path.join(MODEL_DIR, "model_checkpoint.pt")

WEIGHTS_DIR = os.path.join(BASE_DIR, "artifacts", "weights")
USER_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "user_embeddings")
ANIME_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "anime_embeddings")