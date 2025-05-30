import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *

logger = get_logger(__name__)

class DataProcessor:
    def __init__(self, input_file, output_dir):
        self.input_file = input_file
        self.output_dir = output_dir

        self.rating_df = None
        self.anime_df = None
        self.Xtrain_array = None
        self.Xtest_array = None
        self.Ytrain = None
        self.Ytest = None

        self.userId_2_encodedUserId_mapping = {}
        self.encodedUserId_2_userId_mapping = {}
        self.animeId_2_encodedAnimeId_mapping = {}
        self.encodedAnimeId_2_animeId_mapping = {}

        os.makedirs(self.output_dir, exist_ok=True)
        logger.info("Stated with Data processing stage 1")

    def load_data(self, cols2use):
        try:
            self.rating_df = pd.read_csv(self.input_file, low_memory=True, usecols=cols2use)
            logger.info(f"DataProcessor.data loaded {self.input_file}")

        except Exception as e:
            logger.error(f"Error at data loading in processing {e}")
            raise CustomException("Failed to load input csv")
        
    def filter_exp_users(self, NUM_REVIEW_THRESHOLD):
        '''
        Filters out experienced users who have given more than a certain number of reviews.
        Args:
            NUM_REVIEW_THRESHOLD (int): The minimum number of reviews a user must have to be considered experienced.
        Returns:
            None: The method modifies the rating_df in place.
        '''
        try:
            n_ratings = self.rating_df["user_id"].value_counts()
            rating_df = rating_df[rating_df['user_id'].isin(n_ratings[n_ratings >= NUM_REVIEW_THRESHOLD].index)]
            logger.info("Filtered data successfully ...")

        except Exception as e:
            logger.error(f"Error while filtering data : {e}")
            raise CustomException("Failed to filter data", sys)

    def scale_ratings(self):
        '''
        Scales the ratings in the rating_df to a range of 0 to 1.
        This method modifies the rating_df in place.
        Returns:
            None
        '''
        try:
            min_rating = min(self.rating_df['rating'])
            max_rating = max(self.rating_df['rating'])

            self.rating_df['rating'] = self.rating_df['rating'].apply(lambda x : (x-min_rating)/(max_rating-min_rating)).values.astype(np.float16)    
            logger.info(f"Data scaling done successfully ...")

        except Exception as e:
            logger.error(f"Error at scaling DF")
            raise CustomException("Failed at data scaling", sys)
        
    def encode_user_anime_ids(self):
        try:
            self.userId_2_encodedUserId_mapping = {user_id: idx for idx, user_id in enumerate(self.rating_df['user_id'].unique())}
            self.encodedUserId_2_userId_mapping = {idx: user_id for user_id, idx in self.userId_2_encodedUserId_mapping.items()}

            self.animeId_2_encodedAnimeId_mapping = {anime_id: idx for idx, anime_id in enumerate(self.rating_df['anime_id'].unique())}
            self.encodedAnimeId_2_animeId_mapping = {idx: anime_id for anime_id, idx in self.animeId_2_encodedAnimeId_mapping.items()}

            logger.info("Encoded user and anime IDs successfully ...")
        except Exception as e:
            logger.error(f"Error at encoding user and anime IDs: {e}")
            raise CustomException("Failed to encode user and anime IDs", sys)
        
    def split_data(self, test_size, random_state=11):
        try:
            self.rating_df = self.rating_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
            X = self.rating_df[['encoded_userID', 'encoded_animeID']].values
            y = self.rating_df['rating'].values

            X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
            self.Xtrain_array = X_train
            self.Xtest_array = X_test
            self.Ytrain = Y_train
            self.Ytest = Y_test
            logger.info("Data split into train and test sets successfully ...")
            return X_train, X_test, Y_train, Y_test 

        except Exception as e:
            logger.error(f"Failed to split data for rating_df")
            raise CustomException("Failed to split data", sys)
        
    