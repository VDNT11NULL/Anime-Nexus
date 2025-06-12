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
        
    def filter_exp_users(self, NUM_REVIEW_THRESHOLD=5):
        '''
        Filters out experienced users who have given more than a certain number of reviews.
        Args:
            NUM_REVIEW_THRESHOLD (int): The minimum number of reviews a user must have to be considered experienced.
        Returns:
            None: The method modifies the rating_df in place.
        '''
        try:
            n_ratings = self.rating_df["user_id"].value_counts()
            # print(f"Number of users before filtering: {len(self.rating_df['user_id'].unique())}")
            # print(f"Number of users with more than {NUM_REVIEW_THRESHOLD} reviews: {len(n_ratings[n_ratings >= NUM_REVIEW_THRESHOLD])}")
            self.rating_df = self.rating_df[self.rating_df['user_id'].isin(n_ratings[n_ratings >= NUM_REVIEW_THRESHOLD].index)]
            # print(f"Filtered out users with less than {NUM_REVIEW_THRESHOLD} reviews. Remaining users: {len(self.rating_df['user_id'].unique())}")
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
            ## DEBug
            print(self.rating_df.columns)
            X = self.rating_df[['user_id', 'anime_id']].values
            y = self.rating_df['rating'].values

            X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
            self.Xtrain_array = [X_train[:, 0], X_train[:, 1]]
            self.Xtest_array = [X_test[:, 0], X_test[:, 1]]
            self.Ytrain = Y_train
            self.Ytest = Y_test
            logger.info("Data split into train and test sets successfully ...")
            return X_train, X_test, Y_train, Y_test 

        except Exception as e:
            logger.error(f"Failed to split data for rating_df")
            raise CustomException("Failed to split data", sys)
        
    def save_mappings(self):
        try:
            mappings = {
                'userId_2_encodedUserId_mapping': self.userId_2_encodedUserId_mapping,
                'encodedUserId_2_userId_mapping': self.encodedUserId_2_userId_mapping,
                'animeId_2_encodedAnimeId_mapping': self.animeId_2_encodedAnimeId_mapping,
                'encodedAnimeId_2_animeId_mapping': self.encodedAnimeId_2_animeId_mapping
            }
            for name, mapping in mappings.items():
                joblib.dump(mapping, os.path.join(self.output_dir, f"{name}.pkl"))
                logger.info(f"Saved {name} mapping successfully ...")

        except Exception as e:
            logger.error(f"Error at saving mappings: {e}")
            raise CustomException("Failed to save mappings", sys)
               
    def save_data(self):
        try:
            np.save(os.path.join(self.output_dir, 'Xtrain_array.npy'), self.Xtrain_array)
            np.save(os.path.join(self.output_dir, 'Xtest_array.npy'), self.Xtest_array)
            np.save(os.path.join(self.output_dir, 'Ytrain.npy'), self.Ytrain)
            np.save(os.path.join(self.output_dir, 'Ytest.npy'), self.Ytest)
            logger.info("Saved train and test data successfully ...")
        
            self.rating_df.to_csv(os.path.join(PROCESSED_DIR, "processed_rating_df.csv"), index=False)
        
        except Exception as e:
            logger.error(f"Error at saving data: {e}")
            raise CustomException("Failed to save data", sys)
        
    def process_anime_data(self): # for anime.csv
        try:
            self.anime_df = pd.read_csv(ANIME_CSV)
            # print(f'anime_df columns: {self.anime_df.columns}')

            self.anime_df = self.anime_df.replace("Unknown", np.nan)

            self.synopsis_df = pd.read_csv(SYNOPSIS_CSV)

            def get_anime_name(anime_id): ## org anime id (not the encoded ones from animelist)
                try:
                    row = self.anime_df[self.anime_df.MAL_ID == anime_id]
                    name = row["English name"].values[0]
                    if pd.isna(name):
                        name = row["Name"].values[0]
                    return name
                except:
                    print("error while retriving anime name")
    
            
            self.anime_df["eng_version"] = self.anime_df.MAL_ID.apply(lambda x: get_anime_name(x))
            self.anime_df = self.anime_df[["MAL_ID", "eng_version", "Score", "Genres", "Episodes", "Type", "Studios", "Completed", "Rating", "Producers", "Duration", "Members"]]
            # print(f'anime_df columns: {self.anime_df.columns}')
            self.anime_df.sort_values(by=["Score"], inplace=True, ascending=False, kind="quicksort", na_position="last")

            self.anime_df.to_csv(PROCESSED_ANIME_DF, index=False)
            self.synopsis_df.to_csv(PROCESSED_SYNOPSIS_DF, index=False)
            logger.info("Saved processed anime and synopsis data successfully ...")
            logger.info("Processed anime data successfully ...")

        except Exception as e:
            logger.error(f"Error at processing anime data: {e}")
            raise CustomException("Failed to process anime data", sys)            
            

    def run(self):
        try:
            self.load_data(cols2use=['user_id', 'anime_id', 'rating'])
            self.filter_exp_users(NUM_REVIEW_THRESHOLD=5)
            self.scale_ratings()
            self.encode_user_anime_ids()
            self.split_data(test_size=0.2)
            self.save_mappings()
            self.save_data()
            self.process_anime_data()
            logger.info("Data processing completed successfully ...")

        except Exception as e:
            logger.error(f"Error during data loading and filtering: {e}")
            raise CustomException("Failed to load and filter data", sys)
        
        
if __name__ == "__main__":
    input_file = ANIMELIST_CSV  # rating data file
    output_dir = PROCESSED_DIR  # directory to save processed data
    
    processor = DataProcessor(input_file, output_dir)
    processor.run()
