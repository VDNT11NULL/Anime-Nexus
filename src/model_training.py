import os 
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib
from config.paths_config import *
from src.logger import get_logger
from src.custom_exception import CustomException
from utils.common_functions import read_yaml
from src.base_model import BaseModel


logger = get_logger(__name__)

class ModelTrainer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.config = read_yaml(CONFIG_PATH)
        self.config = self.config['model']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.base_model = BaseModel(config_path=CONFIG_PATH)
        self.model = self.base_model.RecommenderNet(self.config['embedding_dim'], self.base_model.n_users, self.base_model.n_animes)

        logger.info(f"ModelTrainer initialized with config: {self.config}")
        logger.info(f"Model moved to {self.device}")

    def load_data(self):
        try:
            Xtrain = np.load(X_TRAIN_ARRAY)
            Xtest = np.load(X_TEST_ARRAY)
            Ytrain = np.load(Y_TRAIN)
            Ytest = np.load(Y_TEST)

            logger.info(f"Training data loaded successfully")

            return Xtrain, Xtest, Ytrain, Ytest
        
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise CustomException("Failed to load training data", sys)
        
    def dataloader_pass(self):
        try:
            logger.info("Defining DataLoader for training...")
            X_train, X_test, Y_train, Y_test = self.load_data()
            
            print(f"X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")
            print(f"X_test shape: {X_test.shape}, Y_test shape: {Y_test.shape}")

            user_ids_train = torch.tensor(X_train[:, 0], dtype=torch.long)
            anime_ids_train = torch.tensor(X_train[:, 1], dtype=torch.long)
            ratings_train = torch.tensor(Y_train, dtype=torch.float32)

            user_ids_test = torch.tensor(X_test[:, 0], dtype=torch.long)
            anime_ids_test = torch.tensor(X_test[:, 1], dtype=torch.long)
            ratings_test = torch.tensor(Y_test, dtype=torch.float32)

            print(f"user_ids_train shape: {user_ids_train.shape}, anime_ids_train shape: {anime_ids_train.shape}, ratings_train shape: {ratings_train.shape}")
            print(f"user_ids_test shape: {user_ids_test.shape}, anime_ids_test shape: {anime_ids_test.shape}, ratings_test shape: {ratings_test.shape}")

            if len(ratings_train.shape) == 1:
                ratings_train = ratings_train.unsqueeze(1)
            if len(ratings_test.shape) == 1:
                ratings_test = ratings_test.unsqueeze(1)

            train_dataset = TensorDataset(user_ids_train, anime_ids_train, ratings_train)
            test_dataset = TensorDataset(user_ids_test, anime_ids_test, ratings_test)

            BATCH_SIZE = self.config['batch_size']
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

            logger.info(f"DataLoader created with batch size {BATCH_SIZE}")
            return train_loader, test_loader
        except Exception as e:
            print(f"Error creating DataLoader: {e}")
            logger.error(f"Error creating DataLoader: {e}")
            raise CustomException("Failed to create DataLoader", sys)

    # def dataloader_pass(self):
    #     try:
    #         logger.info("Defining Data Loader for training...")
    #         X_train, X_test, Y_train, Y_test = self.load_data()
    #         user_ids_train = torch.tensor(X_train[:, 0], dtype=torch.long)
    #         anime_ids_train = torch.tensor(X_train[:, 1], dtype=torch.long)
    #         ratings_train = torch.tensor(Y_train, dtype=torch.float)
    #         print(f"X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")
    #         print(f"X_test shape: {X_test.shape}, Y_test shape: {Y_test.shape}")
    #         logger.info(f"Training data shapes - X: {X_train.shape}, Y: {Y_train.shape}")
    #         logger.info(f"Testing data shapes - X: {X_test.shape}, Y: {Y_test.shape}")

    #         user_ids_test = torch.tensor(X_test[:, 0], dtype=torch.long)
    #         anime_ids_test = torch.tensor(X_test[:, 1], dtype=torch.long)
    #         ratings_test = torch.tensor(Y_test, dtype=torch.float)

    #         if len(ratings_train.shape) == 1:
    #             ratings_train = ratings_train.unsqueeze(1)  # Shape: (num_samples, 1)
    #         if len(ratings_test.shape) == 1:
    #             ratings_test = ratings_test.unsqueeze(1)

    #         train_dataset = TensorDataset(user_ids_train, anime_ids_train, ratings_train)
    #         test_dataset = TensorDataset(user_ids_test, anime_ids_test, ratings_test)

    #         BATCH_SIZE = self.config['batch_size']
    #         train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    #         test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False) 

    #         logger.info(f"DataLoader created with batch size {BATCH_SIZE}")
    #         return train_loader, test_loader

    #     except Exception as e:
    #         logger.error(f"Error creating DataLoader: {e}")
    #         raise CustomException("Failed to create DataLoader", sys)
        
    def train(self):
        try:

            optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
            criterion = nn.MSELoss()
            DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            train_loader, test_loader = self.dataloader_pass()
            logger.info("Starting training process...")
            self.model.train()
            self.model.to(self.device)
            
            metrics = {'train_loss': [], 'test_loss': []}
            for epoch in range(self.config['epochs']):
                train_loss = 0.0
                num_train_batches = 0

                for user, anime, rating in train_loader:
                    user = user.to(DEVICE)
                    anime = anime.to(DEVICE)
                    rating = rating.to(DEVICE)

                    optimizer.zero_grad()

                    predictions = self.model(user, anime)
                    loss = criterion(predictions, rating)
                    mae = torch.mean(torch.abs(predictions - rating))

                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()
                    num_train_batches+=1
                
                avg_train_loss = train_loss / num_train_batches
                metrics['train_loss'].append(avg_train_loss)
            
            logger.info(f"Model Training Done with {avg_train_loss} loss in epoch {epoch+1}")

            return metrics
        
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise CustomException("Failed to train the model", sys) 
        
    def extract_weights(self, layer_name):
        try:
            if layer_name == 'user_embedding':
                weights = self.model.user_emb_layer.weight.data
            elif layer_name == 'anime_embedding':
                weights = self.model.anime_emb_layer.weight.data
            else:
                raise ValueError(f"Unknown layer: {layer_name}")

            # Normalize weights (L2 normalization)
            weights_norm = weights / (torch.norm(weights, dim=1, keepdim=True) + 1e-8)
            weights_norm_np = weights_norm.cpu().numpy()
            logger.info(f"Extracted and normalized weights for {layer_name}")
            return weights_norm_np

        except Exception as e:
            logger.error(f"Error extracting weights for {layer_name}: {e}")
            raise CustomException("Failed to extract weights", sys)

    def save_model_weights(self):
        try:
            # Save full model
            os.makedirs(os.path.dirname(MODEL_CKPT_FILE), exist_ok=True)
            os.makedirs(os.path.dirname(USER_WEIGHTS_PATH), exist_ok=True)
            os.makedirs(os.path.dirname(ANIME_WEIGHTS_PATH), exist_ok=True)
            logger.info("Saving model and weights...")

            torch.save(self.model.state_dict(), MODEL_CKPT_FILE)
            logger.info(f"Model saved to {MODEL_CKPT_FILE}")

            # Extract and save embedding weights
            user_weights = self.extract_weights('user_embedding')
            anime_weights = self.extract_weights('anime_embedding')

            joblib.dump(user_weights, USER_WEIGHTS_PATH)
            joblib.dump(anime_weights, ANIME_WEIGHTS_PATH)
            logger.info(f"User weights saved to {USER_WEIGHTS_PATH}")
            logger.info(f"Anime weights saved to {ANIME_WEIGHTS_PATH}")

        except Exception as e:
            logger.error(f"Error saving model or weights: {e}")
            raise CustomException("Failed to save model or weights", sys)
        
    def run(self):
        try:
            logger.info("Starting model training...")
            metrics = self.train()
            self.save_model_weights()
            print(f"Training metrics: {metrics}")
            logger.info("Model training completed successfully.")
            return metrics
        
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            raise CustomException("Failed to run model training", sys)
        
if __name__ == "__main__":
    trainer = ModelTrainer(data_path=PROCESSED_DIR)
    try:
        metrics = trainer.run()
        print(f"Training metrics: {metrics}")
    except CustomException as e:
        logger.error(f"CustomException occurred: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")

        