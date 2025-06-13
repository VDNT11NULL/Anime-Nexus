import os
import sys
import comet_ml
import numpy as np
from dotenv import load_dotenv
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

load_dotenv()
COMET_API_KEY = os.getenv('COMET_API_KEY')

class ModelTrainer:
    def __init__(self, data_path, exp_tracking=False):
        self.data_path = data_path
        self.exp_tracking = exp_tracking
        if self.exp_tracking:
            self.Experiment = comet_ml.Experiment(
                api_key=COMET_API_KEY,
                project_name='anime-rec-sys',
                workspace='non-llm-wspace',
            )

        self.config = read_yaml(CONFIG_PATH)
        self.config = self.config['model']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.base_model = BaseModel(config_path=CONFIG_PATH)
        self.model = self.base_model.RecommenderNet(self.config['embedding_dim'], self.base_model.n_users, self.base_model.n_animes)

        logger.info(f"ModelTrainer initialized with config: {self.config}")
        logger.info(f"Model moved to {self.device}")
        logger.info(f"Model expects - Users: [0, {self.base_model.n_users-1}], Animes: [0, {self.base_model.n_animes-1}]")
    
    def validate_indices(self, user_ids, anime_ids, data_type=""):
        """Validate that all indices are within bounds"""
        try:
            # Check user indices
            max_user_id = user_ids.max().item()
            min_user_id = user_ids.min().item()
            
            # Check anime indices
            max_anime_id = anime_ids.max().item()
            min_anime_id = anime_ids.min().item()
            
            print(f"=== {data_type.upper()} VALIDATION ===")
            print(f"Model expects - Users: [0, {self.base_model.n_users-1}], Animes: [0, {self.base_model.n_animes-1}]")
            print(f"Data contains - Users: [{min_user_id}, {max_user_id}], Animes: [{min_anime_id}, {max_anime_id}]")
            
            logger.info(f"{data_type} - User IDs range: [{min_user_id}, {max_user_id}]")
            logger.info(f"{data_type} - Anime IDs range: [{min_anime_id}, {max_anime_id}]")
            
            # Validate bounds
            validation_passed = True
            if max_user_id >= self.base_model.n_users:
                error_msg = f"❌ User ID {max_user_id} >= n_users ({self.base_model.n_users})"
                print(error_msg)
                logger.error(error_msg)
                validation_passed = False
                
            if min_user_id < 0:
                error_msg = f"❌ User ID {min_user_id} < 0"
                print(error_msg)
                logger.error(error_msg)
                validation_passed = False
                
            if max_anime_id >= self.base_model.n_animes:
                error_msg = f"❌ Anime ID {max_anime_id} >= n_animes ({self.base_model.n_animes})"
                print(error_msg)
                logger.error(error_msg)
                validation_passed = False
                
            if min_anime_id < 0:
                error_msg = f"❌ Anime ID {min_anime_id} < 0"
                print(error_msg)
                logger.error(error_msg)
                validation_passed = False
            
            if not validation_passed:
                raise ValueError(f"Index validation failed for {data_type} data")
                
            print(f"✅ {data_type} indices validation passed!")
            logger.info(f"{data_type} indices validation passed!")
            
        except Exception as e:
            logger.error(f"Index validation failed for {data_type}: {e}")
            raise CustomException(f"Index validation failed: {e}", sys)
    
    def load_data(self):
        try:
            Xtrain = np.load(X_TRAIN_ARRAY)
            Xtest = np.load(X_TEST_ARRAY)
            Ytrain = np.load(Y_TRAIN)
            Ytest = np.load(Y_TEST)

            # Ensure Xtrain is a 2D array with shape (num_samples, 2)
            if Xtrain.ndim == 1:
                Xtrain = np.stack(Xtrain, axis=0)
            if Xtest.ndim == 1:
                Xtest = np.stack(Xtest, axis=0)

            if Xtrain.shape[0] == 2:
                Xtrain = Xtrain.T
            if Xtest.shape[0] == 2:
                Xtest = Xtest.T

            # Log unique IDs before filtering
            unique_users_train = np.unique(Xtrain[:, 0])
            unique_animes_train = np.unique(Xtrain[:, 1])
            logger.info(f"Before filtering - Unique user IDs in Xtrain: {min(unique_users_train)} to {max(unique_users_train)}")
            logger.info(f"Before filtering - Unique anime IDs in Xtrain: {min(unique_animes_train)} to {max(unique_animes_train)}")

            # Filter invalid IDs
            max_user_id = self.base_model.n_users - 1  # 17661
            max_anime_id = self.base_model.n_animes - 1  # 17224
            valid_train_mask = (Xtrain[:, 0] <= max_user_id) & (Xtrain[:, 1] <= max_anime_id)
            Xtrain = Xtrain[valid_train_mask]
            Ytrain = Ytrain[valid_train_mask]
            valid_test_mask = (Xtest[:, 0] <= max_user_id) & (Xtest[:, 1] <= max_anime_id)
            Xtest = Xtest[valid_test_mask]
            Ytest = Ytest[valid_test_mask]

            # Log shapes and unique IDs after filtering
            unique_users_train = np.unique(Xtrain[:, 0])
            unique_animes_train = np.unique(Xtrain[:, 1])
            logger.info(f"After filtering - Xtrain shape: {Xtrain.shape}")
            logger.info(f"After filtering - Unique user IDs in Xtrain: {min(unique_users_train)} to {max(unique_users_train)}")
            logger.info(f"After filtering - Unique anime IDs in Xtrain: {min(unique_animes_train)} to {max(unique_animes_train)}")
            logger.info(f"After filtering - Xtest shape: {Xtest.shape}")

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

            # CRITICAL: Validate indices before proceeding - FORCED VALIDATION
            print("\n" + "="*50)
            print("PERFORMING INDEX VALIDATION")
            print("="*50)
            self.validate_indices(user_ids_train, anime_ids_train, "Training")
            self.validate_indices(user_ids_test, anime_ids_test, "Testing")
            print("="*50)
            print("VALIDATION COMPLETE")
            print("="*50 + "\n")

            print(f"user_ids_train shape: {user_ids_train.shape}, anime_ids_train shape: {anime_ids_train.shape}, ratings_train shape: {ratings_train.shape}")
            print(f"user_ids_test shape: {user_ids_test.shape}, anime_ids_test shape: {anime_ids_test.shape}, ratings_test shape: {ratings_test.shape}")

            if len(ratings_train.shape) == 1:
                ratings_train = ratings_train.unsqueeze(1)
            if len(ratings_test.shape) == 1:
                ratings_test = ratings_test.unsqueeze(1)

            print(f'user_ids_train: {user_ids_train.shape}')
            print(f'anime_ids_train: {anime_ids_train.shape}')
            print(f'ratings_train: {ratings_train.shape}')
            
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
        
    def train(self):
        try:
            optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
            criterion = nn.MSELoss()
            train_loader, test_loader = self.dataloader_pass()
            
            logger.info("Starting training process...")
            self.model.train()
            self.model.to(self.device)
            
            metrics = {'train_loss': [], 'test_loss': []}
            
            for epoch in range(self.config['epochs']):
                train_loss, mae_loss = 0.0, 0.0
                num_train_batches = 0

                for batch_idx, (user, anime, rating) in enumerate(train_loader):
                    try:
                        user = user.to(self.device)
                        anime = anime.to(self.device)
                        rating = rating.to(self.device)

                        # Additional safety check for this batch
                        if user.max() >= self.base_model.n_users or user.min() < 0:
                            logger.error(f"Batch {batch_idx}: Invalid user IDs - range [{user.min()}, {user.max()}]")
                            continue
                            
                        if anime.max() >= self.base_model.n_animes or anime.min() < 0:
                            logger.error(f"Batch {batch_idx}: Invalid anime IDs - range [{anime.min()}, {anime.max()}]")
                            continue

                        optimizer.zero_grad()
                        predictions = self.model(user, anime)
                        loss = criterion(predictions, rating)
                        mae = torch.mean(torch.abs(predictions - rating))

                        loss.backward()
                        optimizer.step()

                        train_loss += loss.item()
                        mae_loss += mae.item()
                        num_train_batches += 1
                        
                    except RuntimeError as e:
                        logger.error(f"Error in batch {batch_idx}: {e}")
                        if "index" in str(e).lower():
                            logger.error(f"Batch user range: [{user.min()}, {user.max()}]")
                            logger.error(f"Batch anime range: [{anime.min()}, {anime.max()}]")
                        continue
                
                if num_train_batches > 0:
                    avg_train_loss = train_loss / num_train_batches
                    avg_mae_loss = mae_loss / num_train_batches
                    metrics['train_loss'].append(avg_train_loss)

                    if self.exp_tracking:
                        self.Experiment.log_metric('train_loss', avg_train_loss, step=epoch)
                        self.Experiment.log_metric('train_mae', avg_mae_loss, step=epoch)
                    
                    logger.info(f"Epoch {epoch+1}/{self.config['epochs']}: Loss={avg_train_loss:.4f}, MAE={avg_mae_loss:.4f}")
                else:
                    logger.warning(f"No valid batches processed in epoch {epoch+1}")
            
            logger.info("Model Training completed successfully")
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
            if self.exp_tracking:
                self.Experiment.log_asset(MODEL_CKPT_FILE)
    
            logger.info(f"Model saved to {MODEL_CKPT_FILE}")

            # Extract and save embedding weights
            user_weights = self.extract_weights('user_embedding')
            anime_weights = self.extract_weights('anime_embedding')
            logger.info("Extracted user and anime embedding weights.")
            
            joblib.dump(user_weights, USER_WEIGHTS_PATH)
            joblib.dump(anime_weights, ANIME_WEIGHTS_PATH)

            if self.exp_tracking:
                self.Experiment.log_asset(USER_WEIGHTS_PATH)
                self.Experiment.log_asset(ANIME_WEIGHTS_PATH)            
                
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
    trainer = ModelTrainer(data_path=PROCESSED_DIR, exp_tracking=True)
    try:
        metrics = trainer.run()
        print(f"Training metrics: {metrics}")
    except CustomException as e:
        logger.error(f"CustomException occurred: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")