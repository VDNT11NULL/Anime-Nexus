import os
import sys
import torch
import torch.nn as nn
import joblib
from config.paths_config import *
from src.logger import get_logger
from src.custom_exception import CustomException
from utils.common_functions import read_yaml

logger = get_logger(__name__)

class BaseModel:
    def __init__(self, config_path=CONFIG_PATH):
        try:
            self.config = read_yaml(config_path)
            logger.info(f"BaseModel initialized with config: {self.config}")

            # Load mappings
            self.userId_2_encodedUserId_mapping = joblib.load(USERID_2_ENCODEDUSERID_MAPPING)
            self.animeId_2_encodedAnimeId_mapping = joblib.load(ANIMEID_2_ENCODEDANIMEID_MAPPING)

            self.n_users = len(self.userId_2_encodedUserId_mapping)
            print(f"Loaded userId_2_encodedUserId_mapping with {self.n_users} users")
            self.n_animes = len(self.animeId_2_encodedAnimeId_mapping)
            print(f"Loaded animeId_2_encodedAnimeId_mapping with {self.n_animes} animes")
            # self.n_users = 17662
            # self.n_animes = 17224
            
            logger.info(f"Loaded mappings: n_users={self.n_users}, n_animes={self.n_animes}")

            ## Embd dim
            self.embedding_size = self.config.get('embedding_size', 128)

            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = self.RecommenderNet(
                EMBEDDING_SIZE=self.embedding_size,
                N_USERS=self.n_users,
                N_ANIMES=self.n_animes
            ).to(self.device)
            logger.info(f"RecommenderNet initialized and moved to {self.device}")

        except FileNotFoundError as e:
            logger.error(f"Error loading mappings or config: {e}")
            raise CustomException("Failed to load mappings or config", sys)
        except Exception as e:
            logger.error(f"Error initializing BaseModel: {e}")
            raise CustomException("Failed to initialize BaseModel", sys)

    class RecommenderNet(nn.Module):
        def __init__(self, EMBEDDING_SIZE, N_USERS, N_ANIMES):
            super().__init__()

            self.embedding_dim = EMBEDDING_SIZE
            self.n_users = N_USERS
            self.n_animes = N_ANIMES

            # Embedding layers for users and animes
            self.user_emb_layer = nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.embedding_dim)
            self.anime_emb_layer = nn.Embedding(num_embeddings=self.n_animes, embedding_dim=self.embedding_dim)

            self.dense1 = nn.Linear(in_features=1, out_features=1)
            self.bn1 = nn.BatchNorm1d(num_features=1)
            self.sigmoid1 = nn.Sigmoid()
            self.dropout1 = nn.Dropout(p=0.2)

        def forward(self, user, anime):
            # user: tensor of shape (batch_size,)
            # anime: tensor of shape (batch_size,)

            # Get embeddings
            user_emb = self.user_emb_layer(user)  # Shape: (batch_size, embedding_dim)
            anime_emb = self.anime_emb_layer(anime)  # Shape: (batch_size, embedding_dim)

            # Normalize embeddings (L2 norm)
            user_emb_norm = user_emb / torch.norm(user_emb, dim=1, keepdim=True)
            anime_emb_norm = anime_emb / torch.norm(anime_emb, dim=1, keepdim=True)

            # Compute cosine similarity
            x = torch.sum(user_emb_norm * anime_emb_norm, dim=1, keepdim=True)  # Shape: (batch_size, 1)

            # Pass through dense layers
            x = self.dense1(x)
            x = self.bn1(x)
            x = self.sigmoid1(x)
            x = self.dropout1(x)

            return x

    def get_model(self):
        """Return the initialized RecommenderNet model."""
        return self.model

    def summary(self):
        """Print model summary."""
        try:
            from torchinfo import summary
            summary(self.model, input_size=[(1,), (1,)], dtypes=[torch.long, torch.long],
                    col_names=["input_size", "output_size", "num_params", "trainable"])
        except ImportError:
            logger.warning("torchinfo not installed. Install it with `pip install torchinfo` for model summary.")
        except Exception as e:
            logger.error(f"Error generating model summary: {e}")
            raise CustomException("Failed to generate model summary", sys)

if __name__ == "__main__":
    try:
        # Initialize BaseModel
        base_model = BaseModel()
        logger.info("BaseModel created successfully")


        base_model.summary()

    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise CustomException("Failed in main execution", sys)