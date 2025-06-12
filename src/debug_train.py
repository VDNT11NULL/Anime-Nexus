import numpy as np
import torch
from config.paths_config import *
from src.base_model import BaseModel
from utils.common_functions import read_yaml

def debug_data_ranges():
    """Debug script to check data ranges and model expectations"""
    
    # Load model configuration
    config = read_yaml(CONFIG_PATH)
    base_model = BaseModel(config_path=CONFIG_PATH)
    
    print(f"Model expects:")
    print(f"  n_users: {base_model.n_users} (valid range: 0 to {base_model.n_users-1})")
    print(f"  n_animes: {base_model.n_animes} (valid range: 0 to {base_model.n_animes-1})")
    print()
    
    # Load data
    print("Loading data...")
    X_train = np.load(X_TRAIN_ARRAY)
    X_test = np.load(X_TEST_ARRAY)
    Y_train = np.load(Y_TRAIN)
    Y_test = np.load(Y_TEST)
    
    print(f"Original shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  Y_train: {Y_train.shape}")
    print(f"  X_test: {X_test.shape}")
    print(f"  Y_test: {Y_test.shape}")
    print()
    
    # Handle transpose if needed
    if X_train.shape[0] != Y_train.shape[0]:
        X_train = X_train.T
        print(f"Transposed X_train to: {X_train.shape}")
    if X_test.shape[0] != Y_test.shape[0]:
        X_test = X_test.T
        print(f"Transposed X_test to: {X_test.shape}")
    print()
    
    # Check data ranges
    print("Data ranges:")
    print(f"Training data:")
    print(f"  User IDs: min={X_train[:, 0].min()}, max={X_train[:, 0].max()}")
    print(f"  Anime IDs: min={X_train[:, 1].min()}, max={X_train[:, 1].max()}")
    print(f"  Ratings: min={Y_train.min()}, max={Y_train.max()}")
    print()
    
    print(f"Test data:")
    print(f"  User IDs: min={X_test[:, 0].min()}, max={X_test[:, 0].max()}")
    print(f"  Anime IDs: min={X_test[:, 1].min()}, max={X_test[:, 1].max()}")
    print(f"  Ratings: min={Y_test.min()}, max={Y_test.max()}")
    print()
    
    # Check for issues
    issues = []
    
    if X_train[:, 0].min() < 0:
        issues.append("Negative user IDs in training data")
    if X_train[:, 1].min() < 0:
        issues.append("Negative anime IDs in training data")
    if X_train[:, 0].max() >= base_model.n_users:
        issues.append(f"User ID {X_train[:, 0].max()} exceeds model capacity {base_model.n_users}")
    if X_train[:, 1].max() >= base_model.n_animes:
        issues.append(f"Anime ID {X_train[:, 1].max()} exceeds model capacity {base_model.n_animes}")
    
    if X_test[:, 0].min() < 0:
        issues.append("Negative user IDs in test data")
    if X_test[:, 1].min() < 0:
        issues.append("Negative anime IDs in test data")
    if X_test[:, 0].max() >= base_model.n_users:
        issues.append(f"User ID {X_test[:, 0].max()} exceeds model capacity {base_model.n_users}")
    if X_test[:, 1].max() >= base_model.n_animes:
        issues.append(f"Anime ID {X_test[:, 1].max()} exceeds model capacity {base_model.n_animes}")
    
    if issues:
        print("❌ ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nThese issues will cause CUDA device-side assert errors!")
    else:
        print("✅ No issues found - data should work with the model")
    
    # Check data types
    print(f"\nData types:")
    print(f"  X_train dtype: {X_train.dtype}")
    print(f"  Y_train dtype: {Y_train.dtype}")
    print(f"  X_test dtype: {X_test.dtype}")
    print(f"  Y_test dtype: {Y_test.dtype}")
    
    # Sample a few records to inspect
    print(f"\nSample records (first 5):")
    for i in range(min(5, len(X_train))):
        print(f"  Record {i}: user_id={X_train[i, 0]}, anime_id={X_train[i, 1]}, rating={Y_train[i]}")

if __name__ == "__main__":
    debug_data_ranges()