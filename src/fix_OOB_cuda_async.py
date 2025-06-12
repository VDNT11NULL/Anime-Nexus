import numpy as np
import joblib
import os
from config.paths_config import *
from src.logger import get_logger

logger = get_logger(__name__)

def create_id_mappings():
    """Create proper ID mappings for users and animes"""
    
    print("Loading data...")
    X_train = np.load(X_TRAIN_ARRAY)
    X_test = np.load(X_TEST_ARRAY)
    
    # Handle transpose if needed
    if X_train.shape[0] != np.load(Y_TRAIN).shape[0]:
        X_train = X_train.T
    if X_test.shape[0] != np.load(Y_TEST).shape[0]:
        X_test = X_test.T
    
    print(f"Data shapes after transpose: X_train={X_train.shape}, X_test={X_test.shape}")
    
    # Get all unique user and anime IDs from both train and test sets
    all_user_ids = np.unique(np.concatenate([X_train[:, 0], X_test[:, 0]]))
    all_anime_ids = np.unique(np.concatenate([X_train[:, 1], X_test[:, 1]]))
    
    print(f"Found {len(all_user_ids)} unique users (range: {all_user_ids.min()} to {all_user_ids.max()})")
    print(f"Found {len(all_anime_ids)} unique animes (range: {all_anime_ids.min()} to {all_anime_ids.max()})")
    
    # Create mappings: original_id -> new_sequential_id
    user_id_mapping = {original_id: new_id for new_id, original_id in enumerate(all_user_ids)}
    anime_id_mapping = {original_id: new_id for new_id, original_id in enumerate(all_anime_ids)}
    
    print(f"Created mappings: {len(user_id_mapping)} users, {len(anime_id_mapping)} animes")
    
    # Save mappings for future use
    mappings_dir = os.path.join(os.path.dirname(USER_WEIGHTS_PATH), 'mappings')
    os.makedirs(mappings_dir, exist_ok=True)
    
    user_mapping_path = os.path.join(mappings_dir, 'user_id_mapping.pkl')
    anime_mapping_path = os.path.join(mappings_dir, 'anime_id_mapping.pkl')
    
    joblib.dump(user_id_mapping, user_mapping_path)
    joblib.dump(anime_id_mapping, anime_mapping_path)
    
    print(f"Saved user mapping to: {user_mapping_path}")
    print(f"Saved anime mapping to: {anime_mapping_path}")
    
    return user_id_mapping, anime_id_mapping, len(all_user_ids), len(all_anime_ids)

def remap_data(user_id_mapping, anime_id_mapping):
    """Apply the ID mappings to training and test data"""
    
    print("Remapping training and test data...")
    
    # Load original data
    X_train = np.load(X_TRAIN_ARRAY)
    X_test = np.load(X_TEST_ARRAY)
    Y_train = np.load(Y_TRAIN)
    Y_test = np.load(Y_TEST)
    
    # Handle transpose if needed
    if X_train.shape[0] != Y_train.shape[0]:
        X_train = X_train.T
    if X_test.shape[0] != Y_test.shape[0]:
        X_test = X_test.T
    
    # Apply mappings to training data
    X_train_remapped = X_train.copy()
    for i in range(len(X_train)):
        X_train_remapped[i, 0] = user_id_mapping[X_train[i, 0]]  # Remap user ID
        X_train_remapped[i, 1] = anime_id_mapping[X_train[i, 1]]  # Remap anime ID
    
    # Apply mappings to test data
    X_test_remapped = X_test.copy()
    for i in range(len(X_test)):
        X_test_remapped[i, 0] = user_id_mapping[X_test[i, 0]]  # Remap user ID
        X_test_remapped[i, 1] = anime_id_mapping[X_test[i, 1]]  # Remap anime ID
    
    print(f"Remapped data ranges:")
    print(f"  Training - Users: [{X_train_remapped[:, 0].min()}, {X_train_remapped[:, 0].max()}]")
    print(f"  Training - Animes: [{X_train_remapped[:, 1].min()}, {X_train_remapped[:, 1].max()}]")
    print(f"  Test - Users: [{X_test_remapped[:, 0].min()}, {X_test_remapped[:, 0].max()}]")
    print(f"  Test - Animes: [{X_test_remapped[:, 1].min()}, {X_test_remapped[:, 1].max()}]")
    
    # Save remapped data
    backup_dir = os.path.join(os.path.dirname(X_TRAIN_ARRAY), 'backup')
    os.makedirs(backup_dir, exist_ok=True)
    
    # Backup original files
    backup_files = [
        (X_TRAIN_ARRAY, os.path.join(backup_dir, 'X_train_original.npy')),
        (X_TEST_ARRAY, os.path.join(backup_dir, 'X_test_original.npy')),
    ]
    
    for original, backup in backup_files:
        if not os.path.exists(backup):
            original_data = np.load(original)
            np.save(backup, original_data)
            print(f"Backed up {original} to {backup}")
    
    # Save remapped data (overwrite originals)
    np.save(X_TRAIN_ARRAY, X_train_remapped)
    np.save(X_TEST_ARRAY, X_test_remapped)
    
    print(f"Saved remapped training data to: {X_TRAIN_ARRAY}")
    print(f"Saved remapped test data to: {X_TEST_ARRAY}")
    
    return X_train_remapped, X_test_remapped, Y_train, Y_test

def fix_data_id_mapping():
    """Main function to fix ID mapping issues"""
    
    print("=" * 60)
    print("FIXING ID MAPPING ISSUES")
    print("=" * 60)
    
    try:
        # Step 1: Create proper ID mappings
        user_id_mapping, anime_id_mapping, n_users, n_animes = create_id_mappings()
        
        # Step 2: Apply mappings to data
        X_train_remapped, X_test_remapped, Y_train, Y_test = remap_data(user_id_mapping, anime_id_mapping)
        
        # Step 3: Verify the fix
        print("\n" + "=" * 40)
        print("VERIFICATION")
        print("=" * 40)
        
        print(f"New data ranges:")
        print(f"  Users: 0 to {X_train_remapped[:, 0].max()} (expected: 0 to {n_users-1})")
        print(f"  Animes: 0 to {X_train_remapped[:, 1].max()} (expected: 0 to {n_animes-1})")
        
        # Check if fix was successful
        user_range_ok = X_train_remapped[:, 0].max() < n_users and X_test_remapped[:, 0].max() < n_users
        anime_range_ok = X_train_remapped[:, 1].max() < n_animes and X_test_remapped[:, 1].max() < n_animes
        
        if user_range_ok and anime_range_ok:
            print("✅ ID mapping fix successful!")
            print(f"✅ Users now range from 0 to {max(X_train_remapped[:, 0].max(), X_test_remapped[:, 0].max())}")
            print(f"✅ Animes now range from 0 to {max(X_train_remapped[:, 1].max(), X_test_remapped[:, 1].max())}")
        else:
            print("❌ ID mapping fix failed!")
            
        print(f"\nUpdate your BaseModel to use:")
        print(f"  n_users = {n_users}")
        print(f"  n_animes = {n_animes}")
        
        return n_users, n_animes
        
    except Exception as e:
        logger.error(f"Error fixing ID mappings: {e}")
        raise e

if __name__ == "__main__":
    fix_data_id_mapping()