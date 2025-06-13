import os
import pandas as pd
import numpy as np
import joblib
from config.paths_config import *
from src.custom_exception import CustomException
import torch
from collections import defaultdict

def getAnimeFrame(anime, path):
    df = pd.read_csv(path) if isinstance(path, str) else path
    if isinstance(anime, int):
        result = df[df.MAL_ID == anime]
    elif isinstance(anime, str):
        result = df[df.eng_version == anime]
    else:
        raise ValueError(f"Invalid anime identifier type: {type(anime)}")
        
    if result.empty:
        raise ValueError(f"{anime} not found in dataframe")
    return result

def getSynopsis(anime, df_path=SYNOPSIS_CSV):
    try:
        df = pd.read_csv(df_path) if isinstance(df_path, str) else df_path
        if isinstance(anime, int):
            result = df[df.MAL_ID == anime].synopsis
        elif isinstance(anime, str):
            result = df[df.Name == anime].synopsis
        else:
            raise ValueError(f"Invalid anime identifier type: {type(anime)}")
            
        if result.empty:
            raise ValueError(f"{anime} not found in synopsis dataframe")
        return result.values[0]
            
    except Exception as e:
        raise CustomException('Failed to get synopsis', e)

def getSimilarAnimes(anime, path_anime_embeddings_norm, path_anime_df, path_synopsis_df, 
                    path_animeId_2_encodedAnimeId_mapping, path_encodedAnimeId_2_animeId_mapping,
                    k=10, return_dist=False, neg=False):
    try:
        # Load data if input is a path, otherwise use as-is
        anime_embeddings_norm = joblib.load(path_anime_embeddings_norm) if isinstance(path_anime_embeddings_norm, str) else path_anime_embeddings_norm
        anime_df = pd.read_csv(path_anime_df) if isinstance(path_anime_df, str) else path_anime_df
        synopsis_df = pd.read_csv(path_synopsis_df) if isinstance(path_synopsis_df, str) else path_synopsis_df
        animeId_2_encodedAnimeId_mapping = joblib.load(path_animeId_2_encodedAnimeId_mapping) if isinstance(path_animeId_2_encodedAnimeId_mapping, str) else path_animeId_2_encodedAnimeId_mapping
        encodedAnimeId_2_animeId_mapping = joblib.load(path_encodedAnimeId_2_animeId_mapping) if isinstance(path_encodedAnimeId_2_animeId_mapping, str) else path_encodedAnimeId_2_animeId_mapping
        
        # Get the anime frame and verify it exists
        anime_frame = getAnimeFrame(anime, anime_df)
        org_idx = anime_frame.MAL_ID.values[0]
        
        # Verify the encoded mapping exists
        encoded_idx = animeId_2_encodedAnimeId_mapping.get(org_idx)
        if encoded_idx is None:
            raise ValueError(f"No encoded ID found for anime ID {org_idx}")
            
        # Convert embeddings to numpy if needed
        if isinstance(anime_embeddings_norm, torch.Tensor):
            anime_wts = anime_embeddings_norm.cpu().numpy() if anime_embeddings_norm.is_cuda else anime_embeddings_norm.numpy()
        elif not isinstance(anime_embeddings_norm, np.ndarray):
            raise TypeError("anime_embeddings_norm must be a torch.Tensor or numpy.ndarray")
        else:
            anime_wts = anime_embeddings_norm

        # Calculate similarities
        dot_prod = np.dot(anime_wts, anime_wts[encoded_idx])
        sorted_idxs = np.argsort(dot_prod)

        k += 1  # Include curr anime too
        closest = sorted_idxs[:k] if neg else sorted_idxs[-k:]
        
        if return_dist:
            return sorted_idxs, closest
        
        # Build results
        SimilarityArr = []
        for close in closest:
            decoded_id = encodedAnimeId_2_animeId_mapping.get(close)
            if decoded_id is None:
                continue
                
            try:
                synopsis = getSynopsis(decoded_id, synopsis_df)
                anime_frame = getAnimeFrame(decoded_id, anime_df)
                
                SimilarityArr.append({
                    "anime_id": decoded_id,
                    "name": anime_frame.eng_version.values[0],
                    "similarity": dot_prod[close],
                    "genre": anime_frame.Genres.values[0],
                    "synopsis": synopsis
                })
            except Exception as e:
                print(f"Skipping anime ID {decoded_id} due to error: {str(e)}")
                continue

        # Return sorted results excluding the original anime
        result_frame = pd.DataFrame(SimilarityArr).sort_values(by=["similarity"], ascending=False)
        return result_frame[result_frame.anime_id != org_idx].drop(["anime_id"], axis=1)

    except Exception as e:
        print(f"Error in getSimilarAnimes: {str(e)}")
        return None
    

def similar_users(user_id, path_user_embeddings, path_userId_2_encodedUserId_mapping,
                   path_encodedUserId_2_userId_mapping, k=10, return_dist=False, neg=False):
    """
    Find k most similar users to a given user_id based on user embeddings.
    Args:
        user_id (int): The user ID to find similar users for.
        path_user_embeddings (str or np.ndarray): Path to user embeddings or the embeddings array itself.
        path_userId_2_encodedUserId_mapping (str or dict): Path to mapping from userId to encodedUserId or the mapping itself.
        path_encodedUserId_2_userId_mapping (str or dict): Path to mapping from encodedUserId to userId or the mapping itself.
        k (int): Number of similar users to return.
        return_dist (bool): If True, returns distances instead of just user IDs and similarity scores.
        neg (bool): If True, returns least similar users instead of most similar.
    Returns:
        pd.DataFrame: DataFrame containing user IDs and their similarity scores, sorted by similarity.
    """
    try:
        user_embeddings = joblib.load(path_user_embeddings) if isinstance(path_user_embeddings, str) else path_user_embeddings
        userId_2_encodedUserId_mapping = joblib.load(path_userId_2_encodedUserId_mapping) if isinstance(path_userId_2_encodedUserId_mapping, str) else userId_2_encodedUserId_mapping
        encodedUserId_2_userId_mapping = joblib.load(path_encodedUserId_2_userId_mapping) if isinstance(path_encodedUserId_2_userId_mapping, str) else path_encodedUserId_2_userId_mapping
        # Validate user_id
        if not isinstance(user_id, int):
            raise ValueError(f"user_id must be an integer, got {type(user_id)}")
        if user_id not in userId_2_encodedUserId_mapping:
            raise ValueError(f"user_id {user_id} not found in userId_2_encodedUserId_mapping.") 


        if isinstance(user_embeddings, torch.Tensor):
            user_embeddings = user_embeddings.squeeze()
            if user_embeddings.is_cuda:
                user_wts = user_embeddings.cpu().numpy()
            else:
                user_wts = user_embeddings.numpy()
        elif isinstance(user_embeddings, np.ndarray):
            user_wts = np.squeeze(user_embeddings)
        else:
            raise TypeError("user_embeddings must be a torch.Tensor or numpy.ndarray")

        # print(f"user embd shape: {user_wts.shape}")

        # Fetch encoded index and validate
        encoded_idx = userId_2_encodedUserId_mapping.get(user_id)
        if encoded_idx is None:
            raise ValueError(f"user_id {user_id} not found in userId_2_encodedUserId_mapping.")

        user_vec = user_wts[encoded_idx]  # shape (128,)
        # print(f"user_vec shape: {user_vec.shape}")

        # Compute similarity
        dot_prod = np.dot(user_wts, user_vec)  # shape (7013,)
        sorted_idxs = np.argsort(dot_prod)

        k += 1  # to include the user itself
        closest = sorted_idxs[:k] if neg else sorted_idxs[-k:]

        SimilarityArr = []
        for close in closest:
            if close == encoded_idx: 
                continue
            similarity = dot_prod[close]
            decoded_id = encodedUserId_2_userId_mapping[close]
            SimilarityArr.append({
                "user_id": decoded_id,
                "similarity_score": similarity
            })

        frame = pd.DataFrame(SimilarityArr).sort_values(by=["similarity_score"],ascending=False,  kind="heapsort")
        return frame if not return_dist else (dot_prod, closest)

    except Exception as e:
        print(f"Failed to find {k} most similar users to {user_id} : {e}")

def getFavGenre(frame):
    frame.dropna(inplace=False)
    all_genres = defaultdict(int)

    genre_list = []
    for genres in frame["Genres"]:
        if isinstance(genres, str):
            for genre in genres.split(','):
                genre_list.append(genre)
                all_genres[genre.strip()] +=1 

    return genre_list

def get_user_preferences(user_id, rating_df_path = PROCESSED_RATING_DF, anime_df_path = PROCESSED_ANIME_DF, top_percentile=10, plot=False):
    try:
        rating_df = pd.read_csv(rating_df_path) if isinstance(rating_df_path, str) else rating_df_path
        anime_df = pd.read_csv(anime_df_path) if isinstance(anime_df_path, str) else anime_df_path
        animes_watched_by_user = rating_df[rating_df.user_id == user_id]
        ratings_by_user = rating_df[rating_df.user_id == user_id]["rating"]
        user_rating_percentile = np.percentile(ratings_by_user, 100-top_percentile) 
        ## Finf top 100-top_percentile rated animes
        animes_watched_by_user = animes_watched_by_user[animes_watched_by_user.rating >= user_rating_percentile]
        
        top_animes_rated_by_user = (
            animes_watched_by_user.sort_values(by=["rating"], ascending=False).anime_id.values.tolist()
        )
        top_anime_df = anime_df[anime_df["MAL_ID"].isin(top_animes_rated_by_user)]
        top_anime_df = top_anime_df[["eng_version", "Genres"]]

        if plot:
            getFavGenre(top_anime_df, plot)

        # print(top_animes_rated_by_user)
        # display(top_anime_df)
        return top_anime_df
    
    except Exception as e:
        print(f"error while finding user preferences {e}")

##### ACTUAL USER BASED REC SYS

def get_user_recommendations(similar_users, user_pref, anime_df_path, synopsis_df_path, rating_df_path,top_percentile=30, n=10):

    recommended_animes = []
    anime_list = []

    try:
        rating_df = pd.read_csv(rating_df_path) if isinstance(rating_df_path, str) else rating_df_path
        anime_df = pd.read_csv(anime_df_path) if isinstance(anime_df_path, str) else anime_df_path
        synopsis_df = pd.read_csv(synopsis_df_path) if isinstance(synopsis_df_path, str) else synopsis_df_path


        for user_id in similar_users.user_id.values:
            # what does users simialar to me prefer
            pref_list = get_user_preferences(user_id, rating_df, anime_df, top_percentile=top_percentile)
            # print(f"pref list :{pref_list.shape} for user {user_id} before filtering")
            
            # Rec some animes which (I) the user has no rated/watched yet
            pref_list = pref_list[~pref_list['eng_version'].isin(user_pref.eng_version.values)]

            # print(f"pref list :{pref_list.shape} for user {user_id} after filtering")
            ### ISSUE: if te user(I) is experienced, this filetrring out almost empty pref list of my similar users

            if not pref_list.empty:
                anime_list.extend(pref_list.eng_version.values.tolist())

            # print(f"User {user_id} has rated {len(pref_list)} animes with top percentile {top_percentile}")

            if anime_list:
                sorted_list = pd.Series(anime_list).value_counts().head(n)
                # print(f"anime list count: {len(anime_list)}, sorted list shape: {sorted_list.shape}")
                # print(f'Sorted list:\n{sorted_list}')

                for anime_name in sorted_list.index:
                    n_user_pref = sorted_list.loc[anime_name]

                    frame = getAnimeFrame(anime_name, anime_df)
                    if frame.empty:
                        # print(f"Frame not found for {anime_name}")
                        continue

                    anime_id = frame.MAL_ID.values[0]
                    genre = frame.Genres.values[0]
                    synopsis = getSynopsis(int(anime_id), synopsis_df)

                    recommended_animes.append({
                        "n_pref": n_user_pref,
                        "anime_name": anime_name,
                        "Genres": genre,
                        "Synopsis": synopsis
                    })
        return pd.DataFrame(recommended_animes).head(n)
    
    except Exception as e:
        print(f"Error {e} at find user recommendations")
        
def hybrid_rec_sys(user_id, anime_df_path, rating_df_path, synopsis_df, user_embeddings_norm_path, anime_embeddings_norm_path, 
                   userId_2_encodedUserId_mapping_path, encodedUserId_2_userId_mapping_path, 
                   animeId_2_encodedAnimeId_mapping_path, encodedAnimeId_2_animeId_mapping_path,
                   top_percentile=30, user_wts=0.5 , content_wts=0.5, num_rec2return=10, num_similar_animes2rec=10, final_num2_rec=10):

    anime_df = pd.read_csv(anime_df_path) if isinstance(anime_df_path, str) else anime_df_path
    rating_df = pd.read_csv(rating_df_path) if isinstance(rating_df_path, str) else rating_df_path
    synopsis_df = pd.read_csv(synopsis_df) if isinstance(synopsis_df, str) else synopsis_df
    user_embeddings_norm = joblib.load(user_embeddings_norm_path) if isinstance(user_embeddings_norm_path, str) else user_embeddings_norm_path
    anime_embeddings_norm = joblib.load(anime_embeddings_norm_path) if isinstance(anime_embeddings_norm_path, str) else anime_embeddings_norm_path
    userId_2_encodedUserId_mapping = joblib.load(userId_2_encodedUserId_mapping_path) if isinstance(userId_2_encodedUserId_mapping_path, str) else userId_2_encodedUserId_mapping_path
    encodedUserId_2_userId_mapping = joblib.load(encodedUserId_2_userId_mapping_path) if isinstance(encodedUserId_2_userId_mapping_path, str) else encodedUserId_2_userId_mapping_path
    animeId_2_encodedAnimeId_mapping = joblib.load(animeId_2_encodedAnimeId_mapping_path) if isinstance(animeId_2_encodedAnimeId_mapping_path, str) else animeId_2_encodedAnimeId_mapping_path
    encodedAnimeId_2_animeId_mapping = joblib.load(encodedAnimeId_2_animeId_mapping_path) if isinstance(encodedAnimeId_2_animeId_mapping_path, str) else encodedAnimeId_2_animeId_mapping_path

    ## USER RECOMMENDATIONS
    similar_users_list = similar_users(user_id, user_embeddings_norm, anime_df, rating_df,userId_2_encodedUserId_mapping, encodedUserId_2_userId_mapping)
    user_prefs = get_user_preferences(user_id, rating_df, anime_df, top_percentile)
    user_recommendations = get_user_recommendations(similar_users_list, user_prefs, anime_df, synopsis_df, rating_df, top_percentile, n=num_rec2return)
    user_recommendations_anime_list = user_recommendations.anime_name.values

    ## CONTENT REC 
    content_recommended_anime_list = []
    for anime in user_recommendations_anime_list:
        simi_animes = getSimilarAnimes(anime, anime_embeddings_norm, anime_df, synopsis_df, animeId_2_encodedAnimeId_mapping, encodedAnimeId_2_animeId_mapping, k=num_similar_animes2rec).name.values.tolist()
        
        if simi_animes is not None and not len(simi_animes)==0:
            content_recommended_anime_list.extend(simi_animes)

    ### WEIGHTED COMB of both
    combined_scores = {}

    for anime in user_recommendations_anime_list:
        combined_scores[anime] = combined_scores.get(anime, 0.0) + user_wts
        
    for anime in content_recommended_anime_list:
        combined_scores[anime] = combined_scores.get(anime, 0.0) + content_wts

    sorted_animes = sorted(combined_scores.items(), key=lambda x:x[1], reverse=True)    
    return [anime for anime, score in sorted_animes[:final_num2_rec]]
