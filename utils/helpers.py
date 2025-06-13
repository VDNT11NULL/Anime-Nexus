import os
import pandas as pd
import numpy as np
import joblib
from collections import defaultdict
import torch
from config.paths_config import *
from src.custom_exception import CustomException

def getAnimeFrame(anime, path, debug=False):
    try:
        df = pd.read_csv(path) if isinstance(path, str) else path
        if isinstance(anime, int):
            result = df[df.MAL_ID == anime]
        elif isinstance(anime, str):
            result = df[df.eng_version == anime]
        else:
            raise ValueError(f"Invalid anime identifier type: {type(anime)}")
        
        if result.empty:
            if debug:
                print(f"{anime} not found in dataframe")
            return pd.DataFrame()
        return result
    except Exception as e:
        if debug:
            print(f"Error in getAnimeFrame: {e}")
        return pd.DataFrame()

def getSynopsis(anime, df_path=SYNOPSIS_CSV, debug=False):
    try:
        df = pd.read_csv(df_path) if isinstance(df_path, str) else df_path
        if isinstance(anime, int):
            result = df[df.MAL_ID == anime].synopsis
        elif isinstance(anime, str):
            result = df[df.Name == anime].synopsis
        else:
            raise ValueError(f"Invalid anime identifier type: {type(anime)}")
            
        if result.empty:
            if debug:
                print(f"{anime} not found in synopsis dataframe")
            return ""
        return result.values[0]
            
    except Exception as e:
        if debug:
            print(f"Failed to get synopsis for {anime}: {e}")
        return ""

def getSimilarAnimes(anime, path_anime_embeddings_norm, path_anime_df, path_synopsis_df, 
                    path_animeId_2_encodedAnimeId_mapping, path_encodedAnimeId_2_animeId_mapping,
                    k=10, return_dist=False, neg=False, debug=False):
    try:
        anime_embeddings_norm = joblib.load(path_anime_embeddings_norm) if isinstance(path_anime_embeddings_norm, str) else path_anime_embeddings_norm
        anime_df = pd.read_csv(path_anime_df) if isinstance(path_anime_df, str) else path_anime_df
        synopsis_df = pd.read_csv(path_synopsis_df) if isinstance(path_synopsis_df, str) else path_synopsis_df
        animeId_2_encodedAnimeId_mapping = joblib.load(path_animeId_2_encodedAnimeId_mapping) if isinstance(path_animeId_2_encodedAnimeId_mapping, str) else path_animeId_2_encodedAnimeId_mapping
        encodedAnimeId_2_animeId_mapping = joblib.load(path_encodedAnimeId_2_animeId_mapping) if isinstance(path_encodedAnimeId_2_animeId_mapping, str) else path_encodedAnimeId_2_animeId_mapping
        
        if not isinstance(anime_df, pd.DataFrame) or not isinstance(synopsis_df, pd.DataFrame):
            if debug:
                print("Error: anime_df or synopsis_df is not a DataFrame")
            return pd.DataFrame()
        if not isinstance(animeId_2_encodedAnimeId_mapping, dict) or not isinstance(encodedAnimeId_2_animeId_mapping, dict):
            if debug:
                print("Error: anime ID mappings are not dictionaries")
            return pd.DataFrame()
        
        anime_frame = getAnimeFrame(anime, anime_df, debug=debug)
        if anime_frame.empty:
            if debug:
                print(f"Anime {anime} not found")
            return pd.DataFrame()
        org_idx = anime_frame.MAL_ID.values[0]
        
        encoded_idx = animeId_2_encodedAnimeId_mapping.get(org_idx)
        if encoded_idx is None:
            if debug:
                print(f"No encoded ID found for anime {org_idx}")
            return pd.DataFrame()
            
        if isinstance(anime_embeddings_norm, torch.Tensor):
            anime_wts = anime_embeddings_norm.cpu().numpy() if anime_embeddings_norm.is_cuda else anime_embeddings_norm.numpy()
        elif not isinstance(anime_embeddings_norm, np.ndarray):
            if debug:
                print("Error: anime_embeddings_norm must be a torch.Tensor or numpy.ndarray")
            return pd.DataFrame()
        else:
            anime_wts = anime_embeddings_norm

        dot_prod = np.dot(anime_wts, anime_wts[encoded_idx])
        sorted_idxs = np.argsort(dot_prod)

        k += 1
        closest = sorted_idxs[:k] if neg else sorted_idxs[-k:]
        
        if return_dist:
            return sorted_idxs, closest
        
        SimilarityArr = []
        for close in closest:
            decoded_id = encodedAnimeId_2_animeId_mapping.get(close)
            if decoded_id is None:
                if debug:
                    print(f"No decoded ID found for encoded index {close}")
                continue
                
            try:
                synopsis = getSynopsis(decoded_id, synopsis_df, debug=debug)
                anime_frame = getAnimeFrame(decoded_id, anime_df, debug=debug)
                
                if anime_frame.empty:
                    if debug:
                        print(f"Skipping anime ID {decoded_id} due to empty anime frame")
                    continue
                
                SimilarityArr.append({
                    "anime_id": decoded_id,
                    "name": anime_frame.eng_version.values[0],
                    "similarity": float(dot_prod[close]),
                    "genre": anime_frame.Genres.values[0] if not anime_frame.Genres.empty else "",
                    "synopsis": synopsis
                })
            except Exception as e:
                if debug:
                    print(f"Skipping anime ID {decoded_id} due to error: {str(e)}")
                continue

        if not SimilarityArr:
            if debug:
                print("No similar animes found")
            return pd.DataFrame()
        
        result_frame = pd.DataFrame(SimilarityArr).sort_values(by="similarity", ascending=False)
        return result_frame[result_frame.anime_id != org_idx].drop(["anime_id"], axis=1)

    except Exception as e:
        if debug:
            print(f"Error in getSimilarAnimes: {str(e)}")
        return pd.DataFrame()

def similar_users(user_id, path_user_embeddings, path_userId_2_encodedUserId_mapping,
                 path_encodedUserId_2_userId_mapping, k=10, return_dist=False, neg=False, debug=False):
    try:
        user_embeddings = joblib.load(path_user_embeddings) if isinstance(path_user_embeddings, str) else path_user_embeddings
        userId_2_encodedUserId_mapping = joblib.load(path_userId_2_encodedUserId_mapping) if isinstance(path_userId_2_encodedUserId_mapping, str) else path_userId_2_encodedUserId_mapping
        encodedUserId_2_userId_mapping = joblib.load(path_encodedUserId_2_userId_mapping) if isinstance(path_encodedUserId_2_userId_mapping, str) else path_encodedUserId_2_userId_mapping
        
        if not isinstance(userId_2_encodedUserId_mapping, dict) or not isinstance(encodedUserId_2_userId_mapping, dict):
            if debug:
                print("Error: user ID mappings are not dictionaries")
            return pd.DataFrame()

        if not isinstance(user_id, int):
            if debug:
                print(f"Error: user_id must be an integer, got {type(user_id)}")
            return pd.DataFrame()
        if user_id not in userId_2_encodedUserId_mapping:
            if debug:
                print(f"Error: user_id {user_id} not found in userId_2_encodedUserId_mapping")
            return pd.DataFrame()

        if isinstance(user_embeddings, torch.Tensor):
            user_embeddings = user_embeddings.squeeze()
            user_wts = user_embeddings.cpu().numpy() if user_embeddings.is_cuda else user_embeddings.numpy()
        elif isinstance(user_embeddings, np.ndarray):
            user_wts = np.squeeze(user_embeddings)
        else:
            if debug:
                print("Error: user_embeddings must be a torch.Tensor or numpy.ndarray")
            return pd.DataFrame()

        encoded_idx = userId_2_encodedUserId_mapping.get(user_id)
        if encoded_idx is None:
            if debug:
                print(f"Error: user_id {user_id} not found in userId_2_encodedUserId_mapping")
            return pd.DataFrame()

        user_vec = user_wts[encoded_idx]
        dot_prod = np.dot(user_wts, user_vec)
        sorted_idxs = np.argsort(dot_prod)

        k += 1
        closest = sorted_idxs[:k] if neg else sorted_idxs[-k:]

        SimilarityArr = []
        for close in closest:
            if close == encoded_idx: 
                continue
            similarity = float(dot_prod[close])
            decoded_id = encodedUserId_2_userId_mapping.get(close)
            if decoded_id is None:
                if debug:
                    print(f"No decoded ID found for encoded index {close}")
                continue
            SimilarityArr.append({
                "user_id": decoded_id,
                "similarity_score": similarity
            })

        if not SimilarityArr:
            if debug:
                print(f"No similar users found for user_id {user_id}")
            return pd.DataFrame()
        
        frame = pd.DataFrame(SimilarityArr).sort_values(by="similarity_score", ascending=False, kind="heapsort")
        return frame if not return_dist else (dot_prod, closest)

    except Exception as e:
        if debug:
            print(f"Failed to find {k} most similar users to {user_id}: {e}")
        return pd.DataFrame()

def getFavGenre(frame, plot=False, debug=False):
    try:
        frame = frame.dropna()
        all_genres = defaultdict(int)
        genre_list = []
        for genres in frame["Genres"]:
            if isinstance(genres, str):
                for genre in genres.split(','):
                    genre = genre.strip()
                    genre_list.append(genre)
                    all_genres[genre] += 1 

        if plot:
            import matplotlib.pyplot as plt
            genres, counts = zip(*all_genres.items())
            plt.bar(genres, counts, color='#1f77b4')
            plt.xticks(rotation=45)
            plt.xlabel("Genres")
            plt.ylabel("Count")
            plt.title("Favorite Genres")
            plt.tight_layout()
            plt.show()
        
        return genre_list
    except Exception as e:
        if debug:
            print(f"Error in getFavGenre: {e}")
        return []

def get_user_preferences(user_id, rating_df_path=PROCESSED_RATING_DF, anime_df_path=PROCESSED_ANIME_DF, top_percentile=10, plot=False, debug=False):
    try:
        rating_df = pd.read_csv(rating_df_path) if isinstance(rating_df_path, str) else rating_df_path
        anime_df = pd.read_csv(anime_df_path) if isinstance(anime_df_path, str) else anime_df_path
        animes_watched_by_user = rating_df[rating_df.user_id == user_id]
        if animes_watched_by_user.empty:
            if debug:
                print(f"No ratings found for user_id {user_id}")
            return pd.DataFrame()
        ratings_by_user = animes_watched_by_user["rating"]
        user_rating_percentile = np.percentile(ratings_by_user, 100 - top_percentile)
        animes_watched_by_user = animes_watched_by_user[animes_watched_by_user.rating >= user_rating_percentile]
        
        top_animes_rated_by_user = animes_watched_by_user.sort_values(by=["rating"], ascending=False).anime_id.values.tolist()
        top_anime_df = anime_df[anime_df["MAL_ID"].isin(top_animes_rated_by_user)]
        top_anime_df = top_anime_df[["eng_version", "Genres"]]

        if plot:
            getFavGenre(top_anime_df, plot=True, debug=debug)

        return top_anime_df
    
    except Exception as e:
        if debug:
            print(f"Error while finding user preferences for user_id {user_id}: {e}")
        return pd.DataFrame()

def get_user_recommendations(similar_users, user_pref, anime_df_path, synopsis_df_path, rating_df_path, top_percentile=30, n=10, debug=False):
    if similar_users is None or similar_users.empty:
        if debug:
            print("Error: similar_users is None or empty")
        return pd.DataFrame()
    
    recommended_animes = []
    anime_list = []

    try:
        rating_df = pd.read_csv(rating_df_path) if isinstance(rating_df_path, str) else rating_df_path
        anime_df = pd.read_csv(anime_df_path) if isinstance(anime_df_path, str) else anime_df_path
        synopsis_df = pd.read_csv(synopsis_df_path) if isinstance(synopsis_df_path, str) else synopsis_df_path

        for user_id in similar_users.user_id.values:
            pref_list = get_user_preferences(user_id, rating_df, anime_df, top_percentile=top_percentile, debug=debug)
            if pref_list.empty:
                if debug:
                    print(f"No preferences found for user_id {user_id}")
                continue
            pref_list = pref_list[~pref_list['eng_version'].isin(user_pref.eng_version.values)]

            if not pref_list.empty:
                anime_list.extend(pref_list.eng_version.values.tolist())

            if anime_list:
                sorted_list = pd.Series(anime_list).value_counts().head(n)
                for anime_name in sorted_list.index:
                    n_user_pref = sorted_list.loc[anime_name]
                    frame = getAnimeFrame(anime_name, anime_df, debug=debug)
                    if frame.empty:
                        if debug:
                            print(f"Frame not found for {anime_name}")
                        continue
                    anime_id = frame.MAL_ID.values[0]
                    genre = frame.Genres.values[0] if not frame.Genres.empty else ""
                    synopsis = getSynopsis(int(anime_id), synopsis_df, debug=debug)
                    recommended_animes.append({
                        "n_pref": n_user_pref,
                        "anime_name": anime_name,
                        "Genres": genre,
                        "Synopsis": synopsis
                    })
        return pd.DataFrame(recommended_animes).head(n)
    
    except Exception as e:
        if debug:
            print(f"Error at find user recommendations: {e}")
        return pd.DataFrame()

def hybrid_rec_sys(user_id, top_percentile=30, user_wts=0.6, content_wts=0.4, 
                   num_rec2return=10, num_similar_animes2rec=10, final_num2_rec=10, debug=False):
    try:
        # Load data from config.paths_config
        anime_df = pd.read_csv(PROCESSED_ANIME_DF) if isinstance(PROCESSED_ANIME_DF, str) else PROCESSED_ANIME_DF
        rating_df = pd.read_csv(PROCESSED_RATING_DF) if isinstance(PROCESSED_RATING_DF, str) else PROCESSED_RATING_DF
        synopsis_df = pd.read_csv(PROCESSED_SYNOPSIS_DF) if isinstance(PROCESSED_SYNOPSIS_DF, str) else PROCESSED_SYNOPSIS_DF
        user_embeddings_norm = joblib.load(USER_WEIGHTS_PATH) if isinstance(USER_WEIGHTS_PATH, str) else USER_WEIGHTS_PATH
        anime_embeddings_norm = joblib.load(ANIME_WEIGHTS_PATH) if isinstance(ANIME_WEIGHTS_PATH, str) else ANIME_WEIGHTS_PATH
        userId_2_encodedUserId_mapping = joblib.load(USERID_2_ENCODEDUSERID_MAPPING) if isinstance(USERID_2_ENCODEDUSERID_MAPPING, str) else USERID_2_ENCODEDUSERID_MAPPING
        encodedUserId_2_userId_mapping = joblib.load(ENCODEDUSERID_2_USERID_MAPPING) if isinstance(ENCODEDUSERID_2_USERID_MAPPING, str) else ENCODEDUSERID_2_USERID_MAPPING
        animeId_2_encodedAnimeId_mapping = joblib.load(ANIMEID_2_ENCODEDANIMEID_MAPPING) if isinstance(ANIMEID_2_ENCODEDANIMEID_MAPPING, str) else ANIMEID_2_ENCODEDANIMEID_MAPPING
        encodedAnimeId_2_animeId_mapping = joblib.load(ENCODEDANIMEID_2_ANIMEID_MAPPING) if isinstance(ENCODEDANIMEID_2_ANIMEID_MAPPING, str) else ENCODEDANIMEID_2_ANIMEID_MAPPING

        # USER RECOMMENDATIONS
        similar_users_list = similar_users(user_id, user_embeddings_norm, userId_2_encodedUserId_mapping, 
                                     encodedUserId_2_userId_mapping, k=num_rec2return, neg=False, debug=debug)
        
        user_prefs = get_user_preferences(user_id, rating_df, anime_df, top_percentile, debug=debug)
        user_recommendations = get_user_recommendations(similar_users_list, user_prefs, anime_df, synopsis_df, 
                                                       rating_df, top_percentile, n=num_rec2return, debug=debug)
        
        user_recommendations_anime_list = user_recommendations.anime_name.values if not user_recommendations.empty else []

        # CONTENT RECOMMENDATIONS
        content_recommended_anime_list = []
        for anime in user_recommendations_anime_list:
            simi_animes = getSimilarAnimes(anime, anime_embeddings_norm, anime_df, synopsis_df, 
                                           animeId_2_encodedAnimeId_mapping, encodedAnimeId_2_animeId_mapping, 
                                           k=num_similar_animes2rec, debug=debug)
            if simi_animes is not None and not simi_animes.empty:
                content_recommended_anime_list.extend(simi_animes.name.values.tolist())

        # WEIGHTED COMBINATION
        combined_scores = {}
        for anime in user_recommendations_anime_list:
            combined_scores[anime] = combined_scores.get(anime, 0.0) + user_wts
        for anime in content_recommended_anime_list:
            combined_scores[anime] = combined_scores.get(anime, 0.0) + content_wts

        sorted_animes = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)    
        return [anime for anime, score in sorted_animes[:final_num2_rec]]
    
    except Exception as e:
        if debug:
            print(f"Error in hybrid_rec_sys: {e}")
        return []

if __name__ == "__main__":
    TMP_USER_ID = 373
    # Minimal call with only user_id
    result = hybrid_rec_sys(TMP_USER_ID)
    print("Recommendations:", result)

    # Example with custom parameters
    result = hybrid_rec_sys(
        TMP_USER_ID,
        top_percentile=30,
        user_wts=0.6,
        content_wts=0.4,
        num_rec2return=10,
        num_similar_animes2rec=10,
        final_num2_rec=10,
        debug=False
    )
    print("Recommendations with custom parameters:", result)


# import os
# import pandas as pd
# import numpy as np
# import joblib
# from collections import defaultdict
# import torch
# from config.paths_config import *
# from src.custom_exception import CustomException

# def getAnimeFrame(anime, path, debug=False):
#     try:
#         df = pd.read_csv(path) if isinstance(path, str) else path
#         if isinstance(anime, int):
#             result = df[df.MAL_ID == anime]
#         elif isinstance(anime, str):
#             result = df[df.eng_version == anime]
#         else:
#             raise ValueError(f"Invalid anime identifier type: {type(anime)}")
        
#         if result.empty:
#             if debug:
#                 print(f"{anime} not found in dataframe")
#             return pd.DataFrame()
#         return result
#     except Exception as e:
#         if debug:
#             print(f"Error in getAnimeFrame: {e}")
#         return pd.DataFrame()

# def getSynopsis(anime, df_path=SYNOPSIS_CSV, debug=False):
#     try:
#         df = pd.read_csv(df_path) if isinstance(df_path, str) else df_path
#         if isinstance(anime, int):
#             result = df[df.MAL_ID == anime].synopsis
#         elif isinstance(anime, str):
#             result = df[df.Name == anime].synopsis
#         else:
#             raise ValueError(f"Invalid anime identifier type: {type(anime)}")
            
#         if result.empty:
#             if debug:
#                 print(f"{anime} not found in synopsis dataframe")
#             return ""
#         return result.values[0]
            
#     except Exception as e:
#         if debug:
#             print(f"Failed to get synopsis for {anime}: {e}")
#         return ""

# def getSimilarAnimes(anime, path_anime_embeddings_norm, path_anime_df, path_synopsis_df, 
#                     path_animeId_2_encodedAnimeId_mapping, path_encodedAnimeId_2_animeId_mapping,
#                     k=10, return_dist=False, neg=False, debug=False):
#     try:
#         anime_embeddings_norm = joblib.load(path_anime_embeddings_norm) if isinstance(path_anime_embeddings_norm, str) else path_anime_embeddings_norm
#         anime_df = pd.read_csv(path_anime_df) if isinstance(path_anime_df, str) else path_anime_df
#         synopsis_df = pd.read_csv(path_synopsis_df) if isinstance(path_synopsis_df, str) else path_synopsis_df
#         animeId_2_encodedAnimeId_mapping = joblib.load(path_animeId_2_encodedAnimeId_mapping) if isinstance(path_animeId_2_encodedAnimeId_mapping, str) else path_animeId_2_encodedAnimeId_mapping
#         encodedAnimeId_2_animeId_mapping = joblib.load(path_encodedAnimeId_2_animeId_mapping) if isinstance(path_encodedAnimeId_2_animeId_mapping, str) else path_encodedAnimeId_2_animeId_mapping
        
#         if not isinstance(anime_df, pd.DataFrame) or not isinstance(synopsis_df, pd.DataFrame):
#             if debug:
#                 print("Error: anime_df or synopsis_df is not a DataFrame")
#             return pd.DataFrame()
#         if not isinstance(animeId_2_encodedAnimeId_mapping, dict) or not isinstance(encodedAnimeId_2_animeId_mapping, dict):
#             if debug:
#                 print("Error: anime ID mappings are not dictionaries")
#             return pd.DataFrame()
        
#         anime_frame = getAnimeFrame(anime, anime_df, debug=debug)
#         if anime_frame.empty:
#             if debug:
#                 print(f"Anime {anime} not found")
#             return pd.DataFrame()
#         org_idx = anime_frame.MAL_ID.values[0]
        
#         encoded_idx = animeId_2_encodedAnimeId_mapping.get(org_idx)
#         if encoded_idx is None:
#             if debug:
#                 print(f"No encoded ID found for anime {org_idx}")
#             return pd.DataFrame()
            
#         if isinstance(anime_embeddings_norm, torch.Tensor):
#             anime_wts = anime_embeddings_norm.cpu().numpy() if anime_embeddings_norm.is_cuda else anime_embeddings_norm.numpy()
#         elif not isinstance(anime_embeddings_norm, np.ndarray):
#             if debug:
#                 print("Error: anime_embeddings_norm must be a torch.Tensor or numpy.ndarray")
#             return pd.DataFrame()
#         else:
#             anime_wts = anime_embeddings_norm

#         dot_prod = np.dot(anime_wts, anime_wts[encoded_idx])
#         sorted_idxs = np.argsort(dot_prod)

#         k += 1
#         closest = sorted_idxs[:k] if neg else sorted_idxs[-k:]
        
#         if return_dist:
#             return sorted_idxs, closest
        
#         SimilarityArr = []
#         for close in closest:
#             decoded_id = encodedAnimeId_2_animeId_mapping.get(close)
#             if decoded_id is None:
#                 if debug:
#                     print(f"No decoded ID found for encoded index {close}")
#                 continue
                
#             try:
#                 synopsis = getSynopsis(decoded_id, synopsis_df, debug=debug)
#                 anime_frame = getAnimeFrame(decoded_id, anime_df, debug=debug)
                
#                 if anime_frame.empty:
#                     if debug:
#                         print(f"Skipping anime ID {decoded_id} due to empty anime frame")
#                     continue
                
#                 SimilarityArr.append({
#                     "anime_id": decoded_id,
#                     "name": anime_frame.eng_version.values[0],
#                     "similarity": float(dot_prod[close]),
#                     "genre": anime_frame.Genres.values[0] if not anime_frame.Genres.empty else "",
#                     "synopsis": synopsis
#                 })
#             except Exception as e:
#                 if debug:
#                     print(f"Skipping anime ID {decoded_id} due to error: {str(e)}")
#                 continue

#         if not SimilarityArr:
#             if debug:
#                 print("No similar animes found")
#             return pd.DataFrame()
        
#         result_frame = pd.DataFrame(SimilarityArr).sort_values(by="similarity", ascending=False)
#         return result_frame[result_frame.anime_id != org_idx].drop(["anime_id"], axis=1)

#     except Exception as e:
#         if debug:
#             print(f"Error in getSimilarAnimes: {str(e)}")
#         return pd.DataFrame()

# def similar_users(user_id, path_user_embeddings, path_userId_2_encodedUserId_mapping,
#                  path_encodedUserId_2_userId_mapping, k=10, return_dist=False, neg=False, debug=False):
#     try:
#         user_embeddings = joblib.load(path_user_embeddings) if isinstance(path_user_embeddings, str) else path_user_embeddings
#         userId_2_encodedUserId_mapping = joblib.load(path_userId_2_encodedUserId_mapping) if isinstance(path_userId_2_encodedUserId_mapping, str) else path_userId_2_encodedUserId_mapping
#         encodedUserId_2_userId_mapping = joblib.load(path_encodedUserId_2_userId_mapping) if isinstance(path_encodedUserId_2_userId_mapping, str) else path_encodedUserId_2_userId_mapping
        
#         if not isinstance(userId_2_encodedUserId_mapping, dict) or not isinstance(encodedUserId_2_userId_mapping, dict):
#             if debug:
#                 print("Error: user ID mappings are not dictionaries")
#             return pd.DataFrame()

#         if not isinstance(user_id, int):
#             if debug:
#                 print(f"Error: user_id must be an integer, got {type(user_id)}")
#             return pd.DataFrame()
#         if user_id not in userId_2_encodedUserId_mapping:
#             if debug:
#                 print(f"Error: user_id {user_id} not found in userId_2_encodedUserId_mapping")
#             return pd.DataFrame()

#         if isinstance(user_embeddings, torch.Tensor):
#             user_embeddings = user_embeddings.squeeze()
#             user_wts = user_embeddings.cpu().numpy() if user_embeddings.is_cuda else user_embeddings.numpy()
#         elif isinstance(user_embeddings, np.ndarray):
#             user_wts = np.squeeze(user_embeddings)
#         else:
#             if debug:
#                 print("Error: user_embeddings must be a torch.Tensor or numpy.ndarray")
#             return pd.DataFrame()

#         encoded_idx = userId_2_encodedUserId_mapping.get(user_id)
#         if encoded_idx is None:
#             if debug:
#                 print(f"Error: user_id {user_id} not found in userId_2_encodedUserId_mapping")
#             return pd.DataFrame()

#         user_vec = user_wts[encoded_idx]
#         dot_prod = np.dot(user_wts, user_vec)
#         sorted_idxs = np.argsort(dot_prod)

#         k += 1
#         closest = sorted_idxs[:k] if neg else sorted_idxs[-k:]

#         SimilarityArr = []
#         for close in closest:
#             if close == encoded_idx: 
#                 continue
#             similarity = float(dot_prod[close])
#             decoded_id = encodedUserId_2_userId_mapping.get(close)
#             if decoded_id is None:
#                 if debug:
#                     print(f"No decoded ID found for encoded index {close}")
#                 continue
#             SimilarityArr.append({
#                 "user_id": decoded_id,
#                 "similarity_score": similarity
#             })

#         if not SimilarityArr:
#             if debug:
#                 print(f"No similar users found for user_id {user_id}")
#             return pd.DataFrame()
        
#         frame = pd.DataFrame(SimilarityArr).sort_values(by="similarity_score", ascending=False, kind="heapsort")
#         return frame if not return_dist else (dot_prod, closest)

#     except Exception as e:
#         if debug:
#             print(f"Failed to find {k} most similar users to {user_id}: {e}")
#         return pd.DataFrame()

# def getFavGenre(frame, plot=False, debug=False):
#     try:
#         frame = frame.dropna()
#         all_genres = defaultdict(int)
#         genre_list = []
#         for genres in frame["Genres"]:
#             if isinstance(genres, str):
#                 for genre in genres.split(','):
#                     genre = genre.strip()
#                     genre_list.append(genre)
#                     all_genres[genre] += 1 

#         if plot:
#             import matplotlib.pyplot as plt
#             genres, counts = zip(*all_genres.items())
#             plt.bar(genres, counts, color='#1f77b4')
#             plt.xticks(rotation=45)
#             plt.xlabel("Genres")
#             plt.ylabel("Count")
#             plt.title("Favorite Genres")
#             plt.tight_layout()
#             plt.show()
        
#         return genre_list
#     except Exception as e:
#         if debug:
#             print(f"Error in getFavGenre: {e}")
#         return []

# def get_user_preferences(user_id, rating_df_path=PROCESSED_RATING_DF, anime_df_path=PROCESSED_ANIME_DF, top_percentile=10, plot=False, debug=False):
#     try:
#         rating_df = pd.read_csv(rating_df_path) if isinstance(rating_df_path, str) else rating_df_path
#         anime_df = pd.read_csv(anime_df_path) if isinstance(anime_df_path, str) else anime_df_path
#         animes_watched_by_user = rating_df[rating_df.user_id == user_id]
#         if animes_watched_by_user.empty:
#             if debug:
#                 print(f"No ratings found for user_id {user_id}")
#             return pd.DataFrame()
#         ratings_by_user = animes_watched_by_user["rating"]
#         user_rating_percentile = np.percentile(ratings_by_user, 100 - top_percentile)
#         animes_watched_by_user = animes_watched_by_user[animes_watched_by_user.rating >= user_rating_percentile]
        
#         top_animes_rated_by_user = animes_watched_by_user.sort_values(by=["rating"], ascending=False).anime_id.values.tolist()
#         top_anime_df = anime_df[anime_df["MAL_ID"].isin(top_animes_rated_by_user)]
#         top_anime_df = top_anime_df[["eng_version", "Genres"]]

#         if plot:
#             getFavGenre(top_anime_df, plot=True, debug=debug)

#         return top_anime_df
    
#     except Exception as e:
#         if debug:
#             print(f"Error while finding user preferences for user_id {user_id}: {e}")
#         return pd.DataFrame()

# def get_user_recommendations(similar_users, user_pref, anime_df_path, synopsis_df_path, rating_df_path, top_percentile=30, n=10, debug=False):
#     if similar_users is None or similar_users.empty:
#         if debug:
#             print("Error: similar_users is None or empty")
#         return pd.DataFrame()
    
#     recommended_animes = []
#     anime_list = []

#     try:
#         rating_df = pd.read_csv(rating_df_path) if isinstance(rating_df_path, str) else rating_df_path
#         anime_df = pd.read_csv(anime_df_path) if isinstance(anime_df_path, str) else anime_df_path
#         synopsis_df = pd.read_csv(synopsis_df_path) if isinstance(synopsis_df_path, str) else synopsis_df_path

#         for user_id in similar_users.user_id.values:
#             pref_list = get_user_preferences(user_id, rating_df, anime_df, top_percentile=top_percentile, debug=debug)
#             if pref_list.empty:
#                 if debug:
#                     print(f"No preferences found for user_id {user_id}")
#                 continue
#             pref_list = pref_list[~pref_list['eng_version'].isin(user_pref.eng_version.values)]

#             if not pref_list.empty:
#                 anime_list.extend(pref_list.eng_version.values.tolist())

#             if anime_list:
#                 sorted_list = pd.Series(anime_list).value_counts().head(n)
#                 for anime_name in sorted_list.index:
#                     n_user_pref = sorted_list.loc[anime_name]
#                     frame = getAnimeFrame(anime_name, anime_df, debug=debug)
#                     if frame.empty:
#                         if debug:
#                             print(f"Frame not found for {anime_name}")
#                         continue
#                     anime_id = frame.MAL_ID.values[0]
#                     genre = frame.Genres.values[0] if not frame.Genres.empty else ""
#                     synopsis = getSynopsis(int(anime_id), synopsis_df, debug=debug)
#                     recommended_animes.append({
#                         "n_pref": n_user_pref,
#                         "anime_name": anime_name,
#                         "Genres": genre,
#                         "Synopsis": synopsis
#                     })
#         return pd.DataFrame(recommended_animes).head(n)
    
#     except Exception as e:
#         if debug:
#             print(f"Error at find user recommendations: {e}")
#         return pd.DataFrame()

# def hybrid_rec_sys(user_id, top_percentile=30, user_wts=0.6, content_wts=0.4, 
#                    num_rec2return=10, num_similar_animes2rec=10, final_num2_rec=10, debug=False):
#     try:
#         # Load data from config.paths_config
#         anime_df = pd.read_csv(PROCESSED_ANIME_DF) if isinstance(PROCESSED_ANIME_DF, str) else PROCESSED_ANIME_DF
#         rating_df = pd.read_csv(PROCESSED_RATING_DF) if isinstance(PROCESSED_RATING_DF, str) else PROCESSED_RATING_DF
#         synopsis_df = pd.read_csv(PROCESSED_SYNOPSIS_DF) if isinstance(PROCESSED_SYNOPSIS_DF, str) else PROCESSED_SYNOPSIS_DF
#         user_embeddings_norm = joblib.load(USER_WEIGHTS_PATH) if isinstance(USER_WEIGHTS_PATH, str) else USER_WEIGHTS_PATH
#         anime_embeddings_norm = joblib.load(ANIME_WEIGHTS_PATH) if isinstance(ANIME_WEIGHTS_PATH, str) else ANIME_WEIGHTS_PATH
#         userId_2_encodedUserId_mapping = joblib.load(USERID_2_ENCODEDUSERID_MAPPING) if isinstance(USERID_2_ENCODEDUSERID_MAPPING, str) else USERID_2_ENCODEDUSERID_MAPPING
#         encodedUserId_2_userId_mapping = joblib.load(ENCODEDUSERID_2_USERID_MAPPING) if isinstance(ENCODEDUSERID_2_USERID_MAPPING, str) else ENCODEDUSERID_2_USERID_MAPPING
#         animeId_2_encodedAnimeId_mapping = joblib.load(ANIMEID_2_ENCODEDANIMEID_MAPPING) if isinstance(ANIMEID_2_ENCODEDANIMEID_MAPPING, str) else ANIMEID_2_ENCODEDANIMEID_MAPPING
#         encodedAnimeId_2_animeId_mapping = joblib.load(ENCODEDANIMEID_2_ANIMEID_MAPPING) if isinstance(ENCODEDANIMEID_2_ANIMEID_MAPPING, str) else ENCODEDANIMEID_2_ANIMEID_MAPPING

#         # USER RECOMMENDATIONS
#         similar_users_list = similar_users(user_id, user_embeddings_norm, userId_2_encodedUserId_mapping, 
#                                      encodedUserId_2_userId_mapping, k=num_rec2return, neg=False, debug=debug)
        
#         user_prefs = get_user_preferences(user_id, rating_df, anime_df, top_percentile, debug=debug)
#         user_recommendations = get_user_recommendations(similar_users_list, user_prefs, anime_df, synopsis_df, 
#                                                        rating_df, top_percentile, n=num_rec2return, debug=debug)
        
#         user_recommendations_anime_list = user_recommendations.anime_name.values if not user_recommendations.empty else []

#         # CONTENT RECOMMENDATIONS
#         content_recommended_anime_list = []
#         for anime in user_recommendations_anime_list:
#             simi_animes = getSimilarAnimes(anime, anime_embeddings_norm, anime_df, synopsis_df, 
#                                            animeId_2_encodedAnimeId_mapping, encodedAnimeId_2_animeId_mapping, 
#                                            k=num_similar_animes2rec, debug=debug)
#             if simi_animes is not None and not simi_animes.empty:
#                 content_recommended_anime_list.extend(simi_animes.name.values.tolist())

#         # WEIGHTED COMBINATION
#         combined_scores = {}
#         for anime in user_recommendations_anime_list:
#             combined_scores[anime] = combined_scores.get(anime, 0.0) + user_wts
#         for anime in content_recommended_anime_list:
#             combined_scores[anime] = combined_scores.get(anime, 0.0) + content_wts

#         # Fetch additional details for final recommendations
#         sorted_animes = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:final_num2_rec]
#         recommendations = []
#         for anime_name, score in sorted_animes:
#             anime_frame = getAnimeFrame(anime_name, anime_df, debug=debug)
#             synopsis = getSynopsis(anime_name, synopsis_df, debug=debug)
#             genres = anime_frame.Genres.values[0] if not anime_frame.empty and not anime_frame.Genres.empty else ""
#             fallback = ""
#             if not genres and not anime_frame.empty:
#                 # Fallback to Type or Score if available
#                 if "Type" in anime_frame and not anime_frame.Type.empty:
#                     fallback = f"Type: {anime_frame.Type.values[0]}"
#                 elif "Score" in anime_frame and not anime_frame.Score.empty:
#                     fallback = f"Score: {anime_frame.Score.values[0]}"
#             recommendations.append({
#                 "title": anime_name,
#                 "synopsis": synopsis if synopsis else "Synopsis not available",
#                 "genres": genres if genres else "Genres not available",
#                 "fallback": fallback if fallback else "No additional info available"
#             })
        
#         return recommendations
    
#     except Exception as e:
#         if debug:
#             print(f"Error in hybrid_rec_sys: {e}")
#         return []

# if __name__ == "__main__":
#     TMP_USER_ID = 373
#     # Minimal call with only user_id
#     result = hybrid_rec_sys(TMP_USER_ID, debug=True)
#     print("Recommendations:", result)

#     # Example with custom parameters
#     result = hybrid_rec_sys(
#         TMP_USER_ID,
#         top_percentile=30,
#         user_wts=0.6,
#         content_wts=0.4,
#         num_rec2return=10,
#         num_similar_animes2rec=10,
#         final_num2_rec=5,  # Match UI
#         debug=True
#     )
#     print("Recommendations with custom parameters:", result)