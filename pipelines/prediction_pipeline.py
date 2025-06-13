from config.paths_config import *
from utils.helpers import *


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