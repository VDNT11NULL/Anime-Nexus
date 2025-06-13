from utils.helpers import *
from config.paths_config import *

if __name__ == "__main__":
    # Test getAnimeFrame
    # print(getAnimeFrame(471, ANIME_CSV))
    # print(getAnimeFrame("Fullmetal Alchemist:Brotherhood", PROCESSED_ANIME_DF))
    
    # Test getSynopsis
    # print(getSynopsis("Monster", SYNOPSIS_CSV))

    # Test getSimilarAnimes
    # print(getSimilarAnimes("Naruto", ANIME_WEIGHTS_PATH, PROCESSED_ANIME_DF, PROCESSED_SYNOPSIS_DF,                          ANIMEID_2_ENCODEDANIMEID_MAPPING, ENCODEDANIMEID_2_ANIMEID_MAPPING, k=10, return_dist=True))
    
    # Test Similar_users
    simiusers = similar_users(8881, USER_WEIGHTS_PATH, USERID_2_ENCODEDUSERID_MAPPING, ENCODEDUSERID_2_USERID_MAPPING, k=10)

    # Test getFavGenre
    # print(getFavGenre(frame=getAnimeFrame("Naruto", PROCESSED_ANIME_DF)))

    # Test Getuserpref
    # print(get_user_preferences(16999, PROCESSED_RATING_DF, PROCESSED_ANIME_DF, top_percentile=30))
    user_pref_tmp = get_user_preferences(16999, PROCESSED_RATING_DF, PROCESSED_ANIME_DF, top_percentile=30)

    #test getuserrecommendations
    # print(get_user_recommendations(simiusers, user_pref_tmp, PROCESSED_ANIME_DF, PROCESSED_SYNOPSIS_DF, PROCESSED_RATING_DF, top_percentile=10, n=10))

    # Test hybrid_rec_sys
    TMP_USER_ID = 7881
    result = hybrid_rec_sys(
        TMP_USER_ID, 
        PROCESSED_ANIME_DF, 
        PROCESSED_RATING_DF, 
        PROCESSED_SYNOPSIS_DF, 
        USER_WEIGHTS_PATH, 
        ANIME_WEIGHTS_PATH, 
        USERID_2_ENCODEDUSERID_MAPPING, 
        ENCODEDUSERID_2_USERID_MAPPING, 
        ANIMEID_2_ENCODEDANIMEID_MAPPING, 
        ENCODEDANIMEID_2_ANIMEID_MAPPING,
        top_percentile=30, 
        user_wts=0.6, 
        content_wts=0.4, 
        num_rec2return=10, 
        num_similar_animes2rec=10,
        debug=False  # Set to True to see debug messages
    )
    print("Recommendations:", result)