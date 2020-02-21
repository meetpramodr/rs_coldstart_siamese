import pandas as pd
import pickle
import numpy as np
from scripts.utils import *

if __name__ == "__main__":
    seed = 11
    cold_size = 277
    factors = 100

    #Fetch the ratings dataset
    ratings = read_ratings_data()

    #Filter ratings dataset to contain users who have givig min ratings and movies with min ratings
    ratings_filtered = filter_min_users_movies_from_ratings(ratings,min_users=50,min_movies=50)

    #Get the AlexNet features dictionary to subset the ratings
    trailer = pd.concat([pd.read_csv('../data/trailer/merged_alexnet_avg1.csv',encoding='latin-1'),
                    pd.read_csv('../data/trailer/merged_alexnet_avg2.csv',encoding='latin-1'),
                    pd.read_csv('../data/trailer/merged_alexnet_avg3.csv',encoding='latin-1')])

    #Get the intersection of the ratings and trailer features
    trailer_ids = set(trailer['movieId'].values.tolist())
    rating_ids = set(ratings_filtered['movieId'].values.tolist())
    common_ids = trailer_ids.intersection(rating_ids)
    intersection_df = trailer[trailer['movieId'].isin(common_ids)]

    unique_imdbIds = [movieIdToImdb[i] for i in common_ids]
    warm_imdbs,cold_imdbs = get_warm_cold_movies(unique_imdbIds,cold_size,seed,'../data/trailer/cold_imdb.pkl')

    final_training_data = intersection_df.loc[intersection_df['movieId'].isin([imdbToMovieId[j] for j in warm_imdbs])]

    with open('../data/trailer/trailer_imdbToCollabFeatures.pkl','wb') as fp:
        pickle.dump(fit_collab(final_training_data,factors,userkey='userId',moviekey='movieId',rating='rating'),fp)