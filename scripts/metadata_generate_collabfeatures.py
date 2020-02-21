import pandas as pd
import pickle
import numpy as np
from scripts.utils import *

if __name__ == "__main__":
    seed = 11
    cold_size = 700
    factors = 100

    #Fetch the ratings dataset
    ratings = read_ratings_data()

    #Filter ratings dataset to contain users who have givig min ratings and movies with min ratings
    ratings_filtered = filter_min_users_movies_from_ratings(ratings,min_users=50,min_movies=50)

    #Get the AlexNet features dictionary to subset the ratings
    metadata_selected_imdbids =pd.read_csv('../data/metadata/good_movies_meta.csv',names=['imdb_id'])

    #Get the intersection of the ratings and trailer features
    metadata_ids = set(metadata_selected_imdbids['imdb_id'].values.tolist())
    rating_ids = set(ratings_filtered['imdb_id'].values.tolist())
    common_ids = metadata_ids.intersection(rating_ids)
    intersection_df = ratings_filtered[ratings_filtered['imdb_id'].isin(common_ids)]

    warm_imdbs,cold_imdbs = get_warm_cold_movies(common_ids,cold_size,seed,'../data/metadata/cold_imdb.pkl')

    final_training_data = intersection_df.loc[intersection_df['movieId'].isin([imdbToMovieId[j] for j in warm_imdbs])]

    with open('../data/metadata/metadata_imdbToCollabFeatures.pkl','wb') as fp:
        pickle.dump(fit_collab(final_training_data,factors,userkey='userId',moviekey='imdb_id',rating='rating'),fp)