import pandas as pd
import numpy as np
import pickle
from surprise import SVD
from surprise import Dataset
from surprise import Reader

def get_warm_cold_movies(unique_imdbIds,cold_size,seed,path_to_save_cold):
    np.random.seed(seed)
    cold_imdbs = np.random.choice(unique_imdbIds,size=cold_size,replace=False).tolist()
    warm_imdbs = [i for i in unique_imdbIds if i not in cold_imdbs]

    with open(path_to_save_cold,'wb') as fp:
        pickle.dump(cold_imdbs,fp)
    return (warm_imdbs,cold_imdbs)

def fit_collab(training_df,factors,userkey,moviekey,rating):
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(training_df[[userkey, moviekey, rating]],reader)
    dataset = data.build_full_trainset()
    algo = SVD(n_factors=factors)
    algo.fit(dataset)

    unique_movieid = data.df[moviekey].unique().tolist()
    iidToMovieid  = {dataset.to_inner_iid(i):i for i in unique_movieid}
    iidToCollabFeatures = {iidToMovieid[i]:algo.qi[n] for n,i in enumerate(list(iidToMovieid.keys()))}
    return iidToCollabFeatures

def read_ratings_data():
    '''
    Method to read the ratings dataset and links table from ml-20m
    Creates imdb id in the format that is used to lookup with other datasets
    '''
    ratings = pd.read_csv('../data/ml_20m/ratings.csv')
    links = pd.read_csv('../data/ml_20m/links.csv')
    links['imdb_id'] = 'tt' + links['imdbId'].astype(str).str.zfill(7)
    return pd.merge(left=ratings,right=links[['movieId','imdb_id']],on='movieId',how='inner')

def filter_min_users_movies_from_ratings(ratings_df,min_users=50,min_movies=50):
    '''
    Method to subset the dataset based on users having minimum ratings and movies having minimum ratings
    '''
    df_with_minusers = ratings_df.groupby("userId").filter(lambda x: len(x) > min_users)
    df_with_minmovies = df_with_minusers.groupby("movieId").filter(lambda x: len(x) > min_movies)
    return df_with_minmovies

def get_imdb_movieid_dicts():
    '''
    Method to get imdb id and movie id dictionaries for lookups
    '''
    df = read_ratings_data()
    imdbToMovieId = dict(zip(df['imdb_id'].values,df['movieId'].values))
    movieIdToImdb = dict(zip(df['movieId'].values,df['imdb_id'].values))
    return (imdbToMovieId,movieIdToImdb)

imdbToMovieId,movieIdToImdb = get_imdb_movieid_dicts()