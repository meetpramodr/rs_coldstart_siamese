import pandas as pd
import numpy as np
import pickle
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from .utils import *

def getimdbToPlots(df):
    '''
    return a dictionary of imdbid to plots
    '''
    unique_movie_plots = df[['imdb_id','plot_synopsis']].drop_duplicates()
    return dict(zip(unique_movie_plots['imdb_id'].values,unique_movie_plots['plot_synopsis'].values))

def trainW2V(imdbToPlots,sg,size,min_count):
    '''
    train a w2v model based on the parameters and return the model
    '''
    plot_list_tokens = []
    for plot in list(imdbToPlots.values()):
        plot_list_tokens.append(word_tokenize(plot.lower()))
    model = Word2Vec(plot_list_tokens, min_count=min_count,size= size,sg=sg)
    return model

def get_sentence_vectors(words, model, vector_len):
    '''
    use elementwise averaging to of word vectors and return sentence level vectors
    '''
    featureVec = np.zeros((vector_len,), dtype="float32")
    nwords = 0
    for word in words:
        if word in model.wv.vocab:
            nwords += 1
            featureVec = np.add(featureVec, model[word])
    if (nwords>0):
        featureVec = np.divide(featureVec, nwords)  
    return featureVec 

if __name__ == "__main__":
    #Fetch the synopsis dataset
    mpst = pd.read_csv("../data/synopsis/mpst_full_data.csv")

    #Fetch the ratings dataset
    ratings = read_ratings_data()

    #Get the intersection between the synopsis and ratings dataset for overalpping movies
    intersection_df = pd.merge(left=ratings,right=mpst[['imdb_id','plot_synopsis']],on='imdb_id',how='inner')
    imdbToPlots = getimdbToPlots(intersection_df)

    #Train w2v model
    model = trainW2V(imdbToPlots,0,100)

    #Get synopsis2vec
    imdbToW2VFeatures = {}
    for k,v in imdbToPlots.items():
        imdbToW2VFeatures[k] = get_sentence_vectors(word_tokenize(v.lower()),model,100)
    with open('../data/synopsis/synopsis_imdbToFeatures.pkl','wb') as fp:
        pickle.dump(imdbToW2VFeatures,fp)

    

