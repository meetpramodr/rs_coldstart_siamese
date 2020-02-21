#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path = ["/mnt/home/my_lib/"] + sys.path

import warnings
warnings.filterwarnings("ignore")

get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import pickle

from tqdm.notebook import tqdm

import scipy

import seaborn as sns
sns.set()


# ## Read Data

# In[4]:


imdbidToCollabFeatures_meta = pickle.load(open('./redo/imdbToCollabFeatures_synopsisdata3.pkl','rb'))

movies = []
collab_feats = []
for k,v in imdbidToCollabFeatures_meta.items():
    movies.append(k)
    collab_feats.append(v)
collab_df = pd.DataFrame()
collab_df['movies'] = movies
collab_df['feats'] = collab_feats
imdbidToCollabFeatures_meta = ''
movies = []
collab_feats = []


# In[5]:


synopsis_feats = pd.read_pickle('./redo/imdbToW2VFeatures_synopsisdata3.pkl')


# In[6]:


l = []
for k,v in synopsis_feats.items():
    l.append([k,v])
synopsis_feats_df = pd.DataFrame(l,columns = ['movie_id','embeddings'])


# In[7]:


synopsis_feats_df.shape


# In[8]:


# feat_cols = list(movie_features.columns)[2:]
movie_syn_df = pd.DataFrame()
movie_syn_df['movies'] = synopsis_feats_df['movie_id']
movie_syn_df['feats'] = synopsis_feats_df['embeddings']


# In[9]:


feats_float = []
for ind,row in movie_syn_df.iterrows():
    feats_float.append(np.array(row['feats']))
movie_syn_df['feats_float'] = feats_float


# In[10]:


from sklearn.preprocessing import StandardScaler


# In[11]:


sc = StandardScaler(with_mean=True,with_std=True)
movie_syn_df['scaled_feats'] = list(sc.fit_transform(feats_float))
pickle.dump(sc,open('scaler_syn_goog.pkl','wb'))


# In[12]:


movie_syn_dict = {}
for ind,row in movie_syn_df.iterrows():
    movie_syn_dict[row['movies']] = row['scaled_feats']


# In[13]:


pickle.dump(movie_syn_dict,open('movie_syn_dict.pkl','wb'))


# In[14]:


len(set(movie_syn_df['movies']).intersection(collab_df['movies']))


# In[15]:


cold_movies = pickle.load(open('./redo/cold_imdb_3.pkl','rb'))
common_cold_movies = list(set(cold_movies).intersection(set(movie_syn_df['movies'])))
len(common_cold_movies)


# In[16]:


featsize = movie_syn_df['feats_float'][0].shape[0]
featsize


# ## Network

# In[18]:


import keras

from keras.models import Sequential
import time
from keras.optimizers import Adam
from keras.layers import Activation, Input, concatenate, Dropout
from keras.models import Model
import seaborn as sns
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from sklearn.preprocessing import LabelBinarizer
from keras.optimizers import *
from keras.engine.topology import Layer
from keras import backend as K
from keras.regularizers import l2
from keras.utils import plot_model
from keras import regularizers


# In[19]:


from keras.layers import Dot,Reshape


# In[20]:


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)


# In[21]:


def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


# In[22]:


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def cosine_distance(vecs):
    #I'm not sure about this function too
    y_true, y_pred = vecs
    y_true = K.l2_normalize(y_true, axis=-1)
    y_pred = K.l2_normalize(y_pred, axis=-1)
    return K.mean(K.sum((y_true * y_pred), axis=-1))

def cosine_dist_output_shape(shapes):
    shape1, shape2 = shapes
    print((shape1[0], 1))
    return (shape1[0], 1)


# In[24]:


input_shape = (featsize,)
left_input = Input(input_shape)
right_input = Input(input_shape)
model = Sequential()
reg_param = 0#.0000000001
model.add(Dense(256, activation='tanh', input_shape=input_shape,kernel_regularizer=regularizers.l2(reg_param),
                activity_regularizer=regularizers.l1(reg_param),kernel_initializer='random_normal',
                bias_initializer='zeros'))
model.add(Dropout(0.05))

model.add(Dense(128, activation='tanh',kernel_regularizer=regularizers.l2(reg_param),
                activity_regularizer=regularizers.l1(reg_param),kernel_initializer='random_normal',
                bias_initializer='zeros'))
model.add(Dropout(0.05))

model.add(Dense(64, activation='tanh',kernel_regularizer=regularizers.l2(reg_param),
                activity_regularizer=regularizers.l1(reg_param),kernel_initializer='random_normal',
                bias_initializer='zeros'))
model.add(Dropout(0.05))

encoded_l = model(left_input)
encoded_r = model(right_input)

######### L1 distance ################
L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
L1_distance = L1_layer([encoded_l, encoded_r])
L1_distance_tan = Dense(1,activation='tanh')(L1_distance)
prediction = Dense(1,activation='linear')(L1_distance_tan)
neg_layer = Lambda(lambda tensor:-1*tensor)
neg_prediction = Dense(1)(neg_layer(prediction))
siamese_net = Model(inputs=[left_input,right_input],outputs=neg_prediction)


optimizer = Adam(lr = 0.0005)
siamese_net.compile(loss='mse',optimizer=optimizer)

siamese_net.summary()


# ## Training data make

# In[28]:


from scipy.spatial.distance import pdist,squareform,cdist

sc = StandardScaler(with_mean=True, with_std=True)
all_collab_feats = sc.fit_transform(np.array([np.array(x) for x in collab_df['feats']]))


# In[29]:


len(all_collab_feats)


# In[30]:


all_dists = 1 - squareform(pdist(all_collab_feats,metric='cosine'))
all_dists.mean(),all_dists.std()


# In[31]:


all_dists.min(),all_dists.max()


# In[32]:


training_dists = []
all_movies = list(collab_df['movies'])
i=0
for movie1 in tqdm(all_movies):
    for j,movie2 in enumerate(all_movies):
        if j<i or np.random.random()>1.0:
            continue
        else:
            training_dists.append(((movie1,movie2),all_dists[i,j]))
    i+=1


# In[33]:


training_inputs1 = []
training_inputs2 = []
training_outputs = []
for item in tqdm(training_dists):
    training_inputs1.append(movie_syn_dict[item[0][0]])
    training_inputs2.append(movie_syn_dict[item[0][1]])
    training_outputs.append(item[1])


# In[35]:


training_df = pd.DataFrame()
training_df['X1'] = training_inputs1
training_df['X2'] = training_inputs2
training_df['Y'] = training_outputs


# ## Test-Train split

# In[42]:


from sklearn.model_selection import train_test_split


# In[43]:


train_df,test_df = train_test_split(training_df, test_size=0.2, random_state=2)


# In[44]:


# training_inputs = ''
# training_outputs = ''


# ## Training

# In[45]:


step = 15000000


# In[46]:


losses = []
train_df = training_df
for epoch in tqdm(range(50),desc = 'Epochs'):
    loss = []
    for i in tqdm(range(0,train_df.shape[0],step),desc = 'steps'):
        X_train1 = np.array(list(train_df['X1'].values)[i:i+step])
        X_train2 = np.array(list(train_df['X2'].values)[i:i+step])
        y_train = list(train_df['Y'])[i:i+step]
        siamese_net.fit([X_train1,X_train2],y_train,batch_size=2048,validation_split=0.2,epochs=1,)
        loss.append(siamese_net.history.history)
    losses.append(loss)


# In[47]:


plt.figure(figsize=(12,8))
plt.plot([np.mean(np.array([y['loss'] for y in x])) for x in losses],'-o',label='Training')
plt.ylabel('Loss')

plt.plot([np.mean(np.array([y['val_loss'] for y in x])) for x in losses],'-o',label='Validation')
plt.xlabel('iter')

plt.title('Synopsis Google Vec Training Validation redo')
plt.legend()
plt.show()


# In[53]:


siamese_net.save('siamese_net_syn_goog_cosine_3_layer_relutanh_redo3.model')


# ## Cold movies

# In[48]:


# movie_syn_dict = pickle.load(open('movie_syn_dict.pkl','rb'))

total_warm_movies = collab_df['movies'].shape[0]
all_warm_movies = list(collab_df['movies'])

closest_movies_list = []
input1 = np.array([movie_syn_dict[warm_movie] for warm_movie in all_warm_movies])
for cold_movie in tqdm(common_cold_movies):
    input2 = np.repeat(movie_syn_dict[cold_movie].reshape(1,featsize),total_warm_movies,axis =0)
    preds = siamese_net.predict([input1,input2])
    closest_movies = [all_warm_movies[x] for x in list(np.argsort(-1*preds[:,0]))[:100]]#
    closest_movies_list.append(closest_movies)

closest_movies_df = pd.DataFrame()
closest_movies_df['cold_movie'] = common_cold_movies
closest_movies_df['closest_movies'] = closest_movies_list

movieIdToImdbId = pd.read_pickle('movieIdToImdbid.pkl')

movies = pd.read_csv('movies.csv')
movies['imdbId'] = [movieIdToImdbId[x] for x in movies['movieId']]
imdbIdToMovie = {}
for k,row in movies.iterrows():
    imdbIdToMovie[row['imdbId']] = row['title']

cold_movie_names = []
closest_movie_names = []
for k,row in closest_movies_df.iterrows():
    cold_movie_names.append(imdbIdToMovie[row['cold_movie']])
    closest_movieIds = row['closest_movies']
    closest_movies = [imdbIdToMovie[x] for x in closest_movieIds]
    closest_movie_names.append(closest_movies)
closest_movies_df['cold_movie_names'] = cold_movie_names
closest_movies_df['closest_movie_names'] = closest_movie_names

closest_movies_df.to_csv('siamese_pred_top100_googsyn_redo3.csv',index = False)


# In[49]:


min(preds[:,0]),max(preds[:,0]), np.mean(preds[:,0])
# min(preds),max(preds), np.mean(preds)


# In[50]:


closest_movies_df[['cold_movie_names','closest_movie_names']].head(60)


# In[51]:


from scipy.spatial.distance import cdist

movie_meta_dict = movie_syn_dict
closest_movies_list = []
input1 = np.array([movie_meta_dict[warm_movie] for warm_movie in all_warm_movies])
for cold_movie in tqdm(common_cold_movies):
    input2 = movie_meta_dict[cold_movie]
    preds = 1 - cdist(input2.reshape((1,featsize)),input1,metric = 'cosine')
    closest_movies = [all_warm_movies[x] for x in list(np.argsort(-1*np.array([a for a in list(preds[0])]))[:100])]
    closest_movies_list.append(closest_movies)

closest_movies_df = pd.DataFrame()
closest_movies_df['cold_movie'] = common_cold_movies
closest_movies_df['closest_movies'] = closest_movies_list

movies = pd.read_csv('movies.csv')
movies['imdbId'] = [movieIdToImdbId[x] for x in movies['movieId']]
imdbIdToMovie = {}
for k,row in movies.iterrows():
    imdbIdToMovie[row['imdbId']] = row['title']

cold_movie_names = []
closest_movie_names = []
for k,row in closest_movies_df.iterrows():
    cold_movie_names.append(imdbIdToMovie[row['cold_movie']])
    closest_movieIds = row['closest_movies']
    closest_movies = [imdbIdToMovie[x] for x in closest_movieIds]
    closest_movie_names.append(closest_movies)
closest_movies_df['cold_movie_names'] = cold_movie_names
closest_movies_df['closest_movie_names'] = closest_movie_names

closest_movies_df.to_csv('w2vfeats_top100_googsyn_redo3.csv',index = False)


# In[54]:


pickle.dump(losses,open('losses_synopsis_redo3.pkl','wb'))


# In[ ]:




