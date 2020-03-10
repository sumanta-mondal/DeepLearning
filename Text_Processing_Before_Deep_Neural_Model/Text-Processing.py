# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 08:19:12 2020

@author: smondal
"""
import tensorflow as tf
import numpy as np # allows array opperation
import pandas as pd # will use it to read and manipulate files and column content

from nltk.corpus import stopwords # provides list of english stopwords
stop = stopwords.words('english')

# How these init variabes contribute
# max_features used in tokenizer steps
# maxlen is uaed in padding steps
max_features=5000 #we set maximum number of words to 5000
maxlen=100 #and maximum sequence length to 100

#Load a file content
def LoadFile(filename):
    train = pd.read_csv(filename)
    return train

#put everything to lowercase and then replace undesired characters
def ToLowercase(train):
    train['commentBody_lower'] = train["commentBody"].str.lower()
    train['commentBody_no_punctiation'] = train['commentBody_lower'].str.replace('[^\w\s]','')
    return train

#remove stopwords and then fill empy cells with "fillna" word
def RemoveStopwords(train):
    train['commentBody_no_stopwords'] = train['commentBody_no_punctiation'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    train["commentBody_no_stopwords"] = train["commentBody_no_stopwords"].fillna("fillna")
    return train
    
# generate tokens 
def GetTokens():
    tok = tf.keras.preprocessing.text.Tokenizer(num_words=max_features) #tokenizer step
    return tok

# fit tokens to cleaned text and create sequences 
def FitTokens(tok, tf_train):
    tok.fit_on_texts(list(tf_train['commentBody_no_stopwords'])) #fit to cleaned text
    tf_train=tok.texts_to_sequences(list(tf_train['commentBody_no_stopwords'])) #this is how we create sequences
    return tf_train

# execute pad step 
def ExecutePadding(tf_train):
    tf_train=tf.keras.preprocessing.sequence.pad_sequences(tf_train, maxlen=maxlen)
    tf_train = pd.DataFrame(tf_train)
    return tf_train
# -------------------------------- driver code ---------------------------------------------------
#PLEASE DOWNLOAD THE FILE HERE: https://www.kaggle.com/aashita/nyt-comments
train = LoadFile('CommentsApril2017.csv')
train = ToLowercase(train)
train['commentBody_no_punctiation'].head()  #just for testing purpose
train = RemoveStopwords(train)
train['commentBody_no_stopwords'].head()    #just for testing purpose
# We first assign our current data frame to another to keep track of our work 
tf_train = train
tok = GetTokens()#tokenizer step
tf_train = FitTokens(tok, tf_train)
print(len(tf_train[1]))                    #just for testing purpose
tf_train = ExecutePadding(tf_train)
print(len(tf_train[1]))                    #just for testing purpose
tf_train[1] #this is how our sentece looks like after the pad step we don't have anymore 16 words but 100 (equivalent to maxlen)