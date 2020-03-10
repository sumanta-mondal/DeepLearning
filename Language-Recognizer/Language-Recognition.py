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
from sklearn import model_selection
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split #divide into train and test set
from sklearn.metrics import confusion_matrix #we import this package from sklearn and output it

# How these init variabes contribute
# max_features used in tokenizer steps
# maxlen is uaed in padding steps
# embedding dimension is embedding output which is required to define the Sequential model 
max_features=5000 #we set maximum number of words to 5000
maxlen=400 # maximum sequence length to 400
embedding_dim = 50 #this is the final dimension of the embedding space.

#Load a file content
def LoadFile(filename):
    train = pd.read_csv(filename)
    return train
#to encode the target variable( that means label named 'Language' here) from text to number
#this is done using LabelEncoding not OnehotEncoding
def EncodeTargetVariable(train_df):
    Y = train_df['language']
    encoder = LabelEncoder()
    encoder.fit(Y)
    Y = encoder.transform(Y)
    Y = tf.keras.utils.to_categorical(
    Y,
    num_classes=4 #equals to the number of languages
   
    )
    return Y

#put everything to lowercase and then replace undesired characters
def ToLowercase(train):
    train['sentence_lower'] = train["sentence"].str.lower()
    train['sentence_no_punctuation'] = train['sentence_lower'].str.replace('[^\w\s]','')
    return train

#remove stopwords and then fill empy cells with "fillna" word
def RemoveStopwords(train):
    #train['sentence_no_punctuation'] = train['sentence_no_punctuation'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    train["sentence_no_punctuation"] = train["sentence_no_punctuation"].fillna("fillna")
    return train
    
# generate tokens 
def GetTokens():
    tok = tf.keras.preprocessing.text.Tokenizer(num_words=max_features) #tokenizer step
    return tok

# fit tokens to cleaned text and create sequences 
def FitTokens(tok, tf_train):
    tok.fit_on_texts(list(tf_train['sentence_no_punctuation'])) #fit to cleaned text
    tf_train=tok.texts_to_sequences(list(tf_train['sentence_no_punctuation'])) #this is how we create sequences
    return tf_train

# execute pad step 
def ExecutePadding(tf_train):
    tf_train=tf.keras.preprocessing.sequence.pad_sequences(tf_train, maxlen=maxlen)
    tf_train = pd.DataFrame(tf_train)
    return tf_train

def CreateSequenceandPadding(new_text):
    test_text = tok.texts_to_sequences(new_text) #this is how we create sequences
    test_text = tf.keras.preprocessing.sequence.pad_sequences(test_text, maxlen=maxlen) #let's execute pad step
    return test_text
# method contributes in model definition
# this method calculates Embedding input size which is used to define Sequential model fo Keras 
def CalculateEmbeddingSpace(tok):
    vocab_size = len(tok.word_index) + 1
    return vocab_size

def DivideintoTrainandTestSet(train_df, Y):
    X_train, X_test, y_train, y_test = train_test_split(train_df, Y, test_size=0.1, random_state=42)
    return X_train, X_test, y_train, y_test

# A sequential model for for multiclass classifiction problems
# input params:
# vocab_size contributes in  embedding input size one of the importent parameters for defining a sequential model
# embedding_dim contributes in embedding output which is again one of the importent parameters for defining a sequential model
# maxlen will deine the maximum length of an input sequence which is again one of the importent parameters for defining a sequential model
# no_of_probable_outpts will say the dense layer how many possibilities are there for this classification problem - as we are concentrating on 4 languages 
#   we have 4 possibilities arround
def CreateMultiClassClassificationModel(vocab_size, embedding_dim, maxlen, no_of_probable_outpts):
    model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(input_dim=vocab_size, #embedding input
                           output_dim=embedding_dim,#embedding output
                           input_length=maxlen), #maximum length of an input sequence
  tf.keras.layers.Flatten(), #flatten layer

  tf.keras.layers.Dense(no_of_probable_outpts, activation=tf.nn.softmax) #ouput layer a Dense layer with 4 probabilities
  #we also define our final activation function which is the softmax function typical for multiclass
  #classifiction problems

])
    return model

# define loss fnction 
def DefineLoosFunction(model):
    model.compile(optimizer='adam',
              loss='categorical_crossentropy', #we recommend this loss function you
              metrics=['accuracy'])
    return model

# fit data into model
def FitDataintoModel(model, X_train, y_train, no_of_epochs):
    model.fit(np.array(X_train), np.array(y_train), epochs=no_of_epochs) #let's fit the model
    return model

# we use the test to evaluate our model
def EvaluateModel(model, X_test, y_test):
    model.evaluate(np.array(X_test), np.array(y_test)) 
    return model

# get predictions
def GetPredictions(model, test_text):
    predictions = model.predict(test_text)
    print(predictions.argmax())
    print(predictions) #spanish you can get confused with italian which makes sense since they are more similar languages
# -------------------------------- driver code ----------------------------------------------------------
#PLEASE DOWNLOAD THE FILE HERE: https://www.kaggle.com/aashita/nyt-comments
train = LoadFile('train_languages.csv')
Y = EncodeTargetVariable(train)
train = ToLowercase(train)
train['sentence_no_punctuation'].head()  #just for testing purpose
train = RemoveStopwords(train)
train['sentence_no_punctuation'].head()    #just for testing purpose
# We first assign our current data frame to another to keep track of our work 
tf_train = train
tok = GetTokens()#tokenizer step
tf_train = FitTokens(tok, tf_train)
print(len(tf_train[1]))                    #just for testing purpose

embedding_input = CalculateEmbeddingSpace(tok)
tf_train = ExecutePadding(tf_train)
print(len(tf_train[1]))                    #just for testing purpose
tf_train[1] #this is how our sentece looks like after the pad step we don't have anymore 16 words but 100 (equivalent to maxlen)
X_train, X_test, y_train, y_test = DivideintoTrainandTestSet(tf_train, Y)
# Create the model for multiclass classification 
model = CreateMultiClassClassificationModel(embedding_input, 50, maxlen, 4)
model = DefineLoosFunction(model)
model.summary() #here we show the architecture 
model = FitDataintoModel(model, X_train, y_train, 3)
model = EvaluateModel(model, X_test, y_test)
predictions = model.predict(X_test) #here we make predictions
cm = confusion_matrix(predictions.argmax(axis=1), y_test.argmax(axis=1))#we generate the confusion matrix
cm
#new_text = ["tensorflow is a great tool you can find a lot of tutorials from packt"]
#new_text = ["tensorflow est un excellent outil vous pouvez trouver beaucoup de tutoriels de packt"]
#new_text = ["tensorflow è un ottimo strumento puoi trovare molti tutorial di packt"]
new_text = ["tensorflow es una gran herramienta puedes encontrar muchos tutoriales de packt"]
test_text = CreateSequenceandPadding(new_text)
np.set_printoptions(suppress=True)
GetPredictions(model, test_text)
