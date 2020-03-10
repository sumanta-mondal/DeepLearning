
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 00:18:31 2020

@author: smondal
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf 
import numpy as np
from nltk.corpus import stopwords #provides list of english stopwords
stop = stopwords.words('english')

#PRINT VERSION!!!
tf.__version__

#Load a file content
def LoadFile(filename):
    train = pd.read_csv(filename)
    return train
#some refined task on train data
def RefineTrain(train):
    train = train[['Summary','Text']]
    train['text_length'] = train['Text'].str.count(' ')
    train['text_length'].describe()
    train['summary_length'] = train['Summary'].str.count(' ')
    train['summary_length'].describe()
    train = train.loc[train['summary_length']<8]
    train = train.loc[train['text_length']<30]
    return train
#put everything to lowercase and then replace undesired characters
def ToLowercase(train, key1, key2, key3, key4, start, end):
    train[key2] = train[key1].str.lower()
    if start.strip():
        #print("it's not an empty or blank string")
        train[key4] =  start + ' ' +train[key3].str.replace('[^\w\s]','')+ ' ' + end
    else:
        train[key4] = train[key3].str.replace('[^\w\s]','')
        #print("it's an empty or blank string")
    return train

# generate tokens 
def GetTokens(max_features, filters):
    if filters.strip():
        print("it's not an empty or blank filters")
        tok = tf.keras.preprocessing.text.Tokenizer(num_words=max_features, filters = '*') #tokenizer step
    else:
        tok = tf.keras.preprocessing.text.Tokenizer(num_words=max_features) #tokenizer step   
    return tok

# fit tokens to cleaned text and create sequences 
def FitTokens(tok, tf_train, key1, key2):
    tok.fit_on_texts(list(tf_train[key1].astype(str))) #fit to cleaned text
    tf_train=tok.texts_to_sequences(list(tf_train[key2].astype(str))) #this is how we create sequences
    return tf_train
    
#####################Driver Code################################################################
train = LoadFile('reviews.csv')
train.head()
train = RefineTrain(train)
print(train.shape)
print(train.head())
train = ToLowercase(train, 'Text', 'text_lower', 'text_lower', 'text_no_punctuation','','')
train = ToLowercase(train, 'Summary', 'summary_lower', 'summary_lower', 'summary_no_punctuation','_start_','_end_')
# NOTICE THAT WE ADD "_start_" and "_end_" EXACTLY AT THE BEGINNING AND THE END OF EACH SENTENCE TO HAVE SOME KIND OF'DELIMITERS' 
#THAT WILL TELL OUR DECODER TO START AND FINISH. BECAUSE WE DON'T HAVE GENERAL SIGNALS OF START AND FINISH IN NATURAL LANGUAGE. BASICALLY '_end_' REFLECTS THE POINT IN WHICH OUR OUTPUT SENTENCE IS MORE LIKELY TO END.
#train['text_lower'] = train['Text'].str.lower()
#train['text_no_punctuation'] = train['text_lower'].str.replace('[^\w\s]','')
#train['english_no_stopwords'] = train['english_no_punctuation'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
#train["english_no_stopwords"] = train["english_no_stopwords"].fillna("fillna")
#train["english_no_stopwords"] = train["english_no_stopwords"] 
#train['summary_lower'] = train["Summary"].str.lower()
#train['summary_no_punctuation'] =  '_start_' + ' ' +train['summary_lower'].str.replace('[^\w\s]','')+ ' ' +'_end_'

max_features1 = 5000
maxlen1 = 30

max_features2 = 5000
maxlen2 = 8

#tok1 = tf.keras.preprocessing.text.Tokenizer(num_words=max_features1) 
tok1 = GetTokens(max_features1, '')
#tok1.fit_on_texts(list(train['text_no_punctuation'].astype(str))) #fit to cleaned text
#tf_train_text =tok1.texts_to_sequences(list(train['text_no_punctuation'].astype(str)))
tf_train_text = FitTokens(tok1, train, 'text_no_punctuation', 'text_no_punctuation')
tf_train_text =tf.keras.preprocessing.sequence.pad_sequences(tf_train_text, maxlen=maxlen1) #let's execute pad step 
#the processing has to be done for both 
#two different tokenizers
#tok2 = tf.keras.preprocessing.text.Tokenizer(num_words=max_features2, filters = '*') 
tok2 = GetTokens(max_features2, '*')
#tok2.fit_on_texts(list(train['summary_no_punctuation'].astype(str))) #fit to cleaned text
#tf_train_summary = tok2.texts_to_sequences(list(train['summary_no_punctuation'].astype(str)))
tf_train_summary = FitTokens(tok2, train, 'summary_no_punctuation', 'summary_no_punctuation')
tf_train_summary = tf.keras.preprocessing.sequence.pad_sequences(tf_train_summary, maxlen=maxlen2, padding ='post')


vectorized_summary = tf_train_summary
# For Decoder Input, you don't need the last word as that is only for prediction
# when we are training using Teacher Forcing.
decoder_input_data = vectorized_summary[:, :-1]

# Decoder Target Data Is Ahead By 1 Time Step From Decoder Input Data (Teacher Forcing)
decoder_target_data = vectorized_summary[:, 1:]

print(f'Shape of decoder input: {decoder_input_data.shape}')
print(f'Shape of decoder target: {decoder_target_data.shape}')

vectorized_text = tf_train_text
# Encoder input is simply the body of the issue text
encoder_input_data = vectorized_text
doc_length = encoder_input_data.shape[1]
print(f'Shape of encoder input: {encoder_input_data.shape}')

vocab_size_encoder = len(tok1.word_index) + 1 #remember vocab size?
vocab_size_decoder = len(tok2.word_index) + 1

#arbitrarly set latent dimension for embedding and hidden units
latent_dim = 50

encoder_inputs = tf.keras.Input(shape=(doc_length,), name='Encoder-Input')

# Word embeding for encoder (English text)
x = tf.keras.layers.Embedding(vocab_size_encoder, latent_dim, name='Body-Word-Embedding', mask_zero=False)(encoder_inputs)


#Batch normalization is used so that the distribution of the inputs 
#to a specific layer doesn't change over time
x = tf.keras.layers.BatchNormalization(name='Encoder-Batchnorm-1')(x)


# We do not need the `encoder_output` just the hidden state.
_, state_h = tf.keras.layers.GRU(latent_dim, return_state=True, name='Encoder-Last-GRU')(x)

# Encapsulate the encoder as a separate entity so we can just 
#  encode without decoding if we want to.
encoder_model = tf.keras.Model(inputs=encoder_inputs, outputs=state_h, name='Encoder-Model')

seq2seq_encoder_out = encoder_model(encoder_inputs)

########################
#### Decoder Model ####
decoder_inputs = tf.keras.Input(shape=(None,), name='Decoder-Input')  # for teacher forcing

# Word Embedding For Decoder (Italian text)
dec_emb = tf.keras.layers.Embedding(vocab_size_decoder, latent_dim, name='Decoder-Word-Embedding', mask_zero=False)(decoder_inputs)
#again batch normalization
dec_bn = tf.keras.layers.BatchNormalization(name='Decoder-Batchnorm-1')(dec_emb)

# Set up the decoder, using `decoder_state_input` as initial state.
decoder_gru = tf.keras.layers.GRU(latent_dim, return_state=True, return_sequences=True, name='Decoder-GRU')
decoder_gru_output, _ = decoder_gru(dec_bn, initial_state=seq2seq_encoder_out) #the decoder "decodes" the encoder output.
x = tf.keras.layers.BatchNormalization(name='Decoder-Batchnorm-2')(decoder_gru_output)

# Dense layer for prediction
decoder_dense = tf.keras.layers.Dense(vocab_size_decoder, activation='softmax', name='Final-Output-Dense')
decoder_outputs = decoder_dense(x)

########################
#### Seq2Seq Model ####
seq2seq_Model = tf.keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

seq2seq_Model.compile(optimizer=tf.keras.optimizers.Nadam(lr=0.001), loss='sparse_categorical_crossentropy')

#from seq2seq_utils import viz_model_architecture
seq2seq_Model.summary()
#viz_model_architecture(seq2seq_Model)

#Train Model

batch_size = 64
epochs = 2

history = seq2seq_Model.fit([encoder_input_data, decoder_input_data], np.expand_dims(decoder_target_data, -1),
          batch_size=batch_size,  epochs=epochs ,  validation_split=0.12) 
seq2seq_Model.save("seq2seq_subsample_1_epochs.h5")

#---------------------------------Code should run on saved model file ----------------------------------------------------------------------------------
seq2seq_Model = tf.keras.models.load_model('seq2seq_subsample_1_epochs.h5')

#test_text = ['apparently they used too much synthetic flavors that it just burns your tongue also theres too much oil  almost made me chok']
test_text = ['this stuff is awesome  for best flavor boil it in water drain the water add spice packet and then add hot water']
#test_text = ['This dataset consists of reviews of fine foods from amazon. The data span a period of more than 10 years, including all ~500,000 reviews up to October 2012. Reviews include product and user information, ratings, and a plain text review. It also includes reviews from all other Amazon categories.']
#seq2seq_Model = tf.keras.models.load_model('seq2seq_subsample_1_epochs.h5')
#seq2seq_Model = tf.keras.models.load_model('seq2seq_full_data_3_epochs.h5')
#max_len_title = 30
# get the encoder's features for the decoder
tok1.fit_on_texts(test_text)
raw_tokenized = tok1.texts_to_sequences(test_text)
raw_tokenized = tf.keras.preprocessing.sequence.pad_sequences(raw_tokenized, maxlen=maxlen1)
body_encoding = encoder_model.predict(raw_tokenized) #predict the encoder state of the new sentence
latent_dim = seq2seq_Model.get_layer('Decoder-Word-Embedding').output_shape[-1]

#remember the get layer methodo for getting the embedding (word clusters)
decoder_inputs = seq2seq_Model.get_layer('Decoder-Input').input 
dec_emb = seq2seq_Model.get_layer('Decoder-Word-Embedding')(decoder_inputs)
dec_bn = seq2seq_Model.get_layer('Decoder-Batchnorm-1')(dec_emb)

gru_inference_state_input = tf.keras.Input(shape=(latent_dim,), name='hidden_state_input')
gru_out, gru_state_out = seq2seq_Model.get_layer('Decoder-GRU')([dec_bn, gru_inference_state_input])

# Reconstruct dense layers
dec_bn2 = seq2seq_Model.get_layer('Decoder-Batchnorm-2')(gru_out)
dense_out = seq2seq_Model.get_layer('Final-Output-Dense')(dec_bn2)

decoder_model = tf.keras.Model([decoder_inputs, gru_inference_state_input],
                          [dense_out, gru_state_out])

# we want to save the encoder's embedding before its updated by decoder
#   because we can use that as an embedding for other tasks.
original_body_encoding = body_encoding

state_value = np.array(tok2.word_index['_start_']).reshape(1, 1)

state_value

decoded_sentence = []
stop_condition = False

vocabulary_inv = dict((v, k) for k, v in tok2.word_index.items())
#vocabulary_inv[0] = "<PAD/>"
#vocabulary_inv[1] = "unknown"

vocabulary_inv

while not stop_condition:
    #print(1)
    preds, st = decoder_model.predict([state_value, body_encoding])

    pred_idx = np.argmax(preds[:, :, 2:]) + 2
    pred_word_str = vocabulary_inv[pred_idx]
    print(pred_word_str)
    if pred_word_str == '_end_' or len(decoded_sentence) >= maxlen2:
        stop_condition = True
        break
    decoded_sentence.append(pred_word_str)

    # update the decoder for the next word
    body_encoding = st
    state_value = np.array(pred_idx).reshape(1, 1)
    #print(state_value)
    
    train.tail()
