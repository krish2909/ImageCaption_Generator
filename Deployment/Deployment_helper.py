#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
import re
import nltk
from nltk.corpus import stopwords
import string
import json
from time import time
import pickle
import cv2
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.models import Model, load_model
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Input, Dense, Dropout, Embedding, LSTM
from keras.layers.merge import add


# In[2]:


model = load_model('../model_weights/model_9.h5')


# In[3]:


model_temp = ResNet50(weights="imagenet",input_shape=(224,224,3))


# In[4]:


# creating a new model, by removing the last output layer
model_resnet = Model(model_temp.input,model_temp.layers[-2].output)


# In[5]:


from keras.preprocessing import image
def preprocess_img(img):
    img = image.load_img(img,target_size=(224,224))
    img = image.img_to_array(img)
    # As the input to a ResNet is a 4D tensor.Therefore expanding the size (1,224,224,3)
    img = np.expand_dims(img,axis=0) 
     #Normalisation
    img = preprocess_input(img)    
    return img


# In[23]:


def encode_image(img):
    img = preprocess_img(img)
    feature_vector = model_resnet.predict(img)
    feature_vector = feature_vector.reshape((1,2048))
    return feature_vector


# In[24]:


with open('../saved/index_to_word.pkl', 'rb') as f:
    index_to_word = pickle.load(f)
    
with open('../saved/word_to_index.pkl', 'rb') as f:
    word_to_index = pickle.load(f)


# In[25]:


# Predicting output
def predict_caption(photo):
    in_text = "startseq"
    max_len = 35
    for i in range(max_len):
        sequence = [word_to_index[w] for w in in_text.split() if w in word_to_index]
        sequence = pad_sequences([sequence],maxlen = max_len,padding='post')
        ypred = model.predict([photo,sequence])
        ypred = ypred.argmax() #greedy sampling of the words.
        word = index_to_word[ypred]
        in_text += ' ' + word
        
        if word == "endseq":
            break
    final_caption = in_text.split()[1:-1] # Removing startseq and endseq from the captions
    final_caption = ' '.join(final_caption)
    return final_caption


# In[26]:


def caption_this_image(img):
    enc = encode_image(img)
    caption = predict_caption(enc)
    return caption






