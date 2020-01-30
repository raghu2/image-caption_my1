#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import string
from PIL import Image
from time import time
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector, Activation, Flatten, Reshape, concatenate, Dropout, BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import Bidirectional
from keras.layers.merge import add
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras import Input, layers
from keras import optimizers
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import cv2
import os
import glob

#%%

filename = "flickr/Flickr8k_text/Flickr8k.token.txt"
file = open(filename, 'r')
doc = file.read()
file.close()
count = 0
for line in doc.split('\n'):
    count = count + 1
print(count)
print(doc[:400])

#%%

def load_descriptions(doc):
    mapping = dict()
    for line in doc.split('\n'):
        tokens = line.split()
        if len(line) < 2:
            continue
        image_id = tokens[0]
        image_desc = tokens[1:]
        image_desc = ' '.join(image_desc)
        image_id = image_id.split('.')[0]
        if image_id in mapping:
            mapping[image_id].append(image_desc)
        if image_id not in mapping:
            mapping[image_id] = list()
            mapping[image_id].append(image_desc)
    return mapping
descriptions = load_descriptions(doc)
print('loaded=', len(descriptions))

#%%

#print(descriptions)
print(list(descriptions.keys())[:5])

#%%

print(len(descriptions))
print(descriptions['1000268201_693b08cb0e'])
print(descriptions['3712923460_1b20ebb131'])

#%%

table = str.maketrans('', '', string.punctuation)
for key, desc_list in descriptions.items():
    #print(key, desc_list)
    for i in range(len(desc_list)):
        #print('1',desc_list[i])
        desc = desc_list[i]
        #print('2',desc)
        desc = desc.split()
        #print('3',desc)
        desc = [word.lower() for word in desc]
        #print('4',desc)
        desc = [w.translate(table) for w in desc]
        #print('5',desc)
        desc = [word for word in desc if (len(word) > 1)]
        #print('6',desc)
        desc = [word for word in desc if word.isalpha()]
        #print('7',desc)
        desc_list[i] = ' '.join(desc)
        #print('8',desc_list)
#print(desc_list)
#print(len(descriptions))
print(len(descriptions.keys()))

#%%

print(descriptions['3712923460_1b20ebb131'])
print(desc_list['3712923460_1b20ebb131'])

#%%

vocabulary = set()
for key in descriptions.keys():
    [vocabulary.update(d.split()) for d in descriptions[key]]
print('size = ', len(vocabulary))
print(len(descriptions))
print(len(descriptions.keys()))

#%%

train_captions = []
for key,val in descriptions.items():
    for captions in val:
        train_captions.append(captions)

wordcount_max = 10
word_count = {}
nsents = 0
for sent in train_captions:
    nsents = nsents + 1
    for w in sent.split():
        word_count[w] = word_count.get(w, 0) + 1
vocab = [w for w in word_count if word_count[w] > wordcount_max]
print('words = ', len(vocab))


#%%

print(vocab)

#%%

doc = open("flickr/Flickr8k_Dataset/Flicker8k_Dataset")

#%%

doc = load_doc("flickr/Flickr8k_text/Flickr_8k.trainImages.txt")

#%%

file = open('flickr/Flickr8k_text/Flickr_8k.trainImages.txt')
doc = file.read()
train_images = list()
for word in doc.split('\n'):
    image_identifier = word.split('.')[0]
    train_images.append(image_identifier)
print(len(train_images))

#%%

train_descriptions = dict()
#train_descriptions_intermediate = list()
for line in str(descriptions).split('\n'):
    for image_id in descriptions:
        train_descriptions_intermediate = list()
        for i in range(len(descriptions[image_id])):
            train_descriptions_intermediate.append('start_sequence ' + ''.join(descriptions[image_id][i]) + ' end_sequence')
        train_descriptions[image_id] = train_descriptions_intermediate

#%%

print(train_descriptions['3712923460_1b20ebb131'])
print(train_descriptions['2279980395_989d48ae72'])
print(len(train_descriptions))
#print(train_descriptions.keys())

#%%

model = InceptionV3(weights = 'imagenet')

#%%

model_new = Model(model.input, model.layers[-2].output)

#%%

model_new.summary()

#%%

image_path = "flickr/Flickr8k_Dataset/Flicker8k_Dataset"
#images = [cv2.imread(file) for file in glob.glob('image_path/*.jpg')]

#%%



#%%



#%%

path = glob.glob('flickr/Flickr8k_Dataset/Flicker8k_Dataset/*.jpg')
images = list()
for img in path:
    #print(img)
    n = cv2.imread(img)
    #print(n)
    x = image.img_to_array(n)
    #print(x)
    x = np.resize(x, (299, 299))
    #print(x.shape)
    x = np.expand_dims(x, axis = 0)
    #print(x.shape)
    images.append(x)
    #x = x.reshape(1, 299, 299, 3)
    #x = np.expand_dims(x, axis = 3)
    #x = preprocess_input(x)

#%%

#fea_vec_2048 = image_to_vector(x)

#%%

def image_to_vector(img):
    x = model_new.predict(img)
    feature_vector_2048 = np.reshape(x, x.shape[1])
    return feature_vector_2048

#%%



#%% raw

#%%

fea_vec_2048 = image_to_vector(x)

#%%

print(len(images))

#%%

print(images[0].shape)
type(images[0])

#%%

print(images[0].size)
print(images[0])
plt.imshow(images[1])

#%%

resized_images = []
inter1, inter2, inter3 = [], [], []
resized_images_array = []
for i in range(len(images)):
    resized_images.append(np.resize(images[i], (299, 299)))
    inter2.append(np.expand_dims(resized_images[i], axis = 0))
    inter3.append(preprocess_input(inter2[i]))
    resized_images_array.append(np.reshape(inter3[i], inter3[i].shape[-1]))

#%%

print(resized_images[1].shape)
plt.imshow(resized_images[1])

#%%

abc = preprocess_input(inter2[0])
print(abc.shape)

#%%



#%%


