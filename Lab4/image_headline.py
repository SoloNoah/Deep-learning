import argparse
import csv
import io
import gensim
import pandas as pd
import os
import numpy as np
import pickle
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import KeyedVectors
from keras import Sequential
from keras.engine.saving import load_model
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense
import cv2
from numpy import array
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from tqdm import tqdm


def RemoveEmptyLines(df):
    df[df.isnull().any(axis=1)]
    df.drop(df[df.isnull().any(axis=1)].index, inplace=True)
    return df
def Tokenize(df,header):
    list,list1 = [],[]
    for d in df[header]:
        list = d.split()
        list1.append(list)
    return list1
def preProcessForRNN(df, header):
    def RemovePunctuation(df):
        df[header] = df[header].str.replace('[^\s\w]','')
        return df
    def RemoveEmptyLines(df):
        df[df.isnull().any(axis=1)]
        df.drop(df[df.isnull().any(axis=1)].index, inplace=True)
        return df
    return RemovePunctuation(RemoveEmptyLines(df))
def load_fast_text_vectors(file_name):
    if os.path.isfile(file_name):
        with io.open(file_name, 'r', encoding='utf-8', newline='\n', errors='ignore') as file:
           # number_of_lines, vector_size = map(int, file.readline().split())
            data = {}
            for line in file:
                tokens = line.rstrip().split(' ')
                data[tokens[0]] = list(map(float, tokens[1:]))
    return data
def PreProcessVector(predicted_vector):
    word_seq = []
    word_seq.append(predicted_vector)
    for i in range(1,seqLen):
        word_seq.append(pd.np.zeros((300,), dtype=float))
    word_seq_data = array(word_seq)
    word_seq_data1= np.expand_dims(word_seq_data, axis=0)
    return word_seq_data1
def unpickle(file):
    with open(file, 'rb') as fo:
        res = pickle.load(fo, encoding='bytes')
    return res
def get_mean_vector(word2vec_model, words):
    # remove out-of-vocabulary words
    words = [word for word in words if word in word2vec_model.vocab]
    if len(words) >= 1:
        return np.mean(word2vec_model[words], axis=0)
    else:
        return []
def get_one_vector_for_RNN(word2vec_model, words):
    words = [word for word in words if word in word2vec_model.vocab]
    if len(words) >= 1:
        return pd.np.mean(word2vec_model[words], axis=0)
    else:
        return np.zeros((300,), dtype=float)
def createModel(filename):
    model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=False)
    return model
def similarity(predict):
    print("----------Calculate Cosine similarity for each word in cifar-100 dictionary:")
    i = 0
    cosine_similarities_check = {}
    for key in fast_text_vectors:
        if (i != 0):
            cosine_similarities_check[key] = cosine_similarity(np.asarray(predict).reshape(1, -1),
                                                               np.asarray([fast_text_vectors[key]]).reshape(1, -1))
        i = i + 1
        if (i%100000 == 0):
            print (i)
    Predicted_Word = sorted(cosine_similarities_check, key=cosine_similarities_check.get, reverse=True)[:1]

    for w in Predicted_Word:
        word = w
    return word,fast_text_vectors[word]
def createNetworkModel():
    chanDim = -1
    inputShape = (32, 32, 3)
    pool_size = (2, 2)
    model = Sequential()
    cnn_size = 3
    for i in range(cnn_size):
        model.add(Conv2D(32, (3, 3), input_shape=inputShape, activation='relu', padding="same"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size))
        model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(300, activation='relu'))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Dropout(0.5))
    model.add(Dense(400, activation='relu'))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Dropout(0.5))
    model.add(Dense(300))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Dropout(0.5))

    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    return model
def pre_process_labels(fine_label_names):
    split_names = []
    for w in fine_label_names:
        w = w.split('_')
        split_names.append(w)
    return split_names
def CreateYtestYtrain(fine_labels_train,fine_labels_test,Lables_Vectors_list):
    Y_train,Y_test = [],[]
    i = 0
    for i in range(len(fine_labels_train)):
        Y_train.append(Lables_Vectors_list[fine_labels_train[i]])
    for i in range(len(fine_labels_test)):
        Y_test.append(Lables_Vectors_list[fine_labels_test[i]])
    return (Y_train,Y_test)
def CreateRNNModel():
    RNN_Model = Sequential()
    RNN_Model.add(LSTM(300, input_shape=(seqLen, 300), return_sequences=True, activation='tanh'))
    RNN_Model.add(LSTM(300, input_shape=(seqLen, 300), return_sequences=False, activation='tanh'))
    RNN_Model.add(Dense(300))
    RNN_Model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    print("--------------------------Done")
    print("----------------------------Fit LSTM model1")
    RNN_Model.fit(np.asarray(X_train), np.asarray(y_train), epochs=10, batch_size=500)
    score, accuracy = RNN_Model.evaluate(np.asarray(X_test), np.asarray(y_test))
    print("Model Score = {:.2f}".format(score))
    print("Accuracy = {:.2f}".format(accuracy * 100))
    return RNN_Model
def preProcess(df, header):
    def RemoveStopWords(df):
        stop_words = set(stopwords.words('english'))
        df[header] = df[header].apply(lambda x: " ".join(x for x in x.split() if x not in stop_words))
        return df
    def ToLower(df):
        df[header] = df[header].apply(lambda x: x.lower())
        return df
    def RemovePunctuation(df):
        df[header] = df[header].str.replace('[^\s\w]','')
        return df
    def RemoveEmptyLines(df):
        df[df.isnull().any(axis=1)]
        df.drop(df[df.isnull().any(axis=1)].index, inplace=True)
        return df
    return ToLower(RemoveStopWords(RemovePunctuation(RemoveEmptyLines(df))))
parser = argparse.ArgumentParser()
parser.add_argument('--task',metavar='Task',choices=['train', 'test'],type=str, help='Choose between train / test')
parser.add_argument('--image',metavar='Image',type=str,help='image file name')
parser.add_argument('--model3',metavar='Model3',type=str,help='Lab3 CNN model file name')
parser.add_argument('--model4',metavar='Model4',type=str,help='Lab4 RNN model file name')
parser.add_argument('--wordvec',metavar='WordVec',type=str,help='word vectors file')
parser.add_argument('--model2',metavar='Model2',type=str,help='Lab2 regression model')
args = parser.parse_args()
CIFAR100_META_PATH = 'cifar-100-python/meta'
CIFAR100_TRAIN_PATH = 'cifar-100-python/train'
CIFAR100_TEST_PATH = 'cifar-100-python/test'
seqLen =2

if (args.task == 'train'):

    print('\n\n--------LAB4 Train: train a RNN model & save it:\n ')
    print ("----------load vec model------------")
    vec_model = createModel(args.wordvec)
    print("-----------Done")

    print("----------Load BBC Data:")
    df = pd.read_csv('BBC news dataset.csv')
    print("----------Done")

    print("-------Pre process CNN data & create a sequences list & create labels list:")
    listOfDescriptionWords = preProcessForRNN(df, 'description')
    listOfDescriptionWords = Tokenize(df, 'description')
    total, tags = [], []
    i = 0
    for i in range(len(listOfDescriptionWords)):
        len_of_words = len(listOfDescriptionWords[i])
        for j in range(len_of_words - 1):
            counter = 0
            if (len_of_words - j > seqLen):
                newSeq = []
                while (counter < seqLen and j <= len_of_words):
                    newSeq.append((listOfDescriptionWords[i])[j])
                    j = j + 1
                    counter = counter + 1
                total.append(newSeq)
    for i in range(len(listOfDescriptionWords)):
        for ind in (listOfDescriptionWords[i])[seqLen:]:
            tags.append([ind])
    print("-------------done")

    print("-------------  Make average word embedding for sentence and tags:")
    label_vec = []
    total_cp = total
    for i in range(len(total_cp)):
        for j in range(len(total_cp[i])):
            total[i][j] = get_one_vector_for_RNN(vec_model, total_cp[i][j])
    for tag in tags:
        single_vec = get_one_vector_for_RNN(vec_model, tag)
        label_vec.append(single_vec)
    print("-----------Done")

    print("-------------LSTM Model:")
    print("-----------Split to train and test data:")
    X_train, X_test, y_train, y_test = train_test_split(total, label_vec, test_size=0.2, random_state=42)
    print("-------------------Done")

    print ("----------------------------Create RNN model1")
    RNN_Model = CreateRNNModel()
    print("--------------------------Done")

    print("----------------------------SAVE LSTM model1")
    RNN_Model.save(args.model4)
    print("----------------Done")

    print("--------------------------Train process is done")

if (args.task == 'test'):
    print("\n\n----------------Test the image model!------------------")
    print("-------------Loading model and vectors-------")
    CNN_Model = load_model(args.model3)
    fast_text_vectors = load_fast_text_vectors(args.wordvec)
    print('------Done')

    print("----------Loading image and resizing-------")
    image = cv2.imread(args.image)
    image = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
    image = np.expand_dims(image, axis=0)
    print("--------Done")

    print("----------------Load image to the CNN model:")
    prediction = CNN_Model.predict(image)
    new_prediction = list(prediction[0])
    print("-----DONE")

    print("----------Calculate Cosine similarity for each word in cifar-100 dictionary:")
    first_word, first_word_predicted_vector = similarity(new_prediction)

    print('------------Preprocess to first_word for feed the Rnn model:')
    first_seq_data = PreProcessVector(first_word_predicted_vector)
    print('------------done')

    print("----------load text_model (RNN MODEL)")
    RNN_Model = load_model(args.model4)
    print("--------Done")

    print('---------- Predict second and third words from RNN model:')
    second_prediction = RNN_Model.predict(first_seq_data)
    second_word, second_vector = similarity(second_prediction)
    second_seq_data = PreProcessVector(second_vector)
    third_prediction = RNN_Model.predict(second_seq_data)
    third_word,third_vector = similarity(third_prediction)
    print('---------- Done')

    print('CNN+RNN headline is "{0},{1},{2}"'.format(first_word, second_word, third_word))

    print('------------------------------Bonus Task:---------------------')

    print('-------------Preprocess headline for feed it to lab 2 model 2:')
    predicted_headline = []
    predicted_headline.append([first_word])
    predicted_headline.append([second_word])
    predicted_headline.append([third_word])
    print('------------Done')

    print("---------- load word2vec_model:")
    word2vec_model = createModel(args.wordvec)
    print("---------- Done")

    print("---------- create CNN+RNN headline vector presentation:")
    predicted_headline_vector = get_one_vector_for_RNN(word2vec_model, [first_word, second_word, third_word])
    print("-------- Done")
    print("------------- Loading and preprocess training data file:")
    header = ['description', 'tags']
    df = pd.read_csv(os.path.join(os.getcwd(), 'BBC news dataset.csv'),
                     names=header)
    df = preProcess(df, 'description')
    listOfDescriptionWords = Tokenize(df, 'description')
    print("------------- Done.")

    listOfDescriptionVectors, listOfTagsVectors = [], []
    print("-------------  Make average word embedding for sentence and tags:")
    for d in listOfDescriptionWords:
        vec1 = get_mean_vector(word2vec_model, d)
        listOfDescriptionVectors.append(vec1)

    print('---------Done')
    print('------------- Loading NN model:')
    loaded_model = load_model(args.model2)
    print("------------- NN Model is loaded from disk.")

    loaded_model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

    print("-------- create predicte vector:")
    predicted_Vector = loaded_model.predict(predicted_headline_vector.reshape(1, 300))
    print('---------Done')

    print("---------- Create dictionary of [vector] -> sentence:")
    dict_sentences = {}
    count = 0
    for vec in listOfDescriptionVectors:
        dict_sentences[str(vec)] = listOfDescriptionWords[count]
        count += 1
    print('--------Done')

    print("----------- Find the most similar sentence for a given query\n:")
    max_similar = 0
    print("-------- create predicte vector:")
    predicted_Vector = loaded_model.predict(predicted_headline_vector.reshape(1, 300))

    print("---------- Create dictionary of [vector] -> sentence:")
    dict_sentences = {}
    count = 0
    for vec in listOfDescriptionVectors:
        dict_sentences[str(vec)] = listOfDescriptionWords[count]
        count += 1

    print("----------- Find the most similar sentence for a given query\n:")
    max_similar = 0
    i = 0
    for vec in listOfDescriptionVectors:
        similar = cosine_similarity(vec.reshape(1, -1), predicted_Vector.reshape(1, -1))
        if similar[0][0] >= max_similar:
            max_similar = similar[0][0]
            most_sim_sent = dict_sentences[str(vec)]
            similar_index = i
        i = i + 1
    print('--------Done')

    df1 = pd.read_csv(os.path.join(os.getcwd(), 'BBC news dataset.csv'),
                      names=header)
    df1 = RemoveEmptyLines(df1)

    print("The similarity between the prediction and the most similar phrase is :{:.2f}% ".format(
        max_similar * 100))
    print("------------------------------------------ the new headline sentence is:\n")
    print(df1.iloc[similar_index, 0])


    print('Test Script is Finished.')







