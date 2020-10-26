import argparse
import csv
import json
import os
from copy import deepcopy
import gensim
import nltk
from keras.engine.saving import load_model
from keras.optimizers import SGD
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import io
import string
from collections import Counter
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = np.array(list(map(float,tokens[1:])))
    print(data)
    # return data
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
def preProcessScript(query):
    def RemoveStopWords(query):
        stop_words = set(stopwords.words('english'))
        updated_query = [w for w in query if w not in stop_words and w.isalpha()]
        return updated_query

    def ToLower(query):
        updated_query = [w.lower() for w in query]
        return updated_query

    def RemovePunctuation(query):
        table = str.maketrans('', '', string.punctuation)
        updated_query = [w.translate(table) for w in query]
        return updated_query
    return RemoveStopWords(ToLower(RemovePunctuation(query)))
def Tokenize(df,header):
    list,list1 = [],[]
    flag = False
    for d in df[header]:
        if flag:
            list = d.split()
            list1.append(list)
        flag = True
    return list1
def TokenizeScript(query):
    tokenized_sents = [word_tokenize(i) for i in query]
    query_list_updated = [val for sublist in tokenized_sents for val in sublist]
    return query_list_updated
def createModel():
    model = gensim.models.KeyedVectors.load_word2vec_format('wikinews.vec', binary=False)
    return model
def get_mean_vector(word2vec_model, words):
    # remove out-of-vocabulary words
    words = [word for word in words if word in word2vec_model.vocab]
    if len(words) >= 1:
        return np.mean(word2vec_model[words], axis=0)
    else:
        return []
def CreateWordDict(df,header,Dict):
    Flag = False;
    for index, row in df.iterrows():
        if Flag:
            for word in row[header].split():
                if word in Dict:
                    Dict[word] += 1
                else:
                    Dict[word] = 1
        Flag = True
    return Dict
def most_common_inclusive(freq_dict, n):
    # find the nth most common value
    c = Counter(freq_dict)
    if (len(freq_dict) < n):
        n = len(freq_dict)
    nth_most_common = sorted(c.values(), reverse=True)[n-1]
    return { k: v for k, v in c.items() if v >= nth_most_common }
def Create_1hot_foreachWord(wordsSet):
    def createVector(num):
        vector_length = len(wordsSet)
        vec=[]
        [vec.append(0) for i in range(0,vector_length)]
        vec[num]=1
        return vec
    dictOfWords={}
    for count,word in enumerate(wordsSet):
        dictOfWords[word]=createVector(count)
    return dictOfWords
def replaceWords(vector_dict,final_sentences):
    temp=deepcopy(final_sentences)
    temp2 = []
    for count1,sent in enumerate(temp):
        for count2,word in enumerate(sent):
            if word in vector_dict.keys():
                temp2.append(vector_dict[word])
    return ([temp2])
def replaceWords2(vector_dict,final_sentences):
    temp,temp2 = [],[]
    for count1, sent in enumerate(final_sentences):
        for count2, word in enumerate(sent):
            if word in vector_dict.keys():
                temp2.append(vector_dict[word])
        if not temp2:
            temp.append([[0]*len(vector_dict)])
        else:
            temp.append(temp2)
            temp2 = []
    return (temp)
def Create_N_hot(hot1):
    n_hot = []
    for sentence in hot1:
        x = [sum(i) for i in zip(*sentence)]
        n_hot.append(x)
        for x in n_hot:
            for count, num in enumerate(x):
                if (x[count] > 1):
                    x[count] = 1
    return n_hot

listOfDescriptionWords, listOfTagsWords,descriptionList1,descriptionList,tagList,tagList1 = [],[],[],[],[],[]
listOfDescriptionVectors, listOfTagsVectors = [],[]
query_list, query_list_updated, sent_list, sent_list_updated, sent_list_processed = [], [], [], [], []
WordFreqDict= {}

 # -------- Parser configuration ------------:
parser = argparse.ArgumentParser()
parser.add_argument('--query',metavar='Query',type=str,help='File containing query sentence.')
parser.add_argument('--text',metavar='Text',type=str,help='File containing text to match query sentence')
parser.add_argument('--task',metavar='Task',choices=['train', 'test'],type=str, help='Choose between train / test')
parser.add_argument('--data',metavar='Data',type=str,help='Training dataset in CSV format.')
parser.add_argument('--model',metavar='Model',type=str,help='Trained model')
parser.add_argument('--representation',metavar='Representation',choices=['w2v', 'n-hot'],type=str, help='Choose between w2v / n-hot representation')
args = parser.parse_args()

if (args.task == 'train'):
    print("------------- You chose TRAIN:")
    print("------------- Loading and preprocess training data file:")
    header = ['description', 'tags']
    df = pd.read_csv(os.path.join(os.getcwd(), args.data),names=header)
    df = preProcess(df, 'description')
    df = preProcess(df, 'tags')
    listOfDescriptionWords = Tokenize(df, 'description')
    listOfTagsWords = Tokenize(df, 'tags')
    print("------------- Done.")

    if (args.representation == 'n-hot'):
        print("----------------------- You chose to train a N-HOT represntation of the train-data sentences:")
        print("------------- Create a Word-Dict from the train-data file.")
        WordFreqDict = CreateWordDict(df,'description', WordFreqDict)
        WordFreqDict = CreateWordDict(df,'tags', WordFreqDict)
        print("------------- Done.")

        WordFreqDict = most_common_inclusive(WordFreqDict,1000)

        print("------------- Create a 1-hot dictionary:")
        Wordslist = list(WordFreqDict.keys())
        hot1_dict = Create_1hot_foreachWord(Wordslist)
        print("------------- Done.")

        print("------------- save 1-hot represntation dictionary as a json file")
        with open('wordVectorDict.json', 'w') as fp:
            json.dump(hot1_dict, fp)
        print("------------- Done.")

        print("------------- Create n-hot represntaion + Split the data: train-70%, text- 30%:")
        hot1_description =replaceWords2(hot1_dict,listOfDescriptionWords)
        hot1_tags = replaceWords2(hot1_dict,listOfTagsWords)
        n_hot_tags = Create_N_hot(hot1_tags)
        n_hot_description = Create_N_hot(hot1_description)

        X_train_hot, X_test_hot, y_train_hot, y_test_hot = train_test_split(n_hot_tags, n_hot_description, test_size=0.3)

        for x in n_hot_tags:
            VectorSize = len(x)
        print("------------- Done.")

        X_test_hot = np.asarray(X_test_hot)
        y_test_hot = np.asarray(y_test_hot)
        X_train_hot = np.asarray(X_train_hot)
        y_train_hot = np.asarray(y_train_hot)
        print(X_test_hot)
        print('------------- Construct neural network:')
        model = Sequential()
        model.add(Dense(VectorSize, input_dim=VectorSize, activation='relu', kernel_initializer='he_normal'))
        model.add(Dense(200, activation='sigmoid'))
        model.add(Dense(250))
        model.add(Dense(300, activation='relu'))
        model.add(Dense(VectorSize))
        print(model.summary())

        opt = SGD(learning_rate=0.005, momentum=0.01)
        model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])
        print('------------- Done.')

        print('------------- Training NN model:')
        model.fit(X_train_hot, y_train_hot, epochs=100, batch_size=16, verbose=2)
        print('------------- Done.')

        score, accuracy = model.evaluate(X_test_hot, y_test_hot, batch_size=16, verbose=0)
        print("------------- Test fraction correct (NN-Score) = {:.2f}".format(score))
        print("------------- Test fraction correct (NN-Accuracy) = {:.2f}".format(accuracy))

        print('------------- Saving Model to disk')
        model.save(args.model)
        print('------------- Done.')

    if (args.representation == 'w2v'):
        print("----------------------- You chose w2v represntation:")
        print("------------- Loading word vectors:")
        vec_model = createModel()
        print("------------- Done.")

        print("-------------  Make average word embedding for sentence and tags:")
        for d in listOfDescriptionWords:
            vec1 = get_mean_vector(vec_model,d)
            listOfDescriptionVectors.append(vec1)
        for t in listOfTagsWords:
            vec2 = get_mean_vector(vec_model,t)
            listOfTagsVectors.append(vec2)
        print("------------- Done.")

        X_train,X_test,y_train,y_test = train_test_split(listOfTagsVectors,listOfDescriptionVectors,test_size = 0.3)
        y_test = np.asarray(y_test)
        y_train = np.asarray(y_train)
        X_test = np.asarray(X_test)
        X_train = np.asarray(X_train)


        print('------------- Construct neural network:')
        model = Sequential()
        model.add(Dense(300, input_dim=300, activation='relu'))
        model.add(Dense(500, activation='relu'))
        model.add(Dense(600,activation='relu'))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(150, activation='relu'))
        model.add(Dense(200, activation='relu'))
        model.add(Dense(250, activation='relu'))
        model.add(Dense(300))
        print('------------- Done.')

        print(model.summary())
        opt = SGD(learning_rate=0.005, momentum=0.01)
        model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])
        print('------------- Neural Network is ready!')

        print('------------- Training NN model:')
        model.fit(X_train, y_train, epochs=50,batch_size=64)
        print('------------- NN model is Trained.')
        score, accuracy = model.evaluate(X_test, y_test, batch_size=16, verbose=0)
        print("------------- Test fraction correct (NN-Score) = {:.2f}".format(score))
        print("------------- Test fraction correct (NN-Accuracy) = {:.2f}".format(accuracy))

        print('------------- Saving Model to disk')
        model.save(args.model)
        print('------------- Model is saved to disk.')
        print('************************ Train is finished *******************************8')
if (args.task == 'test'):
    print("------------- You chose Test:")

    if (args.representation == 'n-hot'):
        print("----------------------- You chose n-hot represntation:")
        print('------------- Loading NN model:')
        loaded_model = load_model(args.model)
        print("------------- Done.")

        print("----------------------- Compile loaded model:")
        loaded_model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
        print("------------- Done.")

        print("------------- Read and preprocess query from txt file:")
        with open(args.query, 'r') as file:
            query = file.read()
        query_list = list(query.split())
        query_list = preProcessScript(query_list)
        print("------------- Done.")

        print("------------- Read and preprocess text from the text file:")
        with open(args.text, 'r') as file:
            text = file.read()
        final_sentences, query_list_updated = [], []
        tokenizer = nltk.RegexpTokenizer(r"\w+")
        Tokenized_sentences = nltk.sent_tokenize(text)
        for sent in Tokenized_sentences:
            final_sentences.append((preProcessScript(tokenizer.tokenize(sent))))
        print("------------- Done.")

        print("------------- Load Words-Vector Dictionary (Json File):")
        with open('wordVectorDict.json', 'r') as fp:
            Words_Vector_Dict = json.load(fp)
        print("------------- Done.")

        print("------------- Create n-hot represntion for text and query:")
        list_query = []
        list_query.append(query_list)
        hot1_query = replaceWords2(Words_Vector_Dict, list_query)
        hot1_text = replaceWords2(Words_Vector_Dict, final_sentences)
        n_hot_query = Create_N_hot(hot1_query)
        n_hot_text = Create_N_hot(hot1_text)
        print("------------- Done.")


        print("---------- Create dictionary of [vector] -> sentence:")
        dict_sentences = {}
        count = 0
        for vec in n_hot_text:
            dict_sentences[str(vec)] = Tokenized_sentences[count]
            count += 1
        n_hot_query = np.asarray(n_hot_query)
        print("------------- Done.")
        # if (n_hot_query.size==0):
        #     print("you entered a query with no words from the dictionary.")
        #     exit(0)
        print(n_hot_query.reshape(1, len(Words_Vector_Dict)))
        print("---------- supply query as n-hot vector to the model, and obtain vector represntation of the sentence:")
        predicted_Vector = loaded_model.predict(n_hot_query.reshape(1, len(Words_Vector_Dict)))
        print("------------- Done.")
        print("predicted_Vector: {0}".format(predicted_Vector))

        print("----------- Find the most similar sentence for a given query:")
        max_similar = 0
        for vec in n_hot_text:
            vecaraay = np.asarray(vec)
            similar = cosine_similarity(vecaraay.reshape(1, -1), predicted_Vector.reshape(1, -1))
            if similar[0][0] >= max_similar:
                max_similar = similar[0][0]
                most_sim_sent = dict_sentences[str(vec)]

        print("The similarity between the prediction and the most similar phrase is :{:.2f}% ".format(max_similar * 100))
        print("------------------------------------------ the similar sentewnce is:{0}\n".format(most_sim_sent))

        print("----------- Save the most similar sentence in a txt file:")
        with open("most_similar.txt", "w") as text_file:
            text_file.write(most_sim_sent)
        print("----------- Test is finished!")
    if (args.representation == 'w2v'):
        print('------------- Loading NN model:')
        loaded_model = load_model(args.model)
        print("------------- NN Model is loaded from disk.")

        loaded_model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

        print("------------- Read query from txt file.")
        with open(args.query, 'r') as file:
            query = file.read()
        print("------------- Query is loaded.")

        print("------------- PreProcess Query:.")
        query_list = list(query.split())
        query_list = preProcessScript(query_list)
        print("------------- query is preprocessed.")

        print("------------- Read the text file.")
        with open(args.text, 'r') as file:
            text = file.read()
        print("------------- file is loaded.")
        print("------------- Preproccess Text:")
        final_sentences, query_list_updated = [], []
        tokenizer = nltk.RegexpTokenizer(r"\w+")
        Tokenized_sentences = nltk.sent_tokenize(text)
        for sent in Tokenized_sentences:
            final_sentences.append((preProcessScript(tokenizer.tokenize(sent))))
        print("------------- Text is preprocessed.")

        print("------------- Load vec_model.")
        vec_model = createModel()
        print("------------- vec_model is loaded.")

        print("---------- Create vector represnation of query:")
        query_vec = get_mean_vector(vec_model, query_list)

        print("---------- Create vector represnation of text:")
        text_vec, sentence_vec = [], []
        for sentence in final_sentences:
            sentence_vec = get_mean_vector(vec_model, sentence)
            text_vec.append(sentence_vec)


        print("-------- create predicte vector:")
        predicted_Vector = loaded_model.predict(query_vec.reshape(1, 300))

        print("---------- Create dictionary of [vector] -> sentence:")
        dict_sentences = {}
        count = 0
        for vec in text_vec:
            dict_sentences[str(vec)] = Tokenized_sentences[count]
            count += 1

        print("----------- Find the most similar sentence for a given query\n:")
        max_similar = 0
        for vec in text_vec:
            similar = cosine_similarity(vec.reshape(1, -1), predicted_Vector.reshape(1, -1))
            print("similar of the sentence: {0} is {1}".format(dict_sentences[str(vec)], similar[0][0]))
            if similar[0][0] >= max_similar:
                max_similar = similar[0][0]
                most_sim_sent = dict_sentences[str(vec)]

        print("The similarity between the prediction and the most similar phrase is :{:.2f}% ".format(
            max_similar * 100))
        print("------------------------------------------ the similar sentewnce is:\n")
        print(most_sim_sent)

        print("********************* Test is finished! ******************************")



