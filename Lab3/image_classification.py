import argparse
import csv
import io
import gensim
import pandas as pd
import os
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import KeyedVectors
from keras import Sequential
from keras.engine.saving import load_model
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense
import cv2
from tqdm import tqdm

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
    return data
def load_fast_text_vectors(file_name):
    if os.path.isfile(file_name):
        with io.open(file_name, 'r', encoding='utf-8', newline='\n', errors='ignore') as file:
           # number_of_lines, vector_size = map(int, file.readline().split())
            data = {}
            for line in file:
                tokens = line.rstrip().split(' ')
                data[tokens[0]] = list(map(float, tokens[1:]))
    return data
def ReadCsv(Filename):
    df = pd.read_csv(Filename)
    return df
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
def createModel():
    model = gensim.models.KeyedVectors.load_word2vec_format('wiki-news-300d-1M.vec', binary=False)
    return model
def unpickle(file):
    with open(file, 'rb') as fo:
        res = pickle.load(fo, encoding='bytes')
    return res
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
def word2vec_labels():
    # expects labels to be a list
    # each member of labels is a list of words (can be lenght 1)
    # process each member into vector of 300 floats

    meta_data = unpickle(CIFAR100_META_PATH)

    labels = np.array([name.decode('utf8')
                       for name in meta_data[b'fine_label_names']])  # named labels
    # some labels have multiple words
    split_labels = [label.split('_') for label in labels]
    # process labels into Word Vectors, use MEAN on multi word labels

    vectorized_labels = []
    for label in split_labels:
        if len(label) > 1:
            vectorized_labels.append(
                np.mean([word_vectors[word] for word in label], axis=0))
        else:
            vectorized_labels.append(word_vectors[label[0]])
    return vectorized_labels
def get_top_3_similar(prediction):

    vectorized_labels = word2vec_labels()  # bring in all possible labels as vectors
    scores = []  # calc our prediction vs each label
    for label in vectorized_labels:
        scores.append(cos_sim(label, prediction))
    # gather indexes of highest scores => corresponds to index of vectorized label
    top_n_similar_indexes = np.argsort(scores)[::-1][:len(scores)][:3]
    # get list of label where each label index corresponds to vectorized label index
    meta_data = unpickle('cifar-100-python/meta')
    labels = np.array([name.decode('utf8')
                       for name in meta_data[b'fine_label_names']])  # named labels
    most_relevant = [labels[i] for i in top_n_similar_indexes]

    return most_relevant
def cos_sim(a, b):

    from numpy import dot
    from numpy.linalg import norm

    return dot(a, b)/(norm(a)*norm(b))
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

parser = argparse.ArgumentParser()
parser.add_argument('--task',metavar='Task',choices=['train', 'test'],type=str, help='Choose between train / test')
parser.add_argument('--image',metavar='Image',type=str,help='image file name')
parser.add_argument('--model',metavar='Model',type=str,help='Trained model')
args = parser.parse_args()
CIFAR100_META_PATH = 'cifar-100-python/meta'
CIFAR100_TRAIN_PATH = 'cifar-100-python/train'
CIFAR100_TEST_PATH = 'cifar-100-python/test'

if (args.task == 'train'):
    print("--------Import cifar100 Train data--------\n")
    meta = unpickle(CIFAR100_META_PATH)
    fine_label_names = [t.decode('utf8') for t in meta[b'fine_label_names']]
    train = unpickle(CIFAR100_TRAIN_PATH)
    filenames_train = [t.decode('utf8') for t in train[b'filenames']]
    fine_labels_train = train[b'fine_labels']
    data_train = train[b'data']
    print("--------Import cifar100 Test data--------\n")
    test = unpickle(CIFAR100_TEST_PATH)
    filenames_test = [t.decode('utf8') for t in test[b'filenames']]
    fine_labels_test = test[b'fine_labels']
    data_test = test[b'data']
    print("--------Done\n")

    print("-------Create train_images List (x_train_list)\n")
    train_images = list()
    for d in data_train:
        image = np.zeros((32,32,3), dtype=np.uint8)
        image[...,0] = np.reshape(d[:1024], (32,32)) # Red channel
        image[...,1] = np.reshape(d[1024:2048], (32,32)) # Green channel
        image[...,2] = np.reshape(d[2048:], (32,32)) # Blue channel
        train_images.append(image)
    print("---------Done")
    print("-------Create test_images List (x_test_list)\n")
    test_images = list()
    for d in data_test:
        image = np.zeros((32,32,3), dtype=np.uint8)
        image[...,0] = np.reshape(d[:1024], (32,32)) # Red channel
        image[...,1] = np.reshape(d[1024:2048], (32,32)) # Green channel
        image[...,2] = np.reshape(d[2048:], (32,32)) # Blue channel
        test_images.append(image)
    print("----------Done")

    # print("---------Create train_images_list and save images to train_images directory---------------")
    # with open('cifar100train.csv', 'w+') as f:
    #     for index,image in tqdm(enumerate(train_images)):
    #         filename = filenames_train[index]
    #         label = fine_labels_train[index]
    #         label = fine_label_names[label]
    #         f.write('%s,%s\n'%(filename,label))
    # print("---------Done")

    # print("---------Create test_images_list and save images to test_images directory---------------")
    # with open('cifar100test.csv', 'w+') as f:
    #     for index,image in tqdm(enumerate(test_images)):
    #         filename = filenames_test[index]
    #         label = fine_labels_test[index]
    #         label = fine_label_names[label]
    #         f.write('%s,%s\n'%(filename,label))
    # print("----------Done")


    print("-------Pre-proccess for labels")
    split_names = pre_process_labels(fine_label_names)
    print("--------Done")

    print ("----------load vec model------------")
    vec_model = createModel()
    print("-----------Done")

    print("----------Create Dictionary of label->>>>W2V vector")
    i=0
    Lables_Vectors_list = []
    LabelsVecDict = {}
    for d in split_names:
        vec1 = get_mean_vector(vec_model, d)
        LabelsVecDict[fine_label_names[i]] = vec1
        Lables_Vectors_list.append(vec1)
        i=i+1
    print("----------Done")

    print("----------Create Y_test And Y_train:")
    Y_train, Y_test = CreateYtestYtrain(fine_labels_train,fine_labels_test,Lables_Vectors_list)
    print("----------Done")

    print("\n\n-----------------------------------------------------------SAVE!---------------------------------")
    print("-------Save Dictionary to CSV File")
    print ("*************************************************************\n")
    w = csv.writer(open("LabelsVecDict.csv", "w", encoding="utf-8",newline=''))
    for key, val in LabelsVecDict.items():
        w.writerow([key, val])
    print("--------Done")

    print("---------------Saving Y_train,Y_test AND listOfLablesVectors as TXT Files:")
    with open("Y_train.txt", "wb") as fp:
        pickle.dump(Y_train, fp)
    with open("Y_test.txt", "wb") as fp:
        pickle.dump(Y_test, fp)
    with open("Lables_Vectors_list.txt", "wb") as fp:
        pickle.dump(Lables_Vectors_list, fp)
    print("---------------Done-------------------")

    print("\n\n----------------------------------------------------Load----------------------------------")

    print("---------Load csv file to a Dictionary")
    LabelsVecDict2 = {}
    with open('LabelsVecDict.csv', mode='r', encoding="utf-8") as infile:
        reader = csv.reader(infile)
        LabelsVecDict2 = {rows[0]: rows[1].translate(str.maketrans({']': '', '[': ''})) for rows in reader}
    for key,value in LabelsVecDict2.items():
        LabelsVecDict2[key] = [float(x) for x in LabelsVecDict2[key].split()]
    print("----------Done")

    print("---------Load Y_TRAIN AND Y_TEST listOfLablesVectors INTO A LISTS:")
    Y_train_load, Y_test_load,Lables_Vectors_list_load = [],[],[]
    with open("Y_train.txt", "rb") as fp:
        Y_train_load = pickle.load(fp)
    with open("Y_test.txt", "rb") as fp:
        Y_test_load = pickle.load(fp)
    with open("Lables_Vectors_list.txt", "rb") as fp:
        Lables_Vectors_list_load = pickle.load(fp)
    print("---------------------Done\n\n")

    print("--------Create CNN Model-------------")
    model = createNetworkModel()
    model.fit(np.asarray(train_images), np.asarray(Y_train_load), epochs=25, batch_size=500)
    score, accuracy = model.evaluate(np.asarray(test_images), np.asarray(Y_test_load), batch_size=500)
    print("Model Score = {:.2f}".format(score))
    print("Accuracy = {:.2f}".format(accuracy*100))

    model.save(args.model)

if (args.task == 'test'):
    print("\n\n----------------Test the model!------------------")
    print("-------------Loading model and vectors-------")
    model = load_model(args.model)
    fast_text_vectors = load_fast_text_vectors('wiki-news-300d-1M.vec')
    print('------Done')

    print("----------Loading image and resizing-------")
    # word2vec_model = fastText.load_model('wiki-news-300d-1M.vec')
    image = cv2.imread(args.image)
    image = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
    image = np.expand_dims(image, axis=0)
    print("--------Done")

    print("----------------Load image to the CNN model:")
    prediction = model.predict(image)
    new_prediction = list(prediction[0])
    print("-----DONE")

    print("----------Calculate Cosine similarity for each word in cifar-100 dictionary:")
    i = 0
    cosine_similarities_check = {}
    for key in fast_text_vectors:
        if (i != 0):
            cosine_similarities_check[key] = cosine_similarity(np.asarray(new_prediction).reshape(1, -1),
                                                               np.asarray([fast_text_vectors[key]]).reshape(1, -1))
        i = i + 1
        if (i%100000 == 0):
            print (i)
    print(sorted(cosine_similarities_check, key=cosine_similarities_check.get, reverse=True)[:3])

    print("-----------Done")

    # print("Vizualiztion results:")
    # import matplotlib.pyplot as plt
    # top3_CS_Results = sorted(cosine_similarities_check, key=cosine_similarities_check.get, reverse=True)[:3]
    # mydict = {}
    # for label in top3_CS_Results:
    #     mydict[label] = cosine_similarities_check[label]
    # print ("-----------------mydict:{0}\n".format(mydict))
    # labels = list(mydict.keys())
    # C_similarity = list(mydict.values().reshape(1,-1))
    # fig, axs = plt.subplots()
    # axs.bar(labels, C_similarity)
    # fig.suptitle('Categorical Plotting')
    # plt.show()



    # print("----------Calculate Cosine similarity for each word in cifar-100 dictionary:")
    # i=0
    # cosine_similarities_check = {}
    # for key in LabelsVecDict2:
    #     if (i!=0):
    #         cosine_similarities_check[key] = cosine_similarity(np.asarray(new_prediction).reshape(1,-1), np.asarray([LabelsVecDict2[key]]).reshape(1,-1))
    #     i=i+1
    # print(sorted(cosine_similarities_check, key=cosine_similarities_check.get, reverse=True)[:3])
    # print("-----------Done")










