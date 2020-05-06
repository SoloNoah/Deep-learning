import argparse
import sys
from turtle import pd

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import *
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from wordcloud import WordCloud
import io
import string
from numpy import array, argmax
from collections import Counter

'''Noah Solomon & Emilia Zorin & Asaf Arditi'''

'''
    remove punctuation and split data into sentences
    calls word tokenize at the end
'''

'''try'''
def sentance_splitting(sent_list, word_list,filename):
    file_reader = open(filename, "r")
    data = file_reader.readline()
    data.translate(string.punctuation)
    while data:
        no_punctuation = re.sub('[!#?,.:";{}()]', ' ', data)
        sent_list.append(sent_tokenize(no_punctuation))
        data = file_reader.readline()
    print(sent_list)
    tokenize(sent_list, word_list)
    file_reader.close()

'''
    the split sentences are split into tokens
'''


def tokenize(sent_list, word_list):
    for sentence in [item for sublist in sent_list for item in sublist]:
       word_list.append(word_tokenize(sentence))
    print(word_list)


'''
    set of stopwords is made and each word within our wordlist is compared to these stopwords.
    if the word is in the set, it is excluded. else included within new list.
    that list, total, is sent to stemmer.
'''


def stopwords_removal(word_list):
    stop_words = set(stopwords.words('english'))
    sentences = []
    for w in word_list:
        sentance = []
        for i in w:
            if i not in stop_words:
                sentance.append(i)
        sentences.append(sentance)
    return sentences


'''
    onlywords list, that holds the data without stop words, is sent to stemmer and returns stemmed data
'''


def stemmer(only_words):
    stemmer = PorterStemmer()
    stems = [stemmer.stem(single) for single in only_words]
    return stems


def onehot_encoding(stems):
    out = open(args.txt_filename.replace('.txt', '1hot.txt'), "w")
    values = array(stems)
    label_encoder = LabelEncoder()
    one_hot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = label_encoder.fit_transform(values)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    one_hot_encoded = one_hot_encoder.fit_transform(integer_encoded)
    for w in one_hot_encoded:
        out.write("\n" + str(w))
    out.close()


'''
    a dictionary is created. the purpose is to count each of the words in the given text.
'''


def create_dictionary(word_list):
    dictionary = dict()
    for w in word_list:
        sentance = []
        sentance.append(w)
        dictionary[w] = dictionary.get(w, 0) + 1
    return dictionary


'''
    return 20 most common words
'''


def common_words(words):
    final = []
    str1 = ""
    for element in words:
        str1 += " " + element
    split_it = str1.split()
    counter = Counter(split_it)
    most_occur = counter.most_common(20)
    # most_occur holds a list of tuples (word,freq) that converted into a list that contains the top 20 most frequent words
    for tuple_ele in most_occur:
        tup1 = ' '.join(map(str, tuple_ele))
        final.append((tup1.split())[0])
    return final

'''
    creates wordcloud with the given format and data supplied.
'''


def create_wordcloud(data):
    background_color = "arial"
    height = 720
    width = 1080
    word_cloud = WordCloud(
        background_color,
        width,
        height
    )
    word_cloud.generate_from_frequencies(data)
    word_cloud.to_file(args.txt_filename.replace('.txt', '_cloud')+'.png')


def read_col(taglist):
    df = pd.read_csv("BBC news dataset.csv")
    taglist.append(df['tags'])



def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data



parser = argparse.ArgumentParser()
parser.add_argument('txt_filename', nargs='?', help= "write the name of the txt file such as <name>.txt")
args = parser.parse_args()

sent_list, word_list, words_only_list = [], [], []
sentance_splitting(sent_list, word_list, args.txt_filename)

print("\nafter sentence spliting, tokenizing and punctuation:")
print(sent_list)
print(word_list, "\n")

word_list = stopwords_removal(word_list)
print("after stopwords removal:")
print(word_list, "\n")

word_list = stemmer([item for sublist in word_list for item in sublist])
print("after stemming:")
print(word_list, "\n")

onehot_encoding(word_list)
print("onehot encoding output in output1hot.txt file")

common = common_words(word_list)
dictionary = create_dictionary(common)
print(dictionary)
create_wordcloud(dictionary)
print("created image for words within top 20 frequencies")

print("Vector:\n")
print(load_vectors("wikinews.vec"))
x = []
print("excel spread sheet:\n")
read_col(x)
print(x)
