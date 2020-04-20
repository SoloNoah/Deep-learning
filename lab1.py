from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import *
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from wordcloud import WordCloud
import string
from numpy import array

'''
    remove punctuation and split data into sentences
    calls word tokenize at the end
'''


def sentance_splitting(sent_list, word_list):
    file_reader = open("sample.txt", "r")
    data = file_reader.readline()
    data.translate(string.punctuation)
    while data:
        no_punctuation = re.sub('[!#?,.:";{}()]', ' ', data)
        sent_list.append(sent_tokenize(no_punctuation))
        data = file_reader.readline()
    print(sent_list)
    tokenize(sent_list, word_list)

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
    values = array(stems)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    out = open("output1hot.txt", "w")
    for w in onehot_encoded:
        out.write(str(w))
        out.write("\n")

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
    word_cloud.to_file('image_cloud.png')


def main():
    sent_list, word_list, words_only_list = [], [], []
    sentance_splitting(sent_list, word_list)

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

    dictionary = create_dictionary(word_list)
    print(dictionary)
    create_wordcloud(dictionary)
    print("created image for words within top 20 frequencies")


main()
