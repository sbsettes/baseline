# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from nltk.tokenize import word_tokenize
import nltk
import string


# %%
def get_line(id,file_lines):
    for line in file_lines:
        if id in line:
            return line


# %%
def prepare_data(arguments,keypoints,labels):
    if(arguments and keypoints and labels):
        corpus = []
        matches = []
        for line in labels:
            arg_id = line.split(",")[0]
            keypoint_id = line.split(",")[1]
            
            match = int(line.split(",")[2])
            argument = get_line(arg_id,arguments)
            keypoint = get_line(keypoint_id,keypoints)
            corpus.append(argument + " " + keypoint)
            matches.append(match)
    matches = array(matches)
    return corpus,matches


# %%
# tokenize and count words
def tokenize_and_count(corpus):
    if corpus:
        all_words = []
        for line in corpus:
            table = str.maketrans(dict.fromkeys(string.punctuation))
            line = line.translate(table)
            line = line.lower()
            try:
                tokenize_word = word_tokenize(line)
            except:
                nltk.download('punkt')
                tokenize_word = word_tokenize(line)
            for word in tokenize_word:
                all_words.append(word)
        print ("number of words in arguments+keypoints: "+str(len(all_words)))
        unique_words = list(dict.fromkeys(all_words))
        vocab_length = len(unique_words) + 5
        print ("unique words: " + str(vocab_length))
        return vocab_length
    else:
        print("error while creating corpus")


# %%
def get_padded_sentences(corpus,vocab_length):
    embedded_sentences = [one_hot(sent, vocab_length) for sent in corpus]
    #print(embedded_sentences )

    # we need the length of each embedded sentence to be the same
    # so we calculate the length of the longest sentence embedding
    word_count = lambda sentence: len(word_tokenize(sentence))
    longest_sentence = max(corpus, key=word_count)
    length_long_sentence = len(word_tokenize(longest_sentence))
    print("length of the longest sentence: " + str(length_long_sentence))

    # now we add padding to the sentences that have a length smaller than length_long_sentence
    padded_sentences = pad_sequences(embedded_sentences, length_long_sentence, padding='post')
    print(padded_sentences)
    return padded_sentences,length_long_sentence


# %%
# simple one layer model to see if arguments and key points match or not
def create_and_compile_model(vocab_length,length_long_sentence):
    model = Sequential()
    model.add(Embedding(vocab_length, 20, input_length=length_long_sentence))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    print(model.summary())
    return model


# %%
# train the model
def train_model(model,padded_sentences,matches):
    model.fit(padded_sentences, matches, epochs=100, verbose=1)


# %%
# test the model
def test_model(model,padded_sentences, matches):
    loss, accuracy = model.evaluate(padded_sentences, matches, verbose=1)
    print('Accuracy: %f' % (accuracy*100))
    print('Loss: %f' % (loss*100))


# %%
# load training data
with open("./data/train/arguments_train.csv","r",encoding="utf-8") as f:
    arguments_train = f.readlines()[1:]
with open("./data/train/key_points_train.csv","r",encoding="utf-8") as f:
    keypoints_train = f.readlines()[1:]
with open("./data/train/labels_train.csv","r",encoding="utf-8") as f:
    labels_train = f.readlines()[1:]

print("keypoints training data size: "+str(len(keypoints_train)))
print("arguments training data size: "+str(len(arguments_train)))
print("labels training data size: "+str(len(labels_train)))

# train data embeddings
corpus_train,matches_train = prepare_data(arguments_train,keypoints_train,labels_train)
vocab_length_train = tokenize_and_count(corpus_train)


# %%
# load test data
with open("./data/test/arguments_dev.csv","r",encoding="utf-8") as f:
    arguments_dev = f.readlines()[1:]
with open("./data/test/key_points_dev.csv","r",encoding="utf-8") as f:
    keypoints_dev = f.readlines()[1:]
with open("./data/test/labels_dev.csv","r",encoding="utf-8") as f:
    labels_dev = f.readlines()[1:]

print("keypoints test data size: "+str(len(keypoints_dev)))
print("arguments test data size: "+str(len(arguments_dev)))
print("labels test data size: "+str(len(labels_dev)))

# test data embeddings
corpus_test,matches_test = prepare_data(arguments_dev,keypoints_dev,labels_dev)
vocab_length_test = tokenize_and_count(corpus_test)


if(vocab_length_test > vocab_length_train):
    vocab_length_train = vocab_length_test
else:
    vocab_length_test = vocab_length_train

print("generating train data embeddings.... ")
padded_sentences_train,length_long_sentence_train = get_padded_sentences(corpus_train,vocab_length_train)

print("generating test data embeddings.... ")
padded_sentences_test,length_long_sentence_test = get_padded_sentences(corpus_test,vocab_length_test)

if(length_long_sentence_test > length_long_sentence_train):
    length_long_sentence_train = length_long_sentence_test
else:
    length_long_sentence_test = length_long_sentence_train


# %%
# training phase

model_train = create_and_compile_model(vocab_length_train,length_long_sentence_train)
model_test = create_and_compile_model(vocab_length_test,length_long_sentence_test)

train_model(model_train,padded_sentences_train,matches_train)


# %%
print("going to test model...")
test_model(model_test,padded_sentences_test,matches_test)


