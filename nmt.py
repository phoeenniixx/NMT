
#import libraries
import tensorflow as tf
import numpy as np
import pandas as pd
import keras
import string

from nltk.tokenize import word_tokenize
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils import plot_model

from keras.models import Sequential
from keras.layers import LSTM,Bidirectional
from keras.layers import Dense,GRU,Input, Concatenate, Attention
from keras.layers import Embedding,Add
from keras.layers import RepeatVector
from keras.layers import TimeDistributed,Input
from keras.callbacks import ModelCheckpoint
from keras.models import Model

from nltk.translate.bleu_score import corpus_bleu

# Step-1: Download and clean the data
class Preprocess:
    def __init__(self, file_path, split_per,dataset_size):
        self.file_path = file_path
        self.split_per = split_per
        self.dataset_size=dataset_size
    def load_raw_data(self):
        with open(self.file_path, 'r') as file:
            text = file.read()
        lines = text.splitlines()
        self.raw_context = [line.split('\t')[0] for line in lines]
        self.raw_target = [line.split('\t')[1] for line in lines]
        return self.raw_target, self.raw_context

    def clean(self, t):
        s = set(string.punctuation)
        cleaned = []
        x = "startseq"
        for word in word_tokenize(t):
            if (word.lower() not in s):
                cleaned.append(word.lower())
        for i in cleaned:
            if i == cleaned[len(cleaned) - 1]:
                x = x + " " + i + " endseq"
            else:
                x = x + " " + i
        return x

    def cleaned_data(self):
        cleaned_target = [self.clean(i) for i in self.raw_target]
        cleaned_context = [self.clean(i) for i in self.raw_context]
        self.target=np.array(cleaned_target[:self.dataset_size])
        self.context=np.array(cleaned_context[:self.dataset_size])

    def train_test_split(self):
        self.vocab()
        index=int(self.split_per*len(self.sequence_target))
        indices= np.arange(len(self.sequence_target))
        np.random.shuffle(indices)
        training_indices=indices[:index]
        testing_indices=indices[index:]
        self.train_target=self.sequence_target[training_indices]
        self.test_target=self.sequence_target[testing_indices]
        self.test_raw_test=self.target[testing_indices]
        self.train_context=self.sequence_context[training_indices]
        self.testing_context=self.sequence_context[testing_indices]
        return  self.train_target, self.test_target,self.train_context,self.testing_context


    def tokenize(self):
        tokenizer = Tokenizer(oov_token='<UNK>')
        return tokenizer

    def vocab(self):
        self.cleaned_data()
        self.token_target = self.tokenize()
        self.token_context = self.tokenize()
        self.token_target.fit_on_texts(self.target)
        self.token_context.fit_on_texts(self.context)
        self.vocab_target = self.token_target.word_index.keys()
        self.vocab_context = self.token_context.word_index.keys()
        self.sequence_target = np.array(self.token_target.texts_to_sequences(self.target),dtype=object)
        self.sequence_context = np.array(self.token_context.texts_to_sequences(self.context),dtype=object)

    def max_seq_len(self, input_sequences):
        max_seq_len = max([len(seq) for seq in input_sequences])
        return max_seq_len

    def pad_sequence(self,x,lang):
      if lang=='target':
        max_seq_len = self.max_seq_len(self.sequence_target)
      elif lang=='context':
        max_seq_len = self.max_seq_len(self.sequence_context)
      padded_sequences = np.array(pad_sequences(x, maxlen=max_seq_len))
      return padded_sequences

    def flatten_sequence(self, sequences):
        return [item for sublist in sequences for item in sublist]

    def one_hot_encode(self, targets, vocab_size):
        # flattened_targets = self.flatten_sequence(targets)
        one_hot_targets = to_categorical(targets, num_classes=vocab_size)
        return one_hot_targets


file_path="fra-eng/fra.txt"
preprocessor=Preprocess(file_path,0.7,20000)
raw_target, raw_context=preprocessor.load_raw_data()


# Step-2: Split and Prepare the Data for Training
train_target,test_target,train_context,test_context = preprocessor.train_test_split()



padded_train_target=preprocessor.pad_sequence(train_target,'target')
padded_test_target=preprocessor.pad_sequence(test_target,'target')
padded_train_context=preprocessor.pad_sequence(train_context,'context')
padded_test_context=preprocessor.pad_sequence(test_context,'context')



target_vocab_size=len(preprocessor.vocab_target)+1
context_vocab_size=len(preprocessor.vocab_context)+1

trainY = preprocessor.one_hot_encode(padded_train_target, target_vocab_size)
testY=preprocessor.one_hot_encode(padded_test_target, target_vocab_size)

# Step 3: Define and Train the RNN-based Encoder-Decoder Model

def define_model(context_vocab, target_vocab, context_timesteps, target_timesteps, n_units):
    
    model = Sequential()
    model.add(Embedding(context_vocab, n_units, input_length=context_timesteps, mask_zero=True))
    model.add(Bidirectional(GRU(n_units)))
    model.add(RepeatVector(target_timesteps))
    model.add(LSTM(n_units, return_sequences=True))
    model.add(TimeDistributed(Dense(target_vocab, activation='softmax')))
    return model

max_seq_len_context=preprocessor.max_seq_len(preprocessor.sequence_context)
max_seq_len_target=preprocessor.max_seq_len(preprocessor.sequence_target)

model = define_model(context_vocab_size, target_vocab_size, max_seq_len_context, max_seq_len_target, 256)

train_context

trainY.shape

model.compile(optimizer='adam', loss='categorical_crossentropy')
# summarize defined model
print(model.summary())
plot_model(model, to_file='model.png', show_shapes=True)
# fit model
filename = 'model.keras'
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')



model.fit(padded_train_context, trainY, epochs=30, batch_size=64, validation_data=(padded_test_context, testY), callbacks=[checkpoint], verbose=1)

# Step-4 : Evaluate The Model

from keras.models import load_model
def predict_sequence(model, tokenizer, source):
    prediction = model.predict(source, verbose=0)[0]
    integers = [np.argmax(vector) for vector in prediction]
    target = []
    for i in integers:
        word = word_for_id(i, tokenizer)
        if word is None:
            continue  # Skip None values
        target.append(word)
    return ' '.join(target)  # Return None for empty sequences


def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def evaluate_model(model, tokenizer, sources, raw_targets):
    actual, predicted = list(), list()
    for i, source in enumerate(sources):
        # Translate encoded source text
        source = source.reshape((1, source.shape[0]))
        translation = predict_sequence(model, tokenizer, source)
        if translation is not None:
            raw_target = raw_targets[i]
            actual.append([raw_target.split()])  # Split the reference sentence into words
            predicted.append(translation.split())  # Split the predicted sentence into words
    if not actual or not predicted:
        print("No valid translations found.")
        return None, None
    # Calculate BLEU score
    print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))
    return actual, predicted


model=load_model('model.tf')

actual, predicted = evaluate_model(model, preprocessor.token_target, padded_test_context, preprocessor.test_raw_test)




