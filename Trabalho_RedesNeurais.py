# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 16:32:04 2020
"""
#===============================================================================
#   EEL817 (Redes Neurais)
#   TRABALHO:   Um estudo sobre NLP (Processamento de Linguagem Natural) -
#               Análise de sentimento de filme usando redes neurais profundas
#               com a biblioteca Keras.
#
#   GRUPO: 
#       GIOVANNI PAES LEME DA GAMA RODRIGUES
#       RAUL BAPTISTA DE SOUZA
#
#   DATASET: Large Movie Review Dataset v1.0
#     @InProceedings{maas-EtAl:2011:ACL-HLT2011,
#       author    = {Maas, Andrew L.  and  Daly, Raymond E.  and  Pham, Peter T.  and  Huang, Dan  and  Ng, Andrew Y.  and  Potts, Christopher},
#       title     = {Learning Word Vectors for Sentiment Analysis},
#       booktitle = {Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies},
#       month     = {June},
#       year      = {2011},
#       address   = {Portland, Oregon, USA},
#       publisher = {Association for Computational Linguistics},
#       pages     = {142--150},
#       url       = {http://www.aclweb.org/anthology/P11-1015}
#     }
#   Disponível em: http://ai.stanford.edu/~amaas/data/sentiment/
#    
#===============================================================================
#-----------------------------------------------------------------------------
# Importando Bibliotecas:
#-----------------------------------------------------------------------------
import os
import re
# import nltk
# from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import scikitplot as skplt
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score
tf.__version__
seed = 2020
# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.
np.random.seed(seed)
# The below set_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see:
# https://www.tensorflow.org/api_docs/python/tf/random/set_seed
tf.random.set_seed(seed)
#%%
#-----------------------------------------------------------------------------
# Criar CSV: Essa etapa pode levar alguns minutos.
#-----------------------------------------------------------------------------
def loading_dataset(path_d):
    labels = {'pos': 'positive', 'neg': 'negative'}
    dataset = pd.DataFrame()
    for directory in ('test', 'train'):
        for sentiment in ('pos', 'neg'):
            path = path_d.format(directory, sentiment)                       
            for review_file in os.listdir(path):
                with open(os.path.join(path, review_file), 'r', encoding='utf8') as input_file:
                    review = input_file.read()
                dataset = dataset.append([[review, labels[sentiment]]],
                                         ignore_index=True)
    dataset.columns = ['review', 'sentiment']
    indices = dataset.index.tolist()
    np.random.shuffle(indices)
    indices = np.array(indices)
    dataset = dataset.reindex(index=indices)
    dataset.to_csv('movie_reviews.csv', index=False)

# Lembre-se de mudar o path, onde se encontra o dataset no seu diretório.
# Essa célula irá gerar um csv 'movie_reviews.csv' que pode ser lida abaixo.
path = r'C:\Users\Raul\Desktop\UFRJ\9 periodo\Redes Neurais\Trabalho\aclImdb/{}/{}'
loading_dataset(path_d = path)
#%%
#-----------------------------------------------------------------------------
# Ler o arquivo CSV:
#-----------------------------------------------------------------------------
df_imdb = pd.read_csv('movie_reviews.csv')  
#%%
#-----------------------------------------------------------------------------
# Pré-processamento: Limpando os textos
#-----------------------------------------------------------------------------
# Limpando texto:
TAG_RE = re.compile(r'<[^>]+>')
def remove_tags(text):
    return TAG_RE.sub('', text)

def preprocess_text(sentence):
    # Removendo tags HTML
    review = remove_tags(sentence)
    # Removendo pontuação e números
    review = re.sub('[^a-zA-Z]', ' ', review)
    # Removendo caracteres soltos
    review = re.sub(r"\s+[a-zA-Z]\s+", ' ', review)
    # Removendo espaços múltiplos
    review = re.sub(r'\s+', ' ', review)
    # Texto em letras minúsculas
    review = review.lower()
    return review

X = []
reviews = list(df_imdb['review'])
for i in reviews:
    X.append(preprocess_text(i))
#%%
#-----------------------------------------------------------------------------
# Binarizando variável alvo:
#-----------------------------------------------------------------------------
y = df_imdb['sentiment']
y = np.array(list(map(lambda x: 1 if x=="positive" else 0, y)))
#%%
#-----------------------------------------------------------------------------
# Dividindo em conjunto de treino e teste:
#-----------------------------------------------------------------------------
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = seed)
#%%
#-----------------------------------------------------------------------------
# Tokenizer - https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text/Tokenizer
#-----------------------------------------------------------------------------
tok = tf.keras.preprocessing.text.Tokenizer(num_words=3000)
tok.fit_on_texts(X_train)
X_train = tok.texts_to_sequences(X_train)
X_test = tok.texts_to_sequences(X_test)

# Adding 1 because of reserved 0 index
vocab_size = len(tok.word_index) + 1
maxlen = 150
#%%
#-----------------------------------------------------------------------------
# pad_sequences - https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/sequence/pad_sequences
#-----------------------------------------------------------------------------
X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, padding='post', maxlen=maxlen)
#%%
#-----------------------------------------------------------------------------
# Preparando camada Embedding usando GloVe: essa etapa pode demorar alguns minutos
#-----------------------------------------------------------------------------
# Baixar “glove.6B.zip": https://nlp.stanford.edu/projects/glove/
# É o menor pacote de embeddings (822Mb) Ele foi treinado em um conjunto de 
# dados de um bilhão de tokens (palavras) com um vocabulário de 400 mil palavras.

embeddings_dictionary = dict()
# Lembre-se de mudar o path, onde se encontra o pacote de embeddings no seu diretório.
glove_file = open(r"C:\Users\Raul\Desktop\UFRJ\9 periodo\Redes Neurais\Trabalho\Embeddings\glove.6B.300d.txt", encoding="utf8")

for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = np.asarray(records[1:], dtype='float32')
    embeddings_dictionary [word] = vector_dimensions
glove_file.close()
# Lembre-se de alterar o tamanho do vetor de incorporação de acordo com o pacote que você está utilizando
vec_embeddings = 300
embedding_matrix = np.zeros((vocab_size, vec_embeddings))
for word, index in tok.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector

#%%
#-----------------------------------------------------------------------------
# Criando modelo de rede neural profunda:
#-----------------------------------------------------------------------------
# Executar modelo com ou sem validação cruzada
cross_val = False
if(cross_val):
    kfold = StratifiedKFold(n_splits = 4, shuffle = True, random_state = seed)
    cvscores = []
    for train, test in kfold.split(X_train, y_train):
        
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Embedding(vocab_size, vec_embeddings, 
                                            weights=[embedding_matrix], 
                                            input_length = maxlen, 
                                            trainable=False))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.LSTM(100))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        # compile the model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) #binary_crossentropy/categorical_crossentropy 
        # summarize the model
        print(model.summary())
        # fit the model
        model.fit(X_train, y_train, epochs = 10, batch_size=64, validation_split=0.25)  
        scores = model.evaluate(X_test, y_test, verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        cvscores.append(scores[1] * 100)
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
else:
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Embedding(vocab_size, vec_embeddings, 
                                            weights=[embedding_matrix], 
                                            input_length = maxlen, 
                                            trainable=False))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.LSTM(100))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    # compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) #binary_crossentropy/categorical_crossentropy 
    # summarize the model
    print(model.summary())
    # fit the model
    model.fit(X_train, y_train, epochs = 10, batch_size=64, validation_split=0.25)  
    
#%%
scores = model.evaluate(X_test, y_test)
print ("Avaliando...")
print("Accuracy: %.2f%%" % (scores[1]*100))
#%%
#-------------------------------------------------------------------------------
# Matriz de Confusão
#-------------------------------------------------------------------------------
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)
print(accuracy_score(y_test, y_pred))

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        title = 'Confusion matrix, without normalization'
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

plot_confusion_matrix(y_test, y_pred, classes=[1, 0], title='Confusion matrix')
# plt.savefig("ConfusionMatrix.png")
#%%
#-------------------------------------------------------------------------------
# Curva ROC
#-------------------------------------------------------------------------------
y_true = y_test
y_probas = np.concatenate((1-y_pred,y_pred),axis=1)
skplt.metrics.plot_roc(y_true,y_probas)
plt.show()