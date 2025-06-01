import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import nltk
import data_augmentation
import preprocess_data 
import train_ml_model
import train_DL_model
import encode_data
import vectorizer
import time
import feature_extraction
from data_augmentation import *
from preprocess_data import *
from train_ml_model import *
from train_DL_model import *
from encode_data import *
from vectorizer import *
from feature_extraction import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score, precision_score, roc_auc_score
from nltk import word_tokenize, pos_tag, pos_tag_sents
from sklearn import metrics
from sklearn.metrics import mean_squared_error,log_loss
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from keras.initializers import Constant
from keras.layers import Embedding,LSTM,Dense,Dropout,Bidirectional,Input,GlobalMaxPool1D,SpatialDropout1D
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
from keras.models import Model,load_model
import keras.optimizers
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Flatten
from keras.layers import Dense
from keras.initializers import Constant
from keras.callbacks import ReduceLROnPlateau,EarlyStopping
from keras.layers import Embedding,LSTM,Dense,Dropout,Bidirectional
import keras.optimizers
import tensorflow as tf
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier,BaggingClassifier,GradientBoostingClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import multilabel_confusion_matrix,classification_report,confusion_matrix,accuracy_score,f1_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV

class Metrics(tf.keras.callbacks.Callback):

    def __init__(self, validation_data=()):
        super().__init__()
        self.validation_data = validation_data

    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        xVal, yVal, target_type = self.validation_data
        if target_type == 'multi_class':
          val_predict_classes = model.predict_classes(xVal, verbose=0) # Multiclass
        else:
          val_predict_classes = (np.asarray(self.model.predict(xVal))).round() # Multilabel
        
        
        val_targ = yVal

        _val_f1 = f1_score(val_targ, val_predict_classes, average='micro')
        _val_recall = recall_score(val_targ, val_predict_classes, average='micro')
        _val_precision = precision_score(val_targ, val_predict_classes, average='micro')
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        #print("— train_f1: %f — train_precision: %f — train_recall %f" % (_val_f1, _val_precision, _val_recall))
        return

def read_data(path):
    df=pd.read_csv(path)
    return df

def augment_mydata(df,alpha_sr,alpha_ri,alpha_rs,alpha_rd,num_aug):
  df_upsampled=gen_eda(df,alpha_sr,alpha_ri,alpha_rs,alpha_rd,num_aug) 
  return df_upsampled

def dl_clean(df):
    df["Description_DL"] = df["Description"].apply(lambda x: clean_DL_data1(x))
    return df

def ml_clean(df):
    df["Description_ML"] = df["Description"].apply(lambda x: clean_data(x))
    return df

def ml_models(df,num_of_features):
    count_train_tfidf_ML_2,features_tfidf_ML_2 = tfidf_vectorizer_features(df.Description_ML,2,int(num_of_features))
    x_ML_tfidf_2=pd.DataFrame(count_train_tfidf_ML_2,columns=list(features_tfidf_ML_2))
    lb_make = LabelEncoder()
    df['Accident_Level'] = lb_make.fit_transform(df['Accident_Level'])
    y_ML = pd.get_dummies(df['Accident_Level']).values
    x_ML_tfidf_2 = x_ML_tfidf_2.join(df['Accident_Level'].reset_index(drop=True))
    X=x_ML_tfidf_2.drop(['Accident_Level'],axis=1)
    Y=x_ML_tfidf_2.Accident_Level
    X_train, X_test, y_train, y_test = train_test_split(X,  Y, test_size = 0.20, random_state = 1, stratify = y_ML)
    results_df = train_test_allmodels(X_train, X_test, y_train, y_test, 'no','no','no')
    return results_df

def dl_models(df,num_of_features,path):
    my_corpus = []
    for text in df['Description_DL']:
        words = [word.lower() for word in word_tokenize(text)] 
        my_corpus.append(words)
    num_words = len(my_corpus)
    X = df['Description_DL']
    Y = df['Accident_Level']
    Y = LabelEncoder().fit_transform(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20, random_state = 1, stratify = Y)
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    tokenizer = Tokenizer(num_words)
    tokenizer.fit_on_texts(X_train)

    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)

    vocab_size = len(tokenizer.word_index) + 1

    max_length = 750
    X_train = pad_sequences(X_train, padding='post', maxlen=max_length)
    X_test = pad_sequences(X_test, padding='post', maxlen=max_length)
    X_test, X_val, y_test, y_val = train_test_split(X_test,y_test, test_size = 0.5, random_state=2)

    embedding = {}
    with open(path) as file:
        for line in file:
            values = line.split()
            word = values[0]
            vectors = np.asarray(values[1:], 'float32')
            embedding[word] = vectors
    file.close()

    embedding_size = 200
    embeddings_dictionary = dict()

    embedding_matrix = np.zeros((vocab_size, embedding_size))
    for i, word in tokenizer.index_word.items():
        if i < (num_words+1):
            vector = embedding.get(word)
            if vector is not None:
                embedding_matrix[i] = vector

    # Build a Bi-directional LSTM Neural Network
    epochs=20
    deep_inputs = Input(shape=(max_length,))
    embedding_layer = Embedding(vocab_size, embedding_size, weights=[embedding_matrix], trainable=False)(deep_inputs)

    LSTM_Layer_1 = Bidirectional(LSTM(128, return_sequences = True))(embedding_layer)
    max_pool_layer_1 = GlobalMaxPool1D()(LSTM_Layer_1)
    # drop_out_layer_1 = Dropout(0.5, input_shape = (256,))(max_pool_layer_1)
    # dense_layer_1 = Dense(128, activation = 'relu')(drop_out_layer_1)
    # drop_out_layer_2 = Dropout(0.5, input_shape = (128,))(dense_layer_1)
    # dense_layer_2 = Dense(64, activation = 'relu')(max_pool_layer_1)
    # drop_out_layer_3 = Dropout(0.01, input_shape = (64,))(dense_layer_2)
    #(drop_out_layer_3)
    dense_layer_3 = Dense(32, activation = 'relu')(max_pool_layer_1)
    drop_out_layer_4 = Dropout(0.01, input_shape = (32,))(dense_layer_3)

    dense_layer_4 = Dense(10, activation = 'relu')(drop_out_layer_4)
    drop_out_layer_5 = Dropout(0.01, input_shape = (10,))(dense_layer_4)

    dense_layer_5 = Dense(5, activation='softmax')(drop_out_layer_5)
    #dense_layer_3 = Dense(5, activation='softmax')(drop_out_layer_3)

    # LSTM_Layer_1 = LSTM(128)(embedding_layer)
    # dense_layer_1 = Dense(5, activation='softmax')(LSTM_Layer_1)
    # model = Model(inputs=deep_inputs, outputs=dense_layer_1)

    model = Model(inputs=deep_inputs, outputs=dense_layer_5)
    #model = Model(inputs=deep_inputs, outputs=dense_layer_3)

    opt = SGD(learning_rate=0.001, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])

    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=7, min_delta=1E-3)
    rlrp = ReduceLROnPlateau(monitor='val_loss', factor=0.0001, patience=5, min_delta=1E-4)

    target_type = 'multi_label'
    metrics = Metrics(validation_data=(X_train, y_train, target_type))

    # fit the keras model on the dataset
    training_history = model.fit(X_train, y_train, epochs=epochs, batch_size=8, verbose=1,validation_split=0.2, callbacks=[rlrp, metrics])
    # evaluate the keras model
    train_accuracy = model.evaluate(X_train, y_train, batch_size=5, verbose=0)
    test_accuracy = model.evaluate(X_test, y_test, batch_size=5, verbose=0)
    return train_accuracy,test_accuracy