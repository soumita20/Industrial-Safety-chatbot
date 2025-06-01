from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression 
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import conlltags2tree, tree2conlltags
from pprint import pprint
import pandas as pd
import numpy as np

#important parameters
#1. max-df : it is used to remove terms that are very frequent, also known as "corpus specific stop words."
    #e.g. max_df = 0.50 means "ignore terms that appear in more than 50% of the documents".
          #max_df = 25 means "ignore terms that appear in more than 25 documents"
          #default value of max_df is 1.0 which means "ignore terms that appear in more than 100% of the documents". This means the default setting does not ignore any of the documents.
          #When using a float in the range [0.0, 1.0] they refer to the document frequency. That is the percentage of documents that contain the term.
          #When using an int it refers to absolute number of documents that hold this term.

#2. min_df : it is used for removing terms that appear too infrequently.
    #e.g. min_df = 0.01 means "ignore terms that appear in less than 1% of the documents".
          #min_df = 5 means "ignore terms that appear in less than 5 documents".
          #The default min_df is 1, which means "ignore terms that appear in less than 1 document". This means, the default setting does not ignore any terms.

#3. ngram_range tuple (min_n, max_n), default=(1, 1)
    #The lower and upper boundary of the range of n-values for different word n-grams or char n-grams to be extracted. All values of n such such that min_n <= n <= max_n will be used. For example an ngram_range of (1, 1) means only unigrams, (1, 2) means unigrams and bigrams, and (2, 2) means only bigrams. Only applies if analyzer is not callable.

#4. vocabularyMapping or iterable, default=None
    #Either a Mapping (e.g., a dict) where keys are terms and values are indices in the feature matrix, or an iterable over terms. If set to None, it takes the vocabulary of the input data.

#5.max_features : max_featuresint, default=None
#If not None, build a vocabulary that only consider the top max_features ordered by term frequency across the corpus.
#This parameter is ignored if vocabulary is not None.

#count vectorizer unigrams
def count_vectorizer_features(data,ngrams,max_features=None):
  vectorizer = CountVectorizer(ngram_range=(1, ngrams), max_df=1.0, min_df=1, max_features=max_features)
  transformed_data=vectorizer.fit_transform(data)
  features=vectorizer.get_feature_names()
  #df = pd.DataFrame(transformed_data.toarray(),columns=list(features))
  #return df,transformed_data.toarray(),features
  return transformed_data.toarray(),features

#TF IDF vectorizer
#count vectorizer unigrams
def tfidf_vectorizer_features(data,ngrams,max_features=None):
  vectorizer = TfidfVectorizer(ngram_range=(1, ngrams), max_df=1.0, min_df=1, max_features=max_features)
  transformed_data=vectorizer.fit_transform(data)
  features=vectorizer.get_feature_names()
  # df = pd.DataFrame(transformed_data.toarray(),columns=list(features))
  # return df,transformed_data.toarray(),features
  return transformed_data.toarray(),features

#named entity recognition
def named_entity_recognition(sent):
    sent = nltk.word_tokenize(sent)
    #We get a list of tuples containing the individual words in the sentence and their associated part-of-speech.
    sent = nltk.pos_tag(sent)
    #Now we’ll implement noun phrase chunking to identify named entities using a regular expression consisting of rules that indicate how sentences should be chunked.Now we’ll implement noun phrase chunking to identify named entities using a regular expression consisting of rules that indicate how sentences should be chunked.
    #Our chunk pattern consists of one rule, that a noun phrase, NP, should be formed whenever the chunker finds an optional determiner, DT, followed by any number of adjectives, JJ, and then a noun, NN.
    pattern = 'NP: {<DT>?<JJ>*<NN>}'
    #Chunking
    cp = nltk.RegexpParser(pattern)
    cs = cp.parse(sent)
    iob_tagged = tree2conlltags(cs)
    #In this representation, there is one token per line, each with its part-of-speech tag and its named entity tag. Based on this training corpus, we can construct a tagger that can be used to label new sentences; and use the nltk.chunk.conlltags2tree() function to convert the tag sequences into a chunk tree.
    pprint(iob_tagged)
    return cs
