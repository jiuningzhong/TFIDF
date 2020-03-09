import flask
import requests
import json
import re, nltk, spacy, string
from flask import jsonify
from flask import request
import pandas as pd
import pyLDAvis
import pyLDAvis.sklearn
import numpy as np
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer 
import warnings
warnings.filterwarnings('ignore')
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import chart_studio.plotly as py
# pip install -U textblob sklearn matplotlib seaborn chart-studio cufflinks bokeh
# conda install -c plotly chart-studio
# conda install -c conda-forge cufflinks-py
import cufflinks
pd.options.display.max_columns = 30
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'
cufflinks.go_offline()
cufflinks.set_config_file(world_readable=True, theme='pearl')
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.manifold import TSNE
from bokeh.plotting import figure, output_file, show
from bokeh.models import Label
from bokeh.io import output_notebook
output_notebook()
from collections import Counter
import scattertext as st
# reinstall numpy, then
# conda install -c conda-forge spacy
# python -m spacy validate
# python -m spacy download en

from pprint import pprint
import en_core_web_sm
nlp = en_core_web_sm.load()
app = flask.Flask(__name__)
app.config["DEBUG"] = True

predict_url = "http://192.168.0.15:9098/predicttest"

requestJson = {"requestID": "a1ad23456",
               "jsonString": "what evidence does the worm home so sandy",
               "requestorName": "Vernor Vinge"}

df = pd.read_csv('Train_all_types.csv', encoding='utf-8')


def get_top_n_words(corpus, n=None):
    vec = CountVectorizer(stop_words='english', ngram_range=(1,2), min_df=0.03, max_features=100).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

def get_top_n_bigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(2, 2), stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

def get_top_n_trigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(3, 3), stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

# Show top 20 keywords for each topic
def show_topics(vectorizer=vectorizer, lda_model=lda_model, n_words=20):
    keywords = np.array(vectorizer.get_feature_names())
    topic_keywords = []
    for topic_weights in lda_model.components_:
        top_keyword_locs = (-topic_weights).argsort()[:n_words]
        topic_keywords.append(keywords.take(top_keyword_locs))
    return topic_keywords

def clean_text(text):
    '''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

def lemmatizer(text):
    sent = []
    doc = nlp(text)
    for word in doc:
        sent.append(word.lemma_)
    return " ".join(sent)

@app.route('/', methods=['GET'])
def home():
    return "<h1>Default REST API Tutorial Site</h1><p>This site is a prototype API for interacting with LightSide.</p>"

@app.route('/lightside', methods=['GET', 'POST'])
def json_example():
    # get json request
    req = request.get_json()

    print('We have', len(df), 'nodes in the data')
    df_clean = pd.DataFrame(df.text.apply(lambda x: clean_text(x)))
    df_clean["text_lemmatize"] = df_clean.apply(lambda x: lemmatizer(x['text']), axis=1)
    df_clean.to_csv('df_clean.csv', index=False)
    df_clean = pd.read_csv('df_clean.csv')
    df_clean.head()

    df_clean['text_lemmatize_clean'] = df_clean['text_lemmatize'].str.replace('-PRON-', '')

    df_u_clean = df_clean['text_lemmatize_clean'].values.astype('U')

    vec = CountVectorizer(stop_words='english', ngram_range=(1, 2), min_df=0.05, max_features=100).fit(df_u_clean)

    common_words = get_top_n_words(df_u_clean, 30)
    df2 = pd.DataFrame(common_words, columns=['unigram', 'count'])

    vectorizer = CountVectorizer(analyzer='word',
                                 min_df=3,  # minimum required occurences of a word
                                 stop_words='english',  # remove stop words
                                 lowercase=True,  # convert all words to lowercase
                                 token_pattern='[a-zA-Z0-9]{3,}',  # num chars > 3
                                 max_features=5000,
                                 # max number of unique words. Build a vocabulary that only consider the top max_features ordered by term frequency across the corpus
                                 )
    data_vectorized = vectorizer.fit_transform(df_u_clean)

    lda_model = LatentDirichletAllocation(n_components=20,  # Number of topics
                                          learning_method='online',
                                          random_state=0,
                                          n_jobs=-1  # Use all available CPUs
                                          )
    lda_output = lda_model.fit_transform(data_vectorized)

    topic_keywords = show_topics(vectorizer=vectorizer, lda_model=lda_model, n_words=20)

    # json to String
    jsonObj = json.dumps(requestJson)
    # print("jsonString: " + req["jsonString"])
    r = requests.post(predict_url, "", jsonObj)
    responseText = r.json()

    print("predicted: " + responseText["predicted"])
    print("feedbackText: " + responseText["feedbackText"])
    print("requestTimestamp: " + responseText["requestTimestamp"])
    print("accuracy: " + responseText["accuracy"])
    return r.text


app.run()
