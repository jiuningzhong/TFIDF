import re, string
import pandas as pd
import numpy as np
import en_core_web_sm
import warnings
from IPython.core.interactiveshell import InteractiveShell
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
import csv
warnings.filterwarnings('ignore')
pd.options.display.max_columns = 30
InteractiveShell.ast_node_interactivity = 'all'
nlp = en_core_web_sm.load()

csv.register_dialect("comma", delimiter=",")

class Dict(dict):
    def __missing__(self, key):
        return 0

text_dict = Dict()
tf_idf_dict = Dict()

def write_to_csv(dict1, dict2, file_name):
    with open(file_name, 'w', newline='', encoding="utf8") as csvfile:
        writer = csv.writer(csvfile, dialect="comma")
        keys = dict1.keys()
        for k in keys:
            writer.writerow((dict2[k], dict1[k]))

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
def show_topics(vectorizer, lda_model, n_words=20):
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

if __name__ == "__main__":
    # os.getcwd()
    # df = pd.read_csv('./api/Train_all_types.csv', encoding='utf-8')
    df = pd.read_csv('Train_all_types_comma.csv', encoding='utf-8')

    # print('We have', len(df), 'nodes in the data')
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

    data_vectorized.shape

    tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    tfidf_transformer.fit(data_vectorized)

    # print idf values
    df_idf = pd.DataFrame(tfidf_transformer.idf_, index=vectorizer.get_feature_names(),columns=["idf_weights"])

    # sort ascending
    df_idf.sort_values(by=['idf_weights'], ascending=False)

    # count matrix
    count_vector = vectorizer.transform(df_u_clean)

    # tf-idf scores
    tf_idf_vector = tfidf_transformer.transform(count_vector)

    feature_names = vectorizer.get_feature_names()

    # get tfidf vector for first document
    first_document_vector = tf_idf_vector[0]

    # print the scores
    df = pd.DataFrame(first_document_vector.T.todense(), index=feature_names, columns=["tfidf"])
    df.sort_values(by=["tfidf"], ascending=False)

    # settings that you use for count vectorizer will go here
    tfidf_vectorizer = TfidfVectorizer(use_idf=True)

    # just send in all your docs here
    tfidf_vectorizer_vectors = tfidf_vectorizer.fit_transform(df_u_clean)

    # get the first vector out (for the first document)
    first_vector_tfidfvectorizer = tfidf_vectorizer_vectors[0]

    # place tf-idf values in a pandas data frame
    df = pd.DataFrame(first_vector_tfidfvectorizer.T.todense(), index=tfidf_vectorizer.get_feature_names(),
                      columns=["tfidf"])
    df.sort_values(by=["tfidf"], ascending=False)

    lda_model = LatentDirichletAllocation(n_components=20,  # Number of topics
                                          learning_method='online',
                                          random_state=0,
                                          n_jobs=-1  # Use all available CPUs
                                          )
    lda_output = lda_model.fit_transform(data_vectorized)

    topic_keywords = show_topics(vectorizer=vectorizer, lda_model=lda_model, n_words=20)

    # print(topic_keywords[1])

    # learn tfidf using TfidfVectorizer from sklean
    tfidf_vectorizer = TfidfVectorizer(use_idf=True, ngram_range=(1,5), stop_words='english', # ngram_range=(1,6)
                                        analyzer = 'word',
                                        min_df = 3,  # minimum required occurences of a word
                                        # min_df = 3
                                        lowercase = True,  # convert all words to lowercase
                                        token_pattern = '[a-zA-Z0-9]{3,}',  # num chars > 3
                                        max_features = 5000,
                                        sublinear_tf = True,
    # max number of unique words. Build a vocabulary that only consider the top max_features ordered by term frequency across the corpus
    )
    X = tfidf_vectorizer.fit_transform(df_u_clean)
    X.data
    feature_names = tfidf_vectorizer.get_feature_names()
    count = 0
    Y = np.zeros(X.shape[0])

    # print(X.shape[0])
    # print(len(df_u_clean))

    for i in range(X.shape[0]):
        Y[i] = X[i].sum()
        if Y[i] > 7: #8
            # print('line number: ' + str((i+2)) + ' tf-idf value: ' + str(Y[i]))
            # print(df_u_clean[i])

            text_dict[count] = 'line number: ' + str((i+2)) + ' node text: ' + df_u_clean[i]
            tf_idf_dict[count] = Y[i]
            count = count + 1
    # print('tf_idf count: ' + str(count))

    write_to_csv(tf_idf_dict, text_dict, 'tf_idf_novelty_nodes_hashes.csv')

    doc = 115
    feature_index = X[doc, :].nonzero()[1]
    tfidf_scores = zip(feature_index, [X[doc, x] for x in feature_index])

    for w, s in [(feature_names[i], s) for (i, s) in tfidf_scores]:
        print (w, s)

    # just send in all your docs here
    # fitted_vectorizer = tfidf_vectorizer.fit(df_u_clean)
    # tfidf_vectorizer_vectors = fitted_vectorizer.transform(df_u_clean)
    # Clustering text documents using k-means
    # print("n_samples: %d, n_features: %d" % X.shape)

    svd = TruncatedSVD(100)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)
    X = lsa.fit_transform(X)
    explained_variance = svd.explained_variance_ratio_.sum()
    # print("Explained variance of the SVD step: {}%".format(int(explained_variance * 100)))
    km = KMeans(n_clusters=4, init='k-means++', max_iter=100, n_init=1, verbose=0)
    # print("Clustering sparse data with %s" % km)
    km.fit(X)
    # print("Top terms per cluster:")
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    terms = tfidf_vectorizer.get_feature_names()
    #for i in range(4):
        # print("Cluster %d:" % i, end='')
        #for ind in order_centroids[i, :10]:
            # print(' %s' % terms[ind], end='')
        # print()