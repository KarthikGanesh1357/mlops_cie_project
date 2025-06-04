
from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer

def extract_ngrams(corpus, n: int, top_n: int):
    '''
    Description:
        Extract n-grams from dataframe column.
    Parameters:
        corpus: dataframe column.
            Dataframe column with text to extract n-grams.
        n: int
            Type of n-grams (2 for bi-grams, 3 for tri-grams, etc.)
        top_n: int
            Most commons n-grams.
    Return:
        Top n-gram.
    '''
    vector = CountVectorizer(ngram_range=(n, n)).fit(corpus)
    bag_of_words = vector.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) 
                  for word, idx in vector.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1],
                        reverse=True)
    
    return words_freq[:top_n]