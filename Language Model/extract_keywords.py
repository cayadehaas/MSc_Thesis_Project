#https://towardsdatascience.com/extract-keywords-from-documents-unsupervised-d6474ed38179
import os
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from tika import parser
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import spacy
nlp = spacy.load('en_core_web_sm')
ignored_words = list(stopwords.words('dutch'))
from nltk.collocations import BigramAssocMeasures
from nltk.collocations import BigramCollocationFinder
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize


def pdf2text(pdf):
    """Iterate over pages and extract text"""
    rawText = parser.from_file(pdf)
    try:
        rawList = rawText['content'].splitlines()
        # Pre-processing
        while '' in rawList: rawList.remove('')
        while ' ' in rawList: rawList.remove(' ')
        while '\t' in rawList: rawList.remove('\t')
        while '\n' in rawList: rawList.remove('\n')
        text = ' '.join(rawList)
        text = text.lower()
    except:
        text = ''

    return text


def build_corpus_from_dir(dir_path):
    corpus = []
    for root, dirs, filenames in os.walk(dir_path, topdown=False):
        for name in filenames:
            if '.pdf' in name:
                pdf = root + '/' + name
                document = pdf2text(pdf)
                document = lemmatization(document)
                corpus.append(document)

    return corpus


def lemmatization(texts, allowed_postags=None):
    if allowed_postags is None:
        allowed_postags = ['NOUN']
    doc = nlp(texts)
    document = [token.lemma_ for token in doc if token.pos_ in allowed_postags]
    text = ' '.join(document)

    return text


def upload_file(file_path):
    filename = []
    for root, dirs, filenames in os.walk(file_path, topdown=False):
        for name in filenames:
            if '.pdf' in name:
                pdf = root + '/' + name
                document = pdf2text(pdf)
                document = lemmatization(document)
                filename.append(document)

    return filename


if __name__ == '__main__':
    corpus = build_corpus_from_dir(
            r'/Users/cayadehaas/PycharmProjects/Transparency-Lab/MBO Raad/MBO Raad selectie Topic 1')

    filename = upload_file(
            r'/Users/cayadehaas/PycharmProjects/Transparency-Lab/MBO Raad/Document')

    corpus.extend(filename)

    # Single word extraction
    count_vec = CountVectorizer(
        ngram_range=(1, 1)
        , stop_words=ignored_words
    )
    text_set = [doc for doc in corpus]
    tf_result = count_vec.fit_transform(text_set)
    tf_result_df = pd.DataFrame(tf_result.toarray()
                                , columns=count_vec.get_feature_names())
    the_sum_s = tf_result_df.sum(axis=0)
    the_sum_df = pd.DataFrame({
        'keyword': the_sum_s.index
        , 'tf_sum': the_sum_s.values
    })
    the_sum_df = the_sum_df[
        the_sum_df['tf_sum'] > 2
        ].sort_values(by=['tf_sum'], ascending=False)

    start_index = int(len(the_sum_df) * 0.01)
    my_word_df = the_sum_df.iloc[start_index:]
    my_word_df = my_word_df[my_word_df['keyword'].str.len() > 4]
    my_word_df = my_word_df[my_word_df['keyword'].str.isalpha()]

    # Bigram extraction
    text_set_biwords = [word_tokenize(doc) for doc in corpus]
    bigram_measures = BigramAssocMeasures()
    biword_finder = BigramCollocationFinder.from_documents(text_set_biwords)
    biword_finder.apply_freq_filter(3)
    biword_finder.apply_word_filter(lambda w:
                                    len(w) < 3
                                    or len(w) > 15
                                    or w.lower() in ignored_words)
    biword_phrase_result = biword_finder.nbest(bigram_measures.pmi, 20000)
    biword_colloc_strings = [w1 + ' ' + w2 for w1, w2 in biword_phrase_result]

    my_vocabulary = []
    my_vocabulary.extend(my_word_df['keyword'].tolist())
    my_vocabulary.extend(biword_colloc_strings)

    # TF-IDF calculation
    vec = TfidfVectorizer(
        analyzer='word'
        , ngram_range=(1, 2)
        , vocabulary=my_vocabulary)
    text_set = [doc for doc in corpus]
    tf_idf = vec.fit_transform(text_set)
    result_tfidf = pd.DataFrame(tf_idf.toarray()
                                , columns=vec.get_feature_names())

    file_index = -1  # This is the PDF that is added to the corpus as last item
    test_tfidf_row = result_tfidf.iloc[file_index]
    keywords_df = pd.DataFrame({
        'keyword': test_tfidf_row.index,
        'tf-idf': test_tfidf_row.values
    })
    keywords_df = keywords_df[
        keywords_df['tf-idf'] > 0
        ].sort_values(by=['tf-idf'], ascending=False)

    bigram_words = [item.split()
                    for item in keywords_df['keyword'].tolist()
                    if len(item.split()) == 2]
    bigram_words_set = set(subitem
                           for item in bigram_words
                           for subitem in item)
    keywords_df_new_biwords = keywords_df[~keywords_df['keyword'].isin(bigram_words_set)]
    keywords = keywords_df_new_biwords['keyword'][:50]  # change number depending on amount of keywords wanted
    keywords_30 = []
    for keyword in keywords:
        keywords_30.append(keyword)
    print(keywords_30)
