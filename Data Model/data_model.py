from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import os
import nltk
import spacy
from nltk.corpus import stopwords
stop = stopwords.words('dutch')
import re
import csv
import string
from tika import parser
from nltk.sentiment.vader import SentimentIntensityAnalyzer
vader_model = SentimentIntensityAnalyzer()
nlp = spacy.load('en_core_web_sm')


# Signaalwoorden
tegenstelling = ' maar |echter| toch |niettemin|desalniettemin|desondanks|daarentegen|aan de andere kant|enerzijds|anderzijds|hoewel|ofschoon|integendeel|daar staat tegenover|behalve als'
vergelijking = 'net zoals|net als|hetzelfde als|evenals|evenzeer|overeenkomstig|lijkt op|is vergelijkbaar met|in vergelijking met|op dezelfde wijze|hetzelfde is het geval'
opsomming = ' en | ook | verder |ten eerste|ten tweede|in de eerste plaats|in de tweede plaats|daarnaast|bovendien|vervolgens|ten slotte|als laatste| maar ook | als |een ander argument|er is nog een reden waarom|daar komt nog bij dat|tevens'

# Frequency of numbers
n9 = '[0-9]+\.[0-9][0-9]+ [a-z]+'
n16 = '[0-9]+ procent'
n17 = '[0-9]+%'

# SMART
smart_jaar = '(\d+(\.\d+)?%|procent|[0-9]+\.[0-9][0-9]+ [a-z]+) (?:[a-zA-Z]+\s+){5,15}?[1-2][0-9][0-9][0-9]' #nummer tov jaar
smart_maanden = '(\d+(\.\d+)?%|procent|[0-9]+\.[0-9][0-9]+ [a-z]+) (?:[a-zA-Z]+\s+){5,15}?(januari|februari|maart|april|mei|juni|juli|augustus|september|oktober|november|december)'
smart_kwartaal = '(\d+(\.\d+)?%|procent|[0-9]+\.[0-9][0-9]+ [a-z]+) (?:[a-zA-Z]+\s+){5,15}?(maand|maanden|jaar|jaren)'

def run_vader(textual_unit,
              lemmatize=False,
              parts_of_speech_to_consider=set(),
              verbose=0):
    """
    Run VADER on a sentence from spacy

    :param str textual unit: a textual unit, e.g., sentence, sentences (one string)
    (by looping over doc.sents)
    :param bool lemmatize: If True, provide lemmas to VADER instead of words
    :param set parts_of_speech_to_consider:
    -empty set -> all parts of speech are provided
    -non-empty set: only these parts of speech are considered
    :param int verbose: if set to 1, information is printed
    about input and output

    :rtype: dict
    :return: vader output dict
    """
    doc = nlp(textual_unit)

    input_to_vader = []

    for sent in doc.sents:
        for token in sent:

            to_add = token.text

            if lemmatize:
                to_add = token.lemma_

                if to_add == '-PRON-':
                    to_add = token.text

            if parts_of_speech_to_consider:
                if token.pos_ in parts_of_speech_to_consider:
                    input_to_vader.append(to_add)
            else:
                input_to_vader.append(to_add)

    scores = vader_model.polarity_scores(' '.join(input_to_vader))

    if verbose >= 1:
        pass

    return scores


def pdf2text(pdf):
    """
    Iterate over pages and extract text
    """
    dutch_stopwords = stopwords.words('dutch')
    rawText = parser.from_file(pdf)
    rawList = rawText['content'].splitlines()

    tokenized_text = nltk.word_tokenize(rawList)  # tokenize sentences

    nltk.pos_tag(tokenized_text)  # add pos tags to text
    words = []
    for word, tag in nltk.pos_tag(tokenized_text):
        if tag in ['NN', 'NNS', 'NNP', 'NNPS']:  # only accept words with these tags
            words.append(word)
    table = {ord(char): '' for char in string.punctuation}
    cleaned_messy_sentence = []
    for messy_word in words:
        cleaned_word = messy_word.translate(table)
        cleaned_messy_sentence.append(cleaned_word)

    without_stopwords = []

    for token in cleaned_messy_sentence:  # exclude words that contain the following
        if token.isalpha() and 3 < len(token) < 18 and token not in dutch_stopwords:
            without_stopwords.append(token)
    text_without_stopwords = ' '.join(without_stopwords)

    return text_without_stopwords


def tokenize(document):
    """return words longer than 2 chars and all alpha"""
    tokens = [w for w in document.split() if len(w) > 2 and w.isalpha()]
    return tokens


def build_corpus_from_dir(dir_path):
    corpus = []
    filenamess = []
    for root, dirs, filenames in os.walk(dir_path, topdown=False):
        for name in filenames:
            filename = name
            filenamess.append([filename])

            if '.pdf' in name:
                pdf = root + '/' + name
                document = pdf2text(pdf)
                corpus.append(document)

    return corpus, filenamess


if __name__ == '__main__':
    with open('../../results_data_model_all.csv', 'w', newline='', encoding='utf-8') as f:  # create csv file
        writer = csv.writer(f)
        writer.writerow(
            ["FILENAME", "STORY:", "TEGENSTELLING", "VERGELIJKING", "OPSOMMING", "AVERAGE TF-IDF SCORE", "STRUCTURE:", "NUMMERS", "VRAAGTEKENS", "SMART", "NEGATIVE", "NEUTRAL", "POSITIVE", "SENTENCES"])
        # open directory with papers
        corpus, filenamess = build_corpus_from_dir(
            r'/MBO Raad/MBO Raad selectie na validatie')
        vectorizer = TfidfVectorizer(tokenizer=tokenize, stop_words=stop)
        vectorizer.fit(corpus)
        indices = np.argsort(vectorizer.idf_)[::-1]
        features = vectorizer.get_feature_names()

        for root, dirs, files in os.walk(
                '/MBO Raad/MBO Raad selectie na validatie',
                topdown=False):
            for filename in files:
                compound_sentences = []
                verhaallijn = []
                keywords = []
                tf_idf_scores = []
                scores = []
                rawText = parser.from_file(root + '/' + filename)
                try:
                    text = rawText['content'].replace('\n', '')
                except AttributeError:
                    continue
                doc = ' '.join(text.split())

                tdm = vectorizer.transform([doc])
                dense = tdm.todense()
                episode = dense[0].tolist()[0]
                phrase_scores = [pair for pair in zip(range(0, len(episode)), episode) if pair[1] > 0]
                sorted_phrase_scores = sorted(phrase_scores, key=lambda t: t[1] * -1)
                for phrase, score in [(features[word_id], score) for (word_id, score) in sorted_phrase_scores][:10]:
                    keywords.append(phrase)
                    tf_idf_scores.append('{0} {1}'.format(phrase, score))
                    scores.append(score)
                    print('{0: <20} {1}'.format(phrase, score))
                print()
                average_tf_idf_score = sum(scores) / len(scores)
                average_tf_idf_score = round(average_tf_idf_score, 3)

                lowercase_text = text.lower()
                # STORY
                tegenstelling_list = re.compile(
                    "(%s)" % tegenstelling).findall(lowercase_text)
                vergelijking_list = re.compile(
                    "(%s)" % vergelijking).findall(lowercase_text)
                opsomming_list = re.compile(
                    "(%s)" % opsomming).findall(lowercase_text)
                TEGENSTELLING = len(tegenstelling_list)
                VERGELIJKING = len(vergelijking_list)
                OPSOMMING = len(opsomming_list)

                # STRUCTURE
                nummer_list = re.compile("(%s|%s|%s)" % (n9, n16, n17)).findall(lowercase_text)
                vraagtekens = re.compile('[a-z]+[?]').findall(lowercase_text)
                smart_list = re.compile("(%s|%s|%s)" % (smart_jaar, smart_maanden, smart_kwartaal)).findall(
                    lowercase_text)

                NUMMERS = len(nummer_list)
                VRAAGTEKENS = len(vraagtekens)
                SMART = len(smart_list)

                sentences = nltk.sent_tokenize(text)
                SENTENCES = len(sentences)
                for sentence in sentences:
                    compound_sentences.append(run_vader(sentence, lemmatize=True, verbose=1)['compound'])

                for score in compound_sentences:
                    if score >= 0.05:  # positive
                        verhaallijn.append(3)
                    elif score <= - 0.05:  # negative
                        verhaallijn.append(1)
                    else:  # neutral
                        verhaallijn.append(2)

                negative_sentences = 0
                positive_sentences = 0
                neutral_sentences = 0
                for number in verhaallijn:
                    if number == 1:
                        negative_sentences += 1
                    if number == 2:
                        neutral_sentences += 1
                    if number == 3:
                        positive_sentences += 1

                print([filename, "STORY:", TEGENSTELLING, VERGELIJKING, OPSOMMING, average_tf_idf_score, "STRUCTURE:", NUMMERS, VRAAGTEKENS, SMART, negative_sentences, neutral_sentences, positive_sentences, SENTENCES])
                writer.writerow([filename, "STORY:", TEGENSTELLING, VERGELIJKING, OPSOMMING, average_tf_idf_score, "STRUCTURE:", NUMMERS, VRAAGTEKENS, SMART, negative_sentences, neutral_sentences, positive_sentences, SENTENCES])

    read_file = pd.read_csv(r'../../results_data_model_all.csv')
    read_file.to_excel(r'results_data_model_all.xlsx', index=None, header=True)
