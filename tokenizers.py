"""Tokenize documents in a corpus"""

from gensim import utils
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet as wn, opinion_lexicon
from sklearn.feature_extraction.text import CountVectorizer

con_list = {
    "ain't": "are not",
    "aren't": "are not",
    "can't": "can not",
    "can't've": "can not have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so is",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have"
}
contractions = {}
for key, value in con_list.iteritems():
    contractions[key] = value
    if key != "i'll" and key != "it's":
        contractions[key.replace("'", '')] = value

positive_words = set(opinion_lexicon.positive()) | \
    set(['not_'+nw for nw in opinion_lexicon.negative()]) | \
    set(["n't_"+nw for nw in opinion_lexicon.negative()])
negative_words = set(opinion_lexicon.negative()) | \
    set(['not_'+pw for pw in opinion_lexicon.positive()]) | \
    set(["n't_"+pw for pw in opinion_lexicon.positive()])


def stopwordsTokenizer(review):
    """Make everything lowercase"""
    words = []
    for word in review.split():
        word = word.lower()
        words += [word]
    return ' '.join(words)


def basicStemTokenizer2(review):
    """Stem"""

    # lowercase
    review = review.lower()

    # Stem using Porter Stemmer
    stemmer = PorterStemmer()
    stemmed = [stemmer.stem(word) for word in review.split()]

    return ' '.join(stemmed)


def basicLemmaTokenizer2(review):
    """Lemmatize"""

    # lowercase
    review = review.lower()

    # Lemmatize from parts of speech
    tokens = []
    for lemma in utils.lemmatize(review):
        lemma, pos = lemma.split('/')
        tokens += [lemma]

    return ' '.join(tokens)


def basicStemTokenizer(review):
    """Expand contractions and stem"""

    # Uncontract contractions
    words = []
    for word in review.split():
        word = word.lower()
        if word in contractions:
            word = contractions[word]
        words += [word]
    review = ' '.join(words)

    # Stem using Porter Stemmer
    stemmer = PorterStemmer()
    stemmed = [stemmer.stem(word) for word in review.split()]

    return ' '.join(stemmed)


def basicLemmaTokenizer(review):
    """Expand contractions and lemmatize"""

    # Uncontract contractions
    words = []
    for word in review.split():
        word = word.lower()
        if word in contractions:
            word = contractions[word]
        words += [word]
    review = ' '.join(words)

    # Lemmatize from parts of speech
    tokens = []
    for lemma in utils.lemmatize(review):
        lemma, pos = lemma.split('/')
        tokens += [lemma]

    return ' '.join(tokens)


def polarityTokenizer(review):
    """
    Expand contractions, lemmatize, and add polarity
    codewords for all words.
    """

    # Uncontract contractions
    words = []
    for word in review.split():
        word = word.lower()
        if word in contractions:
            word = contractions[word]
        words += [word]
    review = ' '.join(words)

    # Lemmatize from parts of speech
    tokens = []
    ads = []
    for lemma in utils.lemmatize(review):
        lemma, pos = lemma.split('/')
        tokens += [lemma]
        if pos.startswith('J'):
            ads += [lemma]

    # Re-merge for more processing
    lemmatized_review = ' '.join(tokens)

    # Join not with words in front;
    formatted_lm_review = lemmatized_review.replace(' not ', ' not_')

    # Positive and negative word marking
    words = []
    for word in formatted_lm_review.split():
        if word in positive_words:
            words += [word + ' POSITIVEWORD']
        elif word in negative_words:
            words += [word + ' NEGATIVEWORD']
        else:
            words += [word]

    return ' '.join(words)


def polarAdjectivesTokenizer(review):
    """
    Expand contractions, lemmatize, and add polarity
    codewords for only adjectives.
    """

    # Uncontract contractions
    words = []
    for word in review.split():
        word = word.lower()
        if word in contractions:
            word = contractions[word]
        words += [word]
    review = ' '.join(words)

    # Lemmatize from parts of speech
    tokens = []
    ads = []
    for lemma in utils.lemmatize(review):
        lemma, pos = lemma.split('/')
        tokens += [lemma]
        if pos.startswith('J'):
            ads += [lemma]

    # Re-merge for more processing
    lemmatized_review = ' '.join(tokens)

    # Join not with words in front;
    formatted_lm_review = lemmatized_review.replace(' not ', ' not_')

    # Positive and negative word marking
    words = []
    for word in formatted_lm_review.split():
        if word in ads or word.strip('not_') in ads:
            if word in positive_words:
                words += [word + ' POSITIVEWORD']
            elif word in negative_words:
                words += [word + ' NEGATIVEWORD']
        else:
            words += [word]

    return ' '.join(words)


def foodwordTokenizer(review):
    """
    Expand contractions, lemmatize, and add "foodword"
    codeword for all food-related words.
    """
    # Expand contractions
    words = []
    for word in review.split():
        word = word.lower()
        if word in contractions:
            word = contractions[word]
        words += [word]
    review = ' '.join(words)

    # Lemmatize from parts of speech
    tokens = []
    for lemma in utils.lemmatize(review):
        lemma, pos = lemma.split('/')
        tokens += [lemma]

    # Re-merge for more processing
    lemmatized_review = ' '.join(tokens)

    # Join not with words in front;
    formatted_lm_review = lemmatized_review.replace(' not ', ' not_')

    # Food word marking
    words = []
    for word in formatted_lm_review.split():
        if 'noun.food' in [syn.lexname() for syn in wn.synsets(word)]:
            words += [word, 'FOODWORD']
        else:
            words += [word]

    return ' '.join(words)


def foodwordReplacedTokenizer(review):
    """
    Epand contractions, lemmatize, and replace food-related
    words with "foodword".
    """
    # Expand contractions
    words = []
    for word in review.split():
        word = word.lower()
        if word in contractions:
            word = contractions[word]
        words += [word]
    review = ' '.join(words)

    # Lemmatize from parts of speech
    tokens = []
    for lemma in utils.lemmatize(review):
        lemma, pos = lemma.split('/')
        tokens += [lemma]

    # Re-merge for more processing
    lemmatized_review = ' '.join(tokens)

    # Join not with words in front;
    formatted_lm_review = lemmatized_review.replace(' not ', ' not_')

    # Food word replacement
    words = []
    for word in formatted_lm_review.split():
        if 'noun.food' in [syn.lexname() for syn in wn.synsets(word)]:
            words += ['FOODWORD']
        else:
            words += [word]

    return ' '.join(words)


def foodwordTokenizer2(review):
    """
    Stem and add "foodword" codeword for all food-related words.
    """
    # lowercase
    review = review.lower()

    # Stem using Porter Stemmer
    stemmer = PorterStemmer()
    stemmed = [stemmer.stem(word) for word in review.split()]

    # Re-merge for more processing
    review = ' '.join(stemmed)

    # Food word marking
    words = []
    for word in review.split():
        if 'noun.food' in [syn.lexname() for syn in wn.synsets(word)]:
            words += [word, 'FOODWORD']
        else:
            words += [word]

    return ' '.join(words)


def foodwordReplacedTokenizer2(review):
    """
    Stem and replace food-related words with "foodword".
    """
    # lowercase
    review = review.lower()

    # Stem using Porter Stemmer
    stemmer = PorterStemmer()
    stemmed = [stemmer.stem(word) for word in review.split()]

    # Re-merge for more processing
    review = ' '.join(stemmed)

    # Food word replacement
    words = []
    for word in review.split():
        if 'noun.food' in [syn.lexname() for syn in wn.synsets(word)]:
            words += ['FOODWORD']
        else:
            words += [word]

    return ' '.join(words)


def foodwordReplacedPolarityTokenizer(review):
    """
    Expand contractions, lemmatize, add codewords for positive and negative
    words, and replace food-relatedwords with "foodword".
    """
    # Expand contractions
    words = []
    for word in review.split():
        word = word.lower()
        if word in contractions:
            word = contractions[word]
        words += [word]
    review = ' '.join(words)

    # Lemmatize from parts of speech
    tokens = []
    for lemma in utils.lemmatize(review):
        lemma, pos = lemma.split('/')
        tokens += [lemma]

    # Re-merge for more processing
    lemmatized_review = ' '.join(tokens)

    # Join not with words in front;
    formatted_lm_review = lemmatized_review.replace(' not ', ' not_')

    # Positive and negative word marking + food word replacement
    words = []
    for word in formatted_lm_review.split():
        if word in positive_words:
            words += [word, 'POSITIVEWORD']
        elif word in negative_words:
            words += [word, 'NEGATIVEWORD']
        elif 'noun.food' in [syn.lexname() for syn in wn.synsets(word)]:
            words += ['FOODWORD']
        else:
            words += [word]

    return ' '.join(words)


def foodwordPolarityTokenizer(review):
    """
    Lemmatize, add codewords for positive and negative words, and replace
    add "foodword" after food-related words.
    """
    # Expand contractions
    words = []
    for word in review.split():
        word = word.lower()
        if word in contractions:
            word = contractions[word]
        words += [word]
    review = ' '.join(words)

    # Lemmatize from parts of speech
    tokens = []
    for lemma in utils.lemmatize(review):
        lemma, pos = lemma.split('/')
        tokens += [lemma]

    # Re-merge for more processing
    lemmatized_review = ' '.join(tokens)

    # Join not with words in front;
    formatted_lm_review = lemmatized_review.replace(' not ', ' not_')

    # Positive and negative word marking + food word replacement
    words = []
    for word in formatted_lm_review.split():
        if word in positive_words:
            words += [word, 'POSITIVEWORD']
        elif word in negative_words:
            words += [word, 'NEGATIVEWORD']
        elif 'noun.food' in [syn.lexname() for syn in wn.synsets(word)]:
            words += [word, 'FOODWORD']
        else:
            words += [word]

    return ' '.join(words)


def foodwordPolarAdjTokenizer(review):
    """
    Lemmatize, add codewords for positive and negative adjectives, and replace
    add "foodword" after food-related words.
    """
    # Expand contractions
    words = []
    for word in review.split():
        word = word.lower()
        if word in contractions:
            word = contractions[word]
        words += [word]
    review = ' '.join(words)

    # Lemmatize from parts of speech
    tokens = []
    ads = []
    for lemma in utils.lemmatize(review):
        lemma, pos = lemma.split('/')
        tokens += [lemma]
        if pos.startswith('J'):
            ads += [lemma]

    # Re-merge for more processing
    lemmatized_review = ' '.join(tokens)

    # Join not with words in front;
    formatted_lm_review = lemmatized_review.replace(' not ', ' not_')

    # Positive and negative word marking + food word replacement
    words = []
    for word in formatted_lm_review.split():
        if word in ads or word.strip('not_') in ads:
            if word in positive_words:
                words += [word + ' POSITIVEWORD']
            elif word in negative_words:
                words += [word + ' NEGATIVEWORD']
        elif 'noun.food' in [syn.lexname() for syn in wn.synsets(word)]:
            words += [word, 'FOODWORD']
        else:
            words += [word]

    return ' '.join(words)


def fitApplyVectorizer(corp, tokenizer):
    """
    Fit a count vectorizer to the tokenized corpus and return the fitted count
    vectorizer, the application of the vectorizer, and the documents that
    can't be processed by the tokenizer.
    """
    texts = []
    bad_docs = []
    for doc in corp:
        try:
            texts += [tokenizer(doc)]
        except Exception:
            bad_docs += [doc]
            pass
    count_vec = CountVectorizer(stop_words='english')
    return count_vec.fit(texts), count_vec.transform(texts), bad_docs


def applyVectorizer(corp, tokenizer, count_vec_fit):
    """
    Apply a fitted count vectorizer to a tokenized corpus and return the
    application of the vectorizer and the documents that can't be processed by
    the tokenizer.
    """
    texts = []
    bad_docs = []
    for doc in corp:
        try:
            texts += [tokenizer(doc)]
        except Exception:
            bad_docs += [doc]
            pass
    return count_vec_fit.transform(texts), bad_docs
