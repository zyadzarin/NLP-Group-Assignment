import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd

def lemmatize_word(word, pos_tag):
    # map NLTK POS tags to WordNet POS tags
    pos_map = {
        'N': wordnet.NOUN,
        'V': wordnet.VERB,
        'J': wordnet.ADJ,
        'R': wordnet.ADV
    }
    # lemmatize the word
    pos = pos_map.get(pos_tag[0], wordnet.NOUN)
    lemmatizer = WordNetLemmatizer()
    return lemmatizer.lemmatize(word, pos)

def remove_punctuation(text):
    # replace punctuation marks with an empty string
    from string import punctuation
    punctuation += '’'
    translator = str.maketrans('', '', punctuation)
    return text.translate(translator)

def preprocess_text(df_column:pd.core.series.Series):

    stopwords_list = set(stopwords.words('english'))

    # remove stop words
    df_column = df_column.apply(lambda x: ' '.join([word.lower() for word in x.split() if word.lower() not in stopwords_list]))
    # remove punctuation
    df_column = df_column.apply(remove_punctuation)
    # tokenize and apply POS tagging and lemmatization
    df_column = df_column.apply(lambda x: ' '.join([lemmatize_word(word, tag) for word, tag in pos_tag(word_tokenize(x))]))

    return df_column